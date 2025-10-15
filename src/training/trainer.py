"""
Training utilities and strategies for VoxConverse diarization.
Adapted from curriculum_trainer.py and progressive_training.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
import math

# Import VoxConverse dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voxconverse_dataset import VoxConverseDataset


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions [N, ...] 
            targets: Ground truth [N, ...]
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ProgressiveTrainingStrategy:
    """Progressive training strategy with curriculum learning."""
    
    def __init__(self, curriculum_schedule):
        """
        Args:
            curriculum_schedule: List of (epoch_start, segment_duration, complexity_factor)
        """
        self.curriculum_schedule = curriculum_schedule
        self.current_epoch = 0
        self.current_segment_duration = curriculum_schedule[0][1]
        self.current_complexity = curriculum_schedule[0][2]
        
    def update_epoch(self, epoch):
        """Update configuration for current epoch."""
        self.current_epoch = epoch
        
        # Find appropriate configuration for this epoch
        for epoch_start, duration, complexity in reversed(self.curriculum_schedule):
            if epoch >= epoch_start:
                self.current_segment_duration = duration
                self.current_complexity = complexity
                break
        
        print(f"📚 Epoch {epoch}: segments={self.current_segment_duration}s, complexity={self.current_complexity}")
        
    def get_current_config(self):
        """Get current training configuration."""
        return {
            'segment_duration': self.current_segment_duration,
            'complexity_factor': self.current_complexity,
            'epoch': self.current_epoch
        }


class AdaptiveLossFunction(nn.Module):
    """Adaptive loss function that combines multiple loss types."""
    
    def __init__(self, vad_weight=1.0, osd_weight=1.0, vcn_weight=1.0,
                 use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0,
                 class_weights=None):
        super(AdaptiveLossFunction, self).__init__()
        
        self.vad_weight = vad_weight
        self.osd_weight = osd_weight
        self.vcn_weight = vcn_weight
        self.use_focal_loss = use_focal_loss
        
        if use_focal_loss:
            self.vad_criterion = FocalLoss(focal_alpha, focal_gamma)
            self.osd_criterion = FocalLoss(focal_alpha, focal_gamma)
            self.vcn_criterion = FocalLoss(focal_alpha, focal_gamma)
        else:
            # Use class weights if provided
            vad_pos_weight = class_weights['vad'] if class_weights else 1.0
            osd_pos_weight = class_weights['osd'] if class_weights else 1.0
            vcn_pos_weight = class_weights['vcn'] if class_weights else 1.0
            
            self.vad_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(vad_pos_weight))
            self.osd_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(osd_pos_weight))
            self.vcn_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(vcn_pos_weight))
    
    def forward(self, vad_pred, osd_pred, vcn_pred, vad_target, osd_target, vcn_target):
        """
        Compute combined loss.
        
        Args:
            vad_pred: [batch, seq_len, num_speakers] VAD predictions
            osd_pred: [batch, seq_len] OSD predictions
            vcn_pred: [batch, seq_len] VCN predictions
            vad_target: [batch, seq_len, num_speakers] VAD targets
            osd_target: [batch, seq_len] OSD targets
            vcn_target: [batch, seq_len] VCN targets
        """
        # For VAD, we might need to reduce over speakers or handle differently
        # For now, assume VAD is binary per speaker
        vad_loss = 0
        if vad_pred.dim() == 3:  # [batch, seq_len, num_speakers]
            for speaker in range(vad_pred.size(-1)):
                vad_loss += self.vad_criterion(
                    vad_pred[:, :, speaker], 
                    vad_target[:, :, speaker]
                )
            vad_loss /= vad_pred.size(-1)
        else:
            vad_loss = self.vad_criterion(vad_pred, vad_target)
        
        # OSD and VCN losses
        osd_loss = self.osd_criterion(osd_pred, osd_target)
        vcn_loss = self.vcn_criterion(vcn_pred, vcn_target)
        
        # Weighted combination
        total_loss = (self.vad_weight * vad_loss + 
                     self.osd_weight * osd_loss + 
                     self.vcn_weight * vcn_loss)
        
        return total_loss, {
            'vad_loss': vad_loss.item(),
            'osd_loss': osd_loss.item(),
            'vcn_loss': vcn_loss.item(),
            'total_loss': total_loss.item()
        }


class VoxConverseTrainer:
    """Main trainer for VoxConverse diarization models."""
    
    def __init__(self, model, config: Dict):
        """
        Initialize trainer.
        
        Args:
            model: VoxConverseTCN model
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Setup progressive training if enabled
        self.progressive_strategy = None
        if config.get('use_progressive_training', False):
            self.progressive_strategy = ProgressiveTrainingStrategy(
                config['curriculum_schedule']
            )
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.train_metrics = []
        self.val_metrics = []
        
        print(f"🚀 VoxConverse Trainer initialized")
        print(f"📱 Device: {self.device}")
        print(f"🧠 Model parameters: {self.model.get_num_params():,}")
        print(f"📚 Progressive training: {config.get('use_progressive_training', False)}")
    
    def _create_optimizer(self):
        """Create optimizer based on config."""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        elif opt_config['type'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    def _create_loss_function(self):
        """Create loss function based on config."""
        loss_config = self.config['loss']
        
        return AdaptiveLossFunction(
            vad_weight=loss_config.get('vad_weight', 1.0),
            osd_weight=loss_config.get('osd_weight', 1.0),
            vcn_weight=loss_config.get('vcn_weight', 1.0),
            use_focal_loss=loss_config.get('use_focal_loss', False),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            class_weights=loss_config.get('class_weights', None)
        )
    
    def _create_scheduler(self, train_loader):
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config.get('max_lr', 0.01),
                epochs=self.config['num_epochs'],
                steps_per_epoch=len(train_loader),
                pct_start=scheduler_config.get('pct_start', 0.3)
            )
        elif scheduler_config.get('type') == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {'vad_loss': 0, 'osd_loss': 0, 'vcn_loss': 0}
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            features = batch['features'].to(self.device)
            vad_labels = batch['vad_labels'].to(self.device)
            osd_labels = batch['osd_labels'].to(self.device)
            vcn_labels = batch['vcn_labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            vad_pred, osd_pred, vcn_pred = self.model(features)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                vad_pred, osd_pred, vcn_pred,
                vad_labels, osd_labels, vcn_labels
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key in loss_components:
                    loss_components[key] += value
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'vad': f"{loss_dict['vad_loss']:.4f}",
                'osd': f"{loss_dict['osd_loss']:.4f}",
                'vcn': f"{loss_dict['vcn_loss']:.4f}"
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        loss_components = {'vad_loss': 0, 'osd_loss': 0, 'vcn_loss': 0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                features = batch['features'].to(self.device)
                vad_labels = batch['vad_labels'].to(self.device)
                osd_labels = batch['osd_labels'].to(self.device)
                vcn_labels = batch['vcn_labels'].to(self.device)
                
                # Forward pass
                vad_pred, osd_pred, vcn_pred = self.model(features)
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    vad_pred, osd_pred, vcn_pred,
                    vad_labels, osd_labels, vcn_labels
                )
                
                # Update metrics
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if key in loss_components:
                        loss_components[key] += value
        
        # Calculate average losses
        avg_loss = total_loss / len(val_loader)
        avg_components = {k: v / len(val_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def train(self, train_loader, val_loader=None, save_dir='./checkpoints'):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create scheduler
        scheduler = self._create_scheduler(train_loader)
        
        print(f"🚀 Starting training for {self.config['num_epochs']} epochs")
        print(f"💾 Checkpoints will be saved to: {save_dir}")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n📅 Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Update progressive training if enabled
            if self.progressive_strategy:
                self.progressive_strategy.update_epoch(epoch)
            
            # Training
            train_loss, train_components = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            print(f"🏃 Train Loss: {train_loss:.4f}")
            print(f"   VAD: {train_components['vad_loss']:.4f}, "
                  f"OSD: {train_components['osd_loss']:.4f}, "
                  f"VCN: {train_components['vcn_loss']:.4f}")
            
            # Validation
            if val_loader:
                val_loss, val_components = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                print(f"🎯 Val Loss: {val_loss:.4f}")
                print(f"   VAD: {val_components['vad_loss']:.4f}, "
                      f"OSD: {val_components['osd_loss']:.4f}, "
                      f"VCN: {val_components['vcn_loss']:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(save_dir, 'best_model.pth'),
                        epoch, val_loss, is_best=True
                    )
                    print("🏆 New best model saved!")
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                    epoch, train_loss
                )
        
        print(f"\n✅ Training completed!")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(
            os.path.join(save_dir, 'final_model.pth'),
            self.config['num_epochs'] - 1, 
            self.train_losses[-1]
        )
        
        # Plot training curves
        self.plot_training_curves(save_path=os.path.join(save_dir, 'training_curves.png'))
        
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, path, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'is_best': is_best
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"📂 Checkpoint loaded: {path}")
        print(f"📅 Epoch: {checkpoint['epoch']}")
        print(f"📊 Loss: {checkpoint['loss']:.4f}")
        
        return checkpoint['epoch']
    
    def plot_training_curves(self, save_path=None):
        """Plot training and validation curves."""
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate (if available)
        plt.subplot(1, 2, 2)
        if hasattr(self, 'learning_rates'):
            plt.plot(self.learning_rates, label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Training curves saved: {save_path}")
        
        plt.show()


def create_default_config():
    """Create default training configuration."""
    return {
        'num_epochs': 100,
        'optimizer': {
            'type': 'adamw',
            'learning_rate': 1e-3,
            'weight_decay': 0.01
        },
        'scheduler': {
            'type': 'onecycle',
            'max_lr': 1e-2,
            'pct_start': 0.3
        },
        'loss': {
            'vad_weight': 1.0,
            'osd_weight': 2.0,  # More weight on OSD as it's rarer
            'vcn_weight': 3.0,  # More weight on VCN as it's rarest
            'use_focal_loss': True,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'class_weights': None  # Will be calculated from data analysis
        },
        'gradient_clip': 1.0,
        'save_every': 10,
        'use_progressive_training': False,
        'curriculum_schedule': [
            (0, 10.0, 1.0),    # Start with 10s segments
            (20, 30.0, 1.5),   # Increase to 30s at epoch 20
            (50, 60.0, 2.0),   # Full 60s segments at epoch 50
        ]
    }