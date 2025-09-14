import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

from simple_tcn_model import SimpleDiarizationTCN
from voxconverse_dataset import create_voxconverse_dataloaders, create_test_dataloader
from simple_losses import create_loss_function
from simple_metrics import create_metrics


class SimpleDiarizationTrainer:
    """Simple trainer for speaker diarization."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SimpleDiarizationTCN(
            input_dim=config['model']['input_dim'],
            hidden_channels=config['model']['hidden_channels'],
            kernel_size=config['model']['kernel_size'],
            num_speakers=config['model']['num_speakers'],
            dropout=config['model']['dropout']
        )
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )
        
        # Initialize loss and metrics
        self.criterion = create_loss_function(config['loss'])
        self.metrics = create_metrics(cuda_optimized=True)
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        print(f"🎯 Simple model initialized with {self.model.get_num_params():,} parameters")
        print(f"🖥️  Training on device: {self.device}")
        print(f"⚡ Mixed precision: {self.use_amp}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_vad_loss = 0
        total_osd_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            features = batch['features'].to(self.device)  # [batch, mel_dims, time]
            vad_labels = batch['vad_labels'].to(self.device)  # [batch, time, speakers]
            osd_labels = batch['osd_labels'].to(self.device)  # [batch, time]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    vad_pred, osd_pred = self.model(features)
                    loss_dict = self.criterion(vad_pred, osd_pred, vad_labels, osd_labels)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                vad_pred, osd_pred = self.model(features)
                loss_dict = self.criterion(vad_pred, osd_pred, vad_labels, osd_labels)
                loss = loss_dict['total_loss']
                
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_vad_loss += loss_dict['vad_loss'].item()
            total_osd_loss += loss_dict['osd_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'VAD': f"{loss_dict['vad_loss'].item():.4f}",
                'OSD': f"{loss_dict['osd_loss'].item():.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'vad_loss': total_vad_loss / num_batches,
            'osd_loss': total_osd_loss / num_batches
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0
        total_vad_loss = 0
        total_osd_loss = 0
        num_batches = len(val_loader)
        
        all_vad_preds = []
        all_vad_targets = []
        all_osd_preds = []
        all_osd_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            
            for batch in pbar:
                features = batch['features'].to(self.device)
                vad_labels = batch['vad_labels'].to(self.device)
                osd_labels = batch['osd_labels'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        vad_pred, osd_pred = self.model(features)
                        loss_dict = self.criterion(vad_pred, osd_pred, vad_labels, osd_labels)
                        loss = loss_dict['total_loss']
                else:
                    vad_pred, osd_pred = self.model(features)
                    loss_dict = self.criterion(vad_pred, osd_pred, vad_labels, osd_labels)
                    loss = loss_dict['total_loss']
                
                # Update metrics
                total_loss += loss.item()
                total_vad_loss += loss_dict['vad_loss'].item()
                total_osd_loss += loss_dict['osd_loss'].item()
                
                # Collect predictions for metrics
                all_vad_preds.append(vad_pred.cpu())
                all_vad_targets.append(vad_labels.cpu())
                all_osd_preds.append(osd_pred.cpu())
                all_osd_targets.append(osd_labels.cpu())
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'VAD': f"{loss_dict['vad_loss'].item():.4f}",
                    'OSD': f"{loss_dict['osd_loss'].item():.4f}"
                })
        
        # Compute overall metrics
        vad_preds = torch.cat(all_vad_preds, dim=0)
        vad_targets = torch.cat(all_vad_targets, dim=0)
        osd_preds = torch.cat(all_osd_preds, dim=0)
        osd_targets = torch.cat(all_osd_targets, dim=0)
        
        metrics = self.metrics.compute_metrics_cuda(vad_preds, osd_preds, vad_targets, osd_targets)
        
        return {
            'loss': total_loss / num_batches,
            'vad_loss': total_vad_loss / num_batches,
            'osd_loss': total_osd_loss / num_batches,
            **metrics
        }
    
    def save_checkpoint(self, epoch: int, val_loss: float, save_dir: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(save_dir, f'simple_model_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(save_dir, 'simple_model_best.pth')
            torch.save(checkpoint, best_path)
            return True
        return False
    
    def train(self, train_loader, val_loader, epochs: int, save_dir: str):
        """Main training loop."""
        
        # Create scheduler
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['optimizer']['lr'],
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3
        )
        
        print(f"🚀 Starting simple training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['loss'])
            
            # Learning rate step
            scheduler.step()
            
            # Print epoch summary
            print(f"📊 Train Loss: {train_metrics['loss']:.4f}")
            print(f"📊 Val Loss: {val_metrics['loss']:.4f}")
            print(f"🎯 F1 Score: {val_metrics.get('f1_score', 0):.3f}")
            print(f"🎯 VAD Accuracy: {val_metrics.get('vad_accuracy', 0):.3f}")
            print(f"🎯 OSD Accuracy: {val_metrics.get('osd_accuracy', 0):.3f}")
            
            # Save checkpoint
            is_best = self.save_checkpoint(epoch, val_metrics['loss'], save_dir)
            if is_best:
                print("💾 New best model saved!")
        
        print("✅ Training completed!")
        self.plot_training_curves(save_dir)
    
    def plot_training_curves(self, save_dir):
        """Plot training curves."""
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'simple_training_curves.png'))
        plt.show()


