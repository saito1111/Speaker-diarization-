import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from tcn_diarization_model import DiarizationTCN, EnsembleDiarizationTCN
from optimized_dataloader import create_optimized_dataloaders
from diarization_losses import create_loss_function
from metrics import DiarizationMetrics
from improved_trainer import ImprovedDiarizationTrainer


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class DiarizationTrainer:
    """Comprehensive trainer for speaker diarization models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = create_loss_function(config['loss'])
        
        # Initialize metrics
        self.metrics = DiarizationMetrics(num_speakers=config['model']['num_speakers'])
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        # Initialize wandb if configured
        if config.get('use_wandb', False):
            self._init_wandb()
        
        print(f"Model initialized with {self.model.get_num_params():,} parameters")
        print(f"Training on device: {self.device}")
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_config = self.config['model']
        
        if model_config.get('use_ensemble', False):
            return EnsembleDiarizationTCN(
                input_dim=model_config['input_dim'],
                num_models=model_config.get('num_models', 3),
                hidden_channels=model_config.get('hidden_channels', [256, 256, 256, 512, 512]),
                kernel_size=model_config.get('kernel_size', 3),
                num_speakers=model_config['num_speakers'],
                dropout=model_config.get('dropout', 0.2),
                use_attention=model_config.get('use_attention', True)
            )
        else:
            return DiarizationTCN(
                input_dim=model_config['input_dim'],
                hidden_channels=model_config.get('hidden_channels', [256, 256, 256, 512, 512]),
                kernel_size=model_config.get('kernel_size', 3),
                num_speakers=model_config['num_speakers'],
                dropout=model_config.get('dropout', 0.2),
                use_attention=model_config.get('use_attention', True)
            )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config['optimizer']
        opt_type = opt_config.get('type', 'adamw')
        
        if opt_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config.get('lr', 1e-3),
                weight_decay=opt_config.get('weight_decay', 1e-4),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config.get('lr', 1e-3),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'cosine')
        
        if sched_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 5),
                min_lr=sched_config.get('min_lr', 1e-6)
            )
        else:
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.get('project_name', 'speaker-diarization'),
            config=self.config,
            name=f"tcn-diarization-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        wandb.watch(self.model, log_freq=100)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_vad_loss = 0
        total_osd_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            features = batch['features'].to(self.device)  # [batch, num_features, time]
            vad_labels = batch['vad_labels'].to(self.device)  # [batch, time, num_speakers]
            osd_labels = batch['osd_labels'].to(self.device)  # [batch, time]
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    vad_pred, osd_pred = self.model(features)
                    loss_dict = self.criterion(vad_pred, osd_pred, vad_labels, osd_labels)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                vad_pred, osd_pred = self.model(features)
                loss_dict = self.criterion(vad_pred, osd_pred, vad_labels, osd_labels)
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm', 0) > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_vad_loss += loss_dict.get('vad_loss', 0)
            total_osd_loss += loss_dict.get('osd_loss', 0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'VAD': f"{loss_dict.get('vad_loss', 0):.4f}",
                'OSD': f"{loss_dict.get('osd_loss', 0):.4f}"
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 100 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/vad_loss': loss_dict.get('vad_loss', 0),
                    'train/osd_loss': loss_dict.get('osd_loss', 0),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        return {
            'loss': total_loss / num_batches,
            'vad_loss': total_vad_loss / num_batches,
            'osd_loss': total_osd_loss / num_batches
        }
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
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
                # Move data to device
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
                total_vad_loss += loss_dict.get('vad_loss', 0)
                total_osd_loss += loss_dict.get('osd_loss', 0)
                
                # Collect predictions for metrics
                all_vad_preds.append(vad_pred.cpu())
                all_vad_targets.append(vad_labels.cpu())
                all_osd_preds.append(osd_pred.cpu())
                all_osd_targets.append(osd_labels.cpu())
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'VAD': f"{loss_dict.get('vad_loss', 0):.4f}",
                    'OSD': f"{loss_dict.get('osd_loss', 0):.4f}"
                })
        
        # Compute metrics
        vad_preds = torch.cat(all_vad_preds, dim=0)
        vad_targets = torch.cat(all_vad_targets, dim=0)
        osd_preds = torch.cat(all_osd_preds, dim=0)
        osd_targets = torch.cat(all_osd_targets, dim=0)
        
        metrics = self.metrics.compute_metrics(vad_preds, osd_preds, vad_targets, osd_targets)
        
        return {
            'loss': total_loss / num_batches,
            'vad_loss': total_vad_loss / num_batches,
            'osd_loss': total_osd_loss / num_batches,
            **metrics
        }
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            
        # Save periodic checkpoints
        if epoch % self.config.get('save_every', 10) == 0:
            periodic_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['loss'])
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"DER: {val_metrics.get('der', 0):.3f}")
            print(f"F1 Score: {val_metrics.get('f1_score', 0):.3f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss'],
                    'val/der': val_metrics.get('der', 0),
                    'val/f1_score': val_metrics.get('f1_score', 0),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(epoch, val_metrics['loss'], is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        print("Training completed!")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['save_dir'], 'training_curves.png'))
        plt.show()


def main():
    """Enhanced main training function with improved configuration."""
    
    # Enhanced configuration with new features
    config = {
        'model': {
            'input_dim': 771,  # 257 * 3 (LPS + IPD + AF)
            'hidden_channels': [256, 256, 256, 512, 512],
            'kernel_size': 3,
            'num_speakers': 4,
            'dropout': 0.2,
            'use_attention': True,
            'use_ensemble': False,
            'use_speaker_classifier': True,  # NEW: Enable speaker classification
            'embedding_dim': 256
        },
        'loss': {
            'type': 'multitask',
            'vad_weight': 1.0,
            'osd_weight': 1.0,
            'consistency_weight': 0.1,
            'use_pit': True,
            'use_focal': True,
            'focal_gamma': 2.0,
            'num_speakers': 4
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'type': 'onecycle',  # IMPROVED: Use OneCycleLR for better convergence
            'steps_per_epoch': 100,  # Will be updated based on actual data
            'pct_start': 0.3
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'num_workers': 4
        },
        'data': {
            'audio_dir': './annot_prepare/results/meta',  # Default to available data
            'rttm_dir': './annot_prepare/results/rttm',
            'segment_duration': 4.0,
            'sample_rate': 16000,
            'train_split': 0.8,
            'max_segments': None
        },
        'save_dir': './enhanced_checkpoints',
        'use_amp': True,
        'grad_clip_norm': 1.0,
        'patience': 15,  # Increased patience for better training
        'save_every': 10,
        'use_wandb': False,
        
        # NEW: Advanced training features
        'accumulation_steps': 2,  # Gradient accumulation for effective larger batch size
        'memory_threshold': 0.8,  # Memory management threshold
        'adaptive_batch': True,   # Adaptive batch sizing
        'speaker_loss_weight': 0.5,  # Weight for speaker classification loss
    }
    
    # Auto-detect data paths if they exist
    import os
    if os.path.exists('./annot_prepare/results/meta'):
        # Use the prepared AMI corpus data
        print("Using AMI corpus data from annot_prepare/results/")
        # Look for audio files in meta directory
        meta_files = os.listdir('./annot_prepare/results/meta')
        if any(f.endswith('.json') for f in meta_files):
            print(f"Found {len([f for f in meta_files if f.endswith('.json')])} JSON metadata files")
    else:
        print("Warning: Default data path not found. Please update audio_dir and rttm_dir in config")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create enhanced data loaders with memory management
    try:
        train_loader, val_loader = create_optimized_dataloaders(
            audio_dir=config['data']['audio_dir'],
            rttm_dir=config['data']['rttm_dir'],
            batch_size=config['training']['batch_size'],
            train_split=config['data']['train_split'],
            num_workers=config['training']['num_workers'],
            segment_duration=config['data']['segment_duration'],
            sample_rate=config['data']['sample_rate'],
            max_segments=config['data']['max_segments'],
            memory_threshold=config['memory_threshold'],
            adaptive_batch=config['adaptive_batch'],
            accumulation_steps=config['accumulation_steps']
        )
        
        print(f"Enhanced train batches: {len(train_loader)}")
        print(f"Enhanced val batches: {len(val_loader)}")
        
        # Update scheduler steps per epoch
        config['scheduler']['steps_per_epoch'] = len(train_loader)
        
        # Initialize enhanced trainer
        trainer = ImprovedDiarizationTrainer(config)
        
        # Start enhanced training
        print("Starting enhanced training with:")
        print(f"- Speaker classification: {config['model']['use_speaker_classifier']}")
        print(f"- Gradient accumulation: {config['accumulation_steps']} steps")
        print(f"- Memory management: {'Enabled' if config['adaptive_batch'] else 'Disabled'}")
        print(f"- Mixed precision: {config['use_amp']}")
        
        # Training would be called here
        # trainer.train(train_loader, val_loader)  # Commented out for demo
        print("Enhanced trainer setup complete! Ready for training.")
        
    except Exception as e:
        print(f"Enhanced training setup failed: {e}")
        print("Falling back to basic training setup...")
        
        try:
            # Fallback to basic trainer
            print("Attempting basic trainer fallback...")
            basic_config = config.copy()
            basic_config['model']['use_speaker_classifier'] = False
            basic_trainer = DiarizationTrainer(basic_config)
            print("Basic trainer initialized successfully as fallback.")
            
        except Exception as e2:
            print(f"Basic trainer fallback also failed: {e2}")
            print("Please check your data paths and dependencies.")


if __name__ == "__main__":
    main()