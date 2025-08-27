import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import warnings
import psutil
import gc
from pathlib import Path

warnings.filterwarnings("ignore")

from tcn_diarization_model import DiarizationTCN, EnsembleDiarizationTCN
from optimized_dataloader import create_optimized_dataloaders, MemoryAwareDataLoader
from diarization_losses import create_loss_function, MultiTaskDiarizationLoss
from metrics import DiarizationMetrics


class ImprovedDiarizationTrainer:
    """
    Enhanced trainer for speaker diarization with advanced features:
    - Memory-efficient training
    - Speaker classification
    - Adaptive optimization
    - Real-time monitoring
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with speaker classifier
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Model parallelism if multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # Initialize optimizer with layer-wise learning rates
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss functions
        self.criterion = self._create_loss_functions()
        
        # Initialize metrics
        self.metrics = DiarizationMetrics(num_speakers=config['model']['num_speakers'])
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Early stopping with advanced logic
        self.early_stopping = ImprovedEarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.001),
            restore_best_weights=True
        )
        
        # Tracking and visualization
        self.train_losses = []
        self.val_losses = []
        self.speaker_accuracies = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.global_step = 0
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        # Initialize wandb if configured
        if config.get('use_wandb', False):
            self._init_wandb()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.get('save_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Model initialized with {self.model.get_num_params():,} parameters")
        print(f"Training on device: {self.device}")
        print(f"Using mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")
    
    def _create_model(self) -> nn.Module:
        """Create model with enhanced configuration."""
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
                use_attention=model_config.get('use_attention', True),
                use_speaker_classifier=model_config.get('use_speaker_classifier', True),
                embedding_dim=model_config.get('embedding_dim', 256)
            )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with layer-wise learning rates."""
        opt_config = self.config['optimizer']
        base_lr = opt_config.get('lr', 1e-3)
        
        # Group parameters by layer type for different learning rates
        tcn_params = []
        attention_params = []
        classifier_params = []
        other_params = []
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name, param in model.named_parameters():
            if 'tcn' in name:
                tcn_params.append(param)
            elif 'attention' in name:
                attention_params.append(param)
            elif 'speaker_classifier' in name or 'similarity_head' in name:
                classifier_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': tcn_params, 'lr': base_lr * 0.5, 'name': 'tcn'},
            {'params': attention_params, 'lr': base_lr, 'name': 'attention'},
            {'params': classifier_params, 'lr': base_lr * 2.0, 'name': 'classifier'},
            {'params': other_params, 'lr': base_lr, 'name': 'other'}
        ]
        
        # Filter out empty groups
        param_groups = [group for group in param_groups if group['params']]
        
        opt_type = opt_config.get('type', 'adamw')
        
        if opt_type.lower() == 'adamw':
            return optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=opt_config.get('weight_decay', 1e-4),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type.lower() == 'adam':
            return optim.Adam(
                param_groups,
                lr=base_lr,
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create advanced learning rate scheduler."""
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'onecycle')
        
        if sched_type == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=[group['lr'] for group in self.optimizer.param_groups],
                steps_per_epoch=sched_config.get('steps_per_epoch', 100),
                epochs=self.config['training']['epochs'],
                pct_start=sched_config.get('pct_start', 0.3)
            )
        elif sched_type == 'cosine':
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
    
    def _create_loss_functions(self) -> Dict[str, nn.Module]:
        """Create multiple loss functions for multi-task training."""
        loss_config = self.config['loss']
        
        losses = {
            'main': create_loss_function(loss_config)
        }
        
        # Add speaker classification loss if enabled
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model, 'use_speaker_classifier') and model.use_speaker_classifier:
            losses['speaker_classification'] = nn.CrossEntropyLoss()
            losses['speaker_similarity'] = nn.BCELoss()
        
        return losses
    
    def _init_wandb(self):
        """Initialize enhanced Weights & Biases logging."""
        wandb.init(
            project=self.config.get('project_name', 'enhanced-speaker-diarization'),
            config=self.config,
            name=f"tcn-diarization-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            tags=['tcn', 'diarization', 'multi-channel']
        )
        wandb.watch(self.model, log_freq=100, log_graph=True)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Enhanced training epoch with gradient accumulation and memory management."""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'vad_loss': 0.0,
            'osd_loss': 0.0,
            'speaker_loss': 0.0,
            'similarity_loss': 0.0,
            'memory_usage': 0.0,
            'batches_processed': 0
        }
        
        # Progress bar with memory info
        pbar = tqdm(train_loader, desc='Training')
        
        accumulated_loss = 0
        num_accumulated = 0
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Memory monitoring
                memory_info = self.memory_monitor.get_memory_info()
                epoch_metrics['memory_usage'] += memory_info['gpu_percent']
                
                # Move data to device
                features = batch['features'].to(self.device)
                vad_labels = batch['vad_labels'].to(self.device)
                osd_labels = batch['osd_labels'].to(self.device)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp):
                    # Get model outputs
                    model_outputs = self.model(features, return_embeddings=True)
                    
                    if len(model_outputs) == 4:
                        vad_pred, osd_pred, embeddings, speaker_logits = model_outputs
                    else:
                        vad_pred, osd_pred = model_outputs[:2]
                        embeddings, speaker_logits = None, None
                    
                    # Compute main diarization loss
                    loss_dict = self.criterion['main'](vad_pred, osd_pred, vad_labels, osd_labels)
                    total_loss = loss_dict['total_loss']
                    
                    # Add speaker classification loss if available
                    if speaker_logits is not None and 'speaker_classification' in self.criterion:
                        # Create speaker labels from VAD labels (simplified)
                        speaker_targets = torch.argmax(vad_labels.sum(dim=1), dim=1)
                        speaker_loss = self.criterion['speaker_classification'](speaker_logits, speaker_targets)
                        total_loss += self.config.get('speaker_loss_weight', 0.5) * speaker_loss
                        epoch_metrics['speaker_loss'] += speaker_loss.item()
                    
                    # Normalize loss for accumulation
                    total_loss = total_loss / self.accumulation_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                accumulated_loss += total_loss.item()
                num_accumulated += 1
                
                # Update weights after accumulation steps
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if self.use_amp:
                        # Gradient clipping
                        if self.config.get('grad_clip_norm', 0) > 0:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   self.config['grad_clip_norm'])
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Gradient clipping
                        if self.config.get('grad_clip_norm', 0) > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   self.config['grad_clip_norm'])
                        
                        self.optimizer.step()
                    
                    # Update scheduler if OneCycleLR
                    if isinstance(self.scheduler, OneCycleLR):
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Log accumulated loss
                    avg_accumulated_loss = accumulated_loss * self.accumulation_steps
                    epoch_metrics['total_loss'] += avg_accumulated_loss
                    epoch_metrics['vad_loss'] += loss_dict.get('vad_loss', 0)
                    epoch_metrics['osd_loss'] += loss_dict.get('osd_loss', 0)
                    epoch_metrics['batches_processed'] += 1
                    
                    # Reset accumulation
                    accumulated_loss = 0
                    num_accumulated = 0
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Loss': f"{avg_accumulated_loss:.4f}",
                        'GPU': f"{memory_info['gpu_percent']:.1f}%",
                        'RAM': f"{memory_info['ram_percent']:.1f}%"
                    })
                    
                    # Log to wandb
                    if self.config.get('use_wandb', False) and self.global_step % 50 == 0:
                        log_dict = {
                            'train/batch_loss': avg_accumulated_loss,
                            'train/vad_loss': loss_dict.get('vad_loss', 0),
                            'train/osd_loss': loss_dict.get('osd_loss', 0),
                            'train/lr': self.optimizer.param_groups[0]['lr'],
                            'train/gpu_memory_percent': memory_info['gpu_percent'],
                            'train/ram_memory_percent': memory_info['ram_percent'],
                            'global_step': self.global_step
                        }
                        
                        if 'speaker_loss' in epoch_metrics and epoch_metrics['speaker_loss'] > 0:
                            log_dict['train/speaker_loss'] = epoch_metrics['speaker_loss'] / epoch_metrics['batches_processed']
                        
                        wandb.log(log_dict)
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                # Clear cache and continue
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                continue
        
        # Average metrics
        if epoch_metrics['batches_processed'] > 0:
            for key in ['total_loss', 'vad_loss', 'osd_loss', 'speaker_loss', 'memory_usage']:
                epoch_metrics[key] /= epoch_metrics['batches_processed']
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Enhanced validation with detailed metrics."""
        self.model.eval()
        
        val_metrics = {
            'total_loss': 0.0,
            'vad_loss': 0.0,
            'osd_loss': 0.0,
            'speaker_accuracy': 0.0,
            'batches_processed': 0
        }
        
        all_vad_preds = []
        all_vad_targets = []
        all_osd_preds = []
        all_osd_targets = []
        all_speaker_preds = []
        all_speaker_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    features = batch['features'].to(self.device)
                    vad_labels = batch['vad_labels'].to(self.device)
                    osd_labels = batch['osd_labels'].to(self.device)
                    
                    # Forward pass
                    with autocast(enabled=self.use_amp):
                        model_outputs = self.model(features, return_embeddings=True)
                        
                        if len(model_outputs) == 4:
                            vad_pred, osd_pred, embeddings, speaker_logits = model_outputs
                        else:
                            vad_pred, osd_pred = model_outputs[:2]
                            speaker_logits = None
                        
                        # Compute losses
                        loss_dict = self.criterion['main'](vad_pred, osd_pred, vad_labels, osd_labels)
                        total_loss = loss_dict['total_loss']
                        
                        # Speaker classification accuracy
                        if speaker_logits is not None:
                            speaker_targets = torch.argmax(vad_labels.sum(dim=1), dim=1)
                            speaker_accuracy = (torch.argmax(speaker_logits, dim=1) == speaker_targets).float().mean()
                            val_metrics['speaker_accuracy'] += speaker_accuracy.item()
                            
                            all_speaker_preds.extend(torch.argmax(speaker_logits, dim=1).cpu().numpy())
                            all_speaker_targets.extend(speaker_targets.cpu().numpy())
                    
                    # Update metrics
                    val_metrics['total_loss'] += total_loss.item()
                    val_metrics['vad_loss'] += loss_dict.get('vad_loss', 0)
                    val_metrics['osd_loss'] += loss_dict.get('osd_loss', 0)
                    val_metrics['batches_processed'] += 1
                    
                    # Collect predictions for detailed metrics
                    all_vad_preds.append(vad_pred.cpu())
                    all_vad_targets.append(vad_labels.cpu())
                    all_osd_preds.append(osd_pred.cpu())
                    all_osd_targets.append(osd_labels.cpu())
                    
                    pbar.set_postfix({
                        'Loss': f"{total_loss.item():.4f}",
                        'SpkAcc': f"{val_metrics['speaker_accuracy']/(batch_idx+1):.3f}" if speaker_logits is not None else "N/A"
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Average metrics
        if val_metrics['batches_processed'] > 0:
            for key in ['total_loss', 'vad_loss', 'osd_loss', 'speaker_accuracy']:
                val_metrics[key] /= val_metrics['batches_processed']
        
        # Compute detailed diarization metrics
        if all_vad_preds and all_vad_targets:
            vad_preds = torch.cat(all_vad_preds, dim=0)
            vad_targets = torch.cat(all_vad_targets, dim=0)
            osd_preds = torch.cat(all_osd_preds, dim=0)
            osd_targets = torch.cat(all_osd_targets, dim=0)
            
            detailed_metrics = self.metrics.compute_metrics(
                vad_preds, osd_preds, vad_targets, osd_targets
            )
            val_metrics.update(detailed_metrics)
        
        return val_metrics


class ImprovedEarlyStopping:
    """Enhanced early stopping with multiple criteria."""
    
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.best_epoch = 0
    
    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        should_stop = self.counter >= self.patience
        
        if should_stop and self.restore_best_weights and self.best_weights:
            print(f"Restoring best weights from epoch {self.best_epoch}")
            model.load_state_dict(self.best_weights)
        
        return should_stop


class MemoryMonitor:
    """Monitor system and GPU memory usage."""
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {}
        
        # System RAM
        ram = psutil.virtual_memory()
        info['ram_percent'] = ram.percent
        info['ram_used_gb'] = ram.used / (1024**3)
        info['ram_total_gb'] = ram.total / (1024**3)
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_cached = torch.cuda.memory_reserved()
            
            info['gpu_percent'] = (gpu_allocated / gpu_memory) * 100
            info['gpu_allocated_gb'] = gpu_allocated / (1024**3)
            info['gpu_cached_gb'] = gpu_cached / (1024**3)
            info['gpu_total_gb'] = gpu_memory / (1024**3)
        else:
            info['gpu_percent'] = 0
            info['gpu_allocated_gb'] = 0
            info['gpu_cached_gb'] = 0
            info['gpu_total_gb'] = 0
        
        return info


if __name__ == "__main__":
    # Test the improved trainer
    config = {
        'model': {
            'input_dim': 771,
            'hidden_channels': [256, 256, 256, 512, 512],
            'kernel_size': 3,
            'num_speakers': 4,
            'dropout': 0.2,
            'use_attention': True,
            'use_speaker_classifier': True,
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
            'type': 'onecycle',
            'steps_per_epoch': 100,
            'pct_start': 0.3
        },
        'training': {
            'epochs': 100,
            'batch_size': 8
        },
        'accumulation_steps': 4,
        'use_amp': True,
        'grad_clip_norm': 1.0,
        'patience': 15,
        'save_dir': './enhanced_checkpoints',
        'use_wandb': False
    }
    
    print("Testing ImprovedDiarizationTrainer...")
    trainer = ImprovedDiarizationTrainer(config)
    print("Trainer initialized successfully!")
    
    # Test memory monitor
    memory_monitor = MemoryMonitor()
    memory_info = memory_monitor.get_memory_info()
    print(f"Current memory usage: RAM {memory_info['ram_percent']:.1f}%, GPU {memory_info['gpu_percent']:.1f}%")