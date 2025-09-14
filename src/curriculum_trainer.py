import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple
import warnings

from progressive_training import ProgressiveSegmentDataset, ConcatenatedTrainingStrategy
from long_range_tcn import LongRangeDiarizationModel
from simple_losses import create_loss_function
from simple_metrics import create_metrics

class CurriculumDiarizationTrainer:
    """Entraîneur avec curriculum learning pour la diarisation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Stratégies d'entraînement progressif
        self.curriculum_config = config['curriculum']
        self.use_progressive_segments = config.get('use_progressive_segments', True)
        self.use_concatenated_training = config.get('use_concatenated_training', False)
        
        # Modèle adaptatif
        self.model = self._create_adaptive_model()
        self.model.to(self.device)
        
        # Optimiseur et scheduler adaptatifs
        self.optimizer = self._create_optimizer()
        self.criterion = create_loss_function(config['loss'])
        self.metrics = create_metrics(cuda_optimized=True)
        
        # Tracking
        self.curriculum_history = []
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        print(f"🎓 Curriculum trainer initialisé")
        print(f"📊 Segments progressifs: {self.use_progressive_segments}")
        print(f"🔗 Entraînement concaténé: {self.use_concatenated_training}")
        
    def _create_adaptive_model(self):
        """Crée un modèle adaptatif selon la configuration."""
        if self.config['model'].get('use_long_range_tcn', False):
            return LongRangeDiarizationModel(
                input_dim=self.config['model']['input_dim'],
                num_speakers=self.config['model']['num_speakers'],
                max_sequence_length=self.curriculum_config['max_segment_duration']
            )
        else:
            # Utilise le modèle simple avec architecture adaptative
            from simple_tcn_model import SimpleDiarizationTCN
            return SimpleDiarizationTCN(
                input_dim=self.config['model']['input_dim'],
                hidden_channels=self.config['model']['hidden_channels'],
                kernel_size=self.config['model']['kernel_size'],
                num_speakers=self.config['model']['num_speakers'],
                dropout=self.config['model']['dropout']
            )
    
    def _create_optimizer(self):
        """Crée l'optimiseur avec paramètres adaptatifs."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['optimizer']['base_lr'],
            weight_decay=self.config['optimizer']['weight_decay'],
            betas=self.config['optimizer']['betas']
        )
    
    def _get_current_curriculum_stage(self, epoch):
        """Détermine l'étape actuelle du curriculum."""
        for i, (start_epoch, duration, hop_ratio) in enumerate(self.curriculum_config['schedule']):
            if epoch >= start_epoch:
                current_stage = i
            else:
                break
        
        stage_info = self.curriculum_config['schedule'][current_stage]
        return {
            'stage': current_stage,
            'epoch_start': stage_info[0],
            'segment_duration': stage_info[1],
            'hop_ratio': stage_info[2]
        }
    
    def _update_learning_rate(self, epoch, stage_info):
        """Met à jour le learning rate selon l'étape du curriculum."""
        base_lr = self.config['optimizer']['base_lr']
        
        # Réduit le LR pour les segments plus longs (plus difficiles)
        stage_multiplier = 1.0 / (1.0 + 0.1 * stage_info['stage'])
        
        # Warmup pour chaque nouvelle étape
        epochs_in_stage = epoch - stage_info['epoch_start']
        warmup_epochs = 2
        
        if epochs_in_stage < warmup_epochs:
            lr_multiplier = 0.5 + 0.5 * (epochs_in_stage / warmup_epochs)
        else:
            lr_multiplier = 1.0
        
        new_lr = base_lr * stage_multiplier * lr_multiplier
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return new_lr
    
    def _create_curriculum_dataloader(self, epoch, split='train'):
        """Crée un dataloader adapté à l'étape actuelle du curriculum."""
        stage_info = self._get_current_curriculum_stage(epoch)
        
        # Import de la fonction de création de dataloader
        from voxconverse_dataset import create_voxconverse_dataloaders
        
        # Paramètres adaptés à l'étape
        dataloader_config = {
            'batch_size': self._get_adaptive_batch_size(stage_info['segment_duration']),
            'segment_duration': stage_info['segment_duration'],
            'hop_duration': stage_info['segment_duration'] * stage_info['hop_ratio'],
            'num_workers': self.config['training']['num_workers'],
            'pin_memory': True,
            'persistent_workers': True
        }
        
        if split == 'train':
            train_loader, _ = create_voxconverse_dataloaders(**dataloader_config)
            return train_loader
        else:
            # Pour validation, utilise toujours la même configuration
            val_config = dataloader_config.copy()
            val_config['segment_duration'] = 8.0  # Configuration fixe pour validation
            val_config['batch_size'] = min(val_config['batch_size'], 8)
            _, val_loader = create_voxconverse_dataloaders(**val_config)
            return val_loader
    
    def _get_adaptive_batch_size(self, segment_duration):
        """Calcule la taille de batch adaptée à la durée des segments."""
        base_batch_size = self.config['training']['base_batch_size']
        
        # Réduit la taille de batch pour les segments plus longs
        if segment_duration <= 4.0:
            return base_batch_size
        elif segment_duration <= 8.0:
            return max(4, base_batch_size // 2)
        elif segment_duration <= 16.0:
            return max(2, base_batch_size // 4)
        else:
            return max(1, base_batch_size // 8)
    
    def train_epoch_curriculum(self, epoch):
        """Entraîne une époque avec curriculum learning."""
        self.model.train()
        
        # Met à jour la configuration du curriculum
        stage_info = self._get_current_curriculum_stage(epoch)
        current_lr = self._update_learning_rate(epoch, stage_info)
        
        # Crée le dataloader pour cette étape
        train_loader = self._create_curriculum_dataloader(epoch, 'train')
        
        print(f"\n📚 Epoch {epoch} - Stage {stage_info['stage']}")
        print(f"   Segments: {stage_info['segment_duration']:.1f}s")
        print(f"   Hop ratio: {stage_info['hop_ratio']:.1%}")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Learning rate: {current_lr:.6f}")
        
        # Entraînement standard
        total_loss = 0
        total_vad_loss = 0
        total_osd_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Stage {stage_info["stage"]} Training')
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            vad_labels = batch['vad_labels'].to(self.device)
            osd_labels = batch['osd_labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            vad_pred, osd_pred = self.model(features)
            loss_dict = self.criterion(vad_pred, osd_pred, vad_labels, osd_labels)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping adaptatif
            max_grad_norm = 1.0 + 0.5 * stage_info['stage']  # Plus de clipping pour stages avancés
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            total_vad_loss += loss_dict['vad_loss'].item()
            total_osd_loss += loss_dict['osd_loss'].item()
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'VAD': f"{loss_dict['vad_loss'].item():.4f}",
                'OSD': f"{loss_dict['osd_loss'].item():.4f}",
                'LR': f"{current_lr:.6f}"
            })
        
        # Sauvegarde les informations du curriculum
        self.curriculum_history.append({
            'epoch': epoch,
            'stage': stage_info['stage'],
            'segment_duration': stage_info['segment_duration'],
            'hop_ratio': stage_info['hop_ratio'],
            'learning_rate': current_lr,
            'batch_size': train_loader.batch_size,
            'train_loss': total_loss / num_batches
        })
        
        return {
            'loss': total_loss / num_batches,
            'vad_loss': total_vad_loss / num_batches,
            'osd_loss': total_osd_loss / num_batches,
            'stage_info': stage_info
        }
    
    def validate_epoch(self, epoch):
        """Validation avec segments de référence."""
        self.model.eval()
        
        val_loader = self._create_curriculum_dataloader(epoch, 'val')
        
        total_loss = 0
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
                
                vad_pred, osd_pred = self.model(features)
                loss_dict = self.criterion(vad_pred, osd_pred, vad_labels, osd_labels)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                
                # Collecte pour métriques
                all_vad_preds.append(vad_pred.cpu())
                all_vad_targets.append(vad_labels.cpu())
                all_osd_preds.append(osd_pred.cpu())
                all_osd_targets.append(osd_labels.cpu())
        
        # Calcul des métriques
        vad_preds = torch.cat(all_vad_preds, dim=0)
        vad_targets = torch.cat(all_vad_targets, dim=0)
        osd_preds = torch.cat(all_osd_preds, dim=0)
        osd_targets = torch.cat(all_osd_targets, dim=0)
        
        metrics = self.metrics.compute_metrics_cuda(vad_preds, osd_preds, vad_targets, osd_targets)
        
        return {
            'loss': total_loss / num_batches,
            **metrics
        }
    
    def train_curriculum(self, num_epochs):
        """Entraînement complet avec curriculum learning."""
        print(f"🚀 Début de l'entraînement curriculum ({num_epochs} epochs)")
        
        for epoch in range(num_epochs):
            # Entraînement
            train_results = self.train_epoch_curriculum(epoch)
            self.train_losses.append(train_results['loss'])
            
            # Validation
            val_results = self.validate_epoch(epoch)
            self.val_losses.append(val_results['loss'])
            
            # Logging
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_results['loss']:.6f}")
            print(f"  Val Loss: {val_results['loss']:.6f}")
            print(f"  DER: {val_results.get('der', 'N/A'):.3f}")
            
            # Sauvegarde du meilleur modèle
            if val_results['loss'] < self.best_val_loss:
                self.best_val_loss = val_results['loss']
                self._save_checkpoint(epoch, val_results['loss'], is_best=True)
            
            # Sauvegarde régulière
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, val_results['loss'], is_best=False)
        
        print(f"✅ Entraînement terminé!")
        print(f"🏆 Meilleur validation loss: {self.best_val_loss:.6f}")
        
        # Sauvegarde des courbes et historique
        self._save_training_curves()
        self._save_curriculum_report()
    
    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        """Sauvegarde un checkpoint."""
        save_dir = self.config['training']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'curriculum_history': self.curriculum_history
        }
        
        if is_best:
            path = os.path.join(save_dir, 'curriculum_model_best.pth')
        else:
            path = os.path.join(save_dir, f'curriculum_model_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
    
    def _save_training_curves(self):
        """Sauvegarde les courbes d'entraînement."""
        plt.figure(figsize=(15, 10))
        
        # Courbes de loss
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Curves')
        plt.grid(True, alpha=0.3)
        
        # Évolution du curriculum
        plt.subplot(2, 3, 2)
        epochs = [h['epoch'] for h in self.curriculum_history]
        durations = [h['segment_duration'] for h in self.curriculum_history]
        plt.plot(epochs, durations, 'o-', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Segment Duration (s)')
        plt.title('Curriculum Progression')
        plt.grid(True, alpha=0.3)
        
        # Évolution du learning rate
        plt.subplot(2, 3, 3)
        lrs = [h['learning_rate'] for h in self.curriculum_history]
        plt.plot(epochs, lrs, 'o-', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Évolution de la taille de batch
        plt.subplot(2, 3, 4)
        batch_sizes = [h['batch_size'] for h in self.curriculum_history]
        plt.plot(epochs, batch_sizes, 'o-', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Batch Size')
        plt.title('Adaptive Batch Size')
        plt.grid(True, alpha=0.3)
        
        # Stages du curriculum
        plt.subplot(2, 3, 5)
        stages = [h['stage'] for h in self.curriculum_history]
        plt.plot(epochs, stages, 'o-', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Curriculum Stage')
        plt.title('Curriculum Stages')
        plt.grid(True, alpha=0.3)
        
        # Loss par stage
        plt.subplot(2, 3, 6)
        train_losses = [h['train_loss'] for h in self.curriculum_history]
        colors = plt.cm.viridis(np.array(stages) / max(stages))
        plt.scatter(epochs, train_losses, c=colors, alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.title('Loss by Curriculum Stage')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.config['training']['save_dir'], 'curriculum_training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Courbes sauvegardées: {save_path}")
    
    def _save_curriculum_report(self):
        """Sauvegarde un rapport détaillé du curriculum."""
        report = {
            'config': self.config,
            'curriculum_history': self.curriculum_history,
            'final_metrics': {
                'best_val_loss': self.best_val_loss,
                'total_epochs': len(self.train_losses),
                'final_train_loss': self.train_losses[-1] if self.train_losses else None,
                'final_val_loss': self.val_losses[-1] if self.val_losses else None
            },
            'curriculum_summary': self._analyze_curriculum_performance()
        }
        
        save_path = os.path.join(self.config['training']['save_dir'], 'curriculum_report.json')
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Rapport sauvegardé: {save_path}")
    
    def _analyze_curriculum_performance(self):
        """Analyse les performances par stage du curriculum."""
        stages_analysis = {}
        
        for stage in range(len(self.curriculum_config['schedule'])):
            stage_history = [h for h in self.curriculum_history if h['stage'] == stage]
            
            if stage_history:
                stages_analysis[f'stage_{stage}'] = {
                    'epochs': len(stage_history),
                    'segment_duration': stage_history[0]['segment_duration'],
                    'avg_train_loss': np.mean([h['train_loss'] for h in stage_history]),
                    'loss_improvement': stage_history[0]['train_loss'] - stage_history[-1]['train_loss'],
                    'convergence_rate': self._compute_convergence_rate(stage_history)
                }
        
        return stages_analysis
    
    def _compute_convergence_rate(self, stage_history):
        """Calcule le taux de convergence pour un stage."""
        if len(stage_history) < 2:
            return 0.0
        
        losses = [h['train_loss'] for h in stage_history]
        # Calcule la pente moyenne de décroissance
        epochs = list(range(len(losses)))
        slope = np.polyfit(epochs, losses, 1)[0]
        return -slope  # Pente négative = convergence positive


def create_curriculum_config():
    """Crée une configuration d'entraînement avec curriculum."""
    
    return {
        'model': {
            'input_dim': 80,
            'hidden_channels': [128, 256, 256, 512, 512],
            'kernel_size': 3,
            'num_speakers': 4,
            'dropout': 0.1,
            'use_long_range_tcn': True  # Utilise le modèle amélioré
        },
        'optimizer': {
            'base_lr': 0.001,
            'weight_decay': 0.0001,
            'betas': [0.9, 0.999]
        },
        'loss': {
            'type': 'simple',
            'vad_weight': 1.0,
            'osd_weight': 1.2,
            'focal_gamma': 2.0,
            'focal_alpha': 0.25,
            'label_smoothing': 0.05
        },
        'curriculum': {
            'max_segment_duration': 30.0,
            'schedule': [
                (0, 2.0, 0.5),    # Epochs 0-4: 2s segments
                (5, 4.0, 0.4),    # Epochs 5-9: 4s segments  
                (10, 8.0, 0.3),   # Epochs 10-14: 8s segments
                (15, 16.0, 0.2),  # Epochs 15-19: 16s segments
                (20, 30.0, 0.1),  # Epochs 20+: 30s segments
            ]
        },
        'training': {
            'base_batch_size': 16,
            'num_workers': 4,
            'save_dir': '/root/ai-project/Speaker-diarization-/curriculum_checkpoints'
        },
        'use_progressive_segments': True,
        'use_concatenated_training': False
    }


if __name__ == "__main__":
    print("🎓 Test du curriculum learning pour la diarisation")
    
    config = create_curriculum_config()
    trainer = CurriculumDiarizationTrainer(config)
    
    # Test rapide
    print("\n📊 Configuration du curriculum:")
    for i, (epoch_start, duration, hop_ratio) in enumerate(config['curriculum']['schedule']):
        print(f"  Stage {i}: Epoch {epoch_start}+ → {duration}s segments (hop {hop_ratio*100:.0f}%)")
    
    print("\n✅ Entraîneur curriculum prêt!")
    print("Pour lancer l'entraînement complet:")
    print("trainer.train_curriculum(num_epochs=30)")