import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import math

class ProgressiveSegmentDataset:
    """Dataset avec segments de longueur progressive pour curriculum learning."""
    
    def __init__(self, base_dataset, curriculum_schedule):
        """
        Args:
            base_dataset: Dataset VoxConverse de base
            curriculum_schedule: Liste de (epoch_start, segment_duration, hop_ratio)
        """
        self.base_dataset = base_dataset
        self.curriculum_schedule = curriculum_schedule
        self.current_epoch = 0
        self.current_segment_duration = curriculum_schedule[0][1]
        self.current_hop_ratio = curriculum_schedule[0][2]
        
    def update_epoch(self, epoch):
        """Met à jour la configuration selon l'epoch actuel."""
        self.current_epoch = epoch
        
        # Trouve la configuration appropriée pour cet epoch
        for epoch_start, duration, hop_ratio in reversed(self.curriculum_schedule):
            if epoch >= epoch_start:
                self.current_segment_duration = duration
                self.current_hop_ratio = hop_ratio
                break
                
        print(f"Epoch {epoch}: segments de {self.current_segment_duration}s, hop ratio {self.current_hop_ratio}")
        
        # Recrée les segments avec la nouvelle configuration
        self._create_progressive_segments()
    
    def _create_progressive_segments(self):
        """Crée des segments avec la durée actuelle."""
        # Cette méthode devra être intégrée dans VoxConverseDataset
        pass


class HierarchicalSegmentStrategy:
    """Stratégie de segmentation hiérarchique pour l'entraînement."""
    
    def __init__(self, max_duration=30.0, min_duration=2.0, num_levels=4):
        """
        Args:
            max_duration: Durée maximale des segments
            min_duration: Durée minimale des segments  
            num_levels: Nombre de niveaux hiérarchiques
        """
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.num_levels = num_levels
        
        # Calcule les durées pour chaque niveau
        self.level_durations = self._compute_level_durations()
        
    def _compute_level_durations(self):
        """Calcule les durées pour chaque niveau hiérarchique."""
        # Progression géométrique de min_duration à max_duration
        ratio = (self.max_duration / self.min_duration) ** (1 / (self.num_levels - 1))
        durations = []
        
        for i in range(self.num_levels):
            duration = self.min_duration * (ratio ** i)
            durations.append(duration)
            
        return durations
    
    def create_hierarchical_segments(self, audio_data, annotations, level=0):
        """
        Crée des segments hiérarchiques.
        
        Args:
            audio_data: Audio brut
            annotations: Annotations de diarisation
            level: Niveau hiérarchique (0 = plus court, num_levels-1 = plus long)
            
        Returns:
            Liste de segments et sous-segments
        """
        duration = self.level_durations[level]
        
        # Crée segments de base
        segments = self._create_base_segments(audio_data, annotations, duration)
        
        # Si pas au niveau de base, crée aussi les sous-segments
        if level > 0:
            # Divise chaque segment en sous-segments du niveau inférieur
            sub_segments = []
            for segment in segments:
                sub_segs = self.create_hierarchical_segments(
                    segment['audio'], segment['annotations'], level - 1
                )
                sub_segments.extend(sub_segs)
            
            return segments + sub_segments
        
        return segments
    
    def _create_base_segments(self, audio_data, annotations, duration):
        """Crée des segments de base avec la durée spécifiée."""
        # Implémentation similaire à VoxConverseDataset._prepare_segments
        # mais avec durée variable
        pass


class ConcatenatedTrainingStrategy:
    """
    Stratégie d'entraînement avec concaténation progressive.
    
    Principe: 1 segment → 1+2 concaténés → 1+2+3 concaténés, etc.
    """
    
    def __init__(self, base_segment_duration=2.0, max_concatenations=8):
        """
        Args:
            base_segment_duration: Durée du segment de base
            max_concatenations: Nombre maximal de segments à concaténer
        """
        self.base_duration = base_segment_duration
        self.max_concatenations = max_concatenations
        
    def create_training_schedule(self, total_epochs):
        """
        Crée un planning d'entraînement progressif.
        
        Returns:
            Liste de (epoch_start, num_segments_to_concat)
        """
        schedule = []
        epochs_per_level = max(2, total_epochs // self.max_concatenations)
        
        for i in range(1, self.max_concatenations + 1):
            epoch_start = (i - 1) * epochs_per_level
            schedule.append((epoch_start, i))
            
        return schedule
    
    def concatenate_segments(self, segments_list, num_to_concat):
        """
        Concatène des segments adjacents.
        
        Args:
            segments_list: Liste de segments individuels
            num_to_concat: Nombre de segments à concaténer
            
        Returns:
            Liste de segments concaténés
        """
        concatenated = []
        
        for i in range(0, len(segments_list) - num_to_concat + 1):
            # Prend num_to_concat segments consécutifs
            segments_to_merge = segments_list[i:i + num_to_concat]
            
            # Concatène audio et labels
            merged_segment = self._merge_segments(segments_to_merge)
            concatenated.append(merged_segment)
            
        return concatenated
    
    def _merge_segments(self, segments):
        """Fusionne plusieurs segments en un seul."""
        # Concatène audio
        audio_parts = [seg['audio'] for seg in segments]
        merged_audio = np.concatenate(audio_parts)
        
        # Concatène labels VAD et OSD
        vad_parts = [seg['vad_labels'] for seg in segments]
        osd_parts = [seg['osd_labels'] for seg in segments]
        
        merged_vad = np.concatenate(vad_parts, axis=0)
        merged_osd = np.concatenate(osd_parts, axis=0)
        
        return {
            'audio': merged_audio,
            'vad_labels': merged_vad,
            'osd_labels': merged_osd,
            'start_time': segments[0]['start_time'],
            'end_time': segments[-1]['end_time'],
            'conv_idx': segments[0]['conv_idx']
        }


class AdaptiveArchitecture(nn.Module):
    """Architecture adaptative qui s'ajuste à la longueur des segments."""
    
    def __init__(self, base_model, max_segment_length):
        super().__init__()
        self.base_model = base_model
        self.max_segment_length = max_segment_length
        
        # Ajoute des couches adaptatives pour les longues séquences
        self.long_range_layers = nn.ModuleList([
            self._create_long_range_layer(256, dilation=32),
            self._create_long_range_layer(256, dilation=64),
            self._create_long_range_layer(256, dilation=128),
        ])
        
        self.sequence_length_embedding = nn.Embedding(10, 256)  # Embedding de longueur
        
    def _create_long_range_layer(self, channels, dilation):
        """Crée une couche pour capturer les dépendances long terme."""
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, 
                     dilation=dilation, padding=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, segment_length_level=0):
        """
        Forward avec adaptation à la longueur.
        
        Args:
            x: Input features
            segment_length_level: Niveau de longueur (0=court, 9=long)
        """
        # Forward de base
        vad_out, osd_out = self.base_model(x)
        
        # Si segments longs, ajoute les couches long terme
        if segment_length_level > 3:
            # Utilise les features du bottleneck du modèle de base
            features = self.base_model.bottleneck_norm(
                self.base_model.bottleneck(self.base_model.tcn(x))
            )
            
            # Ajoute l'embedding de longueur
            length_emb = self.sequence_length_embedding(
                torch.tensor(segment_length_level).to(x.device)
            )
            features = features + length_emb.unsqueeze(-1)
            
            # Applique les couches long terme
            for layer in self.long_range_layers:
                features = features + layer(features)  # Connexion résiduelle
            
            # Redécode avec les nouvelles features
            vad_out = self.base_model.vad_decoder(features).transpose(1, 2)
            osd_out = self.base_model.osd_decoder(features).squeeze(1)
        
        return vad_out, osd_out


# Exemple d'utilisation
def create_progressive_training_config():
    """Crée une configuration d'entraînement progressif."""
    
    # Curriculum de durées progressives
    curriculum_schedule = [
        (0, 2.0, 0.5),    # Epochs 0-4: segments 2s, hop 50%
        (5, 4.0, 0.5),    # Epochs 5-9: segments 4s, hop 50%
        (10, 8.0, 0.3),   # Epochs 10-14: segments 8s, hop 30%
        (15, 16.0, 0.2),  # Epochs 15-19: segments 16s, hop 20%
        (20, 30.0, 0.1),  # Epochs 20+: segments 30s, hop 10%
    ]
    
    # Stratégie de concaténation
    concat_strategy = ConcatenatedTrainingStrategy(
        base_segment_duration=2.0,
        max_concatenations=8
    )
    
    # Planning de concaténation
    concat_schedule = concat_strategy.create_training_schedule(total_epochs=30)
    
    return {
        'curriculum_schedule': curriculum_schedule,
        'concat_schedule': concat_schedule,
        'use_adaptive_architecture': True,
        'warmup_with_short_segments': True
    }


if __name__ == "__main__":
    print("🎓 Configuration d'entraînement progressif créée")
    
    config = create_progressive_training_config()
    
    print("\n📚 Curriculum de durées:")
    for epoch_start, duration, hop_ratio in config['curriculum_schedule']:
        print(f"  Epoch {epoch_start}+: {duration}s segments, hop {hop_ratio*100:.0f}%")
    
    print("\n🔗 Planning de concaténation:")
    for epoch_start, num_concat in config['concat_schedule']:
        total_duration = num_concat * 2.0  # base_duration = 2.0s
        print(f"  Epoch {epoch_start}+: {num_concat} segments = {total_duration}s")
    
    print("\n✅ Prêt pour l'entraînement progressif!")