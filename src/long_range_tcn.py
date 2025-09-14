import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import math

class LongRangeTCN(nn.Module):
    """TCN optimisé pour les dépendances long terme en diarisation."""
    
    def __init__(self, input_dim=80, 
                 hidden_channels=[128, 256, 256, 512, 512, 512, 512],
                 kernel_size=3, dropout=0.2, max_sequence_length=30.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.max_sequence_length = max_sequence_length
        
        # Calcule les dilatations pour couvrir toute la séquence
        self.dilations = self._compute_optimal_dilations(
            len(hidden_channels), kernel_size, max_sequence_length
        )
        
        # Couches TCN avec dilatations optimisées
        self.tcn_layers = nn.ModuleList()
        
        for i, (channels, dilation) in enumerate(zip(hidden_channels, self.dilations)):
            in_channels = input_dim if i == 0 else hidden_channels[i-1]
            
            layer = EnhancedTemporalBlock(
                in_channels, channels, kernel_size, 
                dilation=dilation, dropout=dropout
            )
            self.tcn_layers.append(layer)
        
        # Attention temporelle pour les très longues séquences
        self.temporal_attention = MultiHeadTemporalAttention(
            hidden_channels[-1], num_heads=8, max_seq_len=1500  # 30s à 20ms/frame
        )
        
        # Normalisations
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.output_norm = nn.BatchNorm1d(hidden_channels[-1])
        
    def _compute_optimal_dilations(self, num_layers, kernel_size, max_duration):
        """
        Calcule les dilatations optimales pour couvrir max_duration.
        
        Objectif: que le champ réceptif de la dernière couche couvre 
        au moins max_duration secondes (30s = 1500 frames à 20ms/frame).
        """
        target_receptive_field = int(max_duration * 50)  # 50 frames/seconde
        
        # Calcule la dilatation maximale nécessaire
        # Champ réceptif ≈ Σ(kernel_size - 1) * dilation_i
        base_receptive_field = (kernel_size - 1) * num_layers
        
        if base_receptive_field >= target_receptive_field:
            # Dilatations exponentielles classiques suffisent
            return [2**i for i in range(num_layers)]
        
        # Calcule dilatations pour atteindre le champ réceptif cible
        dilations = []
        current_receptive_field = 0
        
        for i in range(num_layers):
            if i < 3:
                # Premières couches: dilatations exponentielles
                dilation = 2 ** i
            else:
                # Couches suivantes: dilatations calculées pour atteindre la cible
                remaining_layers = num_layers - i
                remaining_rf = target_receptive_field - current_receptive_field
                dilation = max(2**i, remaining_rf // (remaining_layers * (kernel_size - 1)))
            
            dilations.append(dilation)
            current_receptive_field += (kernel_size - 1) * dilation
        
        return dilations
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, time] - format standard pour les convolutions 1D
        """
        # Assure le format [batch, channels, time]
        if x.dim() == 3 and x.shape[1] == self.input_dim:
            # Format correct déjà
            pass
        elif x.dim() == 3 and x.shape[2] == self.input_dim:
            # Transpose si nécessaire
            x = x.transpose(1, 2)
        
        # Normalisation d'entrée - utilise BatchNorm1d au lieu de LayerNorm
        # pour être compatible avec le format [batch, channels, time]
        if not hasattr(self, 'input_norm') or isinstance(self.input_norm, nn.LayerNorm):
            self.input_norm = nn.BatchNorm1d(self.input_dim).to(x.device)
        
        x = self.input_norm(x)
        
        # Couches TCN
        for layer in self.tcn_layers:
            x = layer(x)
        
        # Attention temporelle pour les séquences longues
        if x.shape[-1] > 400:  # > 8 secondes
            # Format [batch, time, channels] pour l'attention
            x_att = x.transpose(1, 2)
            x_att = self.temporal_attention(x_att)
            # Retour au format convolutionnel
            x = x_att.transpose(1, 2) + x  # Connexion résiduelle
        
        # Normalisation de sortie
        x = self.output_norm(x)
        
        return x


class EnhancedTemporalBlock(nn.Module):
    """Bloc temporel amélioré avec normalisation et gating."""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        # Première branche
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.GroupNorm(8, n_outputs)  # GroupNorm plus stable que BatchNorm
        self.activation1 = nn.GELU()  # GELU souvent meilleur que ReLU
        self.dropout1 = nn.Dropout(dropout)
        
        # Deuxième branche
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                         padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.GroupNorm(8, n_outputs)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Gating mechanism pour contrôler le flux d'information
        self.gate = nn.Sequential(
            nn.Conv1d(n_outputs, n_outputs, 1),
            nn.Sigmoid()
        )
        
        # Connexion résiduelle
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # Première convolution
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.dropout1(out)
        
        # Deuxième convolution
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.norm2(out)
        out = self.activation2(out)
        out = self.dropout2(out)
        
        # Gating
        gate = self.gate(out)
        out = out * gate
        
        # Connexion résiduelle
        res = x if self.downsample is None else self.downsample(x)
        
        return out + res


class MultiHeadTemporalAttention(nn.Module):
    """Attention multi-têtes optimisée pour les séquences temporelles."""
    
    def __init__(self, embed_dim, num_heads=8, max_seq_len=1500, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projections linéaires
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Encodage positionnel relatif
        self.relative_pos_embedding = nn.Embedding(2 * max_seq_len - 1, self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Projections Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention avec encodage positionnel relatif
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Ajoute l'encodage positionnel relatif
        rel_pos = self._get_relative_positions(seq_len)
        rel_pos_emb = self.relative_pos_embedding(rel_pos.to(x.device))
        rel_scores = torch.einsum('bhid,jid->bhij', q, rel_pos_emb) / self.scale
        scores = scores + rel_scores
        
        # Softmax et dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Application de l'attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Projection finale
        out = self.out_proj(out)
        
        return out
    
    def _get_relative_positions(self, seq_len):
        """Calcule les positions relatives pour l'encodage positionnel."""
        positions = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
        positions = positions + seq_len - 1  # Décalage pour indices positifs
        return positions.clamp(0, 2 * seq_len - 2)


class Chomp1d(nn.Module):
    """Remove padding from the right side of 1D convolution output."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class LongRangeDiarizationModel(nn.Module):
    """Modèle de diarisation optimisé pour les dépendances long terme."""
    
    def __init__(self, input_dim=80, num_speakers=4, max_sequence_length=30.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_speakers = num_speakers
        
        # Backbone TCN long terme
        self.backbone = LongRangeTCN(
            input_dim=input_dim,
            hidden_channels=[128, 256, 256, 512, 512, 512, 512],
            max_sequence_length=max_sequence_length
        )
        
        # Décodeurs avec attention locale
        self.vad_decoder = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, num_speakers, 1)
        )
        
        self.osd_decoder = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 1, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim, time]
        """
        # Backbone
        features = self.backbone(x)
        
        # Décodage
        vad_logits = self.vad_decoder(features)  # [batch, num_speakers, time]
        osd_logits = self.osd_decoder(features)  # [batch, 1, time]
        
        # Transpose pour avoir [batch, time, num_speakers]
        vad_out = vad_logits.transpose(1, 2)
        osd_out = osd_logits.squeeze(1)
        
        return vad_out, osd_out
    
    def get_receptive_field_info(self):
        """Retourne des informations sur le champ réceptif."""
        dilations = self.backbone.dilations
        kernel_size = 3
        
        total_rf = sum((kernel_size - 1) * d for d in dilations)
        rf_seconds = total_rf * 0.02  # 20ms par frame
        
        return {
            'dilations': dilations,
            'receptive_field_frames': total_rf,
            'receptive_field_seconds': rf_seconds
        }


def test_long_range_model():
    """Test du modèle longue portée."""
    
    # Paramètres
    batch_size = 2
    input_dim = 80
    num_speakers = 4
    
    # Test avec différentes longueurs
    test_lengths = [200, 600, 1500]  # 4s, 12s, 30s
    
    model = LongRangeDiarizationModel(
        input_dim=input_dim, 
        num_speakers=num_speakers,
        max_sequence_length=30.0
    )
    
    print(f"🏗️  Modèle créé avec {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Infos sur le champ réceptif
    rf_info = model.get_receptive_field_info()
    print(f"\n📏 Champ réceptif:")
    print(f"   Dilatations: {rf_info['dilations']}")
    print(f"   Frames: {rf_info['receptive_field_frames']}")
    print(f"   Secondes: {rf_info['receptive_field_seconds']:.1f}s")
    
    # Tests
    model.eval()
    with torch.no_grad():
        for length in test_lengths:
            duration = length * 0.02
            x = torch.randn(batch_size, input_dim, length)
            
            vad_out, osd_out = model(x)
            
            print(f"\n✅ Test {duration:.1f}s:")
            print(f"   Input: {x.shape}")
            print(f"   VAD output: {vad_out.shape}")
            print(f"   OSD output: {osd_out.shape}")
    
    print(f"\n🎯 Modèle prêt pour l'entraînement long terme!")


if __name__ == "__main__":
    test_long_range_model()