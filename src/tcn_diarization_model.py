import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import math


class Chomp1d(nn.Module):
    """Remove padding from the right side of 1D convolution output."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Optimized Temporal Block for speaker diarization."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolution
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MultiScaleTCN(nn.Module):
    """Multi-scale Temporal Convolutional Network."""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(MultiScaleTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Adaptive padding to maintain sequence length
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SelfAttention(nn.Module):
    """Self-attention module for temporal dependencies."""
    def __init__(self, embed_dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: [batch, channels, time] -> [batch, time, channels]
        x = x.transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(x + attn_out)
        return out.transpose(1, 2)  # [batch, channels, time]


class DiarizationTCN(nn.Module):
    """Optimized TCN for speaker diarization with VAD, OSD outputs and speaker classification."""
    
    def __init__(self, input_dim=771, hidden_channels=[256, 256, 256, 512, 512], 
                 kernel_size=3, num_speakers=4, dropout=0.2, use_attention=True,
                 use_speaker_classifier=True, embedding_dim=256):
        super(DiarizationTCN, self).__init__()
        
        self.input_dim = input_dim
        self.num_speakers = num_speakers
        self.use_attention = use_attention
        self.use_speaker_classifier = use_speaker_classifier
        self.embedding_dim = embedding_dim
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Multi-scale TCN backbone
        self.tcn = MultiScaleTCN(input_dim, hidden_channels, kernel_size, dropout)
        
        # Optional self-attention
        if use_attention:
            self.attention = SelfAttention(hidden_channels[-1], num_heads=8)
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(hidden_channels[-1], 256, kernel_size=1)
        self.bottleneck_norm = nn.BatchNorm1d(256)
        
        # VAD decoder (Voice Activity Detection per speaker)
        self.vad_decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_speakers, kernel_size=1)
        )
        
        # OSD decoder (Overlapped Speech Detection)
        self.osd_decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, 1, kernel_size=1)
        )
        
        # Speaker embedding extractor
        if use_speaker_classifier:
            self.speaker_embedding = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(dropout),
                nn.Conv1d(256, embedding_dim, kernel_size=1),
                nn.AdaptiveAvgPool1d(1)  # Global average pooling
            )
            
            # Speaker classifier (for supervised training)
            self.speaker_classifier = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_speakers)
            )
            
            # Speaker similarity head (for clustering/assignment)
            self.similarity_head = nn.Sequential(
                nn.Linear(embedding_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_embeddings=False):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, input_dim, seq_len]
            return_embeddings: Whether to return speaker embeddings
            
        Returns:
            vad_out: VAD outputs [batch_size, seq_len, num_speakers]
            osd_out: OSD outputs [batch_size, seq_len]
            embeddings (optional): Speaker embeddings [batch_size, embedding_dim]
            speaker_logits (optional): Speaker classification logits [batch_size, num_speakers]
        """
        batch_size, _, seq_len = x.shape
        
        # Input normalization
        x = self.input_norm(x)
        
        # TCN backbone
        x = self.tcn(x)
        
        # Optional attention
        if self.use_attention:
            x = self.attention(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_norm(x)
        features = F.relu(x)  # Store features for speaker embedding
        
        # VAD prediction
        vad_logits = self.vad_decoder(features)  # [batch, num_speakers, seq_len]
        vad_out = torch.sigmoid(vad_logits).transpose(1, 2)  # [batch, seq_len, num_speakers]
        
        # OSD prediction
        osd_logits = self.osd_decoder(features)  # [batch, 1, seq_len]
        osd_out = torch.sigmoid(osd_logits).squeeze(1)  # [batch, seq_len]
        
        results = [vad_out, osd_out]
        
        # Speaker embedding and classification
        if self.use_speaker_classifier and (return_embeddings or self.training):
            # Extract speaker embeddings
            embeddings = self.speaker_embedding(features)  # [batch, embedding_dim, 1]
            embeddings = embeddings.squeeze(-1)  # [batch, embedding_dim]
            
            # Speaker classification
            speaker_logits = self.speaker_classifier(embeddings)  # [batch, num_speakers]
            
            if return_embeddings:
                results.extend([embeddings, speaker_logits])
            else:
                results.append(speaker_logits)
        
        return tuple(results) if len(results) > 2 else (results[0], results[1])
    
    def get_speaker_similarity(self, embeddings1, embeddings2):
        """
        Compute similarity between speaker embeddings.
        
        Args:
            embeddings1: [batch, embedding_dim]
            embeddings2: [batch, embedding_dim]
            
        Returns:
            similarity: [batch, 1] similarity scores
        """
        if not self.use_speaker_classifier:
            raise ValueError("Speaker classifier must be enabled to compute similarity")
        
        combined = torch.cat([embeddings1, embeddings2], dim=1)
        similarity = self.similarity_head(combined)
        return similarity
    
    def extract_speaker_embeddings(self, x, segments=None):
        """
        Extract speaker embeddings for given segments.
        
        Args:
            x: Input features [batch, input_dim, seq_len]
            segments: Optional segment masks [batch, seq_len, num_speakers]
            
        Returns:
            embeddings: Speaker embeddings [batch, num_speakers, embedding_dim]
        """
        if not self.use_speaker_classifier:
            raise ValueError("Speaker classifier must be enabled to extract embeddings")
        
        # Get features
        batch_size, _, seq_len = x.shape
        x = self.input_norm(x)
        x = self.tcn(x)
        if self.use_attention:
            x = self.attention(x)
        x = self.bottleneck(x)
        x = self.bottleneck_norm(x)
        features = F.relu(x)  # [batch, 256, seq_len]
        
        if segments is not None:
            # Extract embeddings for each speaker segment
            segments = segments.transpose(1, 2)  # [batch, num_speakers, seq_len]
            speaker_embeddings = []
            
            for spk in range(self.num_speakers):
                spk_mask = segments[:, spk:spk+1, :]  # [batch, 1, seq_len]
                spk_features = features * spk_mask  # Apply mask
                
                # Global average pooling for speaker
                spk_embedding = spk_features.sum(dim=2) / (spk_mask.sum(dim=2) + 1e-8)
                speaker_embeddings.append(spk_embedding)
            
            embeddings = torch.stack(speaker_embeddings, dim=1)  # [batch, num_speakers, 256]
        else:
            # Extract global embedding
            embeddings = self.speaker_embedding(features).squeeze(-1)  # [batch, embedding_dim]
            embeddings = embeddings.unsqueeze(1).repeat(1, self.num_speakers, 1)
        
        return embeddings
    
    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnsembleDiarizationTCN(nn.Module):
    """Ensemble of TCN models for improved performance."""
    
    def __init__(self, input_dim=771, num_models=3, **model_kwargs):
        super(EnsembleDiarizationTCN, self).__init__()
        
        self.models = nn.ModuleList([
            DiarizationTCN(input_dim=input_dim, **model_kwargs) 
            for _ in range(num_models)
        ])
        
    def forward(self, x):
        vad_outputs = []
        osd_outputs = []
        
        for model in self.models:
            vad_out, osd_out = model(x)
            vad_outputs.append(vad_out)
            osd_outputs.append(osd_out)
        
        # Average ensemble predictions
        vad_ensemble = torch.stack(vad_outputs).mean(dim=0)
        osd_ensemble = torch.stack(osd_outputs).mean(dim=0)
        
        return vad_ensemble, osd_ensemble


if __name__ == "__main__":
    # Test the model
    batch_size = 8
    seq_len = 1000
    input_dim = 771  # 257 * 3 (LPS + IPD + AF)
    num_speakers = 4
    
    # Create model
    model = DiarizationTCN(input_dim=input_dim, num_speakers=num_speakers)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim, seq_len)
    vad_out, osd_out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"VAD output shape: {vad_out.shape}")  # [8, 1000, 4]
    print(f"OSD output shape: {osd_out.shape}")  # [8, 1000]
    
    # Test ensemble model
    ensemble_model = EnsembleDiarizationTCN(input_dim=input_dim, num_models=2)
    vad_ens, osd_ens = ensemble_model(x)
    
    print(f"Ensemble VAD shape: {vad_ens.shape}")
    print(f"Ensemble OSD shape: {osd_ens.shape}")