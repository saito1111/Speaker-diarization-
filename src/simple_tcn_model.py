import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import torchaudio


class Chomp1d(nn.Module):
    """Remove padding from the right side of 1D convolution output."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Simplified Temporal Block for speaker diarization."""
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


class SimpleTCN(nn.Module):
    """Simple Temporal Convolutional Network."""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(SimpleTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding to maintain sequence length
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MelFeatureExtractor(nn.Module):
    """Extract mel-spectrogram features from raw audio."""
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=256, n_mels=80):
        super(MelFeatureExtractor, self).__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=8000
        )
        
    def forward(self, waveform):
        """
        Args:
            waveform: [batch, 1, samples] or [batch, samples]
        Returns:
            mel_features: [batch, n_mels, time]
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-8)
        return log_mel


class SimpleDiarizationTCN(nn.Module):
    """Simple TCN for speaker diarization without attention or embeddings."""
    
    def __init__(self, input_dim=80, hidden_channels=[128, 128, 256, 256, 512], 
                 kernel_size=3, num_speakers=4, dropout=0.2):
        super(SimpleDiarizationTCN, self).__init__()
        
        self.input_dim = input_dim
        self.num_speakers = num_speakers
        
        # Mel feature extractor (optional, if working with raw audio)
        self.mel_extractor = MelFeatureExtractor(n_mels=input_dim)
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Simplified TCN backbone
        self.tcn = SimpleTCN(input_dim, hidden_channels, kernel_size, dropout)
        
        # Bottleneck layer
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
    
    def forward(self, x, use_mel_extractor=False):
        """
        Forward pass
        
        Args:
            x: Input tensor
               - If use_mel_extractor=True: [batch_size, samples] raw audio
               - If use_mel_extractor=False: [batch_size, input_dim, seq_len] mel features
            use_mel_extractor: Whether to extract mel features from raw audio
            
        Returns:
            vad_out: VAD outputs [batch_size, seq_len, num_speakers]
            osd_out: OSD outputs [batch_size, seq_len]
        """
        if use_mel_extractor:
            # Extract mel features from raw audio
            x = self.mel_extractor(x)  # [batch, n_mels, time]
        
        batch_size, _, seq_len = x.shape
        
        # Input normalization
        x = self.input_norm(x)
        
        # TCN backbone
        x = self.tcn(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_norm(x)
        features = F.relu(x)
        
        # VAD prediction - return logits for AMP compatibility
        vad_logits = self.vad_decoder(features)  # [batch, num_speakers, seq_len]
        vad_out = vad_logits.transpose(1, 2)  # [batch, seq_len, num_speakers]
        
        # OSD prediction - return logits for AMP compatibility
        osd_logits = self.osd_decoder(features)  # [batch, 1, seq_len]
        osd_out = osd_logits.squeeze(1)  # [batch, seq_len]
        
        return vad_out, osd_out
    
    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_simple_model(config):
    """Factory function to create a simple model."""
    return SimpleDiarizationTCN(
        input_dim=config.get('input_dim', 80),
        hidden_channels=config.get('hidden_channels', [128, 128, 256, 256, 512]),
        kernel_size=config.get('kernel_size', 3),
        num_speakers=config.get('num_speakers', 4),
        dropout=config.get('dropout', 0.2)
    )


if __name__ == "__main__":
    # Test the simplified model
    batch_size = 4
    seq_len = 1000
    input_dim = 80  # Mel features
    num_speakers = 4
    
    # Create simplified model
    model = SimpleDiarizationTCN(input_dim=input_dim, num_speakers=num_speakers)
    print(f"Simple model parameters: {model.get_num_params():,}")
    
    # Test with mel features
    x_mel = torch.randn(batch_size, input_dim, seq_len)
    vad_out, osd_out = model(x_mel, use_mel_extractor=False)
    
    print(f"Input shape (mel): {x_mel.shape}")
    print(f"VAD output shape: {vad_out.shape}")  # [4, 1000, 4]
    print(f"OSD output shape: {osd_out.shape}")  # [4, 1000]
    
    # Test with raw audio
    audio_samples = 16000 * 10  # 10 seconds at 16kHz
    x_audio = torch.randn(batch_size, audio_samples)
    vad_out_raw, osd_out_raw = model(x_audio, use_mel_extractor=True)
    
    print(f"Input shape (audio): {x_audio.shape}")
    print(f"VAD output shape (from audio): {vad_out_raw.shape}")
    print(f"OSD output shape (from audio): {osd_out_raw.shape}")
    
    print(f"\n✅ Simple model test passed!")
    print(f"🎯 Model is {model.get_num_params():,} parameters")
    print("📊 Features: 80-dim mel spectrogram")
    print("🚫 No self-attention, no embeddings - simple and fast!")