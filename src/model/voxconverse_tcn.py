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


class VoxConverseTCN(nn.Module):
    """TCN model for VoxConverse diarization: VAD, OSD, VCN prediction."""
    
    def __init__(self, input_dim=80, hidden_channels=[128, 128, 256, 256, 512], 
                 kernel_size=3, num_speakers=4, dropout=0.2):
        super(VoxConverseTCN, self).__init__()
        
        self.input_dim = input_dim
        self.num_speakers = num_speakers
        
        # Mel feature extractor (optional, if working with raw audio)
        self.mel_extractor = MelFeatureExtractor(n_mels=input_dim)
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # TCN backbone
        self.tcn = SimpleTCN(input_dim, hidden_channels, kernel_size, dropout)
        
        # Shared bottleneck layer
        self.bottleneck = nn.Conv1d(hidden_channels[-1], 256, kernel_size=1)
        self.bottleneck_norm = nn.BatchNorm1d(256)
        
        # VAD decoder (Voice Activity Detection) - per speaker
        self.vad_decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_speakers, kernel_size=1)
        )
        
        # OSD decoder (Overlapped Speech Detection) - binary
        self.osd_decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, 1, kernel_size=1)
        )
        
        # VCN decoder (Voice Change Detection) - binary
        self.vcn_decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, 1, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
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
            vcn_out: VCN outputs [batch_size, seq_len]
        """
        if use_mel_extractor:
            # Extract mel features from raw audio
            x = self.mel_extractor(x)  # [batch, n_mels, time]
        
        batch_size, _, seq_len = x.shape
        
        # Input normalization
        x = self.input_norm(x)
        
        # TCN backbone
        x = self.tcn(x)
        
        # Shared bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_norm(x)
        features = F.relu(x)
        
        # VAD prediction - return logits for training
        vad_logits = self.vad_decoder(features)  # [batch, num_speakers, seq_len]
        vad_out = vad_logits.transpose(1, 2)  # [batch, seq_len, num_speakers]
        
        # OSD prediction - return logits for training
        osd_logits = self.osd_decoder(features)  # [batch, 1, seq_len]
        osd_out = osd_logits.squeeze(1)  # [batch, seq_len]
        
        # VCN prediction - return logits for training  
        vcn_logits = self.vcn_decoder(features)  # [batch, 1, seq_len]
        vcn_out = vcn_logits.squeeze(1)  # [batch, seq_len]
        
        return vad_out, osd_out, vcn_out
    
    def predict(self, x, use_mel_extractor=False, threshold=0.5):
        """
        Prediction with thresholding for inference.
        
        Returns:
            vad_pred: [batch_size, seq_len, num_speakers] - sigmoid applied
            osd_pred: [batch_size, seq_len] - sigmoid applied  
            vcn_pred: [batch_size, seq_len] - sigmoid applied
        """
        with torch.no_grad():
            vad_logits, osd_logits, vcn_logits = self.forward(x, use_mel_extractor)
            
            # Apply sigmoid for probability outputs
            vad_pred = torch.sigmoid(vad_logits)
            osd_pred = torch.sigmoid(osd_logits)
            vcn_pred = torch.sigmoid(vcn_logits)
            
            return vad_pred, osd_pred, vcn_pred
    
    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_voxconverse_model(config):
    """Factory function to create VoxConverse TCN model."""
    return VoxConverseTCN(
        input_dim=config.get('input_dim', 80),
        hidden_channels=config.get('hidden_channels', [128, 128, 256, 256, 512]),
        kernel_size=config.get('kernel_size', 3),
        num_speakers=config.get('num_speakers', 4),
        dropout=config.get('dropout', 0.2)
    )


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing VoxConverse TCN Model")
    print("=" * 50)
    
    batch_size = 4
    seq_len = 1000
    input_dim = 80  # Mel features
    num_speakers = 4
    
    # Create model
    model = VoxConverseTCN(input_dim=input_dim, num_speakers=num_speakers)
    print(f"📊 Model parameters: {model.get_num_params():,}")
    
    # Test with mel features
    x_mel = torch.randn(batch_size, input_dim, seq_len)
    vad_out, osd_out, vcn_out = model(x_mel, use_mel_extractor=False)
    
    print(f"\n🎼 Input shape (mel): {x_mel.shape}")
    print(f"🎤 VAD output shape: {vad_out.shape}")  # [4, 1000, 4]
    print(f"🗣️  OSD output shape: {osd_out.shape}")  # [4, 1000]
    print(f"🔄 VCN output shape: {vcn_out.shape}")  # [4, 1000]
    
    # Test prediction mode
    vad_pred, osd_pred, vcn_pred = model.predict(x_mel)
    print(f"\n📈 Prediction mode:")
    print(f"🎤 VAD predictions range: [{vad_pred.min():.3f}, {vad_pred.max():.3f}]")
    print(f"🗣️  OSD predictions range: [{osd_pred.min():.3f}, {osd_pred.max():.3f}]")
    print(f"🔄 VCN predictions range: [{vcn_pred.min():.3f}, {vcn_pred.max():.3f}]")
    
    # Test with raw audio
    audio_samples = 16000 * 10  # 10 seconds at 16kHz
    x_audio = torch.randn(batch_size, audio_samples)
    vad_out_raw, osd_out_raw, vcn_out_raw = model(x_audio, use_mel_extractor=True)
    
    print(f"\n🔊 Input shape (audio): {x_audio.shape}")
    print(f"🎤 VAD output shape (from audio): {vad_out_raw.shape}")
    print(f"🗣️  OSD output shape (from audio): {osd_out_raw.shape}")
    print(f"🔄 VCN output shape (from audio): {vcn_out_raw.shape}")
    
    print(f"\n✅ Model test passed!")
    print(f"🎯 Model: {model.get_num_params():,} parameters")
    print(f"📊 Features: {input_dim}-dim mel spectrogram")
    print(f"🎤 VAD: {num_speakers} speakers detection")
    print(f"🗣️  OSD: Overlap speech detection")
    print(f"🔄 VCN: Voice change detection")