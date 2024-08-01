import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ConvTasNet(nn.Module):
    def __init__(self, input_dim, bottleneck_size=256, num_channels=512, kernel_size=3, num_speakers=4):
        super(ConvTasNet, self).__init__()
        self.tcn = TemporalConvNet(input_dim, [num_channels]*8, kernel_size)
        self.bottleneck = nn.Conv1d(num_channels, bottleneck_size, 1)
        self.decoders_vad = nn.ConvTranspose1d(bottleneck_size, num_speakers, kernel_size=2, stride=2)
        self.decoders_osd = nn.ConvTranspose1d(bottleneck_size, 1, kernel_size=2, stride=2)
    
    def forward(self, lps, ipd, af):
        # Concatenate inputs along the channel dimension
        x = torch.cat((lps, ipd, af), dim=1)
        
        # TCN blocks
        x = self.tcn(x)
        
        # Bottleneck layer
        x = self.bottleneck(x)
        
        # Decoder layers
        y_vad = self.decoders_vad(x)  # Shape: [batch_size, num_speakers, seq_len*2]
        y_osd = self.decoders_osd(x)  # Shape: [batch_size, 1, seq_len*2]
        
        # Reshape outputs
        y_vad = y_vad.permute(0, 2, 1)  # Shape: [batch_size, seq_len*2, num_speakers]
        y_osd = y_osd.squeeze(1)  # Shape: [batch_size, seq_len*2]
        
        return y_vad, y_osd
