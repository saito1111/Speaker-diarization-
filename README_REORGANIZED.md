# VoxConverse Speaker Diarization with TCN

This project implements a Temporal Convolutional Network (TCN) for speaker diarization on the VoxConverse dataset, with multi-task learning for Voice Activity Detection (VAD), Overlap Speech Detection (OSD), and Voice Change Detection (VCN).

## 🏗️ Project Structure

```
Speaker-diarization-/
├── src/
│   ├── model/                    # Model architecture
│   │   ├── __init__.py
│   │   └── voxconverse_tcn.py   # TCN model for VAD/OSD/VCN
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── data_analyzer.py     # Data quality analysis
│   │   ├── dataset_utils.py     # Dataset utilities
│   │   └── trainer.py           # Training logic
│   ├── voxconverse_dataset.py   # Dataset implementation
│   ├── curriculum_trainer.py    # Legacy curriculum trainer
│   ├── progressive_training.py  # Legacy progressive training
│   └── simple_tcn_model.py      # Legacy simple model
├── train_voxconverse.py         # Main training script
├── demo_project.py              # Demo and testing script
├── voxconverse_explorer.ipynb   # Original exploration notebook
├── requirements.txt
└── README.md
```

## 🎯 Key Features

### Multi-Task Learning
- **VAD (Voice Activity Detection)**: Detects speech vs silence per speaker
- **OSD (Overlap Speech Detection)**: Detects when multiple speakers talk simultaneously  
- **VCN (Voice Change Detection)**: Detects speaker transitions

### Data Analysis & Quality Assessment
- Comprehensive dataset quality metrics
- Class imbalance analysis with state-of-the-art recommendations
- Conversation boundary analysis
- Automated training strategy recommendations

### Advanced Training Strategies
- **Adaptive Loss Functions**: Focal Loss and weighted BCE for class imbalance
- **Progressive Training**: Curriculum learning with increasing complexity
- **Class Balancing**: Automatic weight calculation from data analysis
- **Checkpoint Management**: Best model tracking and resume capabilities

### Model Architecture
- **TCN Backbone**: Temporal Convolutional Network with dilated convolutions
- **Multi-Head Prediction**: Separate decoders for VAD, OSD, VCN
- **Efficient Design**: ~2-5M parameters depending on configuration
- **Flexible Input**: Supports both raw audio and mel spectrograms

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure VoxConverse dataset is available
# (Follow VoxConverse dataset setup instructions)
```

### 2. Run Demo

```bash
# Test all components
python demo_project.py
```

### 3. Analyze Data

```bash
# Perform comprehensive data analysis
python -c "from src.training.data_analyzer import main; main()"
```

### 4. Train Model

```bash
# Basic training
python train_voxconverse.py

# Training with data analysis and auto-configuration
python train_voxconverse.py --analyze_data --auto_class_weights --use_focal_loss

# Progressive training
python train_voxconverse.py --use_progressive --num_epochs 150

# Custom configuration
python train_voxconverse.py \
    --segment_duration 60.0 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --num_epochs 100 \
    --output_dir ./my_training \
    --analyze_data
```

## 📊 Data Analysis Features

### Quality Metrics
- VAD ratio distribution (speech vs silence)
- OSD ratio distribution (overlap vs single speaker)
- VCN ratio distribution (change points vs stable)
- Conversation boundary analysis
- Segment truncation detection

### Class Imbalance Analysis
- Frame-level class distribution
- Imbalance ratio calculation (negative:positive)
- State-of-the-art balancing recommendations
- Automatic class weight calculation

### Training Recommendations
Based on your data characteristics, the system automatically recommends:
- **Class Weights**: For moderate imbalance (5:1 to 20:1 ratio)
- **Focal Loss**: For severe imbalance (>20:1 ratio)
- **Undersampling**: For extreme cases
- **Loss Function Configuration**: Optimal α and γ parameters

## 🧠 Model Architecture

### VoxConverseTCN
```python
VoxConverseTCN(
    input_dim=80,           # Mel spectrogram dimensions
    hidden_channels=[128, 128, 256, 256, 512],  # TCN layer sizes
    kernel_size=3,          # Convolution kernel size
    num_speakers=4,         # Maximum speakers for VAD
    dropout=0.2             # Dropout rate
)
```

### Outputs
- **VAD**: `[batch, seq_len, num_speakers]` - Voice activity per speaker
- **OSD**: `[batch, seq_len]` - Overlap speech detection
- **VCN**: `[batch, seq_len]` - Voice change detection

### Loss Function
```python
AdaptiveLossFunction(
    vad_weight=1.0,         # VAD loss weight
    osd_weight=2.0,         # OSD loss weight (higher for rarer class)
    vcn_weight=3.0,         # VCN loss weight (highest for rarest)
    use_focal_loss=True,    # Enable Focal Loss
    class_weights=None      # Auto-calculated from data
)
```

## 📈 Training Configuration

### Default Configuration
```python
{
    "num_epochs": 100,
    "optimizer": {
        "type": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 0.01
    },
    "scheduler": {
        "type": "onecycle",
        "max_lr": 1e-2,
        "pct_start": 0.3
    },
    "loss": {
        "vad_weight": 1.0,
        "osd_weight": 2.0,      # More weight for rarer OSD
        "vcn_weight": 3.0,      # Most weight for rarest VCN
        "use_focal_loss": True,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0
    }
}
```

### Progressive Training
```python
curriculum_schedule = [
    (0, 10.0, 1.0),    # Start: 10s segments, low complexity
    (20, 30.0, 1.5),   # Epoch 20: 30s segments, medium complexity  
    (50, 60.0, 2.0),   # Epoch 50: 60s segments, full complexity
]
```

## 🔧 Command Line Options

### Data Parameters
- `--segment_duration`: Segment length in seconds (default: 60.0)
- `--hop_duration`: Hop between segments (default: 30.0)
- `--sample_rate`: Audio sample rate (default: 16000)
- `--batch_size`: Training batch size (default: 8)

### Model Parameters
- `--hidden_channels`: TCN layer sizes (default: [128,128,256,256,512])
- `--num_speakers`: Maximum speakers (default: 4)
- `--dropout`: Dropout rate (default: 0.2)

### Training Parameters
- `--num_epochs`: Training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--use_progressive`: Enable progressive training
- `--use_focal_loss`: Enable Focal Loss for imbalance

### Analysis Parameters
- `--analyze_data`: Perform data analysis before training
- `--auto_class_weights`: Calculate class weights from data
- `--max_analysis_segments`: Segments to analyze (default: 1000)

## 📋 Results and Monitoring

### Output Structure
```
training_output/
├── analysis/               # Data analysis results
│   ├── quality_analysis_*.csv
│   └── distribution_stats_*.json
├── plots/                  # Visualization plots
│   ├── quality_analysis.png
│   └── training_curves.png
├── logs/                   # Training logs
├── training_config.json   # Used configuration
└── training_results.json  # Final results

checkpoints/
├── best_model.pth         # Best validation model
├── final_model.pth        # Final epoch model
└── checkpoint_epoch_*.pth # Regular checkpoints
```

### Metrics Tracked
- Training/validation loss (total and per-task)
- Learning rate schedule
- Class-specific loss components
- Model checkpoint metadata

## 🔬 Advanced Usage

### Custom Dataset Filtering
```python
from src.training.dataset_utils import VoxConverseDatasetUtils

utils = VoxConverseDatasetUtils(segment_duration=60.0)
custom_dataset = utils.create_custom_dataset(
    min_vad_ratio=0.3,      # Require 30% speech
    max_osd_ratio=0.1,      # Max 10% overlap
    require_voice_changes=True,  # Only segments with speaker changes
    max_segments=5000       # Limit dataset size
)
```

### Manual Class Weight Configuration
```python
# Based on your data analysis
class_weights = {
    'vad': 2.5,   # Speech is moderately rare
    'osd': 15.2,  # Overlap is very rare
    'vcn': 25.8   # Voice changes are extremely rare
}
```

### Model Inference
```python
model = VoxConverseTCN(...)
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])

# Get predictions with probabilities
vad_pred, osd_pred, vcn_pred = model.predict(mel_features)

# Predictions are in [0,1] range after sigmoid
```

## 📚 Implementation Details

### From Notebook to Production
This implementation reorganizes and productionizes the analysis from `voxconverse_explorer.ipynb`:

1. **Data Analysis** → `src/training/data_analyzer.py`
2. **Dataset Utilities** → `src/training/dataset_utils.py`  
3. **Model Architecture** → `src/model/voxconverse_tcn.py`
4. **Training Logic** → `src/training/trainer.py`
5. **Integration** → `train_voxconverse.py`

### Key Improvements
- **Modular Design**: Clean separation of concerns
- **Configuration Management**: JSON-based config system
- **Automatic Recommendations**: Data-driven training strategies
- **Production Ready**: Error handling, logging, checkpointing
- **Extensible**: Easy to add new tasks or model architectures

### State-of-the-Art Techniques
- **Multi-task Learning**: Joint training for related tasks
- **Class Imbalance Handling**: Focal Loss + weighted training
- **Progressive Training**: Curriculum learning from simple to complex
- **Adaptive Optimization**: OneCycleLR for faster convergence

## 🤝 Contributing

1. Follow the modular structure in `src/`
2. Add tests in the appropriate modules
3. Update configuration schemas when adding parameters
4. Document new features in this README

## 📄 License

This project builds upon VoxConverse dataset and PyTorch ecosystem.
Follow respective licensing terms for commercial use.

---

🎯 **Ready to train state-of-the-art speaker diarization models with comprehensive data analysis and adaptive training strategies!**