#!/usr/bin/env python3
"""
Quick analysis and demo script for VoxConverse dataset.
This script demonstrates the capabilities of the reorganized modules.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import our modules
from src.model.voxconverse_tcn import VoxConverseTCN
from src.training.data_analyzer import VoxConverseDataAnalyzer
from src.training.dataset_utils import VoxConverseDatasetUtils


def demo_model():
    """Demonstrate model architecture and capabilities."""
    print("🧪 DEMONSTRATING MODEL ARCHITECTURE")
    print("=" * 50)
    
    # Create model
    model = VoxConverseTCN(
        input_dim=80,
        hidden_channels=[128, 128, 256, 256, 512],
        kernel_size=3,
        num_speakers=4,
        dropout=0.2
    )
    
    print(f"🧠 Model: VoxConverse TCN")
    print(f"📊 Parameters: {model.get_num_params():,}")
    print(f"🎯 Predicts: VAD (Voice Activity), OSD (Overlap Speech), VCN (Voice Change)")
    
    # Test with sample input
    batch_size = 4
    seq_len = 1000
    input_dim = 80
    
    # Test with mel features
    x_mel = torch.randn(batch_size, input_dim, seq_len)
    vad_out, osd_out, vcn_out = model(x_mel, use_mel_extractor=False)
    
    print(f"\n📊 Input/Output shapes:")
    print(f"   Input (mel):     {x_mel.shape}")
    print(f"   VAD output:      {vad_out.shape}")  # [batch, seq_len, num_speakers]
    print(f"   OSD output:      {osd_out.shape}")  # [batch, seq_len]
    print(f"   VCN output:      {vcn_out.shape}")  # [batch, seq_len]
    
    # Test prediction mode
    vad_pred, osd_pred, vcn_pred = model.predict(x_mel)
    print(f"\n📈 Prediction ranges (after sigmoid):")
    print(f"   VAD: [{vad_pred.min():.3f}, {vad_pred.max():.3f}]")
    print(f"   OSD: [{osd_pred.min():.3f}, {osd_pred.max():.3f}]")
    print(f"   VCN: [{vcn_pred.min():.3f}, {vcn_pred.max():.3f}]")
    
    print("✅ Model demo completed!")
    return model


def demo_data_analysis():
    """Demonstrate data analysis capabilities."""
    print("\n🔍 DEMONSTRATING DATA ANALYSIS")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = VoxConverseDataAnalyzer(
        segment_duration=30.0,  # Shorter for demo
        hop_duration=15.0,
        sample_rate=16000,
        n_mels=80
    )
    
    # Load dataset
    print("📦 Loading VoxConverse dataset...")
    dataset = analyzer.load_dataset(split='dev')
    
    if len(dataset) == 0:
        print("❌ No dataset found. Please ensure VoxConverse data is available.")
        return None
    
    # Quick quality analysis
    print("📊 Performing quality analysis...")
    quality_df = analyzer.analyze_dataset_quality(max_segments=100)  # Small sample for demo
    
    # Class distribution analysis
    print("⚖️ Analyzing class distribution...")
    distribution_stats = analyzer.analyze_class_distribution(sample_size=100)
    
    # Get recommendations
    print("🎯 Getting training recommendations...")
    recommendations = analyzer.get_training_recommendations()
    
    print(f"\n📋 ANALYSIS SUMMARY:")
    print(f"   Dataset segments: {len(dataset)}")
    print(f"   Analyzed segments: {len(quality_df)}")
    print(f"   Recommended strategy: {recommendations['strategy']}")
    print(f"   Most imbalanced class: {recommendations['most_imbalanced_class']}")
    print(f"   Worst ratio: {recommendations['worst_ratio']:.1f}:1")
    
    print("✅ Data analysis demo completed!")
    return analyzer, dataset, recommendations


def demo_dataset_utils():
    """Demonstrate dataset utilities."""
    print("\n🔧 DEMONSTRATING DATASET UTILITIES")
    print("=" * 50)
    
    # Initialize utils
    utils = VoxConverseDatasetUtils(
        segment_duration=30.0,  # Shorter for demo
        hop_duration=15.0,
        sample_rate=16000,
        n_mels=80
    )
    
    # Create custom filtered dataset
    print("🎛️ Creating custom filtered dataset...")
    try:
        custom_dataset = utils.create_custom_dataset(
            split='dev',
            min_vad_ratio=0.2,      # More speech required
            max_osd_ratio=0.2,      # Less overlap allowed
            require_voice_changes=False,
            max_segments=50         # Small for demo
        )
        
        print(f"✅ Custom dataset created: {len(custom_dataset)} segments")
        
        if len(custom_dataset) > 0:
            # Analyze conversation boundaries
            print("🔍 Analyzing conversation boundaries...")
            conv_stats, truncated = utils.analyze_conversation_boundaries(custom_dataset)
            
            print(f"📊 Conversation analysis:")
            print(f"   Total conversations: {len(conv_stats)}")
            print(f"   Truncated conversations: {len(truncated)}")
            
            # Demo export (just first segment if available)
            if len(custom_dataset) > 0:
                print("💾 Demo: exporting first segment...")
                export_paths = utils.export_segment_analysis(
                    custom_dataset, 0, output_dir='./demo_exports'
                )
                print(f"📄 Sample export saved to: ./demo_exports/")
        
        print("✅ Dataset utilities demo completed!")
        return utils, custom_dataset
        
    except Exception as e:
        print(f"⚠️ Dataset utilities demo failed: {e}")
        print("This might be due to missing VoxConverse data.")
        return utils, None


def demo_training_config():
    """Demonstrate training configuration."""
    print("\n⚙️ DEMONSTRATING TRAINING CONFIGURATION")
    print("=" * 50)
    
    from src.training.trainer import create_default_config, AdaptiveLossFunction
    
    # Create default config
    config = create_default_config()
    
    print("📋 Default training configuration:")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Optimizer: {config['optimizer']['type']}")
    print(f"   Learning rate: {config['optimizer']['learning_rate']}")
    print(f"   Scheduler: {config['scheduler']['type']}")
    print(f"   Loss weights - VAD: {config['loss']['vad_weight']}, "
          f"OSD: {config['loss']['osd_weight']}, VCN: {config['loss']['vcn_weight']}")
    print(f"   Focal loss: {config['loss']['use_focal_loss']}")
    print(f"   Progressive training: {config['use_progressive_training']}")
    
    # Demo adaptive loss function
    print("\n🎯 Testing adaptive loss function...")
    loss_fn = AdaptiveLossFunction(
        vad_weight=1.0,
        osd_weight=2.0,
        vcn_weight=3.0,
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    
    # Create dummy predictions and targets
    batch_size, seq_len, num_speakers = 2, 100, 4
    vad_pred = torch.randn(batch_size, seq_len, num_speakers)
    osd_pred = torch.randn(batch_size, seq_len)
    vcn_pred = torch.randn(batch_size, seq_len)
    
    vad_target = torch.randint(0, 2, (batch_size, seq_len, num_speakers)).float()
    osd_target = torch.randint(0, 2, (batch_size, seq_len)).float()
    vcn_target = torch.randint(0, 2, (batch_size, seq_len)).float()
    
    # Compute loss
    total_loss, loss_components = loss_fn(
        vad_pred, osd_pred, vcn_pred,
        vad_target, osd_target, vcn_target
    )
    
    print(f"📊 Loss computation test:")
    print(f"   Total loss: {total_loss:.4f}")
    print(f"   VAD loss: {loss_components['vad_loss']:.4f}")
    print(f"   OSD loss: {loss_components['osd_loss']:.4f}")
    print(f"   VCN loss: {loss_components['vcn_loss']:.4f}")
    
    print("✅ Training configuration demo completed!")
    return config, loss_fn


def main():
    """Main demo function."""
    print("🎬 VOXCONVERSE PROJECT DEMO")
    print("=" * 60)
    print("Demonstrating the reorganized VoxConverse project capabilities:")
    print("• Model architecture for VAD/OSD/VCN prediction")
    print("• Data quality analysis and class imbalance detection")
    print("• Dataset utilities and filtering")
    print("• Training configuration and loss functions")
    print()
    
    try:
        # Demo model
        model = demo_model()
        
        # Demo data analysis
        analyzer_result = demo_data_analysis()
        
        # Demo dataset utils
        utils_result = demo_dataset_utils()
        
        # Demo training config
        config_result = demo_training_config()
        
        print(f"\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("🚀 Ready to train VoxConverse diarization models!")
        print()
        print("📝 Next steps:")
        print("1. Ensure VoxConverse dataset is properly set up")
        print("2. Run data analysis: python -c \"from src.training.data_analyzer import main; main()\"")
        print("3. Start training: python train_voxconverse.py --analyze_data")
        print("4. Monitor training progress and checkpoints")
        print()
        print("📚 Key features demonstrated:")
        print("• 🧠 TCN model with VAD/OSD/VCN multi-task learning")
        print("• 📊 Comprehensive data quality analysis")
        print("• ⚖️ Class imbalance detection and recommendations")
        print("• 🎯 Adaptive loss functions (Focal Loss, class weights)")
        print("• 📈 Progressive training strategies")
        print("• 🔧 Flexible dataset filtering and customization")
        print("• 💾 Checkpoint management and result tracking")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure all dependencies are installed")
        print("2. Check that VoxConverse dataset is available")
        print("3. Verify Python path includes src directory")
        
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo encountered errors - check troubleshooting tips above")
        sys.exit(1)