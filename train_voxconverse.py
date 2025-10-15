#!/usr/bin/env python3
"""
Main training script for VoxConverse diarization model.
This script integrates all modules from the reorganized project structure:
- Data analysis from voxconverse_explorer.ipynb
- Model architecture adapted for VAD/OSD/VCN prediction
- Progressive training strategies
- Comprehensive quality analysis and recommendations
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import our modules
from src.model.voxconverse_tcn import VoxConverseTCN, create_voxconverse_model
from src.training.data_analyzer import VoxConverseDataAnalyzer
from src.training.dataset_utils import VoxConverseDatasetUtils
from src.training.trainer import VoxConverseTrainer, create_default_config
from src.voxconverse_dataset import VoxConverseDataset, create_voxconverse_dataloaders


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VoxConverse diarization model')
    
    # Data parameters
    parser.add_argument('--segment_duration', type=float, default=60.0,
                       help='Duration of segments in seconds (default: 60.0)')
    parser.add_argument('--hop_duration', type=float, default=30.0,
                       help='Hop between segments in seconds (default: 30.0)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Audio sample rate (default: 16000)')
    parser.add_argument('--n_mels', type=int, default=80,
                       help='Number of mel bands (default: 80)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=80,
                       help='Input dimension (default: 80)')
    parser.add_argument('--hidden_channels', type=int, nargs='+', 
                       default=[128, 128, 256, 256, 512],
                       help='Hidden channels for TCN (default: [128, 128, 256, 256, 512])')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='Kernel size for TCN (default: 3)')
    parser.add_argument('--num_speakers', type=int, default=4,
                       help='Number of speakers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default: 0.2)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping value (default: 1.0)')
    
    # Analysis parameters
    parser.add_argument('--analyze_data', action='store_true',
                       help='Perform data quality analysis before training')
    parser.add_argument('--max_analysis_segments', type=int, default=1000,
                       help='Max segments for analysis (default: 1000)')
    
    # Training strategy
    parser.add_argument('--use_progressive', action='store_true',
                       help='Use progressive training strategy')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss for class imbalance')
    parser.add_argument('--auto_class_weights', action='store_true',
                       help='Automatically calculate class weights from data')
    
    # I/O parameters
    parser.add_argument('--output_dir', type=str, default='./training_output',
                       help='Output directory for results (default: ./training_output)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory for model checkpoints (default: ./checkpoints)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Logging
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    return parser.parse_args()


def setup_output_directories(output_dir, checkpoint_dir):
    """Setup output directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create subdirectories
    analysis_dir = os.path.join(output_dir, 'analysis')
    plots_dir = os.path.join(output_dir, 'plots')
    logs_dir = os.path.join(output_dir, 'logs')
    
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return {
        'output_dir': output_dir,
        'checkpoint_dir': checkpoint_dir,
        'analysis_dir': analysis_dir,
        'plots_dir': plots_dir,
        'logs_dir': logs_dir
    }


def perform_data_analysis(args, dirs):
    """Perform comprehensive data analysis."""
    print("🔍 PERFORMING DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = VoxConverseDataAnalyzer(
        segment_duration=args.segment_duration,
        hop_duration=args.hop_duration,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels
    )
    
    # Load dataset
    dataset = analyzer.load_dataset(split='dev')
    
    # Quality analysis
    print("📊 Analyzing dataset quality...")
    quality_df = analyzer.analyze_dataset_quality(max_segments=args.max_analysis_segments)
    analyzer.visualize_quality_analysis(
        save_path=os.path.join(dirs['plots_dir'], 'quality_analysis.png')
    )
    
    # Class distribution analysis
    print("⚖️ Analyzing class distribution...")
    distribution_stats = analyzer.analyze_class_distribution(sample_size=args.max_analysis_segments)
    
    # Get training recommendations
    print("🎯 Getting training recommendations...")
    recommendations = analyzer.get_training_recommendations()
    
    # Export analysis results
    analyzer.export_analysis(dirs['analysis_dir'])
    
    return dataset, quality_df, distribution_stats, recommendations


def create_training_config(args, recommendations=None):
    """Create training configuration based on arguments and recommendations."""
    print("⚙️ Creating training configuration...")
    
    # Start with default config
    config = create_default_config()
    
    # Update with arguments
    config['num_epochs'] = args.num_epochs
    config['optimizer']['learning_rate'] = args.learning_rate
    config['optimizer']['weight_decay'] = args.weight_decay
    config['gradient_clip'] = args.gradient_clip
    config['save_every'] = args.save_every
    config['use_progressive_training'] = args.use_progressive
    
    # Update loss configuration based on recommendations
    if recommendations:
        config['loss']['use_focal_loss'] = args.use_focal_loss or recommendations.get('use_focal_loss', False)
        
        if args.auto_class_weights and 'class_weights' in recommendations:
            config['loss']['class_weights'] = recommendations['class_weights']
            print(f"🎯 Using auto-calculated class weights:")
            print(f"   VAD weight: {recommendations['class_weights']['vad']:.2f}")
            print(f"   OSD weight: {recommendations['class_weights']['osd']:.2f}")
            print(f"   VCN weight: {recommendations['class_weights']['vcn']:.2f}")
    
    # Model configuration
    config['model'] = {
        'input_dim': args.input_dim,
        'hidden_channels': args.hidden_channels,
        'kernel_size': args.kernel_size,
        'num_speakers': args.num_speakers,
        'dropout': args.dropout
    }
    
    # Data configuration
    config['data'] = {
        'segment_duration': args.segment_duration,
        'hop_duration': args.hop_duration,
        'sample_rate': args.sample_rate,
        'n_mels': args.n_mels,
        'batch_size': args.batch_size
    }
    
    return config


def create_model_and_dataloaders(config):
    """Create model and data loaders."""
    print("🏗️ Creating model and data loaders...")
    
    # Create model
    model = create_voxconverse_model(config['model'])
    print(f"🧠 Model created with {model.get_num_params():,} parameters")
    
    # Create data loaders
    train_loader, val_loader = create_voxconverse_dataloaders(
        split='dev',
        batch_size=config['data']['batch_size'],
        segment_duration=config['data']['segment_duration'],
        hop_duration=config['data']['hop_duration'],
        sample_rate=config['data']['sample_rate'],
        n_mels=config['data']['n_mels'],
        num_workers=4
    )
    
    print(f"📦 Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader) if val_loader else 'None'}")
    
    return model, train_loader, val_loader


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories
    dirs = setup_output_directories(args.output_dir, args.checkpoint_dir)
    
    print("🚀 VOXCONVERSE DIARIZATION TRAINING")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"💾 Checkpoint directory: {args.checkpoint_dir}")
    print(f"🎯 Target tasks: VAD, OSD, VCN")
    
    # Data analysis (optional)
    recommendations = None
    if args.analyze_data:
        dataset, quality_df, distribution_stats, recommendations = perform_data_analysis(args, dirs)
    
    # Create training configuration
    config = create_training_config(args, recommendations)
    
    # Save configuration
    config_path = os.path.join(dirs['output_dir'], 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"⚙️ Configuration saved: {config_path}")
    
    # Create model and data loaders
    model, train_loader, val_loader = create_model_and_dataloaders(config)
    
    # Create trainer
    trainer = VoxConverseTrainer(model, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)
        print(f"📂 Resumed from epoch {start_epoch}")
    
    # Training
    print(f"\n🏃 STARTING TRAINING")
    print(f"📊 Configuration:")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Batch size: {config['data']['batch_size']}")
    print(f"   Learning rate: {config['optimizer']['learning_rate']}")
    print(f"   Segment duration: {config['data']['segment_duration']}s")
    print(f"   Progressive training: {config['use_progressive_training']}")
    print(f"   Focal loss: {config['loss']['use_focal_loss']}")
    
    # Train the model
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=dirs['checkpoint_dir']
    )
    
    # Save final results
    results = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': trainer.best_val_loss,
        'num_parameters': model.get_num_params(),
        'training_completed_at': datetime.now().isoformat()
    }
    
    results_path = os.path.join(dirs['output_dir'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📈 Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"💾 Best model saved in: {dirs['checkpoint_dir']}")
    print(f"📊 Results saved in: {dirs['output_dir']}")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary
    print(f"\n📋 TRAINING SUMMARY")
    print(f"=" * 40)
    print(f"🧠 Model: VoxConverse TCN")
    print(f"📊 Parameters: {model.get_num_params():,}")
    print(f"🎯 Tasks: VAD, OSD, VCN")
    print(f"📏 Segment duration: {config['data']['segment_duration']}s")
    print(f"🏃 Epochs trained: {config['num_epochs']}")
    print(f"📈 Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"🎯 Final val loss: {val_losses[-1]:.4f}")
        print(f"🏆 Best val loss: {trainer.best_val_loss:.4f}")
    
    return trainer, results


if __name__ == "__main__":
    try:
        trainer, results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)