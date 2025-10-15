#!/usr/bin/env python3
"""
Test script to validate the reorganized VoxConverse project.
Run this script to ensure all modules work correctly together.
"""

import os
import sys
import tempfile
import shutil
import torch
import traceback
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Model imports
        from src.model.voxconverse_tcn import VoxConverseTCN, create_voxconverse_model
        print("✅ Model imports successful")
        
        # Training imports
        from src.training.data_analyzer import VoxConverseDataAnalyzer
        from src.training.dataset_utils import VoxConverseDatasetUtils
        from src.training.trainer import VoxConverseTrainer, create_default_config, AdaptiveLossFunction
        print("✅ Training imports successful")
        
        # Dataset imports
        from src.voxconverse_dataset import VoxConverseDataset
        print("✅ Dataset imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation and forward pass."""
    print("🧪 Testing model creation...")
    
    try:
        from src.model.voxconverse_tcn import VoxConverseTCN
        
        # Create model
        model = VoxConverseTCN(
            input_dim=80,
            hidden_channels=[64, 128, 256],  # Smaller for testing
            kernel_size=3,
            num_speakers=4,
            dropout=0.2
        )
        
        print(f"✅ Model created with {model.get_num_params():,} parameters")
        
        # Test forward pass
        batch_size, seq_len, input_dim = 2, 100, 80
        x = torch.randn(batch_size, input_dim, seq_len)
        
        # Training mode (returns logits)
        vad_out, osd_out, vcn_out = model(x)
        assert vad_out.shape == (batch_size, seq_len, 4), f"VAD shape mismatch: {vad_out.shape}"
        assert osd_out.shape == (batch_size, seq_len), f"OSD shape mismatch: {osd_out.shape}"
        assert vcn_out.shape == (batch_size, seq_len), f"VCN shape mismatch: {vcn_out.shape}"
        
        # Prediction mode (returns probabilities)
        vad_pred, osd_pred, vcn_pred = model.predict(x)
        assert torch.all(vad_pred >= 0) and torch.all(vad_pred <= 1), "VAD predictions not in [0,1]"
        assert torch.all(osd_pred >= 0) and torch.all(osd_pred <= 1), "OSD predictions not in [0,1]"
        assert torch.all(vcn_pred >= 0) and torch.all(vcn_pred <= 1), "VCN predictions not in [0,1]"
        
        print("✅ Model forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        traceback.print_exc()
        return False


def test_loss_function():
    """Test adaptive loss function."""
    print("🧪 Testing loss function...")
    
    try:
        from src.training.trainer import AdaptiveLossFunction
        
        # Create loss function
        loss_fn = AdaptiveLossFunction(
            vad_weight=1.0,
            osd_weight=2.0,
            vcn_weight=3.0,
            use_focal_loss=True,
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        # Create dummy data
        batch_size, seq_len, num_speakers = 2, 50, 4
        
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
        
        assert isinstance(total_loss, torch.Tensor), "Total loss should be tensor"
        assert total_loss.requires_grad, "Loss should require gradients"
        assert 'vad_loss' in loss_components, "Missing VAD loss component"
        assert 'osd_loss' in loss_components, "Missing OSD loss component"
        assert 'vcn_loss' in loss_components, "Missing VCN loss component"
        
        print(f"✅ Loss function test successful: {total_loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        traceback.print_exc()
        return False


def test_config_creation():
    """Test configuration creation."""
    print("🧪 Testing configuration...")
    
    try:
        from src.training.trainer import create_default_config
        
        config = create_default_config()
        
        # Check required keys
        required_keys = ['num_epochs', 'optimizer', 'loss', 'scheduler']
        for key in required_keys:
            assert key in config, f"Missing config key: {key}"
        
        # Check optimizer config
        assert 'type' in config['optimizer'], "Missing optimizer type"
        assert 'learning_rate' in config['optimizer'], "Missing learning rate"
        
        # Check loss config
        assert 'vad_weight' in config['loss'], "Missing VAD weight"
        assert 'osd_weight' in config['loss'], "Missing OSD weight"
        assert 'vcn_weight' in config['loss'], "Missing VCN weight"
        
        print(f"✅ Configuration test successful")
        print(f"   Epochs: {config['num_epochs']}")
        print(f"   Optimizer: {config['optimizer']['type']}")
        print(f"   LR: {config['optimizer']['learning_rate']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_trainer_creation():
    """Test trainer creation and basic functionality."""
    print("🧪 Testing trainer creation...")
    
    try:
        from src.model.voxconverse_tcn import VoxConverseTCN
        from src.training.trainer import VoxConverseTrainer, create_default_config
        
        # Create model and config
        model = VoxConverseTCN(
            input_dim=80,
            hidden_channels=[64, 128],  # Very small for testing
            kernel_size=3,
            num_speakers=2,
            dropout=0.1
        )
        
        config = create_default_config()
        config['num_epochs'] = 2  # Short for testing
        
        # Create trainer
        trainer = VoxConverseTrainer(model, config)
        
        assert hasattr(trainer, 'model'), "Trainer missing model"
        assert hasattr(trainer, 'optimizer'), "Trainer missing optimizer"
        assert hasattr(trainer, 'criterion'), "Trainer missing criterion"
        
        print(f"✅ Trainer creation successful")
        print(f"   Device: {trainer.device}")
        print(f"   Model params: {trainer.model.get_num_params():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        traceback.print_exc()
        return False


def test_data_analyzer():
    """Test data analyzer (without actual data)."""
    print("🧪 Testing data analyzer...")
    
    try:
        from src.training.data_analyzer import VoxConverseDataAnalyzer
        
        # Create analyzer
        analyzer = VoxConverseDataAnalyzer(
            segment_duration=30.0,
            hop_duration=15.0,
            sample_rate=16000,
            n_mels=80
        )
        
        assert analyzer.segment_duration == 30.0, "Wrong segment duration"
        assert analyzer.sample_rate == 16000, "Wrong sample rate"
        
        print("✅ Data analyzer creation successful")
        print(f"   Segment duration: {analyzer.segment_duration}s")
        print(f"   Sample rate: {analyzer.sample_rate} Hz")
        
        # Note: We can't test dataset loading without actual VoxConverse data
        print("⚠️  Dataset loading test skipped (requires VoxConverse data)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data analyzer test failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_utils():
    """Test dataset utilities."""
    print("🧪 Testing dataset utilities...")
    
    try:
        from src.training.dataset_utils import VoxConverseDatasetUtils
        
        # Create utils
        utils = VoxConverseDatasetUtils(
            segment_duration=30.0,
            hop_duration=15.0,
            sample_rate=16000,
            n_mels=80
        )
        
        assert utils.segment_duration == 30.0, "Wrong segment duration"
        assert utils.sample_rate == 16000, "Wrong sample rate"
        
        print("✅ Dataset utils creation successful")
        print(f"   Segment duration: {utils.segment_duration}s")
        print(f"   Sample rate: {utils.sample_rate} Hz")
        
        # Note: We can't test actual dataset operations without VoxConverse data
        print("⚠️  Dataset operations test skipped (requires VoxConverse data)")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset utils test failed: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components."""
    print("🧪 Testing component integration...")
    
    try:
        from src.model.voxconverse_tcn import create_voxconverse_model
        from src.training.trainer import VoxConverseTrainer, create_default_config
        
        # Create model via factory function
        model_config = {
            'input_dim': 80,
            'hidden_channels': [64, 128],
            'kernel_size': 3,
            'num_speakers': 2,
            'dropout': 0.1
        }
        
        model = create_voxconverse_model(model_config)
        
        # Create training config
        train_config = create_default_config()
        train_config['model'] = model_config
        train_config['num_epochs'] = 1
        
        # Create trainer
        trainer = VoxConverseTrainer(model, train_config)
        
        # Test checkpoint saving/loading
        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
            
            # Save checkpoint
            trainer.save_checkpoint(checkpoint_path, epoch=0, loss=1.0)
            assert os.path.exists(checkpoint_path), "Checkpoint not saved"
            
            # Load checkpoint
            epoch = trainer.load_checkpoint(checkpoint_path)
            assert epoch == 0, "Wrong epoch loaded"
            
            print("✅ Integration test successful")
            print(f"   Checkpoint save/load: OK")
            
        finally:
            shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🔬 RUNNING COMPREHENSIVE TESTS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Loss Function", test_loss_function),
        ("Configuration", test_config_creation),
        ("Trainer Creation", test_trainer_creation),
        ("Data Analyzer", test_data_analyzer),
        ("Dataset Utils", test_dataset_utils),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n\n📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The reorganized project is working correctly.")
        print("\n🚀 Ready for production use:")
        print("• Run data analysis: python -c \"from src.training.data_analyzer import main; main()\"")
        print("• Test full pipeline: python demo_project.py")
        print("• Start training: python train_voxconverse.py --analyze_data")
        
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("• Ensure all dependencies are installed: pip install -r requirements.txt")
        print("• Check Python version compatibility (>=3.7)")
        print("• Verify PyTorch installation with CUDA if needed")
        
        return False


def main():
    """Main test function."""
    print("🧪 VoxConverse Project Test Suite")
    print("Testing the reorganized project structure and functionality")
    print()
    
    success = run_all_tests()
    
    print(f"\n{'='*60}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("✅ All tests passed - project is ready to use!")
        return 0
    else:
        print("❌ Some tests failed - please address the issues")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)