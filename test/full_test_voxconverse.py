#!/usr/bin/env python3
"""
Full Test VoxConverse - Script de test complet avec optimisations CUDA
=======================================================================

Ce script teste toutes les fonctionnalités de parallélisme CUDA avec le dataset VoxConverse :
- Pin memory pour transferts GPU optimisés
- Multi-workers (4, 8, 16) pour parallélisation des données
- Persistent workers pour éviter la re-création des processus
- Prefetch factor pour optimiser le pipeline
- Benchmarks de performance détaillés

Usage:
    python full_test_voxconverse.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
import psutil
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict

# Import local modules
sys.path.append(str(Path(__file__).parent / 'src'))
from voxconverse_dataset import create_voxconverse_dataloaders, create_test_dataloader


@dataclass
class TestConfig:
    """Configuration for performance tests."""
    batch_sizes: List[int]
    num_workers_list: List[int]
    pin_memory_options: List[bool]
    persistent_workers_options: List[bool]
    prefetch_factors: List[int]
    segment_duration: float = 4.0
    max_batches_per_test: int = 50
    warmup_batches: int = 5


class DummySpeakerDiarizationModel(nn.Module):
    """
    Dummy model to simulate real workload for speaker diarization.
    Uses typical architecture patterns for benchmarking.
    """
    
    def __init__(self, n_mels=80, max_speakers=8, hidden_dim=256):
        super().__init__()
        self.n_mels = n_mels
        self.max_speakers = max_speakers
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # Temporal modeling with LSTM
        self.lstm = nn.LSTM(512, hidden_dim, num_layers=2, 
                           batch_first=True, bidirectional=True)
        
        # Output heads
        self.vad_head = nn.Linear(hidden_dim * 2, max_speakers)  # Voice Activity Detection
        self.osd_head = nn.Linear(hidden_dim * 2, 1)  # Overlap Speech Detection
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, features):
        """
        Forward pass.
        
        Args:
            features: [batch_size, n_mels, time_frames]
        
        Returns:
            vad_output: [batch_size, time_frames, max_speakers]
            osd_output: [batch_size, time_frames]
        """
        batch_size, n_mels, time_frames = features.shape
        
        # Convolutional feature extraction
        conv_out = self.conv_layers(features)  # [batch, 512, time]
        
        # Transpose for LSTM
        lstm_input = conv_out.transpose(1, 2)  # [batch, timf, 512]
        
        # Temporal modeling
        lstm_out, _ = self.lstm(lstm_input)  # [batch, time, hidden*2]
        lstm_out = self.dropout(lstm_out)
        
        # Predictions
        vad_output = torch.sigmoid(self.vad_head(lstm_out))  # [batch, time, speakers]
        osd_output = torch.sigmoid(self.osd_head(lstm_out)).squeeze(-1)  # [batch, time]
        
        return vad_output, osd_output


class PerformanceBenchmark:
    """Benchmark class for testing different CUDA optimizations."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = []
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize model
        self.model = DummySpeakerDiarizationModel().to(self.device)
        self.criterion_vad = nn.BCELoss()
        self.criterion_osd = nn.BCELoss()
        
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def worker_init_fn(self, worker_id):
        """Initialize worker with different random seed."""
        np.random.seed(torch.initial_seed() % 2**32)
    
    def test_configuration(self, batch_size: int, num_workers: int, 
                          pin_memory: bool, persistent_workers: bool,
                          prefetch_factor: int) -> Dict:
        """Test a specific configuration."""
        
        config_name = f"bs{batch_size}_nw{num_workers}_pm{int(pin_memory)}_pw{int(persistent_workers)}_pf{prefetch_factor}"
        print(f"\n🔥 Testing {config_name}")
        
        try:
            # Create dataloader with specific configuration
            train_loader, val_loader = create_voxconverse_dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=prefetch_factor if num_workers > 0 else 2,
                worker_init_fn=self.worker_init_fn if num_workers > 0 else None,
                segment_duration=self.config.segment_duration,
                validation_split=0.1
            )
            
            # Benchmark training loop
            train_metrics = self._benchmark_dataloader(
                train_loader, "TRAIN", config_name
            )
            
            # Benchmark validation loop
            val_metrics = self._benchmark_dataloader(
                val_loader, "VAL", config_name
            )
            
            # Memory statistics
            if torch.cuda.is_available():
                memory_stats = {
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                    'gpu_memory_cached': torch.cuda.memory_cached() / 1024**3
                }
            else:
                memory_stats = {}
            
            return {
                'config_name': config_name,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'persistent_workers': persistent_workers,
                'prefetch_factor': prefetch_factor,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'memory_stats': memory_stats,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error in configuration {config_name}: {e}")
            return {
                'config_name': config_name,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'persistent_workers': persistent_workers,
                'prefetch_factor': prefetch_factor,
                'error': str(e),
                'success': False
            }
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _benchmark_dataloader(self, dataloader, split_name: str, config_name: str) -> Dict:
        """Benchmark a dataloader."""
        print(f"  🔄 Benchmarking {split_name} dataloader...")
        
        # Set model mode - IMPORTANT: Always use training mode for backward pass
        is_training = split_name == "TRAIN"
        self.model.train()  # Always use training mode to avoid cuDNN issues
        
        times = []
        gpu_times = []
        batch_sizes = []
        memory_usage = []
        cpu_usage = []
        
        # Warmup
        print(f"    🔥 Warmup ({self.config.warmup_batches} batches)...")
        for i, batch in enumerate(dataloader):
            if i >= self.config.warmup_batches:
                break
            self._process_batch(batch, warmup=True, is_training=is_training)
        
        # Actual benchmark
        print(f"    📊 Benchmarking ({self.config.max_batches_per_test} batches)...")
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.config.max_batches_per_test:
                break
            
            # GPU timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_start = time.time()
            
            batch_start = time.time()
            
            # Process batch
            loss, batch_size = self._process_batch(batch, warmup=False, is_training=is_training)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_end = time.time()
                gpu_times.append(gpu_end - gpu_start)
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            
            times.append(batch_time)
            batch_sizes.append(batch_size)
            
            # Memory and CPU usage
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
            cpu_usage.append(psutil.cpu_percent())
            
            if batch_idx % 10 == 0:
                print(f"      Batch {batch_idx:3d}: {batch_time:.3f}s, {batch_size} samples")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_batch_time = np.mean(times)
        std_batch_time = np.std(times)
        total_samples = sum(batch_sizes)
        samples_per_second = total_samples / total_time
        
        metrics = {
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'std_batch_time': std_batch_time,
            'samples_per_second': samples_per_second,
            'total_samples': total_samples,
            'num_batches': len(times),
            'avg_cpu_usage': np.mean(cpu_usage) if cpu_usage else 0,
        }
        
        if gpu_times:
            metrics['avg_gpu_time'] = np.mean(gpu_times)
            metrics['std_gpu_time'] = np.std(gpu_times)
        
        if memory_usage:
            metrics['avg_memory_usage'] = np.mean(memory_usage)
            metrics['max_memory_usage'] = max(memory_usage)
        
        print(f"    ✅ {split_name}: {samples_per_second:.1f} samples/sec, {avg_batch_time:.3f}±{std_batch_time:.3f}s/batch")
        
        return metrics
    
    def _process_batch(self, batch, warmup=False, is_training=True):
        """Process a single batch (simulate training)."""
        features = batch['features'].to(self.device, non_blocking=True)
        vad_labels = batch['vad_labels'].to(self.device, non_blocking=True)
        osd_labels = batch['osd_labels'].to(self.device, non_blocking=True)
        
        batch_size = features.shape[0]
        
        # Use torch.no_grad() for validation to save memory and avoid cuDNN issues
        if not is_training and not warmup:
            with torch.no_grad():
                # Forward pass
                vad_pred, osd_pred = self.model(features)
                
                # Calculate losses
                vad_loss = self.criterion_vad(vad_pred, vad_labels)
                osd_loss = self.criterion_osd(osd_pred, osd_labels)
                total_loss = vad_loss + osd_loss
                
                return total_loss.item(), batch_size
        else:
            # Training mode or warmup - allow gradients
            # Forward pass
            vad_pred, osd_pred = self.model(features)
            
            # Calculate losses
            vad_loss = self.criterion_vad(vad_pred, vad_labels)
            osd_loss = self.criterion_osd(osd_pred, osd_labels)
            total_loss = vad_loss + osd_loss
            
            # Simulate backward pass (only during actual training simulation)
            if not warmup and is_training:
                total_loss.backward()
                # Simulate optimizer step without actually updating
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            
            return total_loss.item(), batch_size
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        print("🚀 Starting Full VoxConverse CUDA Performance Benchmark")
        print("=" * 70)
        
        total_configs = (len(self.config.batch_sizes) * 
                        len(self.config.num_workers_list) * 
                        len(self.config.pin_memory_options) * 
                        len(self.config.persistent_workers_options) * 
                        len(self.config.prefetch_factors))
        
        print(f"📊 Total configurations to test: {total_configs}")
        
        config_count = 0
        
        for batch_size in self.config.batch_sizes:
            for num_workers in self.config.num_workers_list:
                for pin_memory in self.config.pin_memory_options:
                    for persistent_workers in self.config.persistent_workers_options:
                        for prefetch_factor in self.config.prefetch_factors:
                            
                            # Skip invalid combinations
                            if num_workers == 0 and (persistent_workers or prefetch_factor > 2):
                                continue
                            
                            config_count += 1
                            print(f"\n🔄 Configuration {config_count}/{total_configs}")
                            
                            result = self.test_configuration(
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers,
                                prefetch_factor=prefetch_factor
                            )
                            
                            self.results.append(result)
                            
                            # Print intermediate summary
                            if result['success']:
                                train_sps = result['train_metrics']['samples_per_second']
                                val_sps = result['val_metrics']['samples_per_second']
                                print(f"  ✅ Performance: Train {train_sps:.1f} sps, Val {val_sps:.1f} sps")
        
        print("\n🎉 Benchmark completed!")
        self._generate_report()
    
    def _generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "=" * 70)
        print("📊 FULL PERFORMANCE REPORT")
        print("=" * 70)
        
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("❌ No successful configurations found!")
            return
        
        # Find best configurations
        best_train = max(successful_results, 
                        key=lambda x: x['train_metrics']['samples_per_second'])
        best_val = max(successful_results, 
                      key=lambda x: x['val_metrics']['samples_per_second'])
        
        print(f"\n🏆 BEST TRAINING PERFORMANCE:")
        print(f"  Config: {best_train['config_name']}")
        print(f"  Speed: {best_train['train_metrics']['samples_per_second']:.1f} samples/sec")
        print(f"  Batch time: {best_train['train_metrics']['avg_batch_time']:.3f}±{best_train['train_metrics']['std_batch_time']:.3f}s")
        
        print(f"\n🏆 BEST VALIDATION PERFORMANCE:")
        print(f"  Config: {best_val['config_name']}")
        print(f"  Speed: {best_val['val_metrics']['samples_per_second']:.1f} samples/sec")
        print(f"  Batch time: {best_val['val_metrics']['avg_batch_time']:.3f}±{best_val['val_metrics']['std_batch_time']:.3f}s")
        
        # Analysis by parameter
        print(f"\n📈 PARAMETER ANALYSIS:")
        
        # Batch size analysis
        print(f"\n🔹 Batch Size Impact:")
        for bs in sorted(set(r['batch_size'] for r in successful_results)):
            bs_results = [r for r in successful_results if r['batch_size'] == bs]
            avg_train_sps = np.mean([r['train_metrics']['samples_per_second'] for r in bs_results])
            print(f"  Batch Size {bs:2d}: {avg_train_sps:6.1f} avg samples/sec")
        
        # Num workers analysis
        print(f"\n🔹 Number of Workers Impact:")
        for nw in sorted(set(r['num_workers'] for r in successful_results)):
            nw_results = [r for r in successful_results if r['num_workers'] == nw]
            avg_train_sps = np.mean([r['train_metrics']['samples_per_second'] for r in nw_results])
            print(f"  Workers {nw:2d}: {avg_train_sps:6.1f} avg samples/sec")
        
        # Pin memory analysis
        print(f"\n🔹 Pin Memory Impact:")
        for pm in [True, False]:
            pm_results = [r for r in successful_results if r['pin_memory'] == pm]
            if pm_results:
                avg_train_sps = np.mean([r['train_metrics']['samples_per_second'] for r in pm_results])
                print(f"  Pin Memory {str(pm):5s}: {avg_train_sps:6.1f} avg samples/sec")
        
        # Save detailed results
        self._save_results_to_csv()
        
        print(f"\n💾 Detailed results saved to 'voxconverse_benchmark_results.csv'")
        print("=" * 70)
    
    def _save_results_to_csv(self):
        """Save results to CSV for further analysis."""
        rows = []
        
        for result in self.results:
            if not result['success']:
                continue
                
            row = {
                'config_name': result['config_name'],
                'batch_size': result['batch_size'],
                'num_workers': result['num_workers'],
                'pin_memory': result['pin_memory'],
                'persistent_workers': result['persistent_workers'],
                'prefetch_factor': result['prefetch_factor'],
                
                'train_samples_per_sec': result['train_metrics']['samples_per_second'],
                'train_avg_batch_time': result['train_metrics']['avg_batch_time'],
                'train_std_batch_time': result['train_metrics']['std_batch_time'],
                'train_total_time': result['train_metrics']['total_time'],
                
                'val_samples_per_sec': result['val_metrics']['samples_per_second'],
                'val_avg_batch_time': result['val_metrics']['avg_batch_time'],
                'val_std_batch_time': result['val_metrics']['std_batch_time'],
                'val_total_time': result['val_metrics']['total_time'],
            }
            
            # Add memory stats if available
            if 'memory_stats' in result and result['memory_stats']:
                for key, value in result['memory_stats'].items():
                    row[key] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv('../output/voxconverse_benchmark_results.csv', index=False)


def main():
    """Main execution function."""
    print("🎯 VoxConverse CUDA Performance Benchmark")
    print("🔧 Testing all combinations of CUDA optimizations")
    
    # Check prerequisites
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This benchmark requires CUDA.")
        return
    
    # Configuration for comprehensive testing
    config = TestConfig(
        batch_sizes=[8, 16, 32],  # Different batch sizes
        num_workers_list=[0, 4, 8, 16],  # Different worker counts
        pin_memory_options=[True, False],  # Pin memory on/off
        persistent_workers_options=[True, False],  # Persistent workers on/off
        prefetch_factors=[2, 4],  # Different prefetch factors
        segment_duration=4.0,
        max_batches_per_test=20,  # Reduced for faster testing
        warmup_batches=3
    )
    
    # Run benchmark
    benchmark = PerformanceBenchmark(config)
    benchmark.run_full_benchmark()
    
    print("\n🎉 Full benchmark completed!")
    print("📊 Check 'voxconverse_benchmark_results.csv' for detailed results")


if __name__ == "__main__":
    main()