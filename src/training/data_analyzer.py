"""
Data analyzer for VoxConverse dataset.
Implements quality analysis, class distribution analysis, and recommendations
from the voxconverse_explorer.ipynb notebook.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Import the VoxConverse dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voxconverse_dataset import VoxConverseDataset

# Configuration matplotlib
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class VoxConverseDataAnalyzer:
    """
    Analyzer for VoxConverse dataset quality and class distribution.
    Based on the analysis from voxconverse_explorer.ipynb.
    """
    
    def __init__(self, segment_duration=60.0, hop_duration=30.0, 
                 sample_rate=16000, n_mels=80):
        """
        Initialize the data analyzer.
        
        Args:
            segment_duration: Duration of segments in seconds
            hop_duration: Hop between segments in seconds
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
        """
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Quality thresholds
        self.quality_threshold_vad = 0.0
        self.quality_threshold_osd = 100.0
        
        # Color palette
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        print(f"🔧 VoxConverse Data Analyzer initialized")
        print(f"   📏 Segment duration: {segment_duration}s")
        print(f"   🎵 Sample rate: {sample_rate} Hz")
        print(f"   🔊 Mel bands: {n_mels}")
    
    def load_dataset(self, split='dev', min_speaker_duration=0.5):
        """Load VoxConverse dataset with current parameters."""
        print(f"🔄 Loading VoxConverse dataset...")
        print(f"   Split: {split}")
        print(f"   Segment duration: {self.segment_duration}s")
        print(f"   Hop duration: {self.hop_duration}s")
        
        self.dataset = VoxConverseDataset(
            split=split,
            segment_duration=self.segment_duration,
            hop_duration=self.hop_duration,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            min_speaker_duration=min_speaker_duration
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   📦 Total segments: {len(self.dataset)}")
        
        if len(self.dataset) == 0:
            raise ValueError("❌ No segments found in dataset!")
        
        return self.dataset
    
    def analyze_dataset_quality(self, max_segments=1000):
        """
        Comprehensive quality analysis of the dataset.
        
        Args:
            max_segments: Maximum number of segments to analyze
            
        Returns:
            DataFrame with quality statistics
        """
        if not hasattr(self, 'dataset'):
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"🔍 Analyzing dataset quality on {min(len(self.dataset), max_segments)} segments...")
        
        # Collect statistics
        stats = {
            'vad_frames': [],
            'osd_frames': [], 
            'vcn_frames': [],
            'total_frames': [],
            'has_speech': [],
            'has_overlap': [],
            'has_voice_change': [],
            'conv_idx': [],
            'start_time': []
        }
        
        # Analyze sample of segments
        sample_size = min(len(self.dataset), max_segments)
        for i in range(sample_size):
            if i % 100 == 0:
                print(f"   Progress: {i}/{sample_size}")
                
            segment = self.dataset.segments[i]
            stats['vad_frames'].append(segment.get('vad_frames', 0))
            stats['osd_frames'].append(segment.get('osd_frames', 0))
            stats['vcn_frames'].append(segment.get('vcn_frames', 0))
            stats['total_frames'].append(int(self.segment_duration / 0.02))  # 20ms frames
            stats['has_speech'].append(segment.get('vad_frames', 0) > 0)
            stats['has_overlap'].append(segment.get('has_overlap', False))
            stats['has_voice_change'].append(segment.get('has_voice_change', False))
            stats['conv_idx'].append(segment.get('conv_idx', -1))
            stats['start_time'].append(segment.get('start_time', 0))
        
        # Convert to DataFrame
        df = pd.DataFrame(stats)
        
        # Calculate ratios
        df['vad_ratio'] = df['vad_frames'] / df['total_frames']
        df['osd_ratio'] = df['osd_frames'] / df['total_frames'] 
        df['vcn_ratio'] = df['vcn_frames'] / df['total_frames']
        
        self.quality_df = df
        print(f"✅ Quality analysis completed on {len(df)} segments")
        
        return df
    
    def visualize_quality_analysis(self, figsize=(16, 12), save_path=None):
        """Create comprehensive quality visualization."""
        if not hasattr(self, 'quality_df'):
            raise ValueError("Quality analysis not done. Call analyze_dataset_quality() first.")
        
        df = self.quality_df
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('📊 VoxConverse Dataset Quality Analysis', fontsize=16, fontweight='bold')
        
        # 1. VAD distribution
        axes[0,0].hist(df['vad_ratio'], bins=50, alpha=0.7, color=self.color_palette[0], edgecolor='black')
        axes[0,0].axvline(self.quality_threshold_vad, color='red', linestyle='--', 
                         label=f'Quality threshold: {self.quality_threshold_vad}')
        axes[0,0].set_title('VAD Distribution (Voice Activity)')
        axes[0,0].set_xlabel('Ratio of frames with voice')
        axes[0,0].set_ylabel('Number of segments')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. OSD distribution  
        axes[0,1].hist(df['osd_ratio'], bins=50, alpha=0.7, color=self.color_palette[1], edgecolor='black')
        axes[0,1].axvline(self.quality_threshold_osd, color='red', linestyle='--', 
                         label=f'Quality threshold: {self.quality_threshold_osd}')
        axes[0,1].set_title('OSD Distribution (Overlap Speech)')
        axes[0,1].set_xlabel('Ratio of frames with overlap')
        axes[0,1].set_ylabel('Number of segments')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. VCN distribution
        axes[0,2].hist(df['vcn_ratio'], bins=50, alpha=0.7, color=self.color_palette[2], edgecolor='black')
        axes[0,2].set_title('VCN Distribution (Voice Change)')
        axes[0,2].set_xlabel('Ratio of frames with change')
        axes[0,2].set_ylabel('Number of segments')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. VAD vs OSD scatter
        axes[1,0].scatter(df['vad_ratio'], df['osd_ratio'], alpha=0.5, color=self.color_palette[3])
        axes[1,0].set_title('VAD vs OSD')
        axes[1,0].set_xlabel('VAD Ratio')
        axes[1,0].set_ylabel('OSD Ratio')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Segments per conversation
        conv_counts = df['conv_idx'].value_counts().head(20)
        axes[1,1].bar(range(len(conv_counts)), conv_counts.values, color=self.color_palette[4])
        axes[1,1].set_title('Segments per Conversation (Top 20)')
        axes[1,1].set_xlabel('Conversations')
        axes[1,1].set_ylabel('Number of segments')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Activity percentages
        percentages = [
            df['has_speech'].sum() / len(df) * 100,
            df['has_overlap'].sum() / len(df) * 100, 
            df['has_voice_change'].sum() / len(df) * 100
        ]
        labels = ['VAD\n(Speech)', 'OSD\n(Overlap)', 'VCN\n(Change)']
        bars = axes[1,2].bar(labels, percentages, color=self.color_palette[:3])
        axes[1,2].set_title('Percentage of segments with activity')
        axes[1,2].set_ylabel('Percentage (%)')
        axes[1,2].grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Quality visualization saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        self._print_quality_summary()
    
    def _print_quality_summary(self):
        """Print summary statistics."""
        df = self.quality_df
        
        print(f"\n📈 QUALITY STATISTICS SUMMARY")
        print(f"="*50)
        print(f"📊 Segments analyzed: {len(df)}")
        print(f"📏 Duration per segment: {self.segment_duration}s")
        print(f"")
        print(f"🎤 VAD (Voice Activity Detection):")
        print(f"   Segments with speech: {df['has_speech'].sum()} ({df['has_speech'].mean()*100:.1f}%)")
        print(f"   Average VAD ratio: {df['vad_ratio'].mean():.3f} ± {df['vad_ratio'].std():.3f}")
        print(f"   Quality segments (>{self.quality_threshold_vad}): {(df['vad_ratio'] > self.quality_threshold_vad).sum()}")
        print(f"")
        print(f"🗣️  OSD (Overlap Speech Detection):")
        print(f"   Segments with overlap: {df['has_overlap'].sum()} ({df['has_overlap'].mean()*100:.1f}%)")
        print(f"   Average OSD ratio: {df['osd_ratio'].mean():.3f} ± {df['osd_ratio'].std():.3f}")
        print(f"   Low overlap segments (<{self.quality_threshold_osd}): {(df['osd_ratio'] < self.quality_threshold_osd).sum()}")
        print(f"")
        print(f"🔄 VCN (Voice Change Detection):")
        print(f"   Segments with changes: {df['has_voice_change'].sum()} ({df['has_voice_change'].mean()*100:.1f}%)")
        print(f"   Average VCN ratio: {df['vcn_ratio'].mean():.3f} ± {df['vcn_ratio'].std():.3f}")
    
    def analyze_class_distribution(self, sample_size=1000):
        """
        Analyze class distribution for training with imbalance recommendations.
        
        Args:
            sample_size: Number of segments to analyze
            
        Returns:
            Dictionary with distribution statistics
        """
        if not hasattr(self, 'dataset'):
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"🔍 ANALYZING CLASS DISTRIBUTION FOR TRAINING")
        print("=" * 60)
        
        # Collect all frame labels
        all_vad_frames = []
        all_osd_frames = []
        all_vcn_frames = []
        
        print(f"📊 Sampling {min(sample_size, len(self.dataset))} segments...")
        
        for i in range(min(sample_size, len(self.dataset))):
            if i % 100 == 0:
                print(f"   Progress: {i}/{min(sample_size, len(self.dataset))}")
                
            sample = self.dataset[i]
            
            # Get frame-by-frame labels
            vad_labels = sample['vad_labels'].numpy().flatten()
            osd_labels = sample['osd_labels'].numpy().flatten()
            vcn_labels = sample['vcn_labels'].numpy().flatten()
            
            all_vad_frames.extend(vad_labels)
            all_osd_frames.extend(osd_labels)
            all_vcn_frames.extend(vcn_labels)
        
        # Convert to numpy arrays
        vad_array = np.array(all_vad_frames)
        osd_array = np.array(all_osd_frames)
        vcn_array = np.array(all_vcn_frames)
        
        # Calculate statistics
        total_frames = len(vad_array)
        
        # Count positive frames (> 0.5 for continuous labels)
        vad_positive = np.sum(vad_array > 0.5)
        osd_positive = np.sum(osd_array > 0.5)
        vcn_positive = np.sum(vcn_array > 0.5)
        
        # Count negative frames
        vad_negative = total_frames - vad_positive
        osd_negative = total_frames - osd_positive
        vcn_negative = total_frames - vcn_positive
        
        # Calculate percentages
        vad_pos_pct = vad_positive / total_frames * 100
        osd_pos_pct = osd_positive / total_frames * 100
        vcn_pos_pct = vcn_positive / total_frames * 100
        
        # Calculate imbalance ratios
        ratios = {
            'VAD': vad_negative/max(vad_positive,1),
            'OSD': osd_negative/max(osd_positive,1),
            'VCN': vcn_negative/max(vcn_positive,1)
        }
        
        distribution_stats = {
            'total_frames': total_frames,
            'vad_positive': vad_positive,
            'osd_positive': osd_positive,
            'vcn_positive': vcn_positive,
            'vad_negative': vad_negative,
            'osd_negative': osd_negative,
            'vcn_negative': vcn_negative,
            'ratios': ratios,
            'percentages': {
                'vad_pos_pct': vad_pos_pct,
                'osd_pos_pct': osd_pos_pct,
                'vcn_pos_pct': vcn_pos_pct
            }
        }
        
        self.distribution_stats = distribution_stats
        self._print_distribution_summary(distribution_stats)
        
        return distribution_stats
    
    def _print_distribution_summary(self, stats):
        """Print class distribution summary."""
        print(f"\n📈 CLASS DISTRIBUTION (on {stats['total_frames']:,} frames):")
        print("-" * 60)
        print(f"🎤 VAD (Voice Activity Detection):")
        print(f"   Positive (speech):     {stats['vad_positive']:8,} frames ({stats['percentages']['vad_pos_pct']:5.1f}%)")
        print(f"   Negative (silence):    {stats['vad_negative']:8,} frames ({100-stats['percentages']['vad_pos_pct']:5.1f}%)")
        print(f"   Imbalance ratio:       1:{stats['ratios']['VAD']:.1f}")
        
        print(f"\n🗣️  OSD (Overlap Speech Detection):")
        print(f"   Positive (overlap):    {stats['osd_positive']:8,} frames ({stats['percentages']['osd_pos_pct']:5.1f}%)")
        print(f"   Negative (no overlap): {stats['osd_negative']:8,} frames ({100-stats['percentages']['osd_pos_pct']:5.1f}%)")
        print(f"   Imbalance ratio:       1:{stats['ratios']['OSD']:.1f}")
        
        print(f"\n🔄 VCN (Voice Change Detection):")
        print(f"   Positive (change):     {stats['vcn_positive']:8,} frames ({stats['percentages']['vcn_pos_pct']:5.1f}%)")
        print(f"   Negative (no change):  {stats['vcn_negative']:8,} frames ({100-stats['percentages']['vcn_pos_pct']:5.1f}%)")
        print(f"   Imbalance ratio:       1:{stats['ratios']['VCN']:.1f}")
    
    def get_training_recommendations(self):
        """
        Provide training recommendations based on class imbalance analysis.
        """
        if not hasattr(self, 'distribution_stats'):
            raise ValueError("Class distribution not analyzed. Call analyze_class_distribution() first.")
        
        stats = self.distribution_stats
        ratios = stats['ratios']
        
        print(f"\n🎯 TRAINING RECOMMENDATIONS BASED ON STATE-OF-THE-ART")
        print("=" * 60)
        
        # Calculate class weights (inverse frequency)
        vad_weight = stats['total_frames'] / (2 * stats['vad_positive'])
        osd_weight = stats['total_frames'] / (2 * stats['osd_positive']) 
        vcn_weight = stats['total_frames'] / (2 * stats['vcn_positive'])
        
        print(f"🏆 STRATEGY 1: CLASS WEIGHTS (RECOMMENDED)")
        print(f"PyTorch implementation:")
        print(f"```python")
        print(f"# Automatically calculated weights:")
        print(f"class_weights_vad = torch.tensor([1.0, {vad_weight:.2f}])  # [negative, positive]")
        print(f"class_weights_osd = torch.tensor([1.0, {osd_weight:.2f}])")  
        print(f"class_weights_vcn = torch.tensor([1.0, {vcn_weight:.2f}])")
        print(f"")
        print(f"# Usage in training:")
        print(f"criterion_vad = nn.BCEWithLogitsLoss(pos_weight=torch.tensor({vad_weight:.2f}))")
        print(f"criterion_osd = nn.BCEWithLogitsLoss(pos_weight=torch.tensor({osd_weight:.2f}))")
        print(f"criterion_vcn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor({vcn_weight:.2f}))")
        print(f"```")
        
        # Focal Loss recommendation
        print(f"\n🔥 STRATEGY 2: FOCAL LOSS")
        most_imbalanced = max(ratios.items(), key=lambda x: x[1])
        
        if most_imbalanced[1] > 20:
            print("✅ HIGHLY RECOMMENDED - You have severely imbalanced classes")
            print("```python")
            print("# Focal Loss implementation")
            print("class FocalLoss(nn.Module):")
            print("    def __init__(self, alpha=0.25, gamma=2):")
            print("        super().__init__()")
            print("        self.alpha = alpha")
            print("        self.gamma = gamma")
            print("    def forward(self, inputs, targets):")
            print("        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')")
            print("        pt = torch.exp(-bce_loss)")
            print("        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss")
            print("        return focal_loss.mean()")
            print("```")
        else:
            print("⚠️  Optional - Your imbalances are manageable with class weights")
        
        # Final recommendation
        print(f"\n🎯 FINAL RECOMMENDATION:")
        if most_imbalanced[1] > 20:
            strategy = "Focal Loss + Class Weights"
            reasoning = f"Class {most_imbalanced[0]} severely imbalanced ({most_imbalanced[1]:.1f}:1)"
        elif most_imbalanced[1] > 10:
            strategy = "Class Weights + Light undersampling"
            reasoning = f"Moderate to severe imbalance ({most_imbalanced[1]:.1f}:1)"
        else:
            strategy = "Class Weights only"
            reasoning = f"Acceptable imbalance ({most_imbalanced[1]:.1f}:1)"
        
        print(f"✅ Recommended strategy: {strategy}")
        print(f"💡 Reason: {reasoning}")
        
        print(f"\n📚 STATE-OF-THE-ART REFERENCES:")
        print("• Lin et al. (2017): Focal Loss for Dense Object Detection")
        print("• He & Garcia (2009): Learning from Imbalanced Data") 
        print("• Wang et al. (2021): Multi-class imbalanced learning for diarization")
        
        return {
            'strategy': strategy,
            'class_weights': {
                'vad': vad_weight,
                'osd': osd_weight,
                'vcn': vcn_weight
            },
            'use_focal_loss': most_imbalanced[1] > 20,
            'most_imbalanced_class': most_imbalanced[0],
            'worst_ratio': most_imbalanced[1]
        }
    
    def export_analysis(self, output_dir='./analysis_results'):
        """Export analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export quality analysis
        if hasattr(self, 'quality_df'):
            quality_path = os.path.join(output_dir, f"quality_analysis_{timestamp}.csv")
            self.quality_df.to_csv(quality_path, index=False)
            print(f"📊 Quality analysis exported: {quality_path}")
        
        # Export distribution stats
        if hasattr(self, 'distribution_stats'):
            import json
            stats_path = os.path.join(output_dir, f"distribution_stats_{timestamp}.json")
            with open(stats_path, 'w') as f:
                json.dump(self.distribution_stats, f, indent=2)
            print(f"📈 Distribution stats exported: {stats_path}")
        
        return output_dir


def main():
    """Main function for standalone execution."""
    print("🔍 VoxConverse Data Analyzer")
    print("=" * 50)
    
    # Configuration
    SEGMENT_DURATION = 60.0
    HOP_DURATION = 30.0
    SAMPLE_RATE = 16000
    N_MELS = 80
    
    # Initialize analyzer
    analyzer = VoxConverseDataAnalyzer(
        segment_duration=SEGMENT_DURATION,
        hop_duration=HOP_DURATION,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS
    )
    
    # Load dataset
    dataset = analyzer.load_dataset(split='dev')
    
    # Quality analysis
    quality_df = analyzer.analyze_dataset_quality(max_segments=500)
    analyzer.visualize_quality_analysis(save_path='./quality_analysis.png')
    
    # Class distribution analysis
    distribution_stats = analyzer.analyze_class_distribution(sample_size=500)
    
    # Get training recommendations
    recommendations = analyzer.get_training_recommendations()
    
    # Export results
    analyzer.export_analysis()
    
    print(f"\n✅ Analysis completed!")
    print(f"📊 Analyzed {len(quality_df)} segments")
    print(f"💡 Training strategy: {recommendations['strategy']}")


if __name__ == "__main__":
    main()