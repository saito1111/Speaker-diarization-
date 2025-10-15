"""
Dataset utilities for VoxConverse.
Implements dataset generation, filtering, and export functionality
from the voxconverse_explorer.ipynb notebook.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchaudio
from datetime import datetime
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Import the VoxConverse dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voxconverse_dataset import VoxConverseDataset


class VoxConverseDatasetUtils:
    """
    Utilities for VoxConverse dataset manipulation and analysis.
    Based on functionality from voxconverse_explorer.ipynb.
    """
    
    def __init__(self, segment_duration=60.0, hop_duration=30.0, 
                 sample_rate=16000, n_mels=80):
        """
        Initialize dataset utilities.
        
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
        
        print(f"🔧 VoxConverse Dataset Utils initialized")
        print(f"   📏 Segment duration: {segment_duration}s")
        print(f"   🎵 Sample rate: {sample_rate} Hz")
    
    def create_custom_dataset(self, split='dev', min_vad_ratio=0.1, max_osd_ratio=0.3,
                             require_voice_changes=False, max_segments=None,
                             min_speaker_duration=0.5):
        """
        Create a custom filtered dataset with quality criteria.
        
        Args:
            split: Dataset split ('dev', 'test', etc.)
            min_vad_ratio: Minimum VAD ratio required
            max_osd_ratio: Maximum OSD ratio accepted
            require_voice_changes: If True, only keep segments with voice changes
            max_segments: Maximum number of segments (None = all)
            min_speaker_duration: Minimum speaker duration
            
        Returns:
            Filtered VoxConverseDataset
        """
        print(f"🔧 Creating custom dataset...")
        print(f"   📏 Segment duration: {self.segment_duration}s (hop: {self.hop_duration}s)")
        print(f"   🎤 VAD minimum: {min_vad_ratio:.2f}")
        print(f"   🗣️  OSD maximum: {max_osd_ratio:.2f}")
        print(f"   🔄 Voice changes required: {require_voice_changes}")
        print(f"   📦 Max segments: {max_segments or 'None'}")
        
        # Create base dataset
        dataset = VoxConverseDataset(
            split=split,
            segment_duration=self.segment_duration,
            hop_duration=self.hop_duration,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            min_speaker_duration=min_speaker_duration
        )
        
        print(f"📊 Base dataset created: {len(dataset)} segments")
        
        # Filter according to quality criteria
        filtered_segments = []
        
        for i, segment in enumerate(dataset.segments):
            if max_segments and len(filtered_segments) >= max_segments:
                break
                
            total_frames = int(self.segment_duration / 0.02)  # 20ms frames
            vad_ratio = segment.get('vad_frames', 0) / total_frames
            osd_ratio = segment.get('osd_frames', 0) / total_frames
            has_changes = segment.get('has_voice_change', False)
            
            # Apply filters
            if vad_ratio >= min_vad_ratio and osd_ratio <= max_osd_ratio:
                if not require_voice_changes or has_changes:
                    filtered_segments.append(segment)
        
        # Replace segments in dataset
        dataset.segments = filtered_segments
        
        print(f"✅ Filtered dataset created: {len(dataset)} segments")
        reduction_pct = (1 - len(filtered_segments) / len(dataset.segments)) * 100 if len(dataset.segments) > 0 else 0
        print(f"   📉 Reduction: {reduction_pct:.1f}%")
        
        return dataset
    
    def analyze_conversation_boundaries(self, dataset):
        """
        Analyze how segments are distributed across conversations and identify boundary segments.
        
        Args:
            dataset: VoxConverseDataset instance
            
        Returns:
            Tuple of (conversation_stats, truncated_segments)
        """
        print("🔍 ANALYZING CONVERSATION BOUNDARIES")
        print("=" * 60)
        
        # Group segments by conversation
        conv_segments = {}
        for i, segment in enumerate(dataset.segments):
            conv_idx = segment['conv_idx']
            if conv_idx not in conv_segments:
                conv_segments[conv_idx] = []
            conv_segments[conv_idx].append({
                'segment_idx': i,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'conv_idx': conv_idx
            })
        
        print(f"📊 GENERAL STATISTICS:")
        print(f"   Total conversations: {len(conv_segments)}")
        print(f"   Total segments: {len(dataset.segments)}")
        print(f"   Configured segment duration: {self.segment_duration}s")
        print(f"   Hop between segments: {self.hop_duration}s")
        
        # Analyze each conversation
        conversation_stats = []
        segments_truncated = []
        
        for conv_idx in sorted(conv_segments.keys()):
            segments = conv_segments[conv_idx]
            
            # Sort by start time
            segments.sort(key=lambda x: x['start_time'])
            
            # Calculate total coverage
            if segments:
                total_coverage = segments[-1]['end_time']
                num_segments = len(segments)
                
                # Calculate theoretical vs actual duration of last segment
                last_segment = segments[-1]
                expected_duration = self.segment_duration
                actual_duration = last_segment['end_time'] - last_segment['start_time']
                
                # Identify if last segment is truncated
                is_truncated = actual_duration < expected_duration * 0.99  # 1% tolerance
                
                conversation_stats.append({
                    'conv_idx': conv_idx,
                    'num_segments': num_segments,
                    'total_coverage': total_coverage,
                    'last_segment_duration': actual_duration,
                    'is_truncated': is_truncated,
                    'first_segment': segments[0]['segment_idx'],
                    'last_segment': segments[-1]['segment_idx']
                })
                
                if is_truncated:
                    segments_truncated.append({
                        'conv_idx': conv_idx,
                        'segment_idx': last_segment['segment_idx'],
                        'missing_duration': expected_duration - actual_duration
                    })
        
        # Display detailed statistics
        print(f"\n📋 CONVERSATION DETAILS:")
        print(f"{'Conv':<6} {'Segments':<9} {'Duration':<8} {'First':<8} {'Last':<8} {'Truncated':<9}")
        print("-" * 60)
        
        for stats in conversation_stats[:20]:  # Show first 20
            truncated_str = "YES" if stats['is_truncated'] else "NO"
            print(f"{stats['conv_idx']:<6} {stats['num_segments']:<9} {stats['total_coverage']:<8.1f} "
                  f"{stats['first_segment']:<8} {stats['last_segment']:<8} {truncated_str:<9}")
        
        if len(conversation_stats) > 20:
            print(f"... and {len(conversation_stats) - 20} other conversations")
        
        # Analyze truncated segments
        print(f"\n🔪 TRUNCATED SEGMENTS:")
        print(f"   Conversations with truncation: {len(segments_truncated)}")
        print(f"   Percentage truncated: {len(segments_truncated) / len(conversation_stats) * 100:.1f}%")
        
        if segments_truncated:
            print(f"\n   Examples of truncated segments:")
            print(f"   {'Conv':<6} {'Segment':<8} {'Missing Duration (s)':<20}")
            print("   " + "-" * 35)
            for trunc in segments_truncated[:10]:
                print(f"   {trunc['conv_idx']:<6} {trunc['segment_idx']:<8} {trunc['missing_duration']:<20.2f}")
        
        return conversation_stats, segments_truncated
    
    def visualize_conversation_boundaries(self, dataset, max_conversations=10, 
                                        figsize=(16, 8), save_path=None):
        """Visualize conversation boundaries for the first conversations."""
        
        print(f"\n🎨 VISUALIZING CONVERSATION BOUNDARIES")
        
        # Group by conversation with segment indices
        conv_segments = {}
        for i, segment in enumerate(dataset.segments):
            conv_idx = segment['conv_idx']
            if conv_idx not in conv_segments:
                conv_segments[conv_idx] = []
            # Store segment with its original index
            segment_with_idx = {'segment': segment, 'original_idx': i}
            conv_segments[conv_idx].append(segment_with_idx)
        
        # Select first conversations
        selected_convs = sorted(conv_segments.keys())[:max_conversations]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(selected_convs)))
        
        for i, conv_idx in enumerate(selected_convs):
            segments_with_idx = conv_segments[conv_idx]
            
            # Sort by time
            segments_with_idx.sort(key=lambda x: x['segment']['start_time'])
            
            y_pos = i
            for j, seg_data in enumerate(segments_with_idx):
                segment = seg_data['segment']
                original_idx = seg_data['original_idx']
                
                start_time = segment['start_time']
                duration = segment['end_time'] - segment['start_time']
                
                # Different color for last segment if truncated
                is_last = j == len(segments_with_idx) - 1
                is_truncated = duration < self.segment_duration * 0.99
                
                color = colors[i]
                if is_last and is_truncated:
                    color = 'red'  # Red for truncated segments
                    alpha = 0.8
                else:
                    alpha = 0.6
                
                # Draw segment
                rect = plt.Rectangle((start_time, y_pos - 0.4), duration, 0.8, 
                                   facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                # Add segment number at center (use original index)
                ax.text(start_time + duration/2, y_pos, str(original_idx), 
                       ha='center', va='center', fontsize=8, fontweight='bold')
        
        max_time = max([max(seg_data['segment']['end_time'] for seg_data in conv_segments[conv]) 
                       for conv in selected_convs]) + 5
        ax.set_xlim(0, max_time)
        ax.set_ylim(-0.5, len(selected_convs) - 0.5)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Conversations', fontsize=12)
        ax.set_title(f'Conversation Boundaries - Segments per Conversation\\n'
                     f'Red = Truncated segment, Configured duration = {self.segment_duration}s', 
                     fontsize=14, fontweight='bold')
        
        # Conversation legend
        ax.set_yticks(range(len(selected_convs)))
        ax.set_yticklabels([f'Conv {conv}' for conv in selected_convs])
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🎨 Visualization saved to: {save_path}")
        
        plt.show()
        
        print(f"🎨 Legend:")
        print(f"   - Each line = one conversation")
        print(f"   - Each rectangle = one segment")
        print(f"   - Red = truncated segment (< {self.segment_duration}s)")
        print(f"   - Numbers = segment indices in dataset")
    
    def export_segment_analysis(self, dataset, segment_idx, output_dir='./exports'):
        """
        Export complete analysis of a segment (audio, data, visualizations).
        
        Args:
            dataset: VoxConverseDataset instance
            segment_idx: Index of segment to export
            output_dir: Output directory
            
        Returns:
            Dictionary with paths to exported files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if segment_idx >= len(dataset):
            raise ValueError(f"Segment {segment_idx} does not exist")
        
        print(f"💾 Exporting segment #{segment_idx}...")
        
        # Get data
        sample = dataset[segment_idx]
        segment_info = dataset.segments[segment_idx]
        
        # Base filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"segment_{segment_idx}_conv_{segment_info['conv_idx']}_{timestamp}"
        
        # 1. Save audio
        audio_path = os.path.join(output_dir, f"{base_name}.wav")
        torchaudio.save(audio_path, sample['audio'].unsqueeze(0), self.sample_rate)
        print(f"   🎵 Audio saved: {audio_path}")
        
        # 2. Save data as pickle
        data_to_save = {
            'segment_idx': segment_idx,
            'segment_info': segment_info,
            'mel_features': sample['features'].numpy(),
            'vad_labels': sample['vad_labels'].numpy(),
            'osd_labels': sample['osd_labels'].numpy(),
            'vcn_labels': sample['vcn_labels'].numpy(),
            'audio': sample['audio'].numpy(),
            'sample_rate': self.sample_rate,
            'segment_duration': self.segment_duration,
            'export_timestamp': timestamp
        }
        
        data_path = os.path.join(output_dir, f"{base_name}_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"   📊 Data saved: {data_path}")
        
        # 3. Create and save visualization
        self._visualize_segment(dataset, segment_idx)
        
        # Save current figure
        viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   📈 Visualization saved: {viz_path}")
        
        # 4. Create text report
        report_path = os.path.join(output_dir, f"{base_name}_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"ANALYSIS REPORT - SEGMENT #{segment_idx}\\n")
            f.write(f"="*50 + "\\n")
            f.write(f"Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Configuration: Duration={self.segment_duration}s, Sample_rate={self.sample_rate}Hz\\n\\n")
            
            f.write(f"SEGMENT INFORMATION:\\n")
            f.write(f"  Index: {segment_idx}\\n")
            f.write(f"  Conversation: {segment_info['conv_idx']}\\n")
            f.write(f"  Time: {segment_info['start_time']:.1f}s - {segment_info['end_time']:.1f}s\\n")
            f.write(f"  Speakers: {segment_info.get('speaker_ids', [])}\\n\\n")
            
            f.write(f"DETECTION STATISTICS:\\n")
            vad_pct = np.mean(sample['vad_labels'].numpy()) * 100
            osd_pct = np.mean(sample['osd_labels'].numpy()) * 100
            vcn_pct = np.mean(sample['vcn_labels'].numpy()) * 100
            
            f.write(f"  VAD (Voice Activity): {vad_pct:.1f}% of frames\\n")
            f.write(f"  OSD (Overlap Speech): {osd_pct:.1f}% of frames\\n")
            f.write(f"  VCN (Voice Change): {vcn_pct:.1f}% of frames\\n\\n")
            
            f.write(f"GENERATED FILES:\\n")
            f.write(f"  - Audio: {base_name}.wav\\n")
            f.write(f"  - Data: {base_name}_data.pkl\\n")
            f.write(f"  - Visualization: {base_name}_visualization.png\\n")
            f.write(f"  - Report: {base_name}_report.txt\\n")
        
        print(f"   📄 Report saved: {report_path}")
        print(f"✅ Export completed in: {output_dir}")
        
        return {
            'audio_path': audio_path,
            'data_path': data_path,
            'viz_path': viz_path,
            'report_path': report_path
        }
    
    def _visualize_segment(self, dataset, segment_idx):
        """Helper function to visualize a single segment."""
        sample = dataset[segment_idx]
        segment_info = dataset.segments[segment_idx]
        
        # Data
        mel_features = sample['features'].numpy()
        vad_labels = sample['vad_labels'].numpy()
        osd_labels = sample['osd_labels'].numpy()
        vcn_labels = sample['vcn_labels'].numpy()
        audio = sample['audio'].numpy()
        
        # Time axes
        time_frames = mel_features.shape[1]
        time_axis = np.linspace(0, self.segment_duration, time_frames)
        audio_time = np.linspace(0, self.segment_duration, len(audio))
        
        # Create figure
        fig, axes = plt.subplots(5, 1, figsize=(16, 12))
        fig.suptitle(f'🎵 Segment #{segment_idx} - Conv {segment_info["conv_idx"]} - '
                     f't={segment_info["start_time"]:.1f}s-{segment_info["end_time"]:.1f}s', 
                     fontsize=14, fontweight='bold')
        
        # 1. Mel Spectrogram
        im = axes[0].imshow(mel_features, aspect='auto', origin='lower',
                            extent=[0, self.segment_duration, 0, self.n_mels],
                            cmap='viridis')
        axes[0].set_title('🎼 Mel Spectrogram', fontweight='bold')
        axes[0].set_ylabel('Mel Bands')
        plt.colorbar(im, ax=axes[0], shrink=0.8)
        
        # 2. Audio signal
        axes[1].plot(audio_time, audio, color='blue', alpha=0.7, linewidth=0.8)
        axes[1].set_title('🔊 Audio Signal', fontweight='bold')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        # 3. VAD
        axes[2].fill_between(time_axis, 0, vad_labels, alpha=0.7, color='green')
        axes[2].plot(time_axis, vad_labels, color='darkgreen', linewidth=2)
        axes[2].set_title('🎤 VAD - Voice Activity', fontweight='bold')
        axes[2].set_ylabel('Activation')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(True, alpha=0.3)
        
        # 4. OSD
        axes[3].fill_between(time_axis, 0, osd_labels, alpha=0.7, color='orange')
        axes[3].plot(time_axis, osd_labels, color='darkorange', linewidth=2)
        axes[3].set_title('🗣️  OSD - Overlap Speech', fontweight='bold')
        axes[3].set_ylabel('Activation')
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].grid(True, alpha=0.3)
        
        # 5. VCN
        axes[4].fill_between(time_axis, 0, vcn_labels, alpha=0.7, color='red')
        axes[4].plot(time_axis, vcn_labels, color='darkred', linewidth=2)
        axes[4].set_title('🔄 VCN - Voice Change', fontweight='bold')
        axes[4].set_ylabel('Activation')
        axes[4].set_xlabel('Time (seconds)')
        axes[4].set_ylim(-0.1, 1.1)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def export_dataset_summary(self, dataset, output_dir='./exports'):
        """Export complete dataset summary."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(output_dir, f"dataset_summary_{timestamp}.csv")
        
        print(f"📊 Exporting dataset summary...")
        
        # Create DataFrame with all statistics
        summary_data = []
        for i, segment in enumerate(dataset.segments):
            summary_data.append({
                'segment_idx': i,
                'conv_idx': segment.get('conv_idx', -1),
                'start_time': segment.get('start_time', 0),
                'end_time': segment.get('end_time', 0),
                'vad_frames': segment.get('vad_frames', 0),
                'osd_frames': segment.get('osd_frames', 0),
                'vcn_frames': segment.get('vcn_frames', 0),
                'total_frames': int(self.segment_duration / 0.02),
                'vad_ratio': segment.get('vad_frames', 0) / int(self.segment_duration / 0.02),
                'osd_ratio': segment.get('osd_frames', 0) / int(self.segment_duration / 0.02),
                'vcn_ratio': segment.get('vcn_frames', 0) / int(self.segment_duration / 0.02),
                'has_speech': segment.get('vad_frames', 0) > 0,
                'has_overlap': segment.get('has_overlap', False),
                'has_voice_change': segment.get('has_voice_change', False),
                'num_speakers': len(segment.get('speaker_ids', [])),
                'speakers': str(segment.get('speaker_ids', []))
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(summary_path, index=False, encoding='utf-8')
        
        print(f"✅ Summary saved: {summary_path}")
        print(f"   📊 {len(df_summary)} segments exported")
        
        return df_summary, summary_path


def main():
    """Main function for standalone execution."""
    print("🔧 VoxConverse Dataset Utils")
    print("=" * 50)
    
    # Configuration
    SEGMENT_DURATION = 60.0
    HOP_DURATION = 30.0
    SAMPLE_RATE = 16000
    N_MELS = 80
    
    # Initialize utils
    utils = VoxConverseDatasetUtils(
        segment_duration=SEGMENT_DURATION,
        hop_duration=HOP_DURATION,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS
    )
    
    # Create custom dataset
    custom_dataset = utils.create_custom_dataset(
        split='dev',
        min_vad_ratio=0.3,      # More strict on speech
        max_osd_ratio=0.1,      # Less overlap allowed
        require_voice_changes=False,
        max_segments=100        # Limit for testing
    )
    
    # Analyze conversation boundaries
    conv_stats, truncated = utils.analyze_conversation_boundaries(custom_dataset)
    
    # Visualize boundaries
    utils.visualize_conversation_boundaries(custom_dataset, max_conversations=10)
    
    # Export example segment
    if len(custom_dataset) > 0:
        export_paths = utils.export_segment_analysis(custom_dataset, 0)
        print(f"🎯 Example segment exported to: {export_paths['report_path']}")
    
    # Export dataset summary
    df_summary, summary_path = utils.export_dataset_summary(custom_dataset)
    
    print(f"\n✅ Dataset utils demo completed!")
    print(f"📊 Custom dataset: {len(custom_dataset)} segments")
    print(f"🔪 Truncated conversations: {len(truncated)}")


if __name__ == "__main__":
    main()