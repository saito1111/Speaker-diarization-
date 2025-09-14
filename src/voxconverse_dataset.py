import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class VoxConverseDataset(Dataset):
    """
    Dataset adapter for VoxConverse speaker diarization.
    Converts VoxConverse format to training-ready tensors.
    """
    
    def __init__(self, 
                 split='dev',
                 segment_duration=4.0,
                 hop_duration=2.0,
                 sample_rate=16000,
                 n_mels=80,
                 max_speakers=8,
                 min_speaker_duration=0.5,
                 pin_memory=True):
        """
        Args:
            split: 'dev' or 'test'
            segment_duration: Duration of each training segment in seconds
            hop_duration: Hop between segments in seconds
            sample_rate: Target sample rate
            n_mels: Number of mel features
            max_speakers: Maximum number of speakers to handle
            min_speaker_duration: Minimum duration for a speaker to be included
            pin_memory: Legacy parameter, now handled by DataLoader pin_memory setting
        """
        self.split = split
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_speakers = max_speakers
        self.min_speaker_duration = min_speaker_duration
        self.pin_memory = pin_memory
        
        # Load VoxConverse dataset
        print(f"Loading VoxConverse {split} split...")
        self.dataset = load_dataset("diarizers-community/voxconverse")[split]
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=256,  # 16ms hop
            n_mels=n_mels,
            f_min=20,
            f_max=8000
        )
        
        # Prepare segments
        self.segments = self._prepare_segments()
        
        print(f"✅ Prepared {len(self.segments)} segments from {len(self.dataset)} conversations")
    
    def _prepare_segments(self) -> List[Dict]:
        """Prepare training segments from conversations."""
        segments = []
        
        for conv_idx, conversation in enumerate(self.dataset):
            if conv_idx % 50 == 0:
                print(f"  Processing conversation {conv_idx+1}/{len(self.dataset)}")
            
            try:
                audio_array = np.array(conversation['audio']['array'], dtype=np.float32)
                timestamps_start = conversation['timestamps_start']
                timestamps_end = conversation['timestamps_end']
                speakers = conversation['speakers']
                
                # Calculate conversation duration
                conv_duration = len(audio_array) / self.sample_rate
                
                # Create segments with sliding window
                start_time = 0.0
                while start_time + self.segment_duration <= conv_duration:
                    end_time = start_time + self.segment_duration
                    
                    # Extract audio segment
                    start_sample = int(start_time * self.sample_rate)
                    end_sample = int(end_time * self.sample_rate)
                    audio_segment = audio_array[start_sample:end_sample]
                    
                    # Create diarization labels for this segment
                    segment_labels = self._create_segment_labels(
                        start_time, end_time, timestamps_start, timestamps_end, speakers
                    )
                    
                    # Only keep segments with some speech activity
                    if segment_labels['has_speech']:
                        segments.append({
                            'audio': audio_segment,
                            'vad_labels': segment_labels['vad'],
                            'osd_labels': segment_labels['osd'],
                            'speaker_ids': segment_labels['speaker_ids'],
                            'conv_idx': conv_idx,
                            'start_time': start_time,
                            'end_time': end_time
                        })
                    
                    start_time += self.hop_duration
            
            except Exception as e:
                print(f"⚠️  Error processing conversation {conv_idx}: {e}")
                continue
        
        return segments
    
    def _create_segment_labels(self, segment_start, segment_end, 
                              timestamps_start, timestamps_end, speakers) -> Dict:
        """Create labels for a segment."""
        # Parameters for time discretization
        frame_duration = 0.02  # 20ms frames (matches 256 hop_length at 16kHz)
        num_frames = int(self.segment_duration / frame_duration)
        
        # Initialize labels
        vad_labels = np.zeros((num_frames, self.max_speakers), dtype=np.float32)
        osd_labels = np.zeros(num_frames, dtype=np.float32)
        
        # Map speaker names to indices
        unique_speakers = list(set(speakers))
        speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers[:self.max_speakers])}
        
        has_speech = False
        
        # Process each annotation
        for start, end, speaker in zip(timestamps_start, timestamps_end, speakers):
            # Check if this annotation overlaps with our segment
            if end <= segment_start or start >= segment_end:
                continue
                
            # Clip to segment boundaries
            clipped_start = max(start, segment_start)
            clipped_end = min(end, segment_end)
            
            if clipped_end - clipped_start < 0.1:  # Skip very short segments
                continue
                
            # Convert to frame indices
            start_frame = int((clipped_start - segment_start) / frame_duration)
            end_frame = int((clipped_end - segment_start) / frame_duration)
            
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, num_frames))
            
            # Set VAD labels
            if speaker in speaker_to_idx:
                speaker_idx = speaker_to_idx[speaker]
                vad_labels[start_frame:end_frame, speaker_idx] = 1.0
                has_speech = True
        
        # Compute OSD labels (overlap detection)
        speaker_activity = vad_labels.sum(axis=1)  # Number of active speakers per frame
        osd_labels = (speaker_activity > 1).astype(np.float32)  # Overlap when >1 speaker
        
        return {
            'vad': vad_labels,
            'osd': osd_labels,
            'speaker_ids': list(speaker_to_idx.keys()),
            'has_speech': has_speech
        }
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        segment = self.segments[idx]
        
        # Convert audio to tensor
        audio = torch.from_numpy(segment['audio']).float()
        
        # Extract mel features
        with torch.no_grad():
            mel_features = self.mel_transform(audio)
            log_mel = torch.log(mel_features + 1e-8)
        
        # Convert labels to tensors
        vad_labels = torch.from_numpy(segment['vad_labels']).float()
        osd_labels = torch.from_numpy(segment['osd_labels']).float()
        
        # Ensure consistent time dimension
        time_frames = log_mel.shape[1]
        target_frames = int(self.segment_duration / 0.02)  # 20ms frames
        
        # Resize mel features if needed
        if time_frames != target_frames:
            log_mel = F.interpolate(
                log_mel.unsqueeze(0), 
                size=target_frames, 
                mode='linear',
                align_corners=False
            ).squeeze(0)
        
        # Resize labels if needed
        if vad_labels.shape[0] != target_frames:
            vad_labels = F.interpolate(
                vad_labels.transpose(0, 1).unsqueeze(0),
                size=target_frames,
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(0, 1)
            
            osd_labels = F.interpolate(
                osd_labels.unsqueeze(0).unsqueeze(0),
                size=target_frames,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        return {
            'features': log_mel,  # [n_mels, time]
            'vad_labels': vad_labels,  # [time, max_speakers]
            'osd_labels': osd_labels,  # [time]
            'audio': audio,  # Original audio for reference
            'conv_idx': segment['conv_idx'],
            'start_time': segment['start_time']
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    features = torch.stack([item['features'] for item in batch])
    vad_labels = torch.stack([item['vad_labels'] for item in batch])
    osd_labels = torch.stack([item['osd_labels'] for item in batch])
    
    return {
        'features': features,
        'vad_labels': vad_labels,
        'osd_labels': osd_labels,
    }


def create_voxconverse_dataloaders(batch_size=16, 
                                   num_workers=4,
                                   segment_duration=4.0,
                                   validation_split=0.1,
                                   pin_memory=True,
                                   persistent_workers=False,
                                   prefetch_factor=2,
                                   worker_init_fn=None,
                                   **dataset_kwargs):
    """
    Create train and validation dataloaders from VoxConverse.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
        segment_duration: Duration of each training segment
        validation_split: Fraction of dev set to use for validation
        pin_memory: If True, tensors will be allocated in pinned memory for faster GPU transfer
        persistent_workers: If True, keep worker processes alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        worker_init_fn: Function to initialize worker processes
        **dataset_kwargs: Additional arguments for VoxConverseDataset
    """
    
    # Separate DataLoader-specific arguments from Dataset arguments
    dataloader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': collate_fn
    }
    
    # Add DataLoader-specific arguments if using multiple workers
    if num_workers > 0:
        dataloader_args['persistent_workers'] = persistent_workers
        dataloader_args['prefetch_factor'] = prefetch_factor
        if worker_init_fn is not None:
            dataloader_args['worker_init_fn'] = worker_init_fn
    
    # Create dataset (only pass dataset-specific arguments)
    dataset_args = {k: v for k, v in dataset_kwargs.items() 
                    if k not in ['persistent_workers', 'prefetch_factor', 'worker_init_fn']}
    
    full_dataset = VoxConverseDataset(
        split='dev',  # Use dev split for both train/val
        segment_duration=segment_duration,
        pin_memory=pin_memory,
        **dataset_args
    )
    
    # Split into train/validation
    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_args
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_args
    )
    
    print(f"📊 Dataset split:")
    print(f"  Train: {len(train_dataset)} segments")
    print(f"  Val:   {len(val_dataset)} segments")
    print(f"  Total: {total_size} segments")
    
    return train_loader, val_loader


def create_test_dataloader(batch_size=16, 
                          num_workers=4, 
                          pin_memory=True,
                          persistent_workers=False,
                          prefetch_factor=2,
                          worker_init_fn=None,
                          **dataset_kwargs):
    """Create test dataloader from VoxConverse test split."""
    
    # Separate DataLoader-specific arguments from Dataset arguments
    dataloader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': collate_fn,
        'shuffle': False
    }
    
    # Add DataLoader-specific arguments if using multiple workers
    if num_workers > 0:
        dataloader_args['persistent_workers'] = persistent_workers
        dataloader_args['prefetch_factor'] = prefetch_factor
        if worker_init_fn is not None:
            dataloader_args['worker_init_fn'] = worker_init_fn
    
    # Create dataset (only pass dataset-specific arguments)
    dataset_args = {k: v for k, v in dataset_kwargs.items() 
                    if k not in ['persistent_workers', 'prefetch_factor', 'worker_init_fn']}
    
    test_dataset = VoxConverseDataset(
        split='test',
        pin_memory=pin_memory,
        **dataset_args
    )
    
    test_loader = DataLoader(
        test_dataset,
        **dataloader_args
    )
    
    print(f"📊 Test dataset: {len(test_dataset)} segments")
    
    return test_loader


if __name__ == "__main__":
    # Test the dataset
    print("🧪 Testing VoxConverse dataset...")
    
    # Create small dataset for testing
    dataset = VoxConverseDataset(
        split='dev',
        segment_duration=4.0,
        max_speakers=4
    )
    
    print(f"✅ Dataset created with {len(dataset)} segments")
    
    # Test a sample
    sample = dataset[0]
    print(f"📊 Sample shapes:")
    print(f"  Features: {sample['features'].shape}")
    print(f"  VAD labels: {sample['vad_labels'].shape}")
    print(f"  OSD labels: {sample['osd_labels'].shape}")
    
    # Test dataloader
    train_loader, val_loader = create_voxconverse_dataloaders(
        batch_size=4,
        num_workers=0,  # For testing
        segment_duration=4.0,
        persistent_workers=False,  # Not needed with num_workers=0
        pin_memory=False  # For testing
    )
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"📊 Batch shapes:")
    print(f"  Features: {batch['features'].shape}")
    print(f"  VAD labels: {batch['vad_labels'].shape}")
    print(f"  OSD labels: {batch['osd_labels'].shape}")
    
    print("✅ VoxConverse dataset test passed!")