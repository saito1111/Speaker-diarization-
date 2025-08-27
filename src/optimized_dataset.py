import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import warnings
warnings.filterwarnings("ignore")


class AudioFeatureExtractor:
    """Optimized audio feature extraction for diarization."""
    
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=256, 
                 theta_values=[0, np.pi/2, np.pi, 3*np.pi/2], 
                 mic_pairs=[(0, 4), (1, 5), (2, 6), (3, 7)]):
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.theta_values = theta_values
        self.mic_pairs = mic_pairs
        self.c = 343  # Speed of sound in m/s
        
        # Pre-compute transforms
        self.spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.amplitude_to_db = AmplitudeToDB()
        
        # Microphone positions (circular array)
        self.mic_positions = [(0.1 * np.cos(i * np.pi / 4), 0.1 * np.sin(i * np.pi / 4)) 
                             for i in range(8)]
        
        # Pre-compute mic pair info
        self.mic_pairs_info = {}
        for i, j in mic_pairs:
            delta_m = np.sqrt((self.mic_positions[i][0] - self.mic_positions[j][0])**2 + 
                            (self.mic_positions[i][1] - self.mic_positions[j][1])**2)
            theta_m = np.arctan2(self.mic_positions[j][1] - self.mic_positions[i][1],
                               self.mic_positions[j][0] - self.mic_positions[i][0])
            self.mic_pairs_info[(i, j)] = (delta_m, theta_m)
    
    def extract_features(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract LPS, IPD, and AF features from multi-channel audio.
        
        Args:
            waveforms: List of waveforms [num_mics, samples]
            
        Returns:
            features: Concatenated features [num_features, num_frames]
        """
        if not isinstance(waveforms, list):
            waveforms = [waveforms[i] for i in range(waveforms.shape[0])]
        
        # Ensure consistent length
        min_length = min(w.shape[-1] for w in waveforms)
        waveforms = [w[..., :min_length] for w in waveforms]
        
        # Compute spectrograms
        specs = [self.spectrogram_transform(w) for w in waveforms]
        
        # Extract LPS
        lps_features = []
        for spec in specs:
            lps = self.amplitude_to_db(spec.abs())
            lps_features.append(lps.squeeze(0))  # Remove channel dim if present
        
        # Extract IPD
        ipd_features = []
        for i, j in self.mic_pairs:
            if i < len(specs) and j < len(specs):
                ipd = torch.angle(specs[i]) - torch.angle(specs[j])
                ipd_features.append(ipd.squeeze(0))
        
        # Extract AF
        frequencies = torch.linspace(0, self.sample_rate // 2, specs[0].size(-2))
        af_features = []
        
        for theta in self.theta_values:
            af_sum = torch.zeros_like(specs[0].abs().squeeze(0))
            
            for (i, j), ipd in zip(self.mic_pairs, ipd_features):
                if (i, j) in self.mic_pairs_info:
                    delta_m, theta_m = self.mic_pairs_info[(i, j)]
                    v_m_theta = 2 * np.pi * frequencies * delta_m * np.cos(theta_m - theta) / self.c
                    v_m_theta = v_m_theta.unsqueeze(-1)  # [freq, 1]
                    
                    af_sum += torch.cos(v_m_theta - ipd)
            
            af_features.append(af_sum)
        
        # Concatenate all features
        all_features = lps_features + ipd_features + af_features
        
        # Ensure same time dimension
        min_time = min(f.size(-1) for f in all_features)
        all_features = [f[..., :min_time] for f in all_features]
        
        # Stack features
        features = torch.cat(all_features, dim=0)  # [total_features, time]
        
        return features


def parse_rttm_segments(rttm_path: str) -> List[Tuple[str, str, float, float]]:
    """Parse RTTM file and return segments."""
    segments = []
    if not os.path.exists(rttm_path):
        return segments
        
    with open(rttm_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                file_id = parts[1]
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                segments.append((file_id, speaker_id, start_time, duration))
    
    return segments


def segments_to_labels(segments: List[Tuple], num_frames: int, frame_size: float, 
                      num_speakers: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert segments to VAD and OSD labels.
    
    Args:
        segments: List of (file_id, speaker_id, start_time, duration)
        num_frames: Number of time frames
        frame_size: Duration of each frame in seconds
        num_speakers: Maximum number of speakers
        
    Returns:
        vad_labels: [num_frames, num_speakers]
        osd_labels: [num_frames]
    """
    vad_labels = torch.zeros(num_frames, num_speakers)
    osd_labels = torch.zeros(num_frames)
    
    # Map speaker IDs to indices
    speaker_ids = list(set(seg[1] for seg in segments))
    speaker_to_idx = {spk: idx for idx, spk in enumerate(speaker_ids[:num_speakers])}
    
    # Frame-level activity count
    frame_activity = torch.zeros(num_frames)
    
    for file_id, speaker_id, start_time, duration in segments:
        if speaker_id not in speaker_to_idx:
            continue
            
        speaker_idx = speaker_to_idx[speaker_id]
        start_frame = int(start_time / frame_size)
        end_frame = min(int((start_time + duration) / frame_size), num_frames)
        
        if start_frame < num_frames and end_frame > start_frame:
            vad_labels[start_frame:end_frame, speaker_idx] = 1
            frame_activity[start_frame:end_frame] += 1
    
    # OSD: frames with multiple active speakers
    osd_labels = (frame_activity > 1).float()
    
    return vad_labels, osd_labels


class DiarizationDataset(Dataset):
    """Optimized dataset for speaker diarization."""
    
    def __init__(self, audio_dir: str, rttm_dir: str, segment_duration: float = 4.0,
                 hop_length: int = 256, sample_rate: int = 16000, num_speakers: int = 4,
                 max_segments: Optional[int] = None, cache_features: bool = True):
        
        self.audio_dir = audio_dir
        self.rttm_dir = rttm_dir
        self.segment_duration = segment_duration
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.cache_features = cache_features
        
        self.frame_size = hop_length / sample_rate
        self.frames_per_segment = int(segment_duration / self.frame_size)
        
        # Feature extractor
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=sample_rate, hop_length=hop_length
        )
        
        # Find audio files and corresponding RTTM files
        self.audio_files = []
        self.rttm_files = []
        
        if os.path.exists(audio_dir):
            for filename in os.listdir(audio_dir):
                if filename.endswith('.wav'):
                    base_name = filename.replace('.wav', '')
                    rttm_path = os.path.join(rttm_dir, f"{base_name}.rttm")
                    
                    if os.path.exists(rttm_path):
                        self.audio_files.append(os.path.join(audio_dir, filename))
                        self.rttm_files.append(rttm_path)
        
        # Create segments
        self.segments = self._create_segments(max_segments)
        
        # Feature cache
        self.feature_cache = {} if cache_features else None
        
        print(f"Dataset initialized with {len(self.segments)} segments from {len(self.audio_files)} files")
    
    def _create_segments(self, max_segments: Optional[int]) -> List[Dict]:
        """Create audio segments with corresponding labels."""
        segments = []
        
        for audio_path, rttm_path in zip(self.audio_files, self.rttm_files):
            # Load audio info
            try:
                info = torchaudio.info(audio_path)
                audio_duration = info.num_frames / info.sample_rate
                
                # Parse RTTM
                rttm_segments = parse_rttm_segments(rttm_path)
                
                # Create segments
                num_segments = int(audio_duration / self.segment_duration)
                
                for seg_idx in range(num_segments):
                    start_time = seg_idx * self.segment_duration
                    end_time = start_time + self.segment_duration
                    
                    # Filter segments for this time window
                    segment_annotations = []
                    for file_id, speaker_id, seg_start, seg_duration in rttm_segments:
                        seg_end = seg_start + seg_duration
                        
                        # Check overlap with current segment
                        if seg_start < end_time and seg_end > start_time:
                            # Adjust times relative to segment start
                            rel_start = max(0, seg_start - start_time)
                            rel_end = min(self.segment_duration, seg_end - start_time)
                            rel_duration = rel_end - rel_start
                            
                            if rel_duration > 0:
                                segment_annotations.append((file_id, speaker_id, rel_start, rel_duration))
                    
                    segments.append({
                        'audio_path': audio_path,
                        'start_time': start_time,
                        'duration': self.segment_duration,
                        'annotations': segment_annotations
                    })
                    
                    if max_segments and len(segments) >= max_segments:
                        return segments
                        
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        return segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        """Get a single segment with features and labels."""
        segment_info = self.segments[idx]
        
        # Check cache first
        if self.feature_cache and idx in self.feature_cache:
            return self.feature_cache[idx]
        
        try:
            # Load audio segment
            waveforms = self._load_audio_segment(segment_info)
            
            # Extract features
            features = self.feature_extractor.extract_features(waveforms)
            
            # Generate labels
            vad_labels, osd_labels = segments_to_labels(
                segment_info['annotations'], 
                self.frames_per_segment, 
                self.frame_size,
                self.num_speakers
            )
            
            # Ensure consistent dimensions
            actual_frames = min(features.size(-1), vad_labels.size(0), osd_labels.size(0))
            target_frames = self.frames_per_segment
            
            # Pad or truncate to target length
            if actual_frames != target_frames:
                if actual_frames < target_frames:
                    # Pad
                    pad_size = target_frames - actual_frames
                    features = F.pad(features, (0, pad_size))
                    vad_labels = F.pad(vad_labels, (0, 0, 0, pad_size))
                    osd_labels = F.pad(osd_labels, (0, pad_size))
                else:
                    # Truncate
                    features = features[:, :target_frames]
                    vad_labels = vad_labels[:target_frames]
                    osd_labels = osd_labels[:target_frames]
            
            # Verify dimensions are correct
            assert features.shape == (771, target_frames), f"Features shape: {features.shape}, expected: (771, {target_frames})"
            assert vad_labels.shape == (target_frames, self.num_speakers), f"VAD labels shape: {vad_labels.shape}, expected: ({target_frames}, {self.num_speakers})"
            assert osd_labels.shape == (target_frames,), f"OSD labels shape: {osd_labels.shape}, expected: ({target_frames},)"
            
            result = {
                'features': features,  # [num_features, time]
                'vad_labels': vad_labels,  # [time, num_speakers]
                'osd_labels': osd_labels,  # [time]
                'segment_id': idx  # For debugging
            }
            
            # Cache if enabled
            if self.feature_cache is not None:
                self.feature_cache[idx] = result
            
            return result
            
        except Exception as e:
            print(f"Error loading segment {idx}: {e}")
            # Return dummy data with correct dimensions
            return {
                'features': torch.zeros(771, self.frames_per_segment),
                'vad_labels': torch.zeros(self.frames_per_segment, self.num_speakers),
                'osd_labels': torch.zeros(self.frames_per_segment),
                'segment_id': idx
            }
    
    def _load_audio_segment(self, segment_info: Dict) -> List[torch.Tensor]:
        """Load multi-channel audio segment."""
        audio_path = segment_info['audio_path']
        start_time = segment_info['start_time']
        duration = segment_info['duration']
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Extract segment
        start_frame = int(start_time * self.sample_rate)
        segment_length = int(duration * self.sample_rate)
        end_frame = start_frame + segment_length
        
        if end_frame > waveform.size(-1):
            # Pad if segment goes beyond audio
            pad_size = end_frame - waveform.size(-1)
            waveform = F.pad(waveform, (0, pad_size))
        
        segment = waveform[:, start_frame:end_frame]
        
        # Convert to list of tensors (one per channel)
        return [segment[i] for i in range(segment.size(0))]


def collate_fn(batch):
    """Custom collate function for DataLoader with proper error handling."""
    try:
        # Filter out failed samples
        valid_batch = [item for item in batch if item['features'] is not None]
        
        if not valid_batch:
            # Return dummy batch if all samples failed
            batch_size = len(batch)
            return {
                'features': torch.zeros(batch_size, 771, 250),  # Default frames
                'vad_labels': torch.zeros(batch_size, 250, 4),  # 4 speakers
                'osd_labels': torch.zeros(batch_size, 250),
                'segment_ids': list(range(batch_size))
            }
        
        # Stack valid samples
        features = torch.stack([item['features'] for item in valid_batch])
        vad_labels = torch.stack([item['vad_labels'] for item in valid_batch])
        osd_labels = torch.stack([item['osd_labels'] for item in valid_batch])
        segment_ids = [item.get('segment_id', -1) for item in valid_batch]
        
        # Transpose features for model input: [batch, time, features] -> [batch, features, time]
        if features.dim() == 3 and features.size(1) != 771:
            features = features.transpose(1, 2)
        
        return {
            'features': features,  # [batch, num_features, time]
            'vad_labels': vad_labels,  # [batch, time, num_speakers]
            'osd_labels': osd_labels,  # [batch, time]
            'segment_ids': segment_ids
        }
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Return dummy batch
        batch_size = len(batch)
        return {
            'features': torch.zeros(batch_size, 771, 250),
            'vad_labels': torch.zeros(batch_size, 250, 4),
            'osd_labels': torch.zeros(batch_size, 250),
            'segment_ids': list(range(batch_size))
        }


def create_dataloaders(audio_dir: str, rttm_dir: str, batch_size: int = 16,
                      train_split: float = 0.8, num_workers: int = 4,
                      **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Create full dataset
    full_dataset = DiarizationDataset(audio_dir, rttm_dir, **dataset_kwargs)
    
    if len(full_dataset) == 0:
        raise ValueError("No valid audio-RTTM pairs found!")
    
    # Train/val split
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    audio_dir = "/path/to/audio/files"
    rttm_dir = "/path/to/rttm/files"
    
    try:
        dataset = DiarizationDataset(audio_dir, rttm_dir, max_segments=10)
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample features shape: {sample['features'].shape}")
            print(f"Sample VAD labels shape: {sample['vad_labels'].shape}")
            print(f"Sample OSD labels shape: {sample['osd_labels'].shape}")
            
            # Test dataloader
            train_loader, val_loader = create_dataloaders(
                audio_dir, rttm_dir, batch_size=4, max_segments=20
            )
            
            for batch in train_loader:
                print(f"Batch features shape: {batch['features'].shape}")
                print(f"Batch VAD labels shape: {batch['vad_labels'].shape}")
                print(f"Batch OSD labels shape: {batch['osd_labels'].shape}")
                break
        else:
            print("No valid samples found in dataset")
            
    except Exception as e:
        print(f"Error testing dataset: {e}")