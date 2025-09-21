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
                 min_speaker_duration=0.5,
                 pin_memory=True):
        """
        Args:
            split: 'dev' or 'test'
            segment_duration: Duration of each training segment in seconds
            hop_duration: Hop between segments in seconds
            sample_rate: Target sample rate
            n_mels: Number of mel features
            min_speaker_duration: Minimum duration for a speaker to be included
            pin_memory: Legacy parameter, now handled by DataLoader pin_memory setting
        """
        self.split = split
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.sample_rate = sample_rate
        self.n_mels = n_mels
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
        self.segments = self._extract_training_segments()
        
        print(f"✅ Prepared {len(self.segments)} segments from {len(self.dataset)} conversations")
    
    def _extract_training_segments(self) -> List[Dict]:
        """Extract training segments from conversations with sliding window."""
    def _extract_training_segments(self) -> List[Dict]:
        """Extract training segments from conversations with sliding window."""
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
                
                # Create global speaker mapping for this conversation (tous les speakers, pas de limite)
                unique_speakers = sorted(list(set(speakers)))
                global_speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
                
                # Create segments with sliding window
                start_time = 0.0
                while start_time + self.segment_duration <= conv_duration:
                    end_time = start_time + self.segment_duration
                    
                    # Extract audio segment
                    start_sample = int(start_time * self.sample_rate)
                    end_sample = int(end_time * self.sample_rate)
                    audio_segment = audio_array[start_sample:end_sample]
                    
                    # Create frame-wise labels for VAD, OSD, VCN
                    segment_labels = self._compute_frame_labels(
                        start_time, end_time, timestamps_start, timestamps_end, speakers,
                        global_speaker_to_idx
                    )
                    
                    # Keep ALL segments
                    segments.append({
                        'audio': audio_segment,
                        'vad_labels': segment_labels['vad'],      # [num_frames] - Voice Activity
                        'osd_labels': segment_labels['osd'],      # [num_frames] - Overlap Speech
                        'vcn_labels': segment_labels['vcn'],      # [num_frames] - Voice Change
                        'speaker_ids': segment_labels['speaker_ids'],
                        'conv_idx': conv_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'has_overlap': segment_labels['has_overlap'],
                        'has_voice_change': segment_labels['has_voice_change'],
                        'vad_frames': segment_labels['vad_frames'],
                        'osd_frames': segment_labels['osd_frames'],
                        'vcn_frames': segment_labels['vcn_frames']
                    })
                    
                    start_time += self.hop_duration
            
            except Exception as e:
                print(f"⚠️  Error processing conversation {conv_idx}: {e}")
                continue
        
        return segments
    
    def _compute_frame_labels(self, segment_start, segment_end, 
                              timestamps_start, timestamps_end, speakers, 
                              global_speaker_to_idx) -> Dict:
        """
        Compute frame-wise labels for VAD, OSD, and VCN.
        
        VAD: Voice Activity Detection (any speaker talking)
        OSD: Overlap Speech Detection (2+ speakers talking simultaneously) 
        VCN: Voice Change Detection (speaker change detection)
        """
        try:
            # Parameters for time discretization
            frame_duration = 0.02  # 20ms frames (matches 256 hop_length at 16kHz)
            num_frames = int(self.segment_duration / frame_duration)
            
            # Initialize frame-wise labels
            vad_labels = np.zeros(num_frames, dtype=np.float32)  # Any speech
            osd_labels = np.zeros(num_frames, dtype=np.float32)  # Overlap speech
            vcn_labels = np.zeros(num_frames, dtype=np.float32)  # Voice change
            
            # Protection contre les conversations sans speakers
            if not global_speaker_to_idx:
                return {
                    'vad': vad_labels,
                    'osd': osd_labels,
                    'vcn': vcn_labels,
                    'speaker_ids': [],
                    'has_speech': False,
                    'has_overlap': False,
                    'has_voice_change': False,
                    'vad_frames': 0,
                    'osd_frames': 0,
                    'vcn_frames': 0,
                    'total_frames': num_frames
                }
                
            # Créer une matrice pour tracker l'activité de TOUS les speakers (pas de limitation)
            num_speakers = len(global_speaker_to_idx)
            speaker_activity = np.zeros((num_frames, num_speakers), dtype=np.float32)
            
            has_speech = False
            has_overlap = False
            has_voice_change = False
            active_speakers_in_segment = set()
            
            # 1. Remplir la matrice d'activité des speakers
            for start, end, speaker in zip(timestamps_start, timestamps_end, speakers):
                # Protection contre les valeurs None ou invalides
                if start is None or end is None or speaker is None:
                    continue
                    
                # Conversion sécurisée en float
                try:
                    start = float(start)
                    end = float(end)
                except (ValueError, TypeError):
                    continue
                    
                # Check if this annotation overlaps with our segment
                if end <= segment_start or start >= segment_end:
                    continue
                    
                # Clip to segment boundaries
                clipped_start = max(start, segment_start)
                clipped_end = min(end, segment_end)
                
                # Seuil minimal très bas pour capturer même les très courtes interventions
                if clipped_end - clipped_start < 0.01:  # 10ms minimum
                    continue
                    
                # Convert to frame indices avec plus de précision
                try:
                    start_frame = int(np.round((clipped_start - segment_start) / frame_duration))
                    end_frame = int(np.round((clipped_end - segment_start) / frame_duration))
                except (ValueError, TypeError) as e:
                    print(f"    ⚠️  Frame conversion error: {e}")
                    continue
                
                start_frame = max(0, min(start_frame, num_frames - 1))
                end_frame = max(start_frame + 1, min(end_frame, num_frames))
                
                # Set speaker activity
                if speaker in global_speaker_to_idx:
                    speaker_idx = global_speaker_to_idx[speaker]
                    # Plus besoin de vérifier max_speakers - on prend tous les speakers
                    speaker_activity[start_frame:end_frame, speaker_idx] = 1.0
                    has_speech = True
                    active_speakers_in_segment.add(speaker)
            
            # 2. Générer VAD - n'importe qui parle
            vad_labels = np.sum(speaker_activity, axis=1)  # Somme sur tous les speakers
            vad_labels = (vad_labels > 0).astype(np.float32)  # Binaire : 0 ou 1
            
            # 3. Générer OSD - au moins 2 speakers parlent simultanément
            osd_labels = np.sum(speaker_activity, axis=1)  # Somme sur tous les speakers
            osd_labels = (osd_labels >= 2).astype(np.float32)  # Binaire : 1 si >= 2 speakers
            has_overlap = np.any(osd_labels > 0)
            
            # 4. Générer VCN - détection des changements de speaker
            current_speakers = set()
            for frame_idx in range(num_frames):
                # Trouver les speakers actifs dans cette frame
                frame_speakers = set()
                for speaker_idx in range(num_speakers):
                    if speaker_activity[frame_idx, speaker_idx] > 0:
                        speaker_name = list(global_speaker_to_idx.keys())[speaker_idx]
                        frame_speakers.add(speaker_name)
                
                # Détecter un changement de speaker
                if frame_idx == 0:
                    # Première frame - établir la baseline
                    current_speakers = frame_speakers.copy()
                    if len(frame_speakers) > 0:
                        vcn_labels[frame_idx] = 1.0  # Début de parole = changement
                        has_voice_change = True
                else:
                    # Détecter changement par rapport à la frame précédente
                    if frame_speakers != current_speakers and len(frame_speakers) > 0:
                        vcn_labels[frame_idx] = 1.0
                        has_voice_change = True
                    current_speakers = frame_speakers.copy()
            
            # Debug: compter les frames pour chaque métrique
            vad_frames = int(np.sum(vad_labels > 0))
            osd_frames = int(np.sum(osd_labels > 0))
            vcn_frames = int(np.sum(vcn_labels > 0))
            
            return {
                'vad': vad_labels,           # [num_frames] - Voice Activity Detection
                'osd': osd_labels,           # [num_frames] - Overlap Speech Detection  
                'vcn': vcn_labels,           # [num_frames] - Voice Change Detection
                'speaker_ids': list(active_speakers_in_segment),
                'has_speech': has_speech,
                'has_overlap': has_overlap,
                'has_voice_change': has_voice_change,
                'vad_frames': vad_frames,
                'osd_frames': osd_frames,
                'vcn_frames': vcn_frames,
                'total_frames': num_frames
            }
            
        except Exception as e:
            print(f"    ⚠️  Error in _compute_frame_labels: {e}")
            import traceback
            traceback.print_exc()
            # En cas d'erreur, retourner des labels vides mais valides
            frame_duration = 0.02
            error_num_frames = int(self.segment_duration / frame_duration)
            return {
                'vad': np.zeros(error_num_frames, dtype=np.float32),
                'osd': np.zeros(error_num_frames, dtype=np.float32),
                'vcn': np.zeros(error_num_frames, dtype=np.float32),
                'speaker_ids': [],
                'has_speech': False,
                'has_overlap': False,
                'has_voice_change': False,
                'vad_frames': 0,
                'osd_frames': 0,
                'vcn_frames': 0,
                'total_frames': error_num_frames
            }
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get a training sample with frame-wise VAD, OSD, and VCN labels."""
        segment = self.segments[idx]
        
        # Convert audio to tensor
        audio = torch.from_numpy(segment['audio']).float()
        
        # Extract mel features
        with torch.no_grad():
            mel_features = self.mel_transform(audio)
            log_mel = torch.log(mel_features + 1e-8)
        
        # Convert labels to tensors - now all are 1D frame-wise
        vad_labels = torch.from_numpy(segment['vad_labels']).float()  # [num_frames]
        osd_labels = torch.from_numpy(segment['osd_labels']).float()  # [num_frames]
        vcn_labels = torch.from_numpy(segment['vcn_labels']).float()  # [num_frames]
        
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
        
        # Resize labels if needed - all labels are now 1D
        if vad_labels.shape[0] != target_frames:
            vad_labels = F.interpolate(
                vad_labels.unsqueeze(0).unsqueeze(0),
                size=target_frames,
                mode='linear',
                align_corners=False
            ).squeeze()
            
            osd_labels = F.interpolate(
                osd_labels.unsqueeze(0).unsqueeze(0),
                size=target_frames,
                mode='linear',
                align_corners=False
            ).squeeze()
            
            vcn_labels = F.interpolate(
                vcn_labels.unsqueeze(0).unsqueeze(0),
                size=target_frames,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        return {
            'features': log_mel,        # [n_mels, time] - Mel spectrogram
            'vad_labels': vad_labels,   # [time] - Voice Activity Detection
            'osd_labels': osd_labels,   # [time] - Overlap Speech Detection  
            'vcn_labels': vcn_labels,   # [time] - Voice Change Detection
            'audio': audio,             # Original audio for reference
            'conv_idx': segment['conv_idx'],
            'start_time': segment['start_time'],
            # Debug info
            'has_overlap': segment.get('has_overlap', False),
            'has_voice_change': segment.get('has_voice_change', False),
            'vad_frames': segment.get('vad_frames', 0),
            'osd_frames': segment.get('osd_frames', 0),
            'vcn_frames': segment.get('vcn_frames', 0)
        }


def collate_fn(batch):
    """Collate function for DataLoader with VAD, OSD, and VCN labels."""
    features = torch.stack([item['features'] for item in batch])
    vad_labels = torch.stack([item['vad_labels'] for item in batch])  # [batch, time]
    osd_labels = torch.stack([item['osd_labels'] for item in batch])  # [batch, time]
    vcn_labels = torch.stack([item['vcn_labels'] for item in batch])  # [batch, time]
    
    return {
        'features': features,     # [batch, n_mels, time]
        'vad_labels': vad_labels, # [batch, time] - Voice Activity Detection
        'osd_labels': osd_labels, # [batch, time] - Overlap Speech Detection
        'vcn_labels': vcn_labels, # [batch, time] - Voice Change Detection
    }


def create_voxconverse_dataloaders(batch_size=32, 
                                   num_workers=4,
                                   segment_duration=4.0,
                                   validation_split=0.1,
                                   pin_memory=True,
                                   persistent_workers=True,
                                   prefetch_factor=2,
                                   worker_init_fn=None,
                                   debug_overlap_stats=True,
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
        debug_overlap_stats: If True, print overlap statistics
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
    
    # Debug overlap statistics
    if debug_overlap_stats:
        _print_overlap_statistics(full_dataset)
    
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


def _print_overlap_statistics(dataset):
    """Print statistics about VAD, OSD, and VCN detection in the dataset."""
    total_segments = len(dataset.segments)
    
    # Compter les segments avec chaque type d'activité
    vad_segments = sum(1 for seg in dataset.segments if seg.get('vad_frames', 0) > 0)
    osd_segments = sum(1 for seg in dataset.segments if seg.get('osd_frames', 0) > 0)
    vcn_segments = sum(1 for seg in dataset.segments if seg.get('vcn_frames', 0) > 0)
    
    # Compter les frames totales
    total_vad_frames = sum(seg.get('vad_frames', 0) for seg in dataset.segments)
    total_osd_frames = sum(seg.get('osd_frames', 0) for seg in dataset.segments)
    total_vcn_frames = sum(seg.get('vcn_frames', 0) for seg in dataset.segments)
    total_frames = sum(seg.get('total_frames', 0) for seg in dataset.segments)
    
    print(f"\n🔍 Frame-wise Detection Statistics:")
    print(f"  Total segments: {total_segments}")
    print(f"  Total frames: {total_frames}")
    print(f"")
    print(f"  📢 VAD (Voice Activity Detection):")
    print(f"    Segments with speech: {vad_segments}/{total_segments} ({vad_segments/total_segments*100:.1f}%)")
    if total_frames > 0:
        print(f"    VAD frames: {total_vad_frames} ({total_vad_frames/total_frames*100:.2f}% of all frames)")
    else:
        print(f"    VAD frames: {total_vad_frames} (0.00% - no frames to analyze)")
    print(f"")
    print(f"  🗣️  OSD (Overlap Speech Detection):")
    print(f"    Segments with overlap: {osd_segments}/{total_segments} ({osd_segments/total_segments*100:.1f}%)")
    if total_frames > 0:
        print(f"    OSD frames: {total_osd_frames} ({total_osd_frames/total_frames*100:.2f}% of all frames)")
    else:
        print(f"    OSD frames: {total_osd_frames} (0.00% - no frames to analyze)")
    print(f"")
    print(f"  🔄 VCN (Voice Change Detection):")
    print(f"    Segments with voice changes: {vcn_segments}/{total_segments} ({vcn_segments/total_segments*100:.1f}%)")
    if total_frames > 0:
        print(f"    VCN frames: {total_vcn_frames} ({total_vcn_frames/total_frames*100:.2f}% of all frames)")
    else:
        print(f"    VCN frames: {total_vcn_frames} (0.00% - no frames to analyze)")
    
    # Échantillons détaillés
    print(f"\n📋 Sample segments with different activities:")
    
    # Segments avec overlap
    osd_samples = [seg for seg in dataset.segments if seg.get('osd_frames', 0) > 0][:3]
    if osd_samples:
        print(f"  OSD samples:")
        for i, seg in enumerate(osd_samples):
            print(f"    {i+1}. Conv {seg['conv_idx']}, t={seg['start_time']:.1f}-{seg['end_time']:.1f}s, "
                  f"OSD={seg.get('osd_frames', 0)} frames")
    
    # Segments avec changements de voix
    vcn_samples = [seg for seg in dataset.segments if seg.get('vcn_frames', 0) > 0][:3]
    if vcn_samples:
        print(f"  VCN samples:")
        for i, seg in enumerate(vcn_samples):
            print(f"    {i+1}. Conv {seg['conv_idx']}, t={seg['start_time']:.1f}-{seg['end_time']:.1f}s, "
                  f"VCN={seg.get('vcn_frames', 0)} frames")
    print()

