import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class DiarizationMetrics:
    """Comprehensive metrics for speaker diarization evaluation."""
    
    def __init__(self, num_speakers: int = 4, frame_duration: float = 0.02, 
                 collar: float = 0.25, threshold: float = 0.5):
        self.num_speakers = num_speakers
        self.frame_duration = frame_duration
        self.collar = collar  # Collar in seconds
        self.threshold = threshold
        self.collar_frames = int(collar / frame_duration)
    
    def compute_der(self, vad_pred: torch.Tensor, vad_target: torch.Tensor,
                   apply_collar: bool = True, skip_overlap: bool = False) -> float:
        """
        Compute Diarization Error Rate (DER).
        
        Args:
            vad_pred: [batch, time, speakers] - predicted VAD probabilities
            vad_target: [batch, time, speakers] - target VAD labels
            apply_collar: Whether to apply collar around speech boundaries
            skip_overlap: Whether to skip overlapped speech regions
            
        Returns:
            DER as percentage
        """
        vad_pred_binary = (vad_pred > self.threshold).float()
        batch_size = vad_pred.size(0)
        
        total_speech_time = 0
        total_error_time = 0
        
        for b in range(batch_size):
            pred_b = vad_pred_binary[b]  # [time, speakers]
            target_b = vad_target[b]     # [time, speakers]
            
            # Apply collar if requested
            if apply_collar and self.collar_frames > 0:
                mask = self._get_collar_mask(target_b)
                pred_b = pred_b[mask]
                target_b = target_b[mask]
            
            # Skip overlapped regions if requested
            if skip_overlap:
                overlap_mask = target_b.sum(dim=1) <= 1
                pred_b = pred_b[overlap_mask]
                target_b = target_b[overlap_mask]
            
            if pred_b.numel() == 0:
                continue
            
            # Compute error components
            speech_frames = (target_b.sum(dim=1) > 0).sum().item()
            
            # False Alarm: prediction where no target speech
            fa_frames = ((pred_b.sum(dim=1) > 0) & (target_b.sum(dim=1) == 0)).sum().item()
            
            # Miss: target speech where no prediction
            miss_frames = ((pred_b.sum(dim=1) == 0) & (target_b.sum(dim=1) > 0)).sum().item()
            
            # Speaker Error: prediction and target both have speech but wrong assignment
            both_active = (pred_b.sum(dim=1) > 0) & (target_b.sum(dim=1) > 0)
            if both_active.any():
                # Check speaker assignment accuracy
                correct_assignment = (pred_b[both_active] * target_b[both_active]).sum(dim=1) > 0
                speaker_error_frames = (~correct_assignment).sum().item()
            else:
                speaker_error_frames = 0
            
            total_speech_time += speech_frames
            total_error_time += fa_frames + miss_frames + speaker_error_frames
        
        if total_speech_time == 0:
            return 0.0
        
        der = 100 * total_error_time / total_speech_time
        return der
    
    def _get_collar_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Get mask for collar application around speech boundaries."""
        num_frames = target.size(0)
        mask = torch.ones(num_frames, dtype=torch.bool)
        
        # Find speech activity
        speech_activity = target.sum(dim=1) > 0
        
        # Find boundaries
        if speech_activity.sum() > 0:
            boundaries = torch.diff(speech_activity.float())
            boundary_indices = torch.nonzero(boundaries, as_tuple=True)[0]
            
            # Apply collar around boundaries
            for idx in boundary_indices:
                start = max(0, idx - self.collar_frames)
                end = min(num_frames, idx + self.collar_frames + 1)
                mask[start:end] = False
        
        return mask
    
    def compute_frame_metrics(self, vad_pred: torch.Tensor, vad_target: torch.Tensor) -> Dict[str, float]:
        """Compute frame-level classification metrics."""
        vad_pred_binary = (vad_pred > self.threshold).float()
        
        # Flatten for sklearn metrics
        pred_flat = vad_pred_binary.view(-1).cpu().numpy()
        target_flat = vad_target.view(-1).cpu().numpy()
        
        precision = precision_score(target_flat, pred_flat, average='binary', zero_division=0)
        recall = recall_score(target_flat, pred_flat, average='binary', zero_division=0)
        f1 = f1_score(target_flat, pred_flat, average='binary', zero_division=0)
        accuracy = accuracy_score(target_flat, pred_flat)
        
        return {
            'frame_precision': precision,
            'frame_recall': recall,
            'frame_f1': f1,
            'frame_accuracy': accuracy
        }
    
    def compute_speaker_metrics(self, vad_pred: torch.Tensor, vad_target: torch.Tensor) -> Dict[str, float]:
        """Compute per-speaker metrics."""
        vad_pred_binary = (vad_pred > self.threshold).float()
        
        speaker_metrics = {}
        
        for spk in range(self.num_speakers):
            pred_spk = vad_pred_binary[:, :, spk].view(-1).cpu().numpy()
            target_spk = vad_target[:, :, spk].view(-1).cpu().numpy()
            
            if target_spk.sum() > 0:  # Only compute if speaker has activity
                precision = precision_score(target_spk, pred_spk, zero_division=0)
                recall = recall_score(target_spk, pred_spk, zero_division=0)
                f1 = f1_score(target_spk, pred_spk, zero_division=0)
                
                speaker_metrics[f'speaker_{spk}_precision'] = precision
                speaker_metrics[f'speaker_{spk}_recall'] = recall
                speaker_metrics[f'speaker_{spk}_f1'] = f1
        
        return speaker_metrics
    
    def compute_osd_metrics(self, osd_pred: torch.Tensor, osd_target: torch.Tensor) -> Dict[str, float]:
        """Compute Overlapped Speech Detection metrics."""
        osd_pred_binary = (osd_pred > self.threshold).float()
        
        pred_flat = osd_pred_binary.view(-1).cpu().numpy()
        target_flat = osd_target.view(-1).cpu().numpy()
        
        precision = precision_score(target_flat, pred_flat, zero_division=0)
        recall = recall_score(target_flat, pred_flat, zero_division=0)
        f1 = f1_score(target_flat, pred_flat, zero_division=0)
        accuracy = accuracy_score(target_flat, pred_flat)
        
        return {
            'osd_precision': precision,
            'osd_recall': recall,
            'osd_f1': f1,
            'osd_accuracy': accuracy
        }
    
    def compute_jaccard_index(self, vad_pred: torch.Tensor, vad_target: torch.Tensor) -> float:
        """Compute Jaccard Index (IoU) for overall speech detection."""
        vad_pred_binary = (vad_pred > self.threshold).float()
        
        # Any speaker active
        pred_activity = (vad_pred_binary.sum(dim=2) > 0).float()
        target_activity = (vad_target.sum(dim=2) > 0).float()
        
        intersection = (pred_activity * target_activity).sum()
        union = ((pred_activity + target_activity) > 0).float().sum()
        
        if union == 0:
            return 1.0
        
        jaccard = intersection / union
        return jaccard.item()
    
    def compute_coverage(self, vad_pred: torch.Tensor, vad_target: torch.Tensor) -> Tuple[float, float]:
        """Compute speech coverage metrics."""
        vad_pred_binary = (vad_pred > self.threshold).float()
        
        # Total speech time in target and prediction
        target_speech_time = (vad_target.sum(dim=2) > 0).float().sum()
        pred_speech_time = (vad_pred_binary.sum(dim=2) > 0).float().sum()
        
        if target_speech_time == 0:
            return 0.0, 0.0
        
        coverage = pred_speech_time / target_speech_time
        over_coverage = max(0, coverage.item() - 1.0)
        
        return coverage.item(), over_coverage
    
    def compute_purity_coverage(self, vad_pred: torch.Tensor, vad_target: torch.Tensor) -> Tuple[float, float]:
        """Compute purity and coverage for speaker diarization."""
        vad_pred_binary = (vad_pred > self.threshold).float()
        batch_size, time_steps, num_speakers = vad_pred.shape
        
        total_purity = 0
        total_coverage = 0
        total_speakers = 0
        
        for b in range(batch_size):
            for spk in range(num_speakers):
                target_spk = vad_target[b, :, spk]
                
                if target_spk.sum() == 0:
                    continue
                
                total_speakers += 1
                
                # Find best matching predicted speaker
                best_overlap = 0
                best_spk_pred = None
                
                for pred_spk in range(num_speakers):
                    pred_activity = vad_pred_binary[b, :, pred_spk]
                    overlap = (target_spk * pred_activity).sum()
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_spk_pred = pred_spk
                
                if best_spk_pred is not None:
                    pred_activity = vad_pred_binary[b, :, best_spk_pred]
                    
                    # Coverage: how much of target is covered by prediction
                    coverage = best_overlap / target_spk.sum()
                    
                    # Purity: how much of prediction corresponds to this target
                    if pred_activity.sum() > 0:
                        purity = best_overlap / pred_activity.sum()
                    else:
                        purity = 0
                    
                    total_coverage += coverage
                    total_purity += purity
        
        if total_speakers == 0:
            return 0.0, 0.0
        
        avg_purity = total_purity / total_speakers
        avg_coverage = total_coverage / total_speakers
        
        return avg_purity.item(), avg_coverage.item()
    
    def compute_metrics(self, vad_pred: torch.Tensor, osd_pred: torch.Tensor,
                       vad_target: torch.Tensor, osd_target: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}
        
        # DER (main metric)
        metrics['der'] = self.compute_der(vad_pred, vad_target)
        
        # Frame-level metrics
        frame_metrics = self.compute_frame_metrics(vad_pred, vad_target)
        metrics.update(frame_metrics)
        
        # Speaker-specific metrics
        speaker_metrics = self.compute_speaker_metrics(vad_pred, vad_target)
        metrics.update(speaker_metrics)
        
        # OSD metrics
        osd_metrics = self.compute_osd_metrics(osd_pred, osd_target)
        metrics.update(osd_metrics)
        
        # Additional metrics
        metrics['jaccard_index'] = self.compute_jaccard_index(vad_pred, vad_target)
        coverage, over_coverage = self.compute_coverage(vad_pred, vad_target)
        metrics['coverage'] = coverage
        metrics['over_coverage'] = over_coverage
        
        purity, coverage_pc = self.compute_purity_coverage(vad_pred, vad_target)
        metrics['purity'] = purity
        metrics['coverage_pc'] = coverage_pc
        
        # F1 score (balanced measure)
        if 'frame_precision' in metrics and 'frame_recall' in metrics:
            p, r = metrics['frame_precision'], metrics['frame_recall']
            metrics['f1_score'] = 2 * p * r / (p + r + 1e-8)
        
        return metrics


def compute_segment_level_metrics(segments_pred: list, segments_target: list, 
                                total_duration: float, collar: float = 0.25) -> Dict[str, float]:
    """
    Compute segment-level diarization metrics from segment lists.
    
    Args:
        segments_pred: List of (start, end, speaker_id) tuples for predictions
        segments_target: List of (start, end, speaker_id) tuples for targets
        total_duration: Total audio duration in seconds
        collar: Collar in seconds around segment boundaries
        
    Returns:
        Dictionary of metrics
    """
    # Convert segments to frame-level labels
    frame_rate = 100  # 10ms frames
    total_frames = int(total_duration * frame_rate)
    
    def segments_to_frames(segments, num_frames):
        frames = np.zeros((num_frames, max(4, max([s[2] for s in segments] + [0]) + 1)))
        for start, end, speaker in segments:
            start_frame = int(start * frame_rate)
            end_frame = min(int(end * frame_rate), num_frames)
            if end_frame > start_frame:
                frames[start_frame:end_frame, speaker] = 1
        return frames
    
    pred_frames = segments_to_frames(segments_pred, total_frames)
    target_frames = segments_to_frames(segments_target, total_frames)
    
    # Convert to tensors
    pred_tensor = torch.tensor(pred_frames[:, :4], dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(target_frames[:, :4], dtype=torch.float32).unsqueeze(0)
    
    # Compute metrics
    metrics_computer = DiarizationMetrics(frame_duration=0.01, collar=collar)
    metrics = {}
    metrics['der'] = metrics_computer.compute_der(pred_tensor, target_tensor, apply_collar=True)
    
    frame_metrics = metrics_computer.compute_frame_metrics(pred_tensor, target_tensor)
    metrics.update(frame_metrics)
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    batch_size = 4
    time_steps = 1000
    num_speakers = 4
    
    # Generate dummy data
    vad_pred = torch.sigmoid(torch.randn(batch_size, time_steps, num_speakers))
    osd_pred = torch.sigmoid(torch.randn(batch_size, time_steps))
    vad_target = torch.randint(0, 2, (batch_size, time_steps, num_speakers)).float()
    osd_target = torch.randint(0, 2, (batch_size, time_steps)).float()
    
    # Initialize metrics
    metrics_computer = DiarizationMetrics(num_speakers=num_speakers)
    
    # Compute metrics
    metrics = metrics_computer.compute_metrics(vad_pred, osd_pred, vad_target, osd_target)
    
    print("Computed metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    print(f"\nMain DER: {metrics['der']:.2f}%")