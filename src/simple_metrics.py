import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class DiarizationMetrics:
    """Simplified metrics for speaker diarization evaluation."""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def compute_frame_metrics(self, predictions, targets, threshold=None):
        """
        Compute frame-level metrics.
        
        Args:
            predictions: [batch, time, speakers] or [batch, time] 
            targets: [batch, time, speakers] or [batch, time]
            threshold: decision threshold
            
        Returns:
            dict of metrics
        """
        if threshold is None:
            threshold = self.threshold
            
        # Convert to binary predictions
        pred_binary = (predictions > threshold).float()
        targets = targets.float()
        
        # Flatten for sklearn metrics
        pred_flat = pred_binary.cpu().numpy().flatten()
        target_flat = targets.cpu().numpy().flatten()
        
        # Compute precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_flat, pred_flat, average='binary', zero_division=0
        )
        
        # Compute accuracy
        accuracy = (pred_flat == target_flat).mean()
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall), 
            'f1_score': float(f1)
        }
    
    def compute_der(self, vad_pred, vad_target, threshold=None):
        """
        Compute simplified Diarization Error Rate (DER).
        
        DER = (False Alarm + Miss + Speaker Error) / Total Reference Speech
        
        Args:
            vad_pred: [batch, time, speakers] VAD predictions
            vad_target: [batch, time, speakers] VAD targets
            
        Returns:
            float DER score
        """
        if threshold is None:
            threshold = self.threshold
            
        # Convert to binary
        pred_binary = (vad_pred > threshold).float()
        target_binary = vad_target.float()
        
        # Flatten batch dimension
        pred_flat = pred_binary.view(-1, pred_binary.shape[-1])  # [batch*time, speakers]
        target_flat = target_binary.view(-1, target_binary.shape[-1])
        
        # Count active speakers per frame
        pred_active = pred_flat.sum(dim=1)  # [batch*time]
        target_active = target_flat.sum(dim=1)
        
        # Reference speech time (frames with any speaker active)
        reference_speech = (target_active > 0).float().sum()
        
        if reference_speech == 0:
            return 0.0  # No speech to evaluate
        
        # Miss: frames where target has speech but prediction doesn't
        miss = ((target_active > 0) & (pred_active == 0)).float().sum()
        
        # False alarm: frames where prediction has speech but target doesn't  
        false_alarm = ((target_active == 0) & (pred_active > 0)).float().sum()
        
        # Speaker error: frames where both have speech but different number of speakers
        speaker_error = ((target_active > 0) & (pred_active > 0) & 
                        (target_active != pred_active)).float().sum()
        
        # DER calculation
        der = (miss + false_alarm + speaker_error) / reference_speech
        return float(der)
    
    def compute_overlap_metrics(self, osd_pred, osd_target, threshold=None):
        """
        Compute overlap detection metrics.
        
        Args:
            osd_pred: [batch, time] overlap predictions
            osd_target: [batch, time] overlap targets
            
        Returns:
            dict of overlap metrics
        """
        return self.compute_frame_metrics(osd_pred, osd_target, threshold)
    
    def compute_all_metrics(self, vad_pred, osd_pred, vad_target, osd_target):
        """
        Compute comprehensive diarization metrics.
        
        Args:
            vad_pred: [batch, time, speakers] VAD predictions
            osd_pred: [batch, time] OSD predictions
            vad_target: [batch, time, speakers] VAD targets  
            osd_target: [batch, time] OSD targets
            
        Returns:
            dict of all metrics
        """
        # VAD metrics
        vad_metrics = self.compute_frame_metrics(vad_pred, vad_target)
        vad_metrics = {f'vad_{k}': v for k, v in vad_metrics.items()}
        
        # OSD metrics  
        osd_metrics = self.compute_overlap_metrics(osd_pred, osd_target)
        osd_metrics = {f'osd_{k}': v for k, v in osd_metrics.items()}
        
        # DER
        der = self.compute_der(vad_pred, vad_target)
        
        # Combined metrics
        overall_f1 = (vad_metrics['vad_f1_score'] + osd_metrics['osd_f1_score']) / 2
        
        return {
            **vad_metrics,
            **osd_metrics,
            'der': der,
            'overall_f1': overall_f1
        }


class CUDAOptimizedMetrics:
    """CUDA-optimized version for faster computation during training."""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def compute_metrics_cuda(self, vad_logits, osd_logits, vad_target, osd_target):
        """
        Fast CUDA-based metric computation for training.
        Expects logits as input (for AMP compatibility).
        
        Returns essential metrics only to avoid CPU transfers.
        """
        # Convert logits to probabilities
        vad_pred = torch.sigmoid(vad_logits)
        osd_pred = torch.sigmoid(osd_logits)
        
        # Convert to binary predictions
        vad_binary = (vad_pred > self.threshold).float()
        osd_binary = (osd_pred > self.threshold).float()
        
        # VAD accuracy
        vad_acc = (vad_binary == vad_target.float()).float().mean()
        
        # OSD accuracy
        osd_acc = (osd_binary == osd_target.float()).float().mean()
        
        # Simple F1 approximation (faster than full computation)
        vad_flat = vad_binary.view(-1)
        vad_target_flat = vad_target.view(-1).float()
        
        tp = ((vad_flat == 1) & (vad_target_flat == 1)).float().sum()
        fp = ((vad_flat == 1) & (vad_target_flat == 0)).float().sum()
        fn = ((vad_flat == 0) & (vad_target_flat == 1)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'vad_accuracy': vad_acc.item(),
            'osd_accuracy': osd_acc.item(),
            'f1_score': f1.item()
        }


def create_metrics(config=None, cuda_optimized=True):
    """Factory function to create metrics."""
    threshold = config.get('threshold', 0.5) if config else 0.5
    
    if cuda_optimized:
        return CUDAOptimizedMetrics(threshold=threshold)
    else:
        return DiarizationMetrics(threshold=threshold)