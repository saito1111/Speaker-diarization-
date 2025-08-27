import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from typing import Tuple, Optional


class PermutationInvariantLoss(nn.Module):
    """
    Permutation Invariant Training (PIT) Loss for speaker diarization.
    Finds the best permutation of predicted speakers to match ground truth.
    """
    
    def __init__(self, base_loss=None, num_speakers=4):
        super(PermutationInvariantLoss, self).__init__()
        self.base_loss = base_loss or nn.BCELoss(reduction='none')
        self.num_speakers = num_speakers
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            predictions: [batch, time, num_speakers]
            targets: [batch, time, num_speakers]
            
        Returns:
            loss: Scalar loss
            best_perms: Best permutation indices for each batch item
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        # Generate all possible permutations
        perms = list(itertools.permutations(range(self.num_speakers)))
        
        losses = []
        for perm in perms:
            # Apply permutation to predictions
            perm_pred = predictions[:, :, perm]
            
            # Compute loss for this permutation
            loss = self.base_loss(perm_pred, targets)
            loss = loss.mean(dim=(1, 2))  # Average over time and speakers
            losses.append(loss)
        
        # Stack losses: [num_perms, batch]
        losses = torch.stack(losses, dim=0)
        
        # Find best permutation for each batch item
        best_perm_indices = losses.argmin(dim=0)
        best_losses = losses[best_perm_indices, torch.arange(batch_size)]
        
        return best_losses.mean(), best_perm_indices


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for imbalanced speaker activity."""
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, time, num_speakers] or [batch, time]
            targets: Same shape as predictions
        """
        # Compute positive weights if not provided
        if self.pos_weight is None:
            pos_ratio = targets.mean()
            neg_ratio = 1 - pos_ratio
            pos_weight = neg_ratio / (pos_ratio + 1e-8)
        else:
            pos_weight = self.pos_weight.to(predictions.device)
        
        # Compute weighted BCE
        loss = F.binary_cross_entropy(
            predictions, targets, 
            weight=None, reduction='none'
        )
        
        # Apply positive weighting
        weights = targets * pos_weight + (1 - targets) * 1.0
        loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in speaker activity detection."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, time, num_speakers] or [batch, time]
            targets: Same shape as predictions
        """
        # Compute standard BCE
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Compute focal weight
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine weights
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss to encourage smooth speaker activity over time."""
    
    def __init__(self, weight: float = 0.1):
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, time, num_speakers]
        """
        # Compute temporal gradients
        temporal_grad = predictions[:, 1:] - predictions[:, :-1]
        
        # L2 penalty on temporal changes
        consistency_loss = (temporal_grad ** 2).mean()
        
        return self.weight * consistency_loss


class MultiTaskDiarizationLoss(nn.Module):
    """
    Multi-task loss combining VAD, OSD, and additional regularization terms.
    """
    
    def __init__(self, 
                 vad_loss_weight: float = 1.0,
                 osd_loss_weight: float = 1.0,
                 consistency_weight: float = 0.1,
                 use_pit: bool = True,
                 use_focal: bool = False,
                 focal_gamma: float = 2.0,
                 num_speakers: int = 4):
        super(MultiTaskDiarizationLoss, self).__init__()
        
        self.vad_loss_weight = vad_loss_weight
        self.osd_loss_weight = osd_loss_weight
        self.consistency_weight = consistency_weight
        self.use_pit = use_pit
        self.num_speakers = num_speakers
        
        # VAD Loss
        if use_pit:
            base_loss = FocalLoss(gamma=focal_gamma) if use_focal else nn.BCELoss()
            self.vad_loss = PermutationInvariantLoss(base_loss, num_speakers)
        else:
            self.vad_loss = FocalLoss(gamma=focal_gamma) if use_focal else WeightedBCELoss()
        
        # OSD Loss
        self.osd_loss = FocalLoss(gamma=focal_gamma) if use_focal else WeightedBCELoss()
        
        # Temporal consistency
        if consistency_weight > 0:
            self.consistency_loss = TemporalConsistencyLoss(consistency_weight)
        else:
            self.consistency_loss = None
    
    def forward(self, vad_pred: torch.Tensor, osd_pred: torch.Tensor,
                vad_target: torch.Tensor, osd_target: torch.Tensor) -> dict:
        """
        Args:
            vad_pred: [batch, time, num_speakers]
            osd_pred: [batch, time]
            vad_target: [batch, time, num_speakers]
            osd_target: [batch, time]
            
        Returns:
            Dictionary with total loss and component losses
        """
        losses = {}
        
        # VAD Loss
        if self.use_pit:
            vad_loss, best_perms = self.vad_loss(vad_pred, vad_target)
            losses['vad_perms'] = best_perms
        else:
            vad_loss = self.vad_loss(vad_pred, vad_target)
        
        losses['vad_loss'] = vad_loss
        
        # OSD Loss
        osd_loss = self.osd_loss(osd_pred, osd_target)
        losses['osd_loss'] = osd_loss
        
        # Temporal consistency
        if self.consistency_loss is not None:
            consistency_loss = self.consistency_loss(vad_pred)
            losses['consistency_loss'] = consistency_loss
        else:
            consistency_loss = 0
        
        # Total loss
        total_loss = (self.vad_loss_weight * vad_loss + 
                     self.osd_loss_weight * osd_loss + 
                     consistency_loss)
        
        losses['total_loss'] = total_loss
        
        return losses


class DERLoss(nn.Module):
    """
    Differentiable approximation to Diarization Error Rate (DER) for end-to-end training.
    """
    
    def __init__(self, collar: float = 0.25, skip_overlap: bool = False):
        super(DERLoss, self).__init__()
        self.collar = collar
        self.skip_overlap = skip_overlap
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                frame_duration: float = 0.02) -> torch.Tensor:
        """
        Args:
            predictions: [batch, time, num_speakers]
            targets: [batch, time, num_speakers]
            frame_duration: Duration of each frame in seconds
        """
        batch_size, num_frames, num_speakers = predictions.shape
        collar_frames = int(self.collar / frame_duration)
        
        total_der = 0
        
        for b in range(batch_size):
            pred_b = predictions[b]  # [time, speakers]
            target_b = targets[b]   # [time, speakers]
            
            # Apply collar (ignore frames near boundaries)
            if collar_frames > 0:
                # Find speech boundaries
                speech_mask = target_b.sum(dim=1) > 0
                boundaries = torch.diff(speech_mask.float())
                boundary_indices = torch.nonzero(boundaries, as_tuple=True)[0]
                
                # Create collar mask
                collar_mask = torch.ones(num_frames, dtype=torch.bool)
                for idx in boundary_indices:
                    start = max(0, idx - collar_frames)
                    end = min(num_frames, idx + collar_frames + 1)
                    collar_mask[start:end] = False
                
                pred_b = pred_b[collar_mask]
                target_b = target_b[collar_mask]
            
            # Skip overlapped regions if requested
            if self.skip_overlap:
                overlap_mask = target_b.sum(dim=1) <= 1
                pred_b = pred_b[overlap_mask]
                target_b = target_b[overlap_mask]
            
            if pred_b.numel() == 0:
                continue
            
            # Compute differentiable DER components
            # False Alarm: predicted speech where there's no target speech
            fa = ((pred_b > 0.5) & (target_b.sum(dim=1, keepdim=True) == 0)).float().sum()
            
            # Miss: target speech where there's no predicted speech  
            miss = ((pred_b.max(dim=1, keepdim=True)[0] <= 0.5) & (target_b.sum(dim=1, keepdim=True) > 0)).float().sum()
            
            # Speaker Error: predicted speech with wrong speaker assignment
            correct_detection = ((pred_b > 0.5) & (target_b > 0.5)).float().sum()
            total_prediction = (pred_b > 0.5).float().sum()
            speaker_error = total_prediction - correct_detection
            
            # Total speech time for normalization
            total_speech = target_b.sum()
            
            if total_speech > 0:
                der_b = (fa + miss + speaker_error) / total_speech
                total_der += der_b
        
        return total_der / batch_size


def create_loss_function(config: dict) -> nn.Module:
    """Factory function to create loss based on configuration."""
    
    loss_type = config.get('type', 'multitask')
    
    if loss_type == 'multitask':
        return MultiTaskDiarizationLoss(
            vad_loss_weight=config.get('vad_weight', 1.0),
            osd_loss_weight=config.get('osd_weight', 1.0),
            consistency_weight=config.get('consistency_weight', 0.1),
            use_pit=config.get('use_pit', True),
            use_focal=config.get('use_focal', False),
            focal_gamma=config.get('focal_gamma', 2.0),
            num_speakers=config.get('num_speakers', 4)
        )
    
    elif loss_type == 'pit':
        base_loss = WeightedBCELoss() if config.get('weighted', True) else nn.BCELoss()
        return PermutationInvariantLoss(base_loss, config.get('num_speakers', 4))
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('alpha', 1.0),
            gamma=config.get('gamma', 2.0)
        )
    
    elif loss_type == 'der':
        return DERLoss(
            collar=config.get('collar', 0.25),
            skip_overlap=config.get('skip_overlap', False)
        )
    
    else:
        return WeightedBCELoss()


if __name__ == "__main__":
    # Test the loss functions
    batch_size = 8
    time_steps = 500
    num_speakers = 4
    
    # Generate dummy data
    vad_pred = torch.sigmoid(torch.randn(batch_size, time_steps, num_speakers))
    osd_pred = torch.sigmoid(torch.randn(batch_size, time_steps))
    vad_target = torch.randint(0, 2, (batch_size, time_steps, num_speakers)).float()
    osd_target = torch.randint(0, 2, (batch_size, time_steps)).float()
    
    # Test MultiTaskDiarizationLoss
    loss_fn = MultiTaskDiarizationLoss(use_pit=True, use_focal=True)
    losses = loss_fn(vad_pred, osd_pred, vad_target, osd_target)
    
    print("Loss components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"{key}: {value.item():.4f}")
    
    # Test PIT loss standalone
    pit_loss = PermutationInvariantLoss(num_speakers=num_speakers)
    pit_loss_val, perms = pit_loss(vad_pred, vad_target)
    print(f"\nPIT Loss: {pit_loss_val.item():.4f}")
    print(f"Best permutations shape: {perms.shape}")
    
    # Test DER loss
    der_loss = DERLoss()
    der_val = der_loss(vad_pred, vad_target)
    print(f"DER Loss: {der_val.item():.4f}")