import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDiarizationLoss(nn.Module):
    """Simplified loss function for speaker diarization."""
    
    def __init__(self, vad_weight=1.0, osd_weight=1.0, focal_gamma=2.0, focal_alpha=0.25):
        super(SimpleDiarizationLoss, self).__init__()
        self.vad_weight = vad_weight
        self.osd_weight = osd_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
    def focal_loss(self, logits, targets, gamma=2.0, alpha=0.25):
        """
        Focal loss to handle class imbalance in diarization.
        Compatible with AMP (automatic mixed precision).
        
        Args:
            logits: [batch, ...] prediction logits (before sigmoid)
            targets: [batch, ...] target labels (0 or 1)
            gamma: focusing parameter
            alpha: weighting factor for rare class
        """
        # Use binary_cross_entropy_with_logits for AMP compatibility
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate probabilities for focal weighting
        predictions = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, predictions, 1 - predictions)
        
        # Calculate alpha_t
        alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
        
        # Calculate focal weight
        focal_weight = alpha_t * (1 - p_t) ** gamma
        
        return (focal_weight * bce).mean()
    
    def forward(self, vad_pred, osd_pred, vad_target, osd_target):
        """
        Compute total diarization loss.
        
        Args:
            vad_pred: [batch, time, speakers] VAD predictions
            osd_pred: [batch, time] OSD predictions  
            vad_target: [batch, time, speakers] VAD targets
            osd_target: [batch, time] OSD targets
            
        Returns:
            dict with loss components
        """
        # VAD loss with focal loss for class imbalance
        vad_loss = self.focal_loss(
            vad_pred, vad_target, 
            gamma=self.focal_gamma, 
            alpha=self.focal_alpha
        )
        
        # OSD loss with focal loss
        osd_loss = self.focal_loss(
            osd_pred, osd_target,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha
        )
        
        # Total weighted loss
        total_loss = self.vad_weight * vad_loss + self.osd_weight * osd_loss
        
        return {
            'total_loss': total_loss,
            'vad_loss': vad_loss,
            'osd_loss': osd_loss
        }


class PermutationInvariantLoss(nn.Module):
    """
    Simplified Permutation Invariant Training (PIT) loss.
    Only use if you need to handle speaker permutation invariance.
    """
    
    def __init__(self, base_loss_fn=None):
        super(PermutationInvariantLoss, self).__init__()
        self.base_loss_fn = base_loss_fn or nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        Find best permutation and compute loss.
        
        Args:
            predictions: [batch, time, speakers] predictions
            targets: [batch, time, speakers] targets
        """
        batch_size, seq_len, num_speakers = predictions.shape
        
        if num_speakers <= 1:
            # No permutation needed for single speaker
            return self.base_loss_fn(predictions, targets)
        
        # Generate all permutations (only feasible for small num_speakers)
        import itertools
        all_permutations = list(itertools.permutations(range(num_speakers)))
        
        min_loss = float('inf')
        best_perm = None
        
        for perm in all_permutations:
            # Apply permutation to predictions
            perm_pred = predictions[:, :, list(perm)]
            
            # Compute loss for this permutation
            loss = self.base_loss_fn(perm_pred, targets)
            
            if loss < min_loss:
                min_loss = loss
                best_perm = perm
        
        return min_loss


def create_loss_function(config):
    """Factory function to create loss based on config."""
    loss_type = config.get('type', 'simple')
    
    if loss_type == 'simple':
        return SimpleDiarizationLoss(
            vad_weight=config.get('vad_weight', 1.0),
            osd_weight=config.get('osd_weight', 1.0),
            focal_gamma=config.get('focal_gamma', 2.0),
            focal_alpha=config.get('focal_alpha', 0.25)
        )
    elif loss_type == 'pit':
        base_loss = SimpleDiarizationLoss(
            vad_weight=config.get('vad_weight', 1.0),
            osd_weight=config.get('osd_weight', 1.0)
        )
        return PermutationInvariantLoss(base_loss)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")