# ==============================================================================
# --- FOCAL LOSS FOR CLASS IMBALANCE ---
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance with per-class alpha weights."""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Tensor of shape (num_classes,) with per-class weights, or None for no weighting
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply per-class alpha weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)  # Shape: (batch_size,)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
