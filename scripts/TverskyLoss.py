import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        Args:
            alpha (float): Weight for False Positives.
            beta (float): Weight for False Negatives.
            smooth (float): Smoothing constant.

        Note:
            - alpha + beta should ideally equal 1.
            - If alpha=0.5 and beta=0.5, this is equivalent to Dice Loss.
            - For tiny targets (imbalanced), setting beta > alpha (e.g., beta=0.7, alpha=0.3)
              emphasizes recall (penalizing missed pixels more).
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes, H, W) - Raw Logits
            targets: (batch_size, H, W) - Ground truth labels
        """
        num_classes = inputs.shape[1]

        # 1. Apply Softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)

        # 2. One-Hot Encode Targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # 3. Flatten
        inputs_flat = inputs.view(inputs.size(0), num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), num_classes, -1)

        # 4. Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
        TP = (inputs_flat * targets_flat).sum(dim=2)
        FP = ((1 - targets_flat) * inputs_flat).sum(dim=2)
        FN = (targets_flat * (1 - inputs_flat)).sum(dim=2)

        # 5. Calculate Tversky Index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # 6. Loss
        return 1 - tversky.mean()