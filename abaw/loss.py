import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torchmetrics.functional.regression import pearson_corrcoef


def _safe_pearson(predictions, labels, eps=1e-8):
    """Numerically stable per-column Pearson correlation in float32."""
    # Always compute in float32 for stability
    predictions = predictions.float()
    labels = labels.float()
    
    B = predictions.size(0)
    if B < 3:
        # Too few samples for meaningful correlation
        return torch.zeros(predictions.size(1), device=predictions.device)
    
    # De-mean
    pred_mean = predictions.mean(dim=0, keepdim=True)
    label_mean = labels.mean(dim=0, keepdim=True)
    pred_centered = predictions - pred_mean
    label_centered = labels - label_mean
    
    # Variance check
    pred_var = (pred_centered ** 2).sum(dim=0)
    label_var = (label_centered ** 2).sum(dim=0)
    
    # Correlation
    cov = (pred_centered * label_centered).sum(dim=0)
    denom = torch.sqrt(pred_var * label_var + eps)
    corr = cov / denom
    
    # Clamp to valid range
    corr = torch.clamp(corr, -1.0, 1.0)
    
    # Set to 0 where variance is too small
    low_var_mask = (pred_var < eps) | (label_var < eps)
    corr = corr.masked_fill(low_var_mask, 0.0)
    
    return corr


class MSE(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_function = torch.nn.MSELoss(reduction=reduction)

    def forward(self, features, labels, sample_weights=None):
        if sample_weights is not None:
            loss = F.mse_loss(features, labels, reduction='none')  # [B, 6]
            loss = (loss * sample_weights.unsqueeze(1)).mean()
            return loss
        return self.loss_function(features, labels)


class CORR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels, sample_weights=None):
        corr = _safe_pearson(predictions, labels)
        return (1 - corr).mean()


class MixedLoss(nn.Module):
    """Mixed MSE + (1 - Pearson) loss. Directly optimizes eval metric while keeping MSE for stability."""
    def __init__(self, mse_weight=0.3, corr_weight=0.7):
        super().__init__()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight

    def forward(self, features, labels, sample_weights=None):
        # MSE component (works fine in fp16)
        if sample_weights is not None:
            mse = (F.mse_loss(features, labels, reduction='none') * sample_weights.unsqueeze(1)).mean()
        else:
            mse = F.mse_loss(features, labels)

        # Pearson correlation component (computed in float32)
        corr = _safe_pearson(features, labels)  # [6]
        corr_loss = (1 - corr).mean()

        return self.mse_weight * mse + self.corr_weight * corr_loss


class GateIntensityLoss(nn.Module):
    """Combined loss for Gate + Intensity dual-head model.
    Loss = alpha * BCE_logits(gate, 1_{y>0}) + beta * MSE(pred, y) + gamma * (1-Pearson(pred, y))
    """
    def __init__(self, gate_weight=0.3, pred_mse_weight=0.2, pred_corr_weight=0.5):
        super().__init__()
        self.gate_weight = gate_weight
        self.pred_mse_weight = pred_mse_weight
        self.pred_corr_weight = pred_corr_weight

    def forward(self, model_output, labels, sample_weights=None):
        pred, gate_logits, intensity_logits = model_output
        # Gate target: 1 if label > 0, else 0
        gate_target = (labels > 0).float()

        # Gate loss: BCE with logits (safe for autocast)
        gate_loss = F.binary_cross_entropy_with_logits(gate_logits, gate_target, reduction='none')  # [B, 6]
        if sample_weights is not None:
            gate_loss = (gate_loss * sample_weights.unsqueeze(1)).mean()
        else:
            gate_loss = gate_loss.mean()

        # Pred loss: MSE on final pred
        if sample_weights is not None:
            mse_loss = (F.mse_loss(pred, labels, reduction='none') * sample_weights.unsqueeze(1)).mean()
        else:
            mse_loss = F.mse_loss(pred, labels)

        # Pred loss: Pearson correlation (stable, float32)
        corr = _safe_pearson(pred, labels)
        corr_loss = (1 - corr).mean()

        total = self.gate_weight * gate_loss + self.pred_mse_weight * mse_loss + self.pred_corr_weight * corr_loss
        return total


class CCC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, labels, sample_weights=None):
        return 2 * torch.cov(torch.cat([features, labels], dim=1)) / (
                    features.var() + labels.var() + (features.mean() - labels.mean()) ** 2)


class MSECCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, features, labels, sample_weights=None):
        return (self.loss_function(features, labels) + (2 * torch.cov(torch.cat([features, labels], dim=1)) / (
                     features.var() + labels.var() + (features.mean() - labels.mean()) ** 2)) / 2).mean()


class MTLoss(nn.Module):
    def __init__(self, num_task, loss_fn, label_smoothing=0.0):
        super(MTLoss, self).__init__()
        self.requires_grad_(True)
        self.log_std = nn.Parameter(torch.zeros(num_task))
        self.loss_fn = loss_fn.__class__(reduction='none')
        self.label_smoothing = label_smoothing
        self.weight = None

    def forward(self, predict: dict, target: dict):
        l1 = self.loss_fn(predict[0], target).mean(axis=1)
        l2 = self.loss_fn(predict[1], target).mean(axis=1)
        loss = torch.stack((l1, l2)).transpose(0, 1)
        self.weight = torch.exp(-2.0*self.log_std)
        ret = torch.matmul(loss, self.weight)
        ret += self.log_std.sum()
        ret = ret.mean()
        return {'loss': ret, 'task_loss': loss.mean(dim=0), 'task_weight': self.weight,
                'task_uncertainty': torch.exp(self.log_std),}

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)


class EmotionWeightedMSE(nn.Module):
    """Per-emotion frequency-weighted MSE.
    Emotions with fewer non-zero training samples get higher weight,
    so that rare emotions (e.g. Empathic Pain ~10% non-zero) contribute
    proportionally to the Pearson mean metric.

    Training non-zero frequencies (train_split.csv):
        Admiration: 0.447, Amusement: 0.364, Determination: 0.325,
        Empathic Pain: 0.101, Excitement: 0.382, Joy: 0.325
    Weights = 1/freq, normalised so mean weight = 1.
    """

    def __init__(self):
        super().__init__()
        nonzero_freq = torch.tensor([0.447, 0.364, 0.325, 0.101, 0.382, 0.325],
                                    dtype=torch.float32)
        weights = 1.0 / nonzero_freq          # higher weight for rarer emotion
        weights = weights / weights.mean()    # mean weight = 1, total = 6
        self.register_buffer('weights', weights)

    def forward(self, features, labels, sample_weights=None):
        loss = F.mse_loss(features, labels, reduction='none')  # [B, 6]
        loss = loss * self.weights.to(features.device).unsqueeze(0)  # broadcast [1,6]
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(1)
        return loss.mean()
