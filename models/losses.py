"""V4 multitask loss — balanced Huber (speed) + angular (heading) + cross-track + norm reg.

Extracted from dead-sih-main.ipynb cell 25. Depends on a fitted `scaler` for v_std/v_mean.
"""
import torch
import torch.nn as nn

VEL_WEIGHT = 1.0
HEAD_WEIGHT = 1.0
CROSS_TRACK_WEIGHT = 0.5
NORM_REG = 0.1
_huber = nn.HuberLoss(reduction='mean')


def multitask_loss_v4(pred, y, m, scaler):
    """pred, y: (B,3) = [vel_norm, sin, cos]; m: (B,1) heading-valid mask."""
    m = m.squeeze(1)
    denom = m.sum().clamp(min=1.0)

    vel_loss = _huber(pred[:, 0], y[:, 0])

    sc = pred[:, 1:]
    norm = sc.norm(dim=1).clamp(min=1e-6)
    unit = sc / norm.unsqueeze(1)
    head_loss = ((1.0 - (unit * y[:, 1:]).sum(dim=1)) * m).sum() / denom
    reg = (((norm - 1.0) ** 2) * m).sum() / denom

    sin_dpsi = unit[:, 0] * y[:, 2] - unit[:, 1] * y[:, 1]
    v_true = (y[:, 0] * scaler.v_std + scaler.v_mean).clamp(min=0.0) / scaler.v_std
    cross_track = (v_true * sin_dpsi.abs() * m).sum() / denom

    return (VEL_WEIGHT * vel_loss + HEAD_WEIGHT * head_loss
            + CROSS_TRACK_WEIGHT * cross_track + NORM_REG * reg)
