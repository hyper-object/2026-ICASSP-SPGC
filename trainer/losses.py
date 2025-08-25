import torch
import torch.nn as nn
import torch.nn.functional as F

def l1_loss(pred, target, mask=None):
    if mask is None:
        return F.l1_loss(pred, target)
    m = mask[:, None, :, :].float()   # (N,1,H,W)
    num = torch.clamp(m.sum() * pred.shape[1], min=1.0)
    return (torch.abs(pred - target) * m).sum() / num

def sam_loss(pred, target, eps=1e-8, mask=None):
    # pred/target: (N,C,H,W)
    N,C,H,W = pred.shape
    p = pred.permute(0,2,3,1).reshape(-1, C)
    t = target.permute(0,2,3,1).reshape(-1, C)
    if mask is not None:
        m = mask.reshape(-1).bool()
        p = p[m]; t = t[m]
    p = p + eps; t = t + eps
    num = (p * t).sum(dim=1)
    den = torch.norm(p, dim=1) * torch.norm(t, dim=1) + eps
    cos = torch.clamp(num / den, -1 + 1e-7, 1 - 1e-7)
    ang = torch.acos(cos)  # radians
    return ang.mean()

class ReconLoss(nn.Module):
    def __init__(self, lambda_sam=0.1):
        super().__init__()
        self.lambda_sam = lambda_sam
    def forward(self, pred, target, mask=None):
        return l1_loss(pred, target, mask) + self.lambda_sam * sam_loss(pred, target, mask=mask)
