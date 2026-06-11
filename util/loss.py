import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FACL(nn.Module):
    def __init__(self, total_step, const_ratio=0.4, prob_init=1, prob_end=0, include_sigmoid=False):
        super(FACL, self).__init__()
        const_step = int(total_step*const_ratio)
        self.prob_init = prob_init
        self.prob_end = prob_end
        self.prob_thres = torch.linspace(prob_init, prob_end, int(total_step-const_step))
        self.step = 0
        self.out = 0
        self.include_sigmoid = include_sigmoid

    def get_thres(self): ## default micro_batch = 1
        prob = self.prob_thres[self.step] if self.step < len(self.prob_thres) else self.prob_thres[-1] ## init(=1) to end(=0)
        self.step += 1
        # return self.out
        return 1-prob ## from 1-init to 1-end

    def fal(self, fft_pred, fft_gt):
        return nn.MSELoss()(fft_pred.abs(), fft_gt.abs())

    def fcl(self, fft_pred, fft_gt):
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred*fft_gt).sum().real
        denominator = torch.sqrt(((fft_gt).abs()**2).sum()*((fft_pred).abs()**2).sum())
        return 1. - numerator/denominator
    
    def forward(self, pred, gt):
        if self.include_sigmoid:
            pred = F.sigmoid(pred)
            gt = F.sigmoid(gt)

        fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
        fft_gt = torch.fft.fftn(gt, dim=[-1,-2], norm='ortho')
        prob = self.get_thres()
        
        H,W = pred.shape[-2:]
        weight = np.sqrt(H*W)
        loss = prob*self.fal(fft_pred, fft_gt) + (1-prob)*self.fcl(fft_pred, fft_gt)
        loss = loss * weight
        return loss


# class BGSuppressionLoss(nn.Module):
#     def __init__(self, bg_threshold_norm: float):
#         super().__init__()
#         self.bg_threshold_norm = bg_threshold_norm
#
#     def forward(self, pred, gt):
#         # gt, pred: [B, 1, T, H, W]
#         bg_mask = (gt < self.bg_threshold_norm).float()
#         loss = (pred.abs() * bg_mask).sum() / (bg_mask.sum() + 1e-6)
#         return loss

class HeavyRainWeightedMAELoss(nn.Module):
    """
    强降水加权 MAE。

    pred 和 gt 都是归一化后的降水：
        [B, 1, T, H, W]

    用反归一化后的 gt_mm 决定权重；
    误差本身仍然在归一化空间计算，避免 loss 数值过大。
    """

    def __init__(self, cfg):
        super().__init__()
        self.channel_min = cfg.channel_min
        self.channel_max = cfg.channel_max

    def forward(self, pred, gt):
        gt_mm = gt * (self.channel_max - self.channel_min) + self.channel_min

        w = torch.ones_like(gt)

        w = torch.where(gt_mm >= 1.0,  w * 1.2, w)
        w = torch.where(gt_mm >= 5.0,  w * 1.8, w)
        w = torch.where(gt_mm >= 10.0, w * 2.5, w)
        w = torch.where(gt_mm >= 20.0, w * 4.0, w)
        w = torch.where(gt_mm >= 30.0, w * 6.0, w)

        loss = (torch.abs(pred - gt) * w).sum() / (w.sum() + 1e-6)
        return loss




