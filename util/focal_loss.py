import torch.nn.functional as F
import torch
import torch.nn as nn





class FocalLossLogits(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, C, T, H, W]
        # targets: [B, T, H, W]
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()

        targets_unsqueezed = targets.unsqueeze(1)
        pt = prob.gather(1, targets_unsqueezed).squeeze(1)
        log_pt = log_prob.gather(1, targets_unsqueezed).squeeze(1)

        focal_term = (1.0 - pt).pow(self.gamma)
        loss = -focal_term * log_pt

        if self.weight is not None:
            weight_map = self.weight[targets]
            loss = loss * weight_map

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()




class focal_loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.punishment_factor = 2.0

        weights = [1.0, 2.0, 5.0, 10.0, 15.0, 23.0, 25.0, 30.0, 50.0]
        class_weights = torch.FloatTensor(weights).to(cfg.device)
        self.class_loss = FocalLossLogits(weight=class_weights, gamma=2.0)


    def forward(self, out_unet, target):
        # out_unet: [B, C, T, H, W] = (B,9,2,64,64)
        # target: [B, C, T, H, W] = (B,1,2,64,64)

        denorm_true = target * (self.cfg.channel_max - self.cfg.channel_min) + self.cfg.channel_min
        thresholds_tensor = torch.tensor(self.cfg.thresholds, device=target.device)
        target_classes = torch.bucketize(denorm_true, thresholds_tensor, right=True).squeeze(1).long()

        # 1. Focal 分类损失
        loss_unet = self.class_loss(out_unet, target_classes)
        total_loss = loss_unet
        return total_loss

