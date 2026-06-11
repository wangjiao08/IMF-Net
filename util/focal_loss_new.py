import torch
import torch.nn as nn
import torch.nn.functional as F


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


class OrdinalThresholdLoss(nn.Module):
    """
    阈值序关系辅助损失。

    对 9 类 logits 额外构造多个二分类任务：
        y >= 0.1
        y >= 1
        y >= 5
        y >= 10
        y >= 20
        y >= 25
        y >= 30
        y >= 50

    这样可以直接强化中高阈值的判别能力。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 对应 cfg.thresholds = [0.1,1,5,10,20,25,30,50]
        # 越高阈值权重越大，但不要过猛，否则 FAR 会明显上升
        self.threshold_weights = torch.tensor(
            [0.5, 0.8, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0],
            dtype=torch.float32,
            device=cfg.device
        )

    def forward(self, logits, target_classes):
        """
        logits: [B, 9, T, H, W]
        target_classes: [B, T, H, W], 取值 0~8
        """
        losses = []

        # 第 k 个阈值对应 target_classes >= k+1
        # 例如：
        # k=0 -> >=0.1mm/h -> class >=1
        # k=3 -> >=10mm/h  -> class >=4
        for k in range(len(self.cfg.thresholds)):
            class_cutoff = k + 1

            target_bin = (target_classes >= class_cutoff).float()

            # 二分类 logit：
            # 正类 = 所有 >= cutoff 的类别 logsumexp
            # 负类 = 所有 < cutoff 的类别 logsumexp
            pos_logit = torch.logsumexp(logits[:, class_cutoff:, :, :, :], dim=1)
            neg_logit = torch.logsumexp(logits[:, :class_cutoff, :, :, :], dim=1)
            binary_logit = pos_logit - neg_logit

            # 正样本权重：高阈值样本更少，所以给更大权重
            pos_weight = self.threshold_weights[k]

            bce = F.binary_cross_entropy_with_logits(
                binary_logit,
                target_bin,
                reduction='none'
            )

            weight_map = torch.where(
                target_bin > 0.5,
                pos_weight,
                torch.tensor(1.0, device=logits.device)
            )

            losses.append((bce * weight_map).mean())

        return torch.stack(losses).mean()


class focal_loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 原来的多分类 focal loss 权重保留
        weights = [1.0, 2.0, 5.0, 10.0, 15.0, 23.0, 25.0, 30.0, 50.0]
        class_weights = torch.FloatTensor(weights).to(cfg.device)

        self.class_loss = FocalLossLogits(weight=class_weights, gamma=2.0)

        # 新增：阈值序关系辅助损失
        self.ordinal_loss = OrdinalThresholdLoss(cfg)

        # 建议先从 0.2 开始
        self.lambda_ordinal = getattr(cfg, 'lambda_ordinal', 0.2)

    def forward(self, out_unet, target):
        # out_unet: [B, 9, T, H, W]
        # target:   [B, 1, T, H, W]

        denorm_true = target * (self.cfg.channel_max - self.cfg.channel_min) + self.cfg.channel_min

        thresholds_tensor = torch.tensor(
            self.cfg.thresholds,
            device=target.device
        )

        target_classes = torch.bucketize(
            denorm_true,
            thresholds_tensor,
            right=True
        ).squeeze(1).long()

        # 1. 原多分类 focal loss
        loss_cls = self.class_loss(out_unet, target_classes)

        # 2. 新增 ordinal threshold loss
        loss_ord = self.ordinal_loss(out_unet, target_classes)

        total_loss = loss_cls + self.lambda_ordinal * loss_ord

        return total_loss