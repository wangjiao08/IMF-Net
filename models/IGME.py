import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

"ee模块和PFIM模块结合"


# =========================================================================
# PFIM
# =========================================================================
class LowerBoundFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through = (x >= bound) | (grad_output < 0)
        return grad_output * pass_through.to(grad_output.dtype), None


def lower_bound(x, bound):
    if not torch.is_tensor(bound):
        bound = torch.tensor(bound, dtype=x.dtype, device=x.device)
    return LowerBoundFn.apply(x, bound)


class GDN2d(nn.Module):
    """简化版 GDN，按 PFIM 思路用于密度建模。"""
    def __init__(self, channels, inverse=False, beta_min=1e-6, gamma_init=0.1, reparam_offset=2 ** -18):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.reparam_offset = reparam_offset

        pedestal = reparam_offset ** 2
        self.register_buffer('pedestal', torch.tensor(pedestal))

        beta_bound = (beta_min + pedestal) ** 0.5
        gamma_bound = reparam_offset
        self.register_buffer('beta_bound', torch.tensor(beta_bound))
        self.register_buffer('gamma_bound', torch.tensor(gamma_bound))

        beta = torch.sqrt(torch.ones(channels) + pedestal)
        gamma = torch.sqrt(torch.eye(channels) * gamma_init + pedestal)
        self.beta_reparam = nn.Parameter(beta)
        self.gamma_reparam = nn.Parameter(gamma)

    def forward(self, x):
        _, c, _, _ = x.shape

        beta = lower_bound(self.beta_reparam, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        gamma = lower_bound(self.gamma_reparam, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(c, c, 1, 1)

        norm = F.conv2d(x ** 2, gamma, beta)
        norm = torch.sqrt(norm + 1e-6)
        if self.inverse:
            return x * norm
        return x / norm


class PFIM(nn.Module):
    """
    PFIM（面向小散降水的任务对齐版）：
    1) 对浅层特征进行可微量化；
    2) 用 Conv + GDN + ReLU 预测 μ / σ；
    3) 由 sigma 生成 uncertainty map；
    4) 再用 sparse_head 学习“小散降水概率图”；
    5) 两者相乘得到 task_map，并以弱门控方式增强浅层特征。
    """
    def __init__(self, channels, hidden_channels=None, eps=1e-6, alpha=0.05):
        super().__init__()
        hidden = hidden_channels or channels
        self.eps = eps
        self.alpha = alpha

        self.feature_model = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            GDN2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            GDN2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.shared_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)
        self.sigma_head = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)
        self.sparse_head = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)

        nn.init.zeros_(self.sparse_head.weight)
        nn.init.zeros_(self.sparse_head.bias)

    @staticmethod
    def _standard_normal_cdf(x):
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _likelihood(self, y_hat, mu, sigma):
        sigma = torch.clamp(sigma, min=self.eps)
        upper = (y_hat + 0.5 - mu) / sigma
        lower = (y_hat - 0.5 - mu) / sigma
        probs = self._standard_normal_cdf(upper) - self._standard_normal_cdf(lower)
        return torch.clamp(probs, min=1e-9)

    def forward(self, y):
        # y: [B, C, H, W]
        if self.training:
            y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            y_hat = torch.round(y)

        feat = self.feature_model(y)
        feat = self.shared_head(feat)


        mu = self.mu_head(feat)  #均值图
        sigma = F.softplus(self.sigma_head(feat)) + self.eps   #方差图
        topk_vals, _ = torch.topk(sigma, k=min(2, sigma.shape[1]), dim=1)
        uncert_map = topk_vals.mean(dim=1, keepdim=True)
        uncert_map = (uncert_map - uncert_map.mean(dim=(2, 3), keepdim=True)) / (
            uncert_map.std(dim=(2, 3), keepdim=True) + 1e-6
        )
        uncert_map = torch.sigmoid(uncert_map)  #信息图

        sparse_logits = self.sparse_head(feat)
        sparse_prob = torch.sigmoid(sparse_logits)  #稀疏概率图


        task_map = uncert_map * sparse_prob  #任务调制图

        likelihood = self._likelihood(y_hat, mu, sigma)
        bits = -torch.log2(likelihood)
        bits_map = bits.mean(dim=1, keepdim=True)
        pfim_loss = bits_map.mean()    #信息熵损失


        # 弱门控增强
        y_enhanced = y * (1.0 + self.alpha * task_map)


        return y_enhanced,pfim_loss







# ----------------- 基础组件 -----------------
class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# ----------------- 感知块 (Perception Block) -----------------
class RSF(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(RSF, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw (升维)
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw (特征提取)
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear (降维)
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ----------------- EE 模块 (Resizer) -----------------
class IGME(nn.Module):
    def __init__(self, in_channels, scale=1):
        """
        in_channels: 输入通道数 (你的任务中是 12*2=24)
        scale: 放大倍数 (如果是特征增强不改变大小，设为 1)
        """
        super().__init__()
        head_channel = 32

        # Head  先做浅层映射（先把原始输入“变成特征”）
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(head_channel),
            nn.ReLU(inplace=True)
        )

        # Body (4个 RSF Block)  感知模块   先把整张图的特征提出来
        self.body = nn.Sequential(
            RSF(32, 64, 3, 1, 64, True, nl='RE'),
            RSF(64, 64, 3, 1, 72, True, nl='RE'),
            RSF(64, 88, 3, 1, 96, True, nl='RE'),
            RSF(88, 88, 3, 1, 128, True, nl='RE')
        )


        ###判断哪里该重点增强
        self.pfim = PFIM(
            channels=88,
            hidden_channels=88,
            alpha=0.05
        )

        # Tail   把整张特征图放大并映射回输出的。
        modules_tail = []
        # PixelShuffle 放大需要通道数变大 scale^2 倍
        modules_tail.append(nn.Conv2d(88, scale ** 2 * 88, 1, padding=0, stride=1))
        modules_tail.append(nn.PixelShuffle(scale))
        modules_tail.append(nn.BatchNorm2d(88))
        modules_tail.append(nn.ReLU(True))

        # 映射回原始输入通道数
        modules_tail.append(nn.Conv2d(88, in_channels, 1, padding=0, stride=1))



        self.tail = nn.Sequential(*modules_tail)

        # Skip Connection (双线性插值)
        self.interpolate = partial(F.interpolate,
                                   scale_factor=scale,
                                   mode='nearest'
                                  )

    def forward(self, x):
        # x: [B, C, H, W]
        identity = x

        out = self.head(x)
        out = self.body(out)
        out, pfim_loss = self.pfim(out)  # 新增
        out = self.tail(out)

        identity = self.interpolate(identity)

        return out + identity , pfim_loss