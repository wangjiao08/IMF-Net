import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from util.ordered_easydict import OrderedEasyDict as edict
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Linear, Dropout, Softmax, LayerNorm


# =========================================================================
# 1. Nano-TransUnet 配置
# =========================================================================
def get_nano_config(img_size=64):
    config = edict()
    config.patches = edict()
    config.patches.size = (1, 1)
    grid_size = img_size // 4
    config.patches.grid = (grid_size, grid_size)

    config.hidden_size = 256
    config.transformer = edict()
    config.transformer.mlp_dim = 512
    config.transformer.num_heads = 4
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0

    config.classifier = 'seg'
    config.resnet = edict()
    config.resnet.num_layers = (2,)
    config.resnet.width_factor = 0.5

    config.decoder_channels = (64, 32)
    config.skip_channels = [64, 32]
    config.n_skip = 2
    config.activation = 'softmax'
    return config


# =========================================================================
# 2. 基础组件 (保持不变)
# =========================================================================
class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(16, cin, eps=1e-6)
        self.conv1 = StdConv2d(cin, cmid, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(16, cmid, eps=1e-6)
        self.conv2 = StdConv2d(cmid, cmid, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(16, cmid, eps=1e-6)
        self.conv3 = StdConv2d(cmid, cout, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if (stride != 1 or cin != cout):
            self.downsample = StdConv2d(cin, cout, kernel_size=1, stride=stride, bias=False)
            self.gn_proj = nn.GroupNorm(16, cin, eps=1e-6)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(self.relu(self.gn_proj(x)))
        y = self.relu(self.gn1(x))
        y = self.conv1(y)
        y = self.relu(self.gn2(y))
        y = self.conv2(y)
        y = self.relu(self.gn3(y))
        y = self.conv3(y)
        return y + residual


class ResNetV2_Nano(nn.Module):
    def __init__(self, block_units, width_factor, in_channels=12):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.root = nn.Sequential(
            StdConv2d(in_channels, width, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(16, width, eps=1e-6),
            nn.ReLU(inplace=True)
        )
        self.stem_conv = nn.Sequential(
            StdConv2d(width, width * 2, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(16, width * 2, eps=1e-6),
            nn.ReLU(inplace=True)
        )
        self.stem_down2 = nn.Sequential(
            StdConv2d(width * 2, width * 4, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(16, width * 4, eps=1e-6),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            PreActBottleneck(width * 4, width * 4, width * 2),
            *[PreActBottleneck(width * 4, width * 4, width * 2) for _ in range(block_units[0] - 1)]
        )

    def forward(self, x):
        x_64 = self.root(x)
        x_32 = self.stem_conv(x_64)
        x_16 = self.stem_down2(x_32)
        x_final = self.body(x_16)
        return x_final, [x_16, x_32, x_64]


# --- Transformer 核心组件 (Attention, Mlp, Block, Embeddings, Transformer) ---
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer.attention_dropout_rate)
        self.proj_dropout = Dropout(config.transformer.attention_dropout_rate)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return self.proj_dropout(self.out(context_layer))


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer.mlp_dim)
        self.fc2 = Linear(config.transformer.mlp_dim, config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act_fn(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        x = x + self.attn(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels):
        super(Embeddings, self).__init__()
        self.hybrid_model = ResNetV2_Nano(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor,
                                          in_channels=in_channels)
        in_ch = self.hybrid_model.width * 4
        self.patch_embeddings = Conv2d(in_channels=in_ch, out_channels=config.hidden_size, kernel_size=1, stride=1)
        n_patches = (img_size // 4) ** 2
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)  # 对应图中的 Linear Projection (在代码中用 1x1 卷积实现)
        return self.dropout(x + self.position_embeddings), features


class Transformer(nn.Module):
    def __init__(self, config, img_size, in_channels):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = nn.ModuleList([Block(config) for _ in range(config.transformer.num_layers)])  #Transformer Layer  2层
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, input_ids):
        x, features = self.embeddings(input_ids)
        for layer in self.encoder:
            x = layer(x)
        return self.encoder_norm(x), features


# --- Decoder 组件 ---
class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, skip=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_more = Conv2dReLU(config.hidden_size, config.hidden_size//2, kernel_size=3, padding=1)
        in_channels = [config.hidden_size // 2] + list(config.decoder_channels[:-1])
        out_channels = config.decoder_channels # (128, 64, 32)
        skip_channels = config.skip_channels   # (128, 64, 32)
        self.blocks = nn.ModuleList([DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
                                     zip(in_channels, out_channels, skip_channels)])

    def forward(self, hidden_states, features):
        B, n_patch, hidden = hidden_states.size()
        h = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, h)
        x = self.conv_more(x)
        # 这里仅作初始化定义，具体流程在主类 forward 中按需调用以保留中间层
        return x



# =========================================================================
# 4. 最终主模型
# =========================================================================
class Transunet(nn.Module):
    def __init__(self, cfg, in_channels=12, input_timesteps=2, out_channels=9, output_timesteps=2,
                 img_size=64):
        super(Transunet, self).__init__()
        config = get_nano_config(img_size=img_size)
        self.out_c, self.out_t = out_channels, output_timesteps
        total_in, total_out = in_channels * input_timesteps, out_channels * output_timesteps

        self.transformer = Transformer(config, img_size=img_size, in_channels=total_in)
        self.decoder = DecoderCup(config)

        self.segmentation_head = nn.Conv2d(
            in_channels=config.decoder_channels[-1],
            out_channels=total_out,
            kernel_size=1
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)

        # 1. Encoder
        encoded, features = self.transformer(x)  # features: [x_16, x_32, x_64]


        # 2. 基础上采样 (Base Path)
        hidden = encoded.size(-1)
        d16_init = encoded.permute(0, 2, 1).contiguous().view(B, hidden, H // 4, W // 4)

        x_up = self.decoder.conv_more(d16_init)
        d32_base = self.decoder.blocks[0](x_up, features[1])
        d64_base = self.decoder.blocks[1](d32_base, features[2])


        out = self.segmentation_head(d64_base)

        cls_logits = out.reshape(B, self.out_c, self.out_t, H, W)



        return cls_logits

