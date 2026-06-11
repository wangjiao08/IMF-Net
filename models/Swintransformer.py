import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
from util.pad import get_pad2d


#将输入的空间图（64x64）切成 16x16 的网格（每个网格4x4像素）
class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # 计算 Patch 分辨率 (保持不变)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim

        # =========================================================
        # 【核心修改】Overlapping Embedding (重叠切片)
        # 原来: kernel_size=(2, 4, 4), padding=0
        # 现在: kernel_size=(2, 7, 7), padding=(0, 3, 3)
        #
        # 解释:
        # 1. 保持时间维度 T 不变 (kernel=patch_size[0])
        # 2. 空间维度加大到 7 (7x7 卷积核)
        # 3. 加 Padding=3 保证输出尺寸不变 (H_out = (H + 2*3 - 7)/4 + 1 = H/4)
        # =========================================================

        # 确保 patch_size[1] 和 [2] 是 4，才使用这个 padding=3
        assert patch_size[1] == 4 and patch_size[2] == 4, "目前参数仅针对 patch_size=4 调整"

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size[0], 7, 7),  # 空间上由 4 改为 7，实现重叠
            stride=patch_size,  # 步长保持 4 不变
            padding=(0, 3, 3)  # 填充 3，维持输出尺寸一致
        )  #就是说以前的卷积核的大小是4*4,步长也是4,这样的话每个 patch 独立，互不相交，而现在的话就可以有重叠

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        assert T == self.img_size[0] and Lat == self.img_size[1] and Lon == self.img_size[2], \
            f"Input image size ({T}*{Lat}*{Lon}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, num_groups: int, num_residuals: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        res = x + shortcut
        if h % 2 != 0:
            res = res[:, :, :-1, :]
        if w % 2 != 0:
            res = res[:, :, :, :-1]
        return res


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        # =========================================================
        # 【修改点】: 弃用 ConvTranspose2d，改用 Upsample + Conv
        # 转置卷积是造成“棋盘格伪影”的元凶之一
        # 讲解文章Deconvolution and Checkerboard Artifacts
        # =========================================================
        self.conv = nn.Sequential(
            # 1. 也是用双线性插值放大 2 倍
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 2. 接一个卷积层调整通道
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False)
        )

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)
        shortcut = x
        x = self.b(x)
        return x + shortcut

class UTransformer(nn.Module):
    """U-Transformer
    Args:
        embed_dim (int): Patch embedding dimension.
        num_groups (int | tuple[int]): number of groups to separate the channels into.
        input_resolution (tuple[int]): Lat, Lon.
        num_heads (int): Number of attention heads in different layers.
        window_size (int | tuple[int]): Window size.
        depth (int): Number of blocks.
    """
    def __init__(self, embed_dim, num_groups, input_resolution, num_heads, window_size, depth):
        super().__init__()
        num_groups = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)
        padding = get_pad2d(input_resolution, window_size)
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right
        self.down = DownBlock(embed_dim, embed_dim, num_groups[0])
        self.layer = SwinTransformerV2Stage(embed_dim, embed_dim, input_resolution, depth, num_heads, window_size)
        self.up = UpBlock(embed_dim * 2, embed_dim, num_groups[1])

    def forward(self, x):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding
        x = self.down(x)

        shortcut = x

        # pad
        x = self.pad(x)
        _, _, pad_lat, pad_lon = x.shape

        x = x.permute(0, 2, 3, 1)  # B Lat Lon C
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)

        # crop
        x = x[:, :, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right]

        # concat
        x = torch.cat([shortcut, x], dim=1)  # B 2*C Lat Lon

        x = self.up(x)
        return x

#修改后的
class Swintransformer(nn.Module):
    """
    针对 64x64 输入优化的无网格版本
    """

    def __init__(self, img_size=(2, 64, 64), patch_size=(2, 4, 4), in_chans=12, out_chans=1,
                 embed_dim=512, num_groups=32, num_heads=8, window_size=8, out_timesteps=2):
        super().__init__()
        # 保持原有的分辨率计算逻辑 (针对 UTransformer)
        input_resolution = int(img_size[1] / patch_size[1] / 2), int(img_size[2] / patch_size[2] / 2)

        self.cube_embedding = CubeEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.u_transformer = UTransformer(embed_dim, num_groups, input_resolution, num_heads, window_size, depth=8)

        self.patch_size = patch_size
        self.out_chans = out_chans
        self.img_size = img_size
        self.out_timesteps = out_timesteps

        # =========================================================================
        # 【核心修改】: 使用插值上采样 (Bilinear Upsample) 替代 fc+Reshape
        # =========================================================================

        # 1. 计算放大倍数
        # 你的 patch_size 是 4，意味着特征图比原图小 4 倍
        # 所以我们需要放大 4 倍
        scale_factor = patch_size[1]  # = 4

        # 2. 计算最终输出通道
        final_out_channels = out_timesteps * out_chans

        self.upsample_head = nn.Sequential(
            # 第一步：降维 (512 -> 128)，减少计算量
            nn.Conv2d(embed_dim, 128, kernel_size=1),
            nn.GroupNorm(8, 128),  # 归一化，帮助收敛
            nn.SiLU(),

            # 第二步：【关键】双线性插值放大 4 倍
            # mode='bilinear' 会自动计算平滑过渡，彻底消除网格
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),  #整个图片放大4*4倍

            # 第三步：卷积融合（融合那些插出来的值） (Refinement)
            # 在高分辨率图上做一次卷积，提取细节
            nn.Conv2d(128, 64, kernel_size=3, padding=1), #图片大小不变
            nn.SiLU(),

            # 第四步：输出层 (映射到最终的预测值)
            nn.Conv2d(64, final_out_channels, kernel_size=1)
        )

        # =========================================================================
        # 【彻底消除马赛克的升级版 Upsample Head (使用 PixelShuffle)】
        # =========================================================================
        # self.upsample_head = nn.Sequential(
        #     # 1. 降维到 256
        #     nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1, padding_mode='reflect'),  加入padding_mode='reflect'好像会不可复现
        #     nn.GroupNorm(16, 256),
        #     nn.SiLU(),
        #
        #     # 2. 准备 PixelShuffle 需要的通道数 (输出64通道，放大4倍，需 64 * 4^2 = 1024 通道)
        #     nn.Conv2d(256, 64 * (scale_factor ** 2), kernel_size=3, padding=1, padding_mode='reflect'),
        #     nn.PixelShuffle(scale_factor),  # 形状瞬间变为[B, 64, H*4, W*4]，没有传统插值的痕迹
        #
        #     # 3. 后处理打磨：连续两层 3x3 卷积，把重组后的特征完全揉碎融合，磨平所有网格边界
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect'),
        #     nn.SiLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect'),
        #     nn.SiLU(),
        #
        #     # 4. 输出层
        #     nn.Conv2d(64, final_out_channels, kernel_size=1)
        # )
        # =========================================================================

    def forward(self, x: torch.Tensor):
        # x input: [B, C_in, T_in, H, W]
        B, _, _, _, _ = x.shape

        # 1. 骨干网络提取特征
        x = self.cube_embedding(x).squeeze(2)  # [B, embed_dim, Lat/4, Lon/4]
        x = self.u_transformer(x)  # [B, embed_dim, Lat/4, Lon/4]

        # 2. 上采样头 (放大 + 预测)
        # 输入是 16x16，放大4倍 -> 输出 64x64
        x = self.upsample_head(x)  # [B, out_timesteps*out_chans, 64, 64]

        # 3. 维度重塑
        # [B, T*C, H, W] -> [B, T, C, H, W]
        H, W = x.shape[-2], x.shape[-1]
        x = x.reshape(B, self.out_timesteps, self.out_chans, H, W)

        # 4. 调整通道顺序以匹配标签
        # [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        # 5. 安全检查 (对于 64x64 其实不需要，但留着防止未来改尺寸)
        if x.shape[-2:] != self.img_size[1:]:
            x = F.interpolate(x, size=self.img_size[1:], mode="nearest")

        return x

