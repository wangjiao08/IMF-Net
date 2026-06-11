import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFeedForward(nn.Module):
    """
    带门控机制的前馈网络 (GDFN)，用于增强特征表达并抑制噪声变量
    """

    def __init__(self, dim, expansion_factor=2.66):
        super(GatedFeedForward, self).__init__()
        hidden_dim = int(dim * expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        # 门控机制：利用 x2 激活 x1
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class HeteroVariableCrossAttention(nn.Module):
    """[核心算子] 异构变量交叉注意力机制  以前是VariableCrossAttention单输入，以前是当作同一数据源内部去学习所有通道直接的关系，现在是两个数据源之间的通道学习关系
    专门用于处理通道数不同的两个数据源之间的协方差计算
    """

    def __init__(self, dim_q, dim_kv, dim_out):
        super().__init__()
        # 将不同模态的数据映射到各自的特征空间
        self.q_proj = nn.Conv2d(dim_q, dim_q, kernel_size=1)
        self.k_proj = nn.Conv2d(dim_kv, dim_kv, kernel_size=1)
        self.v_proj = nn.Conv2d(dim_kv, dim_kv, kernel_size=1)

        # 将交互后的特征统一映射到全局维度 (dim_out = 24)
        self.out_proj = nn.Conv2d(dim_q, dim_out, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.norm = nn.InstanceNorm2d(dim_out, affine=True)
        self.ffn = GatedFeedForward(dim_out)

    def forward(self, x_q, x_kv):
        """
        x_q: 模态A (例如雷达, [B, 2, H, W])
        x_kv: 模态B (例如卫星, [B, 12, H, W])
        """
        B, C_q, H, W = x_q.shape
        _, C_kv, _, _ = x_kv.shape

        q = self.q_proj(x_q).view(B, C_q, -1)  # [B, C_q, HW]
        k = self.k_proj(x_kv).view(B, C_kv, -1)  # [B, C_kv, HW]
        v = self.v_proj(x_kv).view(B, C_kv, -1)  # [B, C_kv, HW]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 核心：计算异构变量协方差矩阵 (Attention Map: C_q x C_kv)
        # 例如雷达(2)和卫星(12)，这里算出的就是 2x12 的物理相关性矩阵
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 特征融合
        out = attn @ v  # [B, C_q, HW]
        out = out.view(B, C_q, H, W)

        x_attn = self.out_proj(out)  # 映射到统一的通道维度
        x_res = self.norm(x_attn)
        x_res = self.ffn(x_res)

        return x_attn + x_res

class MOEFUSION(nn.Module):
    """
    [顶级创新] 异构跨模态混合专家融合网络 (HCM-MoE)
    让专家负责不同数据源之间的两两交互，完美解决跨源融合与雷达单通道问题
    """

    def __init__(self, in_channels=12, in_timesteps=2):
        super(MOEFUSION, self).__init__()
        dim_total = in_channels * in_timesteps  # 24
        self.dim_rad = 1 * in_timesteps  # 雷达: 2
        self.dim_sfc = 5 * in_timesteps  # 地面: 10
        self.dim_sat = 6 * in_timesteps  # 卫星: 12

        # -----------------------------------------------------------
        # 定义3个跨模态交互专家 (Cross-Modal Interactors)
        # -----------------------------------------------------------
        # 专家1：雷达查询卫星 (微物理过程)
        self.expert_rad_sat = HeteroVariableCrossAttention(dim_q=self.dim_rad, dim_kv=self.dim_sat, dim_out=dim_total)

        # 专家2：地面查询卫星 (大尺度热力动力响应)
        self.expert_sfc_sat = HeteroVariableCrossAttention(dim_q=self.dim_sfc, dim_kv=self.dim_sat, dim_out=dim_total)

        # 专家3：雷达查询地面 (近地层降水强迫)
        self.expert_rad_sfc = HeteroVariableCrossAttention(dim_q=self.dim_rad, dim_kv=self.dim_sfc, dim_out=dim_total)

        # -----------------------------------------------------------
        # 动态门控网络 (Spatial Gating Network)
        # -----------------------------------------------------------
        self.gating_network = nn.Sequential(
            nn.Conv2d(dim_total, dim_total // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_total // 2, 3, kernel_size=1)  # 3个专家，输出3个权重
        )

        self.channel_mixer = nn.Conv2d(dim_total, dim_total, kernel_size=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        # 展平时间维[B, 24, H, W]
        x_flat = x.permute(0, 1, 2, 3, 4).reshape(B, C * T, H, W)

        # 严格按照物理通道切分
        x_rad = x_flat[:, 0:self.dim_rad, :, :]
        x_sfc = x_flat[:, self.dim_rad: self.dim_rad + self.dim_sfc, :, :]
        x_sat = x_flat[:, self.dim_rad + self.dim_sfc:, :, :]

        # 1. 专家独立进行跨源诊断 (都输出 [B, 24, H, W])
        out_expert1 = self.expert_rad_sat(x_rad, x_sat)  # 雷达-卫星交互
        out_expert2 = self.expert_sfc_sat(x_sfc, x_sat)  # 地面-卫星交互
        out_expert3 = self.expert_rad_sfc(x_rad, x_sfc)  # 雷达-地面交互

        # 2. 门控网络生成空间权重[B, 3, H, W]
        gate_weights = F.softmax(self.gating_network(x_flat), dim=1)

        # 3. MoE 加权融合 (广播相乘再相加)
        x_fused = (out_expert1 * gate_weights[:, 0:1, :, :] +
                   out_expert2 * gate_weights[:, 1:2, :, :] +
                   out_expert3 * gate_weights[:, 2:3, :, :])

        # 4. 加上原始宏观特征的残差
        x_fused = x_fused + x_flat

        x_out = self.channel_mixer(x_fused)

        return x_out.view(B, C, T, H, W)
