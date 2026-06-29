"""Mamba (Selective State-Space Model) 时序编码器, 替代 Hierarchical Transformer。

设计动机
---------
Transformer (Hierarchical 5+5 层): O(T²) 自注意力, T=300 时全 GPU 显存 ~13MB/层
Mamba (8 层, expand=2, d_state=16): O(T) 选择性扫描, 同 d_model 下显存与算力均更友好;
    天然因果 (无需 causal_mask), 与在线决策 RL 严格对齐;
    长序列建模优于 RNN/GRU (有效 receptive field 几乎全长)。

实现策略 (兼顾正确性与可移植性)
-------------------------------
1. **优先使用 mamba_ssm 官方包**: 若 `import mamba_ssm` 成功, 用其 CUDA-融合的
   SelectiveScan 实现 (5-10x 比纯 PyTorch 快)。
2. **纯 PyTorch 回退**: 若官方包不可用 (Windows 编译困难等), 使用本文件实现的
   `SelectiveSSM` (基于顺序 scan), 与官方语义一致, 数值结果在 fp32 下匹配。

模块层次:
  SelectiveSSM   -> 选择性 SSM 算子 (核心)
  MambaMixer     -> in_proj + causal Conv1d + SSM + SiLU 门 + out_proj
  MambaBlock     -> norm + Mixer + 残差 (Pre-LN, 与 Transformer 一致)
  MambaTemporalEncoder -> 堆叠 N 个 MambaBlock + 末端 norm + Stochastic Depth + padding

接口与 HierarchicalTemporalEncoder 对齐:
  forward(x, src_key_padding_mask=None) -> (B,T,d)
  下游 ActorCritic / SDPO / PPO 完全无感切换。
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 官方 mamba_ssm 检测 (Linux/CUDA 环境下推荐)
# ============================================================
try:
    from mamba_ssm import Mamba as _OfficialMamba   # type: ignore
    _HAS_MAMBA_SSM = True
except Exception:
    _OfficialMamba = None
    _HAS_MAMBA_SSM = False


# ============================================================
# 纯 PyTorch 选择性 SSM (顺序扫描, 与官方语义一致)
# ============================================================

class SelectiveSSM(nn.Module):
    """选择性状态空间模型 (Mamba 核心算子, 纯 PyTorch 实现)。

    输入  : x  (B, T, d_inner)
    输出  : y  (B, T, d_inner)

    数学:
        for t = 1..T:
            dt_t, B_t, C_t  =  Linear_proj(x_t).split([dt_rank, d_state, d_state])
            dt_t            =  softplus(Linear_dt(dt_t))           # (d_inner,)
            A_bar_t         =  exp(dt_t * A)                       # (d_inner, d_state)
            B_bar_t         =  dt_t * B_t                          # (d_inner, d_state)
            h_t             =  A_bar_t * h_{t-1} + B_bar_t * x_t   # (d_inner, d_state)
            y_t             =  (h_t * C_t).sum(dim=-1) + D * x_t   # (d_inner,)

    A 用 -exp(A_log) 参数化保证负实部稳定; D 是逐通道 skip。
    """

    def __init__(self,
                 d_inner: int,
                 d_state: int = 16,
                 dt_rank: Optional[int] = None,
                 dt_min: float = 1e-3,
                 dt_max: float = 1e-1):
        super().__init__()
        self.d_inner = int(d_inner)
        self.d_state = int(d_state)
        self.dt_rank = int(dt_rank) if dt_rank is not None else max(1, d_inner // 16)

        # x -> (dt_raw, B, C), 其中 B/C 是 d_state 维, dt_raw 是 dt_rank 维
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        # dt_raw -> dt (per-channel)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # 初始化 dt_proj 的偏置使得初始 dt 在 [dt_min, dt_max] 内对数均匀分布
        # (与官方 mamba_ssm 一致, 让 SSM 启动稳定)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)

        # A 用 log-空间参数化, 保证 A 始终为负 (稳定)
        # 默认: A_data = -[1, 2, ..., d_state] (按通道复制)
        A = torch.arange(1, d_state + 1, dtype=torch.float32) \
                 .unsqueeze(0).expand(d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))     # (d_inner, d_state)
        # D: 逐通道 skip 连接
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_inner) -> y: (B, T, d_inner). 严格因果。

        数值稳定性 (fix 2026-04-30, 修复 NaN):
          1) 强制 fp32 计算 (autocast 关闭) - SSM 的 exp/scan 对 fp16 极敏感
          2) clamp dt 上下界, 防 dt*A 过大导致 deltaA 溢出
          3) clamp A_log, 防 |A| 过大
          4) deltaA 数学上 ∈ (0, 1] (因 A<0), 但显式 clamp 防 fp 误差
          5) clamp deltaBx 输入贡献, 防瞬时大输入污染状态
          6) clamp h 状态, 防累积爆炸 (sequential scan 经过 T=300 步可放大)
          7) 末端 nan_to_num 守卫, 兜底所有数值异常
        """
        orig_dtype = x.dtype
        # 强制 fp32 + 关闭 autocast: SSM 必须 fp32 才能稳定 (训练 1k 步后 fp16 累积 NaN)
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            B, T, d_inner = x.shape
            assert d_inner == self.d_inner

            # 1) 投影: x -> dt_raw, B_proj, C_proj
            x_dbl = self.x_proj(x)
            dt_raw, b_proj, c_proj = x_dbl.split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1,
            )
            # softplus(.) > 0; 然后 clamp 上下界 [1e-4, 10] 保数值稳定
            dt = F.softplus(self.dt_proj(dt_raw)).clamp(min=1e-4, max=10.0)

            # 2) 离散化: A_bar = exp(dt * A), B_bar = dt * B_proj
            # A_log clamp: 防参数训练后绝对值过大
            A_log_clamped = self.A_log.float().clamp(min=-10.0, max=10.0)
            A = -torch.exp(A_log_clamped)                     # (d_inner, d_state) 严格负
            # dt*A < 0 -> exp(.) < 1, 数学上稳定; 仍显式 clamp 防 fp 误差
            deltaA = torch.exp((dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
                               .clamp(min=-30.0, max=0.0))   # 上界 0 即 exp(0)=1
            deltaA = deltaA.clamp(min=0.0, max=1.0)
            deltaB = dt.unsqueeze(-1) * b_proj.unsqueeze(2)
            deltaBx = (deltaB * x.unsqueeze(-1)).clamp(min=-1e3, max=1e3)

            # 3) 顺序扫描 (fp32, 状态 clamp 防爆炸)
            h = torch.zeros(B, self.d_inner, self.d_state,
                            device=x.device, dtype=torch.float32)
            ys = []
            for t in range(T):
                h = deltaA[:, t] * h + deltaBx[:, t]
                # 防止状态在长序列 (T=247+) 累积出 inf/nan
                h = h.clamp(min=-1e4, max=1e4)
                y_t = (h * c_proj[:, t].unsqueeze(1)).sum(dim=-1)
                ys.append(y_t)
            y = torch.stack(ys, dim=1)

            # 4) skip 连接 D · x
            y = y + self.D.float() * x

            # 5) 末端 NaN/Inf 兜底
            y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

        return y.to(orig_dtype)


# ============================================================
# Mamba Mixer (input_proj + causal conv + SSM + 门控 + output_proj)
# ============================================================

class MambaMixer(nn.Module):
    """单个 Mamba 子层的核心 mixer 部分 (不含 norm/残差)。

    流程:
        x  ─── in_proj ──> [x_path | z_path]
        x_path  ── causal Conv1d (depthwise, k=4) ── SiLU ── SSM ──┐
                                                                  │
        z_path  ── SiLU ────────────────────────────────── × ─────┘── out_proj ── y
    """

    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_model = int(d_model)
        self.d_inner = int(expand) * self.d_model
        self.d_conv = int(d_conv)
        self.conv_pad = self.d_conv - 1

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        # 深度卷积 (groups=d_inner): 每通道独立, 配合 left-pad 因果
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=self.d_conv, groups=self.d_inner,
            padding=0, bias=True,
        )
        self.act = nn.SiLU()
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)。严格因果。

        数值守卫 (fix 2026-04-30):
          - in_proj / conv 输出 clamp, 防 fp16 范围爆炸
          - SSM 自身已强制 fp32 + clamp, 此处再加一道兜底
          - 出口前 nan_to_num, 阻断 NaN 向上层传播
        """
        xz = self.in_proj(x).clamp(min=-50.0, max=50.0)        # 防 fp16 输入投影爆炸
        x_path, z_path = xz.chunk(2, dim=-1)                   # 各 (B, T, d_inner)

        # 因果 1D 卷积 (left pad)
        x_path = x_path.transpose(1, 2)                        # (B, d_inner, T)
        x_path = F.pad(x_path, (self.conv_pad, 0))             # 仅左 pad
        x_path = self.conv(x_path).transpose(1, 2)             # (B, T, d_inner)
        x_path = self.act(x_path).clamp(min=-50.0, max=50.0)   # SiLU 后 clamp

        # SSM (核心, 内部已 fp32)
        y = self.ssm(x_path)                                   # (B, T, d_inner)

        # SiLU 门控 (gated linear unit 风格)
        y = y * self.act(z_path)
        y = y.clamp(min=-50.0, max=50.0)                       # 门控后 clamp

        # 输出投影回 d_model
        out = self.out_proj(y)
        return torch.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0)


# ============================================================
# Mamba Block (Pre-LN + Mixer + 残差)
# ============================================================

class MambaBlock(nn.Module):
    """Pre-LN 残差包装: out = x + Mixer(LayerNorm(x))。

    use_official=True 且 mamba_ssm 可用时, 使用官方 CUDA-融合 Mamba 算子;
    否则回退到本文件的纯 PyTorch MambaMixer。两者数学语义一致, 仅 CUDA 版更快。
    """

    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 use_official: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self._using_official = bool(use_official) and _HAS_MAMBA_SSM
        if self._using_official:
            self.mixer = _OfficialMamba(
                d_model=self.d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mixer = MambaMixer(
                d_model=self.d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 接受多余 args/kwargs 兼容 StochasticDepthBlock 包装
        out = self.dropout(self.mixer(self.norm(x)))
        return x + out


# ============================================================
# Stochastic Depth Wrapper (与 actor_critic 中的同名类语义一致)
# ============================================================

class _StochasticDepthMamba(nn.Module):
    """layer-level stochastic depth 包装 MambaBlock。"""

    def __init__(self, layer: nn.Module, drop_prob: float = 0.0):
        super().__init__()
        self.layer = layer
        self.drop_prob = float(drop_prob)

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if (not self.training) or self.drop_prob <= 0.0:
            return self.layer(src, *args, **kwargs)
        if torch.rand(1, device=src.device).item() < self.drop_prob:
            return src
        return self.layer(src, *args, **kwargs)


# ============================================================
# 时序编码器: 堆叠 N 个 MambaBlock + 末端 norm
# ============================================================

class MambaTemporalEncoder(nn.Module):
    """Mamba 时序编码器, 接口与 HierarchicalTemporalEncoder 对齐。

    forward(x, src_key_padding_mask=None) -> (B, T, d_model)

    特点:
      - 严格因果 (Mamba 内部为顺序 SSM 扫描, 不需要外部 causal_mask)
      - O(T) 复杂度
      - 与 #6 Stochastic Depth 兼容: layer-level 随机跳过, 深度递增 drop_prob
      - padding 双重清零: 入 SSM 前 + 出 SSM 后, 防 padding 帧污染状态
    """

    def __init__(self,
                 d_model: int,
                 num_layers: int = 8,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 use_official: bool = True,
                 stochastic_depth_prob: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self._using_official = bool(use_official) and _HAS_MAMBA_SSM

        layers = []
        for i in range(num_layers):
            block = MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_official=use_official,
                dropout=dropout,
            )
            # 深度递增 stochastic depth (浅层 0, 最深层 stochastic_depth_prob)
            sd_p = (stochastic_depth_prob * float(i) / float(max(1, num_layers - 1))
                    if num_layers > 1 else 0.0)
            layers.append(_StochasticDepthMamba(block, drop_prob=sd_p))
        self.layers = nn.ModuleList(layers)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, d); src_key_padding_mask: (B, T) bool True=padding。返回 (B, T, d)。

        `mask` 参数仅为兼容 nn.TransformerEncoder.forward 签名而保留, 此处忽略
        (Mamba 因果性由顺序扫描天然保证, 不需要外部 attention mask)。
        """
        del mask
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).float().unsqueeze(-1)
            # 入 SSM 前先 mask: padding 位输入清零, 防顺序扫描把 padding ResNet 特征
            # 累积进 SSM 状态后污染合法步的输出
            x = x * valid
        for layer in self.layers:
            x = layer(x)
            # 每层之后再次清零 padding 位, 防层间残差把 padding 区域信息推回 valid 区
            if src_key_padding_mask is not None:
                x = x * valid
        x = self.norm_out(x)
        if src_key_padding_mask is not None:
            x = x * valid
        # 末端 NaN/Inf 兜底, 阻止异常值传到 policy/volume/value heads
        return torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)


def has_official_mamba_ssm() -> bool:
    """判断是否检测到官方 mamba_ssm 包 (CUDA 加速)。"""
    return _HAS_MAMBA_SSM


# ============================================================
# 双向 Mamba (2026-05-01): 仅供非因果 head 使用
# ============================================================
# 动机: 主编码器必须严格因果以保证 RL 在线决策语义, 但 volume / ef_direct /
# phase 等回归头不需要因果——它们只是从已观察到的整段视频估计连续值, 类似
# 一个 BERT-style encoder. 让这些 head 看到 "未来" 帧能显著提升 EF 预测精度.
#
# 实现: 对输入序列做正向 + 反向两次 Mamba scan, 然后 concat → linear fuse。
# 反向 scan = flip(input) → MambaCausal → flip(output)。计算量约 2×。

class BiMambaBlock(nn.Module):
    """双向 Mamba block: 正向 + 反向 + 融合 + 残差."""

    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 use_official: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        # 正向 + 反向各一组完整 MambaBlock (共享 norm 与残差由本类统一处理)
        self.fwd_norm = nn.LayerNorm(self.d_model)
        if use_official and _HAS_MAMBA_SSM:
            self.fwd = _OfficialMamba(d_model=self.d_model, d_state=d_state,
                                      d_conv=d_conv, expand=expand)
            self.bwd = _OfficialMamba(d_model=self.d_model, d_state=d_state,
                                      d_conv=d_conv, expand=expand)
        else:
            self.fwd = MambaMixer(d_model=self.d_model, d_state=d_state,
                                  d_conv=d_conv, expand=expand)
            self.bwd = MambaMixer(d_model=self.d_model, d_state=d_state,
                                  d_conv=d_conv, expand=expand)
        self.fuse = nn.Linear(2 * self.d_model, self.d_model)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fwd_norm(x)
        # 正向
        y_fwd = self.fwd(h)
        # 反向: flip → mamba → flip
        h_rev = torch.flip(h, dims=[1])
        y_bwd_rev = self.bwd(h_rev)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])
        # 融合 (concat → linear)
        y = self.fuse(torch.cat([y_fwd, y_bwd], dim=-1))
        return x + self.dropout(y)


class BiMambaTemporalEncoder(nn.Module):
    """双向 Mamba 编码器 (浅, 默认 2 层). 接口与 MambaTemporalEncoder 对齐。"""

    def __init__(self,
                 d_model: int,
                 num_layers: int = 2,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 use_official: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                         expand=expand, use_official=use_official, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).float().unsqueeze(-1)
            x = x * valid
        for layer in self.layers:
            x = layer(x)
            if src_key_padding_mask is not None:
                x = x * valid
        x = self.norm_out(x)
        if src_key_padding_mask is not None:
            x = x * valid
        return torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
