"""网络头组件 (2026-05-01 架构优化).

5 个新组件, 与 ActorCritic 共同装配:
  - AttentionPool          : 可学习注意力池化, 替代 masked-mean 用于全局特征
  - MeanCenteredVolumeHead : v = vol_mean + vol_scale·tanh(MLP), 对齐 LV 容积分布
  - EFRegressionHead       : 全局池化 EF 直接回归 + 集成
  - PhaseClassifierHead    : 每帧 3 类 (pre/systole/diastole) 辅助任务
  - ModelEMA               : Polyak averaging 影子权重, 评估期使用

向后兼容: 全部组件可通过 cfg.improvements.* 字段开关, 默认开启获得最好性能。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# AttentionPool — 可学习注意力池化
# ============================================================

class AttentionPool(nn.Module):
    """pool = sum(softmax(z·q / sqrt(d)) · z), 单参数 q 全局共享。

    相对 masked-mean 的优势: 模型自适应聚焦在 ED/ES 段或心动周期峰值,
    避免 mean 把所有有效帧等权稀释。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scale = math.sqrt(float(d_model))

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """z: (B,T,d); mask: (B,T) bool True=valid -> (B,d)。"""
        scores = (z @ self.q) / self.scale                         # (B, T)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)       # (B, T, 1)
        return (z * weights).sum(dim=1)                            # (B, d)


# ============================================================
# MeanCenteredVolumeHead — 替代 sigmoid·max_volume_ml
# ============================================================

class MeanCenteredVolumeHead(nn.Module):
    """v = vol_mean + vol_scale · tanh(MLP(z))。

    动机: EchoNet-Dynamic LV 容积分布中心 ~100mL, 范围 ~[20, 180]mL,
        sigmoid·300 浪费一半动态范围; tanh 在 [-3, 3] 区间近似线性, 输出
        对预测分布的中心和方差更敏感。
    """

    def __init__(self, d_model: int, dropout: float,
                 vol_mean: float = 100.0, vol_scale: float = 80.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 1),
        )
        self.vol_mean = float(vol_mean)
        self.vol_scale = float(vol_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B,T,d) -> (B,T) volume in mL。"""
        out = self.head(z).squeeze(-1)
        return self.vol_mean + self.vol_scale * torch.tanh(out)


# ============================================================
# EFRegressionHead — 直接 EF 回归 (与选帧 EF 集成)
# ============================================================

class EFRegressionHead(nn.Module):
    """从全局池化的编码表征端到端回归 EF (%)。

    与原 EF 估计公式 (max-min)/max·100 互补:
      - 选帧 EF: 关键帧定位 + 容积预测两步, 误差耦合
      - 直接 EF: 端到端从全局表征学, 不同归纳偏置, 适合集成

    输出范围 [10%, 90%], 用 ef_mean=50, ef_scale=40 + tanh 限位.
    """

    def __init__(self, d_model: int, dropout: float,
                 ef_mean: float = 50.0, ef_scale: float = 40.0,
                 use_attention_pool: bool = True):
        super().__init__()
        self.use_attn = bool(use_attention_pool)
        if self.use_attn:
            self.attn_pool = AttentionPool(d_model)
        else:
            self.attn_pool = None
        self.pool_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(d_model // 2, 1),
        )
        self.ef_mean = float(ef_mean)
        self.ef_scale = float(ef_scale)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """z: (B,T,d); mask: (B,T) bool -> (B,) EF (%)。"""
        if self.use_attn:
            pooled = self.attn_pool(z, mask)
        else:
            m = mask.float().unsqueeze(-1)
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = (z * m).sum(dim=1) / denom
        pooled = self.pool_norm(pooled)
        out = self.mlp(pooled).squeeze(-1)
        return self.ef_mean + self.ef_scale * torch.tanh(out)


# ============================================================
# PhaseClassifierHead — 时相 3 分类辅助任务
# ============================================================

class PhaseClassifierHead(nn.Module):
    """每帧 3 分类: 0=pre / 1=systole / 2=diastole。

    伪标签由 ED/ES idx 启发式构造 (utils/labels.py::build_phase_labels)。
    作为辅助监督信号, 强迫编码器学到容积变化方向, 与关键帧选择互补。
    """

    def __init__(self, d_model: int, dropout: float, n_classes: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, int(n_classes)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


# ============================================================
# ModelEMA — Polyak averaging shadow weights
# ============================================================

class ModelEMA:
    """指数滑动平均影子权重. 评估期 apply_to(model) -> evaluate -> restore(model)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            sv = self.shadow[k]
            if v.dtype.is_floating_point and sv.dtype == v.dtype and sv.shape == v.shape:
                sv.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict(self.shadow, strict=True)
        except RuntimeError:
            model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module):
        if self.backup is not None:
            try:
                model.load_state_dict(self.backup, strict=True)
            except RuntimeError:
                model.load_state_dict(self.backup, strict=False)
            self.backup = None
