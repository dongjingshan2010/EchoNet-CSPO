"""MambaPolicyNet — 三动作 RL 策略网络（EchoNet-CSPO-mamba3ActorPriod-R21D-derectEF）。

架构
────
  ResNet backbone  → 逐帧空间特征  (B, T, feat_dim)
  Linear proj      → 降维投影      (B, T, d_model)
  SinCos PE        → 位置编码
  MambaEncoder     → 因果时序编码  (B, T, d_model)   [policy/value 头用]
  BiMamba (可选)   → 双向时序编码  (B, T, d_model)   [volume 头用，看全局]

  policy_head  (B,T,3) : 三类 logits
                          0 = skip（跳过）
                          1 = select_as_ED（标记为舒张末期帧，高容积）
                          2 = select_as_ES（标记为收缩末期帧，低容积）
  value_head   (B,T,1) : Critic V(s_t)
  volume_head  (B,T)   : 逐帧 LV 容积预测 (mL), MeanCenteredVolumeHead

EF_mamba 计算
─────────────
  EDV = mean( volume[t] for t where action[t]==1 )  → 舒张末期容积
  ESV = mean( volume[t] for t where action[t]==2 )  → 收缩末期容积
  EF  = (EDV - ESV) / EDV × 100%

  与二值版本的区别：选帧直接决定 EDV/ESV 的计算角色，
  而非依赖全局 max/min 聚合，因此选帧动作具有真正的独占价值。

设计说明
────────
* policy_head 从 2 类改为 3 类，其余结构与 mambaPriod 完全相同。
* BiMamba 默认启用：volume_head 从双向特征学习。
* forward() 返回三元组 (logits, volume, value)，接口不变。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNetBackbone
from .mamba_encoder import (
    MambaTemporalEncoder,
    BiMambaTemporalEncoder,
    has_official_mamba_ssm,
)
from .heads import MeanCenteredVolumeHead, ModelEMA


# ─────────────────────────────────────────────────────────────────────────────
# 正弦余弦位置编码
# ─────────────────────────────────────────────────────────────────────────────

class SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ─────────────────────────────────────────────────────────────────────────────
# 分层 Transformer（可选，向后兼容 encoder_type=hierarchical）
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalTemporalEncoder(nn.Module):
    """局部窗口 + 全局因果 Transformer。"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout,
                 num_local_layers=4, num_global_layers=4,
                 window_size=8, max_windows=64, fuse_mode='add'):
        super().__init__()
        self.window_size = int(window_size)
        self.fuse_mode   = fuse_mode
        self.d_model     = int(d_model)

        def _enc(n):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
            )
            return nn.TransformerEncoder(layer, n, enable_nested_tensor=False)

        self.local_enc  = _enc(int(num_local_layers))
        self.global_enc = _enc(int(num_global_layers))
        self.window_pos = SinCosPositionalEncoding(d_model, max_len=int(max_windows) + 16)
        self.fuse_proj  = nn.Linear(d_model * 2, d_model) if fuse_mode == 'concat' else None
        self.fuse_norm  = nn.LayerNorm(d_model)

    @staticmethod
    def _causal_mask(L, device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        del mask
        B, T, d = x.shape
        device  = x.device
        k       = self.window_size
        pad_len = (k - (T % k)) % k
        if pad_len > 0:
            x   = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            kpm = (src_key_padding_mask if src_key_padding_mask is not None
                   else torch.zeros(B, T, dtype=torch.bool, device=device))
            kpm = F.pad(kpm, (0, pad_len), value=True)
        else:
            kpm = (src_key_padding_mask if src_key_padding_mask is not None
                   else torch.zeros(B, T, dtype=torch.bool, device=device))
        T_pad = T + pad_len
        nw    = T_pad // k

        x_win     = x.view(B, nw, k, d).reshape(B * nw, k, d)
        kpm_win   = kpm.view(B, nw, k)
        kpm_local = kpm_win.reshape(B * nw, k)
        all_pad   = kpm_local.all(dim=1)
        if all_pad.any():
            kpm_local = kpm_local.clone(); kpm_local[all_pad, 0] = False
        local_out = self.local_enc(
            x_win, mask=self._causal_mask(k, device),
            src_key_padding_mask=kpm_local,
        ).view(B, nw, k, d)

        vis            = (~kpm_win).float().unsqueeze(-1)
        window_summary = (local_out * vis).sum(dim=2) / vis.sum(dim=2).clamp(min=1.0)
        window_summary = self.window_pos(window_summary)
        kpm_global     = kpm_win.all(dim=2)
        all_pad_b      = kpm_global.all(dim=1)
        if all_pad_b.any():
            kpm_global = kpm_global.clone(); kpm_global[all_pad_b, 0] = False
        global_out = self.global_enc(
            window_summary, mask=self._causal_mask(nw, device),
            src_key_padding_mask=kpm_global,
        )
        global_bc = global_out.unsqueeze(2).expand(-1, -1, k, -1)
        if self.fuse_mode == 'concat':
            fused = self.fuse_proj(torch.cat([local_out, global_bc], dim=-1))
        else:
            fused = local_out + global_bc
        fused = self.fuse_norm(fused).reshape(B, T_pad, d)
        if pad_len > 0:
            fused = fused[:, :T, :]
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# 时序编码器工厂
# ─────────────────────────────────────────────────────────────────────────────

def _build_temporal_encoder(d: int, cfg) -> nn.Module:
    m_cfg = getattr(cfg, 'model', None)
    h_cfg = getattr(m_cfg, 'hierarchical', None) if m_cfg else None
    enc_type = str(
        getattr(m_cfg, 'encoder_type', '') if m_cfg else ''
    ).lower().strip()
    if not enc_type:
        enc_type = ('hierarchical'
                    if (h_cfg is not None and bool(getattr(h_cfg, 'enabled', False)))
                    else 'flat')

    if enc_type == 'mamba':
        ma = getattr(m_cfg, 'mamba', None)
        enc = MambaTemporalEncoder(
            d_model=d,
            num_layers=int(getattr(ma, 'num_layers', 4)) if ma else 4,
            d_state=int(getattr(ma, 'd_state', 16)) if ma else 16,
            d_conv=int(getattr(ma, 'd_conv', 4)) if ma else 4,
            expand=int(getattr(ma, 'expand', 2)) if ma else 2,
            dropout=float(cfg.model.dropout),
            use_official=bool(getattr(ma, 'use_official', True)) if ma else True,
            stochastic_depth_prob=float(
                getattr(ma, 'stochastic_depth_prob', 0.0)) if ma else 0.0,
        )
        impl = 'mamba_ssm' if (has_official_mamba_ssm() and
                               bool(getattr(ma, 'use_official', True) if ma else True)) \
               else 'pure-torch'
        print(f'[encoder] MambaTemporalEncoder ({impl})')
        return enc

    if enc_type == 'hierarchical' and h_cfg is not None:
        max_steps   = int(cfg.data.max_steps)
        win         = int(getattr(h_cfg, 'window_size', 8))
        max_windows = max(8, (max_steps + win - 1) // win + 8)
        return HierarchicalTemporalEncoder(
            d_model=d, nhead=int(cfg.model.nhead),
            dim_feedforward=int(cfg.model.dim_ff),
            dropout=float(cfg.model.dropout),
            num_local_layers=int(getattr(h_cfg, 'num_local_layers', 4)),
            num_global_layers=int(getattr(h_cfg, 'num_global_layers', 4)),
            window_size=win, max_windows=max_windows,
            fuse_mode=str(getattr(h_cfg, 'fuse_mode', 'add')),
        )

    # flat Causal Transformer（fallback）
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d, nhead=int(cfg.model.nhead),
        dim_feedforward=int(cfg.model.dim_ff), dropout=float(cfg.model.dropout),
        batch_first=True, activation='gelu', norm_first=True,
    )
    return nn.TransformerEncoder(enc_layer, int(cfg.model.num_layers),
                                 enable_nested_tensor=False)


# ─────────────────────────────────────────────────────────────────────────────
# MambaPolicyNet — 三动作策略网络
# ─────────────────────────────────────────────────────────────────────────────

class MambaPolicyNet(nn.Module):
    """三动作 Mamba RL 策略网络。

    forward(frames, mask) → (logits, volume, value)

    logits  : (B, T, 3)  policy logits  {0=skip, 1=ED, 2=ES}
    volume  : (B, T)     逐帧 LV 容积 (mL)
    value   : (B, T)     Critic V(s_t)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── 空间骨干 ─────────────────────────────────────────────────────
        self.backbone = ResNetBackbone(cfg.model.backbone, cfg.model.pretrained)
        d       = int(cfg.model.d_model)
        dropout = float(cfg.model.dropout)

        self.proj = nn.Linear(self.backbone.feat_dim, d)
        self.pos  = SinCosPositionalEncoding(d, max_len=int(cfg.data.max_steps) + 16)

        # ── 因果时序编码器（policy + value 用）──────────────────────────
        self.enc              = _build_temporal_encoder(d, cfg)
        self._is_mamba        = isinstance(self.enc, MambaTemporalEncoder)
        self._is_hierarchical = isinstance(self.enc, HierarchicalTemporalEncoder)

        # ── BiMamba（volume 头专用，双向，看全局）───────────────────────
        imp = getattr(cfg, 'improvements', None)
        self._bidi_enabled = bool(getattr(imp, 'bidi_encoder_enabled', True)) if imp else True
        if self._bidi_enabled and self._is_mamba:
            bidi_layers = int(getattr(imp, 'bidi_num_layers', 2)) if imp else 2
            ma          = getattr(cfg.model, 'mamba', None)
            self.bidi_enc = BiMambaTemporalEncoder(
                d_model=d, num_layers=bidi_layers,
                d_state=int(getattr(ma, 'd_state', 16)) if ma else 16,
                d_conv=int(getattr(ma, 'd_conv', 4)) if ma else 4,
                expand=int(getattr(ma, 'expand', 2)) if ma else 2,
                dropout=dropout,
                use_official=bool(getattr(ma, 'use_official', True)) if ma else True,
            )
            print(f'[encoder] BiMambaTemporalEncoder ({bidi_layers} layers) for volume_head')
        else:
            self.bidi_enc = None

        # ── 输出头 ───────────────────────────────────────────────────────
        def _head(out_dim):
            return nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, out_dim),
            )

        # ★ 三动作：policy_head 输出 3 类 logits
        self.policy_head = _head(3)    # 0=skip, 1=ED, 2=ES
        self.value_head  = _head(1)    # Critic V(s_t)

        # ── v5(移植自 mamba3Actor) 反塌缩：skip 偏置初始化 ──────────────
        # 末层权重置零 + bias=[skip_bias,0,0]：初始 logits≈bias，softmax 偏向 skip，
        # 策略从"稀疏选帧"起步（b=3.5 → 初始选帧率~6%），避免早期熵/噪声驱动的过选塌缩。
        skip_bias = float(getattr(cfg.model, 'policy_skip_bias_init', 0.0))
        if skip_bias != 0.0:
            with torch.no_grad():
                final = self.policy_head[-1]          # 最后一层 Linear(d, 3)
                nn.init.zeros_(final.weight)
                final.bias.copy_(torch.tensor([skip_bias, 0.0, 0.0]))

        # 容积头（BiMamba 双向特征）
        self._mean_centered_vol = bool(getattr(imp, 'mean_centered_volume', True)) if imp else True
        if self._mean_centered_vol:
            self.volume_head = MeanCenteredVolumeHead(
                d_model=d, dropout=dropout,
                vol_mean=float(getattr(imp, 'volume_mean', 100.0)) if imp else 100.0,
                vol_scale=float(getattr(imp, 'volume_scale', 80.0)) if imp else 80.0,
            )
        else:
            self.volume_head = _head(1)
        self.vol_scale = float(cfg.model.max_volume_ml)

        n = sum(p.numel() for p in self.parameters())
        print(f'[MambaPolicyNet-3Action] params={n/1e6:.1f}M  '
              f'bidi={self._bidi_enabled}  '
              f'vol_mode={"centered" if self._mean_centered_vol else "sigmoid"}  '
              f'policy_classes=3')

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _causal_mask(T: int, device):
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def set_backbone_frozen(self, frozen: bool):
        self.backbone.set_frozen(frozen)

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """ResNet 逐帧特征（供 PPO 内循环缓存复用）。"""
        return self.backbone(frames)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, frames: torch.Tensor, mask: torch.Tensor):
        """frames: (B,T,C,H,W); mask: (B,T) bool True=有效。

        Returns: (logits, volume, value)
          logits : (B, T, 3)
          volume : (B, T)
          value  : (B, T)
        """
        feat = self.extract_features(frames)
        return self.forward_from_feat(feat, mask)

    def forward_from_feat(self, feat: torch.Tensor, mask: torch.Tensor):
        """从缓存骨干特征继续前向（PPO 内循环加速）。"""
        x   = self.proj(feat)
        x   = self.pos(x)
        kpm = ~mask

        # ── 因果编码（policy + value）───────────────────────────────────
        if self._is_mamba:
            z = self.enc(x, src_key_padding_mask=kpm)
        elif self._is_hierarchical:
            z = self.enc(x, src_key_padding_mask=kpm)
        else:
            z = self.enc(x, mask=self._causal_mask(x.size(1), x.device),
                         src_key_padding_mask=kpm)

        logits = self.policy_head(z)              # (B, T, 3)
        value  = self.value_head(z).squeeze(-1)   # (B, T)

        # ── 双向编码（volume 头，利用全局上下文）────────────────────────
        z_nc = self.bidi_enc(z, src_key_padding_mask=kpm) if self.bidi_enc is not None else z

        if self._mean_centered_vol:
            volume = self.volume_head(z_nc)
        else:
            volume = torch.sigmoid(self.volume_head(z_nc).squeeze(-1)) * self.vol_scale

        return logits, volume, value


# 向后兼容别名
ActorCritic = MambaPolicyNet
DualRouteActorCritic = MambaPolicyNet
