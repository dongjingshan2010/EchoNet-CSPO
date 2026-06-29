"""R(2+1)D spatiotemporal encoder (Ouyang et al., Nature 2020).

Memory optimizations (2026-05-31)
----------------------------------
Fix 1 - frame_size 224->112 (configs/default.yaml)
    Paper uses 112x112. Wrong 224 inflates stem activation 4x spatially.
Fix 2 - clip_len=32 temporal subsampling
    3D conv memory scales linearly with T. T=150 vs paper T=32 = 4.7x extra.
    Subsample T->32 INSIDE the encoder before 3D conv; interpolate output back to T.
Fix 3 - gradient checkpointing on stem + layer1
    Recompute stem/layer1 during backward instead of storing activations (~50% saving).
Fix 4 - contiguous() AFTER subsampling, not before
    permute() returns a view (no copy); slice first, then contiguous on the small tensor.

Combined effect: stem activation 919 MB -> ~49 MB (~19x reduction).
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _load_r2plus1d_backbone(pretrained):
    """Load r2plus1d_18, strip avgpool/fc. Returns (stem, layer1-4).

    Compatible with both old (pretrained=True) and new (weights=...) torchvision API.
    """
    try:
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        m = r2plus1d_18(weights=weights)
    except (ImportError, AttributeError):
        try:
            from torchvision.models.video import r2plus1d_18 as _fn
            m = _fn(pretrained=pretrained)
        except Exception as exc:
            raise ImportError(
                "R21DEncoder requires torchvision r2plus1d_18. "
                "Install via: pip install torchvision"
            ) from exc
    return m.stem, m.layer1, m.layer2, m.layer3, m.layer4


class R21DEncoder(nn.Module):
    """R(2+1)D spatiotemporal encoder replacing Mamba as temporal backbone.

    Parameters
    ----------
    d_model        : output feature dim (must match ActorCritic.d_model)
    pretrained     : load Kinetics-400 pretrained weights
    dropout        : dropout probability
    clip_len       : frames fed to 3D conv (temporal subsampling target, paper=32)
                     None disables subsampling (high memory risk with T=150)
    use_checkpoint : gradient checkpointing on stem+layer1 to trade compute for memory
    """

    BACKBONE_FEAT_DIM = 512   # r2plus1d_18 layer4 output channels

    def __init__(self,
                 d_model=256,
                 pretrained=True,
                 dropout=0.1,
                 clip_len=32,
                 use_checkpoint=True):
        super().__init__()
        self.d_model = d_model
        self.clip_len = int(clip_len) if clip_len is not None else None
        self.use_checkpoint = bool(use_checkpoint)

        # R(2+1)D backbone (avgpool + fc removed)
        stem, layer1, layer2, layer3, layer4 = _load_r2plus1d_backbone(pretrained)
        self.stem   = stem    # (B,3,T,H,W) -> (B,64,T,H/2,W/2)
        self.layer1 = layer1  # -> (B,64,T,H/4,W/4)      temporal stride=1
        self.layer2 = layer2  # -> (B,128,T/2,H/8,W/8)   temporal stride=2
        self.layer3 = layer3  # -> (B,256,T/4,H/16,W/16) temporal stride=2
        self.layer4 = layer4  # -> (B,512,T/8,H/32,W/32) temporal stride=2

        # Pool spatial dims only, keep temporal
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Project 512 -> d_model
        self.proj = nn.Sequential(
            nn.Linear(self.BACKBONE_FEAT_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

        total = sum(p.numel() for p in self.parameters())
        bb = sum(p.numel() for m in
                 [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
                 for p in m.parameters())
        print(
            f'[R21DEncoder] d_model={d_model} clip_len={clip_len} '
            f'checkpoint={use_checkpoint} pretrained={pretrained} '
            f'params={total/1e6:.1f}M (backbone={bb/1e6:.1f}M)'
        )

    # ------------------------------------------------------------------ #
    def set_frozen(self, frozen):
        """Freeze/unfreeze backbone (interface matches ResNetBackbone)."""
        for m in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters():
                p.requires_grad = not frozen

    # private helpers for gradient checkpointing
    def _stem_fwd(self, x):   return self.stem(x)
    def _layer1_fwd(self, x): return self.layer1(x)

    # ------------------------------------------------------------------ #
    def forward(self, frames, mask=None, src_key_padding_mask=None):
        """Extract per-frame spatiotemporal features.

        Args
        ----
        frames : (B, T, C, H, W)  normalized video frames
        mask   : (B, T) bool, True=valid (optional; zeros out padding positions)

        Returns
        -------
        z : (B, T, d_model)
        """
        B, T, C, H, W = frames.shape

        # Fix 4: permute returns a non-contiguous VIEW (zero copy).
        # We subsample BEFORE calling .contiguous(), so the later copy
        # operates on the smaller (clip_len) tensor, not the full T.
        x = frames.permute(0, 2, 1, 3, 4)   # (B,C,T,H,W) - view, no copy

        # Fix 2: temporal subsampling T -> clip_len
        if self.clip_len is not None and T > self.clip_len:
            t_stride = max(1, T // self.clip_len)
            x = x[:, :, ::t_stride, :, :]   # slice view (B,C,~clip_len,H,W)

        x = x.contiguous()   # copy only the small subsampled tensor

        # Fix 3: gradient checkpointing on stem + layer1 (most expensive layers)
        # use_reentrant=False avoids issues with in-place ops (PyTorch >= 2.0)
        if self.use_checkpoint and self.training:
            x = checkpoint(self._stem_fwd,   x, use_reentrant=False)
            x = checkpoint(self._layer1_fwd, x, use_reentrant=False)
        else:
            x = self.stem(x)
            x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # (B,512,T_feat,H',W')

        T_feat = x.shape[2]

        # Spatial pool: (B,512,T_feat,H',W') -> (B,512,T_feat)
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)

        # Temporal interpolation T_feat -> T (restore for RL policy heads)
        if T_feat != T:
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)

        # Project 512 -> d_model
        x = x.permute(0, 2, 1)          # (B,T,512)
        z = self.dropout(self.proj(x))   # (B,T,d_model)

        if mask is not None:
            z = z * mask.float().unsqueeze(-1)

        return z
