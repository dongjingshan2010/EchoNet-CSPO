"""R21DCyclicEFHead — 基于心动周期检测的多周期 R21D EF 预测头。

核心设计
────────
1. 不依赖 ED/ES 帧位置标签。
2. 从 Mamba volume_head 预测的逐帧容积曲线中，用峰值检测找到心动周期边界。
3. 对每个检测到的周期提取连续帧段，插值到 clip_len，送入 R21D backbone。
4. 对所有周期的 EF 预测做质量加权平均：
   quality ∝ exp(-((span_len - target_len) / target_len)^2)
   接近目标长度（约一个完整心动周期）的 span 权重更高。
5. 冷启动策略：前 cycle_detect_warmup updates 使用均匀分割（不依赖 volume
   质量），之后切换到峰值检测，保证 R21D 在 Mamba volumes 不稳定时仍能收到
   有效监督信号。

返回 CyclicEFResult(ef_final, ef_formula, ef_direct, edv, esv, cycle_count)
  ef_final    : (B,) 质量加权平均 EF（主输出）
  ef_formula  : (B,) 各周期公式路径 EF 的加权均值
  ef_direct   : (B,) 各周期直接回归 EF 的加权均值
  edv         : (B,) 各周期 EDV 的加权均值（若公式头关闭则 None）
  esv         : (B,) 各周期 ESV 的加权均值（若公式头关闭则 None）
  cycle_count : (B,) 实际检测到的周期数
"""
import math
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .r2plus1d_encoder import _load_r2plus1d_backbone

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


CyclicEFResult = namedtuple(
    'CyclicEFResult',
    ['ef_final', 'ef_formula', 'ef_direct', 'edv', 'esv', 'cycle_count'],
)

# 向后兼容
EFResult = namedtuple('EFResult', ['ef_final', 'ef_formula', 'ef_direct', 'edv', 'esv'])


# ─────────────────────────────────────────────────────────────────────────────
# 心动周期检测（纯 numpy / scipy，在 CPU 上运行，调用时 volumes 已 detach）
# ─────────────────────────────────────────────────────────────────────────────

def _detect_cycles_uniform(valid_len: int,
                            min_cycle: int,
                            max_cycles: int) -> List[Tuple[int, int]]:
    """均匀分割：把有效帧等分为 max_cycles 段，用于冷启动阶段。"""
    n = min(max_cycles, max(1, valid_len // max(min_cycle, 1)))
    seg = valid_len // n
    if seg < min_cycle:
        return [(0, valid_len - 1)]
    spans = []
    for i in range(n):
        t0 = i * seg
        t1 = min(valid_len - 1, t0 + seg - 1)
        if t1 - t0 + 1 >= min_cycle:
            spans.append((t0, t1))
    return spans or [(0, valid_len - 1)]


def _detect_cycles_peaks(vol_np: np.ndarray,
                          min_cycle: int,
                          max_cycles: int,
                          sigma: float,
                          prominence_ratio: float) -> List[Tuple[int, int]]:
    """高斯平滑 + 峰值检测：找局部极大值（EDV候选），相邻峰之间 = 一个心动周期。

    回退策略: 若检测到的峰 < 2，返回 None（调用方切换到均匀分割）。
    """
    if not _SCIPY_OK or len(vol_np) < min_cycle * 2:
        return None

    # 高斯平滑降噪
    smooth = gaussian_filter1d(vol_np.astype(np.float64), sigma=max(1.0, sigma))

    v_range     = float(smooth.max() - smooth.min())
    prominence  = max(1.0, v_range * prominence_ratio)

    peaks, _ = find_peaks(
        smooth,
        distance=max(1, min_cycle // 2),
        prominence=prominence,
    )

    if len(peaks) < 2:
        return None  # 信号不足，fallback

    spans = []
    for i in range(len(peaks) - 1):
        t0, t1 = int(peaks[i]), int(peaks[i + 1])
        if t1 - t0 + 1 >= min_cycle:
            spans.append((t0, t1))
            if len(spans) >= max_cycles:
                break

    return spans if spans else None


def detect_cardiac_cycles(volumes_b: torch.Tensor,
                           valid_len: int,
                           min_cycle: int,
                           max_cycles: int,
                           sigma: float,
                           prominence_ratio: float,
                           use_peaks: bool,
                           margin: int = 0) -> List[Tuple[int, int]]:
    """对单样本的容积曲线检测心动周期边界，并向两端扩展 margin 帧。

    Parameters
    ----------
    volumes_b   : (T,) detached CPU/GPU tensor
    valid_len   : 有效帧数（去掉 padding）
    use_peaks   : True = 峰值检测，False = 均匀分割（冷启动）
    margin      : 每个 span 向两端各扩展的帧数（0=不扩展）。
                  扩展后 t0 = max(0, t0-margin)，t1 = min(valid_len-1, t1+margin)。
                  作用：让 R21D 在 ED 边界帧处获得对称的时间上下文，
                  避免 ED 帧处于感受野边缘导致特征提取不完整。

    Returns
    -------
    list of (t0, t1) — 心动周期 span 列表（最少 1 个，已应用 margin 扩展）
    """
    if use_peaks and valid_len >= min_cycle * 2:
        vol_np = volumes_b[:valid_len].detach().float().cpu().numpy()
        spans  = _detect_cycles_peaks(vol_np, min_cycle, max_cycles,
                                       sigma, prominence_ratio)
        if spans is not None:
            return _apply_margin(spans, valid_len, margin)

    # 均匀分割（冷启动或峰值检测失败时）
    return _apply_margin(
        _detect_cycles_uniform(valid_len, min_cycle, max_cycles),
        valid_len, margin,
    )


def _apply_margin(spans: List[Tuple[int, int]],
                  valid_len: int,
                  margin: int) -> List[Tuple[int, int]]:
    """将每个 span 向两端各扩展 margin 帧，并夹入 [0, valid_len-1]。"""
    if margin <= 0:
        return spans
    return [
        (max(0, t0 - margin), min(valid_len - 1, t1 + margin))
        for (t0, t1) in spans
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 周期质量权重：span 长度接近 target_cycle_frames 时权重最高
# ─────────────────────────────────────────────────────────────────────────────

def _cycle_quality_weight(span_len: int, target: int) -> float:
    """Gaussian quality: exp(-((span - target) / target)^2)。"""
    ratio = (float(span_len) - float(target)) / float(max(target, 1))
    return float(np.exp(-(ratio ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# R21DCyclicEFHead
# ─────────────────────────────────────────────────────────────────────────────

class R21DCyclicEFHead(nn.Module):
    """多心动周期 R21D EF 预测头。

    Parameters
    ----------
    pretrained            : 是否加载 Kinetics-400 预训练权重
    clip_len              : 每段 span 三线性插值到的帧数（32 为论文最优）
    dropout               : Dropout 概率
    use_checkpoint        : stem+layer1 使用梯度检查点（节省显存）
    ef_mean / ef_scale    : EF_direct 输出范围 = ef_mean ± ef_scale
    use_formula_head      : 是否启用公式路径（EDV/ESV → EF_formula）
    vol_edv/esv_mean/scale: EDV/ESV 的先验均值和缩放（tanh 输出范围约束）
    ensemble_alpha_init   : α 初始值（公式路径和直接路径的集成权重）
    min_cycle_frames      : 最短有效周期（更短则跳过）
    max_cycles            : 每样本最多使用的周期数
    target_cycle_frames   : 目标周期长（质量加权基准）
    cycle_quality_weighting: 是否按质量加权（False = 简单均值）
    cycle_detect_warmup   : 前 N updates 均匀分割，之后峰值检测
    cycle_smooth_sigma    : 高斯平滑 σ
    cycle_prominence_ratio: 峰值显著度阈值（volume_range 的比例）
    """

    def __init__(
        self,
        pretrained: bool = True,
        clip_len: int = 32,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
        ef_mean: float = 55.0,
        ef_scale: float = 35.0,
        use_formula_head: bool = False,      # 默认关闭：EDV/ESV 无直接监督，公式路径梯度信号弱
        vol_edv_mean: float = 100.0,
        vol_edv_scale: float = 80.0,
        vol_esv_mean: float = 45.0,
        vol_esv_scale: float = 40.0,
        ensemble_alpha_init: float = 0.5,
        min_cycle_frames: int = 20,
        max_cycles: int = 4,
        target_cycle_frames: int = 64,
        cycle_quality_weighting: bool = True,
        cycle_detect_warmup: int = 2000,
        cycle_smooth_sigma: float = 3.0,
        cycle_prominence_ratio: float = 0.15,
        span_margin: int = 6,                # ED 边界扩展帧数：给 R21D 提供对称时间上下文
    ):
        super().__init__()
        self.clip_len               = int(clip_len)
        self.use_checkpoint         = bool(use_checkpoint)
        self.ef_mean                = float(ef_mean)
        self.ef_scale               = float(ef_scale)
        self.use_formula_head       = bool(use_formula_head)
        self.min_cycle_frames       = max(1, int(min_cycle_frames))
        self.max_cycles             = max(1, int(max_cycles))
        self.target_cycle_frames    = max(1, int(target_cycle_frames))
        self.cycle_quality_weighting = bool(cycle_quality_weighting)
        self.cycle_detect_warmup    = int(cycle_detect_warmup)
        self.cycle_smooth_sigma     = float(cycle_smooth_sigma)
        self.cycle_prominence_ratio = float(cycle_prominence_ratio)
        self.span_margin            = max(0, int(span_margin))

        # ── R(2+1)D-18 backbone（去掉 avgpool + fc）──────────────────────
        stem, layer1, layer2, layer3, layer4 = _load_r2plus1d_backbone(pretrained)
        self.stem    = stem
        self.layer1  = layer1
        self.layer2  = layer2
        self.layer3  = layer3
        self.layer4  = layer4
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # ── 路径 1：直接 EF 回归 ─────────────────────────────────────────
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))

        # ── 路径 2：公式路径（EDV + ESV → EF_formula）───────────────────
        if use_formula_head:
            self.formula_proj = nn.Sequential(
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
            )
            self.edv_head      = nn.Linear(256, 1)
            self.esv_head      = nn.Linear(256, 1)
            self.vol_edv_mean  = float(vol_edv_mean)
            self.vol_edv_scale = float(vol_edv_scale)
            self.vol_esv_mean  = float(vol_esv_mean)
            self.vol_esv_scale = float(vol_esv_scale)
            a = max(1e-4, min(1 - 1e-4, float(ensemble_alpha_init)))
            self.log_alpha = nn.Parameter(torch.tensor(math.log(a / (1.0 - a))))
        else:
            self.formula_proj = self.edv_head = self.esv_head = self.log_alpha = None

        n = sum(p.numel() for p in self.parameters())
        print(f'[R21DCyclicEFHead] clip_len={clip_len} '
              f'min_cycle={min_cycle_frames} max_cycles={max_cycles} '
              f'target_cycle={target_cycle_frames} span_margin={span_margin} '
              f'formula_head={use_formula_head} pretrained={pretrained} '
              f'params={n/1e6:.1f}M  detect_warmup={cycle_detect_warmup}')

    # ─────────────────────────────────────────────────────────────────────
    def set_frozen(self, frozen: bool):
        for m in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters():
                p.requires_grad = not frozen

    def _stem_fwd(self, x):   return self.stem(x)
    def _layer1_fwd(self, x): return self.layer1(x)

    # ─────────────────────────────────────────────────────────────────────
    def _build_cycle_clips(
        self,
        frames: torch.Tensor,
        all_spans: List[List[Tuple[int, int]]],
    ) -> Tuple[torch.Tensor, List[int]]:
        """把所有样本的所有周期 span 构建为 R21D 输入 clip。

        Parameters
        ----------
        frames    : (B, T, C, H, W)
        all_spans : list[B] of list[(t0, t1)]

        Returns
        -------
        clips      : (total_clips, C, clip_len, H, W)
        cycle_counts: list[B] 每个样本的周期数
        """
        B, T, C, H, W = frames.shape
        clips        = []
        cycle_counts = []

        for b in range(B):
            spans   = all_spans[b]
            count_b = 0
            for (t0, t1) in spans:
                # 夹入有效范围
                t0 = max(0, min(t0, T - 1))
                t1 = max(t0, min(t1, T - 1))

                span_frames = frames[b, t0: t1 + 1]      # (span, C, H, W)
                x = span_frames.permute(1, 0, 2, 3).unsqueeze(0).float()  # (1,C,span,H,W)
                if span_frames.shape[0] != self.clip_len:
                    x = F.interpolate(
                        x, size=(self.clip_len, H, W),
                        mode='trilinear', align_corners=False,
                    )
                clips.append(x.squeeze(0))               # (C, clip_len, H, W)
                count_b += 1

            if count_b == 0:
                # 极端情况：无有效 span → 用全有效帧作为 fallback
                valid = frames.new_zeros(1)
                x = frames[b].permute(1, 0, 2, 3).unsqueeze(0).float()
                x = F.interpolate(x, size=(self.clip_len, H, W),
                                   mode='trilinear', align_corners=False)
                clips.append(x.squeeze(0))
                count_b = 1

            cycle_counts.append(count_b)

        clips_tensor = torch.stack(clips, dim=0).contiguous()   # (N_total, C, clip_len, H, W)
        return clips_tensor, cycle_counts

    # ─────────────────────────────────────────────────────────────────────
    def _r21d_forward(self, x: torch.Tensor) -> torch.Tensor:
        """R21D backbone forward，返回 (N, 512) 全局特征。"""
        if self.use_checkpoint and self.training:
            x = checkpoint(self._stem_fwd,   x, use_reentrant=False)
            x = checkpoint(self._layer1_fwd, x, use_reentrant=False)
        else:
            x = self.stem(x)
            x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.global_pool(x).flatten(1)   # (N, 512)

    # ─────────────────────────────────────────────────────────────────────
    def _aggregate(
        self,
        values: torch.Tensor,       # (N_total,) 或 (N_total,)
        cycle_counts: List[int],
        quality_weights: List[List[float]],
    ) -> torch.Tensor:
        """把 N_total 个逐周期预测按质量权重聚合为 (B,)。"""
        results = []
        idx = 0
        for i, cnt in enumerate(cycle_counts):
            seg = values[idx: idx + cnt]               # (cnt,)
            if self.cycle_quality_weighting and quality_weights[i]:
                w = torch.tensor(quality_weights[i],
                                 device=values.device, dtype=values.dtype)
                w = w / w.sum().clamp(min=1e-8)
                results.append((seg * w).sum())
            else:
                results.append(seg.mean())
            idx += cnt
        return torch.stack(results, dim=0)             # (B,)

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        frames: torch.Tensor,
        volumes: torch.Tensor,
        mask: torch.Tensor,
        update: int = 0,
    ) -> CyclicEFResult:
        """多周期 EF 预测。

        Parameters
        ----------
        frames  : (B, T, C, H, W)  视频帧（归一化）
        volumes : (B, T)            Mamba volume_head 预测容积（需 detach）
        mask    : (B, T) bool       有效帧掩码
        update  : 当前训练 update（控制峰值/均匀分割策略切换）

        Returns
        -------
        CyclicEFResult
        """
        B = frames.shape[0]
        use_peaks = (update >= self.cycle_detect_warmup)

        # ── Step 1：检测各样本的心动周期 ─────────────────────────────────
        all_spans    : List[List[Tuple[int, int]]] = []
        quality_weights: List[List[float]]         = []

        for b in range(B):
            valid_len = int(mask[b].sum().item())
            spans     = detect_cardiac_cycles(
                volumes[b], valid_len,
                min_cycle=self.min_cycle_frames,
                max_cycles=self.max_cycles,
                sigma=self.cycle_smooth_sigma,
                prominence_ratio=self.cycle_prominence_ratio,
                use_peaks=use_peaks,
                margin=self.span_margin,
            )
            all_spans.append(spans)
            qw = [_cycle_quality_weight(t1 - t0 + 1, self.target_cycle_frames)
                  for (t0, t1) in spans]
            quality_weights.append(qw)

        # ── Step 2：构建 clips ───────────────────────────────────────────
        clips, cycle_counts = self._build_cycle_clips(frames, all_spans)
        # clips: (N_total, C, clip_len, H, W)

        # ── Step 3：R21D backbone（批量，单次前向）──────────────────────
        feat = self._r21d_forward(clips)   # (N_total, 512)

        # ── Step 4：直接 EF 回归路径 ────────────────────────────────────
        ef_direct_all = (self.ef_mean + self.ef_scale *
                         torch.tanh(self.head(feat).squeeze(-1))
                         ).clamp(0.0, 100.0)              # (N_total,)

        if not self.use_formula_head:
            ef_avg = self._aggregate(ef_direct_all, cycle_counts, quality_weights)
            return CyclicEFResult(
                ef_final=ef_avg, ef_formula=ef_avg, ef_direct=ef_avg,
                edv=None, esv=None,
                cycle_count=torch.tensor(cycle_counts, device=frames.device),
            )

        # ── Step 5：公式路径 ─────────────────────────────────────────────
        h   = self.formula_proj(feat)                          # (N_total, 256)

        edv = (self.vol_edv_mean + self.vol_edv_scale *
               torch.tanh(self.edv_head(h).squeeze(-1))
               ).clamp(min=1.0)                               # (N_total,)
        esv = torch.min(
            self.vol_esv_mean + self.vol_esv_scale *
            torch.tanh(self.esv_head(h).squeeze(-1)),
            edv - 1.0,
        ).clamp(min=0.0)                                       # (N_total,)

        ef_formula_all = ((edv - esv) / edv * 100.0).clamp(0.0, 100.0)  # (N_total,)

        # ── Step 6：学习型内部集成 α ────────────────────────────────────
        alpha        = torch.sigmoid(self.log_alpha)
        ef_final_all = (alpha * ef_formula_all +
                        (1.0 - alpha) * ef_direct_all).clamp(0.0, 100.0)

        # ── Step 7：跨周期加权聚合 → (B,) ───────────────────────────────
        ef_final_b   = self._aggregate(ef_final_all,   cycle_counts, quality_weights)
        ef_formula_b = self._aggregate(ef_formula_all, cycle_counts, quality_weights)
        ef_direct_b  = self._aggregate(ef_direct_all,  cycle_counts, quality_weights)
        edv_b        = self._aggregate(edv,            cycle_counts, quality_weights)
        esv_b        = self._aggregate(esv,            cycle_counts, quality_weights)

        return CyclicEFResult(
            ef_final  = ef_final_b,
            ef_formula= ef_formula_b,
            ef_direct = ef_direct_b,
            edv       = edv_b,
            esv       = esv_b,
            cycle_count = torch.tensor(cycle_counts, device=frames.device,
                                       dtype=torch.float32),
        )


# 向后兼容别名（旧代码可能 import R21DEFHead）
class R21DEFHead(R21DCyclicEFHead):
    """向后兼容别名，接口映射到 R21DCyclicEFHead。"""
    def forward(self, frames, actions_or_volumes, mask, update=0):  # type: ignore[override]
        # 旧接口: forward(frames, actions, mask)  → 忽略 actions，用全帧均匀分割
        # 新接口: forward(frames, volumes, mask, update)
        volumes = actions_or_volumes
        if volumes.dtype in (torch.long, torch.int32, torch.int64):
            # 传入的是 actions（旧接口），生成全零 volumes 触发均匀分割
            volumes = torch.zeros(frames.shape[0], frames.shape[1],
                                  device=frames.device, dtype=torch.float32)
        return super().forward(frames, volumes, mask, update=update)
