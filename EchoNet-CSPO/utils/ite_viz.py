"""Frame-level ITE (τ̂_t) computation and publication-quality visualization.

Entry points
────────────
  compute_frame_level_ite()  – K-pair CSPO averaging, returns (B, T) per-frame ITE
  plot_ite_timeseries()      – publication-quality 2-panel figure

Algorithm (compute_frame_level_ite)
────────────────────────────────────
For k = 1…K:
  1. Generate counterfactual trajectory B via FORCEDIVERGE_3action(A).
  2. Compute discounted return difference ΔG_t = G_A_t − G_B_t.
  3. Accumulate ΔG_t only at divergent steps (a_A_t ≠ a_B_t).

τ̂_t = Σ_k ΔG_t · 𝟙[div] / #{k : a_A_t ≠ a_B_t}

This is an unbiased Monte-Carlo estimate of the step-level ITE
as defined in Theorem 1 (g-formula identifiability).

Figure layout
─────────────
Top panel (60 %):
  • Blue line  = Mamba-predicted LV volume (mL)
  • ▲ green    = ED-labelled frames (action = 1)
  • ▼ red      = ES-labelled frames (action = 2)
Bottom panel (40 %):
  • Bar chart τ̂_t  — green=positive, red=negative
  • Amber outline  = frames where action ∈ {1, 2}
  • Dashed zero line
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / server rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import torch

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Per-frame ITE Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_frame_level_ite(
    volume,          # (B, T) torch float  Mamba volume predictions (mL)
    actions_a,       # (B, T) torch long   policy actions {0,1,2}
    ef_gt,           # (B,)   torch float  ground-truth EF (%)
    mask,            # (B, T) torch bool   valid-frame mask
    cfg,                           # OmegaConf / SimpleNamespace config
    K: int = 32,
    gamma: float = 0.99,
    phi: float = 0.25,
    delta_min: float = 0.10,
):
    """Estimate per-frame causal ITE τ̂_t via K-pair CSPO averaging.

    Parameters
    ----------
    volume    : (B, T)  Mamba volume_head predictions in mL
    actions_a : (B, T)  Reference-trajectory actions (greedy or sampled)
    ef_gt     : (B,)    Ground-truth LVEF (%)
    mask      : (B, T)  True = valid frame, False = padding
    cfg       : Project config (needs cfg.reward, cfg.model, cfg.cspo)
    K         : Number of counterfactual trajectory pairs
    gamma     : Discount factor (use cfg.rl.gamma)
    phi       : Per-step flip probability for FORCEDIVERGE
    delta_min : Minimum divergence fraction guarantee

    Returns
    -------
    tau      : (B, T) float  Per-frame ITE estimate (NaN at padding positions)
    coverage : (B, T) float  Fraction of K pairs divergent at each frame (NaN at padding)
    """
    from rl.rewards import (
        compute_ef_from_3action,
        compute_rewards,
        forcediverge_3action,
    )

    device = actions_a.device
    B, T   = actions_a.shape

    # Reference trajectory A rewards (fixed, computed once)
    ef_a       = compute_ef_from_3action(volume, actions_a, mask)
    rewards_a, _ = compute_rewards(actions_a, volume, ef_a, ef_gt, mask, cfg)

    tau_sum = torch.zeros(B, T, device=device)
    tau_cnt = torch.zeros(B, T, device=device)

    for _ in range(K):
        # Generate counterfactual trajectory B
        actions_b    = forcediverge_3action(actions_a, mask, phi=phi, delta_min=delta_min)
        ef_b         = compute_ef_from_3action(volume, actions_b, mask)
        rewards_b, _ = compute_rewards(actions_b, volume, ef_b, ef_gt, mask, cfg)

        # Discounted return difference ΔG_t = G_A_t − G_B_t
        delta_r = rewards_a - rewards_b          # (B, T)
        mf      = mask.float()
        delta_G = torch.zeros(B, T, device=device)
        running = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            running      = delta_r[:, t] + gamma * running * mf[:, t]
            delta_G[:, t] = running

        # Accumulate only at divergent steps
        div      = (actions_a != actions_b) & mask    # (B, T) bool
        tau_sum += delta_G * div.float()
        tau_cnt += div.float()

    tau      = tau_sum / tau_cnt.clamp(min=1.0)
    tau      = tau.masked_fill(~mask, float("nan"))
    coverage = (tau_cnt / K).masked_fill(~mask, float("nan"))
    return tau, coverage


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Colour Palette (Upgraded for Publication Quality)
# ─────────────────────────────────────────────────────────────────────────────

_C = dict(
    blue       = "#1f77b4",   # LV volume line (classic high-contrast blue)
    green      = "#2ca02c",   # ED markers, positive-ITE bars (vibrant green)
    red        = "#d62728",   # ES markers, negative-ITE bars (vibrant red)
    amber      = "#ff7f0e",   # action-frame bar outline (bright orange)
    lgray      = "#d3d3d3",   # grid lines
    dkgray     = "#333333",   # axis labels / text
    dark_green = "darkgreen", # edge color for ED elements to add sharpness
    dark_red   = "darkred",   # edge color for ES elements to add sharpness
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Publication-Quality Figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_ite_timeseries(
    title: str,
    volume: np.ndarray,            # (T,) in mL
    actions: np.ndarray,           # (T,) int  {0=skip, 1=ED, 2=ES}
    tau: np.ndarray,               # (T,) ITE estimates (may contain NaN)
    ef_pred: float,
    ef_gt: float,
    T: int,
    out_path: str,
    figsize: Tuple[float, float] = (14, 8.5),
    dpi: int = 300,
) -> None:
    """Two-panel ITE figure: LV volume curve (top) + per-frame ITE bars (bottom).

    The figure visually demonstrates that τ̂_t peaks align with volume peaks
    (ED) and troughs (ES), confirming physiological causal understanding.
    """
    # ==================== 全局样式美化 ====================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.linewidth'] = 1.5

    fr   = np.arange(T)
    vol  = np.array(volume[:T], dtype=float)
    act  = np.array(actions[:T], dtype=int)
    tau_ = np.array(tau[:T],    dtype=float)

    ed = np.where(act == 1)[0]
    es = np.where(act == 2)[0]

    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.suptitle(title, fontsize=17, fontweight="bold", color=_C["dkgray"], y=0.96, ha="center")

    gs  = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.08,
                           left=0.08, right=0.97, top=0.91, bottom=0.09)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ─── Top panel: LV Volume ────────────────────────────────────────────────
    ax1.plot(fr, vol, color=_C["blue"], lw=2.8, zorder=3,
             label="Mamba — Predicted LV Volume")
    ax1.fill_between(fr, vol, max(0, vol.min() - 5.0),
                     alpha=0.08, color=_C["blue"], zorder=1)

    if ed.size:
        ax1.scatter(ed, vol[ed], marker="^", s=140, color=_C["green"],
                    zorder=5, edgecolors=_C["dark_green"], linewidths=1.2,
                    label="Action = 1  (ED frame)")
    if es.size:
        ax1.scatter(es, vol[es], marker="v", s=140, color=_C["red"],
                    zorder=5, edgecolors=_C["dark_red"], linewidths=1.2,
                    label="Action = 2  (ES frame)")

    for idx in ed:
        ax1.axvline(idx, color=_C["green"], alpha=0.18, lw=1.0, ls=":")
    for idx in es:
        ax1.axvline(idx, color=_C["red"],   alpha=0.18, lw=1.0, ls=":")

    ax1.set_ylabel("LV Volume (mL)", fontsize=14, fontweight="bold", color=_C["dkgray"])
    ax1.tick_params(labelbottom=False, colors=_C["dkgray"], labelsize=12, length=4)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="both", color=_C["lgray"], lw=0.8, ls="--", alpha=0.6, zorder=0)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax1.set_ylim(bottom=max(0, vol.min() - 3), top=vol.max() + 5)

    ef_box = (f"EF$_{{pred}}$ = {ef_pred:.1f} %    "
              f"EF$_{{GT}}$ = {ef_gt:.1f} %")
    ax1.text(0.98, 0.94, ef_box, transform=ax1.transAxes,
             ha="right", va="top", fontsize=13, fontweight="bold", color=_C["dkgray"],
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa",
                       ec="gray", alpha=0.9))

    leg1 = ax1.legend(loc="upper left", fontsize=12,
                      framealpha=0.95, edgecolor="gray", facecolor="white")
    for t in leg1.get_texts():
        t.set_color(_C["dkgray"])

    # ─── Bottom panel: ITE bars ──────────────────────────────────────────────
    valid = ~np.isnan(tau_)
    if valid.any():
        vf = fr[valid]
        vt = tau_[valid]
        va = act[valid]

        bar_col   = [_C["green"] if v >= 0 else _C["red"]   for v in vt]
        # 优化边缘描边逻辑：未被选中的基础帧使用暗色描边增加锐利度，选中的加上显眼的金色高亮
        bar_edge  = [_C["dark_green"] if v >= 0 else _C["dark_red"] for v in vt]
        final_edge = [_C["amber"] if a in (1, 2) else be for a, be in zip(va, bar_edge)]
        edge_lw    = [2.5 if a in (1, 2) else 1.0 for a in va]

        ax2.bar(vf, vt,
                color=bar_col, edgecolor=final_edge, linewidth=edge_lw,
                width=0.8, zorder=3)

    ax2.axhline(0, color="black", lw=1.2, ls="--", zorder=2)
    for idx in ed:
        ax2.axvline(idx, color=_C["green"], alpha=0.18, lw=1.0, ls=":")
    for idx in es:
        ax2.axvline(idx, color=_C["red"],   alpha=0.18, lw=1.0, ls=":")

    ax2.set_ylabel(r"$\hat{\tau}_t$  (ITE)", fontsize=14, fontweight="bold", color=_C["dkgray"])
    ax2.set_xlabel("Frame Index", fontsize=14, fontweight="bold", color=_C["dkgray"])
    ax2.tick_params(colors=_C["dkgray"], labelsize=12, length=4)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="both", color=_C["lgray"], lw=0.8, ls="--", alpha=0.6, zorder=0)

    legend_handles = [
        mpatches.Patch(color=_C["green"], ec=_C["dark_green"], lw=1.0,
                       label=r"$\hat{\tau}_t > 0$  (positive causal effect)"),
        mpatches.Patch(color=_C["red"], ec=_C["dark_red"], lw=1.0,
                       label=r"$\hat{\tau}_t < 0$  (negative causal effect)"),
        mpatches.Patch(facecolor="none", edgecolor=_C["amber"], linewidth=2.5,
                       label="Selected frame (ED / ES)"),
    ]
    leg2 = ax2.legend(handles=legend_handles, loc="upper right", fontsize=12,
                      framealpha=0.95, edgecolor="gray", facecolor="white")
    for t in leg2.get_texts():
        t.set_color(_C["dkgray"])

    # ─── Save ────────────────────────────────────────────────────────────────
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ite_viz] saved → {out_path}")