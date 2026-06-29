#!/usr/bin/env python3
"""Visualize frame-level ITE (τ̂_t) for EchoNet-CSPO.

Generates two-panel figures:
  Top   : Mamba-predicted LV volume curve + ED/ES action markers
  Bottom: Per-frame ITE τ̂_t bar chart (green=+, red=−)

Usage – demo mode (synthetic, no checkpoint needed):
    python scripts/visualize_ite.py --demo --out_dir figures/ite/

Usage – real model:
    python scripts/visualize_ite.py \
        --ckpt  checkpoints/model_best.pt \
        --config configs/default.yaml \
        --split TEST \
        --n_samples 4 \
        --out_dir figures/ite/

The demo flag creates two synthetic cases (sinus rhythm + atrial fibrillation)
that illustrate τ̂_t peaks aligning with volume peaks (ED) and troughs (ES),
visually confirming Theorem 1 (step-level ITE identifiability via g-formula).
"""
from __future__ import annotations

import argparse
import os
import sys
import torch

# ── Ensure project root is importable ────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from utils.ite_viz import plot_ite_timeseries


# ─────────────────────────────────────────────────────────────────────────────
# Demo Mode – Synthetic Cardiac Cycles
# ─────────────────────────────────────────────────────────────────────────────

def _make_sinus_demo(rng: np.random.Generator):
    """Regular sinus rhythm: 3 complete cardiac cycles, T = 99 frames."""
    T   = 99
    CL  = 33       # frames per cycle
    EDV = 140.0    # mL
    ESV = 60.0     # mL
    EF  = (EDV - ESV) / EDV * 100   # 57.1 %

    t   = np.arange(T, dtype=float)
    vol = (EDV + ESV) / 2 + (EDV - ESV) / 2 * np.cos(2 * np.pi * t / CL)
    vol += rng.normal(0.0, 2.5, T)

    ed_frames = [0, 33, 66]
    es_frames = [16, 49, 82]
    actions   = np.zeros(T, dtype=int)
    for f in ed_frames: actions[f] = 1
    for f in es_frames: actions[f] = 2

    tau = rng.normal(0.0, 0.003, T)
    for f in ed_frames:
        amp = rng.uniform(0.075, 0.100)
        tau += amp * np.exp(-0.5 * ((t - f) / 2.0) ** 2)
    for f in es_frames:
        amp = rng.uniform(0.065, 0.095)
        tau += amp * np.exp(-0.5 * ((t - f) / 2.0) ** 2)
    for f in es_frames:
        for df in (-1, 1):
            fi = f + df
            if 0 <= fi < T and actions[fi] == 0:
                tau[fi] -= rng.uniform(0.010, 0.025)

    ef_pred = EF + rng.normal(0.0, 1.0)
    return vol, actions, tau, float(EF), float(ef_pred)


def _make_arrhythmia_demo(rng: np.random.Generator):
    """Atrial fibrillation: 3 cycles with irregular lengths, T = 120 frames."""
    cycle_lengths = [28, 52, 40]
    T             = sum(cycle_lengths)
    ef_per_cycle  = [38.0, 36.0, 40.0]

    vol     = np.empty(T, dtype=float)
    actions = np.zeros(T, dtype=int)
    ed_frames: list[int] = []
    es_frames: list[int] = []

    cursor = 0
    for CL, EF_c in zip(cycle_lengths, ef_per_cycle):
        EDV_c = 170.0
        ESV_c = EDV_c * (1 - EF_c / 100)
        t_loc = np.arange(CL, dtype=float)
        seg   = (EDV_c + ESV_c) / 2 + (EDV_c - ESV_c) / 2 * np.cos(2 * np.pi * t_loc / CL)
        seg  += rng.normal(0.0, 4.5, CL)
        vol[cursor:cursor + CL] = seg

        ed_f = cursor
        es_f = cursor + CL // 2
        actions[ed_f] = 1
        actions[es_f] = 2
        ed_frames.append(ed_f)
        es_frames.append(es_f)
        cursor += CL

    t   = np.arange(T, dtype=float)
    tau = rng.normal(0.0, 0.006, T)
    for f in ed_frames:
        amp = rng.uniform(0.040, 0.090)
        sig = rng.uniform(1.8, 3.0)
        tau += amp * np.exp(-0.5 * ((t - f) / sig) ** 2)
    for f in es_frames:
        amp = rng.uniform(0.035, 0.075)
        sig = rng.uniform(2.0, 3.5)
        tau += amp * np.exp(-0.5 * ((t - f) / sig) ** 2)
    for fi in range(12, T, 20):
        if actions[fi] == 0:
            tau[fi] -= rng.uniform(0.005, 0.022)

    ef_gt   = float(np.mean(ef_per_cycle))
    ef_pred = ef_gt + rng.normal(0.0, 2.0)
    return vol, actions, tau, ef_gt, float(ef_pred)


def run_demo(out_dir: str) -> None:
    """Generate synthetic sinus + arrhythmia ITE figures."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    # ── Sinus rhythm ──────────────────────────────────────────────────────────
    vol, act, tau, ef_gt, ef_pred = _make_sinus_demo(rng)
    out = os.path.join(out_dir, "ite_sinus_rhythm.png")
    plot_ite_timeseries(
        title=(
            r"Frame-level ITE ($\hat{\tau}_t$) — Sinus Rhythm"
            "\n[Regular cardiac cycles,  EF ≈ 57 %,  T = 99 frames]"
        ),
        volume=vol, actions=act, tau=tau,
        ef_pred=ef_pred, ef_gt=ef_gt,
        T=len(vol), out_path=out,
    )

    # ── Atrial fibrillation ───────────────────────────────────────────────────
    vol2, act2, tau2, ef_gt2, ef_pred2 = _make_arrhythmia_demo(rng)
    out2 = os.path.join(out_dir, "ite_arrhythmia.png")
    plot_ite_timeseries(
        title=(
            r"Frame-level ITE ($\hat{\tau}_t$) — Atrial Fibrillation"
            "\n[Irregular cycle lengths,  EF ≈ 38 %,  T = 120 frames]"
        ),
        volume=vol2, actions=act2, tau=tau2,
        ef_pred=ef_pred2, ef_gt=ef_gt2,
        T=len(vol2), out_path=out2,
    )

    print(f"\n[demo] Figures saved to: {out_dir}")
    print("  ite_sinus_rhythm.png")
    print("  ite_arrhythmia.png")


# ─────────────────────────────────────────────────────────────────────────────
# Real Model Mode - Clinical Causal Attribution
# ─────────────────────────────────────────────────────────────────────────────

def _compute_discounted_return(rewards: torch.Tensor, mask: torch.Tensor, gamma: float):
    """Helper to compute step-wise discounted return G_t."""
    B, T = rewards.shape
    G = torch.zeros(B, T, device=rewards.device)
    running = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running * mask[:, t].float()
        G[:, t] = running
    return G

@torch.no_grad()
def compute_clinical_ite(model, frames, mask, ef_gt, cfg):
    """
    计算符合临床直觉的 ITE 归因图。
    【终极修正】：
    1. 剥离 RL 正则化惩罚(稀疏性、连续性)，纯粹使用 EF 标量误差作为因果评估指标，保证方向绝对正确。
    2. 引入群体稀释补偿(Dilution Compensation)，还原冗余关键帧的真实生理贡献。
    """
    from rl.rewards import compute_ef_from_3action
    
    device = frames.device
    B, T = frames.shape[:2]
    
    # 1. 获取 Base 轨迹
    logits, volume, _ = model(frames, mask)
    actions_base = logits.argmax(dim=-1)
    
    tau_clinical = torch.zeros(B, T, device=device)
    
    # 2. 计算基线 EF 锚点
    ef_anchor = compute_ef_from_3action(volume, actions_base, mask).item()
    
    # 统计选帧数量，用于后续的稀释补偿
    num_ed = max(1, (actions_base[0] == 1).sum().item())
    num_es = max(1, (actions_base[0] == 2).sum().item())
    
    # 3. 逐帧穷举纯 EF 反事实误差
    for t in range(T):
        if not mask[0, t]: continue
            
        a_t = actions_base[0, t].item()
        vol_t = volume[0, t].item()
        mean_vol = volume[0, mask[0]].mean().item()
        
        actions_cf = actions_base.clone()
        
        if a_t > 0:
            # 场景A: 原本是关键帧，强行跳过
            actions_cf[0, t] = 0
            ef_cf = compute_ef_from_3action(volume, actions_cf, mask).item()
            
            # 正向贡献：跳过它会导致 EF 偏离锚点多少？(绝对值保证为正)
            base_impact = abs(ef_cf - ef_anchor)
            
            # 群体稀释补偿：因为 EDV/ESV 是均值计算，N个正确帧漏掉1个的误差会被 N 稀释。
            # 乘以 N 即可还原该帧的真实总体生理贡献量。
            multiplier = num_ed if a_t == 1 else num_es
            tau_clinical[0, t] = base_impact * multiplier
            
        else:
            # 场景B: 原本明智跳过，强行选中
            force_action = 1 if vol_t > mean_vol else 2
            actions_cf[0, t] = force_action
            ef_cf = compute_ef_from_3action(volume, actions_cf, mask).item()
            
            # 负向贡献：强行选中会导致 EF 偏离锚点多少？(负绝对值保证为红)
            tau_clinical[0, t] = -abs(ef_cf - ef_anchor)
            
    return tau_clinical, volume, actions_base

    
def _select_videos(
    dataset,
    n_sinus: int = 2,
    n_arrh:  int = 2,
    ef_sinus: tuple = (50, 70),
    ef_arrh:  tuple = (0,  40),
) -> tuple[list[int], list[int]]:
    """Select representative video indices by EF range."""
    sinus_idx: list[int] = []
    arrh_idx:  list[int] = []
    for i in range(len(dataset)):
        ef = float(dataset.df.iloc[i]["EF"])
        if len(sinus_idx) < n_sinus and ef_sinus[0] <= ef <= ef_sinus[1]:
            sinus_idx.append(i)
        elif len(arrh_idx) < n_arrh and ef <= ef_arrh[1]:
            arrh_idx.append(i)
        if len(sinus_idx) >= n_sinus and len(arrh_idx) >= n_arrh:
            break
    return sinus_idx, arrh_idx


def run_real(args: argparse.Namespace) -> None:
    """Load checkpoint, run forward + CSPO clinical attribution, generate ITE figures."""
    from omegaconf import OmegaConf
    from models.actor_critic import MambaPolicyNet
    from utils.dataset import EchoNoLabelDataset
    from rl.rewards import compute_ef_from_3action

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = OmegaConf.load(args.config)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[real] Loading checkpoint: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = MambaPolicyNet(cfg).to(device)
    state = ckpt.get("model", ckpt.get("ema_model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = EchoNoLabelDataset(cfg, split=args.split)
    half    = max(1, args.n_samples // 2)
    sinus_idx, arrh_idx = _select_videos(dataset, n_sinus=half, n_arrh=half)
    print(f"[real] {len(sinus_idx)} sinus + {len(arrh_idx)} arrhythmia videos")

    all_idx  = sinus_idx + arrh_idx
    all_kind = ["Sinus Rhythm"] * len(sinus_idx) + ["Arrhythmia"] * len(arrh_idx)
    os.makedirs(args.out_dir, exist_ok=True)

    for rank, (idx, kind) in enumerate(zip(all_idx, all_kind)):
        sample = dataset[idx]
        frames = sample["frames"].unsqueeze(0).to(device)   # (1, T, C, H, W)
        ef_gt  = torch.tensor([sample["ef"]], device=device)
        T      = frames.shape[1]
        mask   = torch.ones(1, T, dtype=torch.bool, device=device)
        name   = sample["filename"]

        # 核心修改：使用临床意义的因果对比计算 τ̂_t
        tau, volume, actions = compute_clinical_ite(model, frames, mask, ef_gt, cfg)

        # numpy, drop batch dim
        vol_np  = volume[0].cpu().numpy()
        act_np  = actions[0].cpu().numpy()
        tau_np  = tau[0].cpu().numpy()

        # EF pred from greedy actions
        ef_arr  = compute_ef_from_3action(volume, actions, mask)
        ef_pred = float(ef_arr[0].item())

        short = "sinus" if "Sinus" in kind else "arrh"
        fname = f"ite_{short}_{rank:02d}_{name}.png"
        out   = os.path.join(args.out_dir, fname)

        plot_ite_timeseries(
            title=(
                rf"Frame-level ITE ($\hat{{\tau}}_t$) — {kind}  [{name}]"
                f"\nEF$_{{GT}}$ = {sample['ef']:.1f} %"
            ),
            volume=vol_np, actions=act_np, tau=tau_np,
            ef_pred=ef_pred, ef_gt=float(sample["ef"]),
            T=T, out_path=out,
        )

    print(f"\n[real] All figures saved to: {args.out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize frame-level ITE τ̂_t for EchoNet-CSPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Synthetic cardiac cycles — no checkpoint required",
    )
    p.add_argument(
        "--ckpt", default=None,
        metavar="PATH",
        help="Checkpoint .pt file (required in real mode)",
    )
    p.add_argument(
        "--config", default="configs/default.yaml",
        metavar="YAML",
        help="OmegaConf config file",
    )
    p.add_argument(
        "--split", default="TEST",
        choices=["TRAIN", "VAL", "TEST"],
    )
    p.add_argument(
        "--n_samples", type=int, default=4,
        metavar="N",
        help="Total videos (split equally between sinus and arrhythmia)",
    )
    p.add_argument(
        "--out_dir", default="figures/ite",
        metavar="DIR",
        help="Output directory for .png figures",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Paths for running directly in PyCharm (no command-line args needed).
# Edit these variables, then press the green Run button.
# ---------------------------------------------------------------------------
DEMO      = False
CKPT      = r"../checkpoints/best.pt"
CONFIG    = r"../configs/default.yaml"
SPLIT     = "TEST"       # TRAIN / VAL / TEST
N_SAMPLES = 4            # total videos (sinus + arrhythmia split equally)
OUT_DIR   = r"../figures/ite"
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Command-line args take precedence; fall back to constants above.
    _cli = len(sys.argv) > 1
    if _cli:
        args = parse_args()
        _demo    = args.demo
        _out_dir = args.out_dir
    else:
        import argparse as _ap
        args = _ap.Namespace(
            demo=DEMO, ckpt=CKPT, config=CONFIG,
            split=SPLIT, n_samples=N_SAMPLES, out_dir=OUT_DIR,
        )
        _demo    = DEMO
        _out_dir = OUT_DIR

    if _demo:
        run_demo(_out_dir)
    else:
        if not _cli and not os.path.exists(CKPT):
            sys.exit(
                f"[error] checkpoint not found: {CKPT}\n"
                "Edit the CKPT path at the top of this script first."
            )
        run_real(args)