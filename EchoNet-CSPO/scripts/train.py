"""训练主脚本（EchoNet-CSPO-mamba3ActorPriod-R21D-derectEF-CSPO）."""
import os, sys, math, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from types import SimpleNamespace as SimpleNS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import load_config
from utils.dataset import EchoNoLabelDataset, collate_episodes_nolabel
from models.actor_critic import MambaPolicyNet
from models.r2plus1d_ef_head import R21DCyclicEFHead
from models.heads import ModelEMA
from rl.rewards import compute_rewards, compute_ef_from_3action, forcediverge_3action
from rl.ppo import (
    compute_gae, ppo_policy_loss, value_loss,
    volume_smoothness_loss, volume_curvature_loss,
    mamba_ef_supervised_loss, r21d_cyclic_ef_loss, r21d_per_cycle_ef_loss,
    soft_curve_ef, soft_curve_ef_loss, selection_alignment_loss,
    select_sparsity_loss,
    cspo_loss,
)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def pick_device(name):
    return "cuda" if (name == "auto" and torch.cuda.is_available()) else name

@torch.no_grad()
def evaluate(model, r21d_head, loader, device, cfg, max_batches=0, update=0):
    model.eval(); r21d_head.eval()
    errs_mamba, errs_top1, errs_soft, errs_r21d, ed_counts, es_counts = [], [], [], [], [], []
    max_v = float(cfg.model.max_volume_ml)
    soft_tau = float(getattr(getattr(cfg, "rl", SimpleNS()), "soft_ef_tau", 6.0))
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches: break
        frames = batch["frames"].to(device, non_blocking=True)
        mask   = batch["mask"].to(device, non_blocking=True)
        ef_gt  = batch["ef"].to(device)
        logits, volume, _ = model(frames, mask)
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        volume = torch.nan_to_num(volume.float(), nan=max_v*0.5, posinf=max_v, neginf=0.0).clamp(0.0, max_v)
        actions = logits.argmax(dim=-1) * mask.long()
        ef_mamba = compute_ef_from_3action(volume, actions, mask)
        # 与策略无关的容积曲线 EF（Mamba 分支真实健康度，会随 volume head 一起降）
        ef_soft = soft_curve_ef(volume, mask, tau=soft_tau)[0]
        # v5.1 top-1 稳健解码：取 P(ED)/P(ES) 最高的单帧作 ED/ES（衡量策略排序质量）
        probs   = torch.softmax(logits, dim=-1)
        neg     = ~mask
        ed_top  = probs[..., 1].masked_fill(neg, -1.0).argmax(dim=1)
        es_top  = probs[..., 2].masked_fill(neg, -1.0).argmax(dim=1)
        ridx    = torch.arange(frames.shape[0], device=device)
        edv_t   = volume[ridx, ed_top]; esv_t = volume[ridx, es_top]
        ef_top1 = ((edv_t - esv_t) / edv_t.clamp(min=1.0) * 100.0).clamp(0.0, 100.0)
        try:
            cyclic = r21d_head(frames, volume.detach(), mask, update=update)
            ef_r21d = torch.nan_to_num(cyclic.ef_final.float(), nan=50.0, posinf=100.0, neginf=0.0)
        except Exception:
            ef_r21d = torch.full_like(ef_gt, 50.0)
        errs_mamba.extend(torch.abs(ef_mamba - ef_gt).cpu().tolist())
        errs_top1.extend(torch.abs(ef_top1 - ef_gt).cpu().tolist())
        errs_soft.extend(torch.abs(ef_soft - ef_gt).cpu().tolist())
        errs_r21d.extend(torch.abs(ef_r21d - ef_gt).cpu().tolist())
        ed_counts.extend(((actions == 1) & mask).sum(dim=1).cpu().tolist())
        es_counts.extend(((actions == 2) & mask).sum(dim=1).cpu().tolist())
    model.train(); r21d_head.train()
    return {
        "ef_mae_mamba":      float(np.mean(errs_mamba)) if errs_mamba else 0.0,
        "ef_mae_mamba_top1": float(np.mean(errs_top1))  if errs_top1  else 0.0,
        "ef_mae_soft":       float(np.mean(errs_soft))  if errs_soft  else 0.0,
        "ef_mae_r21d":       float(np.mean(errs_r21d))  if errs_r21d  else 0.0,
        "ed_mean":           float(np.mean(ed_counts))   if ed_counts  else 0.0,
        "es_mean":           float(np.mean(es_counts))   if es_counts  else 0.0,
        "n": len(errs_mamba),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--resume",  default=None)
    ap.add_argument("--reset-training", action="store_true")
    args = ap.parse_args()

    cfg, raw_cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device  = pick_device(cfg.train.device)
    use_amp = bool(cfg.train.amp) and device == "cuda"
    print(f"[info] device={device}  amp={use_amp}")
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.train.log_dir, exist_ok=True)

    cspo_cfg   = getattr(cfg, "cspo", SimpleNS())
    cspo_lam   = float(getattr(cspo_cfg, "lambda",    0.1))
    cspo_phi   = float(getattr(cspo_cfg, "phi",       0.25))
    cspo_dmin  = float(getattr(cspo_cfg, "delta_min", 0.10))
    cspo_burnin = int( getattr(cspo_cfg, "burn_in",   2000))
    cspo_eps   = float(getattr(cspo_cfg, "eps_gate",  0.005))
    print(f"[info] CSPO lambda={cspo_lam} phi={cspo_phi} delta_min={cspo_dmin} burn_in={cspo_burnin} eps_gate={cspo_eps}")

    train_ds = EchoNoLabelDataset(cfg, "TRAIN")
    val_ds   = EchoNoLabelDataset(cfg, "VAL")
    print(f"[info] train={len(train_ds)}  val={len(val_ds)}")
    pin = (device == "cuda"); nw = int(cfg.train.num_workers); B = int(cfg.train.envs_per_update)
    train_loader = DataLoader(train_ds, batch_size=B, shuffle=True, num_workers=nw,
        collate_fn=collate_episodes_nolabel, pin_memory=pin, drop_last=True, persistent_workers=(nw>0))
    val_loader = DataLoader(val_ds, batch_size=B, shuffle=False, num_workers=nw,
        collate_fn=collate_episodes_nolabel, pin_memory=pin, persistent_workers=(nw>0))

    model = MambaPolicyNet(cfg).to(device)
    ef_cfg = getattr(cfg.model, "r2plus1d_ef", None)
    def _g(k, d): return getattr(ef_cfg, k, d) if ef_cfg else d
    r21d_head = R21DCyclicEFHead(
        pretrained=bool(_g("pretrained", True)), clip_len=int(_g("clip_len", 32)),
        dropout=float(cfg.model.dropout), use_checkpoint=bool(_g("use_checkpoint", True)),
        ef_mean=float(_g("ef_mean", 55.0)), ef_scale=float(_g("ef_scale", 35.0)),
        use_formula_head=bool(_g("use_formula_head", False)),
        vol_edv_mean=float(_g("vol_edv_mean", 100.0)), vol_edv_scale=float(_g("vol_edv_scale", 80.0)),
        vol_esv_mean=float(_g("vol_esv_mean", 45.0)),  vol_esv_scale=float(_g("vol_esv_scale", 40.0)),
        ensemble_alpha_init=float(_g("ensemble_alpha_init", 0.5)),
        min_cycle_frames=int(_g("min_cycle_frames", 20)), max_cycles=int(_g("max_cycles", 4)),
        target_cycle_frames=int(_g("target_cycle_frames", 64)),
        cycle_quality_weighting=bool(_g("cycle_quality_weighting", True)),
        cycle_detect_warmup=int(_g("cycle_detect_warmup", 2000)),
        cycle_smooth_sigma=float(_g("cycle_smooth_sigma", 3.0)),
        cycle_prominence_ratio=float(_g("cycle_prominence_ratio", 0.15)),
        span_margin=int(_g("span_margin", 6)),
    ).to(device)

    imp_cfg   = getattr(cfg, "improvements", SimpleNS())
    ema_on    = bool( getattr(imp_cfg, "ema_enabled",        True))
    ema_decay = float(getattr(imp_cfg, "ema_decay",         0.999))
    ema_wu    = int(  getattr(imp_cfg, "ema_warmup_updates",  200))
    ema       = ModelEMA(model, decay=ema_decay) if ema_on else None

    rl_cfg         = getattr(cfg, "rl", SimpleNS())
    mamba_ef_coef  = float(getattr(rl_cfg, "mamba_ef_loss_coef",   0.8))
    vol_sm_coef    = float(getattr(rl_cfg, "vol_smooth_loss_coef", 0.05))
    r21d_coef      = float(getattr(rl_cfg, "r21d_ef_loss_coef",    1.0))
    r21d_pc_coef   = float(getattr(rl_cfg, "r21d_per_cycle_coef",  0.3))
    # ── v5(移植自 mamba3Actor) Mamba 分支无标签弱监督 ───────────────────
    soft_ef_coef   = float(getattr(rl_cfg, "soft_ef_loss_coef",  1.0))
    soft_ef_tau    = float(getattr(rl_cfg, "soft_ef_tau",        6.0))
    sparsity_coef  = float(getattr(rl_cfg, "select_sparsity_coef", 0.0))
    sparsity_target = float(getattr(rl_cfg, "select_target_ratio", 0.20))
    align_cfg      = getattr(cfg, "align", SimpleNS())
    align_on       = bool( getattr(align_cfg, "enabled",        True))
    align_coef     = float(getattr(align_cfg, "coef",           1.0))
    align_warmup   = int(  getattr(align_cfg, "warmup_updates", 150))
    align_ramp     = int(  getattr(align_cfg, "ramp_updates",   200))
    align_ed_thr   = float(getattr(align_cfg, "ed_thresh",         1.5))
    align_sharp    = float(getattr(align_cfg, "sharpness",         3.0))
    align_window   = int(  getattr(align_cfg, "window_size",       0))    # v5.2: 滑动窗口局部 z-score
    align_es_w     = float(getattr(align_cfg, "es_weight_factor",  1.0))  # V1.5: ES梯度加权倍数
    print(f"[info] v5 weak-sup: soft_ef_coef={soft_ef_coef} tau={soft_ef_tau} "
          f"sparsity(coef={sparsity_coef} target={sparsity_target}) "
          f"align(on={align_on} coef={align_coef} warmup={align_warmup} ramp={align_ramp} "
          f"ed_thresh={align_ed_thr} window={align_window} es_w={align_es_w})")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(r21d_head.parameters()),
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    total_upd = int(cfg.train.total_updates)
    lr_min    = float(getattr(cfg.train, "lr_min", float(cfg.train.lr) * 0.05))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_upd, eta_min=lr_min)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_update = 0; best_val_mamba = math.inf; best_val_r21d = math.inf; best_mamba_soft = math.inf
    _resume = args.resume or os.path.join(cfg.train.checkpoint_dir, "best.pt")
    # v5 注意：--reset-training 现在=完全从零（不加载任何权重），以保留 skip-bias 初始化；
    # 套用反塌缩修复必须用它（或先删除旧 checkpoints），否则旧权重会覆盖 skip-bias。
    if os.path.exists(_resume) and not getattr(args, "reset_training", False):
        ck = torch.load(_resume, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("model", ck), strict=False)
        if "r21d" in ck: r21d_head.load_state_dict(ck["r21d"], strict=False)
        if True:
            try: optimizer.load_state_dict(ck["optim"])
            except Exception: pass
            if "scheduler" in ck: scheduler.load_state_dict(ck["scheduler"])
            start_update   = int(ck.get("update", 0))
            best_val_mamba = float(ck.get("best_val_mamba", math.inf))
            best_val_r21d  = float(ck.get("best_val_r21d",  math.inf))
            print(f"[resume] update={start_update} best_mamba={best_val_mamba:.3f} best_r21d={best_val_r21d:.3f}")
        if ema: ema.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    train_iter = iter(train_loader)
    model.train(); r21d_head.train()
    log_path = os.path.join(cfg.train.log_dir, "train.log")
    log_file = open(log_path, "a", encoding="utf-8")
    def _log(s): print(s); log_file.write(s + "\n"); log_file.flush()

    nan_skips = 0; t_last = time.time()
    max_v = float(cfg.model.max_volume_ml); K = int(cfg.rl.epochs_per_update)

    for update in range(start_update, total_upd):
        try: batch = next(train_iter)
        except StopIteration: train_iter = iter(train_loader); batch = next(train_iter)

        frames = batch["frames"].to(device, non_blocking=True)
        mask   = batch["mask"].to(device, non_blocking=True)
        ef_gt  = batch["ef"].to(device)

        freeze_until = int(getattr(cfg.model, "freeze_backbone_warmup", 0))
        model.set_backbone_frozen(update < freeze_until)
        r21d_freeze = int(getattr(ef_cfg, "freeze_warmup", freeze_until)) if ef_cfg else 0
        r21d_head.set_frozen(update < r21d_freeze)
        cspo_active = (update >= cspo_burnin)

        # v5 对齐/采样-EF 监督的线性 ramp：先让 soft_ef 锚把容积曲线训出摆幅，再开启策略对齐
        if update < align_warmup:
            ramp_w = 0.0
        elif update < align_warmup + align_ramp:
            ramp_w = float(update - align_warmup) / float(max(1, align_ramp))
        else:
            ramp_w = 1.0

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_raw, volume_raw, value_raw = model(frames, mask)
            logits_f = torch.nan_to_num(logits_raw.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            volume_f = torch.nan_to_num(volume_raw.float(), nan=max_v*0.5, posinf=max_v, neginf=0.0).clamp(0.0, max_v)
            value_f  = torch.nan_to_num(value_raw.float(),  nan=0.0, posinf=1e4,  neginf=-1e4)
            dist      = Categorical(logits=logits_f)
            actions_a = dist.sample(); old_lp = dist.log_prob(actions_a)
            ef_mamba_a = compute_ef_from_3action(volume_f, actions_a, mask)
            rewards_a, _ = compute_rewards(actions_a, volume_f, ef_mamba_a, ef_gt, mask, cfg)
            adv, ret = compute_gae(rewards_a, value_f, mask, float(cfg.rl.gamma), float(cfg.rl.gae_lambda))
            mf  = mask.float()
            mu  = (adv * mf).sum() / mf.sum().clamp(min=1.0)
            var = ((adv - mu) ** 2 * mf).sum() / mf.sum().clamp(min=1.0)
            adv = (adv - mu) / (var.sqrt() + 1e-6)
            if cspo_active:
                actions_b  = forcediverge_3action(actions_a, mask, phi=cspo_phi, delta_min=cspo_dmin)
                ef_mamba_b = compute_ef_from_3action(volume_f, actions_b, mask)
                rewards_b, _ = compute_rewards(actions_b, volume_f, ef_mamba_b, ef_gt, mask, cfg)
            else:
                actions_b = actions_a; rewards_b = rewards_a

        with torch.amp.autocast("cuda", enabled=use_amp):
            cyclic_res = r21d_head(frames, volume_f.detach(), mask, update=update)
        ef_r21d_avg = torch.nan_to_num(cyclic_res.ef_final.float(), nan=50.0, posinf=100.0, neginf=0.0)
        L_r21d = r21d_cyclic_ef_loss(ef_r21d_avg, ef_gt) * r21d_coef
        L_r21d_pc = torch.zeros(1, device=device).squeeze()
        if r21d_pc_coef > 0.0 and cyclic_res.ef_formula is not None:
            L_r21d_pc = r21d_cyclic_ef_loss(cyclic_res.ef_formula, ef_gt) * r21d_pc_coef

        with torch.amp.autocast("cuda", enabled=use_amp):
            cached_feat = model.extract_features(frames)

        accum_loss = torch.zeros((), device=device)
        accum_pi = accum_v = accum_ent = accum_vol = accum_mamba_ef = accum_cspo = 0.0
        accum_soft = accum_align = accum_sparsity = 0.0
        n_valid = 0

        for _ in range(K):
            with torch.amp.autocast("cuda", enabled=use_amp):
                n_logits_raw, n_volume_raw, n_values_raw = model.forward_from_feat(cached_feat, mask)
            n_logits = torch.nan_to_num(n_logits_raw.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            n_volume = torch.nan_to_num(n_volume_raw.float(), nan=max_v*0.5, posinf=max_v, neginf=0.0)
            n_values = torch.nan_to_num(n_values_raw.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            pol_l, ent_l = ppo_policy_loss(n_logits, old_lp, actions_a, adv, mask,
                float(cfg.rl.clip_epsilon), float(cfg.rl.entropy_coef))
            v_l       = value_loss(n_values, ret, mask) * float(cfg.rl.value_coef)

            # Fix C: 二阶曲率正则（替代一阶 TV，不再压制 ED–ES 摆幅，也利好 R21D 周期检测）
            vol_sm_l  = volume_curvature_loss(n_volume, mask) * vol_sm_coef

            # Fix A: 与策略无关的容积曲线 EF 锚（始终开启，主监督 volume head）
            soft_ef_l, _ = soft_curve_ef_loss(n_volume, ef_gt, mask, tau=soft_ef_tau)
            soft_ef_l = soft_ef_l * soft_ef_coef

            # Fix B: 容积曲线 → 逐帧 ED/ES 自蒸馏对齐（warmup 后 ramp 介入，治策略塌缩）
            # V1.5: 传入 es_weight_factor（三路独立归一化 + ES 梯度放大，治 z-score 不对称）
            if align_on and ramp_w > 0.0:
                align_l = selection_alignment_loss(
                    n_logits, n_volume, mask,
                    ed_thresh=align_ed_thr, sharpness=align_sharp,
                    window_size=align_window,
                    es_weight_factor=align_es_w) * (align_coef * ramp_w)
            else:
                align_l = torch.zeros((), device=device)

            # 采样动作 EF 一致性：同样 ramp 介入（早期与锚冲突，故 warmup 前关闭）
            if ramp_w > 0.0:
                ef_mamba_k = compute_ef_from_3action(n_volume, actions_a, mask)
                mamba_ef_l = mamba_ef_supervised_loss(ef_mamba_k, ef_gt) * (mamba_ef_coef * ramp_w)
            else:
                mamba_ef_l = torch.zeros((), device=device)

            # v5 反塌缩稀疏度损失（始终开启，直接压过选，不依赖 RL 优势）
            if sparsity_coef > 0.0:
                sparsity_l = select_sparsity_loss(
                    n_logits, mask, target_ratio=sparsity_target) * sparsity_coef
            else:
                sparsity_l = torch.zeros((), device=device)

            if cspo_active:
                L_cspo = cspo_loss(n_logits, actions_a, actions_b, rewards_a, rewards_b,
                                   mask, gamma=float(cfg.rl.gamma), eps_gate=cspo_eps) * cspo_lam
            else:
                L_cspo = torch.zeros((), device=device)
            loss_k = (pol_l + ent_l + v_l + vol_sm_l + soft_ef_l
                      + align_l + mamba_ef_l + sparsity_l + L_cspo)
            if not torch.isfinite(loss_k):
                nan_skips += 1; _log(f"[warn] non-finite loss at upd={update}; skipped"); continue
            accum_loss += loss_k
            accum_pi += float(pol_l.detach()); accum_v += float(v_l.detach())
            accum_ent += float(ent_l.detach()); accum_vol += float(vol_sm_l.detach())
            accum_mamba_ef += float(mamba_ef_l.detach()); accum_cspo += float(L_cspo.detach())
            accum_soft += float(soft_ef_l.detach()); accum_align += float(align_l.detach())
            accum_sparsity += float(sparsity_l.detach())
            n_valid += 1

        if n_valid > 0:
            mean_loss = accum_loss / float(n_valid) + L_r21d + L_r21d_pc
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(mean_loss).backward()
            scaler.unscale_(optimizer)
            # v5.1 关键修复：分离梯度裁剪。合并裁剪会让 R21D(31M, loss≈10) 的大梯度把
            # 整体范数推高，按同比例缩小所有梯度——本就微弱的策略梯度被连累再缩 ~16x，
            # 导致策略 argmax 永远停在 skip(eval ed=es=0)。分开裁剪后两分支互不连累。
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.rl.max_grad_norm))
            torch.nn.utils.clip_grad_norm_(
                r21d_head.parameters(), float(getattr(cfg.rl, "r21d_grad_norm", cfg.rl.max_grad_norm)))
            scaler.step(optimizer); scaler.update()
            if ema and update >= ema_wu: ema.update(model)
        scheduler.step()

        if update % int(cfg.train.log_interval) == 0:
            dt = time.time() - t_last; t_last = time.time()
            d  = float(max(1, n_valid))
            mae_m = torch.abs(ef_mamba_a - ef_gt).mean().item()
            mae_r = torch.abs(ef_r21d_avg - ef_gt).mean().item()
            r_mean = rewards_a.sum(dim=1).mean().item()
            sel_ed = ((actions_a == 1) & mask).sum(dim=1).float().mean().item()
            sel_es = ((actions_a == 2) & mask).sum(dim=1).float().mean().item()
            cc_avg = cyclic_res.cycle_count.float().mean().item()
            div_str = (f" div={((actions_a != actions_b) & mask).float().sum() / mask.float().sum().clamp(min=1.0):.2f}"
                       if cspo_active else " div=—")
            _log(f"[upd {update:5d}] ef_mamba={mae_m:.3f} ef_r21d={mae_r:.3f} r={r_mean:.3f} "
                 f"ed={sel_ed:.1f} es={sel_es:.1f} cycles={cc_avg:.1f}{div_str} rampw={ramp_w:.2f} "
                 f"pi={accum_pi/d/K:.3f} V={accum_v/d/K:.3f} ent={accum_ent/d/K:.3f} "
                 f"curv={accum_vol/d/K:.4f} soft_ef={accum_soft/d/K:.3f} align={accum_align/d/K:.3f} "
                 f"mb_ef={accum_mamba_ef/d/K:.3f} spars={accum_sparsity/d/K:.4f} "
                 f"cspo={accum_cspo/d/K:.4f} r21d={float(L_r21d.detach()):.3f} "
                 f"lr={scheduler.get_last_lr()[0]:.2e} skip={nan_skips} dt={dt:.1f}s")

        if update > 0 and update % int(cfg.train.eval_interval) == 0:
            if ema and update >= ema_wu: ema.apply_to(model)
            try:
                m = evaluate(model, r21d_head, val_loader, device, cfg,
                             int(getattr(cfg.train, "eval_max_batches", 0)), update=update)
                ev_n, ev_mm, ev_mr, ev_ed, ev_es = (
                    m['n'], m['ef_mae_mamba'], m['ef_mae_r21d'], m['ed_mean'], m['es_mean'])
                ev_mt = m.get('ef_mae_mamba_top1', float('nan'))
                ev_ms = m.get('ef_mae_soft', float('nan'))
                _log(f"[eval {update}] n={ev_n} ef_mae_soft={ev_ms:.3f} (Mamba健康度) | "
                     f"mamba_argmax={ev_mm:.3f} mamba_top1={ev_mt:.3f} (策略解码) | "
                     f"ef_mae_r21d={ev_mr:.3f} ed={ev_ed:.1f} es={ev_es:.1f}")
                # best.pt：R21D 主指标选取（保持不动）
                if m["ef_mae_r21d"] < best_val_r21d:
                    best_val_r21d = m["ef_mae_r21d"]; best_val_mamba = m["ef_mae_mamba"]
                    torch.save({"model": model.state_dict(), "r21d": r21d_head.state_dict(),
                                "optim": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                "update": update, "best_val_mamba": best_val_mamba,
                                "best_val_r21d": best_val_r21d, "raw_cfg": raw_cfg},
                               os.path.join(cfg.train.checkpoint_dir, "best.pt"))
                    _log(f"[best] r21d_MAE={best_val_r21d:.3f} mamba_MAE={best_val_mamba:.3f}  saved best.pt")
                # best_mamba.pt：按与策略无关的 soft 指标选取（Mamba 分支真实健康度，最稳健）
                if ev_ms < best_mamba_soft:
                    best_mamba_soft = ev_ms
                    torch.save({"model": model.state_dict(), "r21d": r21d_head.state_dict(),
                                "optim": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                "update": update, "best_mamba_soft": best_mamba_soft,
                                "best_val_r21d": best_val_r21d, "raw_cfg": raw_cfg},
                               os.path.join(cfg.train.checkpoint_dir, "best_mamba.pt"))
                    _log(f"[best] mamba_soft_MAE={best_mamba_soft:.3f}  saved best_mamba.pt")
            finally:
                if ema and update >= ema_wu: ema.restore(model)

        if update > 0 and update % int(cfg.train.save_interval) == 0:
            torch.save({"model": model.state_dict(), "r21d": r21d_head.state_dict(),
                        "optim": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "update": update, "best_val_mamba": best_val_mamba,
                        "best_val_r21d": best_val_r21d, "raw_cfg": raw_cfg},
                       os.path.join(cfg.train.checkpoint_dir, f"ckpt_{update}.pt"))
    log_file.close()

if __name__ == "__main__":
    main()
