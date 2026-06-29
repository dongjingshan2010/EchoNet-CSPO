"""evaluate_EchoNet-CSPO-Pediatric.py — CSPO 三动作评估脚本，适配 VideosA4C 小儿超声数据集。

架构
────
  Mamba 路径  MambaPolicyNet(3类) → actions∈{0,1,2} → compute_ef_from_3action → EF_mamba
  R21D  路径  R21DCyclicEFHead(volume曲线→周期检测→多周期EF均值) → EF_r21d
  CSPO  路径  FORCEDIVERGE 生成反事实轨迹 B → role_ite_stats → ITE 角色归因

VideosA4C 数据集差异（相比 EchoNet-Dynamic）
──────────────────────────────────────────────
  * FileName 列已含 .avi 后缀，无需再追加
  * 无 NumberOfFrames 列，跳过该过滤
  * Split 列为数值 0-9（10-fold CV），不是 TRAIN/VAL/TEST 字符串
    --split ALL  使用全部数据（默认，推荐用于 Pediatric 评估）
    --split 0~9  仅使用对应折的数据

ITE 角色归因（4 类）
────────────────────
  ED_to_skip : 轨迹 A=ED(1), 轨迹 B=skip(0)  →  选 ED 帧的边际贡献
  ES_to_skip : 轨迹 A=ES(2), 轨迹 B=skip(0)  →  选 ES 帧的边际贡献
  ED_to_ES   : 轨迹 A=ED(1), 轨迹 B=ES(2)    →  ED 角色 vs ES 角色差异
  ES_to_ED   : 轨迹 A=ES(2), 轨迹 B=ED(1)    →  ES 角色 vs ED 角色差异

用法
────
  python scripts/evaluate_EchoNet-CSPO-Pediatric.py
  python scripts/evaluate_EchoNet-CSPO-Pediatric.py --split ALL --save_csv results_pediatric.csv
  python scripts/evaluate_EchoNet-CSPO-Pediatric.py --ckpt checkpoints/best.pt --split 0 --max 200
  python scripts/evaluate_EchoNet-CSPO-Pediatric.py --n_cspo 5
  # 覆盖数据路径（不改 config）:
  python scripts/evaluate_EchoNet-CSPO-Pediatric.py \\
      --filelist ../../data/VideosA4C/FileList_with_frames.csv \\
      --videos_dir ../../data/VideosA4C/VideosA4C
"""

import os
import sys
import argparse
import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import load_config
from utils.dataset import collate_episodes_nolabel
from utils.video import read_video
from models.actor_critic import MambaPolicyNet
from models.r2plus1d_ef_head import R21DCyclicEFHead
from rl.rewards import compute_rewards, compute_ef_from_3action, forcediverge_3action
from rl.ppo import role_ite_stats, soft_curve_ef


# ─────────────────────────────────────────────────────────────────────────────
# VideosA4C 专用数据集（小儿超声，10-fold CV，FileName 已含 .avi）
# ─────────────────────────────────────────────────────────────────────────────

class VideosA4CPediatricDataset(Dataset):
    """适配 VideosA4C 小儿超声数据集的 Dataset 类。

    与 EchoNoLabelDataset 的三点区别：
      1. FileName 列已含 .avi 后缀，直接拼路径，不再追加
      2. 无 NumberOfFrames 列，跳过该过滤
      3. Split 列为数值 0-9；split_val='ALL' 使用全部数据，
         split_val='0'~'9' 筛选对应折

    窗口选择策略与 EchoNoLabelDataset 完全相同：
      split_val == 'ALL' / 非 '0'~'9' → 视为测试模式，使用居中确定性窗口
      split_val 为数字字符串时：
          训练折（0~7）→ 随机窗口（若想训练复用本脚本，需在外部判断）
          此脚本仅做评估，固定使用居中窗口
    """

    def __init__(self, filelist: str, videos_dir: str, cfg, split_val: str = 'ALL'):
        self.videos_dir = videos_dir
        self.cfg = cfg
        self.split_val = split_val

        df = pd.read_csv(filelist)

        # Split 过滤
        if split_val != 'ALL':
            try:
                fold = int(split_val)
                df = df[df['Split'] == fold].reset_index(drop=True)
            except ValueError:
                print(f'[warn] split_val={split_val!r} 无法转为整数，使用全部数据')

        # 只需要 EF 不为空（无 NumberOfFrames 列，跳过）
        df = df[df['EF'].notna()]
        self.df = df.reset_index(drop=True)

        self.size      = int(cfg.data.frame_size)
        self.stride    = int(cfg.data.frame_stride)
        self.max_steps = int(cfg.data.max_steps)
        self.mean = np.array(cfg.data.normalize_mean,
                             dtype=np.float32).reshape(1, 1, 1, 3)
        self.std  = np.array(cfg.data.normalize_std,
                             dtype=np.float32).reshape(1, 1, 1, 3)

    def __len__(self):
        return len(self.df)

    def _window_start(self, T: int) -> int:
        """评估始终使用居中确定性窗口（可复现）。"""
        span = self.max_steps * self.stride
        if T <= span:
            return 0
        return int(max(0, (T - span) // 2))

    def _load(self, idx: int):
        row = self.df.iloc[idx]
        # FileName 已含 .avi，直接拼路径
        vid_path = os.path.join(self.videos_dir, str(row['FileName']))
        frames = read_video(vid_path, self.size)   # (T, H, W, 3) uint8
        T = frames.shape[0]

        start   = self._window_start(T)
        indices = np.arange(start, T, self.stride)[:self.max_steps]
        sub     = frames[indices].astype(np.float32) / 255.0
        sub     = (sub - self.mean) / self.std       # (T', H, W, 3)
        sub     = np.transpose(sub, (0, 3, 1, 2))    # (T', 3, H, W)

        return {
            'frames':   torch.from_numpy(np.ascontiguousarray(sub)),
            'ef':       float(row['EF']),
            'filename': str(row['FileName']),
        }

    def __getitem__(self, idx):
        for offset in range(len(self.df)):
            try:
                return self._load((idx + offset) % len(self.df))
            except Exception as e:  # noqa: BLE001
                if offset == 0:
                    print(f'[warn] load fail idx={idx}: {e}')
                continue
        raise RuntimeError('all samples failed to load')


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _pearson(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(pred) < 2:
        return float('nan')
    try:
        from scipy.stats import pearsonr
        r, _ = pearsonr(pred, gt)
        return float(r)
    except ImportError:
        pass
    pm = pred - pred.mean()
    gm = gt - gt.mean()
    denom = np.sqrt((pm ** 2).sum() * (gm ** 2).sum())
    return float(np.dot(pm, gm) / denom) if denom > 1e-12 else float('nan')


def _path_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    err = pred - gt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

    # 计算 AUC (心衰分类二分类标准：EF < 50%)
    y_true_hf = (gt < 50.0).astype(int)
    auc = float('nan')
    if roc_auc_score is not None and len(np.unique(y_true_hf)) > 1:
        # 使用 -pred 作为分数，因为预测 EF 越低，对应 EF < 50 的概率应该越高
        auc = float(roc_auc_score(y_true_hf, -pred))

    return {
        'mae': float(np.mean(np.abs(err))),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'r2': r2,
        'r': _pearson(pred, gt),
        'auc': auc,
        'within_5': float(np.mean(np.abs(err) <= 5.0)),
        'within_10': float(np.mean(np.abs(err) <= 10.0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 核心评估
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluate(model: MambaPolicyNet,
                 r21d_head: R21DCyclicEFHead,
                 loader,
                 device: str,
                 cfg,
                 cspo_phi: float,
                 cspo_dmin: float,
                 n_cspo: int,
                 max_samples: int = 0,
                 update: int = 99999):
    model.eval()
    r21d_head.eval()
    max_vol = float(cfg.model.max_volume_ml)
    gamma = float(cfg.rl.gamma)
    soft_tau = float(getattr(cfg.rl, 'soft_ef_tau', 6.0))

    ef_gt_all, ef_m_all, ef_t_all, ef_s_all, ef_r_all = [], [], [], [], []
    ed_count_all, es_count_all = [], []
    cycle_count_all = []

    # ITE 累积（4 角色类型）
    ite_keys = ['ED_to_skip', 'ES_to_skip', 'ED_to_ES', 'ES_to_ED']
    ite_accum = {k: [] for k in ite_keys}

    rows = []
    n_done = 0

    for batch in loader:
        if max_samples and n_done >= max_samples:
            break

        frames = batch['frames'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        ef_gt = batch['ef'].to(device)
        names = batch['names']

        # ── Mamba 三动作路径 ─────────────────────────────────────────────
        logits, volume, _ = model(frames, mask)
        logits = torch.nan_to_num(
            logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        volume = torch.nan_to_num(
            volume.float(), nan=max_vol * 0.5, posinf=max_vol, neginf=0.0
        ).clamp(0.0, max_vol)

        # 贪婪动作；padding 帧强制为 skip(0)
        actions_a = logits.argmax(dim=-1)
        actions_a = actions_a * mask.long()

        ef_mamba = compute_ef_from_3action(volume, actions_a, mask)

        # top-1 稳健解码：P(ED)/P(ES) 最高的单帧作 ED/ES，对 argmax 全 skip 免疫
        probs   = torch.softmax(logits, dim=-1)
        neg     = ~mask
        ed_top  = probs[..., 1].masked_fill(neg, -1.0).argmax(dim=1)
        es_top  = probs[..., 2].masked_fill(neg, -1.0).argmax(dim=1)
        ridx    = torch.arange(frames.shape[0], device=device)
        ef_top1 = ((volume[ridx, ed_top] - volume[ridx, es_top])
                   / volume[ridx, ed_top].clamp(min=1.0) * 100.0).clamp(0.0, 100.0)

        # 与策略无关的容积曲线 EF（Mamba 分支真实健康度）
        ef_soft = soft_curve_ef(volume, mask, tau=soft_tau)[0]

        # 轨迹 A 奖励（用于 ITE 计算）
        rewards_a, _ = compute_rewards(actions_a, volume, ef_mamba, ef_gt, mask, cfg)

        # ── R21D 多周期路径 ──────────────────────────────────────────────
        try:
            cyclic = r21d_head(frames, volume.detach(), mask, update=update)
            ef_r21d = torch.nan_to_num(
                cyclic.ef_final.float(), nan=50.0, posinf=100.0, neginf=0.0)
            cc = cyclic.cycle_count.float().cpu()
        except Exception as e:
            print(f'[warn] r21d_head error: {e}')
            ef_r21d = torch.full_like(ef_gt, 50.0)
            cc = torch.zeros(frames.shape[0])

        # ── CSPO ITE：对 n_cspo 条反事实轨迹求平均 ─────────────────────
        batch_ite = {k: [] for k in ite_keys}
        for _ in range(n_cspo):
            actions_b = forcediverge_3action(
                actions_a, mask, phi=cspo_phi, delta_min=cspo_dmin)
            ef_mamba_b = compute_ef_from_3action(volume, actions_b, mask)
            rewards_b, _ = compute_rewards(
                actions_b, volume, ef_mamba_b, ef_gt, mask, cfg)
            stats = role_ite_stats(
                actions_a, actions_b, rewards_a, rewards_b, mask, gamma)
            for k in ite_keys:
                if not np.isnan(stats[k]):
                    batch_ite[k].append(stats[k])

        # 平均 ITE（若无有效发散步则保留 nan）
        ite_mean = {}
        for k in ite_keys:
            ite_mean[k] = float(np.mean(batch_ite[k])) if batch_ite[k] else float('nan')
            if not np.isnan(ite_mean[k]):
                ite_accum[k].append(ite_mean[k])

        B = frames.shape[0]
        ef_gt_np = ef_gt.cpu().numpy()
        ef_m_np = ef_mamba.cpu().numpy()
        ef_t_np = ef_top1.cpu().numpy()
        ef_s_np = ef_soft.cpu().numpy()
        ef_r_np = ef_r21d.cpu().numpy()
        act_np = actions_a.cpu().numpy()
        msk_np = mask.cpu().numpy()
        cc_np = cc.numpy()

        for b in range(B):
            if max_samples and n_done >= max_samples:
                break

            T_b = int(msk_np[b].sum())
            n_ed = int((act_np[b] == 1).sum())
            n_es = int((act_np[b] == 2).sum())
            n_skip = T_b - n_ed - n_es

            ef_gt_all.append(float(ef_gt_np[b]))
            ef_m_all.append(float(ef_m_np[b]))
            ef_t_all.append(float(ef_t_np[b]))
            ef_s_all.append(float(ef_s_np[b]))
            ef_r_all.append(float(ef_r_np[b]))
            ed_count_all.append(n_ed)
            es_count_all.append(n_es)
            cycle_count_all.append(float(cc_np[b]))

            rows.append({
                'filename': names[b],
                'pred_ef_mamba': round(float(ef_m_np[b]), 4),
                'pred_ef_mamba_top1': round(float(ef_t_np[b]), 4),
                'pred_ef_soft': round(float(ef_s_np[b]), 4),
                'pred_ef_r21d': round(float(ef_r_np[b]), 4),
                'true_ef': round(float(ef_gt_np[b]), 4),
                'n_ed': n_ed,
                'n_es': n_es,
                'n_skip': n_skip,
                'total_frames': T_b,
                'cycle_count': int(cc_np[b]),
                'ite_ED_to_skip': round(ite_mean['ED_to_skip'], 5)
                if not np.isnan(ite_mean['ED_to_skip']) else '',
                'ite_ES_to_skip': round(ite_mean['ES_to_skip'], 5)
                if not np.isnan(ite_mean['ES_to_skip']) else '',
                'ite_ED_to_ES': round(ite_mean['ED_to_ES'], 5)
                if not np.isnan(ite_mean['ED_to_ES']) else '',
                'ite_ES_to_ED': round(ite_mean['ES_to_ED'], 5)
                if not np.isnan(ite_mean['ES_to_ED']) else '',
            })
            n_done += 1

    if not ef_gt_all:
        return {'n': 0}, []

    gt = np.array(ef_gt_all, dtype=np.float32)
    ef_m = np.array(ef_m_all, dtype=np.float32)
    ef_t = np.array(ef_t_all, dtype=np.float32)
    ef_s = np.array(ef_s_all, dtype=np.float32)
    ef_r = np.array(ef_r_all, dtype=np.float32)

    metrics = {
        'n': n_done,
        'soft': _path_metrics(ef_s, gt),
        'mamba': _path_metrics(ef_m, gt),
        'mamba_top1': _path_metrics(ef_t, gt),
        'r21d': _path_metrics(ef_r, gt),
        'ed_mean': float(np.mean(ed_count_all)),
        'es_mean': float(np.mean(es_count_all)),
        'cycle_mean': float(np.mean(cycle_count_all)),
        'ite': {
            k: float(np.mean(v)) if v else float('nan')
            for k, v in ite_accum.items()
        },
    }
    np.savez('results_echonet_cspo.npz', trues=gt, preds=ef_r)
    model.train()
    r21d_head.train()
    return metrics, rows


# ─────────────────────────────────────────────────────────────────────────────
# 输出格式化
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, split: str):
    m = metrics['mamba']
    mt = metrics.get('mamba_top1')
    ms = metrics.get('soft')
    r = metrics['r21d']
    ite = metrics['ite']
    bar = '=' * 116
    print(f'\n{bar}')
    print(f'  EchoNet-CSPO-mamba3ActorPriod-R21D-CSPO  |  Dataset=VideosA4C(Pediatric)  Split={split}  n={metrics["n"]}')
    print(bar)
    print(
        f'  {"":24s}  {"EF MAE":>8s}  {"EF RMSE":>8s}  {"EF R^2":>8s}  {"Pearson r":>10s}  {"AUC(EF<50)":>10s}  {"Acc(<=5%)":>10s}  {"Acc(<=10%)":>11s}')
    if ms is not None:
        print(
            f'  {"Soft-curve(no policy)":24s}  {ms["mae"]:>8.3f}  {ms["rmse"]:>8.3f}  {ms["r2"]:>8.4f}  {ms["r"]:>10.4f}  {ms["auc"]:>10.4f}  {ms["within_5"] * 100:>9.1f}%  {ms["within_10"] * 100:>10.1f}%')
    print(
        f'  {"Mamba(3-action argmax)":24s}  {m["mae"]:>8.3f}  {m["rmse"]:>8.3f}  {m["r2"]:>8.4f}  {m["r"]:>10.4f}  {m["auc"]:>10.4f}  {m["within_5"] * 100:>9.1f}%  {m["within_10"] * 100:>10.1f}%')
    if mt is not None:
        print(
            f'  {"Mamba(top-1 ED/ES)":24s}  {mt["mae"]:>8.3f}  {mt["rmse"]:>8.3f}  {mt["r2"]:>8.4f}  {mt["r"]:>10.4f}  {mt["auc"]:>10.4f}  {mt["within_5"] * 100:>9.1f}%  {mt["within_10"] * 100:>10.1f}%')
    print(
        f'  {"R21D cyclic":24s}  {r["mae"]:>8.3f}  {r["rmse"]:>8.3f}  {r["r2"]:>8.4f}  {r["r"]:>10.4f}  {r["auc"]:>10.4f}  {r["within_5"] * 100:>9.1f}%  {r["within_10"] * 100:>10.1f}%')
    print(bar)
    print(f'  Mean ED frames   = {metrics["ed_mean"]:.2f}')
    print(f'  Mean ES frames   = {metrics["es_mean"]:.2f}')
    print(f'  Mean R21D cycles = {metrics["cycle_mean"]:.2f}')
    print(bar)
    print(f'  CSPO ITE 角色归因（平均折扣回报差 ΔG = G_A - G_B）')

    def _ite_str(v):
        return f'{v:+.4f}' if not np.isnan(v) else '   N/A '

    print(f'  {"ED→skip  (选ED的边际贡献)":32s}  {_ite_str(ite["ED_to_skip"])}')
    print(f'  {"ES→skip  (选ES的边际贡献)":32s}  {_ite_str(ite["ES_to_skip"])}')
    print(f'  {"ED→ES    (ED角色 vs ES角色)":32s}  {_ite_str(ite["ED_to_ES"])}')
    print(f'  {"ES→ED    (ES角色 vs ED角色)":32s}  {_ite_str(ite["ES_to_ED"])}')
    print(f'{bar}\n')


def save_csv(rows: list, path: str):
    if not rows:
        print('[warn] no rows to save')
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[info] CSV ({len(rows)} rows) → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if roc_auc_score is None:
        print(
            '[warn] sklearn is not installed. AUC calculation will be skipped. Run `pip install scikit-learn` to enable it.')

    ap = argparse.ArgumentParser(
        description='EchoNet-CSPO-mamba3ActorPriod-R21D-derectEF-CSPO  Pediatric(VideosA4C) 评估脚本')
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'default.yaml'))
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--split', default='ALL',
                    help='ALL = 全部数据（默认）；0~9 = 指定 10-fold 折编号')
    ap.add_argument('--filelist', default=None,
                    help='覆盖 config 的 data.filelist，默认 ../../data/VideosA4C/FileList_with_frames.csv')
    ap.add_argument('--videos_dir', default=None,
                    help='覆盖 config 的 data.videos_dir，默认 ../../data/VideosA4C/VideosA4C')
    ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--save_csv', default=None)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--n_cspo', type=int, default=5,
                    help='FORCEDIVERGE 重复次数，ITE 取均值（越多越稳，默认5）')
    args = ap.parse_args()

    cfg, _ = load_config(args.config)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[info] device={device}')

    # ── VideosA4C 数据路径（CLI > config > 默认值）────────────────────────
    _default_filelist   = os.path.join(ROOT, '..', '..', 'data', 'VideosA4C',
                                       'FileList_with_frames.csv')
    _default_videos_dir = os.path.join(ROOT, '..', '..', 'data', 'VideosA4C', 'VideosA4C')

    filelist   = args.filelist   or getattr(cfg.data, 'filelist',   _default_filelist)
    videos_dir = args.videos_dir or getattr(cfg.data, 'videos_dir', _default_videos_dir)

    # 将相对路径解析为绝对路径（相对 scripts/ 目录的上上层，即项目根的同级）
    if not os.path.isabs(filelist):
        filelist = os.path.normpath(os.path.join(ROOT, filelist))
    if not os.path.isabs(videos_dir):
        videos_dir = os.path.normpath(os.path.join(ROOT, videos_dir))

    print(f'[info] filelist   = {filelist}')
    print(f'[info] videos_dir = {videos_dir}')

    # CSPO 参数（沿用训练配置）
    from types import SimpleNamespace as SimpleNS
    cspo_cfg = getattr(cfg, 'cspo', SimpleNS())
    cspo_phi = float(getattr(cspo_cfg, 'phi', 0.25))
    cspo_dmin = float(getattr(cspo_cfg, 'delta_min', 0.10))
    print(f'[info] CSPO phi={cspo_phi}  delta_min={cspo_dmin}  n_cspo={args.n_cspo}')

    nw = args.workers if args.workers is not None else int(cfg.train.num_workers)

    # ── 使用 VideosA4C 专用数据集（处理三点格式差异）────────────────────
    ds = VideosA4CPediatricDataset(filelist, videos_dir, cfg, split_val=args.split)
    print(f'[info] Pediatric split={args.split!r}  samples: {len(ds)}')

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=nw, collate_fn=collate_episodes_nolabel,
        pin_memory=(device == 'cuda'), persistent_workers=(nw > 0),
    )

    # ── 模型 ─────────────────────────────────────────────────────────────
    model = MambaPolicyNet(cfg).to(device)
    ef_cfg = getattr(cfg.model, 'r2plus1d_ef', None)

    def _g(k, d):
        return getattr(ef_cfg, k, d) if ef_cfg else d

    r21d_head = R21DCyclicEFHead(
        pretrained=bool(_g('pretrained', True)), clip_len=int(_g('clip_len', 32)),
        dropout=float(cfg.model.dropout), use_checkpoint=bool(_g('use_checkpoint', True)),
        ef_mean=float(_g('ef_mean', 55.0)), ef_scale=float(_g('ef_scale', 35.0)),
        use_formula_head=bool(_g('use_formula_head', False)),
        vol_edv_mean=float(_g('vol_edv_mean', 100.0)), vol_edv_scale=float(_g('vol_edv_scale', 80.0)),
        vol_esv_mean=float(_g('vol_esv_mean', 45.0)), vol_esv_scale=float(_g('vol_esv_scale', 40.0)),
        ensemble_alpha_init=float(_g('ensemble_alpha_init', 0.5)),
        min_cycle_frames=int(_g('min_cycle_frames', 20)), max_cycles=int(_g('max_cycles', 4)),
        target_cycle_frames=int(_g('target_cycle_frames', 64)),
        cycle_quality_weighting=bool(_g('cycle_quality_weighting', True)),
        cycle_detect_warmup=int(_g('cycle_detect_warmup', 2000)),
        cycle_smooth_sigma=float(_g('cycle_smooth_sigma', 3.0)),
        cycle_prominence_ratio=float(_g('cycle_prominence_ratio', 0.15)),
        span_margin=int(_g('span_margin', 6)),
    ).to(device)

    ckpt_path = args.ckpt or os.path.join(cfg.train.checkpoint_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}\n'
            f'  请先运行 train.py 生成 best.pt，或用 --ckpt 指定路径')

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_sd = ck.get('model') or ck.get('ema') or ck
    miss, unexp = model.load_state_dict(model_sd, strict=False)
    print(f'[info] model:  missing={len(miss)}  unexpected={len(unexp)}')
    if miss:
        print(f'       missing (first 5): {miss[:5]}')

    if 'r21d' in ck:
        miss2, unexp2 = r21d_head.load_state_dict(ck['r21d'], strict=False)
        print(f'[info] r21d:   missing={len(miss2)}  unexpected={len(unexp2)}')
    else:
        print('[warn] no "r21d" key in checkpoint; r21d_head uses random weights')

    update = int(ck.get('update', 99999))
    bv_r = ck.get('best_val_r21d', '?')
    bv_m = ck.get('best_val_mamba', '?')
    bv_r_s = f'{bv_r:.4f}' if isinstance(bv_r, float) else str(bv_r)
    bv_m_s = f'{bv_m:.4f}' if isinstance(bv_m, float) else str(bv_m)
    print(f'       update={update}  best_val_r21d={bv_r_s}  best_val_mamba={bv_m_s}')

    print(f'[info] running evaluation on {args.split}...')
    metrics, rows = run_evaluate(
        model, r21d_head, loader, device, cfg,
        cspo_phi=cspo_phi, cspo_dmin=cspo_dmin,
        n_cspo=args.n_cspo,
        max_samples=args.max, update=update)

    print_metrics(metrics, args.split)

    if args.save_csv:
        save_csv(rows, args.save_csv)


if __name__ == '__main__':
    main()