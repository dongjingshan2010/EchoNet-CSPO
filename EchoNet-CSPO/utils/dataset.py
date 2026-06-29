"""EchoNet-Dynamic 数据集（无标签模式）。

核心改变（相比原版）
──────────────────
* no_label_mode=True（默认）：窗口完全随机采样，不依赖 ED_Frame/ES_Frame。
  算法设计不依赖关键帧位置标签，因此训练时的窗口由随机起点决定。
  验证/测试时使用确定性居中窗口（取视频中段）保证可复现性。

* collate_fn（collate_episodes_nolabel）：batch 中不包含 ed_idx / es_idx，
  只返回 frames / mask / ef / filename。EF 标量值仍用于监督训练和评估。

* 兼容性：EchoEpisodeDataset 保留原版逻辑（no_label_mode=False），
  供需要关键帧信息的旧代码继续使用。
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .video import read_video


# ─────────────────────────────────────────────────────────────────────────────
# 无标签模式数据集（本项目主用）
# ─────────────────────────────────────────────────────────────────────────────

class EchoNoLabelDataset(Dataset):
    """仅依赖 EF 标量值，不使用 ED_Frame / ES_Frame / EDV / ESV 标签。

    窗口选择策略
    ────────────
    TRAIN : 随机起点（每次 epoch 不同，增加多样性）
    VAL   : 固定居中窗口（可复现）
    TEST  : 固定居中窗口（可复现）
    """

    def __init__(self, cfg, split: str = 'TRAIN'):
        self.cfg   = cfg
        self.split = split

        df = pd.read_csv(cfg.data.filelist)
        df = df[df['Split'] == split].reset_index(drop=True)
        df = df[df['NumberOfFrames'] >= cfg.data.min_frames_keep]
        # 只需要 EF 不为空
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
        """随机（TRAIN）或居中（VAL/TEST）窗口起点，不依赖关键帧位置。"""
        span = self.max_steps * self.stride
        if T <= span:
            return 0
        if self.split == 'TRAIN':
            return int(np.random.randint(0, T - span + 1))
        # VAL/TEST: 居中
        return int(max(0, (T - span) // 2))

    def _load(self, idx: int):
        row     = self.df.iloc[idx]
        vid_path = os.path.join(
            self.cfg.data.videos_dir, str(row['FileName']) + '.avi')
        frames  = read_video(vid_path, self.size)   # (T, H, W, 3) uint8
        T       = frames.shape[0]

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


def collate_episodes_nolabel(batch):
    """EchoNoLabelDataset 的 collate_fn。

    返回字段: frames (B,T,C,H,W), mask (B,T), ef (B,), names list[str]
    不含 ed_idx / es_idx / edv / esv（算法不使用这些标签）。
    """
    B     = len(batch)
    max_T = max(b['frames'].shape[0] for b in batch)
    C, H, W = batch[0]['frames'].shape[1:]

    frames = torch.zeros(B, max_T, C, H, W, dtype=torch.float32)
    mask   = torch.zeros(B, max_T, dtype=torch.bool)
    ef     = torch.zeros(B, dtype=torch.float32)
    names  = []

    for i, b in enumerate(batch):
        T = b['frames'].shape[0]
        frames[i, :T] = b['frames']
        mask[i, :T]   = True
        ef[i]         = b['ef']
        names.append(b['filename'])

    return {'frames': frames, 'mask': mask, 'ef': ef, 'names': names}


# ─────────────────────────────────────────────────────────────────────────────
# 原版数据集（保留，供兼容旧代码）
# ─────────────────────────────────────────────────────────────────────────────

class EchoEpisodeDataset(Dataset):
    """原版数据集（依赖 ED_Frame/ES_Frame/EDV/ESV 做窗口选择和监督）。
    保留供向后兼容；本项目新代码请使用 EchoNoLabelDataset。
    """

    def __init__(self, cfg, split: str = 'TRAIN'):
        self.cfg   = cfg
        self.split = split
        df = pd.read_csv(cfg.data.filelist)
        df = df[df['Split'] == split].reset_index(drop=True)
        df = df[df['NumberOfFrames'] >= cfg.data.min_frames_keep]
        df = df[df['ED_Frame'].notna() & df['ES_Frame'].notna()]
        df = df[df['EDV'].notna() & df['ESV'].notna() & df['EF'].notna()]
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

    def _window_start(self, T: int, ed: int, es: int) -> int:
        span   = self.max_steps * self.stride
        if T <= span:
            return 0
        key_lo = min(ed, es)
        key_hi = max(ed, es)
        if self.split == 'TRAIN':
            lo = max(0, key_hi - span + self.stride)
            hi = min(T - span, key_lo)
            if hi <= lo:
                return max(0, min(T - span, key_lo))
            return int(np.random.randint(lo, hi + 1))
        mid = (key_lo + key_hi) // 2
        return int(max(0, min(T - span, mid - span // 2)))

    def _load(self, idx: int):
        row      = self.df.iloc[idx]
        vid_path = os.path.join(
            self.cfg.data.videos_dir, str(row['FileName']) + '.avi')
        frames   = read_video(vid_path, self.size)
        T        = frames.shape[0]
        ed = int(row['ED_Frame']); es = int(row['ES_Frame'])
        start    = self._window_start(T, ed, es)
        indices  = np.arange(start, T, self.stride)[:self.max_steps]
        sub      = frames[indices].astype(np.float32) / 255.0
        sub      = (sub - self.mean) / self.std
        sub      = np.transpose(sub, (0, 3, 1, 2))

        half   = self.stride // 2
        ed_rel = es_rel = -1
        for i, g in enumerate(indices):
            gi = int(g)
            if abs(gi - ed) <= half: ed_rel = i
            if abs(gi - es) <= half: es_rel = i

        return {
            'frames':   torch.from_numpy(np.ascontiguousarray(sub)),
            'ed_idx':   ed_rel,
            'es_idx':   es_rel,
            'edv':      float(row['EDV']),
            'esv':      float(row['ESV']),
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


def collate_episodes(batch):
    """原版 collate（含 ed_idx/es_idx/edv/esv）。"""
    B     = len(batch)
    max_T = max(b['frames'].shape[0] for b in batch)
    C, H, W = batch[0]['frames'].shape[1:]
    frames = torch.zeros(B, max_T, C, H, W, dtype=torch.float32)
    mask   = torch.zeros(B, max_T, dtype=torch.bool)
    lens   = torch.zeros(B, dtype=torch.long)
    ed_idx = torch.full((B,), -1, dtype=torch.long)
    es_idx = torch.full((B,), -1, dtype=torch.long)
    edv    = torch.zeros(B, dtype=torch.float32)
    esv    = torch.zeros(B, dtype=torch.float32)
    ef     = torch.zeros(B, dtype=torch.float32)
    names  = []
    for i, b in enumerate(batch):
        T = b['frames'].shape[0]
        frames[i, :T] = b['frames']
        mask[i, :T] = True
        lens[i]   = T
        ed_idx[i] = b['ed_idx']
        es_idx[i] = b['es_idx']
        edv[i]    = b['edv']
        esv[i]    = b['esv']
        ef[i]     = b['ef']
        names.append(b['filename'])
    return {
        'frames': frames, 'mask': mask, 'lens': lens,
        'ed_idx': ed_idx, 'es_idx': es_idx,
        'edv': edv, 'esv': esv, 'ef': ef, 'names': names,
    }
