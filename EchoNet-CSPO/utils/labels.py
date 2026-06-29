"""伪标签构造工具 (Phase Classifier 配套, 2026-05-01)。"""
from typing import Tuple
import torch


def build_phase_labels(
    ed_idx: torch.Tensor,
    es_idx: torch.Tensor,
    B: int,
    T: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造每帧 phase 伪标签 + 有效性 mask。

    标签: 0=pre-cycle, 1=systole (ED→ES, 容积下降), 2=diastole (ES→ED, 容积上升)。

    Returns
    -------
    labels : (B,T) long  ∈ {0, 1, 2}
    valid  : (B,T) bool  样本两索引都有效的位置为 True
    """
    labels = torch.zeros(B, T, dtype=torch.long, device=device)
    valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    ed_cpu = ed_idx.detach().cpu().tolist()
    es_cpu = es_idx.detach().cpu().tolist()
    for b in range(B):
        ed = int(ed_cpu[b])
        es = int(es_cpu[b])
        if ed < 0 or ed >= T or es < 0 or es >= T:
            continue
        valid[b, :] = True
        if ed < es:
            labels[b, :ed] = 0
            labels[b, ed:es] = 1
            labels[b, es:] = 2
        elif ed > es:
            labels[b, :ed] = 2
            labels[b, ed:] = 1
        else:
            valid[b, :] = False
    return labels, valid
