"""奖励函数 — 三动作版（EchoNet-CSPO-mamba3ActorPriod-R21D-derectEF）。

动作空间
────────
  0 = skip      跳过（不参与 EF 计算）
  1 = select_as_ED  标记为舒张末期帧（高容积，贡献 EDV）
  2 = select_as_ES  标记为收缩末期帧（低容积，贡献 ESV）

EF 计算（三动作公式）
─────────────────────
  EDV = mean( volume[t]  for t where action[t]==1 & valid )
  ESV = mean( volume[t]  for t where action[t]==2 & valid )
  EF  = (EDV - ESV) / EDV × 100%

  fallback: 若无 ED 帧 → EDV = max(all valid vol)
            若无 ES 帧 → ESV = min(all valid vol)

与二值版本的本质区别
─────────────────────
  选帧决定角色（EDV vs ESV），而非仅决定是否参与聚合。
  多帧标注同一角色可提升估计鲁棒性（均值 vs 单点）。
  agent 必须学会区分心动周期中的高/低容积时相。

奖励组成（5 项）
────────────────
1. r_ef       [-w_ef × |EF_mamba - EF_gt|]     终端，EF 精度（主信号，自适应权重）
2. r_over     [-w_over × (action>0)]            每步稀疏惩罚（鼓励少选）
3. r_cont     [-w_cont × vol_jump]              容积连续性（平滑曲线）
4. r_min_ed   [-w_min_ed]                       终端，未选任何 ED 帧
5. r_min_es   [-w_min_es]                       终端，未选任何 ES 帧
6. r_role_sep [-w_role_sep × max(0,ESV-EDV)/max_vol]
                                                终端，角色混淆惩罚（ESV > EDV 物理违反）

设计约束
────────
全程不依赖 ED_Frame / ES_Frame 帧索引，不依赖关键帧容积标签。
"""
from typing import Tuple
import torch


# ─────────────────────────────────────────────────────────────────────────────
# EF 公式计算（三动作版）
# ─────────────────────────────────────────────────────────────────────────────

def compute_ef_from_3action(
    volumes: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """三动作 EF (%) 计算。

    EDV = mean(vol[action==1 & valid]);  fallback → max(all valid vol)
    ESV = mean(vol[action==2 & valid]);  fallback → min(all valid vol)
    EF  = (EDV - ESV) / EDV × 100%

    Parameters
    ----------
    volumes : (B, T) float  逐帧容积预测 (mL)
    actions : (B, T) long   RL 动作 {0,1,2}
    mask    : (B, T) bool   有效帧掩码

    Returns
    -------
    (B,) float  EF 估计值 [0%, 100%]
    """
    volumes = torch.nan_to_num(
        volumes, nan=50.0, posinf=300.0, neginf=0.0).clamp(0.0, 300.0)

    sel_ed = (actions == 1) & mask   # ED 候选帧 (B, T)
    sel_es = (actions == 2) & mask   # ES 候选帧 (B, T)

    has_ed = sel_ed.any(dim=1)       # (B,) bool
    has_es = sel_es.any(dim=1)

    # ── ESV：ES 帧容积均值；无 ES 帧时退化到全局容积均值（V1.5 修复）──
    # 旧方案：fallback = 全局 min ≈ 真实 ESV → EF 误差极小 → 策略「不选 ES 也无损」
    # 新方案：fallback = 全局均值（偏高，高于真实 ESV）
    #   → EF = (EDV - mean) / EDV < 真实 EF → EF 系统偏低 → r_ef 惩罚增大
    #   → 联合 w_min_es 终端惩罚，双重逼迫策略真正选出 ES 帧
    es_cnt   = sel_es.float().sum(dim=1).clamp(min=1.0)
    esv_sel  = (volumes * sel_es.float()).sum(dim=1) / es_cnt
    n_valid_vol = mask.float().sum(dim=1).clamp(min=1.0)
    esv_all  = (volumes * mask.float()).sum(dim=1) / n_valid_vol   # 全局均值作 fallback
    esv      = torch.where(has_es, esv_sel, esv_all)

    # ── EDV：ED 帧容积均值；无 ED 帧时 EDV=ESV → EF=0%（v5 移植：堵死 fallback 漏洞）──
    # 旧版无 ED 帧退化到全局 max，等于让策略"不选 ED 也能拿到合理 EF"，是塌缩诱因；
    # 改为 EDV=ESV 后无 ED 帧直接得 EF=0，强逼策略真正选出 ED 帧。
    ed_cnt  = sel_ed.float().sum(dim=1).clamp(min=1.0)
    edv_sel = (volumes * sel_ed.float()).sum(dim=1) / ed_cnt
    edv     = torch.where(has_ed, edv_sel, esv)

    ef = (edv - esv) / edv.clamp(min=1.0) * 100.0
    return torch.nan_to_num(ef, nan=0.0, posinf=100.0, neginf=0.0).clamp(0.0, 100.0)


# 向后兼容别名（供旧代码导入）
def compute_ef_from_selection(volumes, actions, mask):
    """向后兼容：二值动作(0/1)时等价于原 max/min 公式，三值动作转发至 3action 版。"""
    max_act = actions.max().item() if actions.numel() > 0 else 0
    if max_act <= 1:
        # 二值模式：保持原语义 max/min
        volumes = torch.nan_to_num(
            volumes, nan=50.0, posinf=300.0, neginf=0.0).clamp(0.0, 300.0)
        sel     = (actions == 1) & mask
        has_any = sel.any(dim=1, keepdim=True)
        eff     = torch.where(has_any, sel, mask)
        neg_inf = torch.full_like(volumes, float('-inf'))
        pos_inf = torch.full_like(volumes, float('inf'))
        vmax    = torch.where(eff, volumes, neg_inf).max(dim=1).values
        vmin    = torch.where(eff, volumes, pos_inf).min(dim=1).values
        ef      = (vmax - vmin) / vmax.clamp(min=1.0) * 100.0
        return torch.nan_to_num(ef, nan=50.0, posinf=100.0, neginf=0.0).clamp(0.0, 100.0)
    return compute_ef_from_3action(volumes, actions, mask)


# ─────────────────────────────────────────────────────────────────────────────
# 主奖励函数（三动作版）
# ─────────────────────────────────────────────────────────────────────────────

def compute_rewards(
    actions: torch.Tensor,
    volumes: torch.Tensor,
    ef_mamba: torch.Tensor,
    ef_gt: torch.Tensor,
    mask: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算每步奖励（三动作版，无 ED/ES 标签）。

    Parameters
    ----------
    actions  : (B, T) long {0,1,2}
    volumes  : (B, T) float  Mamba volume_head 预测容积
    ef_mamba : (B,)  float   compute_ef_from_3action 的结果
    ef_gt    : (B,)  float   Ground-truth EF
    mask     : (B, T) bool   有效帧掩码
    cfg      : 配置命名空间

    Returns
    -------
    rewards  : (B, T) float  每步奖励（padding 位为 0）
    ef_mamba : (B,)  float   透传（供日志使用）
    """
    device = actions.device
    B, T   = actions.shape
    mask_f = mask.float().to(device)
    ef_gt  = ef_gt.to(device)

    rcfg = cfg.reward
    w_ef              = float(rcfg.w_ef)
    w_over            = float(rcfg.w_over)
    w_cont            = float(rcfg.w_cont)
    cont_thresh_ratio = float(rcfg.cont_threshold_ratio)
    w_min_ed          = float(getattr(rcfg, 'w_min_ed',    0.5))
    w_min_es          = float(getattr(rcfg, 'w_min_es',    0.5))
    w_role_sep        = float(getattr(rcfg, 'w_role_sep',  0.15))
    w_max_es          = float(getattr(rcfg, 'w_max_es',     0.0))   # v5: ES 过选惩罚
    max_es_ratio      = float(getattr(rcfg, 'max_es_ratio', 0.15))
    w_max_ed          = float(getattr(rcfg, 'w_max_ed',     0.0))   # v5: ED 过选惩罚
    max_ed_ratio      = float(getattr(rcfg, 'max_ed_ratio', 0.15))
    adaptive_ef       = bool( getattr(rcfg, 'adaptive_ef_enabled',    True))
    adaptive_tanh_s   = float(getattr(rcfg, 'adaptive_ef_tanh_scale', 12.0))
    adaptive_max_s    = float(getattr(rcfg, 'adaptive_ef_max_scale',  1.5))
    max_vol           = float(cfg.model.max_volume_ml)

    volumes_safe = torch.nan_to_num(
        volumes.float().to(device), nan=max_vol * 0.5, posinf=max_vol, neginf=0.0
    ).clamp(0.0, max_vol)
    ef_safe = torch.nan_to_num(
        ef_mamba.float().to(device), nan=50.0, posinf=100.0, neginf=0.0
    )

    rewards = torch.zeros(B, T, device=device)
    act_f   = actions.float().to(device)

    # ── 1. 稀疏惩罚：action > 0（ED 或 ES 均计入）────────────────────
    sel_any = (actions > 0).float().to(device)
    rewards = rewards - w_over * sel_any * mask_f

    # ── 2. 容积连续性惩罚 ─────────────────────────────────────────────
    thr      = cont_thresh_ratio * max_vol
    dv       = torch.abs(volumes_safe[:, 1:] - volumes_safe[:, :-1])
    cont_pen = torch.clamp(dv - thr, min=0.0)
    pair_m   = (mask[:, 1:] & mask[:, :-1]).float().to(device)
    rewards[:, 1:] = rewards[:, 1:] - w_cont * cont_pen * pair_m

    # ── 3. 终端奖励（加到最后一个有效位置）─────────────────────────
    last_valid = mask.float().cumsum(dim=1).argmax(dim=1)   # (B,)
    row_idx    = torch.arange(B, device=device)

    ef_err = torch.abs(ef_safe - ef_gt)

    # 自适应 EF 权重
    if adaptive_ef:
        tanh_v = torch.tanh(ef_err / max(1e-6, adaptive_tanh_s))
        w_ef_v = w_ef * (1.0 + (adaptive_max_s - 1.0) * tanh_v)
    else:
        w_ef_v = torch.full_like(ef_err, w_ef)

    # r_ef：EF 精度惩罚
    rewards[row_idx, last_valid] -= w_ef_v * ef_err

    # r_min_ed：无 ED 帧惩罚
    sel_ed    = (actions == 1) & mask
    no_ed     = (~sel_ed.any(dim=1)).float()
    rewards[row_idx, last_valid] -= w_min_ed * no_ed

    # r_min_es：无 ES 帧惩罚
    sel_es    = (actions == 2) & mask
    no_es     = (~sel_es.any(dim=1)).float()
    rewards[row_idx, last_valid] -= w_min_es * no_es

    # r_role_sep：角色混淆惩罚（mean(vol[ES]) > mean(vol[ED])）
    if w_role_sep > 0.0:
        ed_cnt   = sel_ed.float().sum(dim=1).clamp(min=1.0)
        es_cnt   = sel_es.float().sum(dim=1).clamp(min=1.0)
        edv_mean = (volumes_safe * sel_ed.float()).sum(dim=1) / ed_cnt
        esv_mean = (volumes_safe * sel_es.float()).sum(dim=1) / es_cnt
        # 若无 ED/ES 帧，用全局极值代替（避免惩罚 fallback 情况）
        has_ed   = sel_ed.any(dim=1).float()
        has_es   = sel_es.any(dim=1).float()
        role_pen = torch.clamp(esv_mean - edv_mean, min=0.0) / max_vol
        # 只有同时选了 ED 和 ES 帧时才施加角色惩罚
        both_sel = has_ed * has_es
        rewards[row_idx, last_valid] -= w_role_sep * role_pen * both_sel

    # r_max_es / r_max_ed：过选惩罚（v5 移植，超过 ratio 的部分线性惩罚，反塌缩）
    T_valid = mask.float().sum(dim=1).clamp(min=1.0)
    if w_max_es > 0.0:
        es_ratio = sel_es.float().sum(dim=1) / T_valid
        rewards[row_idx, last_valid] -= w_max_es * torch.clamp(es_ratio - max_es_ratio, min=0.0)
    if w_max_ed > 0.0:
        ed_ratio = sel_ed.float().sum(dim=1) / T_valid
        rewards[row_idx, last_valid] -= w_max_ed * torch.clamp(ed_ratio - max_ed_ratio, min=0.0)

    # 清零 padding 位
    rewards = rewards * mask_f
    return rewards, ef_safe


# ─────────────────────────────────────────────────────────────────────────────
# CSPO：FORCEDIVERGE 反事实轨迹生成（三动作版）
# ─────────────────────────────────────────────────────────────────────────────

def forcediverge_3action(
    actions: torch.Tensor,   # (B, T) long {0,1,2}  轨迹 A 的动作
    mask: torch.Tensor,      # (B, T) bool            有效帧掩码
    phi: float = 0.25,       # 每个有效帧翻转概率
    delta_min: float = 0.10, # 保证至少 delta_min 比例的有效帧动作不同
) -> torch.Tensor:
    """生成反事实轨迹 B（FORCEDIVERGE 算法，三动作版）。

    步骤
    ────
    1. 对每个有效帧，以概率 phi 随机替换为另一个动作（均匀采样 ≠ 原动作）。
    2. 若翻转后的发散比例 < delta_min，继续随机强制翻转有效帧直到满足阈值。

    Parameters
    ----------
    actions   : (B, T) long  轨迹 A 动作序列
    mask      : (B, T) bool  有效帧（padding=False）
    phi       : 每步翻转概率
    delta_min : 最小发散比例保证

    Returns
    -------
    actions_b : (B, T) long  轨迹 B 动作序列（仅有效帧可能被翻转）
    """
    device = actions.device
    B, T   = actions.shape

    actions_b = actions.clone()

    # ── Phase 1：随机翻转 ────────────────────────────────────────────────
    flip_mask = (torch.rand(B, T, device=device) < phi) & mask

    # 对每个需要翻转的位置，从 {0,1,2}\{a_t} 中均匀采样
    # 策略：先随机 offset ∈ {1,2}，再 (a + offset) % 3 保证 ≠ 原动作
    offset = torch.randint(1, 3, (B, T), device=device)   # {1,2}
    alt_actions = (actions + offset) % 3
    actions_b = torch.where(flip_mask, alt_actions, actions_b)

    # ── Phase 2：强制满足 delta_min ──────────────────────────────────────
    valid_count  = mask.float().sum(dim=1).clamp(min=1.0)   # (B,)
    diff_count   = ((actions_b != actions) & mask).float().sum(dim=1)  # (B,)
    div_ratio    = diff_count / valid_count                             # (B,)

    need_more = div_ratio < delta_min   # (B,) bool

    if need_more.any():
        # 对需要补充翻转的样本，随机打乱有效帧顺序，依次翻转直到满足阈值
        for b in range(B):
            if not need_more[b]:
                continue
            target_count = int(delta_min * valid_count[b].item()) + 1
            current_diff = int(diff_count[b].item())
            if current_diff >= target_count:
                continue
            # 找出还未被翻转的有效帧
            unchanged = ((actions_b[b] == actions[b]) & mask[b]).nonzero(as_tuple=False).squeeze(1)
            if unchanged.numel() == 0:
                continue
            perm  = unchanged[torch.randperm(unchanged.numel(), device=device)]
            need  = target_count - current_diff
            force = perm[:need]
            offset_f = torch.randint(1, 3, (force.numel(),), device=device)
            actions_b[b, force] = (actions[b, force] + offset_f) % 3

    # 确保 padding 位不被改变
    actions_b = torch.where(mask, actions_b, actions)
    return actions_b
