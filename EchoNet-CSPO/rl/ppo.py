"""PPO 损失函数 + 监督损失集合（EchoNet-CSPO-mambaPriod-R21D-derectEF）。

包含两类损失：

A. RL 损失（Mamba 分支）
   compute_gae           : GAE 优势估计
   ppo_policy_loss       : PPO clip + 熵正则
   value_loss            : Critic MSE
   volume_smoothness_loss: 容积曲线 TV 正则化（配合 r_cont 奖励双重约束）

B. 监督损失
   mamba_ef_supervised_loss : SmoothL1(EF_mamba, EF_gt) — Mamba 直接监督
   r21d_cyclic_ef_loss      : SmoothL1(EF_r21d_avg, EF_gt) — R21D 主损失
   r21d_per_cycle_ef_loss   : mean(SmoothL1(EF_cycle_i, EF_gt)) — 每周期一致性
"""
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


# ─────────────────────────────────────────────────────────────────────────────
# A. RL 损失（Mamba 分支）
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
):
    """广义优势估计（GAE-λ）。

    Returns: (adv, ret)  都是 (B, T) float
    """
    B, T   = rewards.shape
    device = rewards.device
    adv    = torch.zeros_like(rewards)
    last   = torch.zeros(B, device=device)
    mf     = mask.float()

    for t in reversed(range(T)):
        nv    = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=device)
        nm    = mf[:, t + 1]     if t + 1 < T else torch.zeros(B, device=device)
        delta = rewards[:, t] + gamma * nv * nm - values[:, t]
        last  = delta + gamma * lam * nm * last
        adv[:, t] = last

    return adv, adv + values


def ppo_policy_loss(
    logits: torch.Tensor,
    old_logprobs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
    entropy_coef: float,
):
    """PPO-clip 策略损失 + 熵正则。Returns: (policy_loss, entropy_loss)。"""
    dist   = Categorical(logits=logits)
    new_lp = dist.log_prob(actions)
    ent    = dist.entropy()
    ratio  = torch.exp(new_lp - old_logprobs)
    s1     = ratio * advantages
    s2     = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    m      = mask.float()
    d      = m.sum().clamp(min=1.0)
    pol_l  = (-torch.min(s1, s2) * m).sum() / d
    ent_l  = (-entropy_coef * ent * m).sum() / d
    return pol_l, ent_l


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Critic MSE 损失（有效位平均）。"""
    m = mask.float()
    return ((values - returns) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def volume_smoothness_loss(
    volumes: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Total Variation 正则化：|v_{t+1} - v_t|² 均值（有效相邻帧对）。

    配合 r_cont RL 奖励从两个方向约束容积曲线平滑度。
    """
    dv        = volumes[:, 1:] - volumes[:, :-1]           # (B, T-1)
    pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
    return (dv ** 2 * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)


def volume_curvature_loss(
    volumes: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """二阶差分（曲率）正则化：|v_{t+1} - 2·v_t + v_{t-1}|² 均值。

    动机（v5 Fix C，移植自 mamba3Actor）：心动周期容积是平滑的「升—降」曲线，其一阶差分非零，
    用一阶 TV (volume_smoothness_loss) 惩罚一阶差分会把 ED–ES 摆幅抹平——而这恰是
    volume head 必须产生、且 R21D 周期检测赖以工作的信号。二阶差分只惩罚「逐帧抖动/曲率突变」，
    对恒定斜率（平滑升降）零惩罚，允许容积曲线自由形成 ED 峰与 ES 谷。
    """
    d2  = volumes[:, 2:] - 2.0 * volumes[:, 1:-1] + volumes[:, :-2]   # (B, T-2)
    m3  = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
    return (d2 ** 2 * m3).sum() / m3.sum().clamp(min=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# B. 监督损失
# ─────────────────────────────────────────────────────────────────────────────

def mamba_ef_supervised_loss(
    ef_mamba: torch.Tensor,
    ef_gt: torch.Tensor,
) -> torch.Tensor:
    """SmoothL1(EF_mamba, EF_gt)：Mamba 分支 EF 直接监督。

    使 volume_head 独立于稀疏 RL 奖励也能学到正确的容积尺度。
    """
    ef_m = torch.nan_to_num(ef_mamba.float(), nan=50.0, posinf=100.0, neginf=0.0)
    return F.smooth_l1_loss(ef_m, ef_gt.to(ef_m.device).float())


# ─────────────────────────────────────────────────────────────────────────────
# B+. 无标签弱监督修复（v5，移植自 EchoNet-mamba3Actor-Vision）
#     soft_curve_ef           — 与策略无关的容积曲线 EF 锚（训练 volume head）
#     selection_alignment_loss — 用容积曲线自蒸馏出逐帧 ED/ES 目标（训练 policy）
#     select_sparsity_loss     — 可微反塌缩稀疏度损失
# ─────────────────────────────────────────────────────────────────────────────

def soft_curve_ef(
    volumes: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 3.0,
):
    """与策略无关的 EF（%）——纯从逐帧容积曲线用 soft-max / soft-min 估计。

    z   = 每样本标准化容积  (v - mean) / std
    EDV = Σ softmax(+τ·z) · v      （软取容积峰 ≈ 舒张末期）
    ESV = Σ softmax(−τ·z) · v      （软取容积谷 ≈ 收缩末期）
    EF  = (EDV − ESV) / EDV · 100

    关键作用：给 volume head 一个**每步都一致**的监督目标（"曲线峰谷比 = 真实 EF"），
    完全不依赖（早期随机塌缩的）策略动作，打破 "随机角色标记→矛盾目标→平坦曲线→EF≈0" 的塌缩链。
    在本项目还有第二重收益：更可靠的容积曲线直接改善 R21D 的周期检测输入。

    Returns: (ef, edv, esv)  均为 (B,) float
    """
    v  = torch.nan_to_num(volumes.float(), nan=50.0, posinf=300.0, neginf=0.0).clamp(0.0, 300.0)
    mf = mask.float()
    n  = mf.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (v * mf).sum(dim=1, keepdim=True) / n
    var  = (((v - mean) ** 2) * mf).sum(dim=1, keepdim=True) / n
    z    = (v - mean) / (var.sqrt() + 1e-3)
    neg  = ~mask
    w_max = torch.softmax((tau * z).masked_fill(neg, -1e4), dim=1)
    w_min = torch.softmax((-tau * z).masked_fill(neg, -1e4), dim=1)
    edv = (w_max * v).sum(dim=1)
    esv = (w_min * v).sum(dim=1)
    ef  = (edv - esv) / edv.clamp(min=1.0) * 100.0
    return ef, edv, esv


def soft_curve_ef_loss(
    volumes: torch.Tensor,
    ef_gt: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 3.0,
):
    """SmoothL1(EF_soft, EF_gt)。Returns: (loss, ef_soft)。"""
    ef_soft, _, _ = soft_curve_ef(volumes, mask, tau=tau)
    loss = F.smooth_l1_loss(ef_soft, ef_gt.to(ef_soft.device).float())
    return loss, ef_soft


def _local_zscore(
    v: torch.Tensor,    # (B, T) float, already detached
    mf: torch.Tensor,   # (B, T) float mask {0,1}
    hw: int,            # half-window size (full window = 2*hw+1)
) -> torch.Tensor:
    """Mask-aware sliding-window z-score.

    每帧 t 的 z-score 只在局部窗口 [t-hw, t+hw] 内计算 mean/std，
    而非全序列。这样无论 volume 曲线是否全局周期，每个心动周期
    的局部峰谷都能独立触发 |z| > ed_thresh 的 ED/ES 目标，
    解决全局 z-score 只有前 ~20 帧能产生对齐目标的问题。

    边界处理：v 用 replicate 填充，mask 用 0 填充（边界窗口自动
    缩短有效样本数，但不引入幻象帧贡献）。
    """
    B, T = v.shape
    v_pad = F.pad(v,  (hw, hw), mode='replicate')
    m_pad = F.pad(mf, (hw, hw), mode='constant', value=0.0)

    v_win = v_pad.unfold(1, 2 * hw + 1, 1)   # (B, T, W)
    m_win = m_pad.unfold(1, 2 * hw + 1, 1)   # (B, T, W)

    cnt   = m_win.sum(-1).clamp(min=1.0)
    lmean = (v_win * m_win).sum(-1) / cnt
    lvar  = ((v_win - lmean.unsqueeze(-1)) ** 2 * m_win).sum(-1) / cnt
    lstd  = lvar.sqrt()
    return (v - lmean) / (lstd + 1e-3)


def selection_alignment_loss(
    logits: torch.Tensor,       # (B, T, 3)  策略 logits（含梯度）
    volumes: torch.Tensor,      # (B, T)     容积曲线（本函数内部 detach）
    mask: torch.Tensor,         # (B, T) bool
    ed_thresh: float = 1.0,
    sharpness: float = 2.0,
    key_balance: bool = True,   # v5.1: ED/ES 关键帧与 skip 帧平衡归一化，防关键帧梯度被稀释
    window_size: int = 0,       # v5.2: >0 启用滑动窗口局部 z-score；0 = 全局（旧行为）
    es_weight_factor: float = 1.0,  # V1.5: ES路径梯度倍数(>1时ES获更强梯度，修正生理z-score不对称)
) -> torch.Tensor:
    """容积曲线 → 逐帧三动作软目标的自蒸馏对齐损失。

    目标构造（z = 标准化容积，detach，不回传到 volume head）：
        高容积帧 (z >  ed_thresh)  → ED (类 1)
        低容积帧 (z < −ed_thresh)  → ES (类 2)
        其余                       → skip (类 0)
    用 sigmoid 软化得到概率目标 q∈(B,T,3)，再对 policy 做 masked 交叉熵。
    给策略稠密、逐帧、一致的梯度，告诉它哪帧是 ED/ES、其余 skip——直接治策略塌缩。
    标签来源是模型自己的容积曲线（自蒸馏），不使用任何关键帧标签。

    window_size > 0 时使用滑动窗口局部 z-score（推荐），解决全局 z-score
    在非周期曲线下只有前 20 帧能触发 ED/ES 目标的问题。
    """
    v  = volumes.detach().float()
    mf = mask.float()

    if window_size > 0:
        # v5.2 滑动窗口局部 z-score：每个心动周期独立归一化
        hw = window_size // 2
        z  = _local_zscore(v, mf, hw)
    else:
        # 旧行为：全序列全局 z-score
        n    = mf.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (v * mf).sum(dim=1, keepdim=True) / n
        var  = (((v - mean) ** 2) * mf).sum(dim=1, keepdim=True) / n
        z    = (v - mean) / (var.sqrt() + 1e-3)

    p_ed = torch.sigmoid((z - ed_thresh) * sharpness) * mf
    p_es = torch.sigmoid((-z - ed_thresh) * sharpness) * mf
    s    = (p_ed + p_es).clamp(min=1.0)        # 防 ED+ES 概率和 > 1
    p_ed = p_ed / s
    p_es = p_es / s
    p_skip = (1.0 - p_ed - p_es).clamp(min=0.0)
    q    = torch.stack([p_skip, p_ed, p_es], dim=-1)         # (B, T, 3)

    logp = F.log_softmax(logits, dim=-1)
    ce   = -(q * logp).sum(dim=-1)                           # (B, T)

    if key_balance:
        # V1.5: 三路独立归一化 + ES 梯度加权
        #
        # 动机（三重不对称叠加导致 ES 永不被选）：
        #   ① 帧数不对称：舒张期(ED附近)占心动周期 ~2/3，收缩期(ES附近)~1/3
        #      → v5.1 旧方案「ED+ES 合并 loss_key」中 ED 帧数多 → ED 主导 → ES 拿不到梯度
        #   ② z-score 强度不对称（局部 z-score 下）：局部窗口含更多高容积舒张帧
        #      → ES 谷的局部均值偏高 → |z_ES| 偏小 (z≈−1.2 恰好踩阈) → p_es≈0.5（软目标弱）
        #      vs ED 峰的 z≈+2.0 → p_ed≈0.92（软目标强）
        #   ③ 两者叠加：ES 在 loss_key 里实际权重仅 ~1/6，argmax 无法翻过 skip_bias
        #
        # 修复（V1.5）：
        #   — ED/ES/skip 三路各自独立归一化（消除帧数不对称）
        #   — ES 路径乘以 es_weight_factor 倍（补偿 z-score 强度不对称）
        #   — 总权重归一化 / (2 + esw) 保持量级稳定
        ed_w   = p_ed * mf                                   # (B,T) ED 目标权重
        es_w   = p_es * mf                                   # (B,T) ES 目标权重
        skip_w = (1.0 - p_ed - p_es).clamp(min=0.0) * mf   # (B,T) skip 目标权重
        loss_ed   = (ce * ed_w).sum()   / ed_w.sum().clamp(min=1.0)
        loss_es   = (ce * es_w).sum()   / es_w.sum().clamp(min=1.0)
        loss_skip = (ce * skip_w).sum() / skip_w.sum().clamp(min=1.0)
        esw = float(es_weight_factor)
        return (loss_ed + esw * loss_es + loss_skip) / (2.0 + esw)
    return (ce * mf).sum() / mf.sum().clamp(min=1.0)


def select_sparsity_loss(
    logits: torch.Tensor,       # (B, T, 3)  策略 logits（含梯度）
    mask: torch.Tensor,         # (B, T) bool
    target_ratio: float = 0.20,
) -> torch.Tensor:
    """可微反塌缩稀疏度损失。

    P(select)_t = softmax(logits)_t[1] + softmax(logits)_t[2]
    损失 = relu(mean_t P(select) − target_ratio)²

    奖励里的 w_over / w_max_es 过选惩罚要经 GAE + 优势归一化，当 batch 内轨迹都已塌缩
    （优势方差≈0）时被抹平。本损失直接对策略概率施加可微梯度，对超过 target 的部分平方惩罚，
    与 alignment 配合：alignment 定位极值帧，sparsity 压制泛滥选取。
    """
    p     = torch.softmax(logits, dim=-1)
    p_sel = p[..., 1] + p[..., 2]                            # (B, T)
    mf    = mask.float()
    mean_p = (p_sel * mf).sum() / mf.sum().clamp(min=1.0)
    excess = torch.clamp(mean_p - float(target_ratio), min=0.0)
    return excess * excess


def r21d_cyclic_ef_loss(
    ef_r21d: torch.Tensor,
    ef_gt: torch.Tensor,
) -> torch.Tensor:
    """SmoothL1(EF_r21d_avg, EF_gt)：R21D 多周期平均 EF 主监督损失。"""
    ef_r = torch.nan_to_num(ef_r21d.float(), nan=50.0, posinf=100.0, neginf=0.0)
    return F.smooth_l1_loss(ef_r, ef_gt.to(ef_r.device).float())


def r21d_per_cycle_ef_loss(
    ef_per_cycle_flat: torch.Tensor,
    cycle_counts,
    ef_gt: torch.Tensor,
) -> torch.Tensor:
    """每周期 EF 一致性损失：鼓励每周期独立预测值也接近 GT EF。

    Parameters
    ----------
    ef_per_cycle_flat : (N_total,)  所有样本所有周期的 EF 预测（flat 拼接）
    cycle_counts      : list[int] 或 Tensor  每个样本的周期数
    ef_gt             : (B,) float  Ground-truth EF
    """
    device = ef_per_cycle_flat.device
    ef_gt  = ef_gt.to(device).float()

    targets = []
    for b, cnt in enumerate(cycle_counts):
        cnt = int(cnt)
        targets.extend([ef_gt[b]] * cnt)
    if not targets:
        return torch.zeros(1, device=device).squeeze()

    targets_t = torch.stack(targets, dim=0)   # (N_total,)
    ef_flat   = torch.nan_to_num(
        ef_per_cycle_flat.float(), nan=50.0, posinf=100.0, neginf=0.0)
    return F.smooth_l1_loss(ef_flat, targets_t)


# ─────────────────────────────────────────────────────────────────────────────
# C. CSPO 损失（反事实步级策略优化）
# ─────────────────────────────────────────────────────────────────────────────

def cspo_loss(
    logits: torch.Tensor,        # (B, T, 3)   当前策略对轨迹 A/B 帧的 logits
    actions_a: torch.Tensor,     # (B, T) long  轨迹 A 动作
    actions_b: torch.Tensor,     # (B, T) long  轨迹 B 动作（FORCEDIVERGE 产生）
    rewards_a: torch.Tensor,     # (B, T) float 轨迹 A 每步奖励
    rewards_b: torch.Tensor,     # (B, T) float 轨迹 B 每步奖励
    mask: torch.Tensor,          # (B, T) bool  有效帧
    gamma: float,
    eps_gate: float = 0.005,
) -> torch.Tensor:
    """CSPO 步级因果损失。

    对每个发散步骤 t（a_A_t ≠ a_B_t），估计：

        Δρ_t = log π(a_B_t | s_t) - log π(a_A_t | s_t)   （策略预测的动作价值差）
        ΔR̂_t = G_A_t - G_B_t                              （实际回报差，折扣到 t）

    损失 = mean_{t∈V, |ΔR̂_t|>ε} (ΔR̂_t - Δρ_t)²

    其中 V = { t | a_A_t ≠ a_B_t, mask_t=True }。

    Parameters
    ----------
    logits    : (B, T, 3)  当前策略 logits（两条轨迹共享同一网络输出）
    actions_a : (B, T) long
    actions_b : (B, T) long
    rewards_a : (B, T) float  逐步奖励（A 轨迹）
    rewards_b : (B, T) float  逐步奖励（B 轨迹）
    mask      : (B, T) bool
    gamma     : 折扣因子（用于计算折扣回报差）
    eps_gate  : 奖励差绝对值低于此值时过滤（减少噪声步的影响）

    Returns
    -------
    scalar Tensor  CSPO 损失（若无有效发散步则返回 0）
    """
    device = logits.device
    B, T   = actions_a.shape

    # ── 1. 计算折扣回报差 ΔR̂_t = G_A_t - G_B_t ─────────────────────────
    delta_r = rewards_a - rewards_b   # (B, T) per-step difference
    mf = mask.float()

    # 从后向前折扣累积（G_A_t - G_B_t = Σ_{s≥t} γ^(s-t) (r_A_s - r_B_s)）
    delta_G = torch.zeros(B, T, device=device)
    running = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        running = delta_r[:, t] + gamma * running * mf[:, t]
        delta_G[:, t] = running

    # ── 2. 计算 Δρ_t = log π(a_B_t) - log π(a_A_t) ──────────────────────
    log_probs = F.log_softmax(logits, dim=-1)   # (B, T, 3)

    # gather log-prob for action a and action b
    lp_a = log_probs.gather(2, actions_a.unsqueeze(-1)).squeeze(-1)   # (B, T)
    lp_b = log_probs.gather(2, actions_b.unsqueeze(-1)).squeeze(-1)   # (B, T)
    delta_rho = lp_b - lp_a   # (B, T)

    # ── 3. 发散步掩码 & eps_gate 过滤 ────────────────────────────────────
    diverge_mask = (actions_a != actions_b) & mask             # (B, T) bool
    gate_mask    = diverge_mask & (delta_G.abs() > eps_gate)   # (B, T) bool

    n_valid = gate_mask.float().sum()
    if n_valid < 1.0:
        return torch.zeros(1, device=device).squeeze()

    # ── 4. CSPO 损失（MSE between ΔR̂_t and Δρ_t）──────────────────────
    loss = ((delta_G - delta_rho) ** 2 * gate_mask.float()).sum() / n_valid
    return loss


def role_ite_stats(
    actions_a: torch.Tensor,     # (B, T) long
    actions_b: torch.Tensor,     # (B, T) long
    rewards_a: torch.Tensor,     # (B, T) float
    rewards_b: torch.Tensor,     # (B, T) float
    mask: torch.Tensor,          # (B, T) bool
    gamma: float,
) -> dict:
    """统计按角色分类的 ITE（用于 evaluate.py 报告）。

    返回 dict：
      'ED_to_skip'  : 轨迹 A 选 ED(1)、轨迹 B 改为 skip(0) 时的平均 ΔG
      'ES_to_skip'  : 轨迹 A 选 ES(2)、轨迹 B 改为 skip(0) 时的平均 ΔG
      'ED_to_ES'    : 轨迹 A 选 ED(1)、轨迹 B 改为 ES(2) 时的平均 ΔG
      'ES_to_ED'    : 轨迹 A 选 ES(2)、轨迹 B 改为 ED(1) 时的平均 ΔG
      'n_diverge'   : 总发散步数
    """
    device = actions_a.device
    B, T   = actions_a.shape
    mf     = mask.float()

    # 折扣回报差（同 cspo_loss）
    delta_r = rewards_a - rewards_b
    delta_G = torch.zeros(B, T, device=device)
    running = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        running = delta_r[:, t] + gamma * running * mf[:, t]
        delta_G[:, t] = running

    results = {}
    pairs = [
        ('ED_to_skip', 1, 0),
        ('ES_to_skip', 2, 0),
        ('ED_to_ES',   1, 2),
        ('ES_to_ED',   2, 1),
    ]
    n_total = 0
    for name, a_val, b_val in pairs:
        sel = ((actions_a == a_val) & (actions_b == b_val) & mask)
        n   = int(sel.float().sum().item())
        n_total += n
        if n > 0:
            results[name] = float((delta_G * sel.float()).sum().item() / n)
        else:
            results[name] = float('nan')
    results['n_diverge'] = n_total
    return results
