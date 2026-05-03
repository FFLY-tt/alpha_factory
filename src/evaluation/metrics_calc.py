# src/evaluation/metrics_calc.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.ast.ast_nodes import FactorNode
from tqdm import tqdm


# ============================================================
# 【原有逻辑·完全不动】
# ============================================================

def clean_factor_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0)


def calculate_rank_ic(df: pd.DataFrame, target_col: str = "target_ret") -> pd.Series:
    if target_col not in df.columns:
        raise ValueError(f"数据中缺失目标收益率列: {target_col}")

    ranked_df = df.groupby(level='datetime').rank(pct=True)

    def calc_daily_corr(group):
        return group.corrwith(group[target_col])

    daily_ic = ranked_df.groupby(level='datetime').apply(calc_daily_corr)
    mean_ic = daily_ic.mean().drop(target_col, errors='ignore')
    return mean_ic.fillna(0)


def select_top_k_orthogonal(
        candidates: List[FactorNode],
        df: pd.DataFrame,
        k: int = 50,
        corr_threshold: float = 0.7
) -> Tuple[List[FactorNode], Dict[str, float]]:
    print("  -> [评估法庭] 正在清洗异常数据 (Inf / NaN)...")
    clean_df = clean_factor_data(df)

    print("  -> [评估法庭] 正在计算横截面 Rank IC (可能需要几十秒)...")
    ic_series = calculate_rank_ic(clean_df)

    abs_ic = ic_series.abs()
    sorted_factor_names = abs_ic.sort_values(ascending=False).index.tolist()

    selected_nodes = []
    selected_names = []
    ic_report = {}

    node_map = {n.name: n for n in candidates}

    print(f"  -> [评估法庭] 开启降维正交化 (目标: Top {k}, 候选: {len(sorted_factor_names)}个)...")

    for factor_name in tqdm(sorted_factor_names, desc="⚔️ 精英正交筛选中", unit="因子"):
        if abs_ic[factor_name] < 0.001:
            continue

        is_orthogonal = True
        candidate_series = clean_df[factor_name]

        for selected_name in selected_names:
            selected_series = clean_df[selected_name]
            corr_val = candidate_series.corr(selected_series)

            if abs(corr_val) > corr_threshold:
                is_orthogonal = False
                break

        if is_orthogonal:
            selected_names.append(factor_name)
            selected_nodes.append(node_map[factor_name])
            ic_report[factor_name] = ic_series[factor_name]

        if len(selected_nodes) >= k:
            print("\n  ✅ 已成功锁定所有席位，提前终止筛选！")
            break

    return selected_nodes, ic_report


# ============================================================
# ↓↓↓ 7 模块工具函数
# ============================================================

def winsorize_cross_section(
        df: pd.DataFrame, factor_cols: List[str],
        lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in factor_cols if c != 'target_ret' and c in df.columns]
    for col in cols:
        df[col] = df.groupby(level='datetime')[col].transform(
            lambda x: x.clip(lower=x.quantile(lower), upper=x.quantile(upper))
        )
    return df


def rank_cross_section(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in factor_cols if c != 'target_ret' and c in df.columns]
    for col in cols:
        df[col] = df.groupby(level='datetime')[col].rank(pct=True)
    return df


def robustify_factor(
        df: pd.DataFrame, factor_cols: List[str],
        winsor_lower: float = 0.01, winsor_upper: float = 0.99
) -> pd.DataFrame:
    df = winsorize_cross_section(df, factor_cols, winsor_lower, winsor_upper)
    df = rank_cross_section(df, factor_cols)
    return df


def calculate_factor_autocorr(
        df: pd.DataFrame, factor_name: str, lag: int = 1
) -> float:
    def stock_autocorr(s: pd.Series) -> float:
        if len(s.dropna()) < lag + 5:
            return np.nan
        return s.autocorr(lag=lag)

    autocorrs = df[factor_name].unstack(level='instrument').apply(stock_autocorr)
    return float(autocorrs.mean())


def calculate_turnover(
        df: pd.DataFrame, factor_name: str, top_pct: float = 0.2
) -> float:
    series = df[factor_name].replace([np.inf, -np.inf], np.nan).fillna(0)
    ranks = series.groupby(level='datetime').rank(pct=True)

    raw = ranks.copy()
    raw[ranks >= (1 - top_pct)] = 1.0
    raw[ranks <= top_pct] = -1.0
    raw[(ranks > top_pct) & (ranks < (1 - top_pct))] = 0.0

    wide = raw.unstack(level='instrument').fillna(0)
    norm = wide.div(
        wide.clip(lower=0).sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0) + wide.div(
        wide.clip(upper=0).abs().sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0)

    return float(norm.diff().abs().sum(axis=1).fillna(0).mean())


def calculate_ic_stability(
        df: pd.DataFrame, factor_name: str, target_col: str = 'target_ret'
) -> Tuple[float, float]:
    if target_col not in df.columns or factor_name not in df.columns:
        return 0.0, 0.0

    def daily_rank_ic(group):
        if len(group) < 5:
            return np.nan
        f = group[factor_name].rank(pct=True)
        t = group[target_col].rank(pct=True)
        c = f.corr(t)
        return c if pd.notna(c) else 0.0

    daily_ic = df.groupby(level='datetime').apply(daily_rank_ic).dropna()
    if len(daily_ic) == 0 or daily_ic.std() == 0:
        return 0.0, 0.0

    mean_ic = float(daily_ic.mean())
    icir = float(mean_ic / daily_ic.std())
    return mean_ic, icir


# ============================================================
# 温和 Fitness 公式（已修复版，保持不变）
# ============================================================

def calculate_local_fitness(
        sharpe: float, turnover: float,
        max_corr: float = 0.0, icir: float = 1.0,
        turnover_threshold: float = 0.25, corr_threshold: float = 0.5
) -> Dict[str, float]:
    """温和分段惩罚 + 稳定性保底"""
    if turnover > 1.2:
        turnover_penalty = 0.7
    elif turnover > 0.8:
        turnover_penalty = (turnover - 0.8) * 1.0 + 0.3
    elif turnover > 0.5:
        turnover_penalty = (turnover - 0.5) * 0.6
    else:
        turnover_penalty = 0.0

    if max_corr > corr_threshold:
        corr_penalty = min((max_corr - corr_threshold) / (1.0 - corr_threshold), 1.0)
    else:
        corr_penalty = 0.0

    stability_factor = max(0.4, 1.0 + icir * 0.3)

    fitness = sharpe * (1 - turnover_penalty) * (1 - corr_penalty) * stability_factor

    return {
        "fitness": round(fitness, 4),
        "sharpe": round(sharpe, 4),
        "turnover": round(turnover, 4),
        "turnover_penalty": round(turnover_penalty, 4),
        "max_corr": round(max_corr, 4),
        "corr_penalty": round(corr_penalty, 4),
        "icir": round(icir, 4),
        "stability_factor": round(stability_factor, 4),
    }


def evaluate_factor_fitness(
        df: pd.DataFrame, factor_name: str, sharpe: float,
        existing_factor_cols: Optional[List[str]] = None
) -> Dict[str, float]:
    turnover = calculate_turnover(df, factor_name)
    _, icir = calculate_ic_stability(df, factor_name)

    max_corr = 0.0
    if existing_factor_cols:
        clean = clean_factor_data(df)
        candidate = clean[factor_name]
        corrs = []
        for col in existing_factor_cols:
            if col in clean.columns:
                c = candidate.corr(clean[col], method='spearman')
                corrs.append(abs(c) if pd.notna(c) else 0.0)
        max_corr = max(corrs) if corrs else 0.0

    return calculate_local_fitness(sharpe, turnover, max_corr, icir)


# ============================================================
# 【改动 1】compute_factor_sharpe：高换手时跳过成本扣除
# ============================================================

# 高换手阈值：超过此值时成本模型失真，不再扣除成本，仅用毛 Sharpe
HIGH_TURN_IGNORE_COST = 2.5


def compute_factor_sharpe(
        df: pd.DataFrame, factor_name: str,
        quantiles: int = 5, cost_bps: float = 5.0
) -> Dict[str, float]:
    """
    独立 Sharpe 计算
    
    新增逻辑：当换手率 > HIGH_TURN_IGNORE_COST 时，
    成本扣除模型失真（线性外推不再可靠），
    此时 sharpe_net = sharpe_gross 并标记 high_turnover=True
    """
    if factor_name not in df.columns or 'target_ret' not in df.columns:
        return {"sharpe_gross": 0.0, "sharpe_net": 0.0,
                "ann_ret": 0.0, "ann_vol": 0.0, "max_dd": 0.0,
                "turnover": 0.0, "high_turnover": False}

    data = df[[factor_name, 'target_ret']].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(data) < 100:
        return {"sharpe_gross": 0.0, "sharpe_net": 0.0,
                "ann_ret": 0.0, "ann_vol": 0.0, "max_dd": 0.0,
                "turnover": 0.0, "high_turnover": False}

    data = data.copy()

    # ── 步骤 3（DeepSeek）：提前淘汰无区分度因子 ─────────────────────────
    # 因子值横截面标准差过小说明因子几乎恒定，分位选股无意义
    factor_cross_std = data.groupby(level='datetime')[factor_name].std().mean()
    if factor_cross_std < 1e-6:
        return {"sharpe_gross": 0.0, "sharpe_net": 0.0,
                "ann_ret": 0.0, "ann_vol": 0.0, "max_dd": 0.0,
                "turnover": 0.0, "high_turnover": False}
    # ─────────────────────────────────────────────────────────────────────

    data['quantile'] = data.groupby(level='datetime')[factor_name].transform(
        lambda x: pd.qcut(x, q=quantiles, labels=False, duplicates='drop')
    )

    daily = data.groupby(['datetime', 'quantile'])['target_ret'].mean().unstack()
    long_ret = daily.get(0, pd.Series(0, index=daily.index))
    short_ret = daily.get(quantiles - 1, pd.Series(0, index=daily.index))
    ls_ret = long_ret - short_ret

    turnover = calculate_turnover(df, factor_name)

    # 高换手时跳过成本扣除（避免成本模型失真导致负 Sharpe）
    if turnover > HIGH_TURN_IGNORE_COST:
        ls_ret_net = ls_ret
        cost_model_reliable = False
    else:
        daily_cost = turnover * cost_bps / 10000
        ls_ret_net = ls_ret - daily_cost
        cost_model_reliable = True

    ann_ret_g = ls_ret.mean() * 252
    ann_ret_n = ls_ret_net.mean() * 252

    # ── 步骤 1（DeepSeek）：年化波动率下限 2%，防止除零/极小值导致 Sharpe 爆炸
    MIN_ANN_VOL = 0.02
    ann_vol = max(ls_ret.std() * np.sqrt(252), MIN_ANN_VOL)

    sharpe_g = float(ann_ret_g / ann_vol)
    sharpe_n = float(ann_ret_n / ann_vol)

    # ── 步骤 2（DeepSeek）：Sharpe 截断 [-5, +5]，双重保险
    sharpe_g = float(np.clip(sharpe_g, -5.0, 5.0))
    sharpe_n = float(np.clip(sharpe_n, -5.0, 5.0))

    cum = (1 + ls_ret.fillna(0)).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    return {
        "sharpe_gross": sharpe_g, "sharpe_net": sharpe_n,
        "ann_ret": float(ann_ret_g), "ann_vol": float(ann_vol),
        "max_dd": max_dd, "turnover": float(turnover),
        "high_turnover": not cost_model_reliable,
    }


# ============================================================
# 【改动 2】方向修正工具函数
# ============================================================

def auto_correct_direction(
        ic: float, sharpe_net: float, sharpe_gross: float
) -> Dict:
    """
    根据 IC 方向自动判断因子的有效方向
    
    背景：compute_factor_sharpe 内部固定为「做多 Q0 / 做空 Q4」，
    这相当于隐含假设「因子值越小，未来收益越高」。
    
    - 当 IC < 0：方向吻合，不需要修正
    - 当 IC > 0：因子值大对应高收益，但代码做多了低值组 → 方向反了，
                需要把 Sharpe 取反才反映真实选股能力
    
    Args:
        ic: 横截面 Rank IC
        sharpe_net: 原始 sharpe_net（未修正方向）
        sharpe_gross: 原始 sharpe_gross（未修正方向）
    Returns:
        {
            "corrected_sharpe_net":   修正后的净 Sharpe,
            "corrected_sharpe_gross": 修正后的毛 Sharpe,
            "needs_negation":         是否需要在 WQ 提交时取反整个因子,
            "direction":              "原始方向" 或 "需取反"
        }
    """
    if ic > 0:
        # 因子值大 → 收益高。代码做多 Q0 不对，需取反
        return {
            "corrected_sharpe_net":   -sharpe_net,
            "corrected_sharpe_gross": -sharpe_gross,
            "needs_negation":         True,
            "direction":              "需取反（WQ 提交时给整个表达式加 -1 *）"
        }
    else:
        # 因子值小 → 收益高。代码做多 Q0 正好，不用动
        return {
            "corrected_sharpe_net":   sharpe_net,
            "corrected_sharpe_gross": sharpe_gross,
            "needs_negation":         False,
            "direction":              "原始方向（直接提交即可）"
        }