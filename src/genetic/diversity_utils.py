# src/genetic/diversity_utils.py
"""
多样性维护工具：
- 种群 pairwise Spearman 相关度计算
- 冗余个体识别与去重
- IC 方向一致性检验（训练集两段对比）
- 新颖性得分计算
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


def compute_pairwise_spearman(df: pd.DataFrame, factor_names: List[str]) -> float:
    """计算种群内所有因子对的平均 Spearman |相关系数|（衡量同质化程度）"""
    valid = [f for f in factor_names if f in df.columns]
    if len(valid) < 2:
        return 0.0

    factor_df = df[valid].replace([np.inf, -np.inf], np.nan).fillna(0)
    corr_matrix = factor_df.corr(method='spearman').abs()

    # 取上三角（不含对角线）的均值
    n = len(valid)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                total += val
                count += 1

    return float(total / count) if count > 0 else 0.0


def find_redundant(
        df: pd.DataFrame,
        factor_names: List[str],
        fitness_map: dict,
        threshold: float = 0.9
) -> List[str]:
    """
    找出相关系数 > threshold 的冗余个体，返回应被淘汰的列名列表。
    两两比较时保留 Fitness 更高的，淘汰 Fitness 更低的。
    """
    valid = [f for f in factor_names if f in df.columns]
    if len(valid) < 2:
        return []

    factor_df = df[valid].replace([np.inf, -np.inf], np.nan).fillna(0)
    corr_matrix = factor_df.corr(method='spearman').abs()

    to_remove = set()
    for i in range(len(valid)):
        if valid[i] in to_remove:
            continue
        for j in range(i + 1, len(valid)):
            if valid[j] in to_remove:
                continue
            if corr_matrix.iloc[i, j] > threshold:
                # 淘汰 Fitness 较低的
                fi = fitness_map.get(valid[i], 0.0)
                fj = fitness_map.get(valid[j], 0.0)
                to_remove.add(valid[j] if fi >= fj else valid[i])

    return list(to_remove)


def compute_ic_sign_consistency(
        df: pd.DataFrame,
        factor_name: str,
        target_col: str = 'target_ret',
        split_ratio: float = 0.5
) -> float:
    """
    将数据集按时间二分，分别计算前半段和后半段的 Spearman IC，
    判断 IC 符号是否一致。
    一致返回 1.0，不一致返回 0.3。
    """
    if factor_name not in df.columns or target_col not in df.columns:
        return 0.3

    dates = df.index.get_level_values('datetime').unique().sort_values()
    split_idx = int(len(dates) * split_ratio)
    if split_idx < 10 or split_idx >= len(dates) - 10:
        return 0.5

    split_date = dates[split_idx]
    df_first  = df[df.index.get_level_values('datetime') <  split_date]
    df_second = df[df.index.get_level_values('datetime') >= split_date]

    def _ic(data: pd.DataFrame) -> float:
        if len(data) < 50:
            return 0.0
        try:
            f = data[factor_name].rank(pct=True)
            t = data[target_col].rank(pct=True)
            c = f.corr(t, method='spearman')
            return float(c) if not np.isnan(c) else 0.0
        except Exception:
            return 0.0

    ic1 = _ic(df_first)
    ic2 = _ic(df_second)

    return 1.0 if ic1 * ic2 > 0 else 0.3


def compute_novelty(
        df: pd.DataFrame,
        factor_name: str,
        all_factor_names: List[str],
        delta: float = 0.02
) -> float:
    """
    计算新颖性奖励：delta × (1 - 与种群均值 Spearman 相关)
    上限 0.05。
    """
    others = [f for f in all_factor_names if f != factor_name and f in df.columns]
    if not others or factor_name not in df.columns:
        return delta

    fv = df[factor_name].replace([np.inf, -np.inf], np.nan).fillna(0)
    corrs = []
    for other in others:
        ov = df[other].replace([np.inf, -np.inf], np.nan).fillna(0)
        try:
            c = fv.corr(ov, method='spearman')
            if not np.isnan(c):
                corrs.append(abs(float(c)))
        except Exception:
            pass

    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    return min(delta * (1 - mean_corr), 0.05)