# src/processing/factor_pipeline.py
"""
因子精炼管道：整合模块一（流动性过滤）、模块三（选择性中性化）、模块六（组合筛选）
模块二的 Winsorize+Rank 在 run() 中调用，模块四在 main.py 中由 build_market_state_alphas 提供
修复：未来信息、选择性稳健化、正确执行顺序
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats

from src.evaluation.metrics_calc import (
    clean_factor_data, robustify_factor,
    calculate_factor_autocorr, calculate_turnover,
    calculate_ic_stability, calculate_local_fitness
   
)


class FactorPipeline:
    def __init__(
            self,
            df: pd.DataFrame,
            factor_cols: List[str],
            volume_col: Optional[str] = None,
            industry_col: Optional[str] = None,
            market_cap_col: Optional[str] = None,
    ):
        self.df = df.copy()
        self.factor_cols = [c for c in factor_cols if c != 'target_ret']
        self.volume_col = volume_col
        self.industry_col = industry_col
        self.market_cap_col = market_cap_col
        self.diagnostics: Dict = {}

    # ============================================================
    # 修复 2：未来信息 → 滚动历史分位数 Winsorize
    # ============================================================
    @staticmethod
    def winsorize_rolling_hist(df: pd.DataFrame, factor_cols: List[str],
                               window: int = 252, lower: float = 0.01, upper: float = 0.99):
        """
        用个股滚动 252 日历史分位数钳位，彻底杜绝未来信息。
        """
        df = df.copy()
        for col in factor_cols:
            if col not in df.columns:
                continue
            # 转为宽表 (日期 × 股票)
            wide = df[col].unstack(level='instrument')
            # 滚动分位数（用 expanding 前 window 天后改为 rolling）
            # 简化：expanding 前252天，后续滚动252天
            lower_hist = wide.expanding(window).quantile(lower)
            upper_hist = wide.expanding(window).quantile(upper)
            # 由于 expanding 会包含当前点，若严格需要仅用过去值，需 shift，但为简洁，用 expanding 后回填
            lower_hist = lower_hist.shift(1).fillna(method='ffill')
            upper_hist = upper_hist.shift(1).fillna(method='ffill')
            # 对齐后 clip
            aligned_lower = lower_hist.reindex(wide.index, columns=wide.columns, fill_value=0)
            aligned_upper = upper_hist.reindex(wide.index, columns=wide.columns,
                                               fill_value=wide.max().max())
            clipped = wide.clip(lower=aligned_lower, upper=aligned_upper, axis=0)
            # 转回长表
            df[col] = clipped.stack().reorder_levels(df.index.names)
        return df

    # ============== 模块一：流动性过滤（不变） ==============
    def liquidity_filter(self, bottom_pct: float = 0.2, min_price: float = 1.0):
        if self.volume_col is None or self.volume_col not in self.df.columns:
            print(f"⚠️ [模块一] 未找到流动性列 '{self.volume_col}'，跳过流动性过滤")
            return self
        print(f"🔍 [模块一] 流动性过滤（剔除底部 {bottom_pct:.0%}）...")
        volume_rank = self.df.groupby(level='datetime')[self.volume_col].rank(pct=True)
        liquidity_mask = volume_rank > bottom_pct
        total = len(self.df)
        filtered = (~liquidity_mask).sum()
        for col in self.factor_cols:
            if col in self.df.columns:
                self.df.loc[~liquidity_mask, col] = np.nan
        self.diagnostics['liquidity_filter'] = {
            'bottom_pct': bottom_pct,
            'filtered_signals': int(filtered),
            'total_signals': int(total),
            'filter_rate': float(filtered / total)
        }
        print(f"   ✅ 剔除 {filtered}/{total} 个信号 ({filtered/total:.1%})")
        return self

    # ============== 模块三：选择性中性化（诊断-决策）==============
    def _diagnose_factor_exposure(self, factor_col: str, control_cols: List[str]) -> Dict:
        clean_df = clean_factor_data(self.df)
        daily_results = []
        for date, group in clean_df.groupby(level='datetime'):
            if len(group) < 20:
                continue
            y = group[factor_col].values
            X_cols = [c for c in control_cols if c in group.columns]
            if not X_cols:
                continue
            X = group[X_cols].values
            X = np.column_stack([np.ones(len(X)), X])
            if np.isnan(X).any() or np.isnan(y).any():
                continue
            try:
                betas = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ betas
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                row = {'r_squared': r2}
                for i, c in enumerate(X_cols):
                    row[f'beta_{c}'] = betas[i + 1]
                daily_results.append(row)
            except Exception:
                continue
        if not daily_results:
            return {'r_squared_mean': 0.0, 'significant': False}
        result_df = pd.DataFrame(daily_results)
        diag = {
            'r_squared_mean': float(result_df['r_squared'].mean()),
            'significant': float(result_df['r_squared'].mean()) > 0.1
        }
        for col in control_cols:
            beta_col = f'beta_{col}'
            if beta_col in result_df.columns:
                betas = result_df[beta_col].dropna()
                t, p = stats.ttest_1samp(betas, 0)
                diag[f't_{col}'] = float(t)
                diag[f'p_{col}'] = float(p)
                diag[f'significant_{col}'] = abs(float(t)) > 2.0
        return diag

    def _neutralize_single_factor(self, factor_col: str, control_cols: List[str]) -> pd.Series:
        residuals = []
        clean_df = clean_factor_data(self.df)
        for date, group in clean_df.groupby(level='datetime'):
            if len(group) < 20:
                residuals.append(group[factor_col])
                continue
            y = group[factor_col].values
            X_cols = [c for c in control_cols if c in group.columns]
            if not X_cols:
                residuals.append(group[factor_col])
                continue
            X = group[X_cols].values
            X = np.column_stack([np.ones(len(X)), X])
            if np.isnan(X).any() or np.isnan(y).any():
                residuals.append(group[factor_col])
                continue
            try:
                betas = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ betas
                residuals.append(pd.Series(resid, index=group.index))
            except Exception:
                residuals.append(group[factor_col])
        return pd.concat(residuals).sort_index()

    def selective_neutralize(self, sharpe_drop_threshold: float = 0.2):
        neutralize_cols = []
        col_labels = []
        if self.industry_col and self.industry_col in self.df.columns:
            ind_dummies = pd.get_dummies(self.df[self.industry_col], prefix='ind')
            self.df = pd.concat([self.df, ind_dummies], axis=1)
            neutralize_cols.extend(list(ind_dummies.columns))
            col_labels.append(('industry', list(ind_dummies.columns)))
        if self.market_cap_col and self.market_cap_col in self.df.columns:
            self.df['log_mktcap'] = np.log(self.df[self.market_cap_col].clip(lower=1))
            neutralize_cols.append('log_mktcap')
            col_labels.append(('market_cap', ['log_mktcap']))
        if not neutralize_cols:
            print("⚠️ [模块三] 未找到行业/市值列，跳过中性化")
            return self
        print(f"🔬 [模块三] 选择性中性化诊断...")
        self.diagnostics['neutralization'] = {}
        from src.evaluation.metrics_calc import calculate_rank_ic
        for factor_col in self.factor_cols:
            if factor_col not in self.df.columns:
                continue
            factor_diag = {}
            original_ic = calculate_rank_ic(
                self.df[[factor_col, 'target_ret']].replace(
                    [np.inf, -np.inf], np.nan).fillna(0)
            ).get(factor_col, 0.0)
            for label, ctrl_cols in col_labels:
                exposure = self._diagnose_factor_exposure(factor_col, ctrl_cols)
                if not exposure.get('significant', False):
                    print(f"   [{factor_col}] {label} 暴露不显著，跳过")
                    factor_diag[label] = {'applied': False, 'reason': 'not_significant'}
                    continue
                neutralized = self._neutralize_single_factor(factor_col, ctrl_cols)
                temp_df = self.df.copy()
                temp_df[f'{factor_col}_n'] = neutralized
                neutral_ic = calculate_rank_ic(
                    temp_df[[f'{factor_col}_n', 'target_ret']].replace(
                        [np.inf, -np.inf], np.nan).fillna(0)
                ).get(f'{factor_col}_n', 0.0)
                if abs(original_ic) > 1e-6:
                    ic_drop = (abs(original_ic) - abs(neutral_ic)) / abs(original_ic)
                else:
                    ic_drop = 0.0
                if ic_drop > sharpe_drop_threshold:
                    print(f"   [{factor_col}] {label} 中性化后 IC 下降 {ic_drop:.1%}，"
                          f"该暴露是 Alpha 来源，放弃！")
                    factor_diag[label] = {'applied': False, 'reason': 'alpha_source',
                                          'ic_drop': float(ic_drop)}
                else:
                    print(f"   [{factor_col}] {label} 应用中性化 "
                          f"({abs(original_ic):.4f}→{abs(neutral_ic):.4f})")
                    self.df[factor_col] = neutralized
                    factor_diag[label] = {'applied': True,
                                          'ic_before': float(original_ic),
                                          'ic_after': float(neutral_ic),
                                          'ic_drop': float(ic_drop)}
            self.diagnostics['neutralization'][factor_col] = factor_diag
        return self

    # ============== 模块六：低相关组合筛选（不变）==============
    def select_portfolio(
            self, sharpe_dict: Dict[str, float],
            max_factors: int = 10,
            min_fitness: float = 0.0,
            max_avg_corr: float = 0.3,
            corr_method: str = 'spearman'
    ) -> List[Dict]:
        print(f"\n🎯 [模块六] 因子组合筛选...")
        print(f"   候选: {len(sharpe_dict)} | 目标: {max_factors} | "
              f"最大平均相关: {max_avg_corr}")
        clean_df = clean_factor_data(self.df)
        candidates = []
        for fname, sharpe in sharpe_dict.items():
            if fname not in clean_df.columns:
                continue
            turnover = calculate_turnover(clean_df, fname)
            _, icir = calculate_ic_stability(clean_df, fname)
            fit_report = calculate_local_fitness(sharpe, turnover, 0.0, icir)
            if fit_report['fitness'] < min_fitness:
                continue
            candidates.append({'name': fname, **fit_report})
        candidates.sort(key=lambda x: x['fitness'], reverse=True)
        print(f"   通过最低 Fitness 门槛: {len(candidates)}")
        names = [c['name'] for c in candidates if c['name'] in clean_df.columns]
        if len(names) > 1:
            corr_mat = clean_df[names].corr(method=corr_method).abs()
        else:
            corr_mat = pd.DataFrame([[1.0]], index=names, columns=names)
        selected = []
        selected_names = []
        for cand in candidates:
            name = cand['name']
            if name not in corr_mat.index:
                continue
            if not selected_names:
                selected.append(cand)
                selected_names.append(name)
                print(f"   ✅ [{len(selected)}] {name} | Fit={cand['fitness']:.4f}")
                continue
            corrs = [corr_mat.loc[name, sn] for sn in selected_names
                     if sn in corr_mat.columns]
            avg_corr = float(np.mean(corrs)) if corrs else 0.0
            if avg_corr <= max_avg_corr:
                max_corr = max(corrs) if corrs else 0.0
                updated = calculate_local_fitness(
                    cand['sharpe'], cand['turnover'], max_corr, cand['icir']
                )
                cand.update(updated)
                cand['avg_corr_with_portfolio'] = avg_corr
                selected.append(cand)
                selected_names.append(name)
                print(f"   ✅ [{len(selected)}] {name} | Fit={cand['fitness']:.4f} | "
                      f"AvgCorr={avg_corr:.3f}")
            else:
                print(f"   ❌ {name} | AvgCorr={avg_corr:.3f} > {max_avg_corr}")
            if len(selected) >= max_factors:
                break
        print(f"\n📊 最终组合 {len(selected)} 个：")
        for i, info in enumerate(selected):
            print(f"  [{i+1}] {info['name']} | Fit={info['fitness']:.4f} | "
                  f"Sharpe={info['sharpe']:.3f} | Turn={info['turnover']:.4f}")
        self.diagnostics['portfolio'] = {
            'n_candidates': len(candidates),
            'n_selected': len(selected),
            'selected_names': selected_names,
            'avg_fitness': float(np.mean([s['fitness'] for s in selected]))
                          if selected else 0,
        }
        return selected

    # ============================================================
    # 修复 3+4+5：新执行顺序 + 选择性稳健化 + 因子延迟 shift
    # ============================================================
    def run(
            self,
            enable_liquidity_filter: bool = True,
            enable_robustify: bool = True,
            enable_neutralization: bool = True,
            liquidity_bottom_pct: float = 0.2,
            winsor_lower: float = 0.01,
            winsor_upper: float = 0.99,
            sharpe_drop_threshold: float = 0.2,
            require_delay_shift: bool = True  # 新增：是否对因子做滞后一期
    ) -> pd.DataFrame:
        print("\n🏭 因子精炼管道启动")
        print("=" * 60)

        # ---------- 0. 延迟因子值（杜绝未来信息） ----------
        if require_delay_shift:
            cols_to_shift = [c for c in self.factor_cols if c in self.df.columns]
            if cols_to_shift:
                print("⏳ 因子延迟 shift(1) 以确保无未来信息...")
                self.df[cols_to_shift] = self.df.groupby(
                    level='instrument')[cols_to_shift].shift(1)

        # ---------- 1. 流动性过滤 ----------
        if enable_liquidity_filter:
            self.liquidity_filter(bottom_pct=liquidity_bottom_pct)
        else:
            print("⏭️  [模块一] 流动性过滤已跳过")

        # ---------- 2. 选择性中性化（在稳健化之前，使用原始因子值） ----------
        if enable_neutralization:
            self.selective_neutralize(sharpe_drop_threshold=sharpe_drop_threshold)
        else:
            print("⏭️  [模块三] 中性化已跳过")

        # ---------- 3. 选择性稳健化 ----------
        if enable_robustify:
            print(f"📐 [模块二] 选择性稳健化...")
            cols = [c for c in self.factor_cols if c in self.df.columns]
            self.diagnostics['robustify'] = {}
            for col in cols:
                # 选择是否跳过 Rank（高自相关 & 高原始 Sharpe 的因子保留原有量纲）
                ac = calculate_factor_autocorr(self.df, col, lag=1)
                # 需要原始 Sharpe 做判断，这里快速估算一个粗略值（调用计算函数）
                from src.evaluation.metrics_calc import compute_factor_sharpe
                raw_sharpe = compute_factor_sharpe(self.df, col)['sharpe_net']
                self.diagnostics['robustify'][col] = {
                    'autocorr': ac, 'raw_sharpe': raw_sharpe,
                    'skip_rank': False
                }
                if ac > 0.7 and raw_sharpe > 1.0:
                    # 高自相关+高Sharpe，只做去极值，不Rank
                    self.df = self.winsorize_rolling_hist(self.df, [col],
                                                          window=252,
                                                          lower=winsor_lower,
                                                          upper=winsor_upper)
                    self.diagnostics['robustify'][col]['skip_rank'] = True
                    self.diagnostics['robustify'][col]['action'] = 'only_winsorize'
                    print(f"   {col}: 自相关={ac:.3f}, Sharpe={raw_sharpe:.3f}"
                          f" → 跳过 Rank，仅 Winsorize")
                else:
                    # 完整 robustify (Winsorize + Rank)
                    self.df = robustify_factor(
                        self.df, [col],
                        winsor_lower=winsor_lower,
                        winsor_upper=winsor_upper
                    )
                    self.diagnostics['robustify'][col]['action'] = 'full_robustify'
                    print(f"   {col}: 自相关={ac:.3f} → 完整 Winsorize+Rank")
        else:
            print("⏭️  [模块二] 信号鲁棒化已跳过")

        print("\n✅ 因子精炼管道完成！")
        return self.df