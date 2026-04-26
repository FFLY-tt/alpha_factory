# src/evaluation/backtester.py
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 【原有逻辑·完全不动】
# ==============================================================================

def quantile_backtest(df: pd.DataFrame, factor_name: str, quantiles: int = 5):
    """
    极速向量化分层回测 (Long-Short 多空组合)
    """
    if factor_name not in df.columns or "target_ret" not in df.columns:
        raise ValueError("数据源中缺失因子列或目标收益率(target_ret)列！")

    data = df[[factor_name, "target_ret"]].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        print("⚠️ 警告：回测数据为空！")
        return

    data['quantile'] = data.groupby(level='datetime')[factor_name].transform(
        lambda x: pd.qcut(x, q=quantiles, labels=False, duplicates='drop')
    )

    daily_returns = data.groupby(['datetime', 'quantile'])['target_ret'].mean().unstack()

    long_ret = daily_returns[0]
    short_ret = daily_returns[quantiles - 1]

    ls_ret = long_ret - short_ret

    cum_ret = (1 + ls_ret.fillna(0)).cumprod()

    annualized_return = ls_ret.mean() * 252
    annualized_vol = ls_ret.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

    rolling_max = cum_ret.cummax()
    drawdown = cum_ret / rolling_max - 1
    max_drawdown = drawdown.min()

    print("\n" + "=" * 40)
    print(f"📈 因子 [{factor_name}] 样本外回测战报")
    print("=" * 40)
    print(f"年化收益率 (Ann. Ret): {annualized_return:.2%}")
    print(f"年化波动率 (Ann. Vol): {annualized_vol:.2%}")
    print(f"夏普比率   (Sharpe):   {sharpe_ratio:.2f}")
    print(f"最大回撤   (Max DD):   {max_drawdown:.2%}")
    print("=" * 40)

    plt.figure(figsize=(10, 5))
    plt.plot(cum_ret.index, cum_ret.values, label='Long-Short Portfolio', color='red')
    plt.title(f"Cumulative Return of {factor_name}")
    plt.xlabel('Date')
    plt.ylabel('Net Value')
    plt.grid(True)
    plt.legend()
    plt.show()


# ==============================================================================
# ↓↓↓ 模块五：Walk-Forward 验证（新增，不动原 quantile_backtest）↓↓↓
# ==============================================================================

def walk_forward_backtest(
        fetch_func, factor_name: str, factor_expr: str,
        instrument_pool: str = "sp500", folds=None,
        quantiles: int = 5, cost_bps: float = 5.0
):
    """
    模块五：Walk-Forward 滚动验证
    使用 metrics_calc.compute_factor_sharpe（独立逻辑），不调用原 quantile_backtest
    """
    from src.evaluation.metrics_calc import compute_factor_sharpe, calculate_ic_stability

    if folds is None:
        folds = [
            ("2015-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
            ("2016-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
            ("2017-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
            ("2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
            ("2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ]

    fold_results = []
    formulas = {factor_name: factor_expr}

    print(f"\n🔄 Walk-Forward 验证启动：共 {len(folds)} 折")
    print("=" * 60)

    for i, (is_start, is_end, oos_start, oos_end) in enumerate(folds):
        print(f"\n📁 Fold {i+1}/{len(folds)}")
        print(f"   训练期: {is_start} ~ {is_end}")
        print(f"   验证期: {oos_start} ~ {oos_end}")

        oos_df = fetch_func(formulas, instrument_pool, oos_start, oos_end)
        metrics = compute_factor_sharpe(oos_df, factor_name, quantiles, cost_bps)
        mean_ic, icir = calculate_ic_stability(oos_df, factor_name)

        fold_results.append({
            "fold": i + 1, "oos_start": oos_start, "oos_end": oos_end,
            "sharpe_gross": metrics["sharpe_gross"],
            "sharpe_net": metrics["sharpe_net"],
            "ann_ret": metrics["ann_ret"], "ann_vol": metrics["ann_vol"],
            "max_dd": metrics["max_dd"], "turnover": metrics["turnover"],
            "mean_ic": mean_ic, "icir": icir,
        })

        print(f"   OOS Sharpe(净): {metrics['sharpe_net']:.3f} | "
              f"Turnover: {metrics['turnover']:.4f} | ICIR: {icir:.4f}")

    sharpes = [r['sharpe_net'] for r in fold_results]
    turnovers = [r['turnover'] for r in fold_results]
    icirs = [r['icir'] for r in fold_results]

    summary = {
        "factor_name": factor_name, "n_folds": len(folds),
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "min_sharpe": float(np.min(sharpes)),
        "max_sharpe": float(np.max(sharpes)),
        "mean_turnover": float(np.mean(turnovers)),
        "mean_icir": float(np.mean(icirs)),
        "sharpe_stability": float(np.mean(sharpes) / np.std(sharpes))
                            if np.std(sharpes) > 0 else 0,
        "fold_results": fold_results
    }

    print("\n" + "=" * 60)
    print(f"📊 Walk-Forward 汇总报告")
    print("=" * 60)
    print(f"平均 OOS Sharpe:    {summary['mean_sharpe']:.3f} ± {summary['std_sharpe']:.3f}")
    print(f"最差折 Sharpe:      {summary['min_sharpe']:.3f}")
    print(f"最佳折 Sharpe:      {summary['max_sharpe']:.3f}")
    print(f"平均换手率:         {summary['mean_turnover']:.4f}")
    print(f"平均 ICIR:          {summary['mean_icir']:.4f}")
    print(f"Sharpe 稳定性:      {summary['sharpe_stability']:.3f}")

    if summary['min_sharpe'] > 0 and summary['sharpe_stability'] > 1.0:
        print("✅ 结论：因子样本外表现稳健，推荐纳入候选库！")
    elif summary['mean_sharpe'] > 0:
        print("⚠️ 结论：因子有效但存在波动，建议结合市场状态使用。")
    else:
        print("❌ 结论：因子样本外表现不稳定，建议丢弃。")

    return summary