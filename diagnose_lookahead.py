# diagnose_lookahead.py
"""
诊断脚本：验证代码是否使用了未来数据
原理：
- 生成 5 个完全随机的"假因子"（与未来收益无任何关系）
- 如果代码无未来数据问题，这些随机因子的 Sharpe 应该集中在 0 附近（±0.3 内）
- 如果 Sharpe 离谱（>1），说明有未来数据泄漏
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from data_pipeline.data_source import init_qlib_engine, fetch_factor_data
from src.evaluation.metrics_calc import compute_factor_sharpe, calculate_ic_stability


POOL = "sp500"
START_DATE = "2020-01-01"
END_DATE = "2021-12-31"


def main():
    print("=" * 60)
    print("🔬 未来数据泄漏诊断")
    print("=" * 60)

    # 1. 初始化 Qlib，先取一个真实因子（仅为了得到正确的索引结构）
    cur = os.path.dirname(os.path.abspath(__file__))
    init_qlib_engine(os.path.join(cur, "data", "qlib_data", "us_data"))

    # 取一个真实因子获取数据框架
    df = fetch_factor_data(
        {"real_close": "$close"},
        POOL, START_DATE, END_DATE
    )

    print(f"\n数据形状: {df.shape}")
    print(f"日期范围: {df.index.get_level_values('datetime').min()} → "
          f"{df.index.get_level_values('datetime').max()}")

    # 2. 生成 5 个完全随机的假因子（与未来收益无关）
    print("\n生成 5 个随机假因子...")
    np.random.seed(42)
    for i in range(5):
        df[f"random_factor_{i+1}"] = np.random.randn(len(df))

    # 3. 评估每个随机因子
    print("\n" + "=" * 60)
    print("🎲 随机因子 Sharpe 测试结果")
    print("=" * 60)
    print(f"{'因子':<20} {'Sharpe(净)':>12} {'IC':>10} {'判定':>20}")
    print("-" * 60)

    suspicious_count = 0
    for i in range(5):
        fname = f"random_factor_{i+1}"
        m = compute_factor_sharpe(df, fname, quantiles=5, cost_bps=5.0)
        ic, _ = calculate_ic_stability(df, fname)

        # 判定标准
        if abs(m['sharpe_net']) > 1.0:
            verdict = "❌ 高度可疑"
            suspicious_count += 1
        elif abs(m['sharpe_net']) > 0.5:
            verdict = "⚠️ 略高"
        else:
            verdict = "✅ 正常"

        print(f"{fname:<20} {m['sharpe_net']:>+12.3f} {ic:>+10.4f} {verdict:>20}")

    print("-" * 60)
    print("\n" + "=" * 60)
    print("📋 诊断结论")
    print("=" * 60)

    if suspicious_count == 0:
        print("✅ 代码无未来数据问题！")
        print("   随机因子 Sharpe 集中在 0 附近，符合理论预期")
        print("\n💡 你看到真实因子 Sharpe 较高，原因是：")
        print("   1. 5 分位多空策略本身容易放大 Sharpe（极端切片）")
        print("   2. 2020-2021 是极端市场（新冠暴跌+暴涨）")
        print("   3. 5bps 交易成本远低于实盘真实成本")
        print("   4. 没考虑做空借券成本、流动性冲击等")
        print("\n   这些是「理论 Sharpe」，实盘要打 3-5 折")
    elif suspicious_count >= 3:
        print("❌ 严重警告：代码可能存在未来数据泄漏！")
        print(f"   {suspicious_count}/5 个随机因子 |Sharpe|>1，这不可能是巧合")
        print("\n🔍 建议检查：")
        print("   1. fetch_factor_data 中 target_ret 定义")
        print("   2. 因子表达式中是否使用了 Ref(x, 负数)")
        print("   3. groupby + transform 是否跨日期混合")
    else:
        print(f"⚠️ 边界情况：{suspicious_count}/5 个随机因子 Sharpe 偏高")
        print("   建议增大样本量重测，或使用更长的时间窗口")


if __name__ == "__main__":
    main()