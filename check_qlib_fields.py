# check_qlib_fields.py
"""
检测 Qlib 数据中有哪些字段可用
主要验证：行业分类、市值等字段是否存在
"""
import warnings
warnings.filterwarnings("ignore")
import os

if __name__ == "__main__":
    from data_pipeline.data_source import init_qlib_engine
    from qlib.data import D

    cur = os.path.dirname(os.path.abspath(__file__))
    init_qlib_engine(os.path.join(cur, "data", "qlib_data", "us_data"))

    # 候选字段列表（逐个尝试，失败则标记不可用）
    candidate_fields = {
        # 基础价量（确认基线）
        "close":          "$close",
        "open":           "$open",
        "high":           "$high",
        "low":            "$low",
        "volume":         "$volume",
        "vwap":           "($high + $low + $close) / 3",

        # 市值相关
        "market_cap":     "$market_cap",
        "cap":            "$cap",
        "float_shares":   "$float_shares",
        "total_shares":   "$total_shares",
        "close_x_vol":    "$close * $volume",      # 用成交额代替市值

        # 行业分类
        "industry":       "$industry",
        "sector":         "$sector",
        "ind":            "$ind",
        "industry_code":  "$industry_code",

        # 换手率
        "turnover_rate":  "$turnover_rate",
        "turn":           "$turn",

        # 其他常见字段
        "pe":             "$pe",
        "pb":             "$pb",
        "ps":             "$ps",
        "adjclose":       "$adjclose",
        "factor":         "$factor",
    }

    # 只取 1 只股票、1 周数据，速度极快
    instruments = D.instruments(market="sp500")

    print("\n🔍 Qlib 字段检测结果：")
    print("=" * 50)

    available   = []
    unavailable = []

    for name, expr in candidate_fields.items():
        try:
            df = D.features(
                instruments=instruments,
                fields=[expr],
                start_time="2015-01-05",
                end_time="2015-01-10",
                freq="day"
            )
            # 有数据且不全是 NaN 才算可用
            if df is not None and not df.empty and df.iloc[:, 0].notna().any():
                available.append((name, expr))
                print(f"  ✅ {name:<18} {expr}")
            else:
                unavailable.append((name, expr))
                print(f"  ⚠️ {name:<18} 字段存在但全为 NaN")
        except Exception:
            unavailable.append((name, expr))
            print(f"  ❌ {name:<18} 不可用")

    print("=" * 50)
    print(f"\n可用字段: {len(available)} 个")
    print(f"不可用字段: {len(unavailable)} 个")

    # 给出中性化建议
    print("\n" + "=" * 50)
    print("📋 中性化可行性判断：")
    print("=" * 50)

    has_industry = any(n in ["industry", "sector", "ind", "industry_code"]
                       for n, _ in available)
    has_mktcap   = any(n in ["market_cap", "cap", "float_shares", "total_shares"]
                       for n, _ in available)
    has_turnover = any(n in ["turnover_rate", "turn"] for n, _ in available)

    if has_industry:
        print("✅ 行业分类字段可用 → 可以开启行业中性化")
    else:
        print("❌ 无行业分类字段 → 行业中性化不可用")
        print("   替代方案：用 sector ETF 收益率回归做手动行业中性化")

    if has_mktcap:
        print("✅ 市值字段可用 → 可以开启市值中性化")
    else:
        print("❌ 无市值字段 → 市值中性化不可用")
        print("   替代方案：用 $close * $volume 作为市值代理变量")

    if has_turnover:
        print("✅ 换手率字段可用 → 可以做流动性分析")
    else:
        print("❌ 无换手率字段 → 用 $volume 代替")

    print()
    if has_industry and has_mktcap:
        print("🎯 结论：可以完整开启模块三中性化（行业 + 市值）")
        print("   下一步：在 fetch_factor_data 时加入行业/市值字段，")
        print("            并在 FactorPipeline 初始化时传入对应列名")
    elif has_industry or has_mktcap:
        print("🎯 结论：可以部分开启中性化")
        if has_industry:
            print("   → 只做行业中性化（市值数据缺失）")
        else:
            print("   → 只做市值中性化（行业数据缺失）")
    else:
        print("🎯 结论：数据不支持中性化，维持 enable_neutralization=False")
        print("   可行的替代方案：")
        print("   1. 用 yahoo finance / WRDS 补充行业/市值数据")
        print("   2. 在因子表达式层面手动做行业相对化")
        print("      例如：factor - Mean(factor_in_same_sector, window)")