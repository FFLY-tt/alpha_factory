# src/mining/seed_library.py
from typing import Dict, Tuple
from src.rules.rule_engine import DimType

# ==========================================
# 初始指标定义，基础指标种子库
# ==========================================


def load_seed_factors() -> Tuple[Dict[str, str], Dict[str, DimType]]:
    """
    获取经过严苛测试的 E100+ 工业级基础因子预制件库。
    返回:
        formulas: Dict[因子名, Qlib表达式字符串]
        dim_types: Dict[因子名, 量纲类型]
    """

    # 定义 VWAP 的日线近似值 (Typical Price)
    pseudo_vwap = "(($high + $low + $close) / 3)"
    formulas = {}
    dim_types = {}

    def add_factor(name: str, expr: str, dtype: DimType):
        formulas[name] = expr
        dim_types[name] = dtype

    # ==========================================
    # 1. 基础原始序列 (Raw Data)
    # 包含补充的 vwap (均价) 和 open (开盘价)
    # ==========================================
    add_factor("raw_close", "$close", DimType.PRICE)
    add_factor("raw_open", "$open", DimType.PRICE)
    add_factor("raw_high", "$high", DimType.PRICE)
    add_factor("raw_low", "$low", DimType.PRICE)
    add_factor("raw_vwap", pseudo_vwap, DimType.PRICE)
    add_factor("raw_volume", "$volume", DimType.VOLUME)

    # ==========================================
    # 2. 数据平滑与极值提纯 (Level 1 算子补充)
    # ==========================================
    # 对数平滑 (解决成交量长尾/偏态分布)
    add_factor("log_volume", "Log($volume + 0.00001)", DimType.RATIO)
    # 绝对涨跌幅 (波动幅度本身)
    add_factor("abs_delta_close", "Abs($close - Ref($close, 1))", DimType.PRICE)
    # 纯粹方向 (去极值，只看涨跌平)
    add_factor("sign_delta_close", "Sign($close - Ref($close, 1))", DimType.RATIO)
    add_factor("sign_delta_volume", "Sign($volume - Ref($volume, 1))", DimType.RATIO)

    # # ==========================================
    # # 3. 横截面相对强弱 (Cross-Sectional Rank)
    # # ==========================================
    # add_factor("rank_close", "CSRank($close)", DimType.RANK)
    # add_factor("rank_vwap", f"CSRank({pseudo_vwap})", DimType.RANK)
    # add_factor("rank_volume", "CSRank($volume)", DimType.RANK)

    # ==========================================
    # 4. 时序均线与通道极值 (Time-Series Smoothing & Envelope)
    # ==========================================
    for d in [5, 10, 20]:
        # 简单移动平均 (SMA)
        add_factor(f"mean_{d}_close", f"Mean($close, {d})", DimType.PRICE)
        add_factor(f"mean_{d}_vwap", f"Mean({pseudo_vwap}, {d})", DimType.PRICE)
        add_factor(f"mean_{d}_volume", f"Mean($volume, {d})", DimType.VOLUME)

        # 线性衰减加权平均 (WMA - Alpha 101 核心平滑器)
        add_factor(f"wma_{d}_close", f"WMA($close, {d})", DimType.PRICE)

        # 通道极值 (Max/Min - 支撑阻力位寻找)
        add_factor(f"max_{d}_high", f"Max($high, {d})", DimType.PRICE)
        add_factor(f"min_{d}_low", f"Min($low, {d})", DimType.PRICE)

    # ==========================================
    # 5. 绝对差分动量 (TS Delta)
    # ==========================================
    for d in [1, 3, 5, 10]:
        add_factor(f"delta_{d}_close", f"($close - Ref($close, {d}))", DimType.PRICE)
        add_factor(f"delta_{d}_volume", f"($volume - Ref($volume, {d}))", DimType.VOLUME)

    # ==========================================
    # 6. 相对比例动量 (TS Ratio)
    # ==========================================
    for d in [1, 3, 5, 10]:
        add_factor(f"mom_{d}_close", f"($close / Ref($close, {d}))", DimType.RATIO)
        add_factor(f"mom_{d}_vwap", f"({pseudo_vwap} / Ref({pseudo_vwap}, {d}))", DimType.RATIO)
        add_factor(f"mom_{d}_volume", f"($volume / Ref($volume, {d}))", DimType.RATIO)

    # ==========================================
    # 7. 日内与隔夜动量 (Intraday & Overnight) - 极具杀伤力
    # ==========================================
    # 日内收益 (今收 / 今开 - 捕捉日内资金抢筹或砸盘)
    add_factor("mom_intraday", "($close / $open)", DimType.RATIO)
    # 隔夜跳空 (今开 / 昨收 - 捕捉盘后消息发酵)
    add_factor("mom_overnight", "($open / Ref($close, 1))", DimType.RATIO)

    # ==========================================
    # 8. 波动率调整与风险度量 (Volatility / Std)
    # ==========================================
    for d in [5, 10, 20]:
        # 价格的绝对波动区间
        add_factor(f"std_{d}_close", f"Std($close, {d})", DimType.PRICE)
        # 收益率的相对波动率 (极其重要：无量纲风险指标)
        # ($close / Ref($close, 1) - 1) 就是当天的收益率
        add_factor(f"volatility_{d}", f"Std(($close / Ref($close, 1) - 1), {d})", DimType.RATIO)

    # ==========================================
    # 9. 量价时序共振 (TS Correlation)
    # ==========================================
    for d in [5, 10, 20]:
        add_factor(f"corr_cv_{d}", f"Corr($close, $volume, {d})", DimType.RATIO)
        # VWAP与量的相关性往往比Close更稳健
        add_factor(f"corr_vwap_v_{d}", f"Corr({pseudo_vwap}, $volume, {d})", DimType.RATIO)

    return formulas, dim_types


if __name__ == "__main__":
    # ======== 本地自测逻辑 ========
    f, dt = load_seed_factors()
    print("=" * 60)
    print(f"✅ 成功加载 AlphaForge 终极种子库，总计数量: {len(f)}")
    print("=" * 60)

    # 随机抽样打印几个不同维度的检查点
    checkpoints = ["wma_10_close", "mom_intraday", "volatility_20", "corr_vwap_v_10", "sign_delta_close"]
    for cp in checkpoints:
        if cp in f:
            print(f"[{dt[cp].name.ljust(6)}] {cp.ljust(18)} : {f[cp]}")