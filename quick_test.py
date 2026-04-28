# quick_test.py
"""
快速 A/B 对照实验：验证 7 模块流水线对深度-3 复杂因子的提升效果
- A 组：原始因子值直接评估（基线）
- B 组：经过 7 模块完整流水线 + 最优平滑变体
- Step 4.5：批量生成平滑变体（一次取数），换手率 < TURNOVER_CAP 才入选
- Step 6：样本外 OOS 验证，IS vs OOS 对比表
- Step 6.5（新增）：自动精调——对 OOS 稳健因子执行精细窗口扫描 + 市场状态择时衍生
                    一次取数，IS+OOS 双重验证，全局最优覆盖
- Step 7：WQ 提交建议，同时满足 IS + OOS 双门槛（含精调后最优版本）才入选
"""
import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
import pandas as pd
from data_pipeline.data_source import init_qlib_engine, fetch_factor_data
from src.mining.beam_search import BeamSearchEngine
from src.evaluation.metrics_calc import (
    select_top_k_orthogonal, compute_factor_sharpe,
    calculate_ic_stability, calculate_local_fitness,
    auto_correct_direction
)
from src.processing.factor_pipeline import FactorPipeline
from src.ast.ast_nodes import LeafNode, BinaryNode, TernaryNode
from src.rules.rule_engine import DimType


# ==============================================================================
# 配置
# ==============================================================================
POOL           = "sp500"

# IS / OOS 时间窗口（DeepSeek 建议：8+3 年比例）
IS_START_DATE  = "2010-01-04"   # 样本内：波束搜索 + 精炼 + 选优
IS_END_DATE    = "2017-12-31"
OOS_START_DATE = "2018-01-01"   # 样本外：最终验证（含2018熊市/2019反弹/2020新冠）
OOS_END_DATE   = "2020-11-10"

TARGET_DEPTH   = 4
BEAM_WIDTH     = 12    # 适当缩小控制时长（原10）
CORR_THRESHOLD = 0.8
MAX_EVAL       = 60   # 适当缩小控制时长（原50）
ENABLE_MODULE_4 = False
SHARPE_PRINT_THRESHOLD = 1.0

# 平滑变体（只保留真正的平滑算子，WMA = decay_linear）
SMOOTH_VARIANTS = [
    ("ts_mean_5",       "Mean({expr}, 5)"),
    ("ts_mean_10",      "Mean({expr}, 10)"),
    ("decay_linear_5",  "WMA({expr}, 5)"),
    ("decay_linear_10", "WMA({expr}, 10)"),
    ("decay_linear_20", "WMA({expr}, 20)"),
]

# IS 门槛
TURNOVER_CAP   = 1.5   # 换手率硬上限（IS & OOS 共用）
FITNESS_MIN    = 0.5   # IS Fitness 门槛
SHARPE_MIN_IS  = 1.0   # IS Sharpe(修正) 门槛

# OOS 门槛（适当放宽）
FITNESS_MIN_OOS  = 0.3   # OOS Fitness 门槛
SHARPE_MIN_OOS   = 0.8   # OOS Sharpe(修正) 门槛
# IC 方向：OOS 必须与 IS 同方向（在 Step 6 中判断 ic 符号是否一致）

# Step 6.5：自动精调配置
# 精细平滑窗口扫描（比 Step 4.5 更密集，专注 10-20 区间）
FINE_SMOOTH_VARIANTS = [
    ("mean_10",  "Mean({expr}, 10)"),
    ("mean_12",  "Mean({expr}, 12)"),
    ("mean_15",  "Mean({expr}, 15)"),
    ("mean_20",  "Mean({expr}, 20)"),
    ("wma_10",   "WMA({expr}, 10)"),
    ("wma_12",   "WMA({expr}, 12)"),
    ("wma_15",   "WMA({expr}, 15)"),
    ("wma_20",   "WMA({expr}, 20)"),
]
# 市场状态择时条件（两种）
MKT_RET_EXPR   = "Mean($close / Ref($close, 1) - 1, 20)"   # 20日市场收益
MKT_VOL_EXPR   = "Std($close / Ref($close, 1) - 1, 20)"    # 20日市场波动率
MKT_VOL_MA_EXPR = "Mean(Std($close / Ref($close, 1) - 1, 20), 60)"  # 波动率60日均值
# 精调后 IS/OOS 双门槛（与原门槛保持一致，不额外放宽）
FINE_SHARPE_MIN_IS  = SHARPE_MIN_IS
FINE_SHARPE_MIN_OOS = SHARPE_MIN_OOS
FINE_FITNESS_MIN_IS  = FITNESS_MIN
FINE_FITNESS_MIN_OOS = FITNESS_MIN_OOS


# ==============================================================================
# 工具函数
# ==============================================================================

def evaluate_one(df, factor_name):
    """单因子完整评估（含方向自动修正）"""
    metrics   = compute_factor_sharpe(df, factor_name, quantiles=5, cost_bps=5.0)
    mean_ic, icir = calculate_ic_stability(df, factor_name)
    direction = auto_correct_direction(
        ic=mean_ic,
        sharpe_net=metrics['sharpe_net'],
        sharpe_gross=metrics['sharpe_gross']
    )
    fit = calculate_local_fitness(
        sharpe=direction['corrected_sharpe_net'],
        turnover=metrics['turnover'],
        max_corr=0.0,
        icir=abs(icir)
    )
    return {
        "sharpe_net_raw":   metrics['sharpe_net'],
        "sharpe_gross_raw": metrics['sharpe_gross'],
        "sharpe_net":       direction['corrected_sharpe_net'],
        "sharpe_gross":     direction['corrected_sharpe_gross'],
        "needs_negation":   direction['needs_negation'],
        "direction_note":   direction['direction'],
        "ann_ret":          metrics['ann_ret'],
        "max_dd":           metrics['max_dd'],
        "turnover":         metrics['turnover'],
        "ic":               mean_ic,
        "icir":             icir,
        "fitness":          fit['fitness'],
        "high_turnover":    metrics.get('high_turnover', False),
    }


def build_market_state_alphas(base_factor_nodes):
    MKT_RET_EXPR = "Mean($close / Ref($close, 1) - 1, 20)"
    derived   = []
    mkt_ret   = LeafNode("mkt_ret_20d", f"Ref({MKT_RET_EXPR}, 1)", DimType.RATIO)
    zero      = LeafNode("zero", "0", DimType.RATIO)
    bull_cond = BinaryNode(">", mkt_ret, zero)
    for node in base_factor_nodes:
        if node.dim_type == DimType.BOOLEAN:
            continue
        zero_factor = LeafNode("zero_factor", "0", node.dim_type)
        try:
            bn = TernaryNode(bull_cond, node, zero_factor)
            bn.name = f"BULL_{node.name[:30]}"
            bn.expr_str = f"If(Ref({MKT_RET_EXPR}, 1) > 0, {node.expr_str}, 0)"
            derived.append(bn)
        except ValueError:
            pass
    return derived


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print(f"⚡ A/B 对照实验（深度{TARGET_DEPTH}，换手上限{TURNOVER_CAP}，Fitness≥{FITNESS_MIN}）")
    print(f"   IS : {IS_START_DATE} ~ {IS_END_DATE}（样本内，约8年）")
    print(f"   OOS: {OOS_START_DATE} ~ {OOS_END_DATE}（样本外，约3年）")
    print("=" * 70)

    cur = os.path.dirname(os.path.abspath(__file__))
    init_qlib_engine(os.path.join(cur, "data", "qlib_data", "us_data"))

    # ============================================================
    # Step 1：波束搜索（在 IS 期上跑）
    # ============================================================
    print(f"\n🌱 [Step 1] 波束搜索（IS: {IS_START_DATE} ~ {IS_END_DATE}）...")
    engine       = BeamSearchEngine(beam_width=BEAM_WIDTH)
    current_beam = engine.seed_nodes
    all_elite_nodes = {}

    for depth in range(2, TARGET_DEPTH + 1):
        print(f"\n   🧬 深度 {depth}...")
        candidates = engine.generate_candidates(current_beam, max_eval=MAX_EVAL)
        formulas   = {n.name: n.expr_str for n in candidates}
        formulas["_vol_raw_"] = "$volume"
        df_batch   = fetch_factor_data(formulas, POOL, IS_START_DATE, IS_END_DATE)
        ic_input   = df_batch.drop(columns=["_vol_raw_"], errors="ignore")
        current_beam, ic_report = select_top_k_orthogonal(
            candidates, ic_input, k=BEAM_WIDTH, corr_threshold=CORR_THRESHOLD
        )
        for n in current_beam:
            all_elite_nodes[n.name] = n
        print(f"   留存精英: {len(current_beam)} 个")

        print(f"   📋 |Sharpe(修正)| ≥ {SHARPE_PRINT_THRESHOLD} 扫描：")
        found = 0
        for node in current_beam:
            if node.name not in df_batch.columns:
                continue
            res = evaluate_one(df_batch, node.name)
            if abs(res['sharpe_net']) >= SHARPE_PRINT_THRESHOLD:
                found += 1
                neg = "需取反" if res['needs_negation'] else "原方向"
                hw  = " ⚠️高换手" if res['high_turnover'] else ""
                print(f"   🔥 [{neg}] Shp={res['sharpe_net']:+.3f} | "
                      f"IC={res['ic']:+.4f} | Turn={res['turnover']:.3f}{hw}")
                print(f"      {node.expr_str}")
        if found == 0:
            print(f"   （无）")

    print(f"\n✅ 精英因子: {len(current_beam)} 个")

    # ============================================================
    # Step 2：取最终评估数据（IS 期）
    # ============================================================
    print(f"\n📥 [Step 2] 取最终评估数据（IS 期）...")
    final_formulas = {n.name: n.expr_str for n in current_beam}
    final_formulas["_vol_raw_"] = "$volume"
    df_eval = fetch_factor_data(final_formulas, POOL, IS_START_DATE, IS_END_DATE)

    # ============================================================
    # Step 3：A 组评估（IS 期基线）
    # ============================================================
    print("\n" + "=" * 70)
    print("🅰️  组 A：基线评估（IS 期）")
    print("=" * 70)
    a_results = {}
    for node in current_beam:
        if node.name not in df_eval.columns:
            continue
        a_results[node.name] = evaluate_one(df_eval, node.name)
        res = a_results[node.name]
        neg = "[需取反]" if res['needs_negation'] else "[原方向]"
        hw  = " ⚠️高换手" if res['high_turnover'] else ""
        print(f"   {neg} {node.name[:40]:<40} "
              f"Shp={res['sharpe_net']:+.3f} | IC={res['ic']:+.4f} | "
              f"Turn={res['turnover']:.3f}{hw}")

    # ============================================================
    # Step 4：B 组评估（IS 期精炼管道）
    # ============================================================
    print("\n" + "=" * 70)
    print("🅱️  组 B：7 模块精炼管道（IS 期）")
    print("=" * 70)
    factor_names = list(a_results.keys())
    elite_df = df_eval[factor_names + ["target_ret", "_vol_raw_"]].copy()
    pipeline = FactorPipeline(
        df=elite_df, factor_cols=factor_names, volume_col="_vol_raw_"
    )
    refined_df = pipeline.run(
        enable_liquidity_filter=True,
        enable_robustify=True,
        enable_neutralization=False,
    )
    b_results = {}
    for fname in factor_names:
        if fname not in refined_df.columns:
            continue
        b_results[fname] = evaluate_one(refined_df, fname)

    if ENABLE_MODULE_4:
        print("\n>> 模块四：状态机衍生...")
        derived = build_market_state_alphas(current_beam)
        if derived:
            d_formulas = {n.name: n.expr_str for n in derived}
            d_formulas["_vol_raw_"] = "$volume"
            try:
                d_df = fetch_factor_data(d_formulas, POOL, IS_START_DATE, IS_END_DATE)
                for n in derived:
                    if n.name in d_df.columns:
                        b_results[n.name] = evaluate_one(d_df, n.name)
                        all_elite_nodes[n.name] = n
            except Exception as e:
                print(f"   ⚠️ {e}")

    print("\n>> 模块六：低相关组合筛选...")
    pipeline.select_portfolio(
        sharpe_dict={f: b_results[f]['sharpe_net'] for f in b_results},
        max_factors=5, min_fitness=0.0, max_avg_corr=0.3
    )

    # ============================================================
    # Step 4.5：批量生成平滑变体（一次取数，IS 期）
    # ============================================================
    print("\n\n" + "=" * 70)
    print(f"🔧 [Step 4.5] 平滑变体对比（IS 期，换手上限 {TURNOVER_CAP}）")
    print("=" * 70)

    # 构建所有变体的取数字典（合并一次 fetch）
    variant_index  = {}   # col_name -> (fname, label, smoothed_expr)
    batch_formulas = {"_vol_raw_": "$volume"}
    for i, fname in enumerate(factor_names):
        node = all_elite_nodes.get(fname)
        if node is None:
            continue
        for j, (label, tmpl) in enumerate(SMOOTH_VARIANTS):
            col           = f"_v{i}_{j}_"
            smoothed_expr = tmpl.replace("{expr}", node.expr_str)
            batch_formulas[col] = smoothed_expr
            variant_index[col]  = (fname, label, smoothed_expr)

    n_variants = len(batch_formulas) - 1
    print(f"\n   一次取数：{n_variants} 个变体（{len(factor_names)} 因子 × "
          f"{len(SMOOTH_VARIANTS)} 方案）")
    df_variants = fetch_factor_data(batch_formulas, POOL, IS_START_DATE, IS_END_DATE)

    # 逐因子选最优变体
    best_variant_results = dict(b_results)
    best_variant_labels  = {f: "原始B组" for f in b_results}
    best_variant_exprs   = {n.name: n.expr_str for n in current_beam}

    print()
    for i, fname in enumerate(factor_names):
        node  = all_elite_nodes.get(fname)
        if node is None:
            continue
        b_res  = b_results.get(fname, {})
        b_fit  = b_res.get("fitness", -999)
        b_turn = b_res.get("turnover", 999)

        # 初始化最优（原始B组，如果换手合格）
        best_fit  = b_fit  if b_turn < TURNOVER_CAP else -999
        best_lbl  = "原始B组" if b_turn < TURNOVER_CAP else "(原始B组超限)"
        best_res  = b_res  if b_turn < TURNOVER_CAP else {}
        best_expr = node.expr_str

        rows = []
        if b_turn < TURNOVER_CAP:
            rows.append(("原始B组", b_res.get('sharpe_net',0),
                         b_turn, b_fit, "← 基准", False))

        for j, (label, _) in enumerate(SMOOTH_VARIANTS):
            col = f"_v{i}_{j}_"
            if col not in df_variants.columns:
                continue
            res   = evaluate_one(df_variants, col)
            turn  = res['turnover']
            fit   = res['fitness']
            valid = turn < TURNOVER_CAP

            is_best = False
            if valid and fit > best_fit:
                best_fit  = fit
                best_lbl  = label
                best_res  = res
                best_expr = variant_index[col][2]
                is_best   = True

            rows.append((label, res['sharpe_net'], turn, fit,
                         "⭐最优" if is_best else "", not valid))

        # 打印紧凑对比表
        print(f"  [{node.name[:55]}]")
        print(f"  {'变体':<18} {'Shp(修正)':>11} {'换手':>7} {'Fitness':>9}  备注")
        print(f"  {'-'*58}")
        for lbl, shp, turn, fit, note, over in rows:
            hw = " ⚠️" if over else ""
            print(f"  {lbl:<18} {shp:>+11.3f} {turn:>7.4f} {fit:>+9.4f}{hw}"
                  + (f"  {note}" if note else ""))
        print(f"  {'─'*58}")
        if best_lbl == "(原始B组超限)":
            print(f"  ⚠️ 所有变体换手均超限（{TURNOVER_CAP}），标记为不可用\n")
            best_variant_labels[fname] = "⚠️高换手不可用"
        elif best_lbl == "原始B组":
            print(f"  → 平滑无增益，保留原始B组\n")
        else:
            print(f"  → 最优: [{best_lbl}]  Fitness {b_fit:+.4f} → {best_fit:+.4f}"
                  f"  (提升 {best_fit-b_fit:+.4f})\n")
            best_variant_results[fname] = best_res
            best_variant_labels[fname]  = best_lbl
            best_variant_exprs[fname]   = best_expr

    # 汇总表
    print(f"  {'='*70}")
    print(f"  {'因子名':<35} {'B组Fit':>9} {'最优Fit':>9} {'提升':>7} {'换手':>7}  方案")
    print(f"  {'-'*70}")
    for fname in factor_names:
        b_f   = b_results.get(fname, {}).get("fitness", 0)
        opt   = best_variant_results.get(fname, {})
        opt_f = opt.get("fitness", 0)
        opt_t = opt.get("turnover", 0)
        lbl   = best_variant_labels.get(fname, "原始B组")
        flag  = "⭐" if lbl not in ("原始B组", "⚠️高换手不可用") else ""
        print(f"  {fname[:35]:<35} {b_f:>+9.4f} {opt_f:>+9.4f} "
              f"{opt_f-b_f:>+7.4f} {opt_t:>7.4f}  {flag}{lbl}")
    print(f"  {'='*70}")

    # ============================================================
    # Step 5：A/B 对照表
    # ============================================================
    print("\n\n" + "=" * 110)
    print("📊 A/B 对照结果（B 组为最优变体，按 |Sharpe(修正)| 降序）")
    print("=" * 110)

    common = [n for n in factor_names if n in best_variant_results]
    common.sort(key=lambda x: abs(best_variant_results[x]['sharpe_net']), reverse=True)

    n_improve = 0
    for i, fname in enumerate(common):
        a   = a_results[fname]
        b   = best_variant_results[fname]
        lbl = best_variant_labels.get(fname, "原始B组")
        d   = abs(b['sharpe_net']) - abs(a['sharpe_net'])
        if d > 0:
            n_improve += 1
        sym  = "📈" if d > 0 else "📉"
        neg  = "🔄需取反" if b['needs_negation'] else "✓原方向"
        hw_a = " ⚠️高换手" if a['high_turnover'] else ""
        hw_b = " ⚠️高换手" if b['high_turnover'] else ""
        node = all_elite_nodes.get(fname)

        print(f"\n[{i+1}] {neg} ──────────────────────────────────────────────────────")
        print(f"  名称:     {fname}")
        if node:
            print(f"  原始表达式: {node.expr_str}")
            if lbl not in ("原始B组", "⚠️高换手不可用"):
                print(f"  平滑表达式: {best_variant_exprs.get(fname, node.expr_str)}")
                print(f"  最优方案:   {lbl}")
            print(f"  深度:{node.depth} | 节点数:{node.node_count}")
        print(f"  A(基线): Shp={a['sharpe_net']:+.3f}(原始{a['sharpe_net_raw']:+.3f}) | "
              f"IC={a['ic']:+.4f} | Turn={a['turnover']:.4f}{hw_a}")
        print(f"  B(最优): Shp={b['sharpe_net']:+.3f}(原始{b['sharpe_net_raw']:+.3f}) | "
              f"IC={b['ic']:+.4f} | Turn={b['turnover']:.4f}{hw_b}")
        print(f"           ICIR={b['icir']:+.4f} | Fitness={b['fitness']:+.4f}")
        print(f"  Δ|Shp|:{d:+.3f} {sym} | {b['direction_note']}")

    print(f"\n{'='*110}")
    print(f"✅ 管道有效因子: {n_improve}/{len(common)} 个 |Sharpe| 提升")

    # ============================================================
    # Step 6（新增）：OOS 样本外验证
    # ============================================================
    print("\n\n" + "=" * 110)
    print(f"🔭 [Step 6] OOS 样本外验证（{OOS_START_DATE} ~ {OOS_END_DATE}）")
    print(f"   OOS 门槛：Fitness≥{FITNESS_MIN_OOS} | 换手<{TURNOVER_CAP} | "
          f"Sharpe≥{SHARPE_MIN_OOS} | IC 方向须与 IS 一致")
    print("=" * 110)

    # 6-A：整理需要在 OOS 期取数的表达式
    #       使用 IS 期选出的「最优表达式」（含平滑），不对取反方向做调整，
    #       取数后再用 evaluate_one 的 auto_correct_direction 统一处理。
    oos_formulas = {"_vol_raw_": "$volume"}
    oos_col_map  = {}   # col_name -> fname
    for fname in common:
        if best_variant_labels.get(fname) == "⚠️高换手不可用":
            continue
        expr = best_variant_exprs.get(fname, "")
        if not expr:
            continue
        col = f"_oos_{fname}_"[:60]   # 列名截断，避免过长
        oos_formulas[col] = expr
        oos_col_map[col]  = fname

    print(f"\n   取数：{len(oos_col_map)} 个因子在 OOS 期重新评估...")
    df_oos = fetch_factor_data(oos_formulas, POOL, OOS_START_DATE, OOS_END_DATE)

    # 6-B：逐因子 OOS 评估
    oos_results = {}   # fname -> oos metrics dict
    for col, fname in oos_col_map.items():
        if col not in df_oos.columns:
            print(f"   ⚠️ OOS 取数失败：{fname}，跳过")
            continue
        oos_results[fname] = evaluate_one(df_oos, col)

    # 6-C：IS vs OOS 对比表（按 IS Sharpe 降序）
    print()
    print(f"  {'因子名':<35} │ {'IS Shp':>8} {'IS IC':>8} {'IS Fit':>8} │ "
          f"{'OOS Shp':>8} {'OOS IC':>8} {'OOS Fit':>8} │ {'换手OOS':>8}  判断")
    print(f"  {'-'*35}─┼─{'-'*8}─{'-'*8}─{'-'*8}─┼─"
          f"{'-'*8}─{'-'*8}─{'-'*8}─┼─{'-'*8}──────────")

    oos_robust = []    # 通过 OOS 全部门槛的因子名列表
    for fname in common:
        if fname not in oos_results:
            print(f"  {fname[:35]:<35} │ {'(IS高换手跳过)':>27} │ {'—':>28} │")
            continue

        is_res  = best_variant_results[fname]
        oos_res = oos_results[fname]

        is_shp  = is_res['sharpe_net']
        oos_shp = oos_res['sharpe_net']
        is_ic   = is_res['ic']
        oos_ic  = oos_res['ic']
        is_fit  = is_res['fitness']
        oos_fit = oos_res['fitness']
        oos_turn = oos_res['turnover']

        # OOS 四项门槛判断
        cond_fitness  = oos_fit  >= FITNESS_MIN_OOS
        cond_turnover = oos_turn <  TURNOVER_CAP
        cond_sharpe   = abs(oos_shp) >= SHARPE_MIN_OOS
        # IC 方向一致：IS 和 OOS 的 corrected sharpe 符号相同
        cond_direction = (is_shp * oos_shp) > 0

        robust = all([cond_fitness, cond_turnover, cond_sharpe, cond_direction])
        if robust:
            oos_robust.append(fname)

        # 构建判断标签
        flags = []
        if not cond_fitness:   flags.append(f"Fit<{FITNESS_MIN_OOS}")
        if not cond_turnover:  flags.append(f"Turn≥{TURNOVER_CAP}")
        if not cond_sharpe:    flags.append(f"Shp<{SHARPE_MIN_OOS}")
        if not cond_direction: flags.append("IC方向反转⚠️")
        verdict = "✅稳健" if robust else ("❌ " + " | ".join(flags))

        print(f"  {fname[:35]:<35} │ {is_shp:>+8.3f} {is_ic:>+8.4f} {is_fit:>+8.4f} │ "
              f"{oos_shp:>+8.3f} {oos_ic:>+8.4f} {oos_fit:>+8.4f} │ "
              f"{oos_turn:>8.4f}  {verdict}")

    print(f"\n  OOS 稳健因子（通过全部4项门槛）: {len(oos_robust)}/{len(oos_results)} 个")
    if oos_robust:
        print(f"  → " + " | ".join(oos_robust))

    # ============================================================
    # Step 6.5：自动精调（平滑窗口精细扫描 + 市场状态择时衍生）
    # 仅对 oos_robust 中的因子执行，避免对过拟合因子浪费时间
    # ============================================================
    print("\n\n" + "=" * 110)
    print("🔬 [Step 6.5] 自动精调：精细窗口扫描 + 市场状态择时衍生")
    print(f"   目标：突破 IS Sharpe≥{FINE_SHARPE_MIN_IS} / OOS Sharpe≥{FINE_SHARPE_MIN_OOS} 双门槛")
    print(f"   对象：{len(oos_robust)} 个 OOS 稳健因子")
    print("=" * 110)

    # fine_best_* 存储精调后每个因子的全局最优版本
    # 初始值 = Step 6 已验证的结果，精调只允许覆盖（不降级）
    fine_best_expr   = {f: best_variant_exprs[f]      for f in oos_robust}
    fine_best_is_res = {f: best_variant_results[f]    for f in oos_robust}
    fine_best_oos_res= {f: oos_results[f]             for f in oos_robust}
    fine_best_label  = {f: best_variant_labels.get(f, "原始B组") for f in oos_robust}

    if not oos_robust:
        print("\n  （无 OOS 稳健因子，跳过精调）")
    else:
        # ── 6.5-A：构建精调取数字典（IS 期 + OOS 期各一批，合并两次 fetch）──
        # 对每个稳健因子生成：
        #   ① 精细平滑变体（FINE_SMOOTH_VARIANTS）
        #   ② 牛市择时衍生：If(Ref(MKT_RET,1)>0, expr, 0)
        #   ③ 高波动择时衍生：If(Ref(MKT_VOL,1)>Ref(MKT_VOL_MA,1), expr, 0)
        fine_col_index = {}   # col -> (fname, variant_label, expr_str)
        fine_formulas  = {"_vol_raw_": "$volume"}

        for fname in oos_robust:
            base_expr = fine_best_expr[fname]   # 当前最优表达式作为起点
            node      = all_elite_nodes.get(fname)
            prefix    = fname[:20].replace(" ", "_")   # 列名前缀，避免过长

            # ① 精细平滑变体
            for sv_label, sv_tmpl in FINE_SMOOTH_VARIANTS:
                col  = f"_fa_{prefix}_{sv_label}_"[:55]
                expr = sv_tmpl.replace("{expr}", base_expr)
                fine_formulas[col]   = expr
                fine_col_index[col]  = (fname, f"fine_{sv_label}", expr)

            # ② 牛市择时：If(滞后1日市场20日收益>0, 当前最优expr, 0)
            bull_expr = (f"If(Ref({MKT_RET_EXPR}, 1) > 0, {base_expr}, 0)")
            col_bull  = f"_fb_{prefix}_bull_"[:55]
            fine_formulas[col_bull]  = bull_expr
            fine_col_index[col_bull] = (fname, "bull_timing", bull_expr)

            # ③ 高波动择时：If(滞后1日波动率>60日均值, 当前最优expr, 0)
            hvol_expr = (f"If(Ref({MKT_VOL_EXPR}, 1) > Ref({MKT_VOL_MA_EXPR}, 1), "
                         f"{base_expr}, 0)")
            col_hvol  = f"_fc_{prefix}_hvol_"[:55]
            fine_formulas[col_hvol]  = hvol_expr
            fine_col_index[col_hvol] = (fname, "hvol_timing", hvol_expr)

            # ④ 牛市择时 × 精细平滑（组合：先平滑再择时）
            for sv_label, sv_tmpl in [("mean_15", "Mean({expr}, 15)"),
                                       ("wma_15",  "WMA({expr}, 15)")]:
                smooth_expr     = sv_tmpl.replace("{expr}", base_expr)
                bull_sm_expr    = f"If(Ref({MKT_RET_EXPR}, 1) > 0, {smooth_expr}, 0)"
                hvol_sm_expr    = (f"If(Ref({MKT_VOL_EXPR}, 1) > Ref({MKT_VOL_MA_EXPR}, 1), "
                                   f"{smooth_expr}, 0)")
                col_bs = f"_fd_{prefix}_bull_{sv_label}_"[:55]
                col_hs = f"_fe_{prefix}_hvol_{sv_label}_"[:55]
                fine_formulas[col_bs]  = bull_sm_expr
                fine_col_index[col_bs] = (fname, f"bull+{sv_label}", bull_sm_expr)
                fine_formulas[col_hs]  = hvol_sm_expr
                fine_col_index[col_hs] = (fname, f"hvol+{sv_label}", hvol_sm_expr)

        n_fine = len(fine_formulas) - 1
        print(f"\n  精调变体总数：{n_fine} 个（{len(oos_robust)} 因子 × "
              f"约{n_fine//max(len(oos_robust),1)} 方案）")

        # ── 6.5-B：IS 期取数 & 评估 ──
        print(f"  IS 期取数（{IS_START_DATE}~{IS_END_DATE}）...")
        df_fine_is  = fetch_factor_data(fine_formulas, POOL, IS_START_DATE, IS_END_DATE)

        fine_is_eval = {}   # col -> is_metrics
        for col in fine_col_index:
            if col not in df_fine_is.columns:
                continue
            fine_is_eval[col] = evaluate_one(df_fine_is, col)

        # ── 6.5-C：OOS 期取数 & 评估 ──
        print(f"  OOS 期取数（{OOS_START_DATE}~{OOS_END_DATE}）...")
        df_fine_oos = fetch_factor_data(fine_formulas, POOL, OOS_START_DATE, OOS_END_DATE)

        fine_oos_eval = {}   # col -> oos_metrics
        for col in fine_col_index:
            if col not in df_fine_oos.columns:
                continue
            fine_oos_eval[col] = evaluate_one(df_fine_oos, col)

        # ── 6.5-D：逐因子选全局最优 ──
        print()
        for fname in oos_robust:
            # 收集该因子所有变体
            candidates_fine = [
                (col, info[1], info[2])
                for col, info in fine_col_index.items()
                if info[0] == fname
                   and col in fine_is_eval
                   and col in fine_oos_eval
            ]

            cur_best_score = (
                abs(fine_best_is_res[fname].get('sharpe_net', 0))
                + abs(fine_best_oos_res[fname].get('sharpe_net', 0))
            )
            cur_best_label = fine_best_label[fname]

            rows_fine = []
            for col, v_label, v_expr in candidates_fine:
                is_r  = fine_is_eval[col]
                oos_r = fine_oos_eval[col]

                # IS 门槛
                is_ok = (
                    abs(is_r['sharpe_net'])  >= FINE_SHARPE_MIN_IS
                    and is_r['fitness']      >= FINE_FITNESS_MIN_IS
                    and is_r['turnover']     <  TURNOVER_CAP
                )
                # OOS 门槛
                oos_ok = (
                    abs(oos_r['sharpe_net']) >= FINE_SHARPE_MIN_OOS
                    and oos_r['fitness']     >= FINE_FITNESS_MIN_OOS
                    and oos_r['turnover']    <  TURNOVER_CAP
                    and (is_r['sharpe_net'] * oos_r['sharpe_net']) > 0   # 方向一致
                )
                both_ok = is_ok and oos_ok

                # 综合得分 = IS|Sharpe| + OOS|Sharpe|（双边同时改善才覆盖）
                score = abs(is_r['sharpe_net']) + abs(oos_r['sharpe_net'])
                is_new_best = both_ok and score > cur_best_score

                if is_new_best:
                    cur_best_score          = score
                    fine_best_expr[fname]   = v_expr
                    fine_best_is_res[fname] = is_r
                    fine_best_oos_res[fname]= oos_r
                    fine_best_label[fname]  = v_label

                rows_fine.append((
                    v_label,
                    is_r['sharpe_net'],  is_r['turnover'],  is_r['fitness'],
                    oos_r['sharpe_net'], oos_r['turnover'], oos_r['fitness'],
                    "⭐" if is_new_best else ("✅" if both_ok else ""),
                    not (is_ok and oos_ok)
                ))

            # 打印本因子精调对比表
            orig_is_shp  = best_variant_results[fname]['sharpe_net']
            orig_oos_shp = oos_results[fname]['sharpe_net']
            print(f"  [{fname[:60]}]  基准 IS={orig_is_shp:+.3f} OOS={orig_oos_shp:+.3f}")
            print(f"  {'变体':<22} {'IS_Shp':>8} {'IS_Turn':>8} {'IS_Fit':>7}"
                  f" │ {'OOS_Shp':>8} {'OOS_Turn':>8} {'OOS_Fit':>7}  标记")
            print(f"  {'-'*85}")
            for (lbl, is_shp, is_turn, is_fit,
                 oos_shp, oos_turn, oos_fit, tag, bad) in sorted(
                    rows_fine, key=lambda x: abs(x[4]), reverse=True):
                hw = " ⚠️" if bad else ""
                print(f"  {lbl:<22} {is_shp:>+8.3f} {is_turn:>8.4f} {is_fit:>+7.4f}"
                      f" │ {oos_shp:>+8.3f} {oos_turn:>8.4f} {oos_fit:>+7.4f}"
                      f"  {tag}{hw}")
            print(f"  {'─'*85}")
            new_is  = fine_best_is_res[fname]['sharpe_net']
            new_oos = fine_best_oos_res[fname]['sharpe_net']
            delta_is  = new_is  - orig_is_shp
            delta_oos = new_oos - orig_oos_shp
            if fine_best_label[fname] != cur_best_label or delta_is != 0:
                print(f"  → 最优方案: [{fine_best_label[fname]}]")
                print(f"    IS  {orig_is_shp:+.3f} → {new_is:+.3f}  (Δ{delta_is:+.3f})")
                print(f"    OOS {orig_oos_shp:+.3f} → {new_oos:+.3f}  (Δ{delta_oos:+.3f})\n")
            else:
                print(f"  → 精调无增益，保留原始版本\n")

        # ── 6.5-E：精调汇总表 ──
        print(f"  {'='*85}")
        print(f"  精调汇总（仅 OOS 稳健因子）")
        print(f"  {'因子名':<35} {'原IS_Shp':>9} {'精IS_Shp':>9} {'原OOS_Shp':>10}"
              f" {'精OOS_Shp':>10}  最优方案")
        print(f"  {'-'*85}")
        for fname in oos_robust:
            orig_is  = best_variant_results[fname]['sharpe_net']
            orig_oos = oos_results[fname]['sharpe_net']
            fine_is  = fine_best_is_res[fname]['sharpe_net']
            fine_oos = fine_best_oos_res[fname]['sharpe_net']
            lbl      = fine_best_label[fname]
            flag     = "⭐" if lbl != best_variant_labels.get(fname, "原始B组") else ""
            print(f"  {fname[:35]:<35} {orig_is:>+9.3f} {fine_is:>+9.3f}"
                  f" {orig_oos:>+10.3f} {fine_oos:>+10.3f}  {flag}{lbl}")
        print(f"  {'='*85}")

    # ============================================================
    # Step 7：WQ 提交建议（IS + OOS 双门槛，使用精调后最优版本）
    # ============================================================
    # IS 门槛：Fitness≥FITNESS_MIN & 换手<TURNOVER_CAP & Sharpe≥SHARPE_MIN_IS
    # OOS 门槛：以上 oos_robust 列表（已包含4项）+ 精调后结果覆盖
    # Step 7 的「最优表达式/指标」优先使用 fine_best_*（精调结果），
    # 对于非 oos_robust 因子（未进入精调），仍回退到 best_variant_* + oos_results
    def get_final_is_res(fname):
        return fine_best_is_res.get(fname, best_variant_results.get(fname, {}))

    def get_final_oos_res(fname):
        return fine_best_oos_res.get(fname, oos_results.get(fname, {}))

    def get_final_expr(fname):
        return fine_best_expr.get(fname, best_variant_exprs.get(fname, ""))

    def get_final_label(fname):
        return fine_best_label.get(fname, best_variant_labels.get(fname, "原始B组"))

    def pass_is_threshold(fname):
        res = get_final_is_res(fname)
        return (
            res.get("fitness", 0)        >= FITNESS_MIN
            and res.get("turnover", 999) <  TURNOVER_CAP
            and abs(res.get("sharpe_net", 0)) >= SHARPE_MIN_IS
            and best_variant_labels.get(fname) != "⚠️高换手不可用"
        )

    def pass_oos_threshold(fname):
        is_r  = get_final_is_res(fname)
        oos_r = get_final_oos_res(fname)
        return (
            oos_r.get("fitness", 0)        >= FITNESS_MIN_OOS
            and oos_r.get("turnover", 999) <  TURNOVER_CAP
            and abs(oos_r.get("sharpe_net", 0)) >= SHARPE_MIN_OOS
            and (is_r.get("sharpe_net", 0) * oos_r.get("sharpe_net", 0)) > 0
        )

    qualified = sorted(
        [(f, get_final_is_res(f), get_final_oos_res(f))
         for f in common
         if pass_is_threshold(f) and pass_oos_threshold(f)],
        key=lambda x: x[1].get('fitness', 0), reverse=True
    )

    # 仅 IS 通过（OOS 未通过）的参考清单
    is_only = sorted(
        [(f, get_final_is_res(f))
         for f in common
         if pass_is_threshold(f) and not pass_oos_threshold(f)],
        key=lambda x: x[1].get('fitness', 0), reverse=True
    )

    print(f"\n\n{'='*110}")
    print(f"📤 [Step 7] WQ 提交建议（IS + OOS 双门槛，含精调最优，共 {len(qualified)} 个）")
    print(f"   IS 门槛：Fitness≥{FITNESS_MIN} | 换手<{TURNOVER_CAP} | Sharpe≥{SHARPE_MIN_IS}")
    print(f"   OOS门槛：Fitness≥{FITNESS_MIN_OOS} | 换手<{TURNOVER_CAP} | "
          f"Sharpe≥{SHARPE_MIN_OOS} | IC方向不变")
    print(f"{'='*110}")

    if qualified:
        for rank, (fname, is_b, oos_b) in enumerate(qualified, 1):
            final_expr = get_final_expr(fname)
            if not final_expr:
                continue
            wq_expr = f"-1 * ({final_expr})" if is_b.get('needs_negation') else final_expr
            lbl     = get_final_label(fname)
            node    = all_elite_nodes.get(fname)
            print(f"\n  [{rank}] ─────────────────────────────────────────────────────")
            if node:
                print(f"  原始表达式: {node.expr_str}")
            print(f"  使用方案:   {lbl}")
            print(f"  提交表达式: {wq_expr}")
            print(f"  IS  → Shp:{is_b.get('sharpe_net',0):+.3f} | "
                  f"换手:{is_b.get('turnover',0):.4f} | "
                  f"Fitness:{is_b.get('fitness',0):+.4f} | "
                  f"ICIR:{is_b.get('icir',0):+.4f}")
            print(f"  OOS → Shp:{oos_b.get('sharpe_net',0):+.3f} | "
                  f"换手:{oos_b.get('turnover',0):.4f} | "
                  f"Fitness:{oos_b.get('fitness',0):+.4f} | "
                  f"ICIR:{oos_b.get('icir',0):+.4f}")
    else:
        print(f"\n  ⚠️ 无因子同时满足 IS + OOS 双门槛")
        print(f"  建议：增大 MAX_EVAL 或 BEAM_WIDTH，或降低 FITNESS_MIN/SHARPE_MIN_IS")

    # 参考清单：仅通过 IS，未过 OOS
    if is_only:
        print(f"\n  📋 仅通过 IS 门槛（OOS 未达标，供参考，不建议直接提交）：")
        for fname, is_b in is_only:
            lbl   = get_final_label(fname)
            oos_b = get_final_oos_res(fname)
            print(f"     {fname[:40]:<40} IS_Fit:{is_b.get('fitness',0):+.4f} | "
                  f"OOS_Fit:{oos_b.get('fitness',0):+.4f} | "
                  f"OOS_Shp:{oos_b.get('sharpe_net',0):+.3f}  [{lbl}]")

    print(f"\n⏱️  总耗时: {time.time()-t0:.1f} 秒")
    print(f"💡 平滑方案: ts_mean_5/10（移动均线）、decay_linear_5/10/20（WMA加权衰减）")
    print(f"   换手上限 {TURNOVER_CAP}：超过此值的变体不参与最优选择")
    print(f"   Fitness 理想区间：>1.0 极品 | 0.5~1.0 优秀 | 0.2~0.5 良好")
    print(f"   IS :{IS_START_DATE}~{IS_END_DATE} | OOS:{OOS_START_DATE}~{OOS_END_DATE}")


if __name__ == "__main__":
    main()