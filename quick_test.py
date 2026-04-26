# quick_test.py
"""
快速 A/B 对照实验：验证 7 模块流水线对深度-3 复杂因子的提升效果
- A 组：原始因子值直接评估（基线，模拟未做任何优化）
- B 组：经过 7 模块完整流水线后评估
- 自动方向修正（IC>0 时取反 Sharpe，并提示 WQ 提交需取反）
"""
import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
from data_pipeline.data_source import init_qlib_engine, fetch_factor_data
from src.mining.beam_search import BeamSearchEngine
from src.evaluation.metrics_calc import (
    select_top_k_orthogonal, compute_factor_sharpe,
    calculate_ic_stability, calculate_local_fitness, evaluate_factor_fitness,
    auto_correct_direction
)
from src.processing.factor_pipeline import FactorPipeline
from src.ast.ast_nodes import LeafNode, BinaryNode, TernaryNode
from src.rules.rule_engine import DimType


# ==============================================================================
# 配置
# ==============================================================================
POOL              = "sp500"
START_DATE        = "2020-01-01"
END_DATE          = "2021-12-31"
TARGET_DEPTH      = 3
BEAM_WIDTH        = 10
CORR_THRESHOLD    = 0.8
MAX_EVAL          = 50

ENABLE_MODULE_4   = False
SHARPE_PRINT_THRESHOLD = 1.0


def evaluate_one(df, factor_name):
    """
    单因子完整评估（含方向自动修正）
    
    返回字段说明：
    - sharpe_net_raw / sharpe_gross_raw: 原始 Sharpe（未修正方向）
    - sharpe_net / sharpe_gross: 方向修正后的 Sharpe（用于 Fitness 计算）
    - needs_negation: 是否需要在 WQ 提交时给表达式加 -1 *
    - fitness: 用修正后的 sharpe_net 计算
    """
    metrics = compute_factor_sharpe(df, factor_name, quantiles=5, cost_bps=5.0)
    mean_ic, icir = calculate_ic_stability(df, factor_name)

    # 方向修正
    direction = auto_correct_direction(
        ic=mean_ic,
        sharpe_net=metrics['sharpe_net'],
        sharpe_gross=metrics['sharpe_gross']
    )

    # 用方向修正后的 Sharpe 计算 Fitness
    fit = calculate_local_fitness(
        sharpe=direction['corrected_sharpe_net'],
        turnover=metrics['turnover'],
        max_corr=0.0,
        icir=abs(icir)  # ICIR 也用绝对值，因为方向已经修正
    )

    return {
        # 原始（未修正）
        "sharpe_net_raw":   metrics['sharpe_net'],
        "sharpe_gross_raw": metrics['sharpe_gross'],
        # 方向修正后（用于 Fitness）
        "sharpe_net":       direction['corrected_sharpe_net'],
        "sharpe_gross":     direction['corrected_sharpe_gross'],
        "needs_negation":   direction['needs_negation'],
        "direction_note":   direction['direction'],
        # 其他
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
    derived = []
    mkt_ret = LeafNode("mkt_ret_20d", f"Ref({MKT_RET_EXPR}, 1)", DimType.RATIO)
    zero    = LeafNode("zero", "0", DimType.RATIO)
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


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"⚡ 快速 A/B 对照实验：7 模块流水线 + 方向自动修正（深度 {TARGET_DEPTH}）")
    print("=" * 70)

    cur = os.path.dirname(os.path.abspath(__file__))
    init_qlib_engine(os.path.join(cur, "data", "qlib_data", "us_data"))

    # ============================================================
    # Step 1：波束搜索
    # ============================================================
    print(f"\n🌱 [Step 1] 波束搜索生成深度-{TARGET_DEPTH} 因子...")
    engine = BeamSearchEngine(beam_width=BEAM_WIDTH)
    current_beam = engine.seed_nodes
    print(f"   起始种子: {len(current_beam)} 个")

    all_elite_nodes = {}

    for depth in range(2, TARGET_DEPTH + 1):
        print(f"\n   🧬 扩增至深度 {depth}...")
        candidates = engine.generate_candidates(current_beam, max_eval=MAX_EVAL)
        formulas = {n.name: n.expr_str for n in candidates}
        formulas["_vol_raw_"] = "$volume"

        df_batch = fetch_factor_data(formulas, POOL, START_DATE, END_DATE)
        ic_input = df_batch.drop(columns=["_vol_raw_"], errors="ignore")
        current_beam, ic_report = select_top_k_orthogonal(
            candidates, ic_input, k=BEAM_WIDTH, corr_threshold=CORR_THRESHOLD
        )

        for n in current_beam:
            all_elite_nodes[n.name] = n

        print(f"   深度 {depth} 留存精英: {len(current_beam)} 个")

        # 输出 |Sharpe| > 阈值 的因子（含方向修正信息）
        print(f"\n   📋 深度 {depth} |Sharpe(净，已修正方向)| ≥ {SHARPE_PRINT_THRESHOLD} 扫描：")
        high_count = 0
        for node in current_beam:
            if node.name not in df_batch.columns:
                continue
            res = evaluate_one(df_batch, node.name)
            shp = res['sharpe_net']

            if abs(shp) >= SHARPE_PRINT_THRESHOLD:
                high_count += 1
                hw = " ⚠️高换手" if res['high_turnover'] else ""
                neg = "需取反" if res['needs_negation'] else "原方向"
                print(f"   🔥 [{high_count}] [{neg}]")
                print(f"      名称:   {node.name}")
                print(f"      深度:   {node.depth} | 节点数: {node.node_count}")
                print(f"      Sharpe(修正): {shp:+.3f} | 原始: {res['sharpe_net_raw']:+.3f}")
                print(f"      IC:     {res['ic']:+.4f}{hw}")
                print(f"      换手率: {res['turnover']:.4f}")
                print(f"      Fitness: {res['fitness']:+.4f}")
                print(f"      表达式: {node.expr_str}")
                print()

        if high_count == 0:
            print(f"   （本轮无 |Sharpe(修正)| ≥ {SHARPE_PRINT_THRESHOLD} 的因子）")

    print(f"\n✅ 最终深度-{TARGET_DEPTH} 精英因子: {len(current_beam)} 个")

    # ============================================================
    # Step 2：取最终评估期数据
    # ============================================================
    print(f"\n📥 [Step 2] 取最终评估期数据...")
    final_formulas = {n.name: n.expr_str for n in current_beam}
    final_formulas["_vol_raw_"] = "$volume"
    df_eval = fetch_factor_data(final_formulas, POOL, START_DATE, END_DATE)

    # ============================================================
    # Step 3：A 组评估
    # ============================================================
    print("\n" + "=" * 70)
    print("🅰️  组 A：基线评估（原始因子值，已自动方向修正）")
    print("=" * 70)
    a_results = {}
    for node in current_beam:
        if node.name not in df_eval.columns:
            continue
        a_results[node.name] = evaluate_one(df_eval, node.name)
        res = a_results[node.name]
        hw = " ⚠️高换手" if res['high_turnover'] else ""
        neg = "[需取反]" if res['needs_negation'] else "[原方向]"
        print(f"   {neg} {node.name[:38]:<38} "
              f"Sharpe(修正)={res['sharpe_net']:+.3f} | "
              f"IC={res['ic']:+.4f} | "
              f"Turn={res['turnover']:.3f}{hw}")

    # ============================================================
    # Step 4：B 组评估（精炼管道）
    # ============================================================
    print("\n" + "=" * 70)
    print("🅱️  组 B：完整 7 模块流水线（已自动方向修正）")
    print("=" * 70)

    factor_names = list(a_results.keys())
    elite_df = df_eval[factor_names + ["target_ret", "_vol_raw_"]].copy()

    pipeline = FactorPipeline(
        df=elite_df, factor_cols=factor_names, volume_col="_vol_raw_",
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
                d_df = fetch_factor_data(d_formulas, POOL, START_DATE, END_DATE)
                for n in derived:
                    if n.name in d_df.columns:
                        b_results[n.name] = evaluate_one(d_df, n.name)
                        all_elite_nodes[n.name] = n
                print(f"   衍生因子: {len(derived)} 个")
            except Exception as e:
                print(f"   ⚠️ 衍生因子失败: {e}")

    print("\n>> 模块六：低相关组合筛选...")
    sharpe_map = {fname: b_results[fname]['sharpe_net'] for fname in b_results}
    portfolio = pipeline.select_portfolio(
        sharpe_dict=sharpe_map, max_factors=5,
        min_fitness=0.0, max_avg_corr=0.3
    )

    # ============================================================
    # Step 5：A/B 对照表
    # ============================================================
    print("\n\n" + "=" * 120)
    print("📊 A/B 对照实验结果（按 B 组 |Sharpe(修正)| 降序排列）")
    print("=" * 120)

    common_names = [n for n in factor_names if n in b_results]
    common_names.sort(key=lambda x: abs(b_results[x]['sharpe_net']), reverse=True)

    n_improve = 0
    for i, fname in enumerate(common_names):
        a, b = a_results[fname], b_results[fname]
        d_shp = abs(b['sharpe_net']) - abs(a['sharpe_net'])  # 用绝对值比较，避免方向影响
        if d_shp > 0:
            n_improve += 1
        sym = "📈" if d_shp > 0 else "📉"

        hw_b = " ⚠️高换手" if b['high_turnover'] else ""
        hw_a = " ⚠️高换手" if a['high_turnover'] else ""
        neg = "🔄需取反" if b['needs_negation'] else "✓原方向"

        node = all_elite_nodes.get(fname)

        print(f"\n[{i+1}] {neg} ──────────────────────────────────────────────────────")
        print(f"  名称:     {fname}")
        if node:
            print(f"  表达式:   {node.expr_str}")
            print(f"  深度:     {node.depth} | 节点数: {node.node_count}")
        print(f"  A(基线):  Sharpe(修正)={a['sharpe_net']:+.3f} (原始 {a['sharpe_net_raw']:+.3f}) | "
              f"IC={a['ic']:+.4f} | 换手率={a['turnover']:.4f}{hw_a}")
        print(f"  B(精炼):  Sharpe(修正)={b['sharpe_net']:+.3f} (原始 {b['sharpe_net_raw']:+.3f}) | "
              f"IC={b['ic']:+.4f} | 换手率={b['turnover']:.4f}{hw_b}")
        print(f"           ICIR={b['icir']:+.4f} | Fitness={b['fitness']:+.4f}")
        print(f"  Δ|Sharpe|: {d_shp:+.3f} {sym}")
        print(f"  方向说明: {b['direction_note']}")

    print(f"\n{'='*120}")
    print(f"\n✅ 管道有效因子: {n_improve}/{len(common_names)} 个 |Sharpe| 提升")

    # ============================================================
    # Step 6：高 Sharpe 因子汇总（修正方向后）
    # ============================================================
    high_b = [(fname, b_results[fname]) for fname in common_names
              if abs(b_results[fname]['sharpe_net']) >= SHARPE_PRINT_THRESHOLD]

    if high_b:
        print(f"\n\n{'='*120}")
        print(f"🔥 精炼后 |Sharpe(修正)| ≥ {SHARPE_PRINT_THRESHOLD} 的因子汇总（共 {len(high_b)} 个）")
        print(f"{'='*120}")
        for rank, (fname, b) in enumerate(high_b, 1):
            a = a_results.get(fname, {})
            node = all_elite_nodes.get(fname)
            d_shp = abs(b['sharpe_net']) - abs(a.get('sharpe_net', 0))
            hw_b = " ⚠️高换手" if b['high_turnover'] else ""
            neg = "🔄 WQ 提交时需要取反整个表达式（前面加 -1 *）" if b['needs_negation'] \
                  else "✅ 直接提交，无需修改"

            print(f"\n  ━━━ [{rank}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  名称:     {fname}")
            if node:
                print(f"  表达式:   {node.expr_str}")
                print(f"  深度:     {node.depth} | 节点数: {node.node_count}")
            print(f"  ──── A 组（基线）────")
            print(f"  Sharpe(修正): {a.get('sharpe_net',0):+.3f}  "
                  f"(原始 {a.get('sharpe_net_raw',0):+.3f})")
            print(f"  IC: {a.get('ic',0):+.4f} | 换手率: {a.get('turnover',0):.4f}")
            print(f"  ──── B 组（精炼后）────")
            print(f"  Sharpe(修正): {b['sharpe_net']:+.3f}  "
                  f"(原始 {b['sharpe_net_raw']:+.3f}){hw_b}")
            print(f"  IC: {b['ic']:+.4f} | 换手率: {b['turnover']:.4f}")
            print(f"  ICIR: {b['icir']:+.4f} | Fitness: {b['fitness']:+.4f}")
            print(f"  ──── WQ 提交说明 ────")
            print(f"  {neg}")

        # WQ 提交建议
        print(f"\n\n{'='*120}")
        print(f"📤 WQ 提交建议")
        print(f"{'='*120}")
        for rank, (fname, b) in enumerate(high_b, 1):
            node = all_elite_nodes.get(fname)
            if not node:
                continue
            if b['needs_negation']:
                wq_expr = f"-1 * ({node.expr_str})"
            else:
                wq_expr = node.expr_str
            print(f"\n[{rank}] {fname}")
            print(f"   提交表达式: {wq_expr}")
            print(f"   预期 Sharpe(修正后): {b['sharpe_net']:+.3f} | Fitness: {b['fitness']:+.4f}")

    else:
        print(f"\n⚠️ 本次运行无 |Sharpe(修正)| ≥ {SHARPE_PRINT_THRESHOLD} 的因子")
        print(f"   建议增大 MAX_EVAL 或 BEAM_WIDTH 后重试")

    print(f"\n⏱️  总耗时: {time.time()-t0:.1f} 秒")
    print(f"💡 注：")
    print(f"   - Sharpe(修正) = 根据 IC 方向自动修正后的 Sharpe（正向选股能力）")
    print(f"   - 高换手因子（turnover>2.5）跳过成本扣除，避免成本模型失真")
    print(f"   - 标记「需取反」的因子，在 WQ 提交时要给整个表达式加 -1 *")
    print(f"   - Fitness 理想区间：>1.0 极品 | 0.5~1.0 优秀 | 0.2~0.5 良好 | <0.2 淘汰")


if __name__ == "__main__":
    main()