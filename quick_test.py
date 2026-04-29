# quick_test.py
"""
快速 A/B 对照实验：验证 7 模块流水线对深度-3 复杂因子的提升效果
(重构版：职责清晰的调度流水线)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import time
import pandas as pd
from data_pipeline.data_source import init_qlib_engine, fetch_factor_data
from src.mining.beam_search import BeamSearchEngine

from src.processing.factor_pipeline import FactorPipeline
from src.evaluation.metrics_calc import (
    select_top_k_orthogonal,
    evaluate_single_factor_comprehensive,
    batch_evaluate_formulas
)

# ==============================================================================
# 主流程 (高度结构化的调度流水线)
# ==============================================================================

# 【外科手术修改】：仅仅扩充了参数列表，直接接收大写变量，保证下方逻辑 0 改动
def run_alpha_pipeline(
    engine, db, batch_id, initial_beam, start_depth, target_depth,
    POOL, IS_START_DATE, IS_END_DATE, OOS_START_DATE, OOS_END_DATE,
    MAX_EVAL, BEAM_WIDTH, CORR_THRESHOLD, TURNOVER_CAP,
    FITNESS_MIN_OOS, SHARPE_MIN_OOS
):
    t0 = time.time()
    print("=" * 70)
    print(f"⚙️ 启动 AlphaForge 加工流水线 (Depth {start_depth} -> {target_depth}，换手上限{TURNOVER_CAP})")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1：IS 样本内波束搜索
    # ---------------------------------------------------------
    print(f"\n🌱 [Step 1] 波束搜索（IS: {IS_START_DATE} ~ {IS_END_DATE}）...")
    current_beam = initial_beam

    for depth in range(start_depth, target_depth + 1):
        print(f"\n   🧬 深度 {depth}...")
        candidates = engine.generate_candidates(current_beam, max_eval=MAX_EVAL)
        formulas = {n.name: n.expr_str for n in candidates}
        formulas["_vol_raw_"] = "$volume"

        df_batch = fetch_factor_data(formulas, POOL, IS_START_DATE, IS_END_DATE)
        ic_input = df_batch.drop(columns=["_vol_raw_"], errors="ignore")

        current_beam, _ = select_top_k_orthogonal(
            candidates, ic_input, k=BEAM_WIDTH, corr_threshold=CORR_THRESHOLD
        )
        print(f"   留存精英: {len(current_beam)} 个")
        # 🎯 ================= 【新增：每层实时落库】 ================= 🎯
        print(f"   💾 正在将 Depth {depth} 的 {len(current_beam)} 个精英实时落库...")
        formulas_current = {n.name: n.expr_str for n in current_beam}
        current_layer_results = batch_evaluate_formulas(formulas_current, POOL, IS_START_DATE, IS_END_DATE)

        db_layer_input = {}
        for node in current_beam:
            if node.name in current_layer_results:
                db_layer_input[node.expr_str] = {
                    "node": node,
                    "is_res": current_layer_results[node.name]
                }

        # 写入数据库，打上该层的专属标签
        db.save_factor_batch(batch_id, "XA_MINING", depth, db_layer_input, variant_tag=f"depth_{depth}_elite")
        # 🎯 ======================================================== 🎯

    print(f"\n✅ 挖掘完毕，获得精英因子: {len(current_beam)} 个")

    # ---------------------------------------------------------
    # Step 2：A 组评估（基准）
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("🅰️  [Step 2] 组 A：基线评估（IS 期）")

    formulas_a = {n.name: n.expr_str for n in current_beam}
    a_results = batch_evaluate_formulas(formulas_a, POOL, IS_START_DATE, IS_END_DATE)

    for fname, res in a_results.items():
        hw = " ⚠️高换手" if res['high_turnover'] else ""
        print(f"   {fname[:40]:<40} Shp={res['sharpe_net']:+.3f} | Turn={res['turnover']:.3f}{hw}")

    # ---------------------------------------------------------
    # Step 3：B 组评估（精炼管道）
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("🅱️  [Step 3] 组 B：精炼管道清洗（IS 期）")

    # 重新取数给 pipeline
    fetch_dict = formulas_a.copy()
    fetch_dict["_vol_raw_"] = "$volume"  # 【核心修复】：告诉 Qlib 顺手把成交量也取出来！

    df_eval = fetch_factor_data(fetch_dict, POOL, IS_START_DATE, IS_END_DATE)
    factor_names = list(a_results.keys())

    # 容错提取：只提取 df_eval 中实际存在的列，防止 KeyError
    cols_to_extract = factor_names + ["target_ret", "_vol_raw_"]
    available_cols = [c for c in cols_to_extract if c in df_eval.columns]
    elite_df = df_eval[available_cols].copy()

    pipeline = FactorPipeline(df=elite_df, factor_cols=factor_names, volume_col="_vol_raw_")
    refined_df = pipeline.run(
        enable_liquidity_filter=True,
        enable_robustify=True,
        enable_neutralization=False,
    )

    b_results = {}
    for fname in factor_names:
        if fname in refined_df.columns:
            b_results[fname] = evaluate_single_factor_comprehensive(refined_df, fname)

    # ---------------------------------------------------------
    # Step 4：定向精调 (调用 engine 的新能力)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"🔧 [Step 4] 自动精调与变异竞争（IS 期）")

    # 4.1 生成变体：平滑版 + 状态择时版
    smooth_variants = engine.generate_smoothing_variants(current_beam)
    state_variants = engine.generate_market_state_variants(current_beam)
    all_variants = smooth_variants + state_variants

    # 4.2 取数并评估所有变体
    formulas_variants = {n.name: n.expr_str for n in all_variants}
    variant_results = batch_evaluate_formulas(formulas_variants, POOL, IS_START_DATE, IS_END_DATE)

    # 4.3 选拔最优版本
    best_is_versions = {}  # base_name -> {node, res}

    for base_node in current_beam:
        base_name = base_node.name
        best_res = b_results.get(base_name, {})
        best_node = base_node
        best_fit = best_res.get('fitness', -999) if best_res.get('turnover', 999) < TURNOVER_CAP else -999

        # 寻找该因子的所有变体（通过名称匹配，或者更好的方式是通过血统字段）
        my_variants = [n for n in all_variants if base_name[:20] in n.name]

        for v_node in my_variants:
            v_res = variant_results.get(v_node.name)
            if not v_res: continue

            if v_res['turnover'] < TURNOVER_CAP and v_res['fitness'] > best_fit:
                best_fit = v_res['fitness']
                best_res = v_res
                best_node = v_node

        best_is_versions[base_name] = {"node": best_node, "res": best_res}

        mark = "⭐精调" if best_node != base_node else "原始"
        print(f"  [{base_name[:30]:<30}] 胜出: {best_node.name[:30]:<30} | Fit: {best_fit:+.4f} ({mark})")

    # ---------------------------------------------------------
    # Step 5：OOS 样本外终极验证
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"🔭 [Step 5] OOS 样本外验证（{OOS_START_DATE} ~ {OOS_END_DATE}）")

    # 收集 IS 赢家的公式
    oos_formulas = {info["node"].name: info["node"].expr_str for info in best_is_versions.values()}
    oos_results = batch_evaluate_formulas(oos_formulas, POOL, OOS_START_DATE, OOS_END_DATE)

    print(f"\n  {'因子名':<40} │ {'IS_Shp':>8} {'IS_Fit':>7} │ {'OOS_Shp':>8} {'OOS_Fit':>7}  结果")
    print(f"  {'-' * 40}─┼─{'-' * 8}─{'-' * 7}─┼─{'-' * 8}─{'-' * 7}──────────")

    qualified_count = 0
    final_qualified = {}

    for base_name, info in best_is_versions.items():
        v_name = info["node"].name
        is_r = info["res"]
        oos_r = oos_results.get(v_name)

        if not oos_r: continue

        # 判断 OOS 门槛
        oos_ok = (
                oos_r['fitness'] >= FITNESS_MIN_OOS and
                oos_r['turnover'] < TURNOVER_CAP and
                abs(oos_r['sharpe_net']) >= SHARPE_MIN_OOS and
                (is_r['sharpe_net'] * oos_r['sharpe_net']) > 0  # 方向一致
        )

        if oos_ok:
            qualified_count += 1
            mark = "✅ 稳健"
        else:
            mark = "❌ 淘汰"
            # 将合格的因子信息打包
        final_qualified[v_name] = {
            "node": info["node"],
            "is_res": is_r,
            "oos_res": oos_r
        }

        print(f"  {v_name[:40]:<40} │ {is_r['sharpe_net']:>+8.3f} {is_r['fitness']:>+7.4f} │ "
              f"{oos_r['sharpe_net']:>+8.3f} {oos_r['fitness']:>+7.4f}  {mark}")

    print(f"\n🎉 验证完毕！合格因子数：{qualified_count} / {len(best_is_versions)}")
    print(f"⏱️ 流水线执行耗时: {time.time() - t0:.1f} 秒")

    return final_qualified