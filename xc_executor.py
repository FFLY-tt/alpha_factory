# xc_executor.py
"""
AlphaForge 执行器：XC 模式 (Cross Breed)
逻辑：真正的混合模因算法 (Memetic Algorithm)。
接收两个精英池，通过 AST 子树互换进行纯正的遗传规划 (GP) 杂交。
"""
import time
from data_pipeline.data_source import fetch_factor_data
from src.evaluation.metrics_calc import select_top_k_orthogonal, batch_evaluate_formulas


def run_xc_pipeline(
        engine, db, batch_id, pool_a, pool_b, target_depth,
        POOL, IS_START_DATE, IS_END_DATE, OOS_START_DATE, OOS_END_DATE,
        MAX_EVAL, BEAM_WIDTH, CORR_THRESHOLD, TURNOVER_CAP,
        FITNESS_MIN_OOS, SHARPE_MIN_OOS
):
    t0 = time.time()
    print("=" * 70)
    print(f"🧬 启动 AlphaForge XC (AST 遗传杂交) 流水线 (基因锁深度: {target_depth})")
    print(f"   Pool A 规模: {len(pool_a)} | Pool B 规模: {len(pool_b)}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1: 基因重组 (调用我们手搓的 AST 手术刀)
    # ---------------------------------------------------------
    print(f"\n💞 [Step 1] 执行跨池 AST 子树重组...")
    # 🎯 传入 target_depth 激活 T_max 基因锁
    candidates = engine.generate_crossover_candidates(pool_a, pool_b, max_eval=MAX_EVAL, target_depth=target_depth)

    if not candidates:
        print("❌ 未能生成合法的杂交后代，请检查父本池。")
        return {}

    formulas = {n.name: n.expr_str for n in candidates}
    formulas["_vol_raw_"] = "$volume"

    # 提取 Qlib 数据
    df_batch = fetch_factor_data(formulas, POOL, IS_START_DATE, IS_END_DATE)
    ic_input = df_batch.drop(columns=["_vol_raw_"], errors="ignore")

    # 正交优选，残酷的自然选择
    current_beam, _ = select_top_k_orthogonal(candidates, ic_input, k=BEAM_WIDTH, corr_threshold=CORR_THRESHOLD)
    print(f"   🏆 自然选择结束，杂交存活精英: {len(current_beam)} 个")

    # 实时落库
    formulas_current = {n.name: n.expr_str for n in current_beam}
    current_layer_results = batch_evaluate_formulas(formulas_current, POOL, IS_START_DATE, IS_END_DATE)
    db_layer_input = {n.expr_str: {"node": n, "is_res": current_layer_results.get(n.name, {})} for n in current_beam}
    db.save_factor_batch(batch_id, "XC_CROSSBREED", target_depth, db_layer_input,
                         variant_tag=f"depth_{target_depth}_xc_elite")

    # ---------------------------------------------------------
    # Step 2 & 3: OOS 样本外终极验证
    # ---------------------------------------------------------
    print(f"\n🔧 [Step 2] 对纯血杂交精英进行 OOS 验证（{OOS_START_DATE} ~ {OOS_END_DATE}）...")
    oos_formulas = {n.name: n.expr_str for n in current_beam}
    oos_results = batch_evaluate_formulas(oos_formulas, POOL, OOS_START_DATE, OOS_END_DATE)

    qualified_count = 0
    final_qualified = {}

    print(f"\n  {'杂交变异体 (AST)':<45} │ {'IS_Shp':>8} │ {'OOS_Shp':>8}  结果")
    print(f"  {'-' * 45}─┼─{'-' * 8}─┼─{'-' * 8}──────────")

    for node in current_beam:
        is_r = current_layer_results.get(node.name, {})
        oos_r = oos_results.get(node.name, {})
        if not oos_r or not is_r: continue

        # 判断 OOS 门槛
        oos_ok = (
                oos_r.get('fitness', -99) >= FITNESS_MIN_OOS and
                oos_r.get('turnover', 99) < TURNOVER_CAP and
                abs(oos_r.get('sharpe_net', 0)) >= SHARPE_MIN_OOS and
                (is_r.get('sharpe_net', 0) * oos_r.get('sharpe_net', 0)) > 0  # 方向一致
        )

        mark = "✅ 极品" if oos_ok else "❌ 淘汰"
        if oos_ok:
            qualified_count += 1
            final_qualified[node.name] = {"node": node, "is_res": is_r, "oos_res": oos_r}

        print(
            f"  {node.expr_str[:45]:<45} │ {is_r.get('sharpe_net', 0):>+8.3f} │ {oos_r.get('sharpe_net', 0):>+8.3f}  {mark}")

    print(f"\n🎉 杂交验证完毕！真正跨越周期的超级因子：{qualified_count} 个")
    return final_qualified