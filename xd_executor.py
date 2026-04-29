# xd_executor.py
"""
AlphaForge 执行器：XD 模式 (Horizontal Exhaustion)
逻辑：停留在指定层级，分块循环计算，利用 DB 哈希去重，实现无损断点穷举。
"""
import time
from data_pipeline.data_source import fetch_factor_data
from src.evaluation.metrics_calc import select_top_k_orthogonal, batch_evaluate_formulas


def run_xd_pipeline(
        engine, db, batch_id, initial_beam, target_depth,
        POOL, IS_START_DATE, IS_END_DATE, MAX_EVAL, BEAM_WIDTH, CORR_THRESHOLD
):
    t0 = time.time()
    print("=" * 70)
    print(f"🔥 启动 AlphaForge XD 水平穷举流水线 (目标深度: {target_depth})")
    print("   [注意]: 该模式为无限循环，直到空间穷尽或按 Ctrl+C 中止，进度实时存盘。")
    print("=" * 70)

    chunk_idx = 0
    total_saved = 0

    while True:
        chunk_idx += 1
        print(f"\n🌀 [Chunk {chunk_idx}] 生成候选因子池...")

        # 1. 暴力生成
        candidates = engine.generate_candidates(initial_beam, max_eval=MAX_EVAL)

        # 2. 数据库哈希校验 (核心：幂等性去重)
        existing_hashes = db.get_existing_hashes(batch_id, target_depth)

        new_candidates = []
        for n in candidates:
            # 重新计算哈希以供比对
            h = int(__import__('hashlib').md5(n.expr_str.encode('utf-8')).hexdigest()[:15], 16)
            if h not in existing_hashes:
                new_candidates.append(n)

        print(
            f"   总生成: {len(candidates)} -> 剔除已算: {len(candidates) - len(new_candidates)} -> 待算: {len(new_candidates)}")
        if not new_candidates:
            print("   ✅ 当前算力切片未发现新基因，尝试下一次碰撞...")
            continue

        # 3. 评测计算
        formulas = {n.name: n.expr_str for n in new_candidates}
        df_batch = fetch_factor_data(formulas, POOL, IS_START_DATE, IS_END_DATE)
        if df_batch.empty: continue

        # 4. 选拔该 Chunk 精英并落库
        current_beam, _ = select_top_k_orthogonal(new_candidates, df_batch, k=BEAM_WIDTH, corr_threshold=CORR_THRESHOLD)
        formulas_current = {n.name: n.expr_str for n in current_beam}
        current_layer_results = batch_evaluate_formulas(formulas_current, POOL, IS_START_DATE, IS_END_DATE)

        db_layer_input = {n.expr_str: {"node": n, "is_res": current_layer_results.get(n.name, {})} for n in
                          current_beam}

        db.save_factor_batch(batch_id, "XD_EXHAUST", target_depth, db_layer_input, variant_tag=f"xd_chunk_{chunk_idx}")
        total_saved += len(db_layer_input)
        print(f"   💾 成功将本轮 {len(db_layer_input)} 个精英落盘。(总落盘: {total_saved})")

    return {}  # XD 模式是无限流水线，不返回最终结果，全部直接写库