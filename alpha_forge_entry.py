# alpha_forge_entry.py
import time
import os
from enum import Enum
from src.db.clickhouse_manager import ClickHouseManager
from src.mining.beam_search import BeamSearchEngine

from xa_executor import run_xa_pipeline
from xb_executor import run_xb_pipeline
from xc_executor import run_xc_pipeline
from xd_executor import run_xd_pipeline

from data_pipeline.data_source import init_qlib_engine  # 【新增导入】


class RunMode(Enum):
    XA = "XA_AUTO_PILOT"  # 自动拓荒：从 0 开始挖到目标深度
    XB = "XB_STEP_FORWARD"  # 单步推进：拿特定批次的某一层，往下挖一层
    XC = "XC_CROSS_BREED"  # 跨代杂交：融合两个不同批次/层级的基因 (预留)
    XD = "XD_HORIZON_EXHAUST"  # 水平穷举：停留在某层全量计算 (预留)


# ==============================================================================
# 调度器配置 (统一配置中心)
# ==============================================================================
CURRENT_MODE = RunMode.XA  # 在这里切换模式！

# --- 挖矿与回测配置 ---
POOL = "sp500"
IS_START_DATE = "2010-01-04"
IS_END_DATE = "2017-12-31"
OOS_START_DATE = "2018-01-01"
OOS_END_DATE = "2020-11-10"

TARGET_DEPTH = 3
BEAM_WIDTH = 10
CORR_THRESHOLD = 0.6
MAX_EVAL = 10  # ✅ 算力限制参数已归位

# --- 门槛配置 ---
TURNOVER_CAP = 1.5
FITNESS_MIN_IS = 0.5
SHARPE_MIN_IS = 1.0
FITNESS_MIN_OOS = 0.3
SHARPE_MIN_OOS = 0.8

# --- XB/XC 模式专用配置 ---
RESUME_BATCH_ID = 123456  # 从数据库查出来的 Batch ID
RESUME_DEPTH = 3  # 你想基于哪一层的精英往下挖？


def main():
    print("=" * 70)
    print("🧠 AlphaForge 终极调度中枢启动")
    print("=" * 70)

    # 【外科手术修改】：把 Qlib 挂载的职责还给系统入口
    cur = os.path.dirname(os.path.abspath(__file__))
    init_qlib_engine(os.path.join(cur, "data", "qlib_data", "us_data"))

    db = ClickHouseManager()
    engine = BeamSearchEngine(beam_width=BEAM_WIDTH)

    current_batch_id = int(time.time() * 1000) % 1000000

    if CURRENT_MODE == RunMode.XA:
        print(f"🚀 [模式 XA] 全新拓荒启动 -> Batch ID: {current_batch_id}")
        start_depth = 2
        initial_parents = engine.seed_nodes

    elif CURRENT_MODE == RunMode.XB:
        print(f"🔄 [模式 XB] 断点续跑启动 -> 将继承 Batch {RESUME_BATCH_ID} 的 Depth {RESUME_DEPTH}")
        start_depth = RESUME_DEPTH + 1
        target_depth_run = start_depth  # XB 通常只向下挖一层
        initial_parents = db.get_elite_factors(RESUME_BATCH_ID, RESUME_DEPTH, limit=BEAM_WIDTH)

        if not initial_parents:
            print("❌ 数据库中未找到指定父本，请检查 RESUME 配置。")
            return

    # 🎯【新增】：XC 模式数据初始化
    elif CURRENT_MODE == RunMode.XC:
        print(f"💞 [模式 XC] 跨代杂交启动 -> 将提取 Batch {RESUME_BATCH_ID} 的 Depth {RESUME_DEPTH} 作为双亲库")
        start_depth = RESUME_DEPTH + 1
        target_depth_run = start_depth
        # XC 模式需要提取更多的父本以划分为两个池子
        initial_parents = db.get_elite_factors(RESUME_BATCH_ID, RESUME_DEPTH, limit=BEAM_WIDTH * 2)

        if len(initial_parents) < 2:
            print("❌ 数据库中未找到足够父本进行杂交，请检查 RESUME 配置。")
            return

    # 🎯【新增】：XD 模式数据初始化
    elif CURRENT_MODE == RunMode.XD:
        print(f"🔥 [模式 XD] 水平穷举启动 -> 将停留在 Depth {RESUME_DEPTH + 1} 疯狂横扫")
        start_depth = RESUME_DEPTH + 1
        target_depth_run = start_depth
        initial_parents = db.get_elite_factors(RESUME_BATCH_ID, RESUME_DEPTH, limit=BEAM_WIDTH)

        if not initial_parents:
            print("❌ 数据库中未找到指定父本，请检查 RESUME 配置。")
            return

    else:
        raise ValueError("未知的运行模式或执行器未就绪")

    # =========================================================
    # 路由分发 (将弹药移交对应的 Executor)
    # =========================================================
    print(f"\n📦 将 {len(initial_parents)} 个父本移交给 {CURRENT_MODE.name} 执行器...")

    qualified_factors = {}

    if CURRENT_MODE == RunMode.XA:
        # 🎯 【新增】：记录 T1 种子血统起点 (仅在 XA 拓荒模式下需要)
        print("开始进行 XA 模式执行器...")
        print(f"📝 正在初始化 Batch {current_batch_id} 的原始血统档案...")
        db.save_initial_seeds(current_batch_id, engine.seed_nodes)

        qualified_factors = run_xa_pipeline(
            engine=engine,
            db=db,
            batch_id=current_batch_id,
            initial_beam=initial_parents,
            start_depth=start_depth,
            target_depth=TARGET_DEPTH,  # XA 模式跑向 TARGET_DEPTH
            POOL=POOL,
            IS_START_DATE=IS_START_DATE,
            IS_END_DATE=IS_END_DATE,
            OOS_START_DATE=OOS_START_DATE,
            OOS_END_DATE=OOS_END_DATE,
            MAX_EVAL=MAX_EVAL,
            BEAM_WIDTH=BEAM_WIDTH,
            CORR_THRESHOLD=CORR_THRESHOLD,
            TURNOVER_CAP=TURNOVER_CAP,
            FITNESS_MIN_OOS=FITNESS_MIN_OOS,
            SHARPE_MIN_OOS=SHARPE_MIN_OOS
        )

    elif CURRENT_MODE == RunMode.XB:
        # XB 执行器调用
        print("开始进行 XB 模式执行器...")
        qualified_factors = run_xb_pipeline(
            engine=engine, db=db, batch_id=current_batch_id, initial_beam=initial_parents,
            start_depth=start_depth, target_depth=target_depth_run,
            POOL=POOL, IS_START_DATE=IS_START_DATE, IS_END_DATE=IS_END_DATE,
            OOS_START_DATE=OOS_START_DATE, OOS_END_DATE=OOS_END_DATE,
            MAX_EVAL=MAX_EVAL, BEAM_WIDTH=BEAM_WIDTH, CORR_THRESHOLD=CORR_THRESHOLD,
            TURNOVER_CAP=TURNOVER_CAP, FITNESS_MIN_OOS=FITNESS_MIN_OOS, SHARPE_MIN_OOS=SHARPE_MIN_OOS
        )

    # 🎯【新增】：XC 执行器调用
    elif CURRENT_MODE == RunMode.XC:
        print("开始进行 XC 模式执行器...")
        # 将 initial_parents 劈成两半，作为池 A 和池 B
        mid = len(initial_parents) // 2
        pool_a = initial_parents[:mid]
        pool_b = initial_parents[mid:]

        qualified_factors = run_xc_pipeline(
            engine=engine, db=db, batch_id=current_batch_id, pool_a=pool_a, pool_b=pool_b,
            target_depth=target_depth_run,
            POOL=POOL, IS_START_DATE=IS_START_DATE, IS_END_DATE=IS_END_DATE,
            OOS_START_DATE=OOS_START_DATE, OOS_END_DATE=OOS_END_DATE,
            MAX_EVAL=MAX_EVAL, BEAM_WIDTH=BEAM_WIDTH, CORR_THRESHOLD=CORR_THRESHOLD,
            TURNOVER_CAP=TURNOVER_CAP, FITNESS_MIN_OOS=FITNESS_MIN_OOS, SHARPE_MIN_OOS=SHARPE_MIN_OOS
        )

    # 🎯【新增】：XD 执行器调用
    elif CURRENT_MODE == RunMode.XD:
        print("开始进行 XD 模式执行器 (无限穷举)...")
        # XD 模式内部是死循环，不断产生 Chunk 并实时入库，无需返回 qualified_factors。
        # 除非按 Ctrl+C 中断，否则不会跳出。
        run_xd_pipeline(
            engine=engine, db=db, batch_id=RESUME_BATCH_ID, initial_beam=initial_parents,
            target_depth=target_depth_run,
            POOL=POOL, IS_START_DATE=IS_START_DATE, IS_END_DATE=IS_END_DATE,
            MAX_EVAL=MAX_EVAL, BEAM_WIDTH=BEAM_WIDTH, CORR_THRESHOLD=CORR_THRESHOLD
        )
        return  # 直接阻断，跳过末尾的统一结果持久化

    # =========================================================
    # 结果极简入库
    # =========================================================
    if not qualified_factors:
        print("\n⚠️ 本次执行未产出合格因子，或执行器未返回最终集。")
        return

    print(f"\n💾 准备将 {len(qualified_factors)} 个终极极品因子存入 ClickHouse...")

    db_input = {info["node"].expr_str: info for info in qualified_factors.values()}

    db.save_factor_batch(
        batch_id=current_batch_id,
        run_mode=CURRENT_MODE.name,
        depth=TARGET_DEPTH if CURRENT_MODE == RunMode.XA else start_depth,
        evaluation_results=db_input,
        variant_tag='fine_tuned_best'
    )

    print("✅ 批次任务圆满结束！")


if __name__ == "__main__":
    main()