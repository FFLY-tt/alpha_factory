# run_genetic.py
"""
遗传规划因子挖掘 —— 主入口脚本

流程：
  1. 初始化 Qlib + Dashboard
  2. 遗传进化（TRAIN 集搜索，VAL 集早停）
  3. 精炼：平滑变体 + 市场状态择时（IS 全段）
  4. OOS 终极验证
  5. 输出 WQ 提交建议

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【学术种子因子】在下方 ACADEMIC_SEED_FRAGMENTS 列表中填写。
  每个元素是一个合法的 Qlib 表达式字符串。
  示例：
      "($close / Mean($close, 20)) - 1",   # 价格偏离均线
      "Std($close / Ref($close, 1) - 1, 20)",  # 20 日波动率
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from data_pipeline.data_source import init_qlib_engine, fetch_factor_data
from src.evaluation.metrics_calc import (
    compute_factor_sharpe,
    calculate_ic_stability,
    auto_correct_direction,
)
from src.processing.factor_pipeline import FactorPipeline
from src.ast.ast_nodes import LeafNode
from src.rules.rule_engine import DimType
from src.genetic.genetic_engine import GeneticEngine, Individual, compute_genetic_fitness
from dashboard import Dashboard


# ══════════════════════════════════════════════════════════════════════════════
# ① 配置
# ══════════════════════════════════════════════════════════════════════════════
POOL = "sp500"

TRAIN_START = "2010-01-01"
TRAIN_END   = "2015-12-31"
VAL_START   = "2016-01-01"
VAL_END     = "2017-12-31"
IS_START    = "2010-01-01"   # IS = TRAIN ∪ VAL，精炼阶段使用
IS_END      = "2017-12-31"
OOS_START   = "2018-01-01"
OOS_END     = "2020-11-10"

# 遗传参数
POP_SIZE    = 120
ELITE_COUNT = 24
MAX_GENS    = 50
PATIENCE    = 5

# IS 精炼门槛
IS_SHARPE_MIN   = 0.6
IS_FITNESS_MIN  = 0.3
IS_TURNOVER_MAX = 1.5

# OOS 终极门槛
OOS_SHARPE_MIN   = 0.5
OOS_TURNOVER_MAX = 1.5

# 精炼：平滑变体
SMOOTH_VARIANTS = [
    ("ts_mean_5",       "Mean({expr}, 5)"),
    ("ts_mean_10",      "Mean({expr}, 10)"),
    ("decay_linear_5",  "WMA({expr}, 5)"),
    ("decay_linear_10", "WMA({expr}, 10)"),
    ("decay_linear_20", "WMA({expr}, 20)"),
]
TURNOVER_CAP_REFINE = 1.5   # 精炼平滑变体换手上限

# 市值代理（用于模块三中性化）
MKTCAP_PROXY_COL  = "_mktcap_proxy_"
MKTCAP_PROXY_EXPR = "Mean($close * $volume, 20)"

# 市场择时表达式
MKT_RET_EXPR = "Mean($close / Ref($close, 1) - 1, 20)"

# Dashboard 端口
DASHBOARD_PORT = 8050
ENABLE_DASHBOARD = True

# WQ 自动提交开关（需要根目录有 .env 文件含 WQ_USERNAME / WQ_PASSWORD）
ENABLE_WQ_SUBMIT = False   # 改为 True 开启自动提交

# 精炼后取 Top N 进行 OOS 验证
TOP_N_REFINE = 20


# ══════════════════════════════════════════════════════════════════════════════
# ② 学术种子因子（在此填写）
# ══════════════════════════════════════════════════════════════════════════════
ACADEMIC_SEED_FRAGMENTS = [
    # ── 在下方添加你的学术因子表达式 ──────────────────────────────────────
    # 每行一个，字符串格式，使用 Qlib 表达式语法
    # 可用变量：$close, $open, $high, $low, $volume
    # 可用算子：Mean, Std, WMA, Max, Min, Ref, Corr, Cov, Abs, Log, Sign, Rank

    # 示例（取消注释后生效）：
    # "($close / Mean($close, 20)) - 1",                          # 价格偏离20日均线
    # "Std($close / Ref($close, 1) - 1, 20)",                     # 20日历史波动率
    # "($close - Mean($close, 5)) / Std($close, 20)",             # Z-score
    # "($high - $low) / Mean(($high - $low), 10)",                # 相对振幅
    # "Corr($close, $volume, 10)",                                 # 价量10日相关
    # "(WMA($close, 5) - WMA($close, 20)) / WMA($close, 20)",     # 双均线偏离
    # "Ref($close, 5) / Mean($close, 5) - 1",                    # 5日前相对均线
    # "$close / Ref($close, 252) - 1",                            # 12月动量
]


# ══════════════════════════════════════════════════════════════════════════════
# ③ 辅助函数：target_alpha 计算
# ══════════════════════════════════════════════════════════════════════════════
def make_target_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    已切换为直接使用原始 target_ret，对齐 WQ 评估口径。
    WQ 评估原始绝对收益 Sharpe，不做大盘剥离。
    """
    return df   # 直接返回，不做任何处理


# ══════════════════════════════════════════════════════════════════════════════
# ④ 评估单因子（IS 全段，含 target_alpha）
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_one(df: pd.DataFrame, factor_name: str, depth: int = 0, nodes: int = 1) -> dict:
    """在已含 target_alpha 的 df 上评估单因子，返回指标 dict"""
    metrics   = compute_factor_sharpe(df, factor_name)
    mean_ic, icir = calculate_ic_stability(df, factor_name)
    direction = auto_correct_direction(mean_ic, metrics['sharpe_net'], metrics['sharpe_gross'])

    from src.genetic.diversity_utils import compute_ic_sign_consistency
    ic_consistency = compute_ic_sign_consistency(df, factor_name)

    fit = compute_genetic_fitness(
        alpha_sharpe   = direction['corrected_sharpe_net'],
        depth          = depth,
        nodes          = nodes,
        turnover       = metrics['turnover'],
        icir           = icir,
        ic_consistency = ic_consistency,
    )

    return {
        **metrics,
        **fit,
        "ic"            : mean_ic,
        "icir"          : icir,
        "alpha_sharpe"  : direction['corrected_sharpe_net'],
        "needs_negation": direction['needs_negation'],
        "direction_note": direction['direction'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 精炼：平滑变体 + 市场择时
# ══════════════════════════════════════════════════════════════════════════════
def refine_candidates(top_individuals: list) -> list:
    """
    对 Top N 个体进行精炼：
    1. 流水线（流动性过滤 + 鲁棒化 + 市值中性化）
    2. 平滑变体批量对比（一次取数）
    3. 市场择时变体（牛市/高波动）
    返回 refined_list: [{
        'name', 'original_expr', 'best_expr', 'best_label',
        'is_res', 'node', 'ind'
    }]
    """
    print(f"\n{'='*60}")
    print(f"🔧 精炼阶段（IS: {IS_START} ~ {IS_END}）")
    print(f"{'='*60}")

    # ── Step 1：取 IS 全段数据（含市值代理）──────────────────────────────
    print("\n  [精炼-1] 取 IS 全段数据 + 市值代理...")
    base_formulas = {"_vol_raw_": "$volume", MKTCAP_PROXY_COL: MKTCAP_PROXY_EXPR}
    for ind in top_individuals:
        base_formulas[ind.col_name] = ind.expr_str

    df_is = fetch_factor_data(base_formulas, POOL, IS_START, IS_END)
    df_is = make_target_alpha(df_is)

    # ── Step 2：流水线精炼（Winsorize + Rank + 中性化）──────────────────
    print("  [精炼-2] 7 模块精炼管道（含市值中性化）...")
    factor_names = [ind.col_name for ind in top_individuals if ind.col_name in df_is.columns]
    elite_df = df_is[factor_names + ["target_ret", "_vol_raw_", MKTCAP_PROXY_COL]].copy()
    pipeline = FactorPipeline(
        df=elite_df,
        factor_cols=factor_names,
        volume_col="_vol_raw_",
        market_cap_col=MKTCAP_PROXY_COL,
    )
    refined_df = pipeline.run(
        enable_liquidity_filter=True,
        enable_robustify=True,
        enable_neutralization=False,   # 关闭：WQ 提交后自动做行业中性化，本地不做避免引入额外变换
    )

    b_results = {}
    for fname in factor_names:
        if fname in refined_df.columns:
            ind = next((i for i in top_individuals if i.col_name == fname), None)
            b_results[fname] = evaluate_one(
                refined_df, fname,
                depth=ind.depth if ind else 0,
                nodes=ind.node_count if ind else 1,
            )

    # ── Step 3：批量生成平滑变体 + 市场择时（一次取数）────────────────────
    print("  [精炼-3] 批量平滑变体 + 市场择时...")
    variant_index  = {}   # col -> (col_name原, label, smoothed_expr)
    batch_formulas = {"_vol_raw_": "$volume"}

    for i, ind in enumerate(top_individuals):
        if ind.col_name not in df_is.columns:
            continue
        orig_expr = ind.expr_str

        # 平滑变体
        for j, (label, tmpl) in enumerate(SMOOTH_VARIANTS):
            col   = f"_sm_{i}_{j}_"
            expr  = tmpl.replace("{expr}", orig_expr)
            batch_formulas[col] = expr
            variant_index[col]  = (ind.col_name, label, expr)

        # 牛市择时
        col_bull = f"_bull_{i}_"
        expr_bull = f"If(Ref({MKT_RET_EXPR}, 1) > 0, {orig_expr}, 0)"
        batch_formulas[col_bull] = expr_bull
        variant_index[col_bull]  = (ind.col_name, "bull_timing", expr_bull)

    df_variants = fetch_factor_data(batch_formulas, POOL, IS_START, IS_END)
    df_variants = make_target_alpha(df_variants)

    # ── Step 4：逐个体选最优变体 ──────────────────────────────────────────
    result_list = []
    for i, ind in enumerate(top_individuals):
        if ind.col_name not in b_results:
            continue

        b_res    = b_results[ind.col_name]
        b_fit    = b_res.get("fitness", -999)
        b_turn   = b_res.get("turnover", 999)
        best_fit = b_fit  if b_turn < TURNOVER_CAP_REFINE else -999
        best_lbl = "原始B组" if b_turn < TURNOVER_CAP_REFINE else "(超限)"
        best_res = b_res  if b_turn < TURNOVER_CAP_REFINE else {}
        best_expr= ind.expr_str

        for col, (orig_col, label, expr) in variant_index.items():
            if orig_col != ind.col_name:
                continue
            if col not in df_variants.columns:
                continue
            res  = evaluate_one(df_variants, col, ind.depth, ind.node_count)
            turn = res.get("turnover", 999)
            fit  = res.get("fitness", -999)
            if turn < TURNOVER_CAP_REFINE and fit > best_fit:
                best_fit  = fit
                best_lbl  = label
                best_res  = res
                best_expr = expr

        result_list.append({
            "name"         : ind.col_name,
            "original_expr": ind.expr_str,
            "best_expr"    : best_expr,
            "best_label"   : best_lbl,
            "is_res"       : best_res,
            "node"         : ind.node,
            "ind"          : ind,
        })

    # 打印 IS 汇总
    print(f"\n  {'因子名':<30} {'IS Fit':>9} {'IS Shp':>9} {'换手':>8}  {'最优方案'}")
    print(f"  {'-'*65}")
    for r in sorted(result_list, key=lambda x: x['is_res'].get('fitness', 0), reverse=True):
        is_r = r['is_res']
        print(f"  {r['name'][:30]:<30} {is_r.get('fitness',0):>+9.4f} "
              f"{is_r.get('alpha_sharpe',0):>+9.4f} "
              f"{is_r.get('turnover',0):>8.4f}  {r['best_label']}")

    return result_list


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ OOS 终极验证
# ══════════════════════════════════════════════════════════════════════════════
def oos_validate(refined_list: list) -> list:
    """
    在 OOS 期（2018-2020）验证精炼后的最优因子。
    门槛：OOS Sharpe ≥ 0.5 且换手 < 1.5 且 IC 方向与 IS 一致。
    返回通过 OOS 的 result 列表（新增 'oos_res' 字段）。
    """
    print(f"\n{'='*60}")
    print(f"🔭 OOS 验证（{OOS_START} ~ {OOS_END}）")
    print(f"{'='*60}")

    # 筛选 IS 通过的候选
    is_passed = [r for r in refined_list
                 if (r['is_res'].get('fitness', 0) >= IS_FITNESS_MIN
                     and r['is_res'].get('turnover', 999) < IS_TURNOVER_MAX
                     and abs(r['is_res'].get('alpha_sharpe', 0)) >= IS_SHARPE_MIN)]

    print(f"\n  IS 通过门槛（Fit≥{IS_FITNESS_MIN} | Shp≥{IS_SHARPE_MIN} | Turn<{IS_TURNOVER_MAX}）: "
          f"{len(is_passed)}/{len(refined_list)} 个")

    if not is_passed:
        print("  ⚠️ 无因子通过 IS 门槛，跳过 OOS 验证")
        return []

    # 批量取 OOS 数据
    oos_formulas = {"_vol_raw_": "$volume"}
    oos_col_map  = {}
    for r in is_passed:
        col = f"_oos_{r['name']}_"[:55]
        oos_formulas[col] = r['best_expr']
        oos_col_map[col]  = r['name']

    print(f"  取数：{len(oos_col_map)} 个因子...")
    df_oos = fetch_factor_data(oos_formulas, POOL, OOS_START, OOS_END)
    df_oos = make_target_alpha(df_oos)

    # 逐因子 OOS 评估
    name_to_result = {r['name']: r for r in is_passed}
    oos_robust = []

    print(f"\n  {'因子名':<30} │ {'IS Shp':>8} {'IS Fit':>8} │ "
          f"{'OOS Shp':>8} {'OOS Fit':>8} │ {'换手':>7}  判断")
    print(f"  {'-'*80}")

    for col, fname in oos_col_map.items():
        r     = name_to_result[fname]
        is_r  = r['is_res']
        node  = r.get('node') or r.get('ind').node
        depth = node.depth if node else 0
        nodes_n = node.node_count if node else 1

        if col not in df_oos.columns:
            print(f"  {fname[:30]:<30} │ {'取数失败':>48} │")
            continue

        oos_r = evaluate_one(df_oos, col, depth, nodes_n)
        r['oos_res'] = oos_r

        is_shp  = is_r.get('alpha_sharpe', 0)
        oos_shp = oos_r.get('alpha_sharpe', 0)
        oos_fit = oos_r.get('fitness', 0)
        oos_turn = oos_r.get('turnover', 999)

        # OOS 四项门槛
        c_shp  = abs(oos_shp) >= OOS_SHARPE_MIN
        c_turn = oos_turn     <  OOS_TURNOVER_MAX
        c_dir  = (is_shp * oos_shp) > 0

        robust = c_shp and c_turn and c_dir
        flags  = []
        if not c_shp:  flags.append(f"Shp<{OOS_SHARPE_MIN}")
        if not c_turn: flags.append(f"Turn≥{OOS_TURNOVER_MAX}")
        if not c_dir:  flags.append("方向反转⚠️")
        verdict = "✅稳健" if robust else ("❌ " + " | ".join(flags))

        print(f"  {fname[:30]:<30} │ {is_shp:>+8.3f} {is_r.get('fitness',0):>+8.4f} │ "
              f"{oos_shp:>+8.3f} {oos_fit:>+8.4f} │ "
              f"{oos_turn:>7.4f}  {verdict}")

        if robust:
            oos_robust.append(r)

    print(f"\n  ✅ OOS 稳健因子: {len(oos_robust)}/{len(is_passed)} 个")
    return oos_robust


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ 输出 WQ 提交建议
# ══════════════════════════════════════════════════════════════════════════════
def output_recommendations(oos_robust: list, is_only: list = None) -> None:
    """打印最终 WQ 提交建议"""
    print(f"\n{'='*80}")
    print(f"📤 WQ 提交建议（IS+OOS 双门槛，共 {len(oos_robust)} 个）")
    print(f"{'='*80}")

    if not oos_robust:
        print("\n  ⚠️ 无因子同时满足 IS + OOS 双门槛。")
        print("  建议：增加 MAX_GENS / POP_SIZE，或在 ACADEMIC_SEED_FRAGMENTS 添加更多学术因子。")
    else:
        for rank, r in enumerate(
                sorted(oos_robust, key=lambda x: x['oos_res'].get('fitness', 0), reverse=True), 1
        ):
            is_r  = r['is_res']
            oos_r = r['oos_res']
            expr  = r['best_expr']
            needs_neg = is_r.get('needs_negation', False)
            wq_expr = f"-1 * ({expr})" if needs_neg else expr

            print(f"\n  [{rank}] ─────────────────────────────────────────────────────")
            print(f"  原始表达式: {r['original_expr']}")
            print(f"  使用方案:   {r['best_label']}")
            print(f"  提交表达式: {wq_expr}")
            print(f"  IS  → Shp:{is_r.get('alpha_sharpe',0):+.3f} | "
                  f"换手:{is_r.get('turnover',0):.4f} | "
                  f"Fitness:{is_r.get('fitness',0):+.4f} | "
                  f"ICIR:{is_r.get('icir',0):+.4f}")
            print(f"  OOS → Shp:{oos_r.get('alpha_sharpe',0):+.3f} | "
                  f"换手:{oos_r.get('turnover',0):.4f} | "
                  f"Fitness:{oos_r.get('fitness',0):+.4f} | "
                  f"ICIR:{oos_r.get('icir',0):+.4f}")

    # 仅 IS 通过的参考清单
    if is_only:
        print(f"\n  📋 仅 IS 通过（OOS 未达标，供参考）：")
        for r in sorted(is_only, key=lambda x: x['is_res'].get('fitness', 0), reverse=True):
            is_r = r['is_res']
            oos_r = r.get('oos_res', {})
            print(f"     {r['name'][:40]:<40} IS_Fit:{is_r.get('fitness',0):+.4f} | "
                  f"OOS_Shp:{oos_r.get('alpha_sharpe',0):+.3f}")

    print(f"\n💡 提交 WQ WebSim 说明：")
    print(f"   · 若仿真 Sharpe ≥ 1.25，可直接提交正式")
    print(f"   · 标注「需取反」的因子，WQ 中给表达式前加 -1 *")
    print(f"   · IS: {IS_START}~{IS_END} | OOS: {OOS_START}~{OOS_END}")


# ══════════════════════════════════════════════════════════════════════════════
# ⑧ 主函数
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    # ── 初始化 Qlib ────────────────────────────────────────────────────────
    cur = os.path.dirname(os.path.abspath(__file__))
    init_qlib_engine(os.path.join(cur, "data", "qlib_data", "us_data"))

    # ── 启动 Dashboard ──────────────────────────────────────────────────────
    dashboard = None
    if ENABLE_DASHBOARD:
        try:
            import dash
            dashboard = Dashboard(port=DASHBOARD_PORT)
            dashboard.start()
        except ImportError:
            print("⚠️ Dashboard 依赖未安装，跳过。请运行：")
            print("   pip install dash dash-bootstrap-components plotly")

    # ── 遗传进化 ───────────────────────────────────────────────────────────
    engine = GeneticEngine(
        pop_size    = POP_SIZE,
        elite_count = ELITE_COUNT,
        max_gens    = MAX_GENS,
        patience    = PATIENCE,
        pool        = POOL,
        train_start = TRAIN_START,
        train_end   = TRAIN_END,
        val_start   = VAL_START,
        val_end     = VAL_END,
        dashboard   = dashboard,
    )

    print("\n🌱 初始化种群...")
    engine.initialize(academic_fragments=ACADEMIC_SEED_FRAGMENTS)

    final_population = engine.run()

    # ── 精炼 ───────────────────────────────────────────────────────────────
    top_individuals = engine.get_top_n(n=TOP_N_REFINE)
    print(f"\n📌 取 Top {TOP_N_REFINE} 个体进入精炼阶段")
    for i, ind in enumerate(top_individuals, 1):
        print(f"   [{i:2d}] Fitness={ind.fitness:+.4f} | {ind.expr_str[:60]}")

    refined_list = refine_candidates(top_individuals)

    # ── OOS 验证 ──────────────────────────────────────────────────────────
    oos_robust = oos_validate(refined_list)

    # 仅 IS 通过的参考列表
    oos_names  = {r['name'] for r in oos_robust}
    is_only    = [r for r in refined_list
                  if r['name'] not in oos_names
                  and r['is_res'].get('fitness', 0) >= IS_FITNESS_MIN
                  and r['is_res'].get('turnover', 999) < IS_TURNOVER_MAX
                  and abs(r['is_res'].get('alpha_sharpe', 0)) >= IS_SHARPE_MIN
                  and 'oos_res' in r]

    # ── 输出建议 ──────────────────────────────────────────────────────────
    output_recommendations(oos_robust, is_only)

    # ── WQ 自动提交（可选）────────────────────────────────────────────────
    if ENABLE_WQ_SUBMIT and oos_robust:
        try:
            from wq_submit import submit_factors
            # 构建提交列表：OOS 稳健因子的最终表达式
            factor_list = []
            for r in sorted(oos_robust,
                            key=lambda x: x['oos_res'].get('fitness', 0), reverse=True):
                is_r  = r['is_res']
                oos_r = r['oos_res']
                expr  = r['best_expr']
                needs_neg = is_r.get('needs_negation', False)
                wq_expr = f"-1 * ({expr})" if needs_neg else expr
                factor_list.append({
                    "wq_expr"  : wq_expr,
                    "local_shp": oos_r.get('alpha_sharpe', 0),
                    "label"    : r['name'][:35],
                })
            submit_factors(factor_list, auto_mode=True)
        except ImportError:
            print("\n⚠️ 未找到 wq_submit.py，跳过自动提交。")
        except Exception as e:
            print(f"\n⚠️ WQ 自动提交失败: {e}")
    elif ENABLE_WQ_SUBMIT and not oos_robust:
        print("\n⚠️ 无 OOS 稳健因子，跳过 WQ 提交。")

    print(f"\n⏱️  总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()