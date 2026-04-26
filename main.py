# main.py
"""
AlphaForge V3 - 七模块因子精炼管道集成版
基因入库标准：abs(IC) > 0.02（与原始版本一致）
Fitness 仅作为附加字段记录，不参与入库决策
"""
import warnings
warnings.filterwarnings("ignore")

import os
import time
import json
import numpy as np
from typing import List, Dict

from data_pipeline.data_source import init_qlib_engine, fetch_factor_data
from src.mining.beam_search import BeamSearchEngine
from src.evaluation.metrics_calc import (
    select_top_k_orthogonal, evaluate_factor_fitness,
    compute_factor_sharpe, calculate_ic_stability
)
from src.evaluation.backtester import quantile_backtest, walk_forward_backtest
from src.processing.factor_pipeline import FactorPipeline
from src.ast.ast_nodes import LeafNode, BinaryNode, TernaryNode
from src.rules.rule_engine import DimType


# ==============================================================================
# 配置
# ==============================================================================
BEAM_WIDTH      = 20
MAX_DEPTH       = 5
CORR_THRESHOLD  = 0.8
MAX_EVAL        = 100
POOL            = "sp500"

IS_START_DATE   = "2015-01-01"
IS_END_DATE     = "2019-12-31"
OOS_START_DATE  = "2020-01-01"
OOS_END_DATE    = "2023-12-31"

# 7 模块开关
ENABLE_LIQUIDITY_FILTER = True   # 模块一
ENABLE_ROBUSTIFY        = True   # 模块二
ENABLE_NEUTRALIZATION   = False  # 模块三（需要行业/市值列）
ENABLE_MARKET_STATE     = True   # 模块四
ENABLE_WALK_FORWARD     = True   # 模块五
ENABLE_PORTFOLIO_SELECT = True   # 模块六

# 基因库入库标准（恢复原始版本：abs(IC) > 0.02）
RETAIN_THRESHOLD = 0.02
GENE_BANK_FILE   = "gene_bank.json"


# ==============================================================================
# 模块四：市场状态机
# ==============================================================================

def build_market_state_alphas(base_factor_nodes: list) -> list:
    """为精英因子生成牛市/高波动反转两个低相关衍生版本"""
    MKT_RET_EXPR = "Mean($close / Ref($close, 1) - 1, 20)"
    VOL_EXPR     = "Std($close / Ref($close, 1) - 1, 20)"

    derived = []
    mkt_ret_node = LeafNode("mkt_ret_20d", f"Ref({MKT_RET_EXPR}, 1)", DimType.RATIO)
    zero_node    = LeafNode("zero", "0", DimType.RATIO)
    vol_now      = LeafNode("vol_20d", f"Ref({VOL_EXPR}, 1)", DimType.RATIO)
    vol_ma       = LeafNode("vol_ma_60d", f"Ref(Mean({VOL_EXPR}, 60), 1)", DimType.RATIO)

    bull_cond     = BinaryNode(">", mkt_ret_node, zero_node)
    high_vol_cond = BinaryNode(">", vol_now, vol_ma)

    for node in base_factor_nodes:
        if node.dim_type == DimType.BOOLEAN:
            continue

        zero_factor = LeafNode("zero_factor", "0", node.dim_type)

        # 牛市版
        try:
            bn = TernaryNode(bull_cond, node, zero_factor)
            bn.name = f"BULL_{node.name[:30]}"
            bn.expr_str = f"If(Ref({MKT_RET_EXPR}, 1) > 0, {node.expr_str}, 0)"
            derived.append(bn)
        except ValueError:
            pass

        # 高波动反转版
        if node.dim_type == DimType.RATIO:
            try:
                neg_one = LeafNode("neg_one", "-1", DimType.RATIO)
                rev = BinaryNode("*", neg_one, node)
                hv = TernaryNode(high_vol_cond, rev, node)
                hv.name = f"HV_REV_{node.name[:25]}"
                hv.expr_str = (f"If(Ref({VOL_EXPR}, 1) > Ref(Mean({VOL_EXPR}, 60), 1),"
                               f" -1 * {node.expr_str}, {node.expr_str})")
                derived.append(hv)
            except ValueError:
                pass

    print(f"   🌐 [模块四] {len(base_factor_nodes)} 个基础因子 → {len(derived)} 个衍生因子")
    return derived


# ==============================================================================
# 批量 Sharpe（用于模块七 Fitness 评估）
# ==============================================================================

def batch_compute_sharpe(df, factor_names: List[str],
                        quantiles: int = 5, cost_bps: float = 5.0) -> Dict[str, float]:
    """批量计算净 Sharpe，调用独立 compute_factor_sharpe（不动原回测）"""
    sharpe_dict = {}
    for fname in factor_names:
        m = compute_factor_sharpe(df, fname, quantiles, cost_bps)
        sharpe_dict[fname] = m["sharpe_net"]
    return sharpe_dict


# ==============================================================================
# 基因库
# ==============================================================================

def load_gene_bank() -> Dict:
    if os.path.exists(GENE_BANK_FILE):
        with open(GENE_BANK_FILE, "r") as f:
            return json.load(f)
    return {}


def save_gene_bank(bank: Dict):
    with open(GENE_BANK_FILE, "w") as f:
        json.dump(bank, f, indent=4, ensure_ascii=False)


def update_gene_bank(
        gene_bank: Dict, ic_report: Dict[str, float],
        expr_map: Dict[str, str], fitness_map: Dict[str, Dict]
) -> int:
    """
    入库标准：abs(IC) > 0.02（保持原始严格筛选标准）
    Fitness 仅作为附加字段记录，方便后续分析
    """
    new_count = 0
    for name, ic in ic_report.items():
        if abs(ic) > RETAIN_THRESHOLD and name in expr_map:
            entry = {"expr": expr_map[name], "ic": round(float(ic), 6)}
            # 附加 Fitness 信息（如果计算了）
            fit_info = fitness_map.get(name, {})
            if fit_info:
                entry.update({
                    "fitness":  round(fit_info.get("fitness", 0), 6),
                    "sharpe":   round(fit_info.get("sharpe", 0), 6),
                    "turnover": round(fit_info.get("turnover", 0), 6),
                    "icir":     round(fit_info.get("icir", 0), 6),
                })
            gene_bank[name] = entry
            new_count += 1
    return new_count


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    print("=" * 60)
    print("🚀 AlphaForge V3 - 七模块因子精炼管道")
    print("=" * 60)

    cur = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur, "data", "qlib_data", "us_data")
    init_qlib_engine(data_path)

    search_engine = BeamSearchEngine(beam_width=BEAM_WIDTH)
    current_beam = search_engine.seed_nodes
    print(f"\n🌱 成功加载底层种子: {len(current_beam)} 个")

    gene_bank = load_gene_bank()
    print(f"📚 已有基因库: {len(gene_bank)} 个历史因子\n")

    for depth in range(2, MAX_DEPTH + 1):
        print(f"\n{'─'*50}")
        print(f"🧬 Depth = {depth}  探索中...")
        print(f"{'─'*50}")
        t0 = time.time()

        # A：生成候选
        print(">> [A] 生成变异候选者...")
        candidates = search_engine.generate_candidates(current_beam, max_eval=MAX_EVAL)
        formulas_dict = {n.name: n.expr_str for n in candidates}
        print(f">> 合法候选公式: {len(candidates)} 个")

        # B：取数（带流动性列）
        print(">> [B] 调取样本内数据...")
        fetch_with_vol = dict(formulas_dict)
        fetch_with_vol["_vol_raw_"] = "$volume"
        is_df = fetch_factor_data(fetch_with_vol, POOL, IS_START_DATE, IS_END_DATE)
       
        # C：IC + 正交化（原版逻辑，未改动）
        print(">> [C] Rank IC 评估 + 正交化剪枝...")
        ic_input = is_df.drop(columns=["_vol_raw_"], errors="ignore")
        current_beam, ic_report = select_top_k_orthogonal(
            candidates, ic_input, k=BEAM_WIDTH, corr_threshold=CORR_THRESHOLD
        )

        # D：精炼管道（模块一/二/三）
        print(">> [D] 精炼管道...")
        elite_names = [n.name for n in current_beam]
        elite_cols = [c for c in elite_names if c in is_df.columns]
        elite_df = is_df[elite_cols + ["target_ret", "_vol_raw_"]].copy()

        pipeline = FactorPipeline(
            df=elite_df, factor_cols=elite_names, volume_col="_vol_raw_",
            industry_col=None, market_cap_col=None,
        )
        refined_df = pipeline.run(
            enable_liquidity_filter=ENABLE_LIQUIDITY_FILTER,
            enable_robustify=ENABLE_ROBUSTIFY,
            enable_neutralization=ENABLE_NEUTRALIZATION,
        )

        # E：批量 Sharpe
        print(">> [E] 计算 IS 期 Sharpe...")
        sharpe_map = batch_compute_sharpe(refined_df, elite_names)

        # F：模块七 Fitness
        print(">> [F] 计算本地 Fitness...")
        fitness_map: Dict[str, Dict] = {}
        existing_cols = list(gene_bank.keys())[:20]
        for fname in elite_names:
            if fname not in refined_df.columns:
                continue
            fitness_map[fname] = evaluate_factor_fitness(
                df=refined_df, factor_name=fname,
                sharpe=sharpe_map.get(fname, 0.0),
                existing_factor_cols=existing_cols
            )

        # G：模块四衍生
        if ENABLE_MARKET_STATE:
            print(">> [G] 模块四：状态机衍生...")
            derived = build_market_state_alphas(current_beam)
            if derived:
                d_formulas = {n.name: n.expr_str for n in derived}
                d_formulas["_vol_raw_"] = "$volume"
                try:
                    d_df = fetch_factor_data(d_formulas, POOL, IS_START_DATE, IS_END_DATE)
                    d_names = [n.name for n in derived if n.name in d_df.columns]
                    if d_names:
                        d_sharpe = batch_compute_sharpe(d_df, d_names)
                        added = 0
                        for dname in d_names:
                            d_fit = evaluate_factor_fitness(
                                df=d_df, factor_name=dname,
                                sharpe=d_sharpe.get(dname, 0.0)
                            )
                            # 衍生因子的 IC 也算一下，决定是否入库
                            d_ic, _ = calculate_ic_stability(d_df, dname)
                            if abs(d_ic) > RETAIN_THRESHOLD:
                                ic_report[dname] = d_ic
                                fitness_map[dname] = d_fit
                                formulas_dict[dname] = d_formulas[dname]
                                added += 1
                        print(f"   衍生因子通过 IC 门槛: {added} 个")
                except Exception as e:
                    print(f"   ⚠️ 衍生因子计算异常: {e}")

        # H：模块六组合筛选
        if ENABLE_PORTFOLIO_SELECT and len(sharpe_map) > 1:
            print(">> [H] 模块六：低相关组合筛选...")
            pipeline.select_portfolio(
                sharpe_dict=sharpe_map,
                max_factors=BEAM_WIDTH,
                min_fitness=0.0,
                max_avg_corr=0.3
            )

        # I：更新基因库（入库标准 abs(IC) > 0.02）
        new_count = update_gene_bank(gene_bank, ic_report, formulas_dict, fitness_map)
        save_gene_bank(gene_bank)

        # 本轮报告
        elapsed = time.time() - t0
        print(f"\n✅ Depth {depth} 完成！耗时: {elapsed:.1f}s | "
              f"新增基因: {new_count} | 库存总量: {len(gene_bank)}")

        top5 = sorted(
            [(n, ic_report.get(n.name, 0)) for n in current_beam],
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        print(f"🏆 Top 5 精英因子（按 |IC|）:")
        for idx, (nd, ic_v) in enumerate(top5):
            fit_v = fitness_map.get(nd.name, {}).get("fitness", 0)
            print(f"  [{idx+1}] IC:{ic_v:+.4f} | Fit:{fit_v:+.4f} | {nd.expr_str}")

    # 样本外验证
    print("\n\n" + "=" * 60)
    print("🎯 挖掘结束！样本外验证最优因子...")
    print("=" * 60)

    if not current_beam:
        print("⚠️ 无精英因子，退出。")
        return

    # 按 |IC| 选最优因子（与入库标准一致）
    best_node = max(current_beam, key=lambda n: abs(ic_report.get(n.name, 0)))
    best_ic = ic_report.get(best_node.name, 0)
    best_fit = fitness_map.get(best_node.name, {})
    print(f"\n🥇 最优因子: {best_node.expr_str}")
    print(f"   IS IC={best_ic:+.4f} | "
          f"IS Sharpe={best_fit.get('sharpe', 0):.3f} | "
          f"IS Fitness={best_fit.get('fitness', 0):.4f}")

    # OOS 标准回测（原版 quantile_backtest）
    oos_df = fetch_factor_data(
        {best_node.name: best_node.expr_str}, POOL, OOS_START_DATE, OOS_END_DATE
    )
    print(f"\n>> 样本外回测（{OOS_START_DATE} ~ {OOS_END_DATE}）")
    quantile_backtest(oos_df, best_node.name, quantiles=5)

    # 模块五：Walk-Forward
    if ENABLE_WALK_FORWARD:
        print(f"\n>> 模块五：Walk-Forward 滚动验证...")
        wf = walk_forward_backtest(
            fetch_func=fetch_factor_data,
            factor_name=best_node.name,
            factor_expr=best_node.expr_str,
            instrument_pool=POOL,
            folds=[
                ("2015-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
                ("2016-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
                ("2017-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
                ("2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
                ("2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
            ]
        )
        if best_node.name in gene_bank:
            gene_bank[best_node.name]["walk_forward"] = {
                "mean_sharpe":   round(wf["mean_sharpe"], 4),
                "std_sharpe":    round(wf["std_sharpe"], 4),
                "min_sharpe":    round(wf["min_sharpe"], 4),
                "stability":     round(wf["sharpe_stability"], 4),
            }
            save_gene_bank(gene_bank)
            print(f"✅ Walk-Forward 结果已写入基因库！")

    # 最终统计
    print(f"\n{'='*60}")
    print(f"📚 最终基因库统计")
    print(f"{'='*60}")
    valid = {k: v for k, v in gene_bank.items()
             if isinstance(v, dict) and abs(v.get("ic", 0)) > RETAIN_THRESHOLD}
    print(f"总因子数: {len(gene_bank)} | 通过 IC 门槛: {len(valid)}")

    if valid:
        ics = [abs(v.get("ic", 0)) for v in valid.values()]
        print(f"平均 |IC|: {float(np.mean(ics)):.4f} | "
              f"最高 |IC|: {float(np.max(ics)):.4f}")

        print(f"\n🏆 基因库 Top 10（按 |IC|）:")
        top10 = sorted(valid.items(), key=lambda x: abs(x[1].get("ic", 0)), reverse=True)[:10]
        for i, (name, info) in enumerate(top10):
            print(f"  [{i+1:2d}] IC={info.get('ic',0):+.4f} | "
                  f"Fit={info.get('fitness',0):+.4f} | "
                  f"Sharpe={info.get('sharpe',0):+.3f}")
            print(f"       {info.get('expr', name)[:80]}")

    print(f"\n🎉 全流程完成！结果已保存至 {GENE_BANK_FILE}")


if __name__ == "__main__":
    main()