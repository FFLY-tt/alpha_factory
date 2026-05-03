# src/genetic/genetic_engine.py
"""
遗传规划主引擎
- 种群管理、适应度计算、选择/交叉/变异
- 早停（验证集）、多样性维护、子树库更新
- 每代向 Dashboard 推送数据
"""
import random
import uuid
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

warnings.filterwarnings("ignore")

from src.ast.ast_nodes import FactorNode, LeafNode, UnaryNode, BinaryNode
from src.rules.rule_engine import DimType
from src.mining.beam_search import BeamSearchEngine
from src.mining.seed_library import load_seed_factors
from src.evaluation.metrics_calc import (
    compute_factor_sharpe,
    calculate_ic_stability,
    auto_correct_direction,
)
from data_pipeline.data_source import fetch_factor_data

from src.genetic.subtree_library import SubtreeLibrary
from src.genetic.diversity_utils import (
    compute_pairwise_spearman,
    find_redundant,
    compute_ic_sign_consistency,
    compute_novelty,
)

# ────────────────────────────────────────────────────────────────────────────
# 全局约束
# ────────────────────────────────────────────────────────────────────────────
MAX_DEPTH      = 5
MAX_NODES      = 20
ALLOWED_WINDOWS = [5, 10, 20, 60, 252]
UNARY_TS_OPS  = ['Mean', 'Std', 'WMA', 'Max', 'Min', 'Ref']
UNARY_ELEM_OPS = ['Abs', 'Log', 'Sign']
BINARY_ARITH  = ['+', '-', '*', '/']
BINARY_TS     = ['Corr', 'Cov']


# ────────────────────────────────────────────────────────────────────────────
# 表达式合法性校验
# ────────────────────────────────────────────────────────────────────────────
import re as _re

def is_valid_expr(expr: str) -> bool:
    """
    校验 Qlib 表达式是否合法：
    - 时序算子的窗口参数必须是正整数
    - 不能含有浮点窗口或负数窗口
    """
    ts_ops = '|'.join(UNARY_TS_OPS + ['Corr', 'Cov', 'Rank'])
    pattern = _re.compile(
        r'\b(?:' + ts_ops + r')\s*\(.*?,\s*([^,)]+)\)',
        _re.DOTALL
    )
    for match in pattern.finditer(expr):
        window_str = match.group(1).strip()
        try:
            w = int(window_str)
            if w <= 0:
                return False
        except ValueError:
            return False
    return True


# ────────────────────────────────────────────────────────────────────────────
# Individual dataclass
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class Individual:
    node    : FactorNode
    col_name: str = field(default_factory=lambda: f"_g_{uuid.uuid4().hex[:8]}_")
    fitness : float = 0.0
    metrics : Dict[str, Any] = field(default_factory=dict)

    @property
    def expr_str(self) -> str:
        return self.node.expr_str

    @property
    def depth(self) -> int:
        return self.node.depth

    @property
    def node_count(self) -> int:
        return self.node.node_count


# ────────────────────────────────────────────────────────────────────────────
# 新遗传适应度公式
# ────────────────────────────────────────────────────────────────────────────
def compute_genetic_fitness(
        alpha_sharpe  : float,
        depth         : int,
        nodes         : int,
        turnover      : float,
        icir          : float,
        ic_consistency: float,
        novelty       : float = 0.0,
) -> Dict[str, float]:
    """
    Fitness_raw = Alpha_Sharpe_net
                  × (1 – depth_penalty)
                  × (1 – node_penalty)
                  × (1 – turnover_penalty)
                  × stability_factor
    Fitness = Fitness_raw + novelty_bonus
    """
    # ── 深度惩罚（深度≤3 不扣）──
    depth_penalty = min(max(depth - 3, 0) * 0.05, 0.3)

    # ── 节点惩罚（节点≤15 不扣）──
    node_penalty = min(max(nodes - 15, 0) * 0.02, 0.3)

    # ── 换手率惩罚（温柔分段）──
    if turnover > 1.2:
        turnover_penalty = 0.7
    elif turnover > 0.8:
        turnover_penalty = 0.18 + (turnover - 0.8) * 1.0
    elif turnover > 0.5:
        turnover_penalty = (turnover - 0.5) * 0.6
    else:
        turnover_penalty = 0.0

    # ── 稳定性因子 ──
    icir_stability   = min(1.0, abs(icir) / 2)
    stability_factor = (icir_stability + ic_consistency) / 2

    # ── 原始适应度 ──
    fitness_raw = (
        alpha_sharpe
        * (1 - depth_penalty)
        * (1 - node_penalty)
        * (1 - turnover_penalty)
        * stability_factor
    )

    fitness = fitness_raw + novelty

    return {
        "fitness"         : round(fitness, 6),
        "fitness_raw"     : round(fitness_raw, 6),
        "depth_penalty"   : round(depth_penalty, 4),
        "node_penalty"    : round(node_penalty, 4),
        "turnover_penalty": round(turnover_penalty, 4),
        "stability_factor": round(stability_factor, 4),
        "novelty"         : round(novelty, 6),
    }


# ────────────────────────────────────────────────────────────────────────────
# target_alpha 计算
# ────────────────────────────────────────────────────────────────────────────
def _make_target_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    target_alpha = target_ret – 当日横截面等权市场收益
    结果覆盖 target_ret 列，现有评估函数无需修改。
    """
    if 'target_ret' not in df.columns:
        return df
    df = df.copy()
    market_ret = df['target_ret'].groupby(level='datetime').mean()
    df['target_ret'] = df['target_ret'] - df.index.get_level_values('datetime').map(market_ret)
    return df


# ────────────────────────────────────────────────────────────────────────────
# 遗传操作：交叉 / 变异
# 注意：项目 AST 为「有损」设计，BinaryNode/UnaryNode 不保留子节点引用，
#       因此无法做子树遍历，改为以整体节点为单位进行组合。
# ────────────────────────────────────────────────────────────────────────────

def crossover(p1: FactorNode, p2: FactorNode) -> Optional[FactorNode]:
    """
    交叉：用随机二元算子将两个父代节点组合成新节点。
    80% 概率四则运算，20% 概率时序相关算子（Corr/Cov）。
    """
    if random.random() < 0.8:
        op = random.choice(BINARY_ARITH)
        try:
            node = BinaryNode(op, p1, p2)
            if node.depth <= MAX_DEPTH and node.node_count <= MAX_NODES:
                return node
        except (ValueError, Exception):
            pass
    else:
        op = random.choice(['Corr', 'Cov'])
        w  = random.choice(ALLOWED_WINDOWS)
        try:
            node = BinaryNode(op, p1, p2, w)
            if node.depth <= MAX_DEPTH and node.node_count <= MAX_NODES:
                return node
        except (ValueError, Exception):
            pass
    return None


def mutate(
        individual: FactorNode,
        library   : SubtreeLibrary,
        seeds     : List[FactorNode],
) -> Optional[FactorNode]:
    """
    变异：
    - 40%：与库中优良节点做二元组合
    - 30%：与随机种子做二元组合
    - 30%：对自身套时序一元算子
    """
    r = random.random()

    if r < 0.4 and len(library) > 0:
        other = library.sample()
        if other:
            op = random.choice(BINARY_ARITH)
            try:
                node = BinaryNode(op, individual, other)
                if node.depth <= MAX_DEPTH and node.node_count <= MAX_NODES:
                    return node
            except (ValueError, Exception):
                pass

    if r < 0.7 and seeds:
        seed = random.choice(seeds)
        op   = random.choice(BINARY_ARITH)
        try:
            node = BinaryNode(op, individual, seed)
            if node.depth <= MAX_DEPTH and node.node_count <= MAX_NODES:
                return node
        except (ValueError, Exception):
            pass

    # 兜底：套时序一元算子
    ts_op = random.choice(UNARY_TS_OPS)
    w     = random.choice(ALLOWED_WINDOWS)
    try:
        node = UnaryNode(ts_op, individual, w)
        if node.depth <= MAX_DEPTH and node.node_count <= MAX_NODES:
            return node
    except (ValueError, Exception):
        pass

    return None


# ────────────────────────────────────────────────────────────────────────────
# GeneticEngine
# ────────────────────────────────────────────────────────────────────────────
class GeneticEngine:
    """
    遗传规划主引擎。

    参数
    ----
    pop_size      : 种群大小（建议 120）
    elite_count   : 每代保留精英数（建议 24，即 20%）
    max_gens      : 最大进化代数（建议 50）
    patience      : 早停耐心代数（建议 5）
    pool          : 股票池名称（如 "sp500"）
    train_start/end : 训练集时间范围
    val_start/end   : 验证集时间范围（不参与进化）
    dashboard     : Dashboard 实例（可选），用于推送数据
    novelty_delta : 新颖性奖励系数（默认 0.02）
    diversity_gens: 每隔多少代执行一次多样性维护（默认 5）
    """

    def __init__(
            self,
            pop_size    : int = 120,
            elite_count : int = 24,
            max_gens    : int = 50,
            patience    : int = 5,
            pool        : str = "sp500",
            train_start : str = "2010-01-01",
            train_end   : str = "2015-12-31",
            val_start   : str = "2016-01-01",
            val_end     : str = "2017-12-31",
            dashboard   = None,
            novelty_delta     : float = 0.02,
            diversity_gens    : int   = 5,
            corr_threshold    : float = 0.8,
            redundant_threshold: float = 0.9,
    ):
        self.pop_size    = pop_size
        self.elite_count = elite_count
        self.max_gens    = max_gens
        self.patience    = patience
        self.pool        = pool
        self.train_start = train_start
        self.train_end   = train_end
        self.val_start   = val_start
        self.val_end     = val_end
        self.dashboard   = dashboard
        self.novelty_delta        = novelty_delta
        self.diversity_gens       = diversity_gens
        self.corr_threshold       = corr_threshold
        self.redundant_threshold  = redundant_threshold

        self.population  : List[Individual]  = []
        self.library     : SubtreeLibrary    = SubtreeLibrary(max_size=100)
        self._seed_nodes : List[FactorNode]  = []

        self.best_val_fitness  : float = -float('inf')
        self.best_val_overall  : float = -float('inf')
        self.patience_counter  : int   = 0
        self.gen_history       : list  = []

    # ── 初始化 ────────────────────────────────────────────────────────────

    def initialize(self, academic_fragments: List[str] = None) -> None:
        """
        从种子库 + 学术片段初始化种群。
        academic_fragments: 用户提供的学术因子表达式列表（在 run_genetic.py 中填写）
        """
        # 1. 加载内置种子因子
        formulas, dim_types = load_seed_factors()
        seed_nodes = []
        for name, expr in formulas.items():
            try:
                node = LeafNode(name, expr, dim_types[name])
                seed_nodes.append(node)
            except Exception:
                pass

        # 2. 加载学术片段（用户手动填写的）
        if academic_fragments:
            for i, expr in enumerate(academic_fragments):
                try:
                    node = LeafNode(f"acad_{i:03d}", expr.strip(), DimType.RATIO)
                    seed_nodes.append(node)
                except Exception as e:
                    print(f"   ⚠️ 学术片段 [{i}] 加载失败：{e}，跳过")

        self._seed_nodes = seed_nodes
        print(f"   种子节点总数: {len(seed_nodes)} 个（含 {len(academic_fragments or [])} 个学术片段）")

        # 3. 用波束搜索引擎生成初始候选
        beam_engine = BeamSearchEngine(beam_width=len(seed_nodes))
        print(f"   正在生成初始种群（目标 {self.pop_size} 个）...")
        candidates = beam_engine.generate_candidates(seed_nodes, max_eval=self.pop_size * 2)

        self.population = []
        for node in candidates[:self.pop_size]:
            self.population.append(Individual(node=node))

        # 4. 不足则随机补充
        attempts = 0
        while len(self.population) < self.pop_size and attempts < self.pop_size * 10:
            attempts += 1
            node = self._random_combine(seed_nodes)
            if node and node.depth <= MAX_DEPTH and node.node_count <= MAX_NODES:
                self.population.append(Individual(node=node))

        print(f"   初始种群实际大小: {len(self.population)} 个")

    def _random_combine(self, seeds: List[FactorNode], depth: int = 0) -> Optional[FactorNode]:
        """递归随机组合种子节点"""
        if not seeds:
            return None
        if depth >= 3 or (depth > 0 and random.random() < 0.35):
            return random.choice(seeds)

        choice = random.random()
        if choice < 0.6:   # 二元运算
            left  = self._random_combine(seeds, depth + 1)
            right = self._random_combine(seeds, depth + 1)
            if left is None or right is None:
                return random.choice(seeds)
            op = random.choice(BINARY_ARITH)
            try:
                return BinaryNode(op, left, right)
            except Exception:
                return random.choice(seeds)
        elif choice < 0.85:  # 一元时序运算（包窗口参数，用 LeafNode 包装）
            child = self._random_combine(seeds, depth + 1)
            if child is None:
                return random.choice(seeds)
            ts_op = random.choice(UNARY_TS_OPS)
            w     = random.choice(ALLOWED_WINDOWS)
            expr  = f"{ts_op}({child.expr_str}, {w})"
            try:
                return LeafNode(f"_tmp_{uuid.uuid4().hex[:4]}", expr, child.dim_type)
            except Exception:
                return child
        else:  # 元素级一元运算
            child = self._random_combine(seeds, depth + 1)
            if child is None:
                return random.choice(seeds)
            op = random.choice(UNARY_ELEM_OPS)
            try:
                return UnaryNode(op, child)
            except Exception:
                return child

    # ── 主进化循环 ────────────────────────────────────────────────────────

    def run(self) -> List[Individual]:
        """执行遗传进化，返回最终种群（已按 fitness 降序排列）"""
        print(f"\n{'='*60}")
        print(f"🧬 遗传规划启动：{self.max_gens} 代 × {self.pop_size} 个体")
        print(f"   训练集: {self.train_start} ~ {self.train_end}")
        print(f"   验证集: {self.val_start}   ~ {self.val_end}")
        print(f"{'='*60}\n")

        for gen in range(self.max_gens):
            print(f"\n── [第 {gen+1}/{self.max_gens} 代] ──────────────────────────")

            # Step A：批量取训练集数据，计算 target_alpha
            df_train = self._fetch_population_data(self.train_start, self.train_end)
            if df_train is None:
                print("  ⚠️ 训练集取数失败，跳过本代")
                continue

            # Step B：评估所有个体（训练集）
            self._evaluate_population(df_train, self.population)
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            top_elite = self.population[:self.elite_count]
            train_best_fitness = top_elite[0].fitness if top_elite else 0.0
            train_best_sharpe  = top_elite[0].metrics.get('alpha_sharpe', 0.0) if top_elite else 0.0

            # Step C：验证集评估精英（不参与选择，只用于早停）
            df_val = self._fetch_population_data(
                self.val_start, self.val_end,
                individuals=top_elite
            )
            val_fitness_list = []
            if df_val is not None:
                self._evaluate_population(df_val, top_elite, novelty_active=False)
                val_fitness_list = [ind.fitness for ind in top_elite]

            val_best_fitness = float(np.mean(val_fitness_list)) if val_fitness_list else 0.0
            val_best_sharpe  = top_elite[0].metrics.get('alpha_sharpe', 0.0) if top_elite else 0.0

            # 恢复训练集 fitness（验证集评估覆盖了 metrics，需要还原）
            self._evaluate_population(df_train, top_elite, novelty_active=False)

            # Step D：早停判断
            early_stop = False
            if val_best_fitness > self.best_val_fitness:
                self.best_val_fitness = val_best_fitness
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if val_best_fitness > self.best_val_overall:
                self.best_val_overall = val_best_fitness

            if self.patience_counter >= self.patience:
                early_stop = True

            # Step E：统计 & 发送 Dashboard
            gen_data = self._build_gen_data(
                gen, train_best_fitness, val_best_fitness,
                train_best_sharpe, val_best_sharpe, early_stop,
                df_train
            )
            self.gen_history.append(gen_data)
            if self.dashboard:
                self.dashboard.send_data(gen_data)
            self._print_summary(gen_data)

            if early_stop:
                print(f"\n🛑 早停触发！连续 {self.patience} 代验证集适应度未提升。")
                break

            # Step F：生成新种群
            new_pop = list(top_elite)   # 精英直接保留
            stagnant = 0
            while len(new_pop) < self.pop_size:
                if random.random() < 0.8:  # 交叉
                    p1 = self._tournament(self.population)
                    p2 = self._tournament(self.population)
                    child_node = crossover(p1.node, p2.node)
                    if child_node and is_valid_expr(child_node.expr_str):
                        new_pop.append(Individual(node=child_node))
                    else:
                        # 交叉失败：注入随机新个体，而不是复制父代
                        rnd = self._random_combine(self._seed_nodes)
                        if rnd and is_valid_expr(rnd.expr_str):
                            new_pop.append(Individual(node=rnd))
                        else:
                            new_pop.append(Individual(node=p1.node))
                        stagnant += 1
                else:  # 变异
                    parent = self._tournament(self.population)
                    mut_node = mutate(parent.node, self.library, self._seed_nodes)
                    if mut_node and is_valid_expr(mut_node.expr_str):
                        new_pop.append(Individual(node=mut_node))
                    else:
                        # 变异失败：同样注入随机新个体
                        rnd = self._random_combine(self._seed_nodes)
                        if rnd and is_valid_expr(rnd.expr_str):
                            new_pop.append(Individual(node=rnd))
                        else:
                            new_pop.append(Individual(node=parent.node))
                        stagnant += 1

            self.population = new_pop

            # Step G：多样性维护（每代都检查，不只是每 N 代）
            new_injected = 0
            redundant_removed = 0

            # 计算当前多样性（用表达式字符串集合快速判断同质化程度）
            expr_set = set(ind.expr_str for ind in self.population)
            unique_ratio = len(expr_set) / max(len(self.population), 1)

            # 触发条件：每 diversity_gens 代做完整去重，或者独特表达式比例 < 0.5（严重同质化）
            if (gen + 1) % self.diversity_gens == 0 or unique_ratio < 0.5:
                redundant_removed, new_injected = self._diversity_maintenance(df_train)
                self.library.update(
                    [ind.node for ind in top_elite],
                    [ind.fitness for ind in top_elite]
                )
                trigger = "定时" if (gen + 1) % self.diversity_gens == 0 else f"⚠️ 同质化({unique_ratio:.0%})"
                print(f"  🔄 多样性维护[{trigger}]：淘汰 {redundant_removed} 个 | 注入 {new_injected} 个新个体 | 独特表达式: {len(expr_set)}/{len(self.population)}")

        # 最终排序
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        print(f"\n✅ 遗传进化完成，最优训练适应度: {self.population[0].fitness:+.4f}")
        return self.population

    # ── 批量取数 & 评估 ──────────────────────────────────────────────────

    def _fetch_population_data(
            self,
            start      : str,
            end        : str,
            individuals: List[Individual] = None
    ) -> Optional[pd.DataFrame]:
        """
        批量取数。individuals 为 None 时取整个 self.population。
        若整批失败，自动逐个过滤非法表达式后重试一次。
        返回已计算 target_alpha 的 DataFrame。
        """
        inds = individuals if individuals is not None else self.population

        def _build_formulas(ind_list):
            f = {"_vol_raw_": "$volume"}
            for ind in ind_list:
                f[ind.col_name] = ind.expr_str
            return f

        # ── 第一次尝试：整批取数 ──────────────────────────────────────────
        try:
            df = fetch_factor_data(_build_formulas(inds), self.pool, start, end)
            df = _make_target_alpha(df)
            return df
        except Exception as e:
            print(f"  ⚠️ 批量取数失败（{e}），过滤非法表达式后重试...")

        # ── 第二次尝试：先用校验器过滤，再取数 ──────────────────────────
        valid_inds = [ind for ind in inds if is_valid_expr(ind.expr_str)]
        removed = len(inds) - len(valid_inds)
        if removed > 0:
            print(f"     过滤掉 {removed} 个含非法窗口的个体")

        # 用种子节点替换被过滤掉的个体（只针对 self.population，不改 elite 列表）
        if individuals is None and removed > 0:
            for ind in inds:
                if not is_valid_expr(ind.expr_str) and self._seed_nodes:
                    seed = random.choice(self._seed_nodes)
                    ind.node     = seed
                    ind.col_name = f"_g_{uuid.uuid4().hex[:8]}_"
                    ind.fitness  = 0.0
                    ind.metrics  = {}

        if not valid_inds:
            print("  ⚠️ 无合法个体，跳过本代")
            return None

        try:
            df = fetch_factor_data(_build_formulas(valid_inds), self.pool, start, end)
            df = _make_target_alpha(df)
            return df
        except Exception as e2:
            print(f"  ⚠️ 重试仍失败: {e2}")
            return None

    def _evaluate_population(
            self,
            df           : pd.DataFrame,
            individuals  : List[Individual],
            novelty_active: bool = True
    ) -> None:
        """原地更新每个 Individual 的 fitness 和 metrics"""
        all_cols = [ind.col_name for ind in individuals if ind.col_name in df.columns]

        for ind in individuals:
            if ind.col_name not in df.columns:
                ind.fitness = -999.0
                continue

            try:
                metrics   = compute_factor_sharpe(df, ind.col_name)
                mean_ic, icir = calculate_ic_stability(df, ind.col_name)
                direction = auto_correct_direction(
                    mean_ic, metrics['sharpe_net'], metrics['sharpe_gross']
                )
                ic_consistency = compute_ic_sign_consistency(df, ind.col_name)
                novelty = (
                    compute_novelty(df, ind.col_name, all_cols, self.novelty_delta)
                    if novelty_active else 0.0
                )

                fit_result = compute_genetic_fitness(
                    alpha_sharpe   = direction['corrected_sharpe_net'],
                    depth          = ind.depth,
                    nodes          = ind.node_count,
                    turnover       = metrics['turnover'],
                    icir           = icir,
                    ic_consistency = ic_consistency,
                    novelty        = novelty,
                )

                ind.fitness = fit_result['fitness']
                ind.metrics = {
                    **metrics,
                    **fit_result,
                    'ic'              : mean_ic,
                    'icir'            : icir,
                    'alpha_sharpe'    : direction['corrected_sharpe_net'],
                    'needs_negation'  : direction['needs_negation'],
                    'direction_note'  : direction['direction'],
                    'ic_consistency'  : ic_consistency,
                }
            except Exception as e:
                ind.fitness  = -999.0
                ind.metrics  = {}

    # ── 锦标赛选择 ───────────────────────────────────────────────────────

    def _tournament(self, pool: List[Individual], k: int = 3) -> Individual:
        contestants = random.sample(pool, min(k, len(pool)))
        return max(contestants, key=lambda x: x.fitness)

    # ── 多样性维护 ───────────────────────────────────────────────────────

    def _diversity_maintenance(self, df_train: pd.DataFrame):
        """
        1. 先用表达式字符串快速去重（比 Spearman 快得多）
        2. 计算种群 pairwise 相关，淘汰冗余个体
        3. 同质化严重时积极注入新个体
        """
        # ── 快速去重：表达式完全相同的只保留一个（精英除外保留首个）
        seen_exprs = {}
        dedup_pop  = []
        for ind in self.population:
            if ind.expr_str not in seen_exprs:
                seen_exprs[ind.expr_str] = True
                dedup_pop.append(ind)
        expr_dedup_removed = len(self.population) - len(dedup_pop)
        self.population = dedup_pop

        # ── Spearman 去重（相关 > redundant_threshold）
        col_names   = [ind.col_name for ind in self.population]
        fitness_map = {ind.col_name: ind.fitness for ind in self.population}
        to_remove   = set(find_redundant(df_train, col_names, fitness_map,
                                          self.redundant_threshold))
        before = len(self.population)
        self.population = [ind for ind in self.population
                           if ind.col_name not in to_remove]
        redundant_removed = expr_dedup_removed + (before - len(self.population))

        # ── 注入新个体
        mean_corr    = compute_pairwise_spearman(df_train, col_names)
        new_injected = 0
        # 基础补位：补到 pop_size
        inject_n = max(0, self.pop_size - len(self.population))
        # 积极注入：多样性不足时额外注入
        if mean_corr > 0.5:    # 原来是 < 0.15（方向反了！高相关=低多样性）
            inject_n = max(inject_n, self.pop_size // 3)

        for _ in range(inject_n * 3):
            if len(self.population) >= self.pop_size:
                break
            node = self._random_combine(self._seed_nodes)
            if node and is_valid_expr(node.expr_str):
                self.population.append(Individual(node=node))
                new_injected += 1

        return redundant_removed, new_injected

    # ── Dashboard 数据打包 ───────────────────────────────────────────────

    def _build_gen_data(
            self, gen, train_best_fitness, val_best_fitness,
            train_best_sharpe, val_best_sharpe, early_stop, df_train
    ) -> dict:
        elite = self.population[:self.elite_count]
        col_names = [ind.col_name for ind in self.population
                     if ind.col_name in (df_train.columns if df_train is not None else [])]
        diversity_corr = (
            compute_pairwise_spearman(df_train, col_names)
            if df_train is not None and col_names else 0.0
        )

        elite_profiles = []
        for ind in elite[:10]:
            elite_profiles.append({
                "expr"        : ind.expr_str[:40],
                "depth"       : ind.depth,
                "nodes"       : ind.node_count,
                "turnover"    : round(ind.metrics.get('turnover', 0), 4),
                "train_sharpe": round(ind.metrics.get('alpha_sharpe', 0), 4),
                "val_sharpe"  : 0.0,   # 填充（可在验证集评估后更新）
                "icir"        : round(ind.metrics.get('icir', 0), 4),
            })

        warnings_list = []
        if diversity_corr < 0.15:
            warnings_list.append("⚠️ 多样性过低（corr<0.15），种群可能坍缩")
        if self.patience_counter >= 3:
            warnings_list.append(f"⚠️ 早停计数器 {self.patience_counter}/{self.patience}")
        if early_stop:
            warnings_list.append("🛑 早停已触发")

        depths   = [ind.depth for ind in self.population]
        nodes    = [ind.node_count for ind in self.population]
        turnovers = [ind.metrics.get('turnover', 0) for ind in self.population
                     if ind.metrics]

        return {
            "gen"                 : gen + 1,
            "train_best_fitness"  : round(train_best_fitness, 4),
            "val_best_fitness"    : round(val_best_fitness, 4),
            "val_best_overall"    : round(self.best_val_overall, 4),
            "train_best_sharpe"   : round(train_best_sharpe, 4),
            "val_best_sharpe"     : round(val_best_sharpe, 4),
            "avg_depth"           : round(float(np.mean(depths)), 2) if depths else 0,
            "max_depth"           : int(max(depths)) if depths else 0,
            "avg_nodes"           : round(float(np.mean(nodes)), 2) if nodes else 0,
            "max_nodes"           : int(max(nodes)) if nodes else 0,
            "avg_turnover"        : round(float(np.mean(turnovers)), 4) if turnovers else 0,
            "diversity_corr"      : round(diversity_corr, 4),
            "redundant_removed"   : 0,
            "new_injected"        : 0,
            "patience_counter"    : self.patience_counter,
            "early_stop"          : early_stop,
            "elite_profiles"      : elite_profiles,
            "warnings"            : warnings_list,
        }

    def _print_summary(self, g: dict) -> None:
        health = ("🟢" if g['diversity_corr'] > 0.2 and g['val_best_fitness'] >= self.best_val_overall
                  else "🟡" if g['diversity_corr'] > 0.1 else "🔴")
        print(f"  {health} Gen={g['gen']:3d} | "
              f"Train_Fit={g['train_best_fitness']:+.4f} | "
              f"Val_Fit={g['val_best_fitness']:+.4f} | "
              f"Val_Best={g['val_best_overall']:+.4f} | "
              f"Patience={g['patience_counter']}/{self.patience} | "
              f"Diversity={g['diversity_corr']:.3f}")
        if g['warnings']:
            for w in g['warnings']:
                print(f"         {w}")

    # ── 取结果 ───────────────────────────────────────────────────────────

    def get_top_n(self, n: int = 20) -> List[Individual]:
        """取 fitness 最高的 n 个个体"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:n]