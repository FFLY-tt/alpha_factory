# src/mining/beam_search.py
import random
from typing import List
from src.rules.rule_engine import Ops, DimType
from src.ast.ast_nodes import FactorNode, LeafNode, UnaryNode, BinaryNode, TernaryNode
from src.mining.seed_library import load_seed_factors


class BeamSearchEngine:
    def __init__(self, beam_width: int = 50, ts_params: List[int] = None):
        """
        初始化波束搜索引擎
        :param beam_width: 每一层保留的精英因子数量 (K值)
        :param ts_params: 时序算子变异时可用的周期参数
        """
        self.beam_width = beam_width
        self.ts_params = ts_params or [5, 10, 20]  # 默认变异周期：5天, 10天, 20天
        self.seed_nodes = self._load_and_convert_seeds()

    def _load_and_convert_seeds(self) -> List[LeafNode]:
        """将 E100 种子库转化为 AST 的叶子节点列表"""
        formulas, dim_types = load_seed_factors()
        seed_nodes = []
        for name, expr in formulas.items():
            seed_nodes.append(LeafNode(name=name, expr_str=expr, dim_type=dim_types[name]))
        return seed_nodes

    def generate_candidates(self, current_beam: List[FactorNode], max_eval: int = 0) -> List[FactorNode]:
        """
        核心扩增引擎：基于当前波束池，穷举生成下一代候选因子
        """
        candidates = []

        # 1. 一元扩增 (Unary Expansion)
        print("  -> 正在进行一元变异扩增...")
        for node in current_beam:
            for op in Ops.UNARY_ALL:
                try:
                    if op in Ops.UNARY_TS_SMOOTH:
                        # 带有时间周期的算子，遍历周期参数
                        for d in self.ts_params:
                            candidates.append(UnaryNode(op, node, d))
                    else:
                        # 无需参数的算子 (如 Rank, Sign)
                        candidates.append(UnaryNode(op, node))
                except ValueError:
                    pass  # 触发规则引擎防御，静默剪枝

        # 2. 二元扩增 (Binary Expansion)
        # 为了扩大基因多样性，我们将 当前波束 与 当前波束/原始种子 分别组合
        print("  -> 正在进行二元交叉重组...")
        pool_for_binary = current_beam + self.seed_nodes

        for left in current_beam:
            for right in pool_for_binary:
                for op in Ops.BIN_ALL:
                    try:
                        if op in Ops.BIN_TS:
                            for d in self.ts_params:
                                candidates.append(BinaryNode(op, left, right, d))
                        else:
                            candidates.append(BinaryNode(op, left, right))
                    except ValueError:
                        pass  # 触发规则引擎防御，静默剪枝

        # # 3. 三元扩增 (Ternary Expansion - 状态机)
        # print("  -> 正在生成高级三元状态机 (If-Else)...")
        # # a. 优先从刚才的二元候选集中，挑出合法的布尔条件节点
        # bool_conditions = [n for n in candidates if n.dim_type == DimType.BOOLEAN]
        #
        # # b. 为了防止算力爆炸，我们在三元组装时进行适度采样
        # # 真实场景中，如果 bool_conditions 和 current_beam 很大，全排列会极慢
        # sampled_conds = random.sample(bool_conditions, min(len(bool_conditions), 50))
        #
        # for cond in sampled_conds:
        #     for true_node in current_beam:
        #         for false_node in current_beam:
        #             try:
        #                 candidates.append(TernaryNode(cond, true_node, false_node))
        #             except ValueError:
        #                 pass  # 触发分支量纲冲突或退化拦截

        # 过滤掉布尔节点 (因为布尔值不能作为最终因子的输出，只能作为If的条件)
        valid_candidates = [n for n in candidates if n.dim_type != DimType.BOOLEAN]

        # 字符串级去重 (极度关键：防止长相不同但表达式完全一样的公式)
        unique_candidates = list({n.expr_str: n for n in valid_candidates}.values())

        # 💥 动态算力熔断
        if 0 < max_eval < len(unique_candidates):
            print(f"  ⚠️ 极速实验模式开启：从 {len(unique_candidates)} 个公式中随机抽样 {max_eval} 个！")
            unique_candidates = random.sample(unique_candidates, max_eval)

        return unique_candidates


# =========== 引擎空转自测 (不连 Qlib) ===========
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AlphaForge 波束引擎试运行 (深度 = 2)")
    print("=" * 60)

    engine = BeamSearchEngine(beam_width=10)

    # 初始状态：深度为 1 的波束池就是我们的 E100 种子库
    beam_depth_1 = engine.seed_nodes
    print(f"🌱 初始种子数量: {len(beam_depth_1)}")

    # 向深处进发！生成 Depth = 2 的公式
    print("\n⏳ 开始向深度 2 扩增...")
    candidates_depth_2 = engine.generate_candidates(current_beam=beam_depth_1)

    print(f"\n✅ 扩增完成！成功绕过防御网的合法因子数量: {len(candidates_depth_2)}")

    # 随机展示 5 个通过了所有物理法则考验的天才因子
    print("\n🎲 随机抽赏 5 个 Depth 2 天才公式:")
    for n in random.sample(candidates_depth_2, 5):
        print(f"[{n.dim_type.name.ljust(5)}] {n.expr_str}")

    print("\n⚠️ 注意：目前只是生成了合法公式，尚未通过 Qlib 计算 IC 进行优胜劣汰！")