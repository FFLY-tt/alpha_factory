# src/mining/beam_search.py
import random
from typing import List
from src.rules.rule_engine import Ops, DimType
from src.ast.ast_nodes import FactorNode, LeafNode, UnaryNode, BinaryNode, TernaryNode
from src.mining.seed_library import load_seed_factors


# beam_search,负责加载种子库，通过一元，二元算子暴力交叉变异出下一代公式，同时调用规则引擎提出不合法的垃圾
# 1.装填弹药 (_load_and_convert_seeds)：
# 开局时，把 seed_library.py 里那 62 个初始指标（比如 $close, $volume, mom_intraday）加载进来，包装成最初始的叶子节点（LeafNode）。
# 2.盲目变异（广度探索） (generate_candidates)：
# 这是它原本就在干的事。 它是引擎的主力。它把手里的节点两两配对，尝试加上 +, -, Corr, Max 等算子。遇到量纲冲突（比如价格加成交量），rule_engine.py 会静默拦截。最终它会从几万种组合中，盲目但符合物理定律地生成出 3000 个合法的候选因子，丢给后面的法庭去评判。
# 3.定向定做（深度挖掘） (generate_market_state_variants & generate_smoothing_variants)：
# 这就是我们刚才搬进来的新能力。 当后面的法庭已经评判出“这几个因子是好苗子”时，调度中心会把这几个精英重新送回兵工厂，要求兵工厂对它们进行“定向改装”（套上平滑算子，或者加上大盘条件）。
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

    # 大盘状态衍生，强行在主流程里造出了“牛市版”和“高波动版”因子
    def generate_market_state_variants(self, base_nodes: List[FactorNode]) -> List[FactorNode]:
        """
        模块四衍生：为基础因子生成大盘状态（牛市/高波动）择时版本
        """
        MKT_RET_EXPR = "Mean($close / Ref($close, 1) - 1, 20)"
        VOL_EXPR = "Std($close / Ref($close, 1) - 1, 20)"

        derived = []
        # 将外部宏观条件转化为合法的 AST 叶子节点
        mkt_ret_node = LeafNode("mkt_ret_20d", f"Ref({MKT_RET_EXPR}, 1)", DimType.RATIO)
        zero_node = LeafNode("zero", "0", DimType.RATIO)
        vol_now = LeafNode("vol_20d", f"Ref({VOL_EXPR}, 1)", DimType.RATIO)
        vol_ma = LeafNode("vol_ma_60d", f"Ref(Mean({VOL_EXPR}, 60), 1)", DimType.RATIO)

        # 构建逻辑判断节点
        bull_cond = BinaryNode(">", mkt_ret_node, zero_node)
        high_vol_cond = BinaryNode(">", vol_now, vol_ma)

        for node in base_nodes:
            if node.dim_type == DimType.BOOLEAN:
                continue

            zero_factor = LeafNode("zero_factor", "0", node.dim_type)

            # 1. 衍生：牛市择时版
            try:
                bn = TernaryNode(bull_cond, node, zero_factor)
                bn.name = f"BULL_{node.name[:30]}"
                derived.append(bn)
            except ValueError:
                pass

            # 2. 衍生：高波动反转版
            if node.dim_type == DimType.RATIO:
                try:
                    neg_one = LeafNode("neg_one", "-1", DimType.RATIO)
                    rev = BinaryNode("*", neg_one, node)
                    hv = TernaryNode(high_vol_cond, rev, node)
                    hv.name = f"HV_REV_{node.name[:25]}"
                    derived.append(hv)
                except ValueError:
                    pass

        print(f"   🌐 [引擎] 定向衍生：{len(base_nodes)} 个基础因子 → 产出 {len(derived)} 个状态择时因子")
        return derived

    # 平滑变体生成：通过字符串替换 Mean({expr}, 5) 的方式，强行批量生成衍生因子
    def generate_smoothing_variants(self, base_nodes: List[FactorNode], windows: List[int] = None) -> List[FactorNode]:
        """
        自动精调衍生：为优秀的因子生成极其密集的平滑变体（替代原来的字符串 replace）
        """
        windows = windows or [5, 10, 12, 15, 20]
        derived = []

        for node in base_nodes:
            if node.dim_type == DimType.BOOLEAN:
                continue
            for d in windows:
                try:
                    # 严格使用 AST 生成 Mean 变体
                    mean_node = UnaryNode("Mean", node, d)
                    derived.append(mean_node)

                    # 严格使用 AST 生成 WMA 变体 (对应 Qlib 的 decay_linear)
                    wma_node = UnaryNode("WMA", node, d)
                    derived.append(wma_node)
                except ValueError:
                    pass

        print(f"   🔧 [引擎] 细粒度精调：为 {len(base_nodes)} 个因子生成了 {len(derived)} 个平滑变体")
        return derived

    def generate_crossover_candidates(self, pool_a, pool_b, max_eval: int, target_depth: int):
        """
        【真·遗传规划 (GP) 引擎】：XC 模式专属，基于 AST 子树交换的交叉算子
        """
        import random
        import hashlib
        from src.ast.ast_nodes import LeafNode
        from src.rules.rule_engine import DimType

        # 🎯 引入我们手搓的 AST 手术刀
        try:
            from src.ast.ast_crossover import parse_formula, crossover_trees
        except ImportError:
            print("❌ 找不到 src.ast.ast_crossover，请确保该文件存在！")
            return []

        candidates = []
        attempts = 0
        seen_exprs = set()  # 局部的防重哈希表

        print(f"   🔬 正在基因实验室中进行 AST 子树重组 (最大允许深度: {target_depth})...")

        # 尝试次数上限设为 max_eval 的 10 倍，因为有些杂交可能不合法被废弃
        while len(candidates) < max_eval and attempts < max_eval * 10:
            attempts += 1

            # 1. 锦标赛选择 (Tournament Selection) 的极简版：
            # 因为传入的 pool 已经是按 Fitness 降序排列的精英了，
            # 为了保证多样性，我们在精英池中随机挑选双亲。
            parent_a = random.choice(pool_a)
            parent_b = random.choice(pool_b)

            if parent_a.expr_str == parent_b.expr_str:
                continue

            # 2. 字符串转 AST 树
            tree_a = parse_formula(parent_a.expr_str)
            tree_b = parse_formula(parent_b.expr_str)

            # 3. 核心基因重组 (带基因锁)
            child_tree_1, child_tree_2 = crossover_trees(tree_a, tree_b, max_depth=target_depth)

            # 4. AST 树转回 Qlib 字符串
            expr_1 = child_tree_1.to_string()
            expr_2 = child_tree_2.to_string()

            # 5. 验证与收编入库
            for expr in [expr_1, expr_2]:
                if len(candidates) >= max_eval:
                    break

                # 防重：不要和父本完全一样，也不要和刚生成的兄弟一样
                if expr in (parent_a.expr_str, parent_b.expr_str) or expr in seen_exprs:
                    continue

                seen_exprs.add(expr)

                new_node = LeafNode(name=f"XC_GP_gen_{attempts}", expr_str=expr, dim_type=DimType.RATIO)

                # 🎯 极其重要的血统追踪！记录这是哪两个公式杂交出来的
                new_node.parent1_hash = int(hashlib.md5(parent_a.expr_str.encode('utf-8')).hexdigest()[:15], 16)
                new_node.parent2_hash = int(hashlib.md5(parent_b.expr_str.encode('utf-8')).hexdigest()[:15], 16)
                new_node.node_count = expr.count('(') + 1  # 简单估算节点数

                candidates.append(new_node)

        print(f"   🧬 基因重组完成：碰撞 {attempts} 次，成功培育合法超级后代 {len(candidates)} 个")
        return candidates


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