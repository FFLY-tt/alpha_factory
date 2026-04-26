# src/ast/ast_nodes.py
from src.rules.rule_engine import DimType, validate_binary_op, validate_unary_op, validate_ternary_op


class FactorNode:
    """所有因子节点的基类"""

    def __init__(self, name: str, expr_str: str, dim_type: DimType, depth: int, node_count: int):
        self.name = name  # 短名称，用于日志打印 (如 mean_5_close)
        self.expr_str = expr_str  # 真正的 Qlib 表达式 (如 Mean($close, 5))
        self.dim_type = dim_type  # 量纲
        self.depth = depth  # 树高 (防止无限嵌套)
        self.node_count = node_count  # 节点总数 (WorldQuant 约束)

    def __repr__(self):
        return f"<{self.dim_type.name}> {self.expr_str}"


class LeafNode(FactorNode):
    """
    叶子节点：也就是我们的 E100 种子库因子。
    它们是树生长的起点，深度永远为 1。
    """

    def __init__(self, name: str, expr_str: str, dim_type: DimType):
        super().__init__(name, expr_str, dim_type, depth=1, node_count=1)


class UnaryNode(FactorNode):
    """
    一元操作符节点 (如 Rank(x), Mean(x, 5))
    """

    def __init__(self, op: str, child: FactorNode, *args):
        # 校验并获取新量纲
        new_dim = validate_unary_op(op, child)

        # 拼装 Qlib 字符串。如果有 args (比如天数 5)，需要拼进去
        if args:
            params = ", ".join(map(str, args))
            expr_str = f"{op}({child.expr_str}, {params})"
            name = f"{op}_{args[0]}_{child.name}"
        else:
            expr_str = f"{op}({child.expr_str})"
            name = f"{op}_{child.name}"

        super().__init__(
            name=name,
            expr_str=expr_str,
            dim_type=new_dim,
            depth=child.depth + 1,
            node_count=child.node_count + 1
        )


class BinaryNode(FactorNode):
    """
    二元操作符节点 (如 A + B, Corr(A, B, 10))
    """

    def __init__(self, op: str, left: FactorNode, right: FactorNode, *args):
        # 【核心剪枝】：如果非法，这里会抛出 ValueError，波束搜索就能直接跳过这个组合
        new_dim = validate_binary_op(op, left, right)

        if op in ['+', '-', '*', '/', '>', '<', '==']:
            # 四则运算语法：(A + B)
            expr_str = f"({left.expr_str} {op} {right.expr_str})"
            name = f"({left.name}{op}{right.name})"
        else:
            # 函数式二元算子：Corr(A, B, 10)
            if args:
                params = ", ".join(map(str, args))
                expr_str = f"{op}({left.expr_str}, {right.expr_str}, {params})"
                name = f"{op}_{args[0]}_{left.name}_{right.name}"
            else:
                expr_str = f"{op}({left.expr_str}, {right.expr_str})"
                name = f"{op}_{left.name}_{right.name}"

        super().__init__(
            name=name,
            expr_str=expr_str,
            dim_type=new_dim,
            depth=max(left.depth, right.depth) + 1,
            node_count=left.node_count + right.node_count + 1
        )


class TernaryNode(FactorNode):
    """
    三元操作符节点：也就是 Qlib 里的 If(Condition, True_Expr, False_Expr)
    """

    def __init__(self, condition: FactorNode, true_node: FactorNode, false_node: FactorNode):
        # 呼叫规则引擎：如果非法，瞬间剪枝抛出异常
        new_dim = validate_ternary_op(condition, true_node, false_node)

        # 拼装 Qlib 认可的三元字符串
        expr_str = f"If({condition.expr_str}, {true_node.expr_str}, {false_node.expr_str})"

        # 构建节点名称 (尽量紧凑，防止波束搜索时名字过长)
        name = f"If_{condition.name}_T({true_node.name})_F({false_node.name})"

        super().__init__(
            name=name,
            expr_str=expr_str,
            dim_type=new_dim,
            # 树高是三个子树的最大值 + 1
            depth=max(condition.depth, true_node.depth, false_node.depth) + 1,
            # 节点数全加起来 + 1
            node_count=condition.node_count + true_node.node_count + false_node.node_count + 1
        )


# =========== 神奇的见证时刻 (Sanity Check) ===========
if __name__ == "__main__":
    print("=" * 60)
    print("🌟 AlphaForge AST 树生成器自测")
    print("=" * 60)

    # 模拟从 E100 取出的两个叶子节点
    node_close = LeafNode("raw_close", "$close", DimType.PRICE)
    node_vol = LeafNode("raw_volume", "$volume", DimType.VOLUME)

    print(f"1. 诞生叶子节点 A: {node_close}")
    print(f"2. 诞生叶子节点 B: {node_vol}")

    # 尝试合法组合：对价格套上均线，对成交量套上排序
    node_mean = UnaryNode("Mean", node_close, 5)
    node_rank = UnaryNode("Rank", node_vol)
    print(f"\n3. 合法生长 (价格变均线): {node_mean} | 树高: {node_mean.depth}")
    print(f"4. 合法生长 (成交量变排名): {node_rank} | 树高: {node_rank.depth}")

    # 尝试合法组合：求价格与成交量的相关性
    node_corr = BinaryNode("Corr", node_close, node_vol, 10)
    print(f"\n5. 合法组装 (量价背离因子): {node_corr} | 树高: {node_corr.depth}")

    # 🔥 尝试触发防御机制 (物理学大忌：价格 + 成交量)
    print("\n6. 试图挑战物理学定律：(价格 + 成交量)...")
    try:
        bad_node = BinaryNode("+", node_close, node_vol)
    except Exception as e:
        print(e)

    # 三元运算符
    node_close = LeafNode("raw_close", "$close", DimType.PRICE)
    node_vwap = LeafNode("raw_vwap", "$vwap", DimType.PRICE)
    node_vol = LeafNode("raw_volume", "$volume", DimType.VOLUME)

    # 1. 造一个布尔条件：今天收盘价是否大于均价 (日内强势)
    cond_node = BinaryNode(">", node_close, node_vwap)
    print(f"👉 条件节点生成: {cond_node} | 量纲: {cond_node.dim_type.name}")

    # 2. 造两个执行分支 (量纲都是 RATIO)
    # True 分支：放量动量
    true_branch = BinaryNode("/", node_vol, LeafNode("ref_vol", "Ref($volume, 5)", DimType.VOLUME))
    # False 分支：缩量动量
    false_branch = BinaryNode("/", LeafNode("ref_vol", "Ref($volume, 5)", DimType.VOLUME), node_vol)

    # 3. 完美结合成三元状态机！
    regime_factor = TernaryNode(cond_node, true_branch, false_branch)
    print(f"\n🌟 终极三元因子诞生: {regime_factor}")
    print(f"📊 该因子量纲: {regime_factor.dim_type.name} | 树高: {regime_factor.depth}")