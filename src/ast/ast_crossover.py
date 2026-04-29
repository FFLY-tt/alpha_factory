# src/ast/ast_crossover.py
import random
import copy


class ASTNode:
    """抽象语法树 (AST) 的基本节点"""

    def __init__(self, value: str, children=None):
        self.value = value.strip()  # 算子名 (如 Ts_Mean) 或 叶子变量 (如 $close)
        self.children = children if children is not None else []

    def to_string(self) -> str:
        """递归地将整棵树还原为 Qlib 认识的字符串公式"""
        if not self.children:
            return self.value
        args = ", ".join(child.to_string() for child in self.children)
        return f"{self.value}({args})"

    def get_depth(self) -> int:
        """获取当前节点的深度 (用于实现你说的 T_max(N, M) 基因锁)"""
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)


def parse_formula(formula: str) -> ASTNode:
    """
    【核心解析器】：将平面的字符串解析成 3D 的树结构
    例如: "Div(Mean($close, 5), $vwap)" -> AST 树
    """
    formula = formula.strip()

    # Base Case: 如果没有括号，说明是叶子节点 (原始变量或常数)
    if "(" not in formula:
        return ASTNode(formula)

    # 找到第一层函数名和其包裹的参数字符串
    first_paren_idx = formula.find("(")
    last_paren_idx = formula.rfind(")")

    func_name = formula[:first_paren_idx]
    args_str = formula[first_paren_idx + 1:last_paren_idx]

    # 根据逗号切分参数，但必须避开嵌套的括号！
    args = []
    bracket_level = 0
    current_arg = ""

    for char in args_str:
        if char == '(':
            bracket_level += 1
        elif char == ')':
            bracket_level -= 1

        # 只有在最外层遇到的逗号，才是真正的参数分隔符
        if char == ',' and bracket_level == 0:
            args.append(current_arg)
            current_arg = ""
        else:
            current_arg += char

    if current_arg:
        args.append(current_arg)

    # 递归解析子节点
    children = [parse_formula(arg) for arg in args]
    return ASTNode(func_name, children)


def get_all_nodes(root: ASTNode) -> list:
    """扁平化提取树中所有的节点（供抛骰子切片用）"""
    nodes = [root]
    for child in root.children:
        nodes.extend(get_all_nodes(child))
    return nodes


def crossover_trees(tree_a: ASTNode, tree_b: ASTNode, max_depth: int = None):
    """
    【基因重组手术台】：将两棵树切开，互换一个子树，然后缝合。
    """
    # 1. 深度克隆，绝对不能污染原始的精英父本
    clone_a = copy.deepcopy(tree_a)
    clone_b = copy.deepcopy(tree_b)

    nodes_a = get_all_nodes(clone_a)
    nodes_b = get_all_nodes(clone_b)

    # 2. 避免直接交换整棵树 (根节点)，那样没有意义，退化为 A 和 B 互换位置
    if len(nodes_a) > 1: nodes_a = nodes_a[1:]
    if len(nodes_b) > 1: nodes_b = nodes_b[1:]

    # 最多尝试 10 次，寻找满足深度约束的合法杂交点
    for _ in range(10):
        point_a = random.choice(nodes_a)
        point_b = random.choice(nodes_b)

        # 3. 核心基因互换 (Python 的引用机制让这个操作极其优雅)
        point_a.value, point_b.value = point_b.value, point_a.value
        point_a.children, point_b.children = point_b.children, point_a.children

        # 4. 基因锁检查 (如果你规定了 T_max 深度上限)
        if max_depth is None or (clone_a.get_depth() <= max_depth and clone_b.get_depth() <= max_depth):
            return clone_a, clone_b

        # 如果超标了，把基因换回去，重新抛骰子！
        point_a.value, point_b.value = point_b.value, point_a.value
        point_a.children, point_b.children = point_b.children, point_a.children

    # 如果实在找不到合法的杂交点，就原样返回
    return clone_a, clone_b


# =====================================================================
# 🧪 实验室：见证奇迹的时刻
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧬 AlphaForge 基因重组实验室")
    print("=" * 60)

    # 假设这是你在 XA 模式跑出来的两个 T3 精英公式
    str_a = "Div(Ts_Mean($close, 5), $vwap)"  # 动量特征因子
    str_b = "Ts_Max(Std($volume, 10), 20)"  # 波动率特征因子

    print(f"👨‍🏫 父本 A (深度 {parse_formula(str_a).get_depth()}): {str_a}")
    print(f"👩‍🏫 父本 B (深度 {parse_formula(str_b).get_depth()}): {str_b}")
    print("-" * 60)

    tree_a = parse_formula(str_a)
    tree_b = parse_formula(str_b)

    # 我们让它们交配 5 次，看看能生出什么怪物！
    for i in range(1, 6):
        # 强制后代的深度不得超过双亲的最大深度 (这就是你说的 T_max(N, M))
        max_allowed_depth = max(tree_a.get_depth(), tree_b.get_depth())

        child_1, child_2 = crossover_trees(tree_a, tree_b, max_depth=max_allowed_depth)

        print(f"\n🔄 第 {i} 次杂交突变:")
        print(f"   👼 后代 1: {child_1.to_string()}")
        print(f"   👼 后代 2: {child_2.to_string()}")