# src/rules/rule_engine.py
from enum import Enum


class DimType(Enum):
    PRICE = 1
    VOLUME = 2
    RATIO = 4  # 比例/无量纲
    BOOLEAN = 5


class Ops:
    UNARY_TRANSFORM = {'Sign', 'Log', 'Abs'}
    # ✅ 把 Rank 加到时序算子里！机器会自动生成 Rank(x, 5), Rank(x, 10)
    UNARY_TS_SMOOTH = {'Mean', 'WMA', 'Max', 'Min', 'Std', 'Ref', 'Rank'}
    UNARY_ALL = UNARY_TRANSFORM | UNARY_TS_SMOOTH

    BIN_ADDITIVE = {'+', '-'}
    BIN_MULTIPLICATIVE = {'*', '/'}
    BIN_TS = {'Corr', 'Cov'}
    BIN_LOGICAL = {'>', '<', '=='}
    BIN_ALL = BIN_ADDITIVE | BIN_MULTIPLICATIVE | BIN_TS | BIN_LOGICAL

    TERNARY_ALL = {'If'}
    COMMUTATIVE = {'+', '*', 'Corr', 'Cov', '=='}


def validate_unary_op(op: str, child) -> DimType:
    if op not in Ops.UNARY_ALL:
        raise ValueError(f"🚨 未注册的一元算子: {op}")

    # 高级剪枝：无效的连续平滑 (如 Mean(Mean(x, 5), 5))
    if op in Ops.UNARY_TS_SMOOTH and child.name.startswith(op):
        raise ValueError(f"🚨 剪枝: 连续 {op} 平滑过度")

    if op in Ops.UNARY_TRANSFORM:
        return DimType.RATIO
    elif op in Ops.UNARY_TS_SMOOTH:
        # Rank 计算出的是排位数值，我们将其归类为无量纲比例
        if op == 'Rank':
            return DimType.RATIO
        return child.dim_type


def validate_binary_op(op: str, left, right) -> DimType:
    if op not in Ops.BIN_ALL:
        raise ValueError(f"🚨 未注册的二元算子: {op}")

    if left.expr_str == right.expr_str and op in {'-', '/', 'Corr', 'Cov'}:
        raise ValueError("🚨 剪枝: 自身运算退化")

    if op in Ops.COMMUTATIVE and left.name >= right.name:
        raise ValueError("🚨 剪枝: 交换律重复组合")

    if op in Ops.BIN_ADDITIVE:
        if left.dim_type != right.dim_type:
            raise ValueError(f"🚨 剪枝: 加减法量纲冲突 ({left.dim_type.name} vs {right.dim_type.name})")
        return left.dim_type

    elif op in Ops.BIN_MULTIPLICATIVE:
        if op == '/' and right.expr_str in {'0', '0.0'}:
            raise ValueError("🚨 剪枝: 静态分母为 0")
        return DimType.RATIO

    elif op in Ops.BIN_TS:
        if left.dim_type != right.dim_type:
            raise ValueError("🚨 剪枝: 跨量纲求相关性无意义")
        return DimType.RATIO

    elif op in Ops.BIN_LOGICAL:
        if left.dim_type != right.dim_type:
            raise ValueError("🚨 剪枝: 跨量纲比较无意义")
        return DimType.BOOLEAN


def validate_ternary_op(condition, true_node, false_node) -> DimType:
    if condition.dim_type != DimType.BOOLEAN:
        raise ValueError("🚨 剪枝: 条件必须是 BOOLEAN")
    if true_node.dim_type != false_node.dim_type:
        raise ValueError("🚨 剪枝: 分支量纲冲突")
    if true_node.expr_str == false_node.expr_str:
        raise ValueError("🚨 剪枝: 三元退化")
    return true_node.dim_type