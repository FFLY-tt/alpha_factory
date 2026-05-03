# src/genetic/subtree_library.py
"""
优良因子节点库：
- 项目 AST 设计为「有损」结构：BinaryNode / UnaryNode 不保存子节点引用，
  只保存计算后的 expr_str / depth / node_count。
- 因此无法做真正的子树遍历，改为以「整体节点」为单位存储和采样。
- 遗传操作（交叉/变异）在 genetic_engine.py 中以节点组合方式实现。
"""
import numpy as np
from typing import List, Optional
from src.ast.ast_nodes import FactorNode


class SubtreeLibrary:
    """
    优良节点库（原「子树库」，适配有损 AST）。
    每 N 代从精英个体中收录节点，按适应度加权采样，供变异操作调用。
    """

    def __init__(self, max_size: int = 100):
        self.max_size  = max_size
        self._nodes    : List[FactorNode] = []
        self._values   : List[float]      = []

    # ── 更新 ──────────────────────────────────────────────────────────────

    def update(
            self,
            elite_nodes  : List[FactorNode],
            elite_fitness: List[float]
    ) -> None:
        """将精英节点（整体）收录入库"""
        pool: dict = {n.expr_str: (n, v)
                      for n, v in zip(self._nodes, self._values)}

        for node, fitness in zip(elite_nodes, elite_fitness):
            expr = node.expr_str
            if expr in pool:
                pool[expr] = (node, max(pool[expr][1], fitness))
            else:
                pool[expr] = (node, fitness)

        sorted_items = sorted(pool.values(), key=lambda x: x[1], reverse=True)
        sorted_items = sorted_items[:self.max_size]

        self._nodes  = [item[0] for item in sorted_items]
        self._values = [item[1] for item in sorted_items]

    # ── 采样 ──────────────────────────────────────────────────────────────

    def sample(self) -> Optional[FactorNode]:
        """按 softmax 加权价值随机采样一个节点"""
        if not self._nodes:
            return None

        vals   = np.array(self._values, dtype=float)
        vals   = vals - vals.min() + 1e-6
        std    = vals.std()
        logits = vals / std if std > 0 else vals
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()

        idx = np.random.choice(len(self._nodes), p=probs)
        return self._nodes[idx]

    def __len__(self) -> int:
        return len(self._nodes)