# src/db/clickhouse_manager.py
import clickhouse_connect
import pandas as pd
import numpy as np
import hashlib
import datetime # 【新增导入】
from typing import List, Dict
from src.ast.ast_nodes import LeafNode
from src.rules.rule_engine import DimType

class ClickHouseManager:
    def __init__(self, host='localhost', port=8123, username='default', password='root', database='alpha_forge'):
        """初始化 ClickHouse 连接"""
        self.client = clickhouse_connect.get_client(
            host=host, port=port, username=username, password=password, database=database
        )
        print(f"🗄️ ClickHouse 客户端连接成功: {host}:{port}")

    # 🎯 【新增核心方法】：统一管理所有列名，拒绝硬编码，以后增删字段只改这里！
    def _get_column_names(self) -> List[str]:
        return [
            'batch_id', 'run_mode', 'depth', 'eval_stage', 'create_time', 'expr_str', 'expr_hash',
            'parent1_hash', 'parent2_hash', 'node_count', 'variant_tag',
            'is_needs_negation', 'is_high_turnover', 'is_sharpe_net', 'is_sharpe_gross',
            'is_turnover', 'is_ic', 'is_icir', 'is_fitness', 'is_ann_ret', 'is_max_dd',
            'oos_sharpe_net', 'oos_turnover', 'oos_ic', 'oos_fitness'
        ]

    def _calculate_expr_hash(self, expr_str: str) -> int:
        return int(hashlib.md5(expr_str.encode('utf-8')).hexdigest()[:15], 16)

    def _safe_float(self, val) -> float:
        if val is None: return 0.0
        try:
            f_val = float(val)
            if pd.isna(f_val) or np.isinf(f_val): return 0.0
            return f_val
        except (ValueError, TypeError):
            return 0.0

    def save_factor_batch(self, batch_id: int, run_mode: str, depth: int,
                          evaluation_results: Dict[str, Dict],
                          variant_tag: str = 'raw'):
        """
        批量保存因子评估结果 (包含显式时间注入)
        """
        if not evaluation_results:
            return

        rows = []
        # 获取当前 Python 运行的时间，代替数据库的 DEFAULT now()
        current_time = datetime.datetime.now()

        for expr_str, info in evaluation_results.items():
            is_r  = info.get("is_res", {})
            oos_r = info.get("oos_res", {})
            expr_hash = self._calculate_expr_hash(expr_str)

            node = info["node"]
            # 【关键】：从 Node 对象中提取父本哈希
            p1_hash = getattr(node, 'parent1_hash', 0)
            p2_hash = getattr(node, 'parent2_hash', 0)

            row = [
                int(batch_id),
                str(run_mode),
                int(depth),
                2 if oos_r else 1,
                current_time,       # 🎯 【完美修复】：把生成好的时间强行塞进去！
                str(expr_str),
                expr_hash,
                p1_hash,  # 记录父本1
                p2_hash,  # 记录父本2
                int(node.node_count),
                str(variant_tag),
                1 if is_r.get('needs_negation') else 0,
                1 if is_r.get('high_turnover') else 0,
                self._safe_float(is_r.get('sharpe_net')),
                self._safe_float(is_r.get('sharpe_gross')),
                self._safe_float(is_r.get('turnover')),
                self._safe_float(is_r.get('ic')),
                self._safe_float(is_r.get('icir')),
                self._safe_float(is_r.get('fitness')),
                self._safe_float(is_r.get('ann_ret')),
                self._safe_float(is_r.get('max_dd')),
                self._safe_float(oos_r.get('sharpe_net')),
                self._safe_float(oos_r.get('turnover')),
                self._safe_float(oos_r.get('ic')),
                self._safe_float(oos_r.get('fitness'))
            ]
            rows.append(row)

        column_names = [
            'batch_id', 'run_mode', 'depth', 'eval_stage', 'create_time', 'expr_str', 'expr_hash', # 🎯 列名新增 create_time
            'parent1_hash', 'parent2_hash', 'node_count', 'variant_tag',
            'is_needs_negation', 'is_high_turnover', 'is_sharpe_net', 'is_sharpe_gross',
            'is_turnover', 'is_ic', 'is_icir', 'is_fitness', 'is_ann_ret', 'is_max_dd',
            'oos_sharpe_net', 'oos_turnover', 'oos_ic', 'oos_fitness'
        ]

        self.client.insert('factors_wide', rows, column_names=column_names)
        print(f"✅ 成功将 {len(rows)} 个因子评估数据同步至 ClickHouse (Batch: {batch_id}, Depth: {depth})")

    def get_elite_factors(self, batch_id: int, depth: int, limit: int = 60) -> List[LeafNode]:
        query = f"SELECT expr_str, is_fitness FROM factors_wide WHERE batch_id = {batch_id} AND depth = {depth} ORDER BY is_fitness DESC LIMIT {limit}"
        result = self.client.query(query)

        rescued_nodes = []
        for i, row in enumerate(result.result_rows):
            node = LeafNode(name=f"db_{batch_id}_d{depth}_{i}", expr_str=row[0], dim_type=DimType.RATIO)
            node.depth = depth
            rescued_nodes.append(node)
        return rescued_nodes

    def save_initial_seeds(self, batch_id: int, seed_nodes: List[LeafNode]):
        """
        [新增] 动态写入 T1 原始指标。
        用于追溯：即使指标值全为 Null，也要记录其 Hash 和名称。
        """
        rows = []
        current_time = datetime.datetime.now()
        for node in seed_nodes:
            expr_hash = self._calculate_expr_hash(node.expr_str)
            row = [
                batch_id, "T1_SEED", 1, 1, current_time,
                node.expr_str, expr_hash, 0, 0, 1, "seed_indicator",
                0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # IS Metrics 为空
                0.0, 0.0, 0.0, 0.0  # OOS Metrics 为空
            ]
            rows.append(row)

        column_names = self._get_column_names()
        self.client.insert('factors_wide', rows, column_names=column_names)
        print(f"🧬 [DB] 已动态记录 {len(seed_nodes)} 个 T1 原始种子指标。")