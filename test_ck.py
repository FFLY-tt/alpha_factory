# test_ck.py
from src.db.clickhouse_manager import ClickHouseManager
import traceback


def run_test():
    print("=" * 60)
    print("🔍 ClickHouse 写入诊断测试")
    print("=" * 60)

    # 1. 初始化连接
    try:
        db = ClickHouseManager()
        client = db.client
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # ==========================================
    # Test 1: 原生 SQL 字符串写入 (最安全的写入方式)
    # ==========================================
    print("\n▶️ Test 1: 执行原生 SQL INSERT...")
    try:
        client.command("""
            INSERT INTO factors_wide (
                batch_id, run_mode, depth, expr_str, expr_hash, is_fitness
            ) VALUES (
                999, 'DIAGNOSTIC_TEST', 1, '$close', 123456789, 1.5
            )
        """)
        print("   ✅ Test 1 成功！(说明建表 DDL 没问题，数据库运转正常)")
    except Exception as e:
        print(f"   ❌ Test 1 失败: {e}")
        traceback.print_exc()

    # ==========================================
    # Test 2: 模拟流水线字典写入 (复现 1001 错误)
    # ==========================================
    print("\n▶️ Test 2: 模拟 AlphaForge 数据结构调用 save_factor_batch...")

    # 模拟 AST Node
    class MockNode:
        def __init__(self):
            self.node_count = 5

    # 模拟 evaluate_single_factor_comprehensive 返回的格式
    mock_eval_results = {
        "Mean($close, 5)": {
            "is_res": {
                "needs_negation": False,
                "high_turnover": False,
                "sharpe_net": 1.5,
                "sharpe_gross": 1.6,
                "turnover": 0.5,
                "ic": 0.05,
                "icir": 0.5,
                "fitness": 1.2,
                "ann_ret": 0.15,
                "max_dd": 0.05
            },
            "oos_res": {
                "sharpe_net": 1.2,
                "turnover": 0.5,
                "ic": 0.04,
                "fitness": 1.0
            },
            "node": MockNode()
        }
    }

    try:
        db.save_factor_batch(
            batch_id=888,
            run_mode="TEST",
            depth=3,
            evaluation_results=mock_eval_results,
            variant_tag="test_tag"
        )
        print("   ✅ Test 2 成功！(说明 ClickHouseManager 类型清洗没问题)")
    except Exception as e:
        print(f"   ❌ Test 2 失败 (准备抓取错误根源): {e}")
        # traceback.print_exc()


if __name__ == "__main__":
    run_test()