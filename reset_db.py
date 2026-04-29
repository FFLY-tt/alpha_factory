# reset_db.py
import clickhouse_connect

def reset_and_test():
    print("=" * 60)
    print("🔌 连接 ClickHouse...")
    try:
        client = clickhouse_connect.get_client(host='localhost', port=8123, username='default', password='root')
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    print("💥 正在核平并重建数据库表...")
    client.command("CREATE DATABASE IF NOT EXISTS alpha_forge")
    client.command("DROP TABLE IF EXISTS alpha_forge.factors_wide")

    # 🎯 极致纯净版 DDL：
    # 1. 移除了所有花哨的 DEFAULT
    # 2. 移除了引发崩溃的 PARTITION BY toYYYYMM(create_time)
    ddl = """
    CREATE TABLE alpha_forge.factors_wide
    (
        `batch_id` UInt32,
        `run_mode` String,
        `depth` UInt8,
        `eval_stage` UInt8,
        `create_time` DateTime,
        `expr_str` String,
        `expr_hash` UInt64,
        `parent1_hash` UInt64,
        `parent2_hash` UInt64,
        `node_count` UInt16,
        `variant_tag` String,
        `is_needs_negation` UInt8,
        `is_high_turnover` UInt8,
        `is_sharpe_net` Float32,
        `is_sharpe_gross` Float32,
        `is_turnover` Float32,
        `is_ic` Float32,
        `is_icir` Float32,
        `is_fitness` Float32,
        `is_ann_ret` Float32,
        `is_max_dd` Float32,
        `oos_sharpe_net` Float32,
        `oos_turnover` Float32,
        `oos_ic` Float32,
        `oos_fitness` Float32
    )
    ENGINE = MergeTree()
    ORDER BY (batch_id, depth, expr_hash)
    SETTINGS index_granularity = 8192
    """
    client.command(ddl)
    print("✅ 建表成功！数据库已恢复纯净状态。")

    print("\n▶️ 正在进行 25 字段全量插入测试...")
    try:
        # 模拟引擎的真实写入行为
        client.command("""
            INSERT INTO alpha_forge.factors_wide (
                batch_id, run_mode, depth, eval_stage, create_time, expr_str, expr_hash,
                parent1_hash, parent2_hash, node_count, variant_tag,
                is_needs_negation, is_high_turnover, is_sharpe_net, is_sharpe_gross,
                is_turnover, is_ic, is_icir, is_fitness, is_ann_ret, is_max_dd,
                oos_sharpe_net, oos_turnover, oos_ic, oos_fitness
            ) VALUES (
                1, 'TEST', 3, 2, now(), '$close', 123456789,
                0, 0, 1, 'raw',
                0, 0, 1.5, 1.6, 
                0.1, 0.05, 0.5, 1.2, 0.2, 0.1,
                1.2, 0.1, 0.04, 1.0
            )
        """)
        print("🎉 插入测试成功！数据库底层 Bug 已被彻底消灭！")
    except Exception as e:
        print(f"❌ 插入测试失败: {e}")

if __name__ == "__main__":
    reset_and_test()