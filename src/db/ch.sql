CREATE DATABASE IF NOT EXISTS alpha_forge;

CREATE TABLE IF NOT EXISTS alpha_forge.factors_wide
(
    -- ================= 1. 元数据与调度标记 =================
    `batch_id` UInt32 COMMENT '批次ID',
    `run_mode` String COMMENT '执行模式: XA/XB/XC',
    `depth` UInt8 COMMENT '当前节点层级(T1-T5)',
    `eval_stage` UInt8 DEFAULT 1 COMMENT '生命周期: 1=IS完结, 2=OOS完结, 3=WF完结',
    `create_time` DateTime COMMENT '创建时间',

    -- ================= 2. 基因与血统追踪 =================
    `expr_str` String COMMENT '因子完整表达式',
    `expr_hash` UInt64 COMMENT '表达式的 CityHash64',
    `parent1_hash` UInt64 DEFAULT 0 COMMENT '父本1的哈希值',
    `parent2_hash` UInt64 DEFAULT 0 COMMENT '父本2的哈希值',
    `node_count` UInt16 COMMENT '语法树节点总数',
    `variant_tag` String DEFAULT 'raw' COMMENT '变体标签(如 wma_10, bull_timing)',

    -- ================= 3. 样本内评估指标 (IS) =================
    `is_needs_negation` UInt8 COMMENT '极其关键: 1=需取反(-1*), 0=原方向',
    `is_high_turnover` UInt8 COMMENT '1=触发高换手熔断, 0=正常',
    `is_sharpe_net` Float32 COMMENT 'IS 扣费后修正夏普',
    `is_sharpe_gross` Float32 COMMENT 'IS 扣费前毛夏普',
    `is_turnover` Float32 COMMENT 'IS 日均换手率',
    `is_ic` Float32 COMMENT 'IS Rank IC均值',
    `is_icir` Float32 COMMENT 'IS ICIR稳定性',
    `is_fitness` Float32 COMMENT 'IS 复合打分',
    `is_ann_ret` Float32 COMMENT 'IS 年化收益率',
    `is_max_dd` Float32 COMMENT 'IS 最大回撤',

    -- ================= 4. 样本外评估指标 (OOS) =================
    `oos_sharpe_net` Float32 DEFAULT 0.0,
    `oos_turnover` Float32 DEFAULT 0.0,
    `oos_ic` Float32 DEFAULT 0.0,
    `oos_fitness` Float32 DEFAULT 0.0,

    -- ================= 5. 滚动验证指标 (Walk-Forward) =================
    `wf_mean_sharpe` Float32 DEFAULT 0.0 COMMENT 'WF 平均净夏普',
    `wf_min_sharpe` Float32 DEFAULT 0.0 COMMENT 'WF 最差折夏普',
    `wf_stability` Float32 DEFAULT 0.0 COMMENT 'WF 夏普稳定性(Mean/Std)'
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(create_time)
ORDER BY (batch_id, depth, expr_hash)
SETTINGS index_granularity = 8192;