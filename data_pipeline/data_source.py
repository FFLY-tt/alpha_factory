# data_source.py
import os
from tqdm import tqdm
import hashlib
import json
import pandas as pd
from qlib.data import D

# ================= 核心修复防线 (Windows 终极保命符) =================
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
custom_temp_dir = os.path.join(project_root, ".joblib_temp")
os.makedirs(custom_temp_dir, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = custom_temp_dir

# 【新增】强行压制所有底层多线程/多进程库的并发数
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# ===================================================================

import qlib
from qlib.config import REG_US, C  # 引入 Qlib 的全局配置字典 C
from qlib.data import D
import pandas as pd


def init_qlib_engine(data_dir: str = "data/qlib_data/us_data"):
    """
    启动引擎：把本地的 .bin 数据挂载到内存里
    """
    print(f"🔌 正在挂载 Qlib 数据源: {data_dir} ...")
    qlib.init(provider_uri=data_dir, region=REG_US)

    # 【新增】强制修改 Qlib 内部的并发进程数为 1
    C.n_jobs = 1

    print("✅ Qlib 引擎挂载成功！(已强制开启单核安全模式)")


def fetch_factor_data(formulas_dict: dict, instrument_pool: str = "sp500",
                      start_date: str = "2018-01-01", end_date: str = "2019-12-31",
                      batch_size: int = 1000) -> pd.DataFrame:
    """
    核心取数接口：分块将公式扔给 C++ 引擎，带有漂亮的进度条反馈
    注意：target_ret 使用 T+1 收益，因子请使用当日可得信息（不要提前做 Ref 延迟，
          如需实盘模拟，在管道中通过可选参数 lag=1 处理）。
    """
    # 强制加入“明日收益”作为预测目标
    formulas_dict["target_ret"] = "Ref($close, -1) / $close - 1"

    names = list(formulas_dict.keys())
    exprs = list(formulas_dict.values())
    total_formulas = len(names)

    print(f"🚀 准备调用底层 C++ 引擎，总计待计算公式: {total_formulas} 个")
    print(f"📦 采用分块计算模式，每个数据块大小: {batch_size}")

    instruments = D.instruments(market=instrument_pool)
    all_dfs = []

    for i in tqdm(range(0, total_formulas, batch_size), desc="⚙️ Qlib 算力矩阵计算中", unit="块"):
        batch_names = names[i: i + batch_size]
        batch_exprs = exprs[i: i + batch_size]
        df_batch = D.features(
            instruments=instruments,
            fields=batch_exprs,
            start_time=start_date,
            end_time=end_date,
            freq='day'
        )
        df_batch.columns = batch_names
        all_dfs.append(df_batch)

    print("\n🧩 计算完成，正在将所有数据块横向拼接到内存中，请稍候...")
    final_df = pd.concat(all_dfs, axis=1)

    # 警告：此处不再做全局 shift，因子对齐已由 target_ret 定义保证
    return final_df


# =========== 连通性自测 (Sanity Check) ===========
if __name__ == "__main__":
    data_path = os.path.join(project_root, "data", "qlib_data", "us_data")

    init_qlib_engine(data_dir=data_path)

    test_formulas = {
        "raw_close": "$close",
        "mean_5_close": "Mean($close, 5)",
        "price_momentum": "$close / Ref($close, 10)"
    }

    result_df = fetch_factor_data(
        formulas_dict=test_formulas,
        instrument_pool="sp500",
        start_date="2018-01-03",
        end_date="2018-01-15"
    )

    print("\n🎯 计算结果返回！前 10 行数据如下：")
    print(result_df.head(10))


def fetch_factor_data_with_cache(formulas_dict: dict, instrument_pool: str = "sp500",
                                 start_date: str = "2018-01-01", end_date: str = "2019-12-31",
                                 batch_size: int = 1000) -> pd.DataFrame:
    """
    带有 MD5 哈希缓存的取数接口：算过的数据秒级加载，没算过的再去调用底层
    """
    # 确保缓存目录存在
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. 制造这批公式的唯一指纹 (MD5 Hash)
    # 只要公式字典的内容、时间段、股票池一模一样，指纹就绝对相同
    dict_str = json.dumps(formulas_dict, sort_keys=True) + instrument_pool + start_date + end_date
    fingerprint = hashlib.md5(dict_str.encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"qlib_cache_{fingerprint}.parquet")

    # 2. 检查指纹：如果本地有，直接 0.1 秒极速加载！
    if os.path.exists(cache_file):
        print(f"  ⚡ 触发本地缓存！跳过 Qlib C++ 计算，极速加载历史数据: {cache_file}")
        return pd.read_parquet(cache_file)

    # 3. 如果本地没有，老老实实调用原版函数去算
    print("  🐌 未命中缓存，开始调用 Qlib 底层引擎...")
    df = fetch_factor_data(formulas_dict, instrument_pool, start_date, end_date, batch_size)

    # 4. 算完之后，立刻落盘存为 Parquet
    print(f"  💾 计算完毕，正在将中间矩阵保存至本地缓存: {cache_file}")
    df.to_parquet(cache_file)

    return df