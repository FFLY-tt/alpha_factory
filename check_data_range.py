# check_data_range.py
import warnings
warnings.filterwarnings("ignore")
import os

if __name__ == "__main__":
    from data_pipeline.data_source import init_qlib_engine, fetch_factor_data

    cur = os.path.dirname(os.path.abspath(__file__))
    init_qlib_engine(os.path.join(cur, "data", "qlib_data", "us_data"))

    # 取一个最简单的因子，只要能拿到日期范围就行
    df = fetch_factor_data(
        {"test": "$close"},
        instrument_pool="sp500",
        start_date="2010-01-01",
        end_date="2030-01-01"   # 故意设得很宽，让 qlib 返回实际有数据的范围
    )

    dates = df.index.get_level_values('datetime')
    print(f"\n📅 数据日期范围：")
    print(f"   最早日期: {dates.min().date()}")
    print(f"   最晚日期: {dates.max().date()}")
    print(f"   总交易日数: {dates.nunique()} 天")
    print(f"\n💡 建议：")
    print(f"   样本内（IS）: 最早日期 ~ 最晚日期前 2-3 年")
    print(f"   样本外（OOS）: 最晚日期前 2-3 年 ~ 最晚日期")