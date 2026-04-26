import os
import yaml
import qlib
from qlib.tests.data import GetData


def update_qlib_data():
    config_path = "config/alpha_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = config["env"]["data_dir"]
    region = config["env"]["region"]

    os.makedirs(data_dir, exist_ok=True)

    print(f"🚀 开始从 Yahoo Finance 同步 {region.upper()} 市场数据到 {data_dir}...")
    print("⏳ 这可能需要 10-20 分钟，请耐心等待...")

    try:
        GetData().qlib_data(
            target_dir=data_dir,
            region=region,
            interval="1d",
            delete_old=False
        )
        print("\n✅ 美股数据同步完成！")
    except Exception as e:
        print(f"\n❌ 数据同步失败: {e}")


if __name__ == "__main__":
    update_qlib_data()