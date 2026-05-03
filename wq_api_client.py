# wq_api_client.py
"""
WorldQuant Brain API 客户端
放在 alpha_factory 根目录，供 wq_submit.py 调用。

使用前在根目录创建 .env 文件，内容：
    WQ_USERNAME=你的WQ邮箱
    WQ_PASSWORD=你的WQ密码
"""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
USERNAME = os.getenv('WQ_USERNAME')
PASSWORD = os.getenv('WQ_PASSWORD')

BASE_URL = "https://api.worldquantbrain.com"


class BrainAPIClient:
    def __init__(self):
        self.session = requests.Session()
        self.authenticate()

    def authenticate(self):
        print("🔌 正在连接 WorldQuant Brain 服务器...")
        self.session.auth = (USERNAME, PASSWORD)
        response = self.session.post(f"{BASE_URL}/authentication")
        if response.status_code == 201:
            print("✅ 鉴权成功！")
        else:
            raise Exception(f"❌ 登录失败: {response.text}")

    def simulate_alpha(
            self,
            expression: str,
            delay: int = 1,
            decay: int = 0,
            universe: str = "TOP3000",
            neutralization: str = "SUBINDUSTRY",
            truncation: float = 0.08,
    ) -> dict:
        """
        提交因子表达式到 WQ 云端回测。

        参数说明：
            delay          : 1（T+1 成交，标准设置）
            decay          : 0（不衰减）
            universe       : TOP3000 / TOP500 / TOP1000
            neutralization : SUBINDUSTRY / INDUSTRY / MARKET / NONE
            truncation     : 单只股票最大权重上限
        """
        print(f"\n🚀 提交因子: {expression[:80]}{'...' if len(expression) > 80 else ''}")

        payload = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": "USA",
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": neutralization,
                "truncation": truncation,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "ON",
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": expression,
        }

        response = self.session.post(f"{BASE_URL}/simulations", json=payload)

        if response.status_code == 201:
            sim_id = response.headers.get('Location', '').split('/')[-1]
            print(f"   任务 ID: {sim_id}，等待云端计算...")
            return self._poll_result(sim_id)
        else:
            print(f"   ❌ 提交失败: {response.status_code} {response.text[:200]}")
            return None

    def _poll_result(self, sim_id: str) -> dict:
        status_url = f"{BASE_URL}/simulations/{sim_id}"
        while True:
            try:
                data = self.session.get(status_url).json()
            except Exception as e:
                print(f"   ⚠️ 查询异常: {e}，5秒后重试...")
                time.sleep(5)
                continue

            status = data.get('status', 'UNKNOWN')
            progress = data.get('progress', 0) * 100

            if status == "ERROR":
                print(f"\n   ❌ 云端回测报错: {data.get('message', data)}")
                return None

            elif status in ("COMPLETE", "COMPLETED", "WARNING"):
                alpha_id = data.get('alpha')
                print(f"\n   ✅ 回测完成（{status}）")

                alpha_resp = self.session.get(f"{BASE_URL}/alphas/{alpha_id}")
                if alpha_resp.status_code == 200:
                    alpha_data = alpha_resp.json()
                    is_metrics = alpha_data.get('is', {})
                    return {
                        "sharpe"   : is_metrics.get('sharpe'),
                        "fitness"  : is_metrics.get('fitness'),
                        "turnover" : is_metrics.get('turnover'),
                        "alpha_id" : alpha_id,
                        "status"   : status,
                    }
                else:
                    print(f"   ❌ 拉取成绩失败: {alpha_resp.status_code}")
                    return None

            else:
                print(f"\r   状态: {status} | 进度: {progress:.1f}%...", end="", flush=True)
                time.sleep(5)