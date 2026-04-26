# check_ic.py
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from data_pipeline.data_source import init_qlib_engine, fetch_factor_data
from src.evaluation.metrics_calc import calculate_ic_stability

if __name__ == '__main__':
    cur = os.path.dirname(os.path.abspath(__file__))
    init_qlib_engine(os.path.join(cur, 'data', 'qlib_data', 'us_data'))

    df = fetch_factor_data({'real': '$close'}, 'sp500', '2020-01-01', '2021-12-31')

    np.random.seed(42)
    df['rand'] = np.random.randn(len(df))

    ic, icir = calculate_ic_stability(df, 'rand')
    print(f'\n随机因子 IC={ic:.4f}, ICIR={icir:.4f}')
    print('✅ 无未来数据泄漏' if abs(ic) < 0.01 else '❌ 有未来数据！')