# wq_submit.py
"""
WorldQuant 因子自动提交模块
由 run_genetic.py 在生成推荐因子后自动调用，
也可以单独运行用于手动提交因子。

依赖：
    pip install python-dotenv requests

根目录创建 .env 文件：
    WQ_USERNAME=你的WQ邮箱
    WQ_PASSWORD=你的WQ密码
"""
import time
from wq_api_client import BrainAPIClient

# ── WQ 回测参数（按需修改）──────────────────────────────────────────────────
WQ_UNIVERSE       = "TOP3000"      # 股票池：TOP3000 / TOP1000 / TOP500
WQ_NEUTRALIZATION = "SUBINDUSTRY"  # 中性化：SUBINDUSTRY / INDUSTRY / MARKET / NONE
WQ_DELAY          = 1              # T+1 成交
WQ_DECAY          = 0              # 不衰减
WQ_TRUNCATION     = 0.08           # 单股最大权重 8%

# WQ 提交成功门槛（Sharpe 超过此值打特殊提示）
WQ_SHARPE_EXCELLENT = 1.25
WQ_SHARPE_GOOD      = 0.69


def submit_factors(factor_list: list, auto_mode: bool = True) -> list:
    """
    批量提交因子到 WQ Brain 回测。

    Args:
        factor_list: 因子列表，每个元素是 dict，包含：
            - 'wq_expr'   : 提交到 WQ 的最终表达式（已含取反和平滑）
            - 'local_shp' : 本地 OOS Sharpe（用于对比）
            - 'label'     : 因子描述标签（用于打印）
        auto_mode: True = 自动全部提交；False = 每个询问是否提交

    Returns:
        results: 每个因子的 WQ 回测结果列表
    """
    if not factor_list:
        print("\n⚠️ 没有因子需要提交到 WQ。")
        return []

    print(f"\n{'='*70}")
    print(f"🌐 WorldQuant Brain 自动提交（共 {len(factor_list)} 个因子）")
    print(f"   股票池: {WQ_UNIVERSE} | 中性化: {WQ_NEUTRALIZATION} | "
          f"Delay: {WQ_DELAY} | Decay: {WQ_DECAY}")
    print(f"{'='*70}")

    try:
        client = BrainAPIClient()
    except Exception as e:
        print(f"❌ WQ 登录失败，跳过自动提交: {e}")
        print("   请检查根目录的 .env 文件是否包含正确的 WQ_USERNAME 和 WQ_PASSWORD")
        return []

    results = []
    for rank, factor in enumerate(factor_list, 1):
        wq_expr   = factor.get('wq_expr', '')
        local_shp = factor.get('local_shp', 0)
        label     = factor.get('label', f'因子{rank}')

        if not wq_expr:
            continue

        print(f"\n[{rank}/{len(factor_list)}] {label}")
        print(f"  本地 OOS Sharpe: {local_shp:+.3f}")

        if not auto_mode:
            ans = input("  是否提交到 WQ？(y/n，默认 y): ").strip().lower()
            if ans == 'n':
                print("  跳过。")
                continue

        try:
            result = client.simulate_alpha(
                expression    = wq_expr,
                delay         = WQ_DELAY,
                decay         = WQ_DECAY,
                universe      = WQ_UNIVERSE,
                neutralization= WQ_NEUTRALIZATION,
                truncation    = WQ_TRUNCATION,
            )

            if result:
                wq_shp  = result.get('sharpe', 0) or 0
                wq_fit  = result.get('fitness', 0) or 0
                wq_turn = result.get('turnover', 0) or 0
                alpha_id = result.get('alpha_id', '')

                print(f"  {'='*55}")
                print(f"  WQ 回测结果:")
                print(f"    Sharpe  : {wq_shp:+.4f}")
                print(f"    Fitness : {wq_fit:+.4f}")
                print(f"    换手率  : {wq_turn:.4f}")
                print(f"    Alpha ID: {alpha_id}")
                print(f"  本地 OOS  : {local_shp:+.3f} → WQ: {wq_shp:+.4f}")

                if wq_shp >= WQ_SHARPE_EXCELLENT:
                    print(f"  🚨 Sharpe 突破 {WQ_SHARPE_EXCELLENT}！强烈建议正式提交！")
                elif wq_shp >= WQ_SHARPE_GOOD:
                    print(f"  📈 Sharpe 超过 {WQ_SHARPE_GOOD}，质量不错，可以考虑提交。")
                elif wq_shp > 0:
                    print(f"  📊 有正向收益，但 Sharpe 偏低，可尝试调整参数后重提。")
                else:
                    print(f"  📉 WQ 上 Sharpe 为负，该因子在 TOP3000 上暂不可用。")
                print(f"  {'='*55}")

                results.append({
                    "label"    : label,
                    "wq_expr"  : wq_expr,
                    "local_shp": local_shp,
                    "wq_shp"   : wq_shp,
                    "wq_fit"   : wq_fit,
                    "wq_turn"  : wq_turn,
                    "alpha_id" : alpha_id,
                })
            else:
                print(f"  ❌ WQ 回测无结果，可能是额度不足或表达式语法问题。")
                results.append({
                    "label"    : label,
                    "wq_expr"  : wq_expr,
                    "local_shp": local_shp,
                    "wq_shp"   : None,
                    "error"    : True,
                })

        except Exception as e:
            print(f"  ❌ 提交异常: {e}")
            results.append({
                "label"    : label,
                "wq_expr"  : wq_expr,
                "local_shp": local_shp,
                "wq_shp"   : None,
                "error"    : True,
            })

        # 相邻两次提交之间稍微等一下，避免触发频率限制
        if rank < len(factor_list):
            time.sleep(2)

    # ── 汇总 ──────────────────────────────────────────────────────────────
    successful = [r for r in results if r.get('wq_shp') is not None]
    if successful:
        print(f"\n{'='*70}")
        print(f"📊 WQ 提交汇总（{len(successful)}/{len(factor_list)} 成功）")
        print(f"{'='*70}")
        print(f"  {'因子标签':<35} {'本地OOS':>9} {'WQ_Shp':>9} {'WQ_Fit':>9}")
        print(f"  {'-'*65}")
        for r in sorted(successful, key=lambda x: x['wq_shp'], reverse=True):
            flag = "🚨" if r['wq_shp'] >= WQ_SHARPE_EXCELLENT else (
                   "📈" if r['wq_shp'] >= WQ_SHARPE_GOOD else "  ")
            print(f"  {r['label'][:35]:<35} {r['local_shp']:>+9.3f} "
                  f"{r['wq_shp']:>+9.4f} {r.get('wq_fit', 0):>+9.4f}  {flag}")

    return results


def manual_submit():
    """单独运行时：交互式手动提交"""
    print("🧪 WorldQuant Brain 因子手动提交模式")
    print("💡 直接粘贴因子公式按回车提交，输入 q 退出\n")

    try:
        client = BrainAPIClient()
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        return

    while True:
        expression = input("\n👇 粘贴因子公式（q 退出）:\n> ").strip()
        if expression.lower() in ('q', 'exit', 'quit'):
            print("👋 退出手动提交模式。")
            break
        if not expression:
            print("⚠️ 公式不能为空！")
            continue

        try:
            result = client.simulate_alpha(
                expression    = expression,
                delay         = WQ_DELAY,
                decay         = WQ_DECAY,
                universe      = WQ_UNIVERSE,
                neutralization= WQ_NEUTRALIZATION,
                truncation    = WQ_TRUNCATION,
            )
            if result:
                shp = result.get('sharpe', 0) or 0
                print(f"\n{'='*50}")
                print(f"  Sharpe  : {shp:+.4f}")
                print(f"  Fitness : {result.get('fitness', 0):+.4f}")
                print(f"  换手率  : {result.get('turnover', 0):.4f}")
                if shp >= WQ_SHARPE_EXCELLENT:
                    print(f"  🚨 突破 {WQ_SHARPE_EXCELLENT}！强烈建议正式提交！")
                elif shp >= WQ_SHARPE_GOOD:
                    print(f"  📈 质量不错，可以考虑提交。")
                else:
                    print(f"  📉 Sharpe 偏低，继续优化。")
                print(f"{'='*50}")
        except Exception as e:
            print(f"❌ 提交失败: {e}")


if __name__ == "__main__":
    manual_submit()