# dashboard.py
"""
遗传规划实时监控面板（Plotly Dash）。
独立线程运行，不阻塞进化主循环。
浏览器访问 http://localhost:8050
"""
import threading
import collections
import numpy as np
import pandas as pd
from typing import List

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ────────────────────────────────────────────────────────────────────────────
# 辅助：颜色映射
# ────────────────────────────────────────────────────────────────────────────
def _sharpe_to_color(val: float, vmin: float = -1.0, vmax: float = 2.0) -> str:
    """验证 Sharpe → 渐变颜色（红→绿）"""
    ratio = max(0.0, min(1.0, (val - vmin) / (vmax - vmin + 1e-6)))
    r = int(255 * (1 - ratio))
    g = int(200 * ratio)
    return f"rgba({r},{g},60,0.25)"


# ────────────────────────────────────────────────────────────────────────────
# Dashboard 类
# ────────────────────────────────────────────────────────────────────────────
class Dashboard:
    """
    用法：
        from dashboard import Dashboard
        dash_board = Dashboard(port=8050)
        dash_board.start()   # 在独立线程启动

        # 进化循环中
        dash_board.send_data(gen_data_dict)
    """

    def __init__(self, port: int = 8050, max_history: int = 200):
        self.port    = port
        self.history : collections.deque = collections.deque(maxlen=max_history)
        self._lock   = threading.Lock()

        # 必须在 _build_app() 之前赋值，因为 layout() 会引用它们
        self.control = {
            "paused"              : False,
            "emergency_stop"      : False,
            "inject_count"        : 0,
            "depth_penalty_mul"   : 1.0,
            "node_penalty_mul"    : 1.0,
            "turnover_penalty_mul": 1.0,
            "novelty_delta"       : 0.02,
        }
        self._log_entries: List[dict] = []   # {"gen", "text", "level"}

        self._app = self._build_app()   # 放在最后，此时 self.control 已存在

    # ── 数据接口 ─────────────────────────────────────────────────────────

    def send_data(self, data: dict) -> None:
        """遗传引擎每代调用一次"""
        with self._lock:
            self.history.append(data)
            self._append_log(data)

    def _append_log(self, data: dict) -> None:
        gen = data.get("gen", "?")
        if data.get("redundant_removed", 0) > 0 or data.get("new_injected", 0) > 0:
            self._log_entries.append({
                "gen"  : gen,
                "text" : f"[Gen {gen}] 去重淘汰 {data['redundant_removed']} 个，注入 {data['new_injected']} 个新个体",
                "level": "warn",
            })
        if data.get("early_stop"):
            self._log_entries.append({
                "gen"  : gen,
                "text" : f"[Gen {gen}] 🛑 早停触发，验证适应度连续未提升",
                "level": "danger",
            })
        for w in data.get("warnings", []):
            self._log_entries.append({
                "gen"  : gen,
                "text" : f"[Gen {gen}] {w}",
                "level": "warn",
            })

    # ── 启动 ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """在独立线程启动 Dash 服务器"""
        t = threading.Thread(
            target=lambda: self._app.run(debug=False, port=self.port, use_reloader=False),
            daemon=True,
        )
        t.start()
        print(f"🖥️  Dashboard 已启动：http://localhost:{self.port}")

    # ── 构建 App ─────────────────────────────────────────────────────────

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True,
        )
        app.layout = self._layout()
        self._register_callbacks(app)
        return app

    # ── 布局 ─────────────────────────────────────────────────────────────

    def _layout(self):
        return dbc.Container(fluid=True, children=[
            dcc.Interval(id="interval", interval=500, n_intervals=0),
            dcc.Store(id="control-store", data=self.control),

            # ── 状态概览栏 ──────────────────────────────────────────────
            dbc.Row(id="status-bar", className="my-2 align-items-center", children=[
                dbc.Col(html.H4("🧬 AlphaForge 遗传规划监控面板", className="text-white mb-0"), width=4),
                dbc.Col(id="status-metrics", width=8),
            ], style={"background": "#1a1a2e", "borderRadius": "8px", "padding": "10px"}),

            # ── 主图表 2×2 ──────────────────────────────────────────────
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-fitness",    style={"height": "320px"}), width=6),
                dbc.Col(dcc.Graph(id="fig-complexity", style={"height": "320px"}), width=6),
            ], className="my-1"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="fig-diversity",  style={"height": "320px"}), width=6),
                dbc.Col(dcc.Graph(id="fig-overfit",    style={"height": "320px"}), width=6),
            ], className="my-1"),

            # ── 精英表格 + 日志 ─────────────────────────────────────────
            dbc.Row([
                dbc.Col(self._elite_table_layout(), width=7),
                dbc.Col(self._log_layout(),         width=5),
            ], className="my-1"),

            # ── 手动控制栏 ──────────────────────────────────────────────
            dbc.Row(self._control_layout(),
                    className="my-2",
                    style={"background": "#1a1a2e", "borderRadius": "8px", "padding": "10px"}),
        ])

    def _elite_table_layout(self):
        return dbc.Card([
            dbc.CardHeader("🏆 当前代精英 Top 10"),
            dbc.CardBody(dash_table.DataTable(
                id="elite-table",
                columns=[
                    {"name": "表达式",     "id": "expr"},
                    {"name": "深度",       "id": "depth"},
                    {"name": "节点",       "id": "nodes"},
                    {"name": "换手",       "id": "turnover"},
                    {"name": "训练Sharpe", "id": "train_sharpe"},
                    {"name": "验证Sharpe", "id": "val_sharpe"},
                    {"name": "ICIR",       "id": "icir"},
                ],
                data=[],
                style_header={"backgroundColor": "#2d2d44", "color": "white"},
                style_cell={"backgroundColor": "#1e1e2e", "color": "white",
                            "fontSize": "12px", "padding": "4px 8px"},
                style_data_conditional=[],
                page_size=10,
            )),
        ], color="dark")

    def _log_layout(self):
        return dbc.Card([
            dbc.CardHeader("📋 警告 & 演化日志"),
            dbc.CardBody([
                html.Div(id="warning-zone", className="mb-2"),
                html.Div(id="log-zone",
                         style={"height": "220px", "overflowY": "auto",
                                "background": "#111", "borderRadius": "6px",
                                "padding": "6px", "fontFamily": "monospace",
                                "fontSize": "12px"}),
            ]),
        ], color="dark")

    def _control_layout(self):
        return [
            dbc.Col(html.Div("⚙️ 手动控制", className="text-white fw-bold"), width=1),
            dbc.Col([
                dbc.Button("⏸ 暂停",        id="btn-pause",  color="warning", size="sm", className="me-1"),
                dbc.Button("🛑 紧急早停",    id="btn-stop",   color="danger",  size="sm", className="me-1"),
            ], width=2),
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(id="inject-n", type="number", value=20, min=1, max=100, size="sm"),
                    dbc.Button("💉 注入随机个体", id="btn-inject", size="sm", color="info"),
                ], size="sm"),
            ], width=3),
            dbc.Col([
                html.Small("深度惩罚", className="text-muted"),
                dcc.Slider(id="sl-depth", min=0, max=3, step=0.1, value=1.0,
                           tooltip={"always_visible": False}),
                html.Small("换手惩罚", className="text-muted"),
                dcc.Slider(id="sl-turn",  min=0, max=3, step=0.1, value=1.0,
                           tooltip={"always_visible": False}),
            ], width=4),
            dbc.Col([
                html.Small("新颖性 δ", className="text-muted"),
                dcc.Slider(id="sl-delta", min=0, max=0.05, step=0.005, value=0.02,
                           tooltip={"always_visible": False}),
                dbc.Button("💾 保存最优", id="btn-save", size="sm", color="success"),
                dcc.Download(id="download-csv"),
            ], width=2),
        ]

    # ── 回调注册 ─────────────────────────────────────────────────────────

    def _register_callbacks(self, app: dash.Dash) -> None:
        dash_inst = self   # 闭包捕获

        @app.callback(
            Output("status-metrics", "children"),
            Output("fig-fitness",    "figure"),
            Output("fig-complexity", "figure"),
            Output("fig-diversity",  "figure"),
            Output("fig-overfit",    "figure"),
            Output("elite-table",    "data"),
            Output("elite-table",    "style_data_conditional"),
            Output("warning-zone",   "children"),
            Output("log-zone",       "children"),
            Input("interval", "n_intervals"),
        )
        def refresh(_):
            with dash_inst._lock:
                history = list(dash_inst.history)
                logs    = list(dash_inst._log_entries[-50:])

            if not history:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                    font_color="white", margin=dict(l=30,r=20,t=30,b=30),
                )
                return (
                    "等待数据...",
                    empty_fig, empty_fig, empty_fig, empty_fig,
                    [], [], "", [],
                )

            latest = history[-1]
            gens             = [h["gen"] for h in history]
            train_fits       = [h["train_best_fitness"] for h in history]
            val_fits         = [h["val_best_fitness"]   for h in history]
            avg_depths       = [h["avg_depth"]   for h in history]
            max_depths       = [h["max_depth"]   for h in history]
            avg_nodes        = [h["avg_nodes"]   for h in history]
            max_nodes        = [h["max_nodes"]   for h in history]
            avg_turns        = [h["avg_turnover"] for h in history]
            diversities      = [h["diversity_corr"] for h in history]
            redunds          = [h["redundant_removed"] for h in history]
            injects          = [h["new_injected"] for h in history]
            train_sharpes    = [h["train_best_sharpe"] for h in history]
            val_sharpes      = [h["val_best_sharpe"]   for h in history]
            overfit          = [t - v for t, v in zip(train_sharpes, val_sharpes)]
            val_overall_list = [h["val_best_overall"] for h in history]
            early_stop_gen   = next((h["gen"] for h in history if h.get("early_stop")), None)

            # ── 状态概览栏 ──────────────────────────────────────────────
            dc = latest["diversity_corr"]
            is_improving = (latest["val_best_fitness"] >= latest["val_best_overall"] - 1e-6)
            health = "🟢" if dc > 0.2 and is_improving else ("🟡" if dc > 0.1 else "🔴")

            status = dbc.Row([
                dbc.Col(html.Span(f"{health} 健康", className="text-white fs-5"), width=1),
                dbc.Col(html.Div([
                    html.Span("当前代: ", className="text-muted"),
                    html.Span(f"{latest['gen']}", className="text-white fw-bold me-3"),
                    html.Span("训练最优: ", className="text-muted"),
                    html.Span(f"{latest['train_best_fitness']:+.4f}", className="text-info fw-bold me-3"),
                    html.Span("验证峰值: ", className="text-muted"),
                    html.Span(f"{latest['val_best_overall']:+.4f}", className="text-success fw-bold me-3"),
                    html.Span("早停计数: ", className="text-muted"),
                    html.Span(f"{latest['patience_counter']}", className="text-warning fw-bold"),
                ]), width=11),
            ])

            # ── 图 1：适应度曲线 ────────────────────────────────────────
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=gens, y=train_fits, name="训练最优",
                                      line=dict(color="#4e9af1", width=2)))
            fig1.add_trace(go.Scatter(x=gens, y=val_fits, name="验证最优",
                                      line=dict(color="#f1a14e", width=2),
                                      mode="lines+markers",
                                      marker=dict(
                                          symbol="diamond",
                                          size=[10 if v == max(val_overall_list) else 5
                                                for v in val_fits],
                                          color="#f1a14e"
                                      )))
            if early_stop_gen:
                fig1.add_vline(x=early_stop_gen, line_dash="dash", line_color="red",
                               annotation_text="早停触发")
            fig1.update_layout(
                title="适应度进化", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                font_color="white", legend=dict(orientation="h"),
                margin=dict(l=40,r=20,t=40,b=30),
                yaxis=dict(rangemode="tozero"),
            )

            # ── 图 2：复杂度 & 换手 ─────────────────────────────────────
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Scatter(x=gens, y=avg_depths, name="平均深度",
                                      line=dict(color="#4ecc71", width=2)), secondary_y=False)
            fig2.add_trace(go.Scatter(x=gens, y=max_depths, name="最大深度",
                                      line=dict(color="#4ecc71", width=1, dash="dot")), secondary_y=False)
            fig2.add_trace(go.Scatter(x=gens, y=avg_nodes, name="平均节点",
                                      line=dict(color="#9b59b6", width=2)), secondary_y=False)
            fig2.add_trace(go.Scatter(x=gens, y=max_nodes, name="最大节点",
                                      line=dict(color="#9b59b6", width=1, dash="dot")), secondary_y=False)
            fig2.add_trace(go.Scatter(x=gens, y=avg_turns, name="平均换手",
                                      line=dict(color="#e74c3c", width=2)), secondary_y=True)
            # 阈值线
            fig2.add_hline(y=5,   line_dash="dash", line_color="red",   secondary_y=False, annotation_text="深度阈值5")
            fig2.add_hline(y=20,  line_dash="dash", line_color="purple", secondary_y=False, annotation_text="节点阈值20")
            fig2.add_hline(y=1.5, line_dash="dash", line_color="orange", secondary_y=True,  annotation_text="换手阈值1.5")
            fig2.update_layout(
                title="复杂度 & 换手率", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                font_color="white", legend=dict(orientation="h"),
                margin=dict(l=40,r=40,t=40,b=30),
            )

            # ── 图 3：多样性 ────────────────────────────────────────────
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=gens, y=diversities, name="多样性",
                                      line=dict(color="#8B4513", width=2), mode="lines+markers",
                                      marker=dict(
                                          symbol=["x" if d < 0.15 else "circle" for d in diversities],
                                          size=8,
                                          color=["red" if d < 0.15 else "#8B4513" for d in diversities],
                                      )))
            fig3.add_hline(y=0.3,  line_dash="dash", line_color="yellow", annotation_text="注意(0.3)")
            fig3.add_hline(y=0.15, line_dash="dash", line_color="red",    annotation_text="危险(0.15)")

            # 标注淘汰/注入
            for i, (g, r, inj) in enumerate(zip(gens, redunds, injects)):
                if r > 0 or inj > 0:
                    fig3.add_annotation(x=g, y=diversities[i],
                                        text=f"淘汰:{r} 注入:{inj}",
                                        showarrow=True, arrowhead=2, font=dict(size=9, color="white"))
            fig3.update_layout(
                title="种群多样性（Spearman相关）", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                font_color="white", margin=dict(l=40,r=20,t=40,b=30),
                yaxis=dict(range=[0, 1]),
            )

            # ── 图 4：过拟合检测 ────────────────────────────────────────
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=gens, y=overfit, name="Train-Val Sharpe差",
                                      line=dict(color="#e74c3c", width=2)))
            fig4.add_hline(y=0.3, line_dash="dash", line_color="yellow", annotation_text="注意(0.3)")
            fig4.add_hline(y=0.5, line_dash="dash", line_color="red",    annotation_text="危险(0.5)")

            # 过拟合警告
            if len(overfit) >= 3 and overfit[-1] > 0.5:
                recent_val = val_overall_list[-3:]
                stagnant   = max(recent_val) - min(recent_val) < 0.01
                if stagnant:
                    fig4.add_annotation(
                        x=gens[-1], y=overfit[-1],
                        text="⚠️ 过拟合风险加剧",
                        showarrow=True, font=dict(size=12, color="red"),
                        bgcolor="rgba(255,0,0,0.3)",
                    )
            fig4.update_layout(
                title="过拟合检测（训练-验证 Sharpe 差）", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                font_color="white", margin=dict(l=40,r=20,t=40,b=30),
                yaxis=dict(rangemode="tozero"),
            )

            # ── 精英表格 ────────────────────────────────────────────────
            elite_data = latest.get("elite_profiles", [])
            val_sharpe_vals = [e.get("val_sharpe", 0) for e in elite_data]
            vmin_ = min(val_sharpe_vals) if val_sharpe_vals else -1
            vmax_ = max(val_sharpe_vals) if val_sharpe_vals else 2
            cond_style = [
                {
                    "if": {"row_index": i},
                    "backgroundColor": _sharpe_to_color(e.get("val_sharpe", 0), vmin_, vmax_),
                }
                for i, e in enumerate(elite_data)
            ]

            # ── 当前警告 ────────────────────────────────────────────────
            active_warns = latest.get("warnings", [])
            warn_div = []
            for w in active_warns:
                warn_div.append(dbc.Alert(w, color="danger", className="py-1 mb-1", style={"fontSize": "13px"}))

            # ── 日志面板 ────────────────────────────────────────────────
            color_map = {"warn": "orange", "danger": "red", "info": "white"}
            log_divs = []
            for entry in reversed(logs):
                log_divs.append(html.Div(
                    entry["text"],
                    style={"color": color_map.get(entry["level"], "white"),
                           "borderBottom": "1px solid #333", "padding": "2px 0"},
                ))

            return (status, fig1, fig2, fig3, fig4,
                    elite_data, cond_style, warn_div, log_divs)

        # ── 控制按钮回调 ────────────────────────────────────────────────

        @app.callback(
            Output("btn-pause", "children"),
            Input("btn-pause", "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_pause(n):
            dash_inst.control["paused"] = not dash_inst.control["paused"]
            state = "▶ 继续" if dash_inst.control["paused"] else "⏸ 暂停"
            print(f"[Dashboard] 控制指令: {'暂停' if dash_inst.control['paused'] else '继续'}")
            return state

        @app.callback(
            Output("btn-stop", "color"),
            Input("btn-stop", "n_clicks"),
            prevent_initial_call=True,
        )
        def emergency_stop(n):
            dash_inst.control["emergency_stop"] = True
            print("[Dashboard] ⚠️ 紧急早停指令已发出")
            return "secondary"

        @app.callback(
            Output("btn-inject", "children"),
            Input("btn-inject", "n_clicks"),
            State("inject-n", "value"),
            prevent_initial_call=True,
        )
        def inject_random(n, count):
            dash_inst.control["inject_count"] = int(count or 20)
            print(f"[Dashboard] 注入指令: {dash_inst.control['inject_count']} 个随机个体")
            return "💉 注入随机个体"

        @app.callback(
            Output("download-csv", "data"),
            Input("btn-save", "n_clicks"),
            prevent_initial_call=True,
        )
        def save_best(n):
            with dash_inst._lock:
                history = list(dash_inst.history)
            if not history:
                return None
            latest = history[-1]
            elite  = latest.get("elite_profiles", [])
            df     = pd.DataFrame(elite)
            return dcc.send_data_frame(df.to_csv, "best_factors.csv", index=False)