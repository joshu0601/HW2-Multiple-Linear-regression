"""HW2 Streamlit app（繁體中文介面）：WHO 平均壽命 — 依 CRISP-DM 的多元線性回歸。

本機執行：  streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import (TARGET, candidate_features, clean, cv_rmse, evaluate, fit_ols,  # noqa: E402
                      load_raw, predict_with_intervals, select_features, split, vif_table)

# 深色科技風配色：青綠為主色、橘色標示實際值、紅色標示負相關
BLUE, BLUE_DARK, BLUE_PI, BLUE_CI = "#22d3ee", "#2dd4bf", "rgba(34,211,238,0.12)", "rgba(34,211,238,0.35)"
ORANGE, RED = "#fb923c", "#f87171"
GRID, MUTED = "#1e293b", "#94a3b8"

pio.templates["hw2_dark"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Noto Sans TC, sans-serif"),
    title=dict(font=dict(color="#e2e8f0", size=16)),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#334155", tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#334155", tickfont=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    hoverlabel=dict(bgcolor="#111a2e", bordercolor="#22d3ee", font=dict(color="#e2e8f0")),
))
pio.templates.default = "plotly_dark+hw2_dark"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Noto Sans TC', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #0e2a3a 0%, #0b1120 55%) fixed; }
h1 { background: linear-gradient(90deg, #22d3ee, #2dd4bf 50%, #a78bfa); -webkit-background-clip: text;
     -webkit-text-fill-color: transparent; font-weight: 700 !important; letter-spacing: .5px; }
h2, h3 { color: #67e8f9 !important; border-left: 4px solid #22d3ee; padding-left: .6rem; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(17,26,46,.95), rgba(15,23,42,.95));
    border: 1px solid rgba(34,211,238,.35); border-radius: 14px; padding: 14px 18px;
    box-shadow: 0 0 18px rgba(34,211,238,.12), inset 0 0 12px rgba(34,211,238,.05);
    transition: box-shadow .2s, transform .2s;
}
[data-testid="stMetric"]:hover { box-shadow: 0 0 26px rgba(34,211,238,.35); transform: translateY(-2px); }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #22d3ee; font-family: 'JetBrains Mono', monospace; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #1e293b; }
.stTabs [data-baseweb="tab"] { background: #111a2e; border-radius: 10px 10px 0 0; padding: 8px 16px; color: #94a3b8; }
.stTabs [aria-selected="true"] { background: rgba(34,211,238,.12) !important; color: #22d3ee !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a, #0b1120); border-right: 1px solid #1e293b; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: rgba(17,26,46,.6); border: 1px solid #1e293b; border-radius: 12px; padding: 6px; }
code { color: #5eead4 !important; background: rgba(45,212,191,.1) !important; }
</style>
"""
KAGGLE_URL = "https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who"

# 特徵中文名稱（內部仍使用英文欄位名）
ZH = {
    "country": "國家", "year": "年份", "status": "開發狀態", TARGET: "平均壽命",
    "adult_mortality": "成人死亡率", "infant_deaths": "嬰兒死亡數", "alcohol": "酒精消費量",
    "pct_expenditure": "衛生支出占 GDP 比", "hepatitis_b": "B 型肝炎疫苗覆蓋率", "measles": "麻疹病例數",
    "bmi": "平均 BMI", "under_five_deaths": "五歲以下死亡數", "polio": "小兒麻痺疫苗覆蓋率",
    "total_expenditure": "政府衛生支出比例", "diphtheria": "白喉疫苗覆蓋率", "hiv_aids": "HIV/AIDS 死亡率",
    "gdp": "人均 GDP", "population": "人口", "thinness_10_19": "10–19 歲消瘦率", "thinness_5_9": "5–9 歲消瘦率",
    "income_composition": "所得組成指數 (HDI)", "schooling": "受教育年數", "is_developed": "已開發國家 (0/1)",
}
METHOD_ZH = {
    "All features": "全部特徵",
    "RFECV": "RFECV 遞迴特徵消去",
    "Lasso (L1)": "Lasso (L1) 正則化",
    "VIF<10 + Backward (p<0.05)": "VIF<10 + 逐步後退法 (p<0.05)",
    "Manual": "手動選擇",
}


def zh(col: str) -> str:
    """'log_gdp' -> 'log 人均 GDP（log_gdp）'"""
    base = col[4:] if col.startswith("log_") else col
    name = ZH.get(base, base)
    return f"{'log ' if col.startswith('log_') else ''}{name}（{col}）"


def method_zh(m: str) -> str:
    if m.startswith("SelectKBest"):
        return m.replace("SelectKBest (F-test, k=", "SelectKBest F 檢定（k=").replace(")", "）")
    return METHOD_ZH.get(m, m)


st.set_page_config(page_title="HW2 平均壽命多元線性回歸", page_icon="📈", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load():
    raw = load_raw()
    df = clean(raw)
    return raw, df, candidate_features(df)


@st.cache_data
def run_selection(_df, features, k_best):
    s = split(_df, features)
    sel = select_features(s, k_best=k_best)
    rows = []
    for name, cols in sel.items():
        m = fit_ols(s, cols)
        t = evaluate(m, s.X_test, s.y_test, cols)
        rows.append({"method": name, "n": len(cols), "CV RMSE": cv_rmse(s, cols),
                     "test R²": t["R2"], "test RMSE": t["RMSE"], "test MAE": t["MAE"],
                     "features": ", ".join(cols)})
    return sel, pd.DataFrame(rows)


raw, df, features = load()
s = split(df, features)

# ---------------------------------------------------------------- 側邊欄
st.sidebar.title("⚙️ 模型設定")
k_best = st.sidebar.slider("SelectKBest 的 k 值", 3, len(features), 8)
selections, comp = run_selection(df, tuple(features), k_best)
best_default = comp.loc[comp["CV RMSE"].idxmin(), "method"]
method = st.sidebar.selectbox("特徵選擇方法", list(selections) + ["Manual"],
                              index=list(selections).index(best_default), format_func=method_zh)
if method == "Manual":
    chosen = st.sidebar.multiselect("特徵", features, default=selections[best_default], format_func=zh)
else:
    chosen = selections[method]
    st.sidebar.caption("選中的特徵：" + "、".join(zh(c) for c in chosen))
alpha = 1 - st.sidebar.slider("區間信心水準 (%)", 80, 99, 95) / 100
if not chosen:
    st.warning("請至少選擇一個特徵。")
    st.stop()

model = fit_ols(s, chosen)
test_m = evaluate(model, s.X_test, s.y_test, chosen)
train_m = evaluate(model, s.X_train, s.y_train, chosen)
pi = predict_with_intervals(model, s.X_test, chosen, alpha=alpha)
pi["actual"] = s.y_test
pi["country"] = df.loc[s.X_test.index, "country"]
pi["year"] = df.loc[s.X_test.index, "year"]
coverage = ((pi["actual"] >= pi["pi_low"]) & (pi["actual"] <= pi["pi_high"])).mean()
lvl = f"{1 - alpha:.0%}"

st.title("HW2 — 以多元線性回歸預測各國平均壽命")
st.caption(f"資料集：WHO Life Expectancy · [Kaggle]({KAGGLE_URL}) · 2,938 筆 · 22 欄（20 個特徵）· CRISP-DM 流程")

tabs = st.tabs(["1 商業理解", "2 資料理解", "3 資料準備", "4 建模", "5 評估", "6 部署 / 預測"])

# ---------------------------------------------------------------- 1
with tabs[0]:
    st.header("商業理解 Business Understanding")
    st.markdown("""
**問題。** 平均壽命是衡量一國衛生體系的關鍵指標。衛生部門與 NGO 需要知道 *哪些可量測的因素*
（死亡率、疫苗接種、所得、教育、HIV 盛行率…）與平均壽命相關，並能依一國的指標估計其預期壽命。

**目標。** 建立可解釋的 **多元線性回歸** 模型：
1. 以 WHO / 聯合國指標預測平均壽命（年）；
2. 透過特徵選擇找出最重要的因子；
3. 以 **信賴區間與預測區間** 量化不確定性。

**成功標準。** 在 **訓練時從未見過的國家** 上達到 Test R² ≥ 0.80、RMSE ≤ 4 年；
約 95% 的實際值落在 95% 預測區間內。
""")

# ---------------------------------------------------------------- 2
with tabs[1]:
    st.header("資料理解 Data Understanding")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("資料筆數", f"{len(raw):,}")
    c2.metric("國家數", raw["country"].nunique())
    c3.metric("年份", f"{raw['year'].min()}–{raw['year'].max()}")
    c4.metric("特徵數", raw.shape[1] - 2)
    st.dataframe(raw.head(20).rename(columns=ZH), width="stretch")
    left, right = st.columns(2)
    miss = raw.isna().sum().loc[lambda m: m > 0].sort_values()
    fig = go.Figure(go.Bar(x=miss.values, y=[ZH.get(c, c) for c in miss.index], orientation="h", marker_color=BLUE))
    fig.update_layout(title="各欄位缺值數", xaxis_title="缺值筆數", height=420, margin=dict(l=10, r=10, t=40, b=10))
    left.plotly_chart(fig, width="stretch")
    tc = df[features + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig = go.Figure(go.Bar(x=tc.values, y=[zh(c) for c in tc.index], orientation="h",
                           marker_color=[RED if v < 0 else BLUE for v in tc.values]))
    fig.update_layout(title="與平均壽命的相關係數", xaxis_title="Pearson r", height=420,
                      margin=dict(l=10, r=10, t=40, b=10))
    right.plotly_chart(fig, width="stretch")
    feat = st.selectbox("探索單一特徵與平均壽命的關係", features, index=features.index("income_composition"),
                        format_func=zh)
    fig = go.Figure(go.Scatter(x=df[feat], y=df[TARGET], mode="markers", marker=dict(size=5, color=BLUE, opacity=0.4),
                               text=df["country"] + " " + df["year"].astype(str),
                               hovertemplate="%{text}<br>" + feat + "=%{x}<br>平均壽命=%{y}<extra></extra>"))
    fig.update_layout(xaxis_title=zh(feat), yaxis_title="平均壽命（年）", height=400, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 3
with tabs[2]:
    st.header("資料準備 Data Preparation")
    st.markdown(f"""
1. **欄名清理** – 去除 Kaggle 欄名中多餘的空白，統一改為 `snake_case`。
2. **目標缺值** – 刪除 {len(raw) - len(df)} 筆沒有平均壽命的資料。
3. **不合理的 0** – `BMI`、`所得組成指數`、`受教育年數` 為 0 屬登錄錯誤 → 設為缺值。
4. **補值** – 先以 **同一國家的中位數** 補（各國指標隨時間穩定），再以全體中位數補。
5. **編碼** – `Status` → `is_developed`（0/1）；`Country` 只作為分組鍵，不當作特徵。
6. **轉換** – 對右偏嚴重的計數 / 金額欄位取 `log1p`：{"、".join(f"`{c}`" for c in
    ["infant_deaths", "pct_expenditure", "measles", "under_five_deaths", "gdp", "population", "hiv_aids"])}。
7. **切分** – 以 `GroupShuffleSplit` 依國家 80/20 切分 → **訓練 {s.groups_train.nunique()} 國
   （{len(s.X_train)} 筆）/ 測試 {df.loc[s.X_test.index, 'country'].nunique()} 國（{len(s.X_test)} 筆）**。
   若隨機以「列」切分，同一國家會同時出現在訓練與測試集，造成資料洩漏、分數虛高。
""")
    desc = df[features + [TARGET]].describe().T.round(2)
    desc.index = [zh(c) for c in desc.index]
    desc.columns = ["筆數", "平均", "標準差", "最小值", "25%", "中位數", "75%", "最大值"]
    st.dataframe(desc, width="stretch")

# ---------------------------------------------------------------- 4
with tabs[3]:
    st.header("建模 Modeling — 特徵選擇與 OLS")
    st.markdown("五種特徵選擇策略 **只在訓練集上** 執行，以 5-fold *依國家分組* 的交叉驗證 RMSE 比較。"
                "預設選 CV RMSE 最低者；測試集只用於最後評估。")
    table = comp.assign(method=comp["method"].map(method_zh)).rename(columns={
        "method": "方法", "n": "特徵數", "CV RMSE": "CV RMSE", "test R²": "測試 R²",
        "test RMSE": "測試 RMSE", "test MAE": "測試 MAE", "features": "選中的特徵"})
    st.dataframe(table.style.format({"CV RMSE": "{:.3f}", "測試 R²": "{:.3f}", "測試 RMSE": "{:.3f}",
                                     "測試 MAE": "{:.3f}"}).highlight_min("CV RMSE", color="rgba(34,211,238,0.25)"),
                 width="stretch", hide_index=True)
    st.subheader(f"OLS 模型：{method_zh(method)}（{len(chosen)} 個特徵）")
    c1, c2 = st.columns([3, 2])
    coefs = pd.DataFrame({"係數": model.params, "標準誤": model.bse, "p 值": model.pvalues})
    ci = model.conf_int(alpha=0.05)
    coefs["95% CI 下界"], coefs["95% CI 上界"] = ci[0], ci[1]
    coefs.index = ["截距" if c == "const" else zh(c) for c in coefs.index]
    c1.dataframe(coefs.round(4), width="stretch")
    vif = vif_table(s.X_train[chosen]).round(2)
    vif["feature"] = vif["feature"].map(zh)
    c2.dataframe(vif.rename(columns={"feature": "特徵"}), width="stretch", hide_index=True)
    c2.caption("VIF > 10 代表嚴重多重共線性（係數不穩定、難以解釋，但不影響預測）。")
    with st.expander("完整 statsmodels 報表"):
        st.text(model.summary().as_text())

# ---------------------------------------------------------------- 5
with tabs[4]:
    st.header("評估 Evaluation")
    c = st.columns(5)
    c[0].metric("測試 R²", f"{test_m['R2']:.3f}", f"訓練 {train_m['R2']:.3f}", delta_color="off")
    c[1].metric("測試調整後 R²", f"{test_m['Adj_R2']:.3f}")
    c[2].metric("測試 RMSE", f"{test_m['RMSE']:.2f} 年")
    c[3].metric("測試 MAE", f"{test_m['MAE']:.2f} 年")
    c[4].metric(f"{lvl} 預測區間涵蓋率", f"{coverage:.1%}")

    srt = pi.sort_values("pred").reset_index(drop=True)
    x = np.arange(len(srt))
    fig = go.Figure([
        go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[srt["pi_high"], srt["pi_low"][::-1]], fill="toself",
                   fillcolor=BLUE_PI, line=dict(width=0), name=f"{lvl} 預測區間 (PI)", hoverinfo="skip"),
        go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[srt["ci_high"], srt["ci_low"][::-1]], fill="toself",
                   fillcolor=BLUE_CI, line=dict(width=0), name=f"{lvl} 信賴區間 (CI)", hoverinfo="skip"),
        go.Scatter(x=x, y=srt["pred"], line=dict(color=BLUE_DARK, width=2), name="預測值"),
        go.Scatter(x=x, y=srt["actual"], mode="markers", marker=dict(color=ORANGE, size=5), name="實際值",
                   text=srt["country"] + " " + srt["year"].astype(str),
                   hovertemplate="%{text}<br>實際 %{y:.1f}<extra></extra>"),
    ])
    fig.update_layout(title="測試集（未見過的國家）預測結果，依預測值排序",
                      xaxis_title="測試樣本", yaxis_title="平均壽命（年）", height=480,
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    lim = [pi[["actual", "pred"]].min().min() - 2, pi[["actual", "pred"]].max().max() + 2]
    fig = go.Figure([
        go.Scatter(x=pi["pred"], y=pi["actual"], mode="markers", marker=dict(color=BLUE, size=5, opacity=0.6),
                   error_x=dict(type="data", symmetric=False, array=pi["pi_high"] - pi["pred"],
                                arrayminus=pi["pred"] - pi["pi_low"], color="rgba(34,211,238,0.18)"),
                   text=pi["country"] + " " + pi["year"].astype(str), name="測試",
                   hovertemplate="%{text}<br>預測 %{x:.1f} / 實際 %{y:.1f}<extra></extra>"),
        go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="gray", dash="dash"), name="y = x"),
    ])
    fig.update_layout(title="實際值 vs 預測值（誤差棒 = 預測區間）", xaxis_title="預測值", yaxis_title="實際值",
                      height=420, showlegend=False)
    left.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Scatter(x=model.fittedvalues, y=model.resid, mode="markers",
                               marker=dict(color=BLUE, size=4, opacity=0.4)))
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(title="殘差 vs 擬合值（訓練集）", xaxis_title="擬合值", yaxis_title="殘差", height=420)
    right.plotly_chart(fig, width="stretch")

    country = st.selectbox("查看某個測試國家的逐年預測", sorted(pi["country"].unique()))
    d = pi[pi["country"] == country].sort_values("year")
    fig = go.Figure([
        go.Scatter(x=np.r_[d["year"], d["year"][::-1]], y=np.r_[d["pi_high"], d["pi_low"][::-1]], fill="toself",
                   fillcolor=BLUE_PI, line=dict(width=0), name="預測區間", hoverinfo="skip"),
        go.Scatter(x=np.r_[d["year"], d["year"][::-1]], y=np.r_[d["ci_high"], d["ci_low"][::-1]], fill="toself",
                   fillcolor=BLUE_CI, line=dict(width=0), name="信賴區間", hoverinfo="skip"),
        go.Scatter(x=d["year"], y=d["pred"], line=dict(color=BLUE_DARK, width=2), name="預測值"),
        go.Scatter(x=d["year"], y=d["actual"], line=dict(color=ORANGE, width=2), mode="lines+markers", name="實際值"),
    ])
    fig.update_layout(height=380, xaxis_title="年份", yaxis_title="平均壽命（年）", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 6
with tabs[5]:
    st.header("部署 Deployment — What-if 預測器")
    st.markdown("調整目前模型所用的指標，即可得到點預測值，以及信賴區間（平均值的不確定性）"
                "與預測區間（單一國家-年份的不確定性）。")
    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(chosen):
        lo, hi, med = float(df[f].min()), float(df[f].max()), float(df[f].median())
        if f == "is_developed":
            vals[f] = cols[i % 3].selectbox(zh(f), [0, 1], index=0)
        else:
            vals[f] = cols[i % 3].slider(zh(f), lo, hi, med, step=(hi - lo) / 200 or 1.0)
    one = pd.DataFrame([vals])
    r = predict_with_intervals(model, one, chosen, alpha=alpha).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("預測平均壽命", f"{r['pred']:.1f} 年")
    c2.metric(f"{lvl} 信賴區間", f"{r['ci_low']:.1f} – {r['ci_high']:.1f}")
    c3.metric(f"{lvl} 預測區間", f"{r['pi_low']:.1f} – {r['pi_high']:.1f}")
    st.caption("`log_*` 滑桿為 log1p 尺度；例如 log_gdp = 8 ≈ 人均 GDP e⁸ − 1 ≈ 2,980 美元。")
    st.markdown("""
**部署說明。** 原始碼放在 GitHub，本網頁部署於 Streamlit Community Cloud。
WHO 每年發布新資料時應重新訓練模型，並監控資料漂移（例如 2015 年後的 COVID-19 衝擊不在訓練資料範圍內）。
""")
