"""HW2 Streamlit app: WHO Life Expectancy — Multiple Linear Regression following CRISP-DM.

Run locally:  streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import (TARGET, candidate_features, clean, cv_rmse, evaluate, fit_ols,  # noqa: E402
                      load_raw, predict_with_intervals, select_features, split, vif_table)

BLUE, BLUE_DARK, BLUE_PI, BLUE_CI = "#2a78d6", "#1c5cab", "rgba(42,120,214,0.18)", "rgba(42,120,214,0.45)"
ORANGE, RED = "#eb6834", "#e34948"
KAGGLE_URL = "https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who"

st.set_page_config(page_title="HW2 Life Expectancy MLR", page_icon="📈", layout="wide")


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

# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Model settings")
k_best = st.sidebar.slider("k for SelectKBest", 3, len(features), 8)
selections, comp = run_selection(df, tuple(features), k_best)
best_default = comp.loc[comp["CV RMSE"].idxmin(), "method"]
method = st.sidebar.selectbox("Feature-selection method", list(selections) + ["Manual"],
                              index=list(selections).index(best_default))
if method == "Manual":
    chosen = st.sidebar.multiselect("Features", features, default=selections[best_default])
else:
    chosen = selections[method]
    st.sidebar.caption("Selected features: " + ", ".join(chosen))
alpha = 1 - st.sidebar.slider("Interval level (%)", 80, 99, 95) / 100
if not chosen:
    st.warning("Select at least one feature.")
    st.stop()

model = fit_ols(s, chosen)
test_m = evaluate(model, s.X_test, s.y_test, chosen)
train_m = evaluate(model, s.X_train, s.y_train, chosen)
pi = predict_with_intervals(model, s.X_test, chosen, alpha=alpha)
pi["actual"] = s.y_test
pi["country"] = df.loc[s.X_test.index, "country"]
pi["year"] = df.loc[s.X_test.index, "year"]
coverage = ((pi["actual"] >= pi["pi_low"]) & (pi["actual"] <= pi["pi_high"])).mean()

st.title("📈 HW2 — Predicting Life Expectancy with Multiple Linear Regression")
st.caption(f"Dataset: WHO Life Expectancy · [Kaggle]({KAGGLE_URL}) · 2938 rows · 22 columns (20 features) · CRISP-DM")

tabs = st.tabs(["1 Business", "2 Data", "3 Preparation", "4 Modeling", "5 Evaluation", "6 Deployment / Predict"])

# ---------------------------------------------------------------- 1
with tabs[0]:
    st.header("Business Understanding")
    st.markdown("""
**Problem.** Life expectancy is a headline indicator of a country's health system. Ministries of health and NGOs need
to know *which measurable factors* (mortality, immunisation, income, schooling, HIV prevalence, …) are associated with
life expectancy, and to estimate the expected value for a country given its indicators.

**Goal.** Build an interpretable **multiple linear regression** model that
1. predicts life expectancy (years) from WHO/UN indicators,
2. tells us which features matter most (feature selection), and
3. quantifies uncertainty with **confidence and prediction intervals**.

**Success criteria.** Test R² ≥ 0.80 and RMSE ≤ 4 years on **countries never seen during training**;
~95 % of actual values inside the 95 % prediction interval.
""")

# ---------------------------------------------------------------- 2
with tabs[1]:
    st.header("Data Understanding")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Countries", raw["country"].nunique())
    c3.metric("Years", f"{raw['year'].min()}–{raw['year'].max()}")
    c4.metric("Features", raw.shape[1] - 2)
    st.dataframe(raw.head(20), width="stretch")
    left, right = st.columns(2)
    miss = raw.isna().sum().loc[lambda m: m > 0].sort_values()
    fig = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h", marker_color=BLUE))
    fig.update_layout(title="Missing values per column", height=420, margin=dict(l=10, r=10, t=40, b=10))
    left.plotly_chart(fig, width="stretch")
    tc = df[features + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig = go.Figure(go.Bar(x=tc.values, y=tc.index, orientation="h",
                           marker_color=[RED if v < 0 else BLUE for v in tc.values]))
    fig.update_layout(title="Correlation with life expectancy", height=420, margin=dict(l=10, r=10, t=40, b=10))
    right.plotly_chart(fig, width="stretch")
    feat = st.selectbox("Explore a feature vs life expectancy", features, index=features.index("income_composition"))
    fig = go.Figure(go.Scatter(x=df[feat], y=df[TARGET], mode="markers", marker=dict(size=5, color=BLUE, opacity=0.4),
                               text=df["country"] + " " + df["year"].astype(str),
                               hovertemplate="%{text}<br>" + feat + "=%{x}<br>life exp=%{y}<extra></extra>"))
    fig.update_layout(xaxis_title=feat, yaxis_title="life expectancy", height=400, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 3
with tabs[2]:
    st.header("Data Preparation")
    st.markdown(f"""
1. **Column cleanup** – strip stray spaces in Kaggle headers and rename to `snake_case`.
2. **Target** – drop the {len(raw) - len(df)} rows with missing life expectancy.
3. **Invalid zeros** – `BMI`, `Income composition` and `Schooling` equal to 0 are recording errors → set to NaN.
4. **Imputation** – fill NaN with the **same country's median** (indicators are stable over time), then the global median.
5. **Encoding** – `Status` → `is_developed` (0/1). `Country` is used only as a grouping key, not as a feature.
6. **Transforms** – `log1p` for heavily right-skewed counts / money: {", ".join(f"`{c}`" for c in
    ["infant_deaths", "pct_expenditure", "measles", "under_five_deaths", "gdp", "population", "hiv_aids"])}.
7. **Split** – `GroupShuffleSplit` by country, 80/20 → **{s.groups_train.nunique()} train countries
   ({len(s.X_train)} rows) / {df.loc[s.X_test.index, 'country'].nunique()} test countries ({len(s.X_test)} rows)**.
   A random row split would leak the same country into train and test and inflate the score.
""")
    st.dataframe(df[features + [TARGET]].describe().T.round(2), width="stretch")

# ---------------------------------------------------------------- 4
with tabs[3]:
    st.header("Modeling — Feature Selection & OLS")
    st.markdown("Five strategies run on the **training set only**, compared by 5-fold *country-grouped* CV RMSE. "
                "The lowest CV RMSE is chosen by default; the test set is used only for final evaluation.")
    st.dataframe(comp.style.format({"CV RMSE": "{:.3f}", "test R²": "{:.3f}", "test RMSE": "{:.3f}",
                                    "test MAE": "{:.3f}"}).highlight_min("CV RMSE", color="#cde2fb"),
                 width="stretch")
    st.subheader(f"OLS model: {method} ({len(chosen)} features)")
    c1, c2 = st.columns([3, 2])
    coefs = pd.DataFrame({"coef": model.params, "std err": model.bse, "p-value": model.pvalues})
    ci = model.conf_int(alpha=0.05)
    coefs["95% CI low"], coefs["95% CI high"] = ci[0], ci[1]
    c1.dataframe(coefs.round(4), width="stretch")
    c2.dataframe(vif_table(s.X_train[chosen]).round(2), width="stretch", hide_index=True)
    c2.caption("VIF > 10 indicates strong multicollinearity (coefficients become unstable, predictions are unaffected).")
    with st.expander("Full statsmodels summary"):
        st.text(model.summary().as_text())

# ---------------------------------------------------------------- 5
with tabs[4]:
    st.header("Evaluation")
    c = st.columns(5)
    c[0].metric("Test R²", f"{test_m['R2']:.3f}", f"train {train_m['R2']:.3f}", delta_color="off")
    c[1].metric("Test Adj. R²", f"{test_m['Adj_R2']:.3f}")
    c[2].metric("Test RMSE", f"{test_m['RMSE']:.2f} yrs")
    c[3].metric("Test MAE", f"{test_m['MAE']:.2f} yrs")
    c[4].metric(f"{1 - alpha:.0%} PI coverage", f"{coverage:.1%}")

    srt = pi.sort_values("pred").reset_index(drop=True)
    x = np.arange(len(srt))
    fig = go.Figure([
        go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[srt["pi_high"], srt["pi_low"][::-1]], fill="toself",
                   fillcolor=BLUE_PI, line=dict(width=0), name=f"{1 - alpha:.0%} prediction interval", hoverinfo="skip"),
        go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[srt["ci_high"], srt["ci_low"][::-1]], fill="toself",
                   fillcolor=BLUE_CI, line=dict(width=0), name=f"{1 - alpha:.0%} confidence interval", hoverinfo="skip"),
        go.Scatter(x=x, y=srt["pred"], line=dict(color=BLUE_DARK, width=2), name="predicted"),
        go.Scatter(x=x, y=srt["actual"], mode="markers", marker=dict(color=ORANGE, size=5), name="actual",
                   text=srt["country"] + " " + srt["year"].astype(str),
                   hovertemplate="%{text}<br>actual %{y:.1f}<extra></extra>"),
    ])
    fig.update_layout(title="Test-set predictions (unseen countries) sorted by predicted value",
                      xaxis_title="test sample", yaxis_title="life expectancy (years)", height=480,
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    lim = [pi[["actual", "pred"]].min().min() - 2, pi[["actual", "pred"]].max().max() + 2]
    fig = go.Figure([
        go.Scatter(x=pi["pred"], y=pi["actual"], mode="markers", marker=dict(color=BLUE, size=5, opacity=0.6),
                   error_x=dict(type="data", symmetric=False, array=pi["pi_high"] - pi["pred"],
                                arrayminus=pi["pred"] - pi["pi_low"], color="rgba(42,120,214,0.15)"),
                   text=pi["country"] + " " + pi["year"].astype(str), name="test",
                   hovertemplate="%{text}<br>pred %{x:.1f} / actual %{y:.1f}<extra></extra>"),
        go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="gray", dash="dash"), name="y = x"),
    ])
    fig.update_layout(title="Actual vs predicted (bars = PI)", xaxis_title="predicted", yaxis_title="actual",
                      height=420, showlegend=False)
    left.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Scatter(x=model.fittedvalues, y=model.resid, mode="markers",
                               marker=dict(color=BLUE, size=4, opacity=0.4)))
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(title="Residuals vs fitted (train)", xaxis_title="fitted", yaxis_title="residual", height=420)
    right.plotly_chart(fig, width="stretch")

    country = st.selectbox("Trajectory of an unseen test country", sorted(pi["country"].unique()))
    d = pi[pi["country"] == country].sort_values("year")
    fig = go.Figure([
        go.Scatter(x=np.r_[d["year"], d["year"][::-1]], y=np.r_[d["pi_high"], d["pi_low"][::-1]], fill="toself",
                   fillcolor=BLUE_PI, line=dict(width=0), name="PI", hoverinfo="skip"),
        go.Scatter(x=np.r_[d["year"], d["year"][::-1]], y=np.r_[d["ci_high"], d["ci_low"][::-1]], fill="toself",
                   fillcolor=BLUE_CI, line=dict(width=0), name="CI", hoverinfo="skip"),
        go.Scatter(x=d["year"], y=d["pred"], line=dict(color=BLUE_DARK, width=2), name="predicted"),
        go.Scatter(x=d["year"], y=d["actual"], line=dict(color=ORANGE, width=2), mode="lines+markers", name="actual"),
    ])
    fig.update_layout(height=380, xaxis_title="year", yaxis_title="life expectancy", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 6
with tabs[5]:
    st.header("Deployment — What-if Predictor")
    st.markdown("Adjust the indicators of the selected model; the app returns the point prediction with its "
                "confidence interval (uncertainty of the mean) and prediction interval (uncertainty of one country-year).")
    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(chosen):
        lo, hi, med = float(df[f].min()), float(df[f].max()), float(df[f].median())
        if f == "is_developed":
            vals[f] = cols[i % 3].selectbox(f, [0, 1], index=0)
        else:
            vals[f] = cols[i % 3].slider(f, lo, hi, med, step=(hi - lo) / 200 or 1.0)
    one = pd.DataFrame([vals])
    r = predict_with_intervals(model, one, chosen, alpha=alpha).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted life expectancy", f"{r['pred']:.1f} yrs")
    c2.metric(f"{1 - alpha:.0%} confidence interval", f"{r['ci_low']:.1f} – {r['ci_high']:.1f}")
    c3.metric(f"{1 - alpha:.0%} prediction interval", f"{r['pi_low']:.1f} – {r['pi_high']:.1f}")
    st.caption("`log_*` sliders are on the log1p scale; e.g. log_gdp = 8 ≈ GDP per capita of e^8 − 1 ≈ 2,980 USD.")
    st.markdown("""
**Deployment notes.** Source code on GitHub; this app is deployed on Streamlit Community Cloud.
The model should be re-trained whenever WHO publishes a new year of data, and monitored for drift
(e.g. post-2015 COVID-19 shocks are outside the training range).
""")
