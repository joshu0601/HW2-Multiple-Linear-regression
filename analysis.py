"""HW2 offline analysis: runs the full CRISP-DM pipeline and writes figures/ and results/.

Usage:  python analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import (TARGET, candidate_features, clean, cv_rmse, evaluate, fit_ols,  # noqa: E402
                      load_raw, predict_with_intervals, select_features, split, vif_table)

ROOT = Path(__file__).parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# palette (validated reference palette: blue / orange, red<->blue diverging, gray neutrals)
BLUE, BLUE_DARK, BLUE_PI, BLUE_CI = "#2a78d6", "#1c5cab", "#cde2fb", "#86b6ef"
ORANGE, RED, GRAY, INK, MUTED = "#eb6834", "#e34948", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#d6d5d0",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


# ---------------------------------------------------------------- Data understanding
raw = load_raw()
missing = raw.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
missing.rename("missing").to_csv(RES / "missing_values.csv")
raw.describe().T.round(3).to_csv(RES / "describe.csv")

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(missing.index[::-1], missing.values[::-1], color=BLUE, height=0.6)
ax.set_title("Missing values per column (raw data, 2938 rows)")
ax.set_xlabel("missing rows")
save(fig, "01_missing_values.png")

df = clean(raw)
features = candidate_features(df)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df[TARGET], bins=40, color=BLUE, edgecolor="#fcfcfb", linewidth=1)
ax.set_title("Distribution of life expectancy")
ax.set_xlabel("life expectancy (years)")
ax.set_ylabel("country-years")
save(fig, "02_target_distribution.png")

corr = df[features + [TARGET]].corr()
tc = corr[TARGET].drop(TARGET).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(tc.index, tc.values, color=[RED if v < 0 else BLUE for v in tc.values], height=0.65)
ax.axvline(0, color=GRAY, lw=1)
ax.set_title("Pearson correlation with life expectancy")
ax.set_xlabel("r")
save(fig, "03_target_correlation.png")

fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="r")
ax.set_title("Correlation matrix (after cleaning & log transforms)")
save(fig, "04_correlation_heatmap.png")

# ---------------------------------------------------------------- Feature selection
s = split(df, features)
selections = select_features(s)
rows = []
for name, cols in selections.items():
    m = fit_ols(s, cols)
    test = evaluate(m, s.X_test, s.y_test, cols)
    rows.append({"method": name, "n_features": len(cols), "CV_RMSE": cv_rmse(s, cols),
                 "train_adj_R2": m.rsquared_adj, "AIC": m.aic,
                 **{f"test_{k}": v for k, v in test.items()}, "features": ", ".join(cols)})
comp = pd.DataFrame(rows)
comp.to_csv(RES / "feature_selection_comparison.csv", index=False)

# model selection rule: lowest grouped-CV RMSE on the training set (test set untouched)
best = comp.loc[comp["CV_RMSE"].idxmin()]
best_name, best_cols = best["method"], selections[best["method"]]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
order = comp.sort_values("CV_RMSE", ascending=False)
for ax, col, title in [(axes[0], "CV_RMSE", "5-fold grouped CV RMSE (train)"),
                       (axes[1], "test_RMSE", "Hold-out RMSE (unseen countries)")]:
    colors = [BLUE_DARK if m == best_name else BLUE_CI for m in order["method"]]
    ax.barh(order["method"], order[col], color=colors, height=0.6)
    for y, v in enumerate(order[col]):
        ax.text(v + 0.03, y, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("RMSE (years) — lower is better")
    ax.set_xlim(0, order[col].max() * 1.15)
fig.suptitle(f"Feature-selection strategies (selected: {best_name})", fontweight="bold")
save(fig, "05_feature_selection.png")

# ---------------------------------------------------------------- Modeling
model = fit_ols(s, best_cols)
(RES / "ols_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
vif_table(s.X_train[best_cols]).round(2).to_csv(RES / "vif_selected.csv", index=False)

# standardised coefficients (+95% CI) for interpretability
Xz = (s.X_train[best_cols] - s.X_train[best_cols].mean()) / s.X_train[best_cols].std()
mz = sm.OLS(s.y_train, sm.add_constant(Xz)).fit()
coef = mz.params.drop("const")
ci = mz.conf_int().drop("const")
o = coef.sort_values().index
fig, ax = plt.subplots(figsize=(7, 0.5 * len(o) + 1.5))
ax.errorbar(coef[o], range(len(o)), xerr=[coef[o] - ci.loc[o, 0], ci.loc[o, 1] - coef[o]],
            fmt="o", color=BLUE, ecolor=BLUE_CI, elinewidth=3, capsize=0, ms=8)
ax.axvline(0, color=GRAY, lw=1)
ax.set_yticks(range(len(o)), o)
ax.set_title("Standardised coefficients with 95% CI")
ax.set_xlabel("change in life expectancy (years) per 1 SD of feature")
save(fig, "06_coefficients.png")

# ---------------------------------------------------------------- Evaluation
test_metrics = evaluate(model, s.X_test, s.y_test, best_cols)
train_metrics = evaluate(model, s.X_train, s.y_train, best_cols)
pi = predict_with_intervals(model, s.X_test, best_cols)
pi["actual"] = s.y_test
pi["country"] = df.loc[s.X_test.index, "country"]
pi["year"] = df.loc[s.X_test.index, "year"]
pi["in_pi"] = (pi["actual"] >= pi["pi_low"]) & (pi["actual"] <= pi["pi_high"])
pi.to_csv(RES / "test_predictions.csv")
coverage = pi["in_pi"].mean()

# (a) main prediction plot: test samples sorted by prediction, CI & PI bands
srt = pi.sort_values("pred").reset_index(drop=True)
x = np.arange(len(srt))
fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(x, srt["pi_low"], srt["pi_high"], color=BLUE_PI, label="95% prediction interval")
ax.fill_between(x, srt["ci_low"], srt["ci_high"], color=BLUE_CI, label="95% confidence interval")
ax.plot(x, srt["pred"], color=BLUE_DARK, lw=2, label="predicted")
out = ~srt["in_pi"]
ax.scatter(x[~out], srt.loc[~out, "actual"], s=10, color=ORANGE, alpha=0.7, label="actual (inside PI)", zorder=3)
ax.scatter(x[out], srt.loc[out, "actual"], s=22, color=RED, marker="x", label="actual (outside PI)", zorder=3)
ax.set_title(f"Test-set predictions with 95% CI / PI  —  PI coverage {coverage:.1%}")
ax.set_xlabel("test samples (sorted by predicted value)")
ax.set_ylabel("life expectancy (years)")
ax.legend(loc="upper left", frameon=False)
save(fig, "07_prediction_intervals.png")

# (b) actual vs predicted
fig, ax = plt.subplots(figsize=(6, 6))
ax.errorbar(pi["pred"], pi["actual"], xerr=[pi["pred"] - pi["pi_low"], pi["pi_high"] - pi["pred"]],
            fmt="none", ecolor=BLUE_PI, elinewidth=1, alpha=0.6)
ax.scatter(pi["pred"], pi["actual"], s=12, color=BLUE, alpha=0.8, edgecolor="#fcfcfb", linewidth=0.5)
lim = [min(pi["pi_low"].min(), pi["actual"].min()) - 1, max(pi["pi_high"].max(), pi["actual"].max()) + 1]
ax.plot(lim, lim, color=GRAY, ls="--", lw=1, label="perfect prediction")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_title(f"Actual vs predicted (test)  R²={test_metrics['R2']:.3f}")
ax.set_xlabel("predicted (bars: 95% PI)")
ax.set_ylabel("actual")
ax.legend(frameon=False)
save(fig, "08_actual_vs_predicted.png")

# (c) residual diagnostics
resid = model.resid
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(model.fittedvalues, resid, s=8, color=BLUE, alpha=0.5)
axes[0].axhline(0, color=GRAY, lw=1)
axes[0].set_title("Residuals vs fitted (train)")
axes[0].set_xlabel("fitted")
axes[0].set_ylabel("residual")
(osm, osr), (sl, ic, _) = stats.probplot(resid, dist="norm")
axes[1].scatter(osm, osr, s=8, color=BLUE, alpha=0.5)
axes[1].plot(osm, sl * osm + ic, color=ORANGE, lw=2)
axes[1].set_title("Normal Q-Q of residuals")
axes[1].set_xlabel("theoretical quantiles")
axes[1].set_ylabel("sample quantiles")
save(fig, "09_residuals.png")

# (d) simple linear regression on the single strongest feature, with CI/PI bands
top = tc.abs().idxmax()
slr = sm.OLS(s.y_train, sm.add_constant(s.X_train[[top]])).fit()
grid = pd.DataFrame({top: np.linspace(df[top].min(), df[top].max(), 200)})
band = slr.get_prediction(sm.add_constant(grid, has_constant="add")).summary_frame(alpha=0.05)
slr_test = evaluate(slr, s.X_test, s.y_test, [top])
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(s.X_test[top], s.y_test, s=10, color=ORANGE, alpha=0.6, label="test data")
ax.fill_between(grid[top], band["obs_ci_lower"], band["obs_ci_upper"], color=BLUE_PI, label="95% PI")
ax.fill_between(grid[top], band["mean_ci_lower"], band["mean_ci_upper"], color=BLUE_CI, label="95% CI")
ax.plot(grid[top], band["mean"], color=BLUE_DARK, lw=2, label="fit")
ax.set_title(f"Simple linear regression: life expectancy ~ {top}  (test R²={slr_test['R2']:.3f})")
ax.set_xlabel(top)
ax.set_ylabel("life expectancy (years)")
ax.legend(frameon=False)
save(fig, "10_simple_regression.png")

# (e) per-country time series for a few unseen test countries
countries = (pi.groupby("country").size().loc[lambda c: c >= 15].index.tolist())
picks = [c for c in ["Zimbabwe", "Brazil", "Germany", "India", "Haiti", "Japan"] if c in countries]
picks = (picks + [c for c in countries if c not in picks])[:4]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
for ax, c in zip(axes.flat, picks):
    d = pi[pi["country"] == c].sort_values("year")
    ax.fill_between(d["year"], d["pi_low"], d["pi_high"], color=BLUE_PI, label="95% PI")
    ax.fill_between(d["year"], d["ci_low"], d["ci_high"], color=BLUE_CI, label="95% CI")
    ax.plot(d["year"], d["pred"], color=BLUE_DARK, lw=2, label="predicted")
    ax.plot(d["year"], d["actual"], color=ORANGE, lw=2, marker="o", ms=4, label="actual")
    ax.set_title(c)
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Unseen test countries: predicted trajectory with 95% CI / PI", fontweight="bold")
save(fig, "11_country_trajectories.png")

# ---------------------------------------------------------------- summary
summary = {
    "dataset": "WHO Life Expectancy (Kaggle: kumarajarshi/life-expectancy-who)",
    "rows_raw": int(len(raw)), "rows_clean": int(len(df)), "n_candidate_features": len(features),
    "n_train": int(len(s.X_train)), "n_test": int(len(s.X_test)),
    "n_train_countries": int(s.groups_train.nunique()),
    "n_test_countries": int(df.loc[s.X_test.index, "country"].nunique()),
    "selected_method": best_name, "selected_features": best_cols,
    "train": train_metrics, "test": test_metrics, "pi_coverage_test": coverage,
    "simple_lr": {"feature": top, "test": slr_test},
    "coefficients": model.params.round(4).to_dict(),
    "pvalues": model.pvalues.round(6).to_dict(),
}
(RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
print(comp[["method", "n_features", "CV_RMSE", "test_R2", "test_RMSE"]].round(3).to_string(index=False))
print(json.dumps({k: summary[k] for k in ["selected_method", "selected_features", "test", "pi_coverage_test", "simple_lr"]},
                 indent=2, default=float))
