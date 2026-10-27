"""Shared pipeline for HW2: WHO Life Expectancy multiple linear regression (CRISP-DM).

Used by both `analysis.py` (offline report / figures) and `app.py` (Streamlit web app).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Life Expectancy Data.csv"
TARGET = "life_expectancy"
SEED = 42

# Kaggle column names contain stray spaces / inconsistent case -> normalise
RENAME = {
    "Country": "country",
    "Year": "year",
    "Status": "status",
    "Life expectancy": TARGET,
    "Adult Mortality": "adult_mortality",
    "infant deaths": "infant_deaths",
    "Alcohol": "alcohol",
    "percentage expenditure": "pct_expenditure",
    "Hepatitis B": "hepatitis_b",
    "Measles": "measles",
    "BMI": "bmi",
    "under-five deaths": "under_five_deaths",
    "Polio": "polio",
    "Total expenditure": "total_expenditure",
    "Diphtheria": "diphtheria",
    "HIV/AIDS": "hiv_aids",
    "GDP": "gdp",
    "Population": "population",
    "thinness  1-19 years": "thinness_10_19",
    "thinness 5-9 years": "thinness_5_9",
    "Income composition of resources": "income_composition",
    "Schooling": "schooling",
}

# Heavily right-skewed count / money columns -> log1p makes the relation more linear
LOG_COLS = ["infant_deaths", "pct_expenditure", "measles", "under_five_deaths",
            "gdp", "population", "hiv_aids"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={k.strip(): v for k, v in RENAME.items()})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without target, fix impossible zeros, impute by country median then global median."""
    df = df.dropna(subset=[TARGET]).copy()
    # Values of 0 for these are data-entry errors in the WHO export (e.g. BMI=0, income index=0)
    for col in ["bmi", "income_composition", "schooling"]:
        df.loc[df[col] == 0, col] = np.nan
    num_cols = df.select_dtypes("number").columns.drop(TARGET)
    df[num_cols] = df.groupby("country")[num_cols].transform(lambda s: s.fillna(s.median()))
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df["is_developed"] = (df["status"] == "Developed").astype(int)
    for col in LOG_COLS:
        df[f"log_{col}"] = np.log1p(df[col])
    return df.reset_index(drop=True)


def candidate_features(df: pd.DataFrame) -> list[str]:
    """All usable predictors (raw skewed cols replaced by their log version)."""
    excluded = {"country", "status", TARGET, *LOG_COLS}
    return [c for c in df.columns if c not in excluded]


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series


def split(df: pd.DataFrame, features: list[str], test_size: float = 0.2) -> Split:
    """Split by *country* so the test set contains countries the model has never seen."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr, te = next(gss.split(df, groups=df["country"]))
    return Split(df.loc[tr, features], df.loc[te, features],
                 df.loc[tr, TARGET], df.loc[te, TARGET], df.loc[tr, "country"])


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X)
    vifs = [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])]
    return pd.DataFrame({"feature": X.columns, "VIF": vifs}).sort_values("VIF", ascending=False)


def drop_high_vif(X: pd.DataFrame, threshold: float = 10.0) -> list[str]:
    cols = list(X.columns)
    while len(cols) > 1:
        v = vif_table(X[cols])
        if v["VIF"].iloc[0] < threshold:
            break
        cols.remove(v["feature"].iloc[0])
    return cols


def backward_elimination(X: pd.DataFrame, y: pd.Series, alpha: float = 0.05) -> list[str]:
    cols = list(X.columns)
    while cols:
        model = sm.OLS(y, sm.add_constant(X[cols])).fit()
        pvals = model.pvalues.drop("const")
        if pvals.max() <= alpha:
            break
        cols.remove(pvals.idxmax())
    return cols


def select_features(s: Split, k_best: int = 8) -> dict[str, list[str]]:
    """Run several feature-selection strategies on the training set only."""
    X, y = s.X_train, s.y_train
    # pre-computed country-grouped folds (avoids metadata-routing for `groups`)
    cv = list(GroupKFold(n_splits=5).split(X, y, groups=s.groups_train))
    out: dict[str, list[str]] = {"All features": list(X.columns)}

    skb = SelectKBest(f_regression, k=k_best).fit(X, y)
    out[f"SelectKBest (F-test, k={k_best})"] = list(X.columns[skb.get_support()])

    rfe = RFECV(LinearRegression(), cv=cv, scoring="neg_root_mean_squared_error",
                min_features_to_select=3)
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    rfe.fit(Xs, y)
    out["RFECV"] = list(X.columns[rfe.support_])

    lasso = make_pipeline(StandardScaler(), LassoCV(cv=cv, random_state=SEED, max_iter=50_000))
    lasso.fit(X, y)
    coefs = lasso[-1].coef_
    out["Lasso (L1)"] = list(X.columns[np.abs(coefs) > 1e-3])

    vif_cols = drop_high_vif(X)
    out["VIF<10 + Backward (p<0.05)"] = backward_elimination(X[vif_cols], y)
    return out


def cv_rmse(s: Split, features: list[str]) -> float:
    cv = list(GroupKFold(n_splits=5).split(s.X_train, s.y_train, groups=s.groups_train))
    scores = cross_val_score(LinearRegression(), s.X_train[features], s.y_train, cv=cv,
                             scoring="neg_root_mean_squared_error")
    return float(-scores.mean())


def fit_ols(s: Split, features: list[str]):
    return sm.OLS(s.y_train, sm.add_constant(s.X_train[features])).fit()


def evaluate(model, X: pd.DataFrame, y: pd.Series, features: list[str]) -> dict[str, float]:
    pred = model.predict(sm.add_constant(X[features], has_constant="add"))
    n, p = len(y), len(features)
    r2 = r2_score(y, pred)
    return {
        "R2": r2,
        "Adj_R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": mean_absolute_error(y, pred),
    }


def predict_with_intervals(model, X: pd.DataFrame, features: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """Point prediction + confidence interval (mean) + prediction interval (single obs)."""
    frame = model.get_prediction(sm.add_constant(X[features], has_constant="add")).summary_frame(alpha=alpha)
    return frame.rename(columns={"mean": "pred", "mean_ci_lower": "ci_low", "mean_ci_upper": "ci_high",
                                 "obs_ci_lower": "pi_low", "obs_ci_upper": "pi_high"})[
        ["pred", "ci_low", "ci_high", "pi_low", "pi_high"]].set_index(X.index)
