from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler


def grey_relational_analysis(X: pd.DataFrame, y: pd.Series, rho: float = 0.5) -> pd.DataFrame:
    X_df = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = MinMaxScaler()
    Xn = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns, index=X_df.index)
    y_float = pd.Series(y).astype(float).reset_index(drop=True)
    denom = y_float.max() - y_float.min()
    y0 = (y_float - y_float.min()) / (denom + 1e-6)
    diff = np.abs(Xn.reset_index(drop=True).to_numpy() - y0.to_numpy().reshape(-1, 1))
    min_diff, max_diff = diff.min(), diff.max()
    xi = (min_diff + rho * max_diff) / (diff + rho * max_diff + 1e-12)
    return pd.DataFrame({"feature": X_df.columns, "gra_score": xi.mean(axis=0)}).sort_values("gra_score", ascending=False)


def model_feature_importance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
    if len(np.unique(y)) < 2:
        return pd.DataFrame({"feature": X.columns, "importance": np.ones(len(X.columns)) / max(len(X.columns), 1)})
    estimator = ExtraTreesClassifier(n_estimators=200, random_state=random_state, class_weight="balanced")
    estimator.fit(X, y)
    return pd.DataFrame({"feature": X.columns, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False)


def hybrid_feature_scores(
    X: pd.DataFrame,
    y: pd.Series,
    gra_weight: float = 0.45,
    importance_weight: float = 0.45,
    collinearity_penalty: float = 0.10,
) -> pd.DataFrame:
    gra = grey_relational_analysis(X, y)
    imp = model_feature_importance(X, y)
    out = gra.merge(imp, on="feature", how="inner")
    out["gra_norm"] = out["gra_score"] / (out["gra_score"].max() + 1e-12)
    out["imp_norm"] = out["importance"] / (out["importance"].max() + 1e-12)
    corr = X.corr(numeric_only=True).abs().fillna(0.0)
    redundancy = []
    for feature in out["feature"]:
        others = corr.loc[feature].drop(labels=[feature], errors="ignore")
        redundancy.append(float(others.max()) if len(others) else 0.0)
    out["redundancy_penalty"] = redundancy
    out["hybrid_score"] = (
        gra_weight * out["gra_norm"]
        + importance_weight * out["imp_norm"]
        - collinearity_penalty * out["redundancy_penalty"]
    )
    return out.sort_values("hybrid_score", ascending=False)


def select_hybrid_features(
    X: pd.DataFrame,
    y: pd.Series,
    min_features: int = 6,
    quantile: float = 0.35,
    max_correlation: float = 0.92,
) -> tuple[List[str], pd.DataFrame]:
    scores = hybrid_feature_scores(X, y)
    if scores.empty:
        return list(X.columns), scores
    threshold = scores["hybrid_score"].quantile(quantile)
    candidates = scores[scores["hybrid_score"] >= threshold]["feature"].tolist()
    selected: List[str] = []
    corr = X.corr(numeric_only=True).abs().fillna(0.0)
    for feature in candidates:
        if not selected:
            selected.append(feature)
            continue
        max_corr = max(float(corr.loc[feature, s]) for s in selected if s in corr.columns)
        if max_corr <= max_correlation:
            selected.append(feature)
    for feature in scores["feature"]:
        if len(selected) >= min_features:
            break
        if feature not in selected:
            selected.append(feature)
    return selected, scores
