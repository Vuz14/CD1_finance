from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import BASE_DIR, DEFAULT_DATASET, FEATURE_COLUMNS, ID_COLUMNS, RESULTS_DIR, TARGET_COLUMN


def resolve_dataset_path(filename: Optional[str] = DEFAULT_DATASET) -> Path:
    filename = filename or DEFAULT_DATASET
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "dataset" / filename,
        BASE_DIR / "uploads" / filename,
        Path.cwd() / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Không tìm thấy dataset '{filename}'. Đã tìm tại: {[str(p) for p in candidates]}")


def normalize_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported CSV schemas to the internal training contract."""
    out = df.copy()
    aliases = {
        "Ticker": "MaCP",
        "ticker": "MaCP",
        "Company": "MaCP",
        "company": "MaCP",
        "Target": TARGET_COLUMN,
        "target": TARGET_COLUMN,
        "Default": TARGET_COLUMN,
        "default": TARGET_COLUMN,
        "label": TARGET_COLUMN,
    }
    rename_map = {source: target for source, target in aliases.items() if source in out.columns and target not in out.columns}
    if rename_map:
        out = out.rename(columns=rename_map)
    if "STT" not in out.columns:
        out.insert(0, "STT", range(1, len(out) + 1))
    if "MaCP" not in out.columns:
        out["MaCP"] = "UNKNOWN"
    if "Year" not in out.columns:
        raise ValueError("Dataset phải có cột Year.")
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    return out


def load_training_frame(filename: Optional[str] = DEFAULT_DATASET) -> pd.DataFrame:
    df = normalize_training_frame(pd.read_csv(resolve_dataset_path(filename)))
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset phải có cột nhãn '{TARGET_COLUMN}' hoặc alias Target/default/label.")
    return df.sort_values(["MaCP", "Year"])


def select_feature_frame(df: pd.DataFrame, feature_names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    names = list(feature_names) if feature_names is not None else [c for c in FEATURE_COLUMNS if c in df.columns]
    if not names:
        names = df.drop(columns=[TARGET_COLUMN, *ID_COLUMNS, "split"], errors="ignore").select_dtypes(include=[np.number]).columns.tolist()
    return df[names].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def safe_auc(y_true: Any, y_prob: Any) -> float:
    y_arr = np.asarray(y_true)
    if len(np.unique(y_arr)) < 2:
        return 0.5
    return float(roc_auc_score(y_arr, y_prob))


def evaluate_binary_classifier(y_true: Any, y_prob: Any, threshold: float = 0.5) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_prob_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "AUC": safe_auc(y_true_arr, y_prob_arr),
        "Accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "Precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "GMean": float(math.sqrt(max(sensitivity * specificity, 0.0))),
        "Brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
    }


def population_stability_index(expected: pd.DataFrame, actual: pd.DataFrame, bins: int = 10) -> Dict[str, float]:
    psi_scores: Dict[str, float] = {}
    common_cols = [c for c in expected.columns if c in actual.columns]
    for col in common_cols:
        exp = pd.to_numeric(expected[col], errors="coerce").dropna().to_numpy()
        act = pd.to_numeric(actual[col], errors="coerce").dropna().to_numpy()
        if len(exp) == 0 or len(act) == 0:
            psi_scores[col] = 0.0
            continue
        cuts = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
        if len(cuts) < 3:
            psi_scores[col] = 0.0
            continue
        exp_counts, _ = np.histogram(exp, bins=cuts)
        act_counts, _ = np.histogram(act, bins=cuts)
        exp_pct = np.maximum(exp_counts / max(exp_counts.sum(), 1), 1e-6)
        act_pct = np.maximum(act_counts / max(act_counts.sum(), 1), 1e-6)
        psi_scores[col] = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    psi_scores["PSI_mean"] = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
    return psi_scores


def smooth_pd_by_cycle(df: pd.DataFrame, prob_col: str = "pd_default", window: int = 3) -> pd.DataFrame:
    out = df.sort_values(["MaCP", "Year"]).copy()
    out["pd_smooth"] = out.groupby("MaCP")[prob_col].transform(lambda x: x.rolling(window, center=True, min_periods=1).mean())
    return out


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_artifact(model_type: str):
    model_dir = RESULTS_DIR / model_type / "final_model"
    candidates = [
        model_dir / f"pd_{model_type}_model.pkl",
        model_dir / "pd_model.pkl",
    ]
    for path in candidates:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"Không tìm thấy model đã train cho '{model_type}' trong {model_dir}")


def load_feature_schema(model_type: str) -> List[str]:
    schema_path = RESULTS_DIR / model_type / "final_model" / "feature_schema.json"
    schema = load_json(schema_path, {"features": FEATURE_COLUMNS})
    return list(schema.get("features", FEATURE_COLUMNS))


def prepare_single_input(features: Dict[str, Any], feature_names: Iterable[str]) -> pd.DataFrame:
    row = {name: float(features.get(name, 0.0) or 0.0) for name in feature_names}
    return pd.DataFrame([row], columns=list(feature_names))
