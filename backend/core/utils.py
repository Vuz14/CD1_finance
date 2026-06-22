from __future__ import annotations

import json
import hashlib
import math
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
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
        "default_final": TARGET_COLUMN,
        "label": TARGET_COLUMN,
    }
    rename_map = {source: target for source, target in aliases.items() if source in out.columns and target not in out.columns}
    if rename_map:
        out = out.rename(columns=rename_map)
    if "MaCP" not in out.columns:
        out["MaCP"] = "UNKNOWN"
    if "Year" not in out.columns:
        raise ValueError("Dataset phải có cột Year.")
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
        key_cols = ["MaCP", "Year"]
        conflicting = out.groupby(key_cols, dropna=False)[TARGET_COLUMN].transform("nunique") > 1
        out = out.loc[~conflicting].copy()

        # Multiple records for one company-year with the same label are collapsed
        # deterministically instead of being counted more than once in evaluation.
        if out.duplicated(key_cols, keep=False).any():
            value_cols = [c for c in out.columns if c not in {*key_cols, "STT", TARGET_COLUMN}]
            numeric_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(out[c])]
            other_cols = [c for c in value_cols if c not in numeric_cols]
            aggregations: Dict[str, str] = {c: "median" for c in numeric_cols}
            aggregations.update({c: "first" for c in other_cols})
            aggregations[TARGET_COLUMN] = "first"
            out = out.groupby(key_cols, as_index=False, dropna=False).agg(aggregations)

    out = out.drop(columns=["STT"], errors="ignore")
    out.insert(0, "STT", range(1, len(out) + 1))
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


def bootstrap_auc_ci(y_true: Any, y_prob: Any, iterations: int = 500) -> tuple[float, float]:
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_arr)) < 2:
        return 0.5, 0.5
    rng = np.random.default_rng(42)
    scores: List[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if len(np.unique(y_arr[idx])) == 2:
            scores.append(float(roc_auc_score(y_arr[idx], p_arr[idx])))
    if not scores:
        return 0.5, 0.5
    low, high = np.quantile(scores, [0.025, 0.975])
    return float(low), float(high)


def find_optimal_threshold(y_true: Any, y_prob: Any, objective: str = "gmean") -> float:
    """Select a threshold on past data only using G-Mean, F2, or Youden's J."""
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if len(np.unique(y_arr)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 199), p_arr]))
    if objective not in {"gmean", "f2", "youden"}:
        raise ValueError(f"Threshold objective không hợp lệ: {objective}")
    best_key = (-1.0, -1.0, -1.0)
    best_threshold = 0.5
    for threshold in candidates:
        pred = (p_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        gmean = math.sqrt(max(sensitivity * specificity, 0.0))
        f1 = f1_score(y_arr, pred, zero_division=0)
        f2 = fbeta_score(y_arr, pred, beta=2, zero_division=0)
        youden = sensitivity + specificity - 1.0
        primary = {"gmean": gmean, "f2": f2, "youden": youden}[objective]
        key = (primary, gmean, -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def evaluate_binary_classifier(
    y_true: Any,
    y_prob: Any,
    threshold: float = 0.5,
    y_pred: Optional[Any] = None,
    include_ci: bool = False,
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred_arr = np.asarray(y_pred).astype(int) if y_pred is not None else (y_prob_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    metrics = {
        "AUC": safe_auc(y_true_arr, y_prob_arr),
        "PR_AUC": float(average_precision_score(y_true_arr, y_prob_arr)) if y_true_arr.any() else 0.0,
        "Accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "Precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "Recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "F1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "F2": float(fbeta_score(y_true_arr, y_pred_arr, beta=2, zero_division=0)),
        "GMean": float(math.sqrt(max(sensitivity * specificity, 0.0))),
        "Brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    if include_ci:
        metrics["AUC_CI_Lower"], metrics["AUC_CI_Upper"] = bootstrap_auc_ci(y_true_arr, y_prob_arr)
    return metrics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    model_type: str,
    params: Dict[str, Any],
    frame: pd.DataFrame,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR.parent,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=BASE_DIR.parent,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
    except OSError:
        git_commit = "unknown"
        git_dirty = True
    code_paths = [
        BASE_DIR / "core" / "utils.py",
        BASE_DIR / "models" / "base_models.py",
        BASE_DIR / "models" / "ensemble_model.py",
        BASE_DIR / "services" / "feature_analysis.py",
    ]
    label_definition_path = BASE_DIR / "config" / "label_definition.json"
    label_definition = load_json(label_definition_path, {})
    packages: Dict[str, str] = {}
    for package in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "torch", "xgboost", "lightgbm"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    source_frame = pd.read_csv(dataset_path)
    aliases = {"Ticker": "MaCP", "Target": TARGET_COLUMN}
    source_frame = source_frame.rename(columns={k: v for k, v in aliases.items() if k in source_frame.columns})
    duplicate_keys = 0
    conflicting_keys = 0
    if {"MaCP", "Year"}.issubset(source_frame.columns):
        grouped = source_frame.groupby(["MaCP", "Year"], dropna=False)
        duplicate_keys = int((grouped.size() > 1).sum())
        if TARGET_COLUMN in source_frame.columns:
            conflicting_keys = int((grouped[TARGET_COLUMN].nunique() > 1).sum())
    manifest: Dict[str, Any] = {
        "model_type": model_type,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit or "unknown",
        "git_dirty": git_dirty,
        "training_code_sha256": {str(p.relative_to(BASE_DIR)): file_sha256(p) for p in code_paths},
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "clean_rows": int(len(frame)),
        "source_rows": int(len(source_frame)),
        "source_duplicate_company_year_keys": duplicate_keys,
        "source_conflicting_label_keys": conflicting_keys,
        "class_counts": {str(k): int(v) for k, v in frame[TARGET_COLUMN].value_counts().sort_index().items()},
        "year_range": [int(frame["Year"].min()), int(frame["Year"].max())],
        "duplicate_policy": "drop conflicting-label company-years; median-collapse same-label duplicates",
        "label_definition": label_definition,
        "label_definition_sha256": file_sha256(label_definition_path) if label_definition_path.exists() else None,
        "params": params,
        "python": platform.python_version(),
        "packages": packages,
    }
    manifest.update(extra or {})
    return manifest


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
