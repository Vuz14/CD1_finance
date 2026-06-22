from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.config import RESULTS_DIR, TARGET_COLUMN
from core.utils import evaluate_binary_classifier, find_optimal_threshold, load_training_frame, save_json, select_feature_frame
from models.base_models import FinancialModelPipeline
from services.feature_analysis import select_hybrid_features


DEVELOPMENT_END_YEAR = 2021
HOLDOUT_YEARS = (2022, 2023)
MODEL_TYPES = ("ann", "lr", "tree", "xgb")


def load_params() -> Dict[str, Dict[str, Any]]:
    path = BACKEND_DIR / "config" / "training_params.json"
    return json.loads(path.read_text(encoding="utf-8"))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["bin"] = pd.qcut(frame["p"], q=min(bins, len(frame)), duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(count=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    return float((((grouped["observed"] - grouped["predicted"]).abs()) * grouped["count"] / len(frame)).sum())


def development_thresholds(model_type: str) -> Dict[str, float]:
    path = RESULTS_DIR / model_type / "predictions_with_probability.csv"
    predictions = pd.read_csv(path)
    history = predictions[predictions["Year"] <= DEVELOPMENT_END_YEAR]
    probability_col = "ProbabilityRaw" if "ProbabilityRaw" in history.columns else "Probability"
    probabilities = history[probability_col].astype(float)
    if probability_col == "Probability":
        probabilities = probabilities / 100.0
    return {
        objective: find_optimal_threshold(history[TARGET_COLUMN], probabilities, objective=objective)
        for objective in ("f2", "gmean", "youden")
    }


def fit_and_score(
    model_type: str,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    params: Dict[str, Any],
    use_feature_selection: bool = True,
    use_resampling: bool = True,
    threshold: float = 0.5,
    compute_importance: bool = True,
) -> tuple[Dict[str, float], pd.DataFrame, List[str], pd.DataFrame]:
    pipeline = FinancialModelPipeline(model_type)
    X_dev_full = select_feature_frame(development)
    X_holdout_full = select_feature_frame(holdout, X_dev_full.columns)
    y_dev = development[TARGET_COLUMN].astype(int)
    if use_feature_selection:
        selected, _ = select_hybrid_features(X_dev_full, y_dev)
    else:
        selected = X_dev_full.columns.tolist()
    X_dev = X_dev_full[selected]
    X_holdout = X_holdout_full[selected]
    if use_resampling:
        X_fit, y_fit = pipeline._resample(X_dev, y_dev)
    else:
        X_fit, y_fit = X_dev, y_dev
    estimator = pipeline.build_estimator(params, input_dim=len(selected))
    estimator.fit(X_fit, y_fit)
    probability = estimator.predict_proba(X_holdout)[:, 1]
    prediction = (probability >= threshold).astype(int)
    metrics = evaluate_binary_classifier(
        holdout[TARGET_COLUMN], probability, threshold=threshold, y_pred=prediction, include_ci=True
    )
    metrics["ECE"] = expected_calibration_error(holdout[TARGET_COLUMN].to_numpy(), probability)
    output = holdout[["STT", "MaCP", "Year", TARGET_COLUMN]].copy()
    output["Prediction"] = prediction
    output["ProbabilityRaw"] = probability
    output["Probability"] = (probability * 100).round(2)
    importance = pd.DataFrame(columns=["feature", "permutation_importance_mean", "permutation_importance_std", "scoring"])
    if compute_importance:
        importance_result = permutation_importance(
            estimator,
            X_holdout,
            holdout[TARGET_COLUMN],
            scoring="average_precision",
            n_repeats=30,
            random_state=42,
            n_jobs=1,
        )
        importance = pd.DataFrame({
            "feature": selected,
            "permutation_importance_mean": importance_result.importances_mean,
            "permutation_importance_std": importance_result.importances_std,
            "scoring": "average_precision",
        }).sort_values("permutation_importance_mean", ascending=False)
    return metrics, output, selected, importance


def paired_bootstrap_auc_difference(
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    iterations: int = 2000,
) -> Dict[str, float]:
    rng = np.random.default_rng(42)
    differences: List[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            differences.append(float(roc_auc_score(y_true[idx], first[idx]) - roc_auc_score(y_true[idx], second[idx])))
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "auc_difference": float(roc_auc_score(y_true, first) - roc_auc_score(y_true, second)),
        "ci_lower": float(low),
        "ci_upper": float(high),
        "significant_at_0_05": bool(low > 0 or high < 0),
    }


def main() -> None:
    frame = load_training_frame()
    frame = frame[frame["Year"] <= max(HOLDOUT_YEARS)].copy()
    development = frame[frame["Year"] <= DEVELOPMENT_END_YEAR].copy()
    holdout = frame[frame["Year"].isin(HOLDOUT_YEARS)].copy()
    if development.empty or holdout.empty:
        raise ValueError("Development hoặc temporal holdout đang rỗng.")

    output_root = RESULTS_DIR / "research_holdout"
    output_root.mkdir(parents=True, exist_ok=True)
    params_by_model = load_params()
    primary_predictions: Dict[str, pd.DataFrame] = {}
    summary: Dict[str, Any] = {}

    cohort = {
        "study_population": "Vietnamese listed companies",
        "development_years": [int(development["Year"].min()), DEVELOPMENT_END_YEAR],
        "retrospective_holdout_years": list(HOLDOUT_YEARS),
        "development_rows": int(len(development)),
        "development_positive": int(development[TARGET_COLUMN].sum()),
        "holdout_rows": int(len(holdout)),
        "holdout_positive": int(holdout[TARGET_COLUMN].sum()),
        "label_semantics": "T+1 financial-distress proxy reconstructed from audited-statement fields in BCTC_2015_2024_TongHop.csv",
        "label_audit_status": "reproducible generation script available; source-document URLs and page-level verification remain pending",
        "label_generation_script": "backend/scripts/build_financial_distress_labels.py",
        "feature_year_column": "Year",
        "event_year_definition": "Year + 1",
    }
    save_json(output_root / "cohort_and_label_audit.json", cohort)

    for model_type in MODEL_TYPES:
        model_dir = output_root / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        thresholds = development_thresholds(model_type)
        threshold = thresholds["f2"]
        metrics, predictions, selected, importance = fit_and_score(
            model_type,
            development,
            holdout,
            params_by_model[model_type],
            use_feature_selection=False,
            use_resampling=False,
            threshold=threshold,
        )
        metrics["DevelopmentThreshold"] = threshold
        metrics["DevelopmentThresholds"] = thresholds
        metrics["ThresholdSelectionObjective"] = "F2 on expanding-window development predictions through 2021"
        metrics["TestProtocol"] = "train <= 2021; retrospective temporal holdout = 2022-2023"
        save_json(model_dir / "metrics_holdout.json", metrics)
        save_json(model_dir / "selected_features.json", {"features": selected})
        predictions.to_csv(model_dir / "predictions_holdout.csv", index=False)
        importance.to_csv(model_dir / "model_specific_permutation_importance.csv", index=False)
        observed, predicted = calibration_curve(
            predictions[TARGET_COLUMN], predictions["ProbabilityRaw"], n_bins=8, strategy="quantile"
        )
        pd.DataFrame({"mean_predicted_probability": predicted, "observed_distress_rate": observed}).to_csv(
            model_dir / "calibration_curve.csv", index=False
        )
        primary_predictions[model_type] = predictions
        summary[model_type] = metrics

    ablations: List[Dict[str, Any]] = []
    for model_type in ("tree", "xgb"):
        for name, feature_selection, resampling in [
            ("selected_features_with_resampling", True, True),
            ("selected_features_without_resampling", True, False),
            ("all_features_with_resampling", False, True),
            ("primary_all_features_without_resampling", False, False),
        ]:
            metrics, _, selected, _ = fit_and_score(
                model_type,
                development,
                holdout,
                params_by_model[model_type],
                use_feature_selection=feature_selection,
                use_resampling=resampling,
                threshold=0.5,
                compute_importance=False,
            )
            ablations.append({
                "model": model_type,
                "variant": name,
                "feature_count": len(selected),
                "AUC": metrics["AUC"],
                "PR_AUC": metrics["PR_AUC"],
                "Brier": metrics["Brier"],
                "ECE": metrics["ECE"],
            })
    pd.DataFrame(ablations).to_csv(output_root / "ablation_study.csv", index=False)

    tree = primary_predictions["tree"]
    xgb = primary_predictions["xgb"]
    comparison = paired_bootstrap_auc_difference(
        tree[TARGET_COLUMN].to_numpy(), tree["ProbabilityRaw"].to_numpy(), xgb["ProbabilityRaw"].to_numpy()
    )
    comparison.update({"first_model": "tree", "second_model": "xgb", "method": "paired bootstrap, 2000 resamples"})
    save_json(output_root / "tree_vs_xgb_auc_comparison.json", comparison)
    save_json(output_root / "metrics_summary.json", summary)
    print(json.dumps({"cohort": cohort, "metrics": summary, "comparison": comparison}, ensure_ascii=False))


if __name__ == "__main__":
    main()
