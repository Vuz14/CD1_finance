from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from core.config import MODEL_TYPES, RANDOM_STATE, RESULTS_DIR, TARGET_COLUMN
from core.utils import (
    build_run_manifest,
    evaluate_binary_classifier,
    find_optimal_threshold,
    load_json,
    load_model_artifact,
    load_training_frame,
    prepare_single_input,
    save_json,
)
from .base_models import FinancialModelPipeline

LOGGER = logging.getLogger("cd1_finance.ensemble")


class EnsembleStackingClassifier:
    """Meta-stacking classifier từ xác suất OOF của ANN, LR, Tree và XGBoost."""

    def __init__(self, base_models: Optional[List[str]] = None) -> None:
        self.base_models = base_models or list(MODEL_TYPES)
        self.meta_model = CalibratedClassifierCV(
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
            cv=3,
        )
        self.output_dir = RESULTS_DIR / "ensemble"
        self.model_dir = self.output_dir / "final_model"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _new_meta_model(y: pd.Series):
        base = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
        class_counts = np.bincount(y.astype(int), minlength=2)
        if len(y) >= 6 and class_counts.min() >= 3:
            return CalibratedClassifierCV(base, method="sigmoid", cv=min(3, int(class_counts.min())))
        return base

    def _load_oof_matrix(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        merged: Optional[pd.DataFrame] = None
        for model_type in self.base_models:
            path = RESULTS_DIR / model_type / "predictions_with_probability.csv"
            if not path.exists():
                LOGGER.info("Chưa có OOF của %s, train nhanh model nền.", model_type)
                FinancialModelPipeline(model_type).fit({})
            df = pd.read_csv(path)
            cols = ["STT", "MaCP", "Year", TARGET_COLUMN, "Probability", "ProbabilityRaw"]
            df = df[[c for c in cols if c in df.columns]].copy()
            if "ProbabilityRaw" in df.columns:
                df = df.rename(columns={"ProbabilityRaw": f"{model_type}_prob"})
                df = df.drop(columns=["Probability"], errors="ignore")
            else:
                df = df.rename(columns={"Probability": f"{model_type}_prob"})
                df[f"{model_type}_prob"] = df[f"{model_type}_prob"] / 100.0
            if merged is None:
                merged = df
            else:
                merged = merged.merge(
                    df[["MaCP", "Year", TARGET_COLUMN, f"{model_type}_prob"]],
                    on=["MaCP", "Year", TARGET_COLUMN],
                    how="inner",
                )
        if merged is None or merged.empty:
            raise ValueError("Không thể tạo ma trận OOF cho ensemble.")
        feature_cols = [f"{m}_prob" for m in self.base_models if f"{m}_prob" in merged.columns]
        return merged[feature_cols], merged[TARGET_COLUMN].astype(int), merged

    def fit(self) -> Dict[str, float]:
        X_meta, y, meta_df = self._load_oof_matrix()
        temporal_parts: List[pd.DataFrame] = []
        year_metrics: List[Dict[str, float]] = []
        for year in sorted(meta_df["Year"].unique()):
            train_mask = meta_df["Year"] < year
            test_mask = meta_df["Year"] == year
            if not train_mask.any() or y.loc[train_mask].nunique() < 2:
                continue
            fold_model = self._new_meta_model(y.loc[train_mask])
            fold_model.fit(X_meta.loc[train_mask], y.loc[train_mask])
            fold_prob = fold_model.predict_proba(X_meta.loc[test_mask])[:, 1]

            if temporal_parts:
                history = pd.concat(temporal_parts, ignore_index=True)
                threshold = find_optimal_threshold(history[TARGET_COLUMN], history["ProbabilityRaw"])
            else:
                threshold = 0.5
            fold_pred = (fold_prob >= threshold).astype(int)
            fold_out = meta_df.loc[test_mask, ["STT", "MaCP", "Year", TARGET_COLUMN]].copy()
            fold_out["Prediction"] = fold_pred
            fold_out["Probability"] = (fold_prob * 100).round(2)
            fold_out["ProbabilityRaw"] = fold_prob
            fold_out["Threshold"] = threshold
            temporal_parts.append(fold_out)
            fold_metrics = evaluate_binary_classifier(
                y.loc[test_mask], fold_prob, threshold=threshold, y_pred=fold_pred
            )
            year_metrics.append({"Year": int(year), **fold_metrics})

        if not temporal_parts:
            raise ValueError("Không đủ lịch sử hai lớp để đánh giá temporal ensemble.")
        out = pd.concat(temporal_parts, ignore_index=True)
        metrics = evaluate_binary_classifier(
            out[TARGET_COLUMN],
            out["ProbabilityRaw"],
            threshold=float(out["Threshold"].mean()),
            y_pred=out["Prediction"],
            include_ci=True,
        )
        metrics["EvaluationStartYear"] = int(out["Year"].min())

        self.meta_model = self._new_meta_model(y)
        self.meta_model.fit(X_meta, y)
        deployment_threshold = find_optimal_threshold(out[TARGET_COLUMN], out["ProbabilityRaw"])
        metrics["DeploymentThreshold"] = deployment_threshold
        out.to_csv(self.output_dir / "predictions_with_probability.csv", index=False)
        pd.DataFrame(year_metrics).to_csv(self.output_dir / "metrics_by_year.csv", index=False)
        save_json(self.output_dir / "metrics_test.json", metrics)
        save_json(
            self.model_dir / "feature_schema.json",
            {
                "features": list(X_meta.columns),
                "base_models": self.base_models,
                "decision_threshold": deployment_threshold,
            },
        )
        save_json(
            self.output_dir / "run_manifest.json",
            build_run_manifest(
                "ensemble",
                {"meta_model": "calibrated_logistic_regression"},
                load_training_frame(),
                {
                    "base_models": self.base_models,
                    "evaluation": "temporal expanding-window meta OOF",
                    "decision_threshold": deployment_threshold,
                },
            ),
        )
        joblib.dump(self.meta_model, self.model_dir / "pd_ensemble_model.pkl")
        return metrics

    def predict_from_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        meta_inputs: List[float] = []
        individual: Dict[str, Any] = {}
        for model_type in self.base_models:
            try:
                model = load_model_artifact(model_type)
                from core.utils import load_feature_schema

                feature_names = load_feature_schema(model_type)
                X = prepare_single_input(features, feature_names)
                prob = float(model.predict_proba(X)[:, 1][0])
                individual[model_type] = {"probability": round(prob * 100, 2)}
                meta_inputs.append(prob)
            except Exception as exc:
                individual[model_type] = {"error": str(exc)}
                meta_inputs.append(0.5)
        meta_path = self.model_dir / "pd_ensemble_model.pkl"
        if not meta_path.exists():
            self.fit()
        meta = joblib.load(meta_path)
        ensemble_prob = float(meta.predict_proba(pd.DataFrame([meta_inputs], columns=[f"{m}_prob" for m in self.base_models]))[:, 1][0])
        schema = load_json(self.model_dir / "feature_schema.json", {})
        threshold = float(schema.get("decision_threshold", 0.5))
        return {
            "individual": individual,
            "ensemble": {
                "probability": round(ensemble_prob * 100, 2),
                "base_models": self.base_models,
                "confidence": round(abs(ensemble_prob - 0.5) * 2, 4),
                "decision_threshold": round(threshold, 6),
                "risk_level": "HIGH RISK" if ensemble_prob >= threshold else "LOW RISK",
            },
        }
