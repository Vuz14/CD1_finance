from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from core.config import MODEL_TYPES, RANDOM_STATE, RESULTS_DIR, TARGET_COLUMN
from core.pipeline import TrainingResult
from core.utils import evaluate_binary_classifier, load_model_artifact, load_training_frame, prepare_single_input, save_json, select_feature_frame
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

    def _load_oof_matrix(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        merged: Optional[pd.DataFrame] = None
        for model_type in self.base_models:
            path = RESULTS_DIR / model_type / "predictions_with_probability.csv"
            if not path.exists():
                LOGGER.info("Chưa có OOF của %s, train nhanh model nền.", model_type)
                FinancialModelPipeline(model_type).fit({})
            df = pd.read_csv(path)
            cols = ["STT", "MaCP", "Year", TARGET_COLUMN, "Probability"]
            df = df[[c for c in cols if c in df.columns]].copy()
            df = df.rename(columns={"Probability": f"{model_type}_prob"})
            df[f"{model_type}_prob"] = df[f"{model_type}_prob"] / 100.0
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df[["STT", "MaCP", "Year", f"{model_type}_prob"]], on=["STT", "MaCP", "Year"], how="inner")
        if merged is None or merged.empty:
            raise ValueError("Không thể tạo ma trận OOF cho ensemble.")
        feature_cols = [f"{m}_prob" for m in self.base_models if f"{m}_prob" in merged.columns]
        return merged[feature_cols], merged[TARGET_COLUMN].astype(int), merged

    def fit(self) -> Dict[str, float]:
        X_meta, y, meta_df = self._load_oof_matrix()
        class_counts = np.bincount(y.astype(int), minlength=2)
        if len(y) < 6 or class_counts.min() < 3:
            self.meta_model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
        self.meta_model.fit(X_meta, y)
        y_prob = self.meta_model.predict_proba(X_meta)[:, 1]
        metrics = evaluate_binary_classifier(y, y_prob)
        out = meta_df[["STT", "MaCP", "Year", TARGET_COLUMN]].copy()
        out["Prediction"] = (y_prob >= 0.5).astype(int)
        out["Probability"] = (y_prob * 100).round(2)
        out.to_csv(self.output_dir / "predictions_with_probability.csv", index=False)
        save_json(self.output_dir / "metrics_test.json", metrics)
        save_json(self.model_dir / "feature_schema.json", {"features": list(X_meta.columns), "base_models": self.base_models})
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
        return {
            "individual": individual,
            "ensemble": {
                "probability": round(ensemble_prob * 100, 2),
                "base_models": self.base_models,
                "confidence": round(abs(ensemble_prob - 0.5) * 2, 4),
            },
        }
