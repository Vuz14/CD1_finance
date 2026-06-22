from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from core.config import FEATURE_COLUMNS, MIN_TEST_SAMPLES, MIN_TRAIN_YEARS, RANDOM_STATE, RESULTS_DIR, TARGET_COLUMN, get_device
from core.pipeline import BaseTrainingPipeline, FoldResult, TrainingResult
from core.utils import (
    build_run_manifest,
    evaluate_binary_classifier,
    load_training_frame,
    population_stability_index,
    save_json,
    select_feature_frame,
    smooth_pd_by_cycle,
)
from services.feature_analysis import select_hybrid_features

LOGGER = logging.getLogger("cd1_finance.models")


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Fold-fitted winsorization without learning from future observations."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):  # type: ignore[no-untyped-def]
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):  # type: ignore[no-untyped-def]
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


class TorchANN(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, dropout: float = 0.2, batchnorm: bool = True) -> None:
        super().__init__()
        layers: List[Any] = []
        dims = [input_dim, 128, 64, 32]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(32, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.model(x)


class ANNSklearnWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, input_dim: int, dropout: float = 0.2, batchnorm: bool = True, lr: float = 1e-3, weight_decay: float = 1e-5, epochs: int = 20):
        self.input_dim = input_dim
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.clipper = QuantileClipper()
        self.scaler = MinMaxScaler()
        self.device = get_device()
        self.model_: Optional[TorchANN] = None

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        X_clipped = self.clipper.fit_transform(np.asarray(X, dtype=float))
        X_arr = self.scaler.fit_transform(X_clipped)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        model = TorchANN(X_arr.shape[1], self.dropout, self.batchnorm).to(self.device)
        self.model_ = model
        pos = max(float(y_arr.sum()), 1.0)
        neg = max(float(len(y_arr) - y_arr.sum()), 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32).to(self.device))
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        batch_size = min(32, max(2, len(X_arr) // 4))
        loader = DataLoader(
            TensorDataset(torch.tensor(X_arr, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(X_arr) > batch_size,
        )
        for _ in range(self.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb.to(self.device)), yb.to(self.device))
                loss.backward()
                optimizer.step()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        if self.model_ is None:
            raise ValueError("ANN model chưa được fit.")
        model = self.model_
        X_clipped = self.clipper.transform(np.asarray(X, dtype=float))
        X_arr = self.scaler.transform(X_clipped)
        probs: List[float] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(X_arr), 128):
                xb = torch.tensor(X_arr[start : start + 128], dtype=torch.float32).to(self.device)
                probs.extend(torch.sigmoid(model(xb)).cpu().numpy().ravel().tolist())
        p = np.clip(np.asarray(probs), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):  # type: ignore[no-untyped-def]
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FinancialModelPipeline(BaseTrainingPipeline):
    def __init__(self, model_type: str, dataset_name: Optional[str] = None) -> None:
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.output_dir = RESULTS_DIR / model_type
        self.model_dir = self.output_dir / "final_model"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.selected_features: List[str] = FEATURE_COLUMNS.copy()

    def build_estimator(self, params: Optional[Dict[str, Any]] = None, input_dim: Optional[int] = None):
        params = params or {}
        if self.model_type == "lr":
            penalty = params.get("penalty", "l2")
            lr_base = LogisticRegression(
                solver="lbfgs" if penalty == "l2" else "saga",
                penalty=penalty,
                C=float(params.get("C", 1.0)),
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
            calibration_cv = int(params.get("calibration_cv", 3))
            classifier = (
                CalibratedClassifierCV(
                    estimator=lr_base,
                    method=params.get("calibration_method", "sigmoid"),
                    cv=calibration_cv,
                )
                if calibration_cv >= 2
                else lr_base
            )
            return Pipeline([
                ("clipper", QuantileClipper()),
                ("scaler", StandardScaler()),
                ("clf", classifier),
            ])
        if self.model_type == "tree":
            return Pipeline([
                ("clipper", QuantileClipper()),
                ("clf", LGBMClassifier(
                    n_estimators=int(params.get("n_estimators", 300)),
                    learning_rate=float(params.get("learning_rate", 0.03)),
                    num_leaves=int(params.get("num_leaves", 15)),
                    max_depth=int(params.get("max_depth", 4)),
                    min_child_samples=int(params.get("min_child_samples", 25)),
                    subsample=float(params.get("subsample", 0.85)),
                    colsample_bytree=float(params.get("colsample_bytree", 0.85)),
                    reg_alpha=float(params.get("reg_alpha", 0.1)),
                    reg_lambda=float(params.get("reg_lambda", 1.0)),
                    class_weight="balanced",
                    n_jobs=int(params.get("n_jobs", -1)),
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                )),
            ])
        if self.model_type == "xgb":
            return Pipeline([
                ("clipper", QuantileClipper()),
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    n_estimators=params.get("n_estimators", 160),
                    max_depth=params.get("max_depth", 4),
                    learning_rate=params.get("learning_rate", 0.05),
                    subsample=params.get("subsample", 0.85),
                    colsample_bytree=params.get("colsample_bytree", 0.85),
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    verbosity=0,
                )),
            ])
        if self.model_type == "ann":
            return ANNSklearnWrapper(
                input_dim=input_dim or len(FEATURE_COLUMNS),
                dropout=float(params.get("dropout", 0.20)),
                batchnorm=bool(params.get("batchnorm", True)),
                lr=float(params.get("lr", params.get("learning_rate", 1e-3))),
                weight_decay=float(params.get("weight_decay", 1e-5)),
                epochs=int(params.get("epochs", 20)),
            )
        raise ValueError(f"Model type không hợp lệ: {self.model_type}")

    def _resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 2:
            return X, y
        k_neighbors = max(1, min(5, int(min(np.bincount(y.astype(int))) - 1)))
        try:
            sampler = SMOTEENN(random_state=RANDOM_STATE, smote=SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE))
            X_res, y_res = cast(Tuple[Any, Any], sampler.fit_resample(X, y))
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        except Exception as exc:
            LOGGER.warning("Bỏ qua SMOTE-ENN cho fold này: %s", exc)
            return X, y

    def fit(self, params: Optional[Dict[str, Any]] = None) -> TrainingResult:
        df = load_training_frame(self.dataset_name)
        params = params or {}
        use_resampling = bool(params.get("use_resampling", False))
        use_feature_selection = bool(params.get("use_feature_selection", False))
        years = sorted(df["Year"].dropna().unique())
        all_predictions: List[pd.DataFrame] = []
        fold_results: List[FoldResult] = []
        previous_train_x: Optional[pd.DataFrame] = None

        for idx, year in enumerate(years):
            if idx < MIN_TRAIN_YEARS:
                continue
            train_df = df[df["Year"] < year]
            test_df = df[df["Year"] == year]
            if train_df.empty or len(test_df) < MIN_TEST_SAMPLES:
                LOGGER.warning("Bỏ qua validation year %s vì chỉ có %s mẫu (< %s).", int(year), len(test_df), MIN_TEST_SAMPLES)
                continue
            X_train_full = select_feature_frame(train_df)
            y_train = train_df[TARGET_COLUMN].astype(int)
            X_test_full = select_feature_frame(test_df, X_train_full.columns.tolist())
            hybrid_selected, score_df = select_hybrid_features(X_train_full, y_train)
            selected = hybrid_selected if use_feature_selection else X_train_full.columns.tolist()
            X_train, X_test = X_train_full[selected], X_test_full[selected]
            X_train_res, y_train_res = self._resample(X_train, y_train) if use_resampling else (X_train, y_train)

            fold_params = dict(params)
            if self.model_type == "lr":
                smallest_class = int(y_train_res.value_counts().min())
                fold_params["calibration_cv"] = min(int(params.get("calibration_cv", 3)), smallest_class)
            estimator = self.build_estimator(fold_params, input_dim=len(selected))
            LOGGER.info("Train %s fold <= %s, test %s, features=%s, resampled=%s", self.model_type, int(year) - 1, int(year), len(selected), len(X_train_res))
            estimator.fit(X_train_res, y_train_res)
            y_prob = estimator.predict_proba(X_test)[:, 1]
            metrics = evaluate_binary_classifier(test_df[TARGET_COLUMN].astype(int).to_numpy(), np.asarray(y_prob, dtype=float))
            psi = population_stability_index(previous_train_x, X_train) if previous_train_x is not None else {"PSI_mean": 0.0}
            metrics["PSI"] = psi.get("PSI_mean", 0.0)
            fold_results.append(FoldResult(year=int(year), metrics=metrics, psi=psi))
            previous_train_x = X_train.copy()

            pred = test_df[["STT", "MaCP", "Year", TARGET_COLUMN]].copy()
            pred["Prediction"] = (y_prob >= 0.5).astype(int)
            pred["Probability"] = (y_prob * 100).round(2)
            pred["ProbabilityRaw"] = y_prob
            pred["pd_default"] = y_prob
            all_predictions.append(pred)

        if not all_predictions:
            raise ValueError("Không đủ dữ liệu theo năm để chạy expanding window.")

        predictions = smooth_pd_by_cycle(pd.concat(all_predictions, ignore_index=True))
        y_true_all = predictions[TARGET_COLUMN].astype(int)
        y_prob_all = predictions["pd_default"].astype(float)
        overall_metrics = evaluate_binary_classifier(y_true_all.to_numpy(), y_prob_all.to_numpy(), include_ci=True)
        overall_metrics["PSI"] = float(np.mean([fr.metrics.get("PSI", 0.0) for fr in fold_results]))

        full_X = select_feature_frame(df)
        full_y = df[TARGET_COLUMN].astype(int)
        hybrid_selected, score_df = select_hybrid_features(full_X, full_y)
        self.selected_features = hybrid_selected if use_feature_selection else full_X.columns.tolist()
        selected_full_X = full_X[self.selected_features]
        final_X, final_y = self._resample(selected_full_X, full_y) if use_resampling else (selected_full_X, full_y)
        self.model = self.build_estimator(params, input_dim=len(self.selected_features))
        self.model.fit(final_X, final_y)

        joblib.dump(self.model, self.model_dir / f"pd_{self.model_type}_model.pkl")
        save_json(self.model_dir / "feature_schema.json", {"features": self.selected_features})
        save_json(self.model_dir / "best_params.json", params or {})
        save_json(
            self.output_dir / "run_manifest.json",
            build_run_manifest(
                self.model_type,
                params or {},
                df,
                {
                    "selected_features": self.selected_features,
                    "evaluation": "expanding-window OOF",
                    "minimum_test_year_samples": MIN_TEST_SAMPLES,
                    "use_resampling": use_resampling,
                    "use_feature_selection": use_feature_selection,
                    "outlier_handling": "1st/99th percentile clipping fitted inside each fold",
                },
            ),
        )
        save_json(self.output_dir / "metrics_test.json", overall_metrics)
        pd.DataFrame([{"Year": fr.year, **fr.metrics} for fr in fold_results]).to_csv(self.output_dir / "metrics_by_year.csv", index=False)
        predictions[["STT", "MaCP", "Year", TARGET_COLUMN, "Prediction", "Probability", "ProbabilityRaw"]].to_csv(
            self.output_dir / "predictions_with_probability.csv", index=False
        )
        score_df.to_csv(self.output_dir / f"feature_influence_analysis_{self.model_type}.csv", index=False)

        return TrainingResult(
            model_type=self.model_type,
            metrics=overall_metrics,
            fold_metrics=fold_results,
            selected_features=self.selected_features,
            output_dir=self.output_dir,
        )

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        if self.model is None:
            self.model = joblib.load(self.model_dir / f"pd_{self.model_type}_model.pkl")
        from core.utils import load_feature_schema

        feature_schema = load_feature_schema(self.model_type)
        X = select_feature_frame(frame, feature_schema)
        return pd.Series(self.model.predict_proba(X)[:, 1], index=frame.index)


def train_model(model_type: str, params: Optional[Dict[str, Any]] = None) -> TrainingResult:
    return FinancialModelPipeline(model_type).fit(params=params)
