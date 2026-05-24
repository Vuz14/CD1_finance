from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from core.config import FEATURE_COLUMNS, MODEL_TYPES, RESULTS_DIR, UPLOAD_DIR, LOGGER, setup_utf8
from core.utils import load_feature_schema, load_json, load_model_artifact, load_training_frame, normalize_training_frame, prepare_single_input, save_json, select_feature_frame
from models.ensemble_model import EnsembleStackingClassifier
from scripts.train_runner import run_training
from services.preprocessing import process_raw_data

setup_utf8()

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
CORS(app)

TRAIN_LOCK = threading.Lock()
ALLOWED_CSV = {"csv"}
ALLOWED_PREPROCESS = {"csv", "xlsx", "xls"}


def allowed_file(filename: str, extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def api_error(message: str, status: int = 500):
    LOGGER.exception(message) if status >= 500 else LOGGER.warning(message)
    return jsonify({"success": False, "error": message}), status


def load_metrics(model_type: str) -> Dict[str, Any]:
    return load_json(RESULTS_DIR / model_type / "metrics_test.json", {})


def load_features(model_type: str) -> List[Dict[str, Any]]:
    candidates = [
        RESULTS_DIR / model_type / f"feature_influence_analysis_{model_type}.csv",
        RESULTS_DIR / model_type / "feature_influence_analysis.csv",
        RESULTS_DIR / model_type / f"feature_importance_{model_type}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path).head(10).to_dict("records")
    return []


def predict_all_models(features: Dict[str, Any]) -> Dict[str, Any]:
    ensemble = EnsembleStackingClassifier().predict_from_features(features)
    ensemble["ensemble"].update(build_mdcl_context(ensemble["ensemble"]["probability"]))
    return {
        "predictions": ensemble["individual"],
        "ensemble": ensemble["ensemble"],
    }


def build_mdcl_context(probability_percent: float) -> Dict[str, Any]:
    base_psis = []
    for model_type in MODEL_TYPES:
        metrics = load_metrics(model_type)
        if "PSI" in metrics:
            base_psis.append(float(metrics["PSI"]))
    psi_mean = float(np.mean(base_psis)) if base_psis else 0.0
    adaptive_threshold = 0.5 - (0.3 * min(psi_mean, 0.5))
    probability = probability_percent / 100.0
    return {
        "psi_mean": round(psi_mean, 6),
        "adaptive_threshold": round(adaptive_threshold, 6),
        "adjusted_risk_level": "HIGH RISK" if probability >= adaptive_threshold else "LOW RISK",
        "mdcl_active": psi_mean > 0.15,
    }


def infer_feature_schema() -> List[str]:
    for model_type in ["lr", "xgb", "tree", "ann"]:
        try:
            schema = load_feature_schema(model_type)
            if schema and schema != FEATURE_COLUMNS:
                return schema
        except Exception:
            continue
    try:
        df = load_training_frame()
        return select_feature_frame(df).columns.tolist()
    except Exception:
        return FEATURE_COLUMNS


@app.get("/health")
def health():
    return jsonify({"status": "ok", "message": "Backend đang chạy UTF-8", "models": list(MODEL_TYPES) + ["ensemble"]})


@app.post("/upload")
def upload_file():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return api_error("Không có file được chọn.", 400)
        if not allowed_file(file.filename, ALLOWED_CSV):
            return api_error("Chỉ hỗ trợ file CSV.", 400)
        filename = secure_filename(file.filename)
        path = UPLOAD_DIR / filename
        file.save(path)
        df = pd.read_csv(path)
        return jsonify({
            "success": True,
            "filename": filename,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "first_rows": df.head().to_dict("records"),
        })
    except Exception as exc:
        return api_error(str(exc), 500)


@app.post("/train")
def train_model():
    try:
        payload = request.get_json(silent=True) or {}
        model_type = payload.get("model_type")
        params = payload.get("params", {})
        if model_type not in list(MODEL_TYPES) + ["ensemble", "all"]:
            return api_error("model_type không hợp lệ.", 400)
        with TRAIN_LOCK:
            result = run_training(model_type, params)
        return jsonify(result)
    except Exception as exc:
        return api_error(str(exc), 500)


@app.post("/train-all")
def train_all_models():
    try:
        payload = request.get_json(silent=True) or {}
        with TRAIN_LOCK:
            return jsonify(run_training("all", payload.get("params", {})))
    except Exception as exc:
        return api_error(str(exc), 500)


@app.get("/results/<model_type>")
def get_results(model_type: str):
    if model_type not in list(MODEL_TYPES) + ["ensemble"]:
        return api_error("model_type không hợp lệ.", 400)
    if not (RESULTS_DIR / model_type).exists():
        return api_error("Chưa có kết quả huấn luyện.", 404)
    history_path = RESULTS_DIR / model_type / "metrics_by_year.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    return jsonify({
        "success": True,
        "model_type": model_type,
        "metrics": load_metrics(model_type),
        "features": load_features(model_type),
        "history": history,
    })


@app.get("/models")
def get_models():
    return jsonify({
        "models": [
            {"id": "ann", "name": "Artificial Neural Network", "params": ["dropout", "batchnorm", "lr", "weight_decay", "epochs"]},
            {"id": "lr", "name": "Logistic Regression", "params": ["C", "penalty"]},
            {"id": "tree", "name": "LightGBM", "params": ["n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples"]},
            {"id": "xgb", "name": "XGBoost", "params": ["max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree"]},
            {"id": "ensemble", "name": "Heterogeneous Stacking Ensemble", "params": []},
        ]
    })


@app.get("/feature-schema")
def get_feature_schema():
    features = infer_feature_schema()
    return jsonify({"success": True, "features": features})


@app.post("/compare")
def compare_models():
    results: Dict[str, Any] = {}
    for model_type in list(MODEL_TYPES) + ["ensemble"]:
        metrics = load_metrics(model_type)
        results[model_type] = {
            "metrics": metrics,
            "features": load_features(model_type),
            "status": "trained" if metrics else "not_trained",
        }
    return jsonify({"success": True, "results": results})


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        raw_features = payload.get("features", payload)
        features = {}
        for name, value in raw_features.items():
            try:
                features[name] = float(value or 0.0)
            except (TypeError, ValueError):
                continue
        if not features:
            return api_error("Không có feature tài chính hợp lệ.", 400)
        data = predict_all_models(features)
        return jsonify({"success": True, "data": data["predictions"], "ensemble": data["ensemble"], "input_features": features})
    except Exception as exc:
        return api_error(str(exc), 500)


@app.post("/predict-from-file")
def predict_from_file():
    try:
        file = request.files.get("file")
        if not file or not allowed_file(file.filename, ALLOWED_CSV):
            return api_error("Vui lòng upload CSV hợp lệ.", 400)
        df = normalize_training_frame(pd.read_csv(file.stream))
        feature_schema = infer_feature_schema()
        required = ["MaCP", "Year"] + [name for name in feature_schema if name in df.columns]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return api_error(f"Thiếu cột: {missing}", 400)
        max_year = int(df["Year"].max())
        next_year = max_year + 1
        max_year_data = df[df["Year"] == max_year]
        selected_company = request.form.get("company", max_year_data.iloc[0]["MaCP"])
        company_row = max_year_data[max_year_data["MaCP"] == selected_company]
        if company_row.empty:
            company_row = max_year_data.iloc[[0]]
        row = company_row.iloc[0]
        features = {name: float(row.get(name, 0.0) if pd.notna(row.get(name, 0.0)) else 0.0) for name in feature_schema}
        prediction = predict_all_models(features)
        return jsonify({
            "success": True,
            "data": {
                "maxYear": max_year,
                "nextYear": next_year,
                "companyCode": row["MaCP"],
                "features": features,
                "predictions": prediction["predictions"],
                "ensemble": prediction["ensemble"],
            },
            "companies": df["MaCP"].dropna().unique().tolist(),
        })
    except Exception as exc:
        return api_error(str(exc), 500)


@app.post("/predict-from-file-by-company")
def predict_from_file_by_company():
    try:
        payload = request.get_json(silent=True) or {}
        company_code = payload.get("company")
        if not company_code:
            return api_error("Thiếu mã công ty.", 400)
        path = UPLOAD_DIR / "processed_BCTC_by_year.csv"
        if not path.exists():
            return api_error("Chưa có file processed_BCTC_by_year.csv trong uploads.", 400)
        df = normalize_training_frame(pd.read_csv(path))
        company_df = df[df["MaCP"] == company_code]
        if company_df.empty:
            return api_error(f"Không tìm thấy công ty {company_code}.", 400)
        row = company_df[company_df["Year"] == company_df["Year"].max()].iloc[0]
        feature_schema = infer_feature_schema()
        features = {name: float(row.get(name, 0.0) if pd.notna(row.get(name, 0.0)) else 0.0) for name in feature_schema}
        prediction = predict_all_models(features)
        return jsonify({
            "success": True,
            "data": {
                "maxYear": int(company_df["Year"].max()),
                "nextYear": int(company_df["Year"].max()) + 1,
                "companyCode": company_code,
                "features": features,
                "predictions": prediction["predictions"],
                "ensemble": prediction["ensemble"],
            },
        })
    except Exception as exc:
        return api_error(str(exc), 500)


@app.get("/company-probability")
def get_company_probability():
    try:
        company = request.args.get("company")
        result = {"company": company, "models": {}}
        for model_type in list(MODEL_TYPES) + ["ensemble"]:
            path = RESULTS_DIR / model_type / "predictions_with_probability.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if company is None and not df.empty:
                company = str(df["MaCP"].iloc[0])
                result["company"] = company
            company_df = df[df["MaCP"] == company]
            if company_df.empty:
                continue
            result["models"][model_type] = (
                company_df.groupby("Year")["Probability"].mean().reset_index().rename(columns={"Probability": "probability"}).to_dict("records")
            )
        if not result["models"]:
            return api_error("Không có dữ liệu xác suất cho công ty.", 404)
        return jsonify({"success": True, "data": result})
    except Exception as exc:
        return api_error(str(exc), 500)


@app.get("/default-rate")
def get_default_rate():
    try:
        model_type = request.args.get("model_type", "ensemble")
        path = RESULTS_DIR / model_type / "predictions_with_probability.csv"
        if not path.exists():
            return api_error("Chưa có file dự báo.", 404)
        df = pd.read_csv(path)
        company_year = df.groupby(["MaCP", "Year"])["Probability"].mean().reset_index()
        company_avg = df.groupby("MaCP")["Probability"].mean().reset_index().sort_values("Probability")
        return jsonify({
            "success": True,
            "model": model_type,
            "data": {
                "byCompanyYear": company_year.rename(columns={"MaCP": "company", "Year": "year", "Probability": "default_rate"}).to_dict("records"),
                "lowest": company_avg.head(5).rename(columns={"MaCP": "company", "Probability": "avg_prob"}).to_dict("records"),
                "highest": company_avg.tail(5).rename(columns={"MaCP": "company", "Probability": "avg_prob"}).to_dict("records"),
            },
        })
    except Exception as exc:
        return api_error(str(exc), 500)


@app.post("/preprocess")
def preprocess_data():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return api_error("Không có file được chọn.", 400)
        if not allowed_file(file.filename, ALLOWED_PREPROCESS):
            return api_error("Chỉ hỗ trợ CSV/XLS/XLSX.", 400)
        filename = secure_filename(file.filename)
        temp_path = UPLOAD_DIR / f"temp_{filename}"
        file.save(temp_path)
        processed_filename = f"processed_{Path(filename).stem}.csv"
        processed_path = UPLOAD_DIR / processed_filename
        processed_df = process_raw_data(str(temp_path), str(processed_path))
        if processed_filename != "processed_BCTC_by_year.csv":
            processed_df.to_csv(UPLOAD_DIR / "processed_BCTC_by_year.csv", index=False, encoding="utf-8-sig")
        temp_path.unlink(missing_ok=True)
        return jsonify({
            "success": True,
            "filename": processed_filename,
            "download_url": f"/download/{processed_filename}",
            "shape": list(processed_df.shape),
            "columns": list(processed_df.columns),
            "preview": processed_df.head(10).to_dict("records"),
            "statistics": {
                "rows": int(len(processed_df)),
                "columns": int(len(processed_df.columns)),
                "companies": int(processed_df["MaCP"].nunique()) if "MaCP" in processed_df.columns else 0,
                "years": int(processed_df["Year"].nunique()) if "Year" in processed_df.columns else 0,
            },
        })
    except Exception as exc:
        return api_error(str(exc), 500)


@app.get("/download/<filename>")
def download_file(filename: str):
    path = (UPLOAD_DIR / secure_filename(filename)).resolve()
    if not path.exists() or not str(path).startswith(str(UPLOAD_DIR.resolve())):
        return api_error("File không tồn tại.", 404)
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name=path.name)


@app.get("/debug-models")
def debug_models():
    info: Dict[str, Any] = {}
    for model_type in list(MODEL_TYPES) + ["ensemble"]:
        model_dir = RESULTS_DIR / model_type / "final_model"
        info[model_type] = {
            "path": str(model_dir),
            "exists": model_dir.exists(),
            "files": [p.name for p in model_dir.iterdir()] if model_dir.exists() else [],
            "metrics": load_metrics(model_type),
        }
    return jsonify(info)


@app.post("/test-single-predict")
def test_single_predict():
    try:
        payload = request.get_json(silent=True) or {}
        model_type = payload.get("model_type", "lr")
        model = load_model_artifact(model_type)
        schema = load_feature_schema(model_type)
        features = {name: 0.5 for name in schema}
        prob = float(model.predict_proba(prepare_single_input(features, schema))[:, 1][0])
        return jsonify({"success": True, "model": model_type, "probability": round(prob * 100, 2), "features": schema})
    except Exception as exc:
        return api_error(str(exc), 500)


@app.get("/best-params")
def get_best_params():
    params = {model_type: load_json(RESULTS_DIR / model_type / "final_model" / "best_params.json", {}) for model_type in MODEL_TYPES}
    return jsonify({"success": True, "best_params": params, "models": list(MODEL_TYPES)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
