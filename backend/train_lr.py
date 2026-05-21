
import os
import json
import optuna
import joblib
import numpy as np
import pandas as pd


from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from params_manager import load_best_params, save_best_params, load_best_metrics, save_best_metrics, compare_metrics

DATA_PATH = "Dataset_CD1_preprocessed_1.csv"
SAVE_DIR = "results/lr"
MODEL_DIR = os.path.join(SAVE_DIR, "final_model")

MIN_TRAIN_YEARS = 1  
N_OPTUNA_TRIALS = 30
RANDOM_STATE = 42

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= HELPER FUNCTIONS =================

def get_dataset_path(filename="Dataset_CD1_preprocessed_1.csv"):
    """Tìm dataset ở vị trí đúng"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(backend_dir, filename),
        os.path.join(backend_dir, "uploads", filename),
        filename
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Dataset '{filename}' không tìm thấy. Tìm kiếm tại: {possible_paths}")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def eval_metrics(y_true, y_pred, y_prob):
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def grey_relational_analysis(X, y, feature_names, rho=0.5):
    """Grey Relational Analysis"""
    from sklearn.preprocessing import MinMaxScaler
    
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X
    
    scaler = MinMaxScaler()
    Xn = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)
    
    if isinstance(y, np.ndarray):
        y_series = pd.Series(y)
    else:
        y_series = y.reset_index(drop=True)
    
    y0 = (y_series - y_series.min()) / (y_series.max() - y_series.min() + 1e-6)
    
    ref = y0.values
    diff = np.abs(Xn.values - ref.reshape(-1, 1))
    min_diff, max_diff = diff.min(), diff.max()
    
    xi = (min_diff + rho * max_diff) / (diff + rho * max_diff)
    gra_scores = xi.mean(axis=0)
    
    df_gra = pd.DataFrame({
        "feature": X_df.columns,
        "gra_score": gra_scores
    }).sort_values("gra_score", ascending=False)
    
    return df_gra

def run_influence_analysis_lr(model, X_train, y_train, save_dir, feature_names):
    """Run complete influence analysis for Logistic Regression"""
    log("Running feature influence analysis...")
    
    # Get coefficients from logistic regression
    clf = model.named_steps['clf']
    if hasattr(clf, 'coef_'):
        coefficients = np.abs(clf.coef_[0])
    else:
        # If it's wrapped by CalibratedClassifierCV, extract the base estimator
        if hasattr(clf, 'base_estimator_'):
            coefficients = np.abs(clf.base_estimator_.named_steps['clf'].coef_[0])
        else:
            log("WARNING: Could not extract coefficients from model")
            return
    
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": coefficients
    }).sort_values("importance", ascending=False)
    
    df_imp.to_csv(os.path.join(save_dir, "feature_importance_lr.csv"), index=False)
    
    # GRA
    df_gra = grey_relational_analysis(X_train, y_train, feature_names)
    df_gra.to_csv(os.path.join(save_dir, "gra_scores.csv"), index=False)
    
    # Combine
    df = pd.merge(df_imp, df_gra, on="feature", how="inner")
    df["imp_rank"] = df["importance"].rank(ascending=False)
    df["gra_rank"] = df["gra_score"].rank(ascending=False)
    df["composite_score"] = 0.6 * (1 - df["imp_rank"] / len(df)) + 0.4 * (1 - df["gra_rank"] / len(df))
    df = df.sort_values("composite_score", ascending=False)
    
    df.to_csv(os.path.join(save_dir, "feature_influence_analysis_lr.csv"), index=False)
    log("Feature analysis saved")


def optuna_objective(trial, X, y):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    C = trial.suggest_float("C", 0.01, 10.0, log=True)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            penalty=penalty,
            C=C,
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_prob)

# ================= PD SMOOTHING =================

def smooth_pd_by_cycle(df, window=3):
    df = df.sort_values(["MaCP", "Year"])
    df["pd_smooth"] = (
        df.groupby("MaCP")["pd_default"]
        .transform(lambda x: x.rolling(window, center=True, min_periods=1).mean())
    )
    return df

# ================= TIME-SERIES BACKTEST =================

def estimate_pd_time_series():
    log("Bắt đầu Logistic Regression TIME-SERIES (Optuna + Expanding Window)")

    df = pd.read_csv(get_dataset_path()).sort_values(["MaCP", "Year"])
    years = sorted(df["Year"].unique())

    all_pd, all_metrics = [], []
    all_params = []  # Track params from each year's Optuna tuning
    DROP_COLS = ["default_final", "STT", "MaCP", "Year", "split"]

    for i, year in enumerate(years):
        if i < MIN_TRAIN_YEARS:
            continue  # Bỏ qua các năm đầu chưa đủ dữ liệu (cần ít nhất MIN_TRAIN_YEARS năm để train)

        log(f"📅 Train ≤ {year-1} → Predict {year}")

        train_df = df[df["Year"] < year]
        test_df  = df[df["Year"] == year]

        X_train = train_df.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=[np.number])
        y_train = train_df["default_final"]
        X_test  = test_df.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=[np.number])
        y_test  = test_df["default_final"]

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: optuna_objective(t, X_train, y_train), n_trials=N_OPTUNA_TRIALS)
        best = study.best_params
        all_params.append(best)  # Collect params from each fold

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="saga",
                penalty=best["penalty"],
                C=best["C"],
                max_iter=4000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])

        try:
            model = CalibratedClassifierCV(model, cv=3)
        except:
            pass

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        m = eval_metrics(y_test, y_pred, y_prob)
        m["Year"] = year
        all_metrics.append(m)

        res = test_df.copy()
        res["pd_default"] = y_prob
        all_pd.append(res)

    pd_ts = pd.concat(all_pd)
    pd_ts = smooth_pd_by_cycle(pd_ts)

    pd_ts.to_csv(os.path.join(SAVE_DIR, "pd_time_series.csv"), index=False)
    pd.DataFrame(all_metrics).to_csv(os.path.join(SAVE_DIR, "metrics_by_year.csv"), index=False)

    # Calculate overall metrics from all years combined
    y_true_all = pd_ts["default_final"]
    y_prob_all = pd_ts["pd_default"]
    y_pred_all = (y_prob_all >= 0.5).astype(int)
    overall_metrics = eval_metrics(y_true_all, y_pred_all, y_prob_all)
    with open(os.path.join(SAVE_DIR, "metrics_test.json"), "w") as f:
        json.dump(overall_metrics, f, indent=4)
    
    # Lưu predictions_with_probability.csv cho frontend (company trends)
    predictions_df = pd_ts[["STT", "MaCP", "Year", "default_final"]].copy()
    predictions_df["Prediction"] = (pd_ts["pd_default"] > 0.5).astype(int)
    predictions_df["Probability"] = (pd_ts["pd_default"] * 100).round(2)
    predictions_df.to_csv(os.path.join(SAVE_DIR, "predictions_with_probability.csv"), index=False)

    log("✅ Hoàn tất backtest time-series + PD smoothing")

    metrics_by_year = pd.DataFrame(all_metrics)
    avg_auc = metrics_by_year["AUC"].mean()
    best_year_idx = (metrics_by_year["AUC"] - avg_auc).abs().idxmin()  # Closest to average
    best_params = all_params[best_year_idx]
    log(f"📊 Selected params from year with AUC closest to average ({avg_auc:.4f})")
    
    return df, overall_metrics, best_params

# ================= FINAL MODEL FOR DEPLOY =================

def train_and_save_final_model(df, best_params_from_cv, current_metrics):
    log("🚀 Train FINAL MODEL cho dự báo tương lai")

    DROP_COLS = ["default_final", "STT", "MaCP", "Year", "split"]
    X = df.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=[np.number])
    y = df["default_final"]

    # Try to load previous best params and metrics
    old_best_params = load_best_params(SAVE_DIR, "LR")
    old_metrics = load_best_metrics(SAVE_DIR, "LR")
    
    # Decide which params to use
    params_to_use = best_params_from_cv
    if old_best_params is not None:
        log(f"📊 Comparing: Old params vs New params from CV")
        log(f"   Old AUC: {old_metrics.get('AUC', 0):.4f}")
        log(f"   New AUC: {current_metrics.get('AUC', 0):.4f}")
        
        if compare_metrics(old_metrics, current_metrics):
            log(f"✅ New params are better!")
            params_to_use = best_params_from_cv
        else:
            log(f"📌 Old params are still better, using previous best")
            params_to_use = old_best_params
    else:
        log(f"📊 First run, using params from CV tuning")
        params_to_use = best_params_from_cv

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            penalty=params_to_use["penalty"],
            C=params_to_use["C"],
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    final_model.fit(X, y)

    joblib.dump(final_model, os.path.join(MODEL_DIR, "pd_lr_model.pkl"))
    json.dump({"features": X.columns.tolist()}, open(os.path.join(MODEL_DIR, "feature_schema.json"), "w"), indent=4)

    log("Generating predictions for 2025 (synthetic future year using final model)...")
    y_prob_2025 = final_model.predict_proba(X)[:, 1]
    predictions_2025 = df[["STT", "MaCP", "default_final"]].copy()
    predictions_2025["Year"] = 2025
    predictions_2025["Prediction"] = (y_prob_2025 > 0.5).astype(int)
    predictions_2025["Probability"] = (y_prob_2025 * 100).round(2)
    
    # Append 2025 predictions to CSV
    csv_path = os.path.join(SAVE_DIR, "predictions_with_probability.csv")
    try:
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            predictions_2025_reordered = predictions_2025[existing_df.columns]
            combined_df = pd.concat([existing_df, predictions_2025_reordered], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
            print(f"✅ Added 2025 predictions to CSV. Total years: {combined_df['Year'].unique()}")
        else:
            print(f"⚠️ CSV not found at {csv_path}")
    except Exception as e:
        print(f"❌ Error appending 2025 predictions: {e}")

    # Feature analysis
    feature_names = X.columns.tolist()
    run_influence_analysis_lr(final_model, X.values, y.values, SAVE_DIR, feature_names)

    # Save the best params we decided to use
    save_best_params(params_to_use, SAVE_DIR)
    
    # Save metrics only if improved (or first time)
    if compare_metrics(old_metrics, current_metrics) or old_metrics is None:
        save_best_metrics(current_metrics, SAVE_DIR)
        log(f"✅ Saved new best metrics")
    else:
        log(f"📌 Kept previous metrics")

    log("✅ Final model đã sẵn sàng cho Web/API")


def train_lr(save_dir="results/lr", C=None, penalty=None, **kwargs):
    """Wrapper function để train_wrapper.py gọi (support hyperparameter overrides)"""
    global SAVE_DIR, MODEL_DIR
    SAVE_DIR = save_dir
    MODEL_DIR = os.path.join(save_dir, "final_model")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Chạy time-series training (now returns metrics and best_params too)
    df_full, current_metrics, best_params = estimate_pd_time_series()
    train_and_save_final_model(df_full, best_params, current_metrics)
    
    log("✅ Hoàn tất huấn luyện Logistic Regression")
    
    # Trả về metrics
    return current_metrics


if __name__ == "__main__":
    df = pd.read_csv(get_dataset_path())
    
    df_full, metrics, best_params = estimate_pd_time_series()
    train_and_save_final_model(df_full, best_params, metrics)
    
    log("🎉 Hệ thống PD Time-Series + PD smoothing hoàn tất")
