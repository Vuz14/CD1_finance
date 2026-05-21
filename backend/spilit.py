# utils.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

def load_data(path="Dataset_CD1_preprocessed.csv", label_col="default_final"):
    # Get the absolute path to the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build list of possible paths - use absolute paths for reliability
    possible_paths = [
        path if os.path.isabs(path) else os.path.join(backend_dir, path),
        os.path.join(backend_dir, "uploads", os.path.basename(path)),
        os.path.join(backend_dir, "uploads", path),
    ]
    
    # Also add relative paths
    possible_paths.extend([
        f"uploads/{path}",
        f"uploads/{os.path.basename(path)}",
    ])
    
    actual_path = None
    for p in possible_paths:
        try:
            if os.path.exists(p):
                actual_path = os.path.abspath(p)
                break
        except:
            continue
    
    if actual_path is None:
        raise FileNotFoundError(
            f"Dataset '{os.path.basename(path)}' not found. "
            f"Checked paths: {[os.path.abspath(p) if not os.path.isabs(p) else p for p in possible_paths]}"
        )
    
    df = pd.read_csv(actual_path)
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col, "STT", "MaCP", "Year"], errors='ignore')
    meta = df[[c for c in ["STT","MaCP","Year"] if c in df.columns]]
    return X, y, meta

def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def get_cv(K=5, random_state=42):
    return StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)

def smote_fit_resample(X_train, y_train, smote_seed=42, k_neighbors=5):
    sm = SMOTE(random_state=smote_seed, k_neighbors=k_neighbors)
    Xs, ys = sm.fit_resample(X_train, y_train)
    return Xs, ys

def save_metrics(dest_folder, model_name, metrics_list):
    os.makedirs(dest_folder, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(dest_folder, f"{model_name}_cv_metrics.csv"), index=False)
    return df

def eval_metrics(y_true, y_pred, y_prob=None):
    y_prob = y_prob if y_prob is not None else np.array(y_pred)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, y_prob))
    }

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".pkl") or path.endswith(".joblib"):
        joblib.dump(obj, path)
    else:
        # fallback
        joblib.dump(obj, path + ".joblib")
