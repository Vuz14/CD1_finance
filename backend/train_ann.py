
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
from params_manager import load_best_params, save_best_params, load_best_metrics, save_best_metrics, compare_metrics

def log(msg):
    import sys
    print(msg, file=sys.stdout)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class ANN(nn.Module):
    def __init__(self, input_dim, dropout=0.2, batchnorm=True):
        super().__init__()
        layers = []

        def block(in_f, out_f, bn=True, drop=0.0):
            m = [nn.Linear(in_f, out_f), nn.ReLU()]
            if bn:
                m.append(nn.BatchNorm1d(out_f))
            if drop > 0:
                m.append(nn.Dropout(drop))
            return m

        layers += block(input_dim, 128, batchnorm, dropout)
        layers += block(128, 64, batchnorm, dropout)
        layers += block(64, 32, batchnorm, dropout)
        layers += [nn.Linear(32, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ANNSklearnWrapper:
    def __init__(self, model, scaler, device=torch.device('cpu')):
        self.model = model
        self.scaler = scaler
        self.device = device
    
    def predict_proba(self, X):
        """Predict probability (sklearn-like interface)"""
        import numpy as np
        import pandas as pd
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 32):
                xb = torch.tensor(X_scaled[i:i+32], dtype=torch.float32)
                preds = torch.sigmoid(self.model(xb.to(self.device))).cpu().numpy().flatten()
                probs.extend(preds)
        
        probs = np.array(probs)
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class (sklearn-like interface)"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


#  DATASET LOADER (STANDARDIZED)
def get_dataset_path():
    """Find dataset in expected locations"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(backend_dir, "Dataset_CD1_preprocessed_1.csv"),
        os.path.join(backend_dir, "uploads", "Dataset_CD1_preprocessed_1.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Dataset not found. Searched: {candidates}")

def load_dataset():
    """Load and preprocess dataset for time-series"""
    dataset_path = get_dataset_path()
    df = pd.read_csv(dataset_path)
    
    # Remove non-feature columns
    to_drop = [col for col in ['STT', 'MaCP', 'split', 'default_final', 'Year'] if col in df.columns]
    X = df.drop(columns=to_drop)
    y = df.get('default_final', pd.Series(0, index=df.index))
    years = df.get('Year', pd.Series(1, index=df.index))
    
    return X, y, years, df[['STT', 'MaCP']].reset_index(drop=True)

# FEATURE ANALYSIS

def grey_relational_analysis(X, y, feature_names, rho=0.5):
    """Grey Relational Analysis"""
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

def run_influence_analysis_ann(model, X_train, y_train, save_dir, feature_names):
    """Run complete influence analysis"""
    print("Running feature influence analysis...")
    
    # Gradient importance
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(DEVICE)
    model.eval()
    with torch.enable_grad():
        output = torch.sigmoid(model(X_tensor))
        loss = output.sum()
        loss.backward()
    
    if X_tensor.grad is not None:
        gradients = X_tensor.grad.abs().mean(dim=0).detach().cpu().numpy()
    else:
        gradients = np.ones(len(feature_names))
    
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": gradients
    }).sort_values("importance", ascending=False)
    
    df_imp.to_csv(os.path.join(save_dir, "gradient_importance_ann.csv"), index=False)
    
    # GRA
    df_gra = grey_relational_analysis(X_train, y_train, feature_names)
    df_gra.to_csv(os.path.join(save_dir, "gra_scores.csv"), index=False)
    
    # Combine
    df = pd.merge(df_imp, df_gra, on="feature", how="inner")
    df["imp_rank"] = df["importance"].rank(ascending=False)
    df["gra_rank"] = df["gra_score"].rank(ascending=False)
    df["composite_score"] = 0.6 * (1 - df["imp_rank"] / len(df)) + 0.4 * (1 - df["gra_rank"] / len(df))
    df = df.sort_values("composite_score", ascending=False)
    
    df.to_csv(os.path.join(save_dir, "feature_influence_analysis_ann.csv"), index=False)
    print(f"Feature analysis saved")

# TIME-SERIES EXPANDING WINDOW

MIN_TRAIN_YEARS = 1 

def estimate_pd_time_series_ann(X, y, years, dropout, batchnorm, lr, weight_decay, n_trials=15):
    metrics_by_year = {}
    all_params = []  # Collect params from each year
    predictions_all = []  # Collect all predictions for saving to CSV
    unique_years = sorted(years.unique())
    
    # Get original dataframe for STT, MaCP, default_final
    dataset_path = get_dataset_path()
    df_original = pd.read_csv(dataset_path)
    
    if len(unique_years) < MIN_TRAIN_YEARS:
        print(f"Need at least {MIN_TRAIN_YEARS} years, have {len(unique_years)}")
        return {}, {}
    
    for predict_year in unique_years[MIN_TRAIN_YEARS:]:
        train_idx = years < predict_year
        test_idx = years == predict_year
        
        if sum(test_idx) == 0:
            continue
        
        X_train, X_test = X[train_idx].values, X[test_idx].values
        y_train, y_test = y[train_idx].values, y[test_idx].values
        
        print(f"  Year {predict_year}: train={len(X_train)}, test={len(X_test)}", end="")
        
        # Optuna for this year
        def objective(trial):
            t_dropout = trial.suggest_float("dropout", 0.05, 0.5)
            t_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            t_wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
            
            from sklearn.model_selection import StratifiedKFold
            n_splits = min(3, max(2, sum(train_idx) // 20))
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            auc_scores = []
            
            scaler = MinMaxScaler()
            X_tr_scaled = scaler.fit_transform(X_train)
            
            for tr_i, val_i in kf.split(X_tr_scaled, y_train):
                X_tr, X_val = X_tr_scaled[tr_i], X_tr_scaled[val_i]
                y_tr, y_val = y_train[tr_i], y_train[val_i]
                
                model = ANN(X_train.shape[1], t_dropout, batchnorm).to(DEVICE)
                pos_w = torch.tensor([(len(y_tr) - sum(y_tr)) / (sum(y_tr) + 1e-6)], dtype=torch.float32).to(DEVICE)
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
                opt = optim.AdamW(model.parameters(), lr=t_lr, weight_decay=t_wd)
                
                # Adjust batch size to avoid BatchNorm issues with small batches
                batch_size = min(32, max(2, len(X_train) // 4))  # Ensure batch_size >= 2 for BatchNorm
                
                loader_tr = DataLoader(
                    TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)),
                    batch_size=batch_size, shuffle=True, drop_last=True
                )
                loader_val = DataLoader(
                    TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)),
                    batch_size=batch_size, shuffle=False, drop_last=False
                )
                
                for _ in range(30):
                    model.train()
                    for xb, yb in loader_tr:
                        opt.zero_grad()
                        loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
                        loss.backward()
                        opt.step()
                
                model.eval()
                val_preds = []
                with torch.no_grad():
                    for xb, _ in loader_val:
                        val_preds.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
                
                auc = roc_auc_score(y_val, val_preds)
                auc_scores.append(auc)
            
            return np.mean(auc_scores)
        
        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_dropout = study.best_params.get("dropout", dropout)
        best_lr = study.best_params.get("lr", lr)
        best_wd = study.best_params.get("weight_decay", weight_decay)
        
        # Train on full train set with best params
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = ANN(X_train.shape[1], best_dropout, batchnorm).to(DEVICE)
        pos_w = torch.tensor([(len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)], dtype=torch.float32).to(DEVICE)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
        
        # Adjust batch size to avoid BatchNorm issues with small batches
        batch_size = min(32, max(2, len(X_train) // 4))  # Ensure batch_size >= 2 for BatchNorm
        
        loader = DataLoader(
            TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        
        for _ in range(50):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
                loss.backward()
                opt.step()
        
        # Predict
        model.eval()
        test_preds = []
        with torch.no_grad():
            for i in range(0, len(X_test_scaled), 32):
                xb = torch.tensor(X_test_scaled[i:i+32], dtype=torch.float32)
                test_preds.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
        
        test_pred_labels = (np.array(test_preds) > 0.5).astype(int)
        
        metrics_by_year[predict_year] = {
            "Accuracy": accuracy_score(y_test, test_pred_labels),
            "Precision": precision_score(y_test, test_pred_labels, zero_division=0),
            "Recall": recall_score(y_test, test_pred_labels, zero_division=0),
            "F1": f1_score(y_test, test_pred_labels, zero_division=0),
            "AUC": roc_auc_score(y_test, test_preds)
        }
        
        print(f", AUC={metrics_by_year[predict_year]['AUC']:.4f}")
        
        # Collect predictions for this year
        test_indices = np.where(test_idx)[0]
        df_test = df_original.iloc[test_indices].copy()
        df_test['Prediction'] = test_pred_labels
        df_test['Probability'] = (np.array(test_preds) * 100).round(2)
        predictions_all.append(df_test)
        
        # Collect params from this year
        all_params.append({
            "dropout": best_dropout,
            "lr": best_lr,
            "weight_decay": best_wd,
            "auc": metrics_by_year[predict_year]['AUC']
        })
    
    # Select best params based on AVERAGE AUC across all years (more stable)
    # This prevents overfitting to a single year's peak performance
    if all_params:
        avg_auc = np.mean([p['auc'] for p in all_params])
        best_params_dict = min(all_params, key=lambda x: abs(x['auc'] - avg_auc))  # Closest to average
        best_params = {
            "dropout": best_params_dict["dropout"],
            "learning_rate": best_params_dict["lr"],
            "weight_decay": best_params_dict["weight_decay"],
            "batchnorm": batchnorm
        }
        log(f"📊 Selected params from year with AUC closest to average ({avg_auc:.4f})")
    else:
        best_params = {
            "dropout": dropout,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batchnorm": batchnorm
        }
    
    return metrics_by_year, best_params, predictions_all

# ===========================================
# 5️⃣ MAIN TRAIN FUNCTION
# ===========================================
def train_ann(save_dir="results/ann", dropout=None, batchnorm=True, lr=None, weight_decay=None, n_trials=15):
    """Time-series expanding window training with Optuna"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load previous best params if available
    old_best_params = load_best_params(save_dir, "ANN")
    old_metrics = load_best_metrics(save_dir, "ANN")
    
    # Use provided params or defaults or previous best
    if dropout is None:
        dropout = old_best_params.get("dropout", 0.1488) if old_best_params else 0.1488
    if lr is None:
        lr = old_best_params.get("learning_rate", 0.00659) if old_best_params else 0.00659
    if weight_decay is None:
        weight_decay = old_best_params.get("weight_decay", 5.98e-06) if old_best_params else 5.98e-06
    if old_best_params:
        batchnorm = old_best_params.get("batchnorm", True)
    
    print(f"Using parameters: dropout={dropout:.4f}, lr={lr:.6f}, weight_decay={weight_decay:.2e}")
    if old_best_params:
        print("(Loaded from previous best params)")
    
    print("Loading dataset...")
    X, y, years, metadata = load_dataset()
    print(f"{len(X)} samples, {X.shape[1]} features, {len(years.unique())} years")
    
    # Time-series validation
    print("Time-series expanding window validation...")
    metrics_by_year, best_params, predictions_all = estimate_pd_time_series_ann(X, y, years, dropout, batchnorm, lr, weight_decay, n_trials)
    
    # Average metrics
    if metrics_by_year:
        avg_metrics = {k: np.mean([m[k] for m in metrics_by_year.values()]) for k in list(metrics_by_year.values())[0].keys()}
        print(f"\nAverage Metrics:")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")
    else:
        print("No metrics calculated")
        avg_metrics = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0, "AUC": 0}
    
    # Compare with previous best metrics
    print("\nComparing with previous training...")
    params_to_use = best_params
    
    if old_best_params:
        log(f"📊 Comparing: Old params vs New params from CV")
        log(f"   Old AUC: {old_metrics.get('AUC', 0):.4f}")
        log(f"   New AUC: {avg_metrics.get('AUC', 0):.4f}")
        
        if compare_metrics(old_metrics, avg_metrics):
            log(f"✅ New params are better!")
            params_to_use = best_params
        else:
            log(f"📌 Old params are still better, using previous best")
            params_to_use = old_best_params
            avg_metrics = old_metrics
    else:
        log(f"📊 First run, using params from CV tuning")
        params_to_use = best_params
    
    # Save the best params we decided to use
    save_best_params(params_to_use, save_dir)
    
    # Save metrics only if improved (or first time)
    if compare_metrics(old_metrics, avg_metrics) or old_metrics is None:
        save_best_metrics(avg_metrics, save_dir)
        log(f"✅ Saved new best metrics")
    else:
        log(f"📌 Kept previous metrics")
    
    # Save metrics by year
    if metrics_by_year:
        pd.DataFrame(metrics_by_year).T.to_csv(os.path.join(save_dir, "pd_time_series.csv"))
    
    # Save predictions to CSV for frontend
    if predictions_all:
        # Concat all prediction DataFrames from each year
        df_predictions = pd.concat(predictions_all, ignore_index=True)
        
        # Select and reorder columns to match expected format
        if 'STT' in df_predictions.columns and 'MaCP' in df_predictions.columns:
            cols = ['STT', 'MaCP', 'Year', 'default_final', 'Prediction', 'Probability']
            df_predictions = df_predictions[[col for col in cols if col in df_predictions.columns]]
            df_predictions.to_csv(os.path.join(save_dir, "predictions_with_probability.csv"), index=False)
            print(f"✅ Saved {len(df_predictions)} predictions to CSV")
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    print(f"Using best params: dropout={params_to_use['dropout']:.4f}, lr={params_to_use['learning_rate']:.6f}, weight_decay={params_to_use['weight_decay']:.2e}")
    
    X_vals = X.values
    y_vals = y.values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_vals)
    
    model = ANN(X_vals.shape[1], params_to_use['dropout'], params_to_use['batchnorm']).to(DEVICE)
    pos_weight = torch.tensor([(len(y_vals) - sum(y_vals)) / (sum(y_vals) + 1e-6)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=params_to_use['learning_rate'], weight_decay=params_to_use['weight_decay'])
    
    # Adjust batch size to avoid BatchNorm issues with small batches
    batch_size = min(32, max(2, len(X_vals) // 4))  # Ensure batch_size >= 2 for BatchNorm
    
    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_vals, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            optimizer.step()
    
    # Save final model
    final_model_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(final_model_dir, "ann_final_model.pth"))
    
    # Create wrapper and save
    wrapper_model = ANNSklearnWrapper(model, scaler, DEVICE)
    joblib.dump(wrapper_model, os.path.join(final_model_dir, "pd_ann_model.pkl"))
    
    with open(os.path.join(final_model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(final_model_dir, "feature_schema.json"), "w") as f:
        json.dump({"features": X.columns.tolist()}, f, indent=4)
    
    # Save best parameters
    save_best_params(best_params, save_dir)
    
    # NOTE: Predictions are already saved from expanding window validation
    # Do NOT generate predictions using final model on all data, as it would violate time-series integrity
    # (model trained on future data would predict past years)
    # Frontend uses predictions_with_probability.csv from expanding window validation (2019-2024)
    
    # Generate predictions for 2025 using final model (synthetic future year)
    print("Generating predictions for 2025 (synthetic future year using final model)...")
    model.eval()
    preds_2025 = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), 32):
            xb = torch.tensor(X_scaled[i:i+32], dtype=torch.float32)
            preds_2025.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
    
    # Create 2025 predictions dataframe
    dataset_path = get_dataset_path()
    df_original = pd.read_csv(dataset_path)
    
    predictions_2025 = df_original[["STT", "MaCP", "default_final"]].copy()
    predictions_2025["Year"] = 2025
    predictions_2025["Prediction"] = (np.array(preds_2025) > 0.5).astype(int)
    predictions_2025["Probability"] = (np.array(preds_2025) * 100).round(2)
    
    # Append 2025 predictions to CSV
    csv_path = os.path.join(save_dir, "predictions_with_probability.csv")
    try:
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            # Ensure column order matches
            predictions_2025_reordered = predictions_2025[existing_df.columns]
            combined_df = pd.concat([existing_df, predictions_2025_reordered], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
            print(f"✅ Added 2025 predictions to CSV. Total years: {combined_df['Year'].unique()}")
        else:
            print(f"⚠️ CSV not found at {csv_path}")
    except Exception as e:
        print(f"❌ Error appending 2025 predictions: {e}")
    
    # Save metrics
    with open(os.path.join(save_dir, "metrics_test.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)
    
    # Feature analysis
    feature_names = X.columns.tolist()
    run_influence_analysis_ann(model, X_vals, y_vals, save_dir, feature_names)
    
    print(f"\nModel saved to {final_model_dir}")
    return avg_metrics

# ===========================================
# 6️⃣ STANDALONE EXECUTION
# ===========================================
if __name__ == "__main__":
    train_ann()
