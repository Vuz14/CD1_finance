"""
Parameter management utility for model training
Handles loading, comparing, and saving optimal parameters
"""
import json
import os
from typing import Dict, Any, Optional

def load_best_params(model_dir: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Load best parameters from previous training"""
    params_file = os.path.join(model_dir, "final_model", "best_params.json")
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading best_params for {model_name}: {e}")
    return None

def save_best_params(params: Dict[str, Any], model_dir: str) -> None:
    """Save best parameters to file"""
    model_final_dir = os.path.join(model_dir, "final_model")
    os.makedirs(model_final_dir, exist_ok=True)
    params_file = os.path.join(model_final_dir, "best_params.json")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)

def load_best_metrics(model_dir: str, model_name: str) -> Optional[Dict[str, float]]:
    """Load best metrics from previous training"""
    metrics_file = os.path.join(model_dir, "metrics_test.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metrics for {model_name}: {e}")
    return None

def save_best_metrics(metrics: Dict[str, float], model_dir: str) -> None:
    """Save best metrics to file"""
    metrics_file = os.path.join(model_dir, "metrics_test.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def compare_metrics(old_metrics: Optional[Dict[str, float]], new_metrics: Dict[str, float]) -> bool:
    """
    Compare metrics and return True if new is better
    Uses AUC as primary metric, then Accuracy as tiebreaker
    """
    if old_metrics is None:
        return True
    
    # Primary metric: AUC
    old_auc = old_metrics.get("AUC", 0)
    new_auc = new_metrics.get("AUC", 0)
    
    if new_auc > old_auc:
        improvement = ((new_auc - old_auc) / old_auc * 100) if old_auc > 0 else 0
        print(f"✓ Better metrics found! AUC: {old_auc:.4f} → {new_auc:.4f} (+{improvement:.2f}%)")
        return True
    elif new_auc == old_auc:
        # Tiebreaker: Accuracy
        old_acc = old_metrics.get("Accuracy", 0)
        new_acc = new_metrics.get("Accuracy", 0)
        if new_acc > old_acc:
            improvement = ((new_acc - old_acc) / old_acc * 100) if old_acc > 0 else 0
            print(f"✓ Better accuracy found! {old_acc:.4f} → {new_acc:.4f} (+{improvement:.2f}%)")
            return True
    
    print(f"No improvement. Keeping previous metrics. (AUC: {new_auc:.4f} vs {old_auc:.4f})")
    return False
