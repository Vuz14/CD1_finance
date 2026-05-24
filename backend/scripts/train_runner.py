from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.config import MODEL_TYPES, setup_utf8
from models.base_models import train_model
from models.ensemble_model import EnsembleStackingClassifier


def run_training(model_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    setup_utf8()
    params = params or {}
    if model_type == "all":
        results: Dict[str, Any] = {}
        for name in MODEL_TYPES:
            result = train_model(name, params.get(name, {}))
            results[name] = result.metrics
        results["ensemble"] = EnsembleStackingClassifier().fit()
        return {"success": True, "model_type": "all", "metrics": results}
    if model_type == "ensemble":
        metrics = EnsembleStackingClassifier().fit()
        return {"success": True, "model_type": "ensemble", "metrics": metrics}
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model type không hợp lệ: {model_type}")
    result = train_model(model_type, params)
    return {
        "success": True,
        "model_type": model_type,
        "metrics": result.metrics,
        "selected_features": result.selected_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CD1 finance training runner")
    parser.add_argument("model_type", choices=list(MODEL_TYPES) + ["ensemble", "all"])
    parser.add_argument("--params", default="{}", help="JSON hyperparameters")
    args = parser.parse_args()
    try:
        params = json.loads(args.params)
        print(json.dumps(run_training(args.model_type, params), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
