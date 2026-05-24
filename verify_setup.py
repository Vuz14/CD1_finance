from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def ok(message: str) -> bool:
    print(f"[OK] {message}")
    return True


def fail(message: str) -> bool:
    print(f"[FAIL] {message}")
    return False


def test_structure() -> bool:
    required = [
        "backend/api/server.py",
        "backend/core/config.py",
        "backend/core/utils.py",
        "backend/services/preprocessing.py",
        "backend/services/feature_analysis.py",
        "backend/models/base_models.py",
        "backend/models/ensemble_model.py",
        "backend/scripts/train_runner.py",
        "backend/tools/generate_sample_data.py",
        "frontend/package.json",
        "frontend/src/App.tsx",
        "frontend/src/api/client.ts",
        "frontend/src/types/api.ts",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    return fail(f"Missing files: {missing}") if missing else ok("Project structure is clean")


def test_backend_imports() -> bool:
    sys.path.insert(0, str(ROOT / "backend"))
    modules = [
        "flask",
        "flask_cors",
        "pandas",
        "numpy",
        "sklearn",
        "imblearn",
        "xgboost",
        "core.config",
        "models.base_models",
        "models.ensemble_model",
        "scripts.train_runner",
    ]
    try:
        for module in modules:
            importlib.import_module(module)
        return ok("Backend imports are available")
    except Exception as exc:
        return fail(f"Backend import failed: {exc}")


def test_flask_health() -> bool:
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from api.server import app

        response = app.test_client().get("/health")
        if response.status_code != 200:
            return fail(f"/health returned {response.status_code}")
        return ok(f"/health => {response.json}")
    except Exception as exc:
        return fail(f"Flask health test failed: {exc}")


def test_frontend_config() -> bool:
    try:
        package = json.loads((ROOT / "frontend/package.json").read_text(encoding="utf-8"))
        scripts = package.get("scripts", {})
        if "start" not in scripts or "build" not in scripts:
            return fail("Frontend package.json must have start and build scripts")
        return ok("Frontend package.json is runnable")
    except Exception as exc:
        return fail(f"Frontend config failed: {exc}")


def main() -> None:
    print("CD1 Finance setup verification")
    results = [
        test_structure(),
        test_backend_imports(),
        test_flask_health(),
        test_frontend_config(),
    ]
    passed = sum(results)
    print(f"\nPassed {passed}/{len(results)} checks")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
