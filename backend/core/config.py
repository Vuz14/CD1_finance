from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for non-ANN workflows
    torch = None


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODEL_TYPES = ("ann", "lr", "tree", "xgb")
FEATURE_COLUMNS = [f"X{i}" for i in range(1, 14)]
TARGET_COLUMN = "default_final"
ID_COLUMNS = ["STT", "MaCP", "Year"]
DROP_COLUMNS = [TARGET_COLUMN, "STT", "MaCP", "Year", "split"]
RANDOM_STATE = int(os.getenv("CD1_RANDOM_STATE", "42"))
MIN_TRAIN_YEARS = int(os.getenv("CD1_MIN_TRAIN_YEARS", "1"))
DEFAULT_DATASET = os.getenv("CD1_DATASET", "dataset/Processed_EWS_Final.csv")

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def setup_utf8() -> None:
    """Force UTF-8 output streams on Windows terminals."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
    if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("CD1_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("cd1_finance")


def get_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


LOGGER = setup_logging()
setup_utf8()
seed_everything()
