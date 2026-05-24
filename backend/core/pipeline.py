from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class FoldResult:
    year: int
    metrics: Dict[str, float]
    psi: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    model_type: str
    metrics: Dict[str, float]
    fold_metrics: List[FoldResult]
    selected_features: List[str]
    output_dir: Path


class BaseTrainingPipeline(ABC):
    model_type: str

    @abstractmethod
    def fit(self, params: Optional[Dict[str, Any]] = None) -> TrainingResult:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
