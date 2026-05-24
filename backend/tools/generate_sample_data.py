from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_dataset(output_path: str | None = None, n_companies: int = 40, start_year: int = 2018, end_year: int = 2024) -> Path:
    rng = np.random.default_rng(42)
    rows = []
    stt = 1
    for company_idx in range(1, n_companies + 1):
        company = f"CP{company_idx:03d}"
        base_risk = rng.beta(2.0, 6.0)
        for year in range(start_year, end_year + 1):
            x = rng.normal(0.0, 1.0, 13)
            logit = -1.4 + 1.1 * x[4] - 0.8 * x[1] + 0.5 * x[8] + 0.9 * base_risk + 0.05 * (year - start_year)
            probability = 1.0 / (1.0 + np.exp(-logit))
            default = int(rng.random() < probability)
            rows.append({
                "STT": stt,
                "Ticker": company,
                "Year": year,
                **{f"X{i + 1}": float(x[i]) for i in range(13)},
                "Target": default,
            })
            stt += 1
    out = pd.DataFrame(rows)
    path = Path(output_path) if output_path else Path(__file__).resolve().parents[1] / "dataset" / "Processed_EWS_Final_sample.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    return path


if __name__ == "__main__":
    generated = generate_sample_dataset()
    print(f"Generated sample dataset: {generated}")
