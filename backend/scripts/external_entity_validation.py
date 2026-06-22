from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
RAW_TRAINING = ROOT / "backend" / "dataset" / "BCTC_2015_2024_TongHop.csv"
OUT = ROOT / "research" / "external_validation"
CACHE = OUT / "api_cache"
FEATURES = [
    "ROA", "ROE", "Gross_Margin", "Current_Ratio", "Debt_to_Assets",
    "Cash_Ratio", "Inventory_Turnover", "Receivables_Turnover",
]
SENTINELS = ["VPC", "VST", "TTF", "HVN"]

IDS = {
    "net_income": ["net_profit_loss_after_tax"],
    "revenue": ["net_sales", "revenue"],
    "cogs": ["cost_of_sales", "cost_of_goods_sold"],
    "gross_profit": ["gross_profit"],
    "current_assets": ["current_assets"],
    "current_liabilities": ["current_liabilities"],
    "cash": ["cash_and_cash_equivalents", "cash"],
    "equity": ["owners_equity", "total_equity"],
    "receivables": ["short_term_receivables", "short_term_accounts_receivable", "accounts_receivable"],
    "inventory": ["inventories", "inventories_net"],
    "assets": ["total_assets"],
    "retained": ["undistributed_earnings", "retained_earnings"],
    "capital": ["paid_in_capital", "owners_contributed_capital"],
}


def normalize(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()


def get_value(frame: pd.DataFrame, field: str, year: int) -> float:
    year_col = str(year)
    if frame.empty or year_col not in frame.columns:
        return np.nan
    selected = frame[frame["item_id"].astype(str).isin(IDS[field])]
    if selected.empty:
        return np.nan
    values = pd.to_numeric(selected[year_col], errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else np.nan


def safe_ratio(a: float, b: float) -> float:
    return a / b if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def fetch_report(ticker: str, report: str, delay: float) -> pd.DataFrame:
    path = CACHE / f"{ticker}_{report}.csv"
    if path.exists():
        return pd.read_csv(path)
    from vnstock.api.financial import Finance
    finance = Finance(source="VCI", symbol=ticker, period="year")
    frame = getattr(finance, report)(period="year", lang="vi")
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    time.sleep(delay)
    return frame


def build_row(ticker: str, delay: float) -> dict[str, object] | None:
    try:
        income = fetch_report(ticker, "income_statement", delay)
        balance = fetch_report(ticker, "balance_sheet", delay)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return {"Ticker": ticker, "status": "api_error", "error": str(exc)[:250]}

    ni = {year: get_value(income, "net_income", year) for year in (2022, 2023, 2024)}
    event_equity = get_value(balance, "equity", 2024)
    event_retained = get_value(balance, "retained", 2024)
    event_capital = get_value(balance, "capital", 2024)
    loss3 = all(np.isfinite(ni[year]) and ni[year] < 0 for year in (2022, 2023, 2024))
    accumulated = (
        np.isfinite(event_retained) and np.isfinite(event_capital)
        and event_retained < 0 and abs(event_retained) > event_capital
    )
    negative_equity = np.isfinite(event_equity) and event_equity < 0

    year = 2023
    values = {name: get_value(income if name in {"net_income", "revenue", "cogs", "gross_profit"} else balance, name, year)
              for name in ("net_income", "revenue", "cogs", "gross_profit", "current_assets",
                           "current_liabilities", "cash", "equity", "receivables", "inventory", "assets")}
    liabilities = values["assets"] - values["equity"]
    features = {
        "ROA": safe_ratio(values["net_income"], values["assets"]),
        "ROE": safe_ratio(values["net_income"], values["equity"]),
        "Gross_Margin": safe_ratio(values["gross_profit"], values["revenue"]),
        "Current_Ratio": safe_ratio(values["current_assets"], values["current_liabilities"]),
        "Debt_to_Assets": safe_ratio(liabilities, values["assets"]),
        "Cash_Ratio": safe_ratio(values["cash"], values["current_liabilities"]),
        "Inventory_Turnover": safe_ratio(abs(values["cogs"]), values["inventory"]),
        "Receivables_Turnover": safe_ratio(values["revenue"], values["receivables"]),
    }
    complete_label = all(np.isfinite(value) for value in [*ni.values(), event_equity, event_retained, event_capital])
    complete_features = all(np.isfinite(value) for value in features.values())
    return {
        "Ticker": ticker, "Year": 2023, "EventYear": 2024,
        "loss_three_consecutive_years": int(loss3),
        "accumulated_loss_exceeds_contributed_capital": int(accumulated),
        "negative_equity": int(negative_equity),
        "financial_distress": int(loss3 or accumulated or negative_equity),
        "complete_label_inputs": complete_label, "complete_features": complete_features,
        "status": "eligible" if complete_label and complete_features else "incomplete",
        **features,
    }


def get_candidate_universe(sample_size: int, seed: int) -> pd.DataFrame:
    listing_path = OUT / "listing_universe.csv"
    if listing_path.exists():
        listing = pd.read_csv(listing_path)
    else:
        from vnstock.api.listing import Listing
        listing = Listing(source="VCI").symbols_by_industries(lang="vi")
        OUT.mkdir(parents=True, exist_ok=True)
        listing.to_csv(listing_path, index=False, encoding="utf-8-sig")
    level1 = listing[pd.to_numeric(listing["icb_level"], errors="coerce").eq(1)].drop_duplicates("symbol")
    level1 = level1[~level1["icb_name"].map(normalize).str.contains("tai chinh")]
    training = set(pd.read_csv(RAW_TRAINING, usecols=[0]).iloc[:, 0].astype(str).str.upper())
    level1["symbol"] = level1["symbol"].astype(str).str.upper()
    level1 = level1[~level1["symbol"].isin(training) & level1["symbol"].str.fullmatch(r"[A-Z]{3}")]
    level1["random_key"] = level1["symbol"].map(lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    random_sample = level1.sort_values("random_key").head(sample_size).copy()
    random_sample["selection_stratum"] = "random_nonfinancial_entity_disjoint"
    sentinel_rows = level1[level1["symbol"].isin(SENTINELS)].copy()
    sentinel_rows["selection_stratum"] = "predeclared_distress_sentinel_exploratory"
    return pd.concat([random_sample, sentinel_rows]).drop_duplicates("symbol", keep="last")


def evaluate(cohort: pd.DataFrame) -> None:
    eligible = cohort[cohort["status"].eq("eligible")].copy()
    predictions: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    for model_type in ("ann", "lr", "tree", "xgb"):
        path = ROOT / "backend" / "results" / model_type / "final_model" / f"pd_{model_type}_model.pkl"
        model = joblib.load(path)
        probability = model.predict_proba(eligible[FEATURES])[:, 1]
        part = eligible[["Ticker", "selection_stratum", "financial_distress"]].copy()
        part["model"] = model_type
        part["probability"] = probability
        predictions.append(part)
        y = eligible["financial_distress"]
        record = {"model": model_type, "n": len(y), "positives": int(y.sum()), "brier": brier_score_loss(y, probability)}
        if y.nunique() == 2:
            record.update({"roc_auc": roc_auc_score(y, probability), "pr_auc": average_precision_score(y, probability)})
        summary.append(record)
    pd.concat(predictions).to_csv(OUT / "external_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(OUT / "external_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_frame = pd.concat(predictions)
    diagnostics: list[dict[str, object]] = []
    for (model, stratum), group in prediction_frame.groupby(["model", "selection_stratum"]):
        predicted = group["probability"].ge(0.5).astype(int)
        y = group["financial_distress"].astype(int)
        diagnostics.append({
            "model": model,
            "selection_stratum": stratum,
            "n": len(group),
            "positives": int(y.sum()),
            "threshold": 0.5,
            "predicted_positive_rate": float(predicted.mean()),
            "recall_if_defined": float(predicted[y.eq(1)].mean()) if y.eq(1).any() else np.nan,
            "false_positive_rate_if_defined": float(predicted[y.eq(0)].mean()) if y.eq(0).any() else np.nan,
        })
    pd.DataFrame(diagnostics).to_csv(OUT / "external_subgroup_diagnostics.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build entity-disjoint 2023->2024 external validation cohort")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20240622)
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between API calls")
    args = parser.parse_args()
    os.environ.setdefault("VNSTOCK_HOME", str(ROOT / ".vnstock"))
    OUT.mkdir(parents=True, exist_ok=True)
    universe = get_candidate_universe(args.sample_size, args.seed)
    universe.to_csv(OUT / "candidate_sample.csv", index=False, encoding="utf-8-sig")
    rows = [build_row(ticker, args.delay) for ticker in universe["symbol"]]
    cohort = pd.DataFrame(rows).merge(universe[["symbol", "organ_name", "icb_name", "selection_stratum"]], left_on="Ticker", right_on="symbol", how="left")
    cohort.to_csv(OUT / "external_cohort_2023_2024.csv", index=False, encoding="utf-8-sig")
    eligible = cohort[cohort["status"].eq("eligible")]
    if not eligible.empty:
        evaluate(cohort)
    print(json.dumps({"sampled": len(cohort), "eligible": len(eligible), "positives": int(eligible.get("financial_distress", pd.Series(dtype=int)).sum())}))


if __name__ == "__main__":
    main()
