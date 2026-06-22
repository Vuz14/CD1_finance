from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_financial_distress_labels import load_event_year_frame


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "backend" / "dataset" / "BCTC_2015_2024_TongHop.csv"
OUTPUT_PATH = ROOT / "backend" / "dataset" / "Processed_EWS_Final.csv"
ARCHIVE_DIR = ROOT / "backend" / "dataset" / "archive"
AUDIT_DIR = ROOT / "research" / "dataset_rebuild"

TICKER = "Mã CP"
YEAR = "Năm"
NET_INCOME = "Lợi nhuận sau thuế thu nhập doanh nghiệp"
REVENUE = "Doanh thu thuần"
COGS = "Giá vốn hàng bán"
GROSS_PROFIT = "Lợi nhuận gộp"
CURRENT_ASSETS = "Tài sản lưu động và đầu tư ngắn hạn"
CURRENT_LIABILITIES = "Nợ ngắn hạn"
CASH = "Tiền và các khoản tương đương tiền"
EQUITY = "Vốn chủ sở hữu"
RECEIVABLES = "Các khoản phải thu ngắn hạn"
INVENTORY = "Tổng hàng tồn kho"
TOTAL_ASSETS = "TỔNG CỘNG TÀI SẢN"
REPORTED_LIABILITIES = "Nợ phải trả"

FEATURE_INPUTS = [
    NET_INCOME,
    REVENUE,
    COGS,
    GROSS_PROFIT,
    CURRENT_ASSETS,
    CURRENT_LIABILITIES,
    CASH,
    EQUITY,
    RECEIVABLES,
    INVENTORY,
    TOTAL_ASSETS,
]
DENOMINATORS = [REVENUE, CURRENT_LIABILITIES, EQUITY, RECEIVABLES, INVENTORY, TOTAL_ASSETS]
FEATURES = [
    "ROA",
    "ROE",
    "Gross_Margin",
    "Current_Ratio",
    "Debt_to_Assets",
    "Cash_Ratio",
    "Inventory_Turnover",
    "Receivables_Turnover",
]


def choose_scraped_record(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = list(dict.fromkeys(FEATURE_INPUTS + [REPORTED_LIABILITIES]))
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
    raw = raw.reset_index(names="raw_row")
    raw["feature_input_count"] = raw[FEATURE_INPUTS].notna().sum(axis=1)
    raw["balance_error"] = (
        raw[TOTAL_ASSETS] - raw[REPORTED_LIABILITIES] - raw[EQUITY]
    ).abs() / raw[TOTAL_ASSETS].abs().replace(0, np.nan)

    ranked = raw.sort_values(
        [TICKER, YEAR, "balance_error", "feature_input_count", "raw_row"],
        ascending=[True, True, True, False, True],
        na_position="last",
    )
    selected = ranked.drop_duplicates([TICKER, YEAR], keep="first").copy()
    decisions = ranked.loc[ranked.duplicated([TICKER, YEAR], keep=False), [
        TICKER,
        YEAR,
        "raw_row",
        "balance_error",
        "feature_input_count",
    ]].copy()
    decisions["selected"] = decisions["raw_row"].isin(selected["raw_row"])
    return selected, decisions


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_features(selected: pd.DataFrame) -> pd.DataFrame:
    frame = selected.copy()
    # The scraped liabilities column is demonstrably shifted for many rows.
    # The accounting identity gives a reproducible total-liabilities value.
    frame["derived_total_liabilities"] = frame[TOTAL_ASSETS] - frame[EQUITY]
    frame["ROA"] = safe_ratio(frame[NET_INCOME], frame[TOTAL_ASSETS])
    frame["ROE"] = safe_ratio(frame[NET_INCOME], frame[EQUITY])
    frame["Gross_Margin"] = safe_ratio(frame[GROSS_PROFIT], frame[REVENUE])
    frame["Current_Ratio"] = safe_ratio(frame[CURRENT_ASSETS], frame[CURRENT_LIABILITIES])
    frame["Debt_to_Assets"] = safe_ratio(frame["derived_total_liabilities"], frame[TOTAL_ASSETS])
    frame["Cash_Ratio"] = safe_ratio(frame[CASH], frame[CURRENT_LIABILITIES])
    frame["Inventory_Turnover"] = safe_ratio(frame[COGS], frame[INVENTORY])
    frame["Receivables_Turnover"] = safe_ratio(frame[REVENUE], frame[RECEIVABLES])
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame


def main() -> None:
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    selected, duplicate_decisions = choose_scraped_record(raw)
    feature_frame = build_features(selected)
    event, label_quality = load_event_year_frame()

    labels = event.rename(columns={TICKER: "Ticker", YEAR: "EventYear"})[
        ["Ticker", "EventYear", "financial_distress_event"]
    ].copy()
    labels["Year"] = labels["EventYear"] - 1

    dataset = feature_frame.rename(columns={TICKER: "Ticker", YEAR: "Year"})[
        ["Ticker", "Year", *FEATURES]
    ].merge(labels, on=["Ticker", "Year"], how="inner")
    dataset = dataset.rename(columns={"financial_distress_event": "Target"})

    # Strict history: event year 2016 cannot establish the three-loss criterion
    # because the raw source starts in 2015 and contains no 2014 net income.
    dataset = dataset.loc[dataset["EventYear"].ge(2017)].copy()
    missing_feature_rows = dataset.loc[dataset[FEATURES].isna().any(axis=1)].copy()
    dataset = dataset.dropna(subset=FEATURES)
    dataset = dataset.sort_values(["Ticker", "Year"]).reset_index(drop=True)
    dataset["Target"] = dataset["Target"].astype(int)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(OUTPUT_PATH, ARCHIVE_DIR / f"Processed_EWS_Final_pre_rebuild_{stamp}.csv")

    dataset.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    duplicate_decisions.to_csv(AUDIT_DIR / "duplicate_record_decisions.csv", index=False, encoding="utf-8-sig")
    missing_feature_rows.to_csv(AUDIT_DIR / "rows_dropped_missing_features.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source": str(RAW_PATH),
        "rows_raw": int(len(raw)),
        "unique_company_years": int(len(selected)),
        "duplicate_company_year_groups": int(raw.groupby([TICKER, YEAR]).size().gt(1).sum()),
        "reported_liabilities_not_used": True,
        "liabilities_reconstruction": "total_assets - equity",
        "strict_feature_year_min": int(dataset["Year"].min()),
        "feature_year_max": int(dataset["Year"].max()),
        "rows_final": int(len(dataset)),
        "companies_final": int(dataset["Ticker"].nunique()),
        "positive_labels": int(dataset["Target"].sum()),
        "rows_dropped_missing_features": int(len(missing_feature_rows)),
        "class_counts": {str(k): int(v) for k, v in dataset["Target"].value_counts().sort_index().items()},
        "label_quality": label_quality,
        "notes": [
            "No global scaling, imputation, or winsorization is applied before temporal splitting.",
            "Feature year 2015 is excluded until 2014 net income is available for the three-year-loss rule.",
            "Feature year 2024 is excluded because event-year 2025 statements are unavailable.",
        ],
    }
    (AUDIT_DIR / "rebuild_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
