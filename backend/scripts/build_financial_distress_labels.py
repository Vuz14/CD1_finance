from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "backend" / "dataset" / "BCTC_2015_2024_TongHop.csv"
PROCESSED_PATH = ROOT / "backend" / "dataset" / "Processed_EWS_Final.csv"
OUTPUT_DIR = ROOT / "research" / "label_reconstruction"

TICKER = "Mã CP"
YEAR = "Năm"
NET_INCOME = "Lợi nhuận sau thuế thu nhập doanh nghiệp"
RETAINED_EARNINGS = "Lợi nhuận sau thuế chưa phân phối ( Lỗ Lũy kế ) "
CONTRIBUTED_CAPITAL = "Vốn đầu tư của chủ sở hữu ( Vốn điều lệ )"
EQUITY = "Vốn chủ sở hữu"
LABEL_FIELDS = [NET_INCOME, RETAINED_EARNINGS, CONTRIBUTED_CAPITAL, EQUITY]


def load_event_year_frame() -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw[LABEL_FIELDS] = raw[LABEL_FIELDS].apply(pd.to_numeric, errors="coerce")

    grouped = raw.groupby([TICKER, YEAR], dropna=False)
    disagreements = grouped[LABEL_FIELDS].nunique(dropna=False).gt(1)
    conflicting_groups = disagreements.any(axis=1)
    if conflicting_groups.any():
        keys = [list(key) for key in conflicting_groups[conflicting_groups].index]
        raise ValueError(f"Conflicting label inputs in duplicate company-years: {keys}")

    # Duplicate scraped rows disagree in some feature fields, but their four label
    # inputs are identical. Collapse only the label inputs here; feature repair is
    # intentionally outside the scope of this script.
    event = grouped[LABEL_FIELDS].first().reset_index().sort_values([TICKER, YEAR])
    by_company = event.groupby(TICKER, group_keys=False)

    three_losses = by_company[NET_INCOME].transform(
        lambda values: values.lt(0) & values.shift(1).lt(0) & values.shift(2).lt(0)
    )
    consecutive_years = by_company[YEAR].transform(
        lambda years: years.diff().eq(1) & years.diff(2).eq(2)
    )
    event["loss_three_consecutive_years"] = three_losses & consecutive_years
    event["accumulated_loss_exceeds_contributed_capital"] = (
        event[RETAINED_EARNINGS].lt(0)
        & event[RETAINED_EARNINGS].abs().gt(event[CONTRIBUTED_CAPITAL])
    )
    event["negative_equity"] = event[EQUITY].lt(0)
    criteria = [
        "loss_three_consecutive_years",
        "accumulated_loss_exceeds_contributed_capital",
        "negative_equity",
    ]
    event["financial_distress_event"] = event[criteria].any(axis=1).astype(int)

    quality = {
        "raw_rows": int(len(raw)),
        "unique_company_years": int(len(event)),
        "companies": int(event[TICKER].nunique()),
        "year_min": int(event[YEAR].min()),
        "year_max": int(event[YEAR].max()),
        "duplicate_company_year_groups": int(grouped.size().gt(1).sum()),
        "duplicate_groups_with_conflicting_label_inputs": int(conflicting_groups.sum()),
        "missing_label_inputs": {field: int(event[field].isna().sum()) for field in LABEL_FIELDS},
        "derived_positive_event_years": int(event["financial_distress_event"].sum()),
    }
    return event, quality


def compare_with_processed(event: pd.DataFrame) -> pd.DataFrame:
    processed = pd.read_csv(PROCESSED_PATH, low_memory=False).rename(
        columns={"Ticker": "company", "Year": "feature_year", "Target": "current_label"}
    )
    label_counts = processed.groupby(["company", "feature_year"])["current_label"].nunique()
    conflicting_keys = label_counts[label_counts.gt(1)].index
    processed = processed.set_index(["company", "feature_year"])
    processed = processed.loc[~processed.index.isin(conflicting_keys)].reset_index()
    processed = processed.groupby(["company", "feature_year"], as_index=False)["current_label"].first()

    shifted = event.rename(columns={TICKER: "company", YEAR: "event_year"}).copy()
    shifted["feature_year"] = shifted["event_year"] - 1
    keep = [
        "company",
        "feature_year",
        "event_year",
        "financial_distress_event",
        "loss_three_consecutive_years",
        "accumulated_loss_exceeds_contributed_capital",
        "negative_equity",
        NET_INCOME,
        RETAINED_EARNINGS,
        CONTRIBUTED_CAPITAL,
        EQUITY,
    ]
    comparison = processed.merge(shifted[keep], on=["company", "feature_year"], how="left")
    comparison["event_data_available"] = comparison["event_year"].notna()
    comparison["derived_label"] = comparison["financial_distress_event"].fillna(0).astype(int)
    comparison["label_matches"] = (
        comparison["event_data_available"]
        & comparison["current_label"].eq(comparison["derived_label"])
    )
    return comparison


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event, quality = load_event_year_frame()
    comparison = compare_with_processed(event)

    event.to_csv(OUTPUT_DIR / "event_year_labels.csv", index=False, encoding="utf-8-sig")
    event.loc[event["financial_distress_event"].eq(1)].to_csv(
        OUTPUT_DIR / "positive_event_years.csv", index=False, encoding="utf-8-sig"
    )
    comparison.to_csv(OUTPUT_DIR / "processed_label_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.loc[
        comparison["current_label"].eq(1) & ~comparison["label_matches"]
    ].to_csv(OUTPUT_DIR / "positive_label_exceptions.csv", index=False, encoding="utf-8-sig")

    quality["processed_positive_labels_after_conflict_removal"] = int(comparison["current_label"].sum())
    quality["positive_labels_reproduced"] = int(
        (comparison["current_label"].eq(1) & comparison["label_matches"]).sum()
    )
    quality["positive_labels_not_reproduced"] = int(
        (comparison["current_label"].eq(1) & ~comparison["label_matches"]).sum()
    )
    (OUTPUT_DIR / "reconstruction_summary.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # ASCII output remains readable in Windows terminals that default to cp1252;
    # the persisted JSON above retains the original Vietnamese field names.
    print(json.dumps(quality, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
