from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "research" / "source_document_manifest.csv"
EVIDENCE_PATH = ROOT / "research" / "source_document_evidence.csv"
AUDIT_PATH = ROOT / "research" / "label_audit_template.csv"

WRONG_DOCUMENTS = {
    "NGC_2020_user_provided_audited.pdf": "PDF is Novaland (NVL), not NGC",
    "SDT_2022_user_provided_audited.pdf": "PDF is Thanh Thanh Cong - Bien Hoa, not SDT",
    "VGI_2017_user_provided_audited.pdf": "PDF is Vietnam National Textile and Garment Group (VGT), not VGI",
    "VGI_2018_user_provided_audited.pdf": "PDF is Vietnam National Textile and Garment Group (VGT), not VGI",
    "VGI_2019_user_provided_audited.pdf": "PDF is Vietnam National Textile and Garment Group (VGT), not VGI",
}

MANUAL_EVIDENCE = [
    {
        "company": "DTC", "source_year": 2022, "page": 10,
        "source_document": "DTC_2022_user_provided_audited.pdf",
        "evidence_fields": "net_income",
        "evidence_snippet": "Lợi nhuận sau thuế TNDN: (7.868.695.603) VND",
    },
    {
        "company": "DTC", "source_year": 2024, "page": 8,
        "source_document": "DTC_2024_user_provided_audited.pdf",
        "evidence_fields": "net_income",
        "evidence_snippet": "Lợi nhuận sau thuế TNDN: (36.955.501.333) VND",
    },
    {
        "company": "HNG", "source_year": 2024, "page": 12,
        "source_document": "HNG_2024_user_provided_audited.pdf",
        "evidence_fields": "net_income",
        "evidence_snippet": "Lỗ thuần sau thuế TNDN: (1.281.878.648) nghìn VND",
    },
    {
        "company": "L43", "source_year": 2021, "page": 8,
        "source_document": "L43_2021_user_provided_audited.pdf",
        "evidence_fields": "net_income",
        "evidence_snippet": "Lợi nhuận sau thuế TNDN: (8.533.077.124) đồng",
    },
    {
        "company": "VNA", "source_year": 2020, "page": 9,
        "source_document": "VNA_2020_user_provided_audited.pdf",
        "evidence_fields": "contributed_capital;retained_earnings;equity",
        "evidence_snippet": (
            "Vốn góp 200.000.000.000 VND; LNST chưa phân phối (218.563.327.214) VND; "
            "vốn chủ sở hữu 64.369.394.640 VND"
        ),
    },
    {
        "company": "NGC", "source_year": 2020, "page": 11,
        "source_document": "NGC_2020_user_provided_audited_v2.pdf",
        "evidence_fields": "contributed_capital;retained_earnings;equity",
        "evidence_snippet": (
            "Vốn góp 22.998.540.000 VND; LNST chưa phân phối (31.021.874.234) VND; "
            "vốn chủ sở hữu (8.023.334.234) VND"
        ),
    },
    {
        "company": "SDT", "source_year": 2022, "page": 10,
        "source_document": "SDT_2022_user_provided_audited_v2.pdf",
        "evidence_fields": "net_income",
        "evidence_snippet": "Lợi nhuận sau thuế TNDN: (15.649.208.664) VND",
    },
    {
        "company": "VGI", "source_year": 2017, "page": 11,
        "source_document": "VGI_2017_user_provided_audited_v2.pdf",
        "evidence_fields": "net_income",
        "evidence_snippet": "Lỗ sau thuế TNDN hợp nhất: (481.058.011.980) VND",
    },
    {
        "company": "VGI", "source_year": 2018, "page": 11,
        "source_document": "VGI_2018_user_provided_audited_v2.pdf",
        "evidence_fields": "net_income",
        "evidence_snippet": "Lỗ sau thuế TNDN hợp nhất: (1.070.924.096.060) VND",
    },
    {
        "company": "VGI", "source_year": 2019, "page": 5,
        "source_document": "VGI_2019_user_provided_audited_v2.pdf",
        "evidence_fields": "net_income",
        "evidence_snippet": "Lỗ sau thuế TNDN sau KTNN trình bày lại: (521.200.083.592) VND",
    },
]


def filename_series(frame: pd.DataFrame) -> pd.Series:
    return frame["local_file"].fillna("").map(lambda value: Path(str(value)).name)


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    local = manifest["source_domain"].eq("local_user_provided")
    filenames = filename_series(manifest)

    for filename, reason in WRONG_DOCUMENTS.items():
        selected = local & filenames.eq(filename)
        manifest.loc[selected, "status"] = "rejected_wrong_company"
        manifest.loc[selected, "document_verified"] = False
        manifest.loc[selected, "verification"] = reason

    valid_documents = {row["source_document"] for row in MANUAL_EVIDENCE}
    for filename in valid_documents:
        selected = local & filenames.eq(filename)
        manifest.loc[selected, "status"] = "downloaded_verified"
        manifest.loc[selected, "document_verified"] = True
        manifest.loc[selected, "verification"] = "company, year and cited statement visually inspected"
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    existing = (
        pd.read_csv(EVIDENCE_PATH)
        if EVIDENCE_PATH.exists() and EVIDENCE_PATH.stat().st_size
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    filenames = filename_series(manifest)
    for item in MANUAL_EVIDENCE:
        selected = manifest[
            filenames.eq(item["source_document"])
            & manifest["source_domain"].eq("local_user_provided")
            & manifest["status"].eq("downloaded_verified")
        ]
        if selected.empty:
            raise RuntimeError(f"Verified source missing from manifest: {item['source_document']}")
        source = selected.sort_values("rank").iloc[0]
        rows.append({
            **item,
            "source_url": source["source_url"],
            "local_file": source["local_file"],
            "extraction_method": "visual_inspection_pending_independent_human_verification",
            "sha256": source["sha256"],
        })

    manual = pd.DataFrame(rows)
    combined = pd.concat([existing, manual], ignore_index=True)
    combined = combined.drop_duplicates(
        ["company", "source_year", "source_document", "page", "evidence_fields"],
        keep="last",
    )
    combined.to_csv(EVIDENCE_PATH, index=False, encoding="utf-8-sig")

    audit = pd.read_csv(AUDIT_PATH)
    for index, audit_row in audit.iterrows():
        event_year = int(audit_row["event_year"])
        required_years = [event_year]
        if int(audit_row["loss_three_consecutive_years"]) == 1:
            required_years += [event_year - 1, event_year - 2]
        evidence = combined[
            combined["company"].eq(audit_row["company"])
            & combined["source_year"].astype(int).isin(required_years)
        ]
        if evidence.empty:
            continue
        page_pairs = sorted({
            (int(row.source_year), int(row.page))
            for row in evidence.itertuples()
        })
        audit.at[index, "source_pages"] = ";".join(
            f"{year}:{page}" for year, page in page_pairs
        )
        audit.at[index, "evidence_fields"] = ";".join(
            sorted(set(";".join(evidence["evidence_fields"].astype(str)).split(";")))
        )
        audit.at[index, "automated_check"] = (
            "page-level evidence extracted/visually inspected; "
            "independent human verification pending"
        )
    audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8-sig")
    print({
        "accepted_local_pdfs": len(valid_documents),
        "rejected_wrong_company": len(WRONG_DOCUMENTS),
    })


if __name__ == "__main__":
    main()
