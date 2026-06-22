from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "research" / "source_documents"
MANIFEST_PATH = ROOT / "research" / "source_document_manifest.csv"


def infer_ticker_year(path: Path) -> tuple[str, int] | None:
    # Accept both drop locations:
    #   source_documents/TICKER/report_2020.pdf
    #   source_documents/TICKER/2020/report.pdf
    ticker = (path.parent.parent.name if path.parent.name.isdigit() else path.parent.name).upper()
    match = re.search(r"\b(20\d{2})\b", path.stem)
    if match is None and path.parent.name.isdigit():
        match = re.search(r"\b(20\d{2})\b", path.parent.name)
    if not re.fullmatch(r"[A-Z0-9]{3}", ticker) or not match:
        return None
    return ticker, int(match.group(1))


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()

    # Repair records created by older/broad scans: if a local-user row has the
    # same hash as an existing crawler row, restore the crawler's original path
    # and discard only the duplicate local-user record.
    if not manifest.empty and {"sha256", "source_domain", "local_file"}.issubset(manifest.columns):
        crawler = manifest[manifest["source_domain"].ne("local_user_provided")]
        duplicate_hashes = set(crawler["sha256"].dropna()) & set(
            manifest.loc[manifest["source_domain"].eq("local_user_provided"), "sha256"].dropna()
        )
        for sha256 in duplicate_hashes:
            original = crawler[crawler["sha256"].eq(sha256)].iloc[0]
            duplicate = manifest[
                manifest["sha256"].eq(sha256)
                & manifest["source_domain"].eq("local_user_provided")
            ].iloc[-1]
            current_path = ROOT / str(duplicate["local_file"])
            original_path = ROOT / str(original["local_file"])
            if current_path.exists() and not original_path.exists():
                original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current_path), str(original_path))
        manifest = manifest[
            ~(
                manifest["source_domain"].eq("local_user_provided")
                & manifest["sha256"].isin(duplicate_hashes)
            )
        ].copy()

    records = manifest.to_dict("records")
    ingested = 0
    known_hashes = set(manifest["sha256"].dropna()) if "sha256" in manifest.columns else set()

    candidates = sorted(SOURCE_ROOT.glob("*/*.pdf")) + sorted(SOURCE_ROOT.glob("*/*/*.pdf"))
    for path in candidates:
        # Canonical files have already been ingested. Only process newly dropped
        # PDFs so rerunning this script is idempotent.
        if re.fullmatch(r"[A-Z0-9]{3}_20\d{2}_user_provided_audited(?:_v\d+)?\.pdf", path.name):
            continue
        inferred = infer_ticker_year(path)
        if inferred is None:
            continue
        content = path.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        if sha256 in known_hashes:
            continue
        ticker, year = inferred
        destination_dir = SOURCE_ROOT / ticker / str(year)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{ticker}_{year}_user_provided_audited.pdf"
        version = 2
        while destination.exists():
            destination = destination_dir / f"{ticker}_{year}_user_provided_audited_v{version}.pdf"
            version += 1
        shutil.move(str(path), str(destination))

        content = destination.read_bytes()
        reader = PdfReader(destination, strict=False)
        relative = str(destination.relative_to(ROOT))

        if not manifest.empty and "sha256" in manifest.columns:
            records = [row for row in records if str(row.get("sha256", "")) != sha256]
        records.append({
            "company": ticker,
            "event_year": year,
            "query": "user-provided audited financial statement",
            "rank": 0,
            "source_score": 200,
            "source_domain": "local_user_provided",
            "source_url": relative,
            "status": "downloaded_unverified",
            "content_type": "application/pdf",
            "bytes": len(content),
            "sha256": sha256,
            "local_file": relative,
            "document_verified": False,
            "page_count": len(reader.pages),
            "verification": "local_pdf_pending_page_evidence",
        })
        known_hashes.add(sha256)
        ingested += 1

    output = pd.DataFrame(records)
    output.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    print({"ingested": ingested, "manifest_rows": len(output)})


if __name__ == "__main__":
    main()
