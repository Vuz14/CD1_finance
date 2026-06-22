from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import re
import unicodedata
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "research" / "label_audit_template.csv"
MANIFEST_PATH = ROOT / "research" / "source_document_manifest.csv"
EVIDENCE_PATH = ROOT / "research" / "source_document_evidence.csv"
CACHE_ROOT = ROOT / "research" / "ocr_cache"

KEYWORDS = {
    "net_income": ("loi nhuan sau thue", "loi nhuan sau thue thu nhap doanh nghiep"),
    "retained_earnings": ("loi nhuan sau thue chua phan phoi", "lo luy ke"),
    "contributed_capital": ("von dau tu cua chu so huu", "von dieu le"),
    "equity": ("von chu so huu",),
}


def normalize(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()


def keyword_hits(text: str) -> list[str]:
    normalized = normalize(text)
    return [name for name, terms in KEYWORDS.items() if any(term in normalized for term in terms)]


def native_pages(path: Path) -> dict[int, str]:
    reader = PdfReader(path, strict=False)
    return {number: page.extract_text() or "" for number, page in enumerate(reader.pages, start=1)}


def ocr_page(reader, document: fitz.Document, page_index: int, cache_dir: Path) -> str:
    cache = cache_dir / f"page_{page_index + 1}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    if reader is None:
        return ""
    page = document.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    results, _ = reader(image)
    text = "\n".join(str(item[1]) for item in (results or []))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def candidate_page_indexes(page_count: int, max_pages: int) -> list[int]:
    # Audited statements normally place the primary statements after the audit
    # opinion. Search the front matter first, then the tail for small attachments.
    if page_count <= 13:
        return list(range(page_count))
    # In the downloaded Vietnamese audited statements, the balance sheet and
    # income statement consistently begin around PDF pages 9-13.
    start = 8
    stop = min(page_count, start + max_pages)
    return list(range(start, stop))


def inspect(
    path: Path,
    sha256: str,
    max_ocr_pages: int,
    reader_holder: dict,
    required_fields: set[str] | None = None,
) -> list[dict[str, object]]:
    required_fields = required_fields or set(KEYWORDS)
    pages = native_pages(path)
    evidence: list[dict[str, object]] = []
    for page_number, text in pages.items():
        hits = keyword_hits(text)
        if hits:
            evidence.append({"page": page_number, "method": "embedded_text", "hits": hits, "text": text})
    native_found = {hit for item in evidence for hit in item["hits"]}
    if native_found.issuperset(required_fields):
        return evidence

    if reader_holder.get("cached_only"):
        reader_holder["reader"] = None
    elif "reader" not in reader_holder:
        from rapidocr_onnxruntime import RapidOCR
        reader_holder["reader"] = RapidOCR()
    document = fitz.open(path)
    cache_dir = CACHE_ROOT / (sha256 or path.stem)
    found: set[str] = set()
    for page_index in candidate_page_indexes(len(document), max_ocr_pages):
        text = ocr_page(reader_holder["reader"], document, page_index, cache_dir)
        hits = keyword_hits(text)
        if hits:
            evidence.append({"page": page_index + 1, "method": "rapidocr_onnx", "hits": hits, "text": text})
            found.update(hits)
        if (native_found | found).issuperset(required_fields):
            break
    return evidence


def compact_snippet(text: str, limit: int = 320) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    selected = [line for line in lines if keyword_hits(line)]
    return " | ".join(selected)[:limit]


def inspect_worker(args: tuple[str, str, int, bool, tuple[str, ...]]) -> list[dict[str, object]]:
    path, sha256, max_ocr_pages, cached_only, required_fields = args
    return inspect(
        Path(path), sha256, max_ocr_pages, {"cached_only": cached_only}, set(required_fields)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract page-level evidence from audited financial statements")
    parser.add_argument("--max-ocr-pages", type=int, default=9)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-ocr", action="store_true", help="Only populate downloaded source metadata")
    parser.add_argument("--cached-only", action="store_true", help="Parse embedded text and existing OCR cache only")
    parser.add_argument("--companies", default="", help="Optional comma-separated ticker filter")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_PATH)
    downloaded = manifest[manifest["status"].isin(["downloaded_verified", "downloaded_unverified"])].copy()
    # Prefer consolidated/larger documents and inspect one document per company-year.
    downloaded["bytes"] = pd.to_numeric(downloaded["bytes"], errors="coerce").fillna(0)
    downloaded = downloaded.sort_values(["company", "event_year", "document_verified", "bytes"], ascending=[True, True, False, False])
    downloaded = downloaded.drop_duplicates(["company", "event_year"], keep="first")
    if args.companies:
        selected_companies = {value.strip().upper() for value in args.companies.split(",") if value.strip()}
        downloaded = downloaded[downloaded["company"].isin(selected_companies)]

    rows: list[dict[str, object]] = []
    sources = [source for source in downloaded.itertuples(index=False)
               if (ROOT / str(source.local_file)).exists() and float(source.page_count or 0) > 1]
    if not args.skip_ocr:
        required_by_source: dict[tuple[str, int], set[str]] = {}
        for audit_row in pd.read_csv(AUDIT_PATH).itertuples(index=False):
            event_year = int(audit_row.event_year)
            if int(audit_row.loss_three_consecutive_years) == 1:
                for year in (event_year - 2, event_year - 1, event_year):
                    required_by_source.setdefault((audit_row.company, year), set()).add("net_income")
            if int(audit_row.accumulated_loss_exceeds_contributed_capital) == 1:
                required_by_source.setdefault((audit_row.company, event_year), set()).update(
                    {"retained_earnings", "contributed_capital"}
                )
            if int(audit_row.negative_equity) == 1:
                required_by_source.setdefault((audit_row.company, event_year), set()).add("equity")
        work = [
            (
                str(ROOT / str(source.local_file)), str(source.sha256), args.max_ocr_pages,
                args.cached_only, tuple(sorted(required_by_source.get(
                    (source.company, int(source.event_year)), set(KEYWORDS)
                ))),
            )
            for source in sources
        ]
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            inspected = executor.map(inspect_worker, work)
            for source, evidence in zip(sources, inspected):
                path = ROOT / str(source.local_file)
                for item in evidence:
                    rows.append({
                        "company": source.company,
                        "source_year": int(source.event_year),
                        "source_document": path.name,
                        "source_url": source.source_url,
                        "local_file": source.local_file,
                        "page": item["page"],
                        "extraction_method": item["method"],
                        "evidence_fields": ";".join(item["hits"]),
                        "evidence_snippet": compact_snippet(item["text"]),
                        "sha256": source.sha256,
                    })
    evidence_frame = pd.DataFrame(rows)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    evidence_frame.to_csv(EVIDENCE_PATH, index=False, encoding="utf-8-sig")

    audit = pd.read_csv(AUDIT_PATH)
    for column in ("source_pages", "evidence_fields", "automated_check", "source_sha256"):
        if column not in audit.columns:
            audit[column] = ""
        audit[column] = audit[column].fillna("").astype(str)
    if not args.companies:
        for column in (
            "source_document", "source_url_or_file", "source_pages", "evidence_fields",
            "automated_check", "source_sha256",
        ):
            if column not in audit.columns:
                audit[column] = ""
            else:
                audit[column] = ""
    for index, audit_row in audit.iterrows():
        required_years = [int(audit_row["event_year"])]
        if int(audit_row["loss_three_consecutive_years"]) == 1:
            required_years.extend([int(audit_row["event_year"]) - 1, int(audit_row["event_year"]) - 2])
        sources = downloaded[
            downloaded["company"].eq(audit_row["company"])
            & downloaded["event_year"].astype(int).isin(required_years)
        ]
        if not sources.empty:
            audit.at[index, "source_document"] = "; ".join(
                f"{int(row.event_year)}:{Path(str(row.local_file)).name}" for row in sources.itertuples(index=False)
            )
            audit.at[index, "source_url_or_file"] = "; ".join(
                f"{int(row.event_year)}:{row.source_url}" for row in sources.itertuples(index=False)
            )
            audit.at[index, "source_sha256"] = ";".join(str(value) for value in sources["sha256"].dropna())
            missing = sorted(set(required_years) - set(sources["event_year"].astype(int)))
            audit.at[index, "automated_check"] = (
                "PDF downloaded; OCR/page evidence pending" if missing else
                "all required PDFs downloaded; OCR/page evidence pending"
            )
            if missing:
                audit.at[index, "notes"] = str(audit.at[index, "notes"]) + f"; missing source years: {missing}"
        if evidence_frame.empty:
            continue
        matched = evidence_frame[
            evidence_frame["company"].eq(audit_row["company"])
            & evidence_frame["source_year"].isin(required_years)
        ]
        if matched.empty:
            continue
        audit.at[index, "source_pages"] = ";".join(
            f"{int(row.source_year)}:{int(row.page)}" for row in matched.itertuples(index=False)
        )
        audit.at[index, "evidence_fields"] = ";".join(sorted(set(";".join(matched["evidence_fields"]).split(";"))))
        audit.at[index, "automated_check"] = "page-level evidence extracted; independent human verification pending"
    audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8-sig")
    print(json.dumps({"documents": int(downloaded.shape[0]), "evidence_pages": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
