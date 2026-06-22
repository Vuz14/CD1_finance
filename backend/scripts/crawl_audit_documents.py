from __future__ import annotations

import argparse
import hashlib
import io
import time
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "research" / "label_audit_template.csv"
OUTPUT_ROOT = ROOT / "research" / "source_documents"
MANIFEST_PATH = ROOT / "research" / "source_document_manifest.csv"
REQUIREMENTS_PATH = ROOT / "research" / "source_document_requirements.csv"
SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (compatible; AcademicFinancialDistressAudit/1.0; research use)"
MAX_FILE_BYTES = 50 * 1024 * 1024
EXCHANGE = {
    "AGM": "HOSE", "APC": "HOSE", "DPC": "HNX", "DTC": "HNX",
    "HBC": "HOSE", "HNG": "HOSE", "L43": "HNX", "LEC": "HOSE",
    "NGC": "HNX", "SCD": "HOSE", "SDT": "HNX", "SMC": "HOSE",
    "VGI": "UPCOM", "VNA": "HNX", "VTL": "UPCOM",
}


def decode_result_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    encoded = parse_qs(parsed.query).get("uddg")
    return unquote(encoded[0]) if encoded else href


def source_score(url: str, ticker: str) -> int:
    host = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    score = 0
    if any(domain in host for domain in ("hsx.vn", "hnx.vn", "ssc.gov.vn")):
        score += 100
    elif host.endswith(".gov.vn"):
        score += 95
    elif ticker.lower() in host or ticker.lower() in path:
        score += 35
    if path.endswith(".pdf"):
        score += 35
    if "kiemtoan" in path or "audit" in path:
        score += 15
    if "static2.vietstock.vn" in host:
        score += 55
    elif "cafef1.mediacdn.vn" in host:
        score += 45
    elif "vietstock.vn" in host or "cafef.vn" in host:
        score += 15
    return score


def direct_pdf_candidates(ticker: str, year: int) -> list[str]:
    yy = str(year)[-2:]
    exchanges = list(dict.fromkeys([EXCHANGE.get(ticker, "HNX"), "HOSE", "HNX", "UPCOM"]))
    names = [
        f"{ticker}_Baocaotaichinh_{year}_Kiemtoan_Hopnhat.pdf",
        f"{ticker}_Baocaotaichinh_{year}_Kiemtoan_Congtyme.pdf",
        f"{ticker}_Baocaotaichinh_{year}_Kiemtoan.pdf",
        f"{ticker}_Baocaotaichinh_{year}_Kiemtoan_HN.pdf",
        f"{ticker}_BCTC_{year}_Kiemtoan_Hopnhat.pdf",
    ]
    candidates = [
        f"https://static2.vietstock.vn/data/{exchange}/{year}/BCTC/VN/NAM/{name}"
        for exchange in exchanges for name in names
    ]
    candidates.extend([
        f"https://cafef1.mediacdn.vn/Images/Uploaded/DuLieuDownload/BCTC/{ticker}_{yy}CN_BCTC_HNKT.pdf",
        f"https://cafef1.mediacdn.vn/Images/Uploaded/DuLieuDownload/BCTC/{ticker}_{yy}CN_BCTC_KT.pdf",
        f"https://cafef1.mediacdn.vn/Images/Uploaded/DuLieuDownload/BCTC/{ticker}_{yy}CN_BCTC_HN.pdf",
        f"https://cafef1.mediacdn.vn/Images/Uploaded/DuLieuDownload/BCTC/{ticker}_{yy}CN_BCTC.pdf",
    ])
    return candidates


def normalize_text(value: str) -> str:
    source = value.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFKD", source).encode("ascii", "ignore").decode("ascii").lower()


def inspect_pdf(content: bytes, event_year: int) -> dict[str, object]:
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        page_count = len(reader.pages)
        texts: list[str] = []
        for page in reader.pages[: min(10, page_count)]:
            texts.append(page.extract_text() or "")
        text = normalize_text("\n".join(texts))
        financial = any(term in text for term in (
            "bao cao tai chinh", "bao cao cua kiem toan", "financial statements",
            "bang can doi ke toan", "bao cao tinh hinh tai chinh",
        ))
        year_present = str(event_year) in text
        if text.strip() and not financial:
            return {"document_verified": False, "page_count": page_count, "verification": "not_financial_statement"}
        return {
            "document_verified": bool(financial and year_present),
            "page_count": page_count,
            "verification": "text_verified" if financial and year_present else "scan_or_year_unverified",
        }
    except Exception as exc:
        return {"document_verified": False, "verification": "pdf_parse_error", "parse_error": str(exc)[:200]}


def search(session: requests.Session, query: str, retries: int = 3) -> list[str]:
    for attempt in range(retries):
        try:
            response = session.get(SEARCH_URL, params={"q": query}, timeout=40)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            urls = [decode_result_url(link.get("href", "")) for link in soup.select("a.result__a")]
            return list(dict.fromkeys(url for url in urls if url.startswith("http")))
        except requests.RequestException:
            if attempt == retries - 1:
                return []
            time.sleep(3 * (attempt + 1))
    return []


def download_pdf(session: requests.Session, url: str, destination: Path) -> dict[str, object]:
    try:
        with session.get(url, timeout=60, stream=True, allow_redirects=True) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(1024 * 128):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    return {"status": "too_large", "bytes": size}
                chunks.append(chunk)
            content = b"".join(chunks)
        if not content.startswith(b"%PDF"):
            return {"status": "not_pdf", "content_type": content_type, "bytes": len(content)}
        inspection = inspect_pdf(content, int(destination.parent.name))
        if inspection.get("verification") == "not_financial_statement":
            return {"status": "rejected_not_financial_statement", "content_type": content_type, "bytes": len(content), **inspection}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return {
            "status": "downloaded_verified" if inspection.get("document_verified") else "downloaded_unverified",
            "content_type": content_type,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "local_file": str(destination.relative_to(ROOT)),
            **inspection,
        }
    except requests.RequestException as exc:
        return {"status": "download_error", "error": str(exc)[:300]}


def crawl(max_documents_per_case: int) -> pd.DataFrame:
    audit = pd.read_csv(AUDIT_PATH)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.8"})
    previous = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()
    records: list[dict[str, object]] = previous.to_dict("records")
    completed: set[tuple[str, int]] = set()
    if not previous.empty:
        ok = previous[previous["status"].isin(["downloaded_verified", "downloaded_unverified"])]
        completed = {(str(row.company), int(row.event_year)) for row in ok.itertuples(index=False)}

    requirements: list[dict[str, object]] = []
    for row in audit.itertuples(index=False):
        years = [int(row.event_year)]
        if int(row.loss_three_consecutive_years) == 1:
            years.extend([int(row.event_year) - 1, int(row.event_year) - 2])
        for source_year in years:
            requirements.append({
                "company": str(row.company).upper(),
                "event_year": int(row.event_year),
                "source_year": source_year,
                "reason": "three_year_loss_sequence" if source_year != int(row.event_year) else "event_year_criteria",
            })
    requirements_frame = pd.DataFrame(requirements).drop_duplicates()
    REQUIREMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    requirements_frame.to_csv(REQUIREMENTS_PATH, index=False, encoding="utf-8-sig")

    for case in requirements_frame[["company", "source_year"]].drop_duplicates().itertuples(index=False):
        ticker = str(case.company).upper()
        year = int(case.source_year)
        if (ticker, year) in completed:
            continue
        queries = [
            f'"{ticker}" "báo cáo tài chính kiểm toán" "{year}" filetype:pdf',
            f'"{ticker}" "Báo cáo tài chính" "{year}" "kiểm toán" pdf',
        ]
        discovered: list[tuple[str, str]] = []
        for query in queries:
            for url in search(session, query):
                discovered.append((query, url))
            time.sleep(1.0)

        unique = {url: "direct historical archive pattern" for url in direct_pdf_candidates(ticker, year)}
        for query, url in discovered:
            unique.setdefault(url, query)
        ranked = sorted(unique.items(), key=lambda item: source_score(item[0], ticker), reverse=True)
        pdf_candidates = [(url, query) for url, query in ranked if urlparse(url).path.lower().endswith(".pdf")]

        downloaded = 0
        for rank, (url, query) in enumerate(pdf_candidates, start=1):
            if downloaded >= max_documents_per_case:
                break
            host = urlparse(url).netloc.lower().replace("www.", "").replace(":", "_")
            destination = OUTPUT_ROOT / ticker / str(year) / f"{ticker}_{year}_{rank}_{host}.pdf"
            result = download_pdf(session, url, destination)
            row: dict[str, object] = {
                "company": ticker,
                "event_year": year,
                "query": query,
                "rank": rank,
                "source_score": source_score(url, ticker),
                "source_domain": urlparse(url).netloc.lower(),
                "source_url": url,
                **result,
            }
            records.append(row)
            if result.get("status") in {"downloaded_verified", "downloaded_unverified"}:
                downloaded += 1
            time.sleep(0.8)

        if not pdf_candidates:
            records.append({
                "company": ticker,
                "event_year": year,
                "status": "no_pdf_discovered",
                "query": " | ".join(queries),
            })

    manifest = pd.DataFrame(records)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and download source PDFs for positive-label audit")
    parser.add_argument("--max-documents-per-case", type=int, default=2)
    args = parser.parse_args()
    manifest = crawl(max(1, args.max_documents_per_case))
    print(manifest["status"].value_counts(dropna=False).to_json())


if __name__ == "__main__":
    main()
