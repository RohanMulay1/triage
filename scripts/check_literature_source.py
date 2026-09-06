"""Inspect an Elsevier XML response without mistaking HTTP 200 for full text.

Public, unauthenticated GET only; no provider calls or paid access. Receipts are
single-use directories. A body is only a candidate for manual reading, never an
automated full-text or novelty verdict. Response bodies are not redistributed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

import requests

ARTICLE_NS = "http://www.elsevier.com/xml/svapi/article/dtd"
PRISM_NS = "http://prismstandard.org/namespaces/basic/2.0/"


def inspect_response(status: int, body: bytes, expected_doi: str) -> dict:
    """Refuse errors, wrong identities, and metadata-only responses.

    Even originalText may contain just an abstract or a truncated document.
    Its presence cannot certify completeness or establish a scientific claim.
    """
    result = {
        "http_status": status,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "expected_doi": expected_doi,
        "full_text_verified": False,
        "novelty_verdict": None,
    }
    if status != 200:
        return {**result, "status": "http_error"}
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return {**result, "status": "invalid_xml"}
    if root.tag != f"{{{ARTICLE_NS}}}full-text-retrieval-response":
        return {**result, "status": "unexpected_document"}
    doi = root.findtext(f"{{{ARTICLE_NS}}}coredata/{{{PRISM_NS}}}doi", "").strip()
    result["returned_doi"] = doi
    if doi.casefold() != expected_doi.strip().casefold():
        return {**result, "status": "identity_mismatch"}
    original = root.find(f"{{{ARTICLE_NS}}}originalText")
    if original is None or not "".join(original.itertext()).strip():
        return {**result, "status": "metadata_only"}
    return {**result, "status": "body_present_requires_manual_review"}


def check_source(doi: str, output_dir: Path) -> dict:
    """Write a provenance receipt once; refuse reuse before making a request."""
    output_dir.mkdir(parents=True, exist_ok=False)
    url = "https://api.elsevier.com/content/article/doi/" + quote(doi, safe="/")
    url += "?httpAccept=text/xml"
    receipt = {
        "url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "access": "public_unauthenticated",
        "response_body_saved": False,
    }
    try:
        response = requests.get(url, timeout=30)
        receipt.update(inspect_response(response.status_code, response.content, doi))
        receipt["final_url"] = response.url
    except requests.RequestException as exc:
        receipt.update(status="transport_error", error_type=type(exc).__name__,
                       full_text_verified=False, novelty_verdict=None,
                       expected_doi=doi)
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = check_source(args.doi, args.output_dir)
    print(json.dumps(result, indent=2))
    # A body still requires reading. Exit 0 means candidate acquired, not a pass.
    return 0 if result["status"] == "body_present_requires_manual_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
