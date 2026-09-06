"""A successful metadata response must never become a full-text audit."""
from types import SimpleNamespace

import pytest
import requests

from scripts.check_literature_source import (
    ARTICLE_NS, PRISM_NS, check_source, inspect_response,
)

DOI = "10.1016/j.knosys.2026.116685"


def xml(original="", doi=DOI):
    return (
        f'<full-text-retrieval-response xmlns="{ARTICLE_NS}" '
        f'xmlns:prism="{PRISM_NS}"><coredata><prism:doi>{doi}</prism:doi>'
        f'</coredata>{original}</full-text-retrieval-response>'
    ).encode()


@pytest.mark.parametrize("original", ["", "<originalText/>", "<originalText> </originalText>"])
def test_http_success_without_article_text_is_metadata_only(original):
    result = inspect_response(200, xml(original), DOI)
    assert result["status"] == "metadata_only"
    assert result["full_text_verified"] is False
    assert result["novelty_verdict"] is None


@pytest.mark.parametrize("status,body,expected", [
    (403, b"blocked", "http_error"),
    (401, b"authentication required", "http_error"),
    (200, b"not XML", "invalid_xml"),
    (200, b"<html>Sign in</html>", "unexpected_document"),
    (200, xml(doi="10.1/wrong"), "identity_mismatch"),
])
def test_access_and_identity_failures_never_verify_full_text(status, body, expected):
    result = inspect_response(status, body, DOI)
    assert result["status"] == expected
    assert result["full_text_verified"] is False
    assert result["novelty_verdict"] is None


def test_article_body_still_requires_manual_reading():
    result = inspect_response(200, xml("<originalText>Partial abstract</originalText>"), DOI)
    assert result["status"] == "body_present_requires_manual_review"
    assert result["full_text_verified"] is False
    assert result["novelty_verdict"] is None


def test_receipt_records_bytes_without_redistributing_body(tmp_path, monkeypatch):
    body = xml()
    monkeypatch.setattr("scripts.check_literature_source.requests.get",
                        lambda *a, **kw: SimpleNamespace(status_code=200, content=body,
                                                         url="https://api.elsevier.com/"))
    output = tmp_path / "attempt"
    result = check_source(DOI, output)
    assert result["expected_doi"] == DOI
    assert result["bytes"] == len(body)
    assert len(result["sha256"]) == 64
    assert result["status"] == "metadata_only"
    assert [p.name for p in output.iterdir()] == ["receipt.json"]


def test_reused_directory_refused_before_network_request(tmp_path, monkeypatch):
    output = tmp_path / "attempt"
    output.mkdir()
    receipt = output / "receipt.json"
    receipt.write_text("previous receipt", encoding="utf-8")

    def unexpected_request(*args, **kwargs):
        pytest.fail("must refuse reuse before making a network request")

    monkeypatch.setattr("scripts.check_literature_source.requests.get", unexpected_request)
    with pytest.raises(FileExistsError):
        check_source(DOI, output)
    assert receipt.read_text(encoding="utf-8") == "previous receipt"


def test_transport_failure_is_recorded_without_a_verdict(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.Timeout("test")

    monkeypatch.setattr("scripts.check_literature_source.requests.get", timeout)
    result = check_source(DOI, tmp_path / "attempt")
    assert result["status"] == "transport_error"
    assert result["full_text_verified"] is False
    assert result["novelty_verdict"] is None
