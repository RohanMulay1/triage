"""Production configuration and cold FastAPI checks.

The suite forces the mock provider and redirects writable state, so this module
exercises application seams without network calls or repository writes.
"""
from fastapi.testclient import TestClient
import pytest

from app.config import default_small_model, get_model, get_settings


def _clear_settings() -> None:
    get_settings.cache_clear()


def test_unset_application_override_preserves_historical_default(monkeypatch):
    monkeypatch.delenv("TRIAGE_DEFAULT_MODEL", raising=False)
    _clear_settings()
    try:
        model = default_small_model()
        assert model["id"] == "llama-3.1-8b"
        assert model["provider_model"] == "meta/llama-3.1-8b-instruct"
        assert model["operational_status"] == "retired"
    finally:
        _clear_settings()


def test_explicit_application_override_selects_verified_groq_pair(monkeypatch):
    monkeypatch.setenv("TRIAGE_DEFAULT_MODEL", "groq-gpt-oss-20b")
    _clear_settings()
    try:
        small = default_small_model()
        assert small["id"] == "groq-gpt-oss-20b"
        assert small["escalate_to"] == "groq-gpt-oss-120b"
        assert get_model(small["escalate_to"])["provider"] == "groq"
    finally:
        _clear_settings()


def test_invalid_application_override_fails_closed(monkeypatch):
    monkeypatch.setenv("TRIAGE_DEFAULT_MODEL", "typo-does-not-exist")
    _clear_settings()
    try:
        with pytest.raises(ValueError, match="TRIAGE_DEFAULT_MODEL names unknown model"):
            default_small_model()
    finally:
        _clear_settings()


def test_fastapi_health_chat_and_telemetry_without_network(monkeypatch):
    monkeypatch.setenv("TRIAGE_FORCE_MOCK", "1")
    monkeypatch.setenv("TRIAGE_DEFAULT_MODEL", "groq-gpt-oss-20b")
    _clear_settings()
    try:
        from app.main import app

        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            assert health.json()["force_mock"] is True

            chat = client.post("/chat", json={"message": "What is 2 + 2?"})
            assert chat.status_code == 200
            body = chat.json()
            assert body["status"] == "OK"
            assert body["cost"]["est_cost_usd"] == 0.0

            telemetry = client.get("/telemetry")
            assert telemetry.status_code == 200
            assert "summary" in telemetry.json()
            assert "recent" in telemetry.json()
    finally:
        _clear_settings()
