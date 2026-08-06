import pytest

from app.ai.config import AISettings


def test_ai_settings_do_not_require_a_secret_at_import_time(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AI_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = AISettings.load()

    assert settings.deepseek_api_key == ""
    assert settings.deepseek_configured is False
    assert settings.deepseek_chat_model == "deepseek-v4-flash"
    assert settings.embedding_dimension == 768
    assert settings.embedded_ai_enabled is False
    assert settings.report_worker_enabled is False


def test_ai_settings_reject_candidate_count_smaller_than_result_count(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAG_TOP_K", "10")
    monkeypatch.setenv("RAG_CANDIDATE_K", "5")

    with pytest.raises(ValueError, match="RAG_CANDIDATE_K"):
        AISettings.load()


def test_ai_settings_can_enable_embedded_report_worker(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDED_AI_ENABLED", "true")
    monkeypatch.setenv("REPORT_WORKER_ENABLED", "true")

    settings = AISettings.load()

    assert settings.embedded_ai_enabled is True
    assert settings.report_worker_enabled is True
