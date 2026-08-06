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


def test_ai_settings_reject_candidate_count_smaller_than_result_count(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAG_TOP_K", "10")
    monkeypatch.setenv("RAG_CANDIDATE_K", "5")

    with pytest.raises(ValueError, match="RAG_CANDIDATE_K"):
        AISettings.load()
