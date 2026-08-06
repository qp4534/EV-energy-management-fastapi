from __future__ import annotations

import os
from dataclasses import dataclass


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class AISettings:
    """Runtime configuration for the non-realtime AI processes.

    Values are read only from process environment variables. This module never
    reads or creates a local ``.env`` file, so production can inject secrets
    through ECS/Secrets Manager without changing application code.
    """

    database_url: str
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-v4-flash"
    deepseek_report_model: str = "deepseek-v4-flash"
    deepseek_timeout_seconds: float = 30.0
    deepseek_chat_max_tokens: int = 1_200
    deepseek_report_max_tokens: int = 2_000
    deepseek_report_thinking: bool = False
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32
    rag_top_k: int = 6
    rag_candidate_k: int = 30
    rag_min_score: float = 0.35
    rag_allow_drafts: bool = False
    inference_base_url: str = "http://127.0.0.1:8000"
    vehicle_state_timeout_seconds: float = 2.0
    vehicle_state_max_age_seconds: int = 30
    allow_general_fallback: bool = True
    report_worker_poll_seconds: float = 2.0
    report_worker_max_retries: int = 3
    internal_api_token: str = ""

    @property
    def deepseek_configured(self) -> bool:
        return bool(self.deepseek_api_key)

    @classmethod
    def load(cls) -> "AISettings":
        database_url = os.getenv("AI_DATABASE_URL") or os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://ev_app:ev_app@127.0.0.1:5433/ev_ai",
        )
        if not database_url.strip():
            raise ValueError("AI_DATABASE_URL or DATABASE_URL must not be empty")

        candidate_k = _positive_int("RAG_CANDIDATE_K", 30)
        top_k = _positive_int("RAG_TOP_K", 6)
        if candidate_k < top_k:
            raise ValueError("RAG_CANDIDATE_K must be greater than or equal to RAG_TOP_K")

        min_score = float(os.getenv("RAG_MIN_SCORE", "0.35"))
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("RAG_MIN_SCORE must be between 0 and 1")

        return cls(
            database_url=database_url.strip(),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).strip().rstrip("/"),
            deepseek_chat_model=os.getenv(
                "DEEPSEEK_CHAT_MODEL", "deepseek-v4-flash"
            ).strip(),
            deepseek_report_model=os.getenv(
                "DEEPSEEK_REPORT_MODEL", "deepseek-v4-flash"
            ).strip(),
            deepseek_timeout_seconds=_positive_float(
                "DEEPSEEK_TIMEOUT_SECONDS", 30.0
            ),
            deepseek_chat_max_tokens=_positive_int(
                "DEEPSEEK_CHAT_MAX_TOKENS", 1_200
            ),
            deepseek_report_max_tokens=_positive_int(
                "DEEPSEEK_REPORT_MAX_TOKENS", 2_000
            ),
            deepseek_report_thinking=_boolean(
                "DEEPSEEK_REPORT_THINKING", False
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
            ).strip(),
            embedding_dimension=_positive_int("EMBEDDING_DIMENSION", 768),
            embedding_batch_size=_positive_int("EMBEDDING_BATCH_SIZE", 32),
            rag_top_k=top_k,
            rag_candidate_k=candidate_k,
            rag_min_score=min_score,
            rag_allow_drafts=_boolean("RAG_ALLOW_DRAFTS", False),
            inference_base_url=os.getenv(
                "INFERENCE_BASE_URL", "http://127.0.0.1:8000"
            ).strip().rstrip("/"),
            vehicle_state_timeout_seconds=_positive_float(
                "VEHICLE_STATE_TIMEOUT_SECONDS", 2.0
            ),
            vehicle_state_max_age_seconds=_positive_int(
                "VEHICLE_STATE_MAX_AGE_SECONDS", 30
            ),
            allow_general_fallback=_boolean("ALLOW_GENERAL_FALLBACK", True),
            report_worker_poll_seconds=_positive_float(
                "REPORT_WORKER_POLL_SECONDS", 2.0
            ),
            report_worker_max_retries=_positive_int(
                "REPORT_WORKER_MAX_RETRIES", 3
            ),
            internal_api_token=os.getenv("AI_INTERNAL_TOKEN", "").strip(),
        )
