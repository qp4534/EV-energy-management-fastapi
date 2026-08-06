"""Shared AI adapters used by the chatbot and report worker."""

from .config import AISettings
from .contracts import RetrievedChunk
from .deepseek import DeepSeekClient, DeepSeekUnavailable

__all__ = [
    "AISettings",
    "DeepSeekClient",
    "DeepSeekUnavailable",
    "RetrievedChunk",
]
