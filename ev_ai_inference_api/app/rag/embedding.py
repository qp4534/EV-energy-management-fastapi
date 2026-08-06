from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any


class EmbeddingDependencyMissing(RuntimeError):
    pass


class SentenceTransformerEmbedder:
    """Lazy multilingual-e5 embedder.

    The large ML dependency is imported only by chatbot/ingestion processes, so
    the existing realtime inference API does not pay its startup or memory cost.
    """

    def __init__(
        self,
        model_name: str,
        *,
        dimension: int = 768,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model

            def load() -> Any:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise EmbeddingDependencyMissing(
                        "Install requirements-ai.txt to use local embeddings"
                    ) from exc
                return SentenceTransformer(self.model_name)

            self._model = await asyncio.to_thread(load)
            detected = self._model.get_sentence_embedding_dimension()
            if detected != self.dimension:
                raise ValueError(
                    f"embedding dimension mismatch: expected {self.dimension}, got {detected}"
                )
            return self._model

    async def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._ensure_model()

        def encode() -> list[list[float]]:
            vectors = model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()

        values = await asyncio.to_thread(encode)
        if any(len(value) != self.dimension for value in values):
            raise ValueError("embedding provider returned an unexpected dimension")
        return values

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._encode([f"passage: {text}" for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        values = await self._encode([f"query: {text}"])
        return values[0]
