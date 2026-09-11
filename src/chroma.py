"""ChromaDB wrapper — §6.2 collections, local-Ollama embeddings.

Collections: `page_blocks_{job_id}` (header/footer dedup) + `documents`
(future search/RAG). Embeddings come from the LOCAL Ollama daemon —
the embed model is configurable per Settings.EmbedFn is
injectable so unit tests run without any model call.
"""

from collections.abc import Callable, Sequence
from typing import Any

import ollama

EmbedFn = Callable[[list[str]], list[list[float]]]


class ChromaStore:
    """Thin store: caller owns ids/documents; we own embeddings + collections."""

    def __init__(
        self,
        path: str,
        embed_model: str,
        ollama_url: str = "http://localhost:11434",
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self._path = path
        self.embed_model = embed_model
        self._ollama_url = ollama_url
        self._embed_fn = embed_fn or self._ollama_embed
        self._client: Any = None

    @staticmethod
    def page_blocks_name(job_id: str) -> str:
        """Deterministic per-job collection name (§6.2)."""
        return f"page_blocks_{job_id}"

    def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        response = ollama.Client(host=self._ollama_url).embed(model=self.embed_model, input=texts)
        return [list(map(float, vec)) for vec in response.embeddings]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Public embed seam (NFR-10): one vector per text, in order.

        Injection point for the eval harness and any external retrieval
        experiment — no private-attribute reach-ins.
        """
        return self._embed_fn(list(texts))

    def delete_collection(self, name: str) -> None:
        """Drop a collection if it exists (idempotent; eval resets)."""
        client = self._client_or_create()
        if any(c.name == name for c in client.list_collections()):
            client.delete_collection(name)

    def _client_or_create(self) -> Any:
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._path)
        return self._client

    def _collection(self, name: str) -> Any:
        return self._client_or_create().get_or_create_collection(name)

    def add_texts(
        self,
        collection: str,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Embed + upsert; caller-chosen stable ids (re-runs overwrite)."""
        coll = self._collection(collection)
        coll.upsert(
            ids=list(ids),
            documents=list(documents),
            embeddings=self._embed_fn(list(documents)),
            metadatas=list(metadatas) if metadatas is not None else None,
        )

    def query(self, collection: str, text: str, n_results: int = 5) -> dict[str, Any]:
        """Nearest-neighbour search; returns the raw Chroma result dict."""
        coll = self._collection(collection)
        result: dict[str, Any] = coll.query(
            query_embeddings=self._embed_fn([text]), n_results=n_results
        )
        return result


__all__ = ["ChromaStore", "EmbedFn"]
