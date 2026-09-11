"""Unit tier: wrapper logic with fake embeddings (no model calls)."""

from src.chroma import ChromaStore


def _keyword_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic one-hot vectors by keyword for predictable neighbours."""
    vectors = []
    for text in texts:
        if "alpha" in text:
            vectors.append([1.0, 0.0])
        elif "beta" in text:
            vectors.append([0.0, 1.0])
        else:
            vectors.append([0.0, 0.0])
    return vectors


def test_page_blocks_name(tmp_path) -> None:
    store = ChromaStore(str(tmp_path), "any-model", embed_fn=_keyword_embed)
    assert store.page_blocks_name("abc123") == "page_blocks_abc123"


def test_add_query_round_trip(tmp_path) -> None:
    store = ChromaStore(str(tmp_path), "any-model", embed_fn=_keyword_embed)
    store.add_texts("docs", ["a", "b"], ["alpha document", "beta document"])
    result = store.query("docs", "alpha query", n_results=1)
    assert result["ids"][0][0] == "a"


def test_embed_model_configurable(tmp_path) -> None:
    seen: list[str] = []

    def _spy(texts: list[str]) -> list[list[float]]:
        seen.append("called")
        return _keyword_embed(texts)

    store = ChromaStore(str(tmp_path), "custom-model", embed_fn=_spy)
    assert store.embed_model == "custom-model"
    store.add_texts("docs", ["a"], ["alpha x"])
    assert seen == ["called"]
