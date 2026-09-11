"""Real components: embedded Chroma + LOCAL Ollama embeddings.

Local-only (no Cloud quota). Embed model comes from Settings (configurable).
"""

import pytest

from src.chroma import ChromaStore
from src.config import load_settings

pytestmark = pytest.mark.integration


def test_real_embed_round_trip(tmp_path) -> None:
    settings = load_settings()
    store = ChromaStore(str(tmp_path), settings.EMBED_MODEL, ollama_url=settings.OLLAMA_LOCAL_URL)
    vectors = store._ollama_embed(["hello world"])
    assert len(vectors) == 1 and len(vectors[0]) > 0
    assert all(isinstance(x, float) for x in vectors[0])

    store.add_texts(
        ChromaStore.page_blocks_name("job1"),
        ["p1", "p2"],
        ["the quick brown fox jumps", "quantum chromodynamics lecture notes"],
    )
    result = store.query(ChromaStore.page_blocks_name("job1"), "brown fox", n_results=1)
    assert result["ids"][0][0] == "p1"
