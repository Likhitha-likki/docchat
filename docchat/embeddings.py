"""Local embeddings via fastembed (ONNX). Default model is ~130 MB and runs fine on 8 GB RAM."""
import numpy as np

from .config import EMBED_CACHE_DIR, EMBED_MODEL


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, cache_dir: str | None = EMBED_CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    @property
    def model(self):
        # Lazy load: the model is only downloaded/loaded when first used
        if self._model is None:
            from fastembed import TextEmbedding

            kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
            self._model = TextEmbedding(model_name=self.model_name, **kwargs)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.model.embed(texts)), dtype="float32")
        return _normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        # query_embed adds the retrieval prefix that BGE models expect
        vectors = np.array(list(self.model.query_embed(query)), dtype="float32")
        return _normalize(vectors)  # shape (1, dim)
