"""Shared local sentence-embedding backend.

Used by both RPD (step-description embeddings) and Correct Answer Clustering
(full-solution embeddings), so the model is loaded once and cached here
instead of once per metric module.
"""

from typing import Sequence

import numpy as np

_ST_MODEL = None
_ST_MODEL_NAME = None


def embed_texts(texts: Sequence[str], model: str) -> np.ndarray:
    """Embed a list of strings with a local sentence-transformers model.

    Returns L2-normalized float32 embeddings, so `A @ B.T` gives cosine
    similarity directly.
    """
    global _ST_MODEL, _ST_MODEL_NAME
    from sentence_transformers import SentenceTransformer

    if _ST_MODEL is None or _ST_MODEL_NAME != model:
        _ST_MODEL = SentenceTransformer(
            model, trust_remote_code=True,
            model_kwargs={"device_map": "auto"},
            device="cuda",
        )
        _ST_MODEL_NAME = model

    return _ST_MODEL.encode(
        list(texts),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
