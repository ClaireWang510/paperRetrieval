from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    model_name_or_path = (
        os.environ.get("BGE_EMBEDDING_MODEL_PATH")
        or os.environ.get("BGE_EMBEDDING_MODEL_NAME")
        or "BAAI/bge-m3"
    )
    device = os.environ.get("BGE_EMBEDDING_DEVICE", "auto").strip().lower()
    if device not in {"cpu", "cuda", "auto"}:
        device = "auto"

    kwargs = {}
    if device != "auto":
        kwargs["device"] = device
    return SentenceTransformer(model_name_or_path, **kwargs)
