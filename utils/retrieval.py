"""
Retrieval utilities for handling and processing retrieval tasks.
"""
import os
from pathlib import Path

import faiss
import numpy as np

from typing import Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from .exceptions import EnvironmentVariableError
from .text import jsonl_read


load_dotenv()


def check_index_exists() -> bool:
    """
    Verifica se os arquivos necessários para a indexação existem.

    Returns:
        bool: True se todos os arquivos de índice existem, False caso contrário.
    """
    try:
        index_path = os.getenv("DATA_INDEX_PATH")
        meta_path = os.getenv("DATA_META_PATH")
        embeddings_path = os.getenv("DATA_EMBEDDINGS_PATH")

        if not all([index_path, meta_path, embeddings_path]):
            return False

        return (
            Path(index_path).exists() and
            Path(meta_path).exists() and
            Path(embeddings_path).exists()
        )
    except Exception:
        return False


def load_index_and_meta() -> tuple[faiss.Index, list[dict[str, Any]], np.ndarray]:

    if (INDEX_PATH := os.getenv("DATA_INDEX_PATH")) is None:
        raise EnvironmentVariableError(
            "DATA_INDEX_PATH environment variable is not set. "
            "Please run: python rag.py index"
        )

    if (META_PATH := os.getenv("DATA_META_PATH")) is None:
        raise EnvironmentVariableError(
            "Metadata not found."
        )

    if (EMBEDDINGS_PATH := os.getenv("DATA_EMBEDDINGS_PATH")) is None:
        raise EnvironmentVariableError(
            "DATA_EMBEDDINGS_PATH environment variable is not set."
        )

    return (
        faiss.read_index(str(INDEX_PATH)),
        jsonl_read(META_PATH),
        np.load(EMBEDDINGS_PATH),
    )


def search(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """
    Search for the top-k most relevant documents for a given query. Uses a pre-trained
    SentenceTransformer model to encode the query, and performs a similarity search
    using a FAISS index.

    Args:
        query: The search query string.
        top_k: The number of top results to return.
    """
    if (EMBEDDINGS_MODEL := os.getenv("EMBEDDINGS_MODEL")) is None:
        raise EnvironmentVariableError(
            "EMBEDDINGS_MODEL variable is not set. "
            "Please check your configuration."
        )

    embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)

    query = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query)

    index, meta, _ = load_index_and_meta()

    distances, indices = index.search(query.astype(np.float32), top_k)
    hits = []

    for rank, idx in enumerate(indices[0]):
        m = meta[int(idx)]
        hits.append({
            "rank": rank + 1,
            "score": float(distances[0][rank]),
            "doc": m["doc"],
            "chunk_id": m["chunk_id"],
            "text": m["text"],
        })

    return hits
