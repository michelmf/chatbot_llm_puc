"""
Indexing utilities for handling various indexing operations.
"""
import os

import faiss
import numpy as np

from glob import glob
from typing import Any

from dotenv import load_dotenv
from .text import chunk_text
from .exceptions import CorpusDataDoesNotExist, EnvironmentVariableError


load_dotenv()


def load_corpus_chunks() -> tuple[list[str], list[dict[str, Any]]]:
    """
    Read txt files from corpus and returns chunks and metadata.
    Each metadata includes: {"doc": filename, "chunk_id": i, "text": chunk}
    """

    if (CORPUS_DIR := os.getenv("DATA_CORPUS_PATH")) is None:
        raise CorpusDataDoesNotExist("Could not find corpus data! Check the folder.")

    if (CHUNK_SIZE := os.getenv("CHUNK_SIZE")) is None:
        raise EnvironmentVariableError("CHUNK_SIZE environment variable is not set.")

    if (CHUNK_OVERLAP := os.getenv("CHUNK_OVERLAP")) is None:
        raise EnvironmentVariableError("CHUNK_OVERLAP environment variable is not set.")

    all_chunks, meta = [], []

    for p in sorted(glob(os.path.join(CORPUS_DIR, "*.txt"))):

        with open(p, 'r', encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        chunks = chunk_text(txt, int(CHUNK_SIZE), int(CHUNK_OVERLAP))

        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({"doc": os.path.basename(p), "chunk_id": i, "text": ch})

    return all_chunks, meta


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Normalize embeddings to use cosine similarity via inner product,
    and build a FAISS index for efficient similarity search.

    Args:
        embeddings: Array of embeddings to index.
    """
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    return index
