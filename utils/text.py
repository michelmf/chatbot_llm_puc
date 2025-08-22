"""
Utils for text processing, including JSONL read/write and text chunking.
"""

from pathlib import Path
from typing import Any

import orjson


def jsonl_write(path: Path, obj: dict[str, Any]) -> None:
    """
    Write a dictionary to a JSON Lines file. Each object is serialized to a single line
    in the file. If the file does not exist, it will be created. If the directory does
    not exist, it will be created.

    Args:
        path: The file path where the JSON Lines data will be written.
        obj: The dictionary object to write to the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(orjson.dumps(obj) + b"\n")


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    """
    Read a JSON Lines file and return a list of dictionaries. Each line in the file
    is expected to contain a valid JSON object. If the file does not exist, an empty
    list is returned.

    Args:
        path: The file path from which to read the JSON Lines data.
    """
    if not path.exists():
        return []
    out = []
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                out.append(orjson.loads(line))
    return out


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a text into chunks of specified size with a defined overlap. The text is
    stripped of leading and trailing whitespace and normalized to use newline characters
    consistently. Each chunk is trimmed of whitespace.

    Args:
        text: The input text to be chunked.
        chunk_size: The maximum size of each chunk.
        overlap: The number of characters that overlap between consecutive chunks.
    """

    text = text.strip().replace("\r\n", "\n")

    chunks = []
    start = 0

    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start: end].strip()

        if chunk:
            chunks.append(chunk)
        if end == n:
            break

        start = end - overlap

        if start < 0:
            start = 0

    return chunks
