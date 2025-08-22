"""
CLI Commands.
"""
import os
from pathlib import Path

import faiss
import numpy as np

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from sentence_transformers import SentenceTransformer

from .indexing import load_corpus_chunks, build_faiss_index
from .model import generate_answer
from .retrieval import search, check_index_exists
from .text import jsonl_write
from .exceptions import EnvironmentVariableError


def index_command(console: Console) -> None:
    """
    CLI command to create/update the FAISS index from text files in data/corpus/*.txt.
    """
    console.rule("[bold green]Indexing Operation")

    texts, meta = load_corpus_chunks()

    console.print(f"Files read: {len(set(m['doc'] for m in meta))}")
    console.print(f"Total Chunks: {len(texts)}")
    console.print("Loading the embedding model...")

    if (EMBEDDINGS_MODEL := os.getenv("EMBEDDINGS_MODEL")) is None:
        raise EnvironmentVariableError("Model name is not set! Check the `.env` file.")

    if (DATA_EMBEDDINGS_PATH := os.getenv("DATA_EMBEDDINGS_PATH")) is None:
        raise EnvironmentVariableError(
            "DATA_EMBEDDINGS_PATH environment variable is not set."
        )

    if (DATA_PATH := os.getenv("DATA_PATH")) is None:
        raise EnvironmentVariableError("DATA_PATH environment variable is not set.")

    if (DATA_INDEX_PATH := os.getenv("DATA_INDEX_PATH")) is None:
        raise EnvironmentVariableError(
            "DATA_INDEX_PATH environment variable is not set."
        )

    if (DATA_META_PATH := os.getenv("DATA_META_PATH")) is None:
        raise EnvironmentVariableError(
            "DATA_META_PATH environment variable is not set."
        )

    emb_model = SentenceTransformer(EMBEDDINGS_MODEL)

    console.print("Creating the embeddings (it may take a while) ...")

    emb = emb_model.encode(
        texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True
    )

    np.save(DATA_EMBEDDINGS_PATH, emb)

    console.print("Creating FAISS index...")
    index = build_faiss_index(emb)

    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(DATA_INDEX_PATH))

    console.print("Storing Metadata...")

    Path(DATA_META_PATH).unlink(missing_ok=True)

    for m in meta:
        jsonl_write(Path(DATA_META_PATH), m)

    console.print(
        Panel.fit(
            "[bold cyan]Done![/] Index saved on [yellow]data/index.faiss[/], "
            "metadata on [yellow]data/meta.jsonl[/].")
        )


def chat_command(console: Console, model_mode: str = "t5"):

    console.rule("[bold green]FAQ Chatbot (RAG)")

    # First check if the indexing task was performed
    if not check_index_exists():
        console.print(
            Panel.fit(
                "[bold red]❌ Index not found![/]\n\n"
                "Index files were not found or are incomplete.\n"
                "Please run the indexing process first:\n\n"
                "[bold cyan]python app.py index[/]\n\n"
                "This will create the necessary files for the chatbot to work.",
                title="[bold red]Error - Indexing Required[/]",
                border_style="red"
            )
        )
        return

    console.print("[bold green]✅ Indexação encontrada! Iniciando chatbot...[/]")
    console.print("Digite sua pergunta. Digite 'sair', 'exit' ou 'quit' para fechar o chat.")

    if (TOP_K := int(os.getenv("TOP_K"))) is None:
        raise EnvironmentVariableError("TOP_K environment variable is not set.")

    while True:
        q = Prompt.ask("[bold cyan]Você")
        if q.strip().lower() in {"sair", "exit", "quit"}:
            console.print("[bold yellow]Closing the chat.[/]")
            break

        hits = search(q, top_k=TOP_K)
        if not hits:
            console.print("[red]Nada encontrado no índice.[/]")
            continue
        answer = generate_answer(q, hits, mode=model_mode)

        refs = "\n".join([
            f"- {h['doc']} (chunk {h['chunk_id']}, score {h['score']:.3f})"
             for h in hits
        ])

        console.print(Panel.fit(answer, title="[bold]Resposta[/]"))
        console.print(Panel.fit(refs, title="[bold]Fontes[/]"))

        jsonl_write("conversations/history.jsonl", {
            "question": q,
            "answer": answer,
            "hits": hits,
        })
