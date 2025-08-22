#%%
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
import orjson

from utils.indexing import build_faiss_index, load_corpus_chunks
from utils.model import build_t5_pipeline, build_causal_pipeline
from utils.retrieval import search
from utils.text import chunk_text, jsonl_write, jsonl_read

console = Console()


def index_command() -> None:
    """
    CLI command to create/update the FAISS index from text files in data/corpus/*.txt.
    """
    console.rule("[bold green]Indexing Operation")

    texts, meta = load_corpus_chunks()

    console.print(f"Files read: {len(set(m['doc'] for m in meta))}")
    console.print(f"Total Chunks: {len(texts)}")
    console.print("Loading the embedding model...")

    if EMBEDDINGS_MODEL := os.getenv("EMBEDDINGS_MODEL") is None:
        raise EnvironmentError("Model name is not set! Check the `.env` file.")

    if DATA_EMBEDDINGS_PATH := os.getenv("DATA_EMBEDDINGS_PATH") is None:
        raise EnvironmentError("DATA_EMBEDDINGS_PATH environment variable is not set.")

    if DATA_PATH := os.getenv("DATA_PATH") is None:
        raise EnvironmentError("DATA_PATH environment variable is not set.")

    if DATA_INDEX_PATH := os.getenv("DATA_INDEX_PATH") is None:
        raise EnvironmentError("DATA_INDEX_PATH environment variable is not set.")

    if DATA_META_PATH := os.getenv("DATA_META_PATH") is None:
        raise EnvironmentError("DATA_META_PATH environment variable is not set.")

    emb_model = SentenceTransformer(EMBEDDINGS_MODEL)

    console.print("Creating the embeddings (it may take a while) ...")

    emb = emb_model.encode(
        texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True
    )

    np.save(DATA_EMBEDDINGS_PATH, emb)

    console.print("Creating FAISS index...")
    index = build_faiss_index(emb)

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(DATA_INDEX_PATH))

    console.print("Storing Metadata...")

    DATA_META_PATH.unlink(missing_ok=True)
    for m in meta:
        jsonl_write(DATA_META_PATH, m)

    console.print(
        Panel.fit(
            "[bold cyan]Done![/] Index saved on [yellow]data/index.faiss[/], "
            "metadata on [yellow]data/meta.jsonl[/].")
        )


T5_SYSTEM_PROMPT = """Responda à pergunta usando SOMENTE o contexto abaixo.
Se a resposta não estiver claramente no contexto, diga honestamente que NÃO há informação suficiente.
Seja conciso e objetivo.

[Contexto]
{context}

[Pergunta]
{question}

[Resposta]
"""

CAUSAL_SYSTEM_PROMPT = """Você é um assistente útil. Responda apenas com base no CONTEXTO. Se o contexto for insuficiente, diga que não há informação suficiente.

[CONTEXTO]
{context}

[PERGUNTA]
{question}

[RESPOSTA]
"""

def generate_answer(question: str, hits: List[Dict], mode: str = "t5") -> str:
    context = "\n\n".join([f"(Trecho {h['rank']} | {h['doc']} #{h['chunk_id']})\n{h['text']}" for h in hits])
    if mode == "t5":
        pipe = build_t5_pipeline()
        prompt = T5_SYSTEM_PROMPT.format(context=context, question=question)
        out = pipe(prompt, max_new_tokens=256, do_sample=False)
        return out[0]["generated_text"].strip()
    else:
        pipe = build_causal_pipeline()
        prompt = CAUSAL_SYSTEM_PROMPT.format(context=context, question=question)
        out = pipe(prompt, max_new_tokens=256, do_sample=False)
        # Alguns modelos retornam o prompt + resposta; corte simples:
        text = out[0]["generated_text"]
        split_marker = "[RESPOSTA]"
        if split_marker in text:
            text = text.split(split_marker, 1)[-1]
        return text.strip()


# =========================
# CLI de Chat
# =========================
def chat_command(model_mode: str = "t5"):
    console.rule("[bold green]FAQ Chatbot (RAG)")
    console.print("Insert your prompt. Type 'exit' or 'quit' to close the chat.")
    console.print("Digite sua pergunta. Use 'sair' para encerrar.")
    while True:
        q = Prompt.ask("[bold cyan]Você")
        if q.strip().lower() in {"sair", "exit", "quit"}:
            console.print("[bold yellow]Closing the chat.[/]")
            break
        # hits = search(q, top_k=TOP_K)
        hits = search(q, top_k=4)
        if not hits:
            console.print("[red]Nada encontrado no índice.[/]")
            continue
        answer = generate_answer(q, hits, mode=model_mode)

        # Mostrar fontes
        refs = "\n".join([f"- {h['doc']} (chunk {h['chunk_id']}, score {h['score']:.3f})" for h in hits])
        console.print(Panel.fit(answer, title="[bold]Resposta[/]"))
        console.print(Panel.fit(refs, title="[bold]Fontes[/]"))

        # Salvar histórico
        # jsonl_write(HISTORY_PATH, {
        jsonl_write("conversations/history.jsonl", {
            "question": q,
            "answer": answer,
            "hits": hits,
        })


def main() -> None:
    """
    Main entry point for the chatbot application.
    Parses command line arguments and executes the appropriate command.
    """
    parser = argparse.ArgumentParser(description="Simple CHATBOT w/ RAG using FAISS + T5")

    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("index", help="Create/update index from data/corpus/*.txt")

    chat_p = sub.add_parser("chat", help="Starts the chatbot interface")
    chat_p.add_argument(
        "--mode",
        choices=["t5", "causal"],
        default="t5",
        help="Generation model: 't5' (default) or 'causal' (ex.: Mistral local)"
    )

    args = parser.parse_args()

    if args.cmd == "index":
        index_command()
    elif args.cmd == "chat":
        chat_command(model_mode=args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":

    main()

# %%
