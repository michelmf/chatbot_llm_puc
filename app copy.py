#%%
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
import orjson

# # =========================
# # Configurações
# # =========================
# DATA_DIR = Path("data")
# CORPUS_DIR = DATA_DIR / "corpus"
# INDEX_PATH = DATA_DIR / "index.faiss"
# META_PATH = DATA_DIR / "meta.jsonl"
# EMB_PATH = DATA_DIR / "embeddings.npy"  # opcional (debug)
# CONV_DIR = Path("conversations")
# HISTORY_PATH = CONV_DIR / "history.jsonl"

# EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# # Geração padrão: FLAN-T5 (text2text)
# GEN_MODEL_T5 = "google/flan-t5-base"
# # Opcional (causal): Mistral 7B Instruct local, se você já tiver
# GEN_MODEL_CAUSAL = "mistralai/Mistral-7B-Instruct-v0.2"

# CHUNK_SIZE = 300     # caracteres
# CHUNK_OVERLAP = 100  # caracteres
# TOP_K = 4            # quantos trechos recuperar

console = Console()


# # =========================
# # Utilidades
# # =========================
# def jsonl_write(path: Path, obj: Dict):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("ab") as f:
#         f.write(orjson.dumps(obj) + b"\n")

# def jsonl_read(path: Path) -> List[Dict]:
#     if not path.exists():
#         return []
#     out = []
#     with path.open("rb") as f:
#         for line in f:
#             if line.strip():
#                 out.append(orjson.loads(line))
#     return out

# def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
#     text = text.strip().replace("\r\n", "\n")
#     chunks = []
#     start = 0
#     n = len(text)
#     while start < n:
#         end = min(start + chunk_size, n)
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#         if end == n:
#             break
#         start = end - overlap  # move com overlap
#         if start < 0:
#             start = 0
#     return chunks


# # =========================
# # Indexação
# # =========================
# def load_corpus_chunks() -> Tuple[List[str], List[Dict]]:
#     """
#     Lê todos .txt em data/corpus e retorna (chunks, metadados).
#     Cada metadado inclui: {"doc": nome_arquivo, "chunk_id": i, "text": chunk}
#     """
#     assert CORPUS_DIR.exists(), f"Pasta {CORPUS_DIR} não existe."
#     all_chunks, meta = [], []
#     for p in sorted(CORPUS_DIR.glob("*.txt")):
#         txt = p.read_text(encoding="utf-8", errors="ignore")
#         chunks = chunk_text(txt, CHUNK_SIZE, CHUNK_OVERLAP)
#         for i, ch in enumerate(chunks):
#             all_chunks.append(ch)
#             meta.append({"doc": p.name, "chunk_id": i, "text": ch})
#     return all_chunks, meta

# def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
#     # Normaliza para usar similaridade de cosseno via produto interno
#     faiss.normalize_L2(embeddings)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings.astype(np.float32))
#     return index

from utils.text import load_corpus_chunks, chunk_text
from utils.indexing import build_faiss_index, jsonl_write, jsonl_read

def index_command():
    console.rule("[bold green]Indexação")
    texts, meta = load_corpus_chunks()
    console.print(f"Arquivos lidos: {len(set(m['doc'] for m in meta))}")
    console.print(f"Total de chunks: {len(texts)}")

    console.print("Carregando modelo de embeddings...")
    emb_model = SentenceTransformer(EMBEDDINGS_MODEL)

    console.print("Gerando embeddings (isso pode demorar na 1ª vez)...")
    emb = emb_model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMB_PATH, emb)

    console.print("Construindo índice FAISS...")
    index = build_faiss_index(emb)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    console.print("Gravando metadados...")
    META_PATH.unlink(missing_ok=True)
    for m in meta:
        jsonl_write(META_PATH, m)

    console.print(Panel.fit("[bold cyan]Pronto![/] Índice salvo em [yellow]data/index.faiss[/], metadados em [yellow]data/meta.jsonl[/]."))


# # =========================
# # Recuperação
# # =========================
# def load_index_and_meta() -> Tuple[faiss.Index, List[Dict], np.ndarray]:
#     assert INDEX_PATH.exists(), "Índice não encontrado. Rode: python rag.py index"
#     assert META_PATH.exists(), "Metadados não encontrados."
#     index = faiss.read_index(str(INDEX_PATH))
#     meta = jsonl_read(META_PATH)
#     emb = np.load(EMB_PATH)
#     return index, meta, emb

# def search(query: str, top_k: int = TOP_K) -> List[Dict]:
#     emb_model = SentenceTransformer(EMBEDDINGS_MODEL)
#     q = emb_model.encode([query], convert_to_numpy=True)
#     faiss.normalize_L2(q)
#     index, meta, _ = load_index_and_meta()
#     distances, indices = index.search(q.astype(np.float32), top_k)
#     hits = []
#     for rank, idx in enumerate(indices[0]):
#         m = meta[int(idx)]
#         hits.append({
#             "rank": rank + 1,
#             "score": float(distances[0][rank]),
#             "doc": m["doc"],
#             "chunk_id": m["chunk_id"],
#             "text": m["text"],
#         })
#     return hits


# =========================
# Geração (Modelos)
# =========================
def build_t5_pipeline():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_T5)
    return pipeline("text2text-generation", model=model, tokenizer=tok)

def build_causal_pipeline(model_name=GEN_MODEL_CAUSAL):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tok)

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
    console.print("Digite sua pergunta. Use 'sair' para encerrar.")
    while True:
        q = Prompt.ask("[bold cyan]Você")
        if q.strip().lower() in {"sair", "exit", "quit"}:
            console.print("[bold yellow]Encerrando.[/]")
            break
        hits = search(q, top_k=TOP_K)
        if not hits:
            console.print("[red]Nada encontrado no índice.[/]")
            continue
        answer = generate_answer(q, hits, mode=model_mode)

        # Mostrar fontes
        refs = "\n".join([f"- {h['doc']} (chunk {h['chunk_id']}, score {h['score']:.3f})" for h in hits])
        console.print(Panel.fit(answer, title="[bold]Resposta[/]"))
        console.print(Panel.fit(refs, title="[bold]Fontes[/]"))

        # Salvar histórico
        jsonl_write(HISTORY_PATH, {
            "question": q,
            "answer": answer,
            "hits": hits,
        })


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="RAG simples com FAISS + FLAN-T5")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("index", help="Cria/atualiza o índice a partir de data/corpus/*.txt")

    chat_p = sub.add_parser("chat", help="Inicia o chat no terminal")
    chat_p.add_argument("--mode", choices=["t5", "causal"], default="t5",
                        help="Modelo de geração: 't5' (padrão) ou 'causal' (ex.: Mistral local)")

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
