"""
Module to build and return Hugging Face pipelines for various models.
"""
from typing import Any

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
)


T5_SYSTEM_PROMPT = (
    "Responda à pergunta usando SOMENTE o contexto a seguir. \n"
    "Se a resposta não estiver claramente no contexto, diga honestamente que NÃO há "
    "informação suficiente. Seja conciso e objetivo."
    "[Contexto] \n"
    "{context} \n\n"
    "[Pergunta] \n"
    "{question} \n\n"
    "[Resposta]"
)


CAUSAL_SYSTEM_PROMPT = (
    "Você é um assistente útil. Responda apenas com base no CONTEXTO. \n"
    "Se o contexto for insuficiente, diga que não há informação suficiente. \n"
    "[Contexto] \n"
    "{context} \n\n"
    "[Pergunta] \n"
    "{question} \n\n"
    "[Resposta]"
)


def build_t5_pipeline(tokenizer_name: str, model_name: str) -> pipeline:
    """
    Build a T5-based text-to-text generation pipeline.

    Args:
        tokenizer_name: Name of the tokenizer to use.
        model_name: Name of the T5 model to use.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tok)


def build_causal_pipeline(tokenizer_name: str, model_name: str) -> pipeline:
    """
    Build a causal language model pipeline for text generation.

    Args:
        model_name: Name of the causal language model to use (e.g., Mistral).
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tok)


def generate_answer(
    question: str,
    hits: list[dict[str, Any]],
    mode: str = "t5"
) -> str:

    context = "\n\n".join([
        f"(Trecho {h['rank']} | {h['doc']} #{h['chunk_id']})\n{h['text']}"
        for h in hits
    ])

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
