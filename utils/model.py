"""
Module to build and return Hugging Face pipelines for various models.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
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
