"""
app/inference.py

Pure inference helpers used by every tab.
No Streamlit imports — these are plain functions so they can also be called
from CLI or tests without a running Streamlit server.

All functions return a dict with at least:
  response   str    — the generated text
  latency_s  float  — wall-clock seconds for generate()
  tokens_in  int    — prompt token count
  tokens_out int    — generated token count
"""

from __future__ import annotations
import time
from pathlib import Path

# ── Shared system prompt ──────────────────────────────────────────────────────
_SYSTEM_BASE = (
    "You are a precise and reliable legal assistant specialised in the "
    "Income-Tax Act, 2025 (as amended by the Finance Act, 2026). "
    "Answer questions based strictly on the provisions of the Act. "
    "Cite the relevant section number in your response. "
    "Do not speculate beyond what the text of the law states."
)

_SYSTEM_RAG = (
    "You are a precise and reliable legal assistant specialised in the "
    "Income-Tax Act, 2025 (as amended by the Finance Act, 2026). "
    "Answer strictly based on the retrieved legal provisions below. "
    "Always cite the relevant section number. "
    "If the retrieved provisions do not contain the answer, state that clearly."
)

_GENERATE_KWARGS = dict(
    do_sample=False,
    temperature=None,
    top_p=None,
    repetition_penalty=1.1,
)


def _build_plain_prompt(question: str) -> str:
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{_SYSTEM_BASE}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = [
        f"[Provision {i} — Section {c['section_num']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ]
    context = "\n\n".join(context_parts)
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{_SYSTEM_RAG}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Based on the following retrieved legal provisions from the Income-Tax Act, 2025, "
        "answer the question accurately:\n\n"
        f"--- RETRIEVED PROVISIONS ---\n{context}\n--- END ---\n\n"
        f"Question: {question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _generate(prompt: str, model, tokenizer, max_new_tokens: int = 512) -> dict:
    import torch
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    tokens_in = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            **_GENERATE_KWARGS,
        )

    latency = time.perf_counter() - t0
    gen_ids  = out_ids[0][tokens_in:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return {
        "response":   response,
        "latency_s":  round(latency, 2),
        "tokens_in":  tokens_in,
        "tokens_out": len(gen_ids),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_base(question: str, model, tokenizer, max_new_tokens: int = 512) -> dict:
    """Plain base-model inference (no RAG, no fine-tuning)."""
    prompt = _build_plain_prompt(question)
    return _generate(prompt, model, tokenizer, max_new_tokens)


def run_finetuned(question: str, model, tokenizer, max_new_tokens: int = 512) -> dict:
    """Fine-tuned model inference (no RAG)."""
    prompt = _build_plain_prompt(question)
    return _generate(prompt, model, tokenizer, max_new_tokens)


def run_rag(
    question: str,
    model,
    tokenizer,
    collection,
    k: int = 6,
    max_new_tokens: int = 512,
) -> dict:
    """Fine-tuned model + RAG: retrieve context, then generate."""
    # 1. Retrieve
    t_ret = time.perf_counter()
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    retrieval_ms = round((time.perf_counter() - t_ret) * 1000, 1)

    chunks = [
        {
            "section_num":   meta.get("section_num", "?"),
            "section_title": meta.get("section_title", ""),
            "text":          doc,
            "distance":      round(dist, 4),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

    # 2. Generate
    prompt = _build_rag_prompt(question, chunks)
    gen    = _generate(prompt, model, tokenizer, max_new_tokens)

    return {
        **gen,
        "chunks":        chunks,
        "retrieval_ms":  retrieval_ms,
    }
