"""
base_eval.py  —  Unified Evaluation Pipeline
Evaluates three configurations on the same held-out test set:

  base          — un-fine-tuned LLaMA (context from test record)
  finetuned     — fine-tuned model    (context from test record)
  finetuned-rag — fine-tuned model    (context retrieved live from ChromaDB)

Pipeline steps :-
split         Create stratified 85/15 train/test split (run ONCE before training).
infer         Run model inference → predictions JSONL.
judge         Score predictions with Groq llama3-70b-8192 LLM-judge (via judge_groq.py).
mock          Score with heuristic judge (no API key, for testing).
report        Reprint a saved report.
compare       Side-by-side delta table: base vs finetuned-rag.
full          split + infer + judge end-to-end.

Output files :-
  eval/base_model_predictions.jsonl   eval/base_judge_results.jsonl   eval/base_eval_report.txt
  eval/after_model_predictions.jsonl  eval/after_judge_results.jsonl  eval/after_eval_report.txt
  eval/rag_model_predictions.jsonl    eval/rag_judge_results.jsonl    eval/rag_eval_report.txt
  eval/compare_report.txt             ← written by --mode compare

Usage :-
# 1. Split (once, before fine-tuning):
python src/base_eval.py --mode split

# 2. Evaluate base model:
python src/base_eval.py --mode full --model-type base

# 3. Evaluate fine-tuned + RAG (the meaningful post-training eval):
python src/base_eval.py --mode full --model-type finetuned-rag

# 4. Compare base vs fine-tuned+RAG side-by-side:
python src/base_eval.py --mode compare

# Evaluate plain fine-tuned (no RAG) if you want a three-way comparison:
python src/base_eval.py --mode full --model-type finetuned

# Inference-only (no judge):
python src/base_eval.py --mode infer --model-type finetuned-rag

# Re-judge existing predictions:
python src/base_eval.py --mode judge --model-type finetuned-rag

# Mock judge (no API key):
python src/base_eval.py --mode mock --model-type finetuned-rag

# Set the Groq API key via environment variable before running:
export GROQ_API_KEY=gsk_...
python src/base_eval.py --mode full --model-type finetuned-rag

Notes :-
- ChromaDB index must be built before using finetuned-rag:
    python src/rag.py build
- Run --mode split ONCE before fine-tuning begins. Never re-run it.
- All three model-types evaluate against the same data/processed/test.jsonl.
"""

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

#  Paths 
ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = ROOT_DIR / "eval"
MODELS_DIR = ROOT_DIR / "models"
FINETUNE_DIR = ROOT_DIR / "finetune"

DATASET_JSONL     = PROCESSED_DIR / "qa_dataset.jsonl"
TRAIN_JSONL       = PROCESSED_DIR / "train.jsonl"
TEST_JSONL        = PROCESSED_DIR / "test.jsonl"

# Base model output paths
BASE_PREDICTIONS_JSONL  = EVAL_DIR / "base_model_predictions.jsonl"
BASE_JUDGE_JSONL        = EVAL_DIR / "base_judge_results.jsonl"
BASE_REPORT_TXT         = EVAL_DIR / "base_eval_report.txt"

# Fine-tuned model output paths
AFTER_PREDICTIONS_JSONL = EVAL_DIR / "after_model_predictions.jsonl"
AFTER_JUDGE_JSONL       = EVAL_DIR / "after_judge_results.jsonl"
AFTER_REPORT_TXT        = EVAL_DIR / "after_eval_report.txt"

# Fine-tuned + RAG output paths
RAG_PREDICTIONS_JSONL   = EVAL_DIR / "rag_model_predictions.jsonl"
RAG_JUDGE_JSONL         = EVAL_DIR / "rag_judge_results.jsonl"
RAG_REPORT_TXT          = EVAL_DIR / "rag_eval_report.txt"

# Compare report
COMPARE_REPORT_TXT      = EVAL_DIR / "compare_report.txt"

# RAG index location (must match step4_rag.py)
RAG_CHROMA_DIR          = Path(".") / "rag" / "chroma_db"
RAG_COLLECTION          = "ita_2025"
RAG_EMBED_MODEL         = "BAAI/bge-base-en-v1.5"
RAG_TOP_K               = 6

# Model directories
BASE_MODEL_DIR      = str(MODELS_DIR / "llama_3_2_1b")    
BASE_MODEL_HF       = "meta-llama/Llama-3.2-1B-Instruct"  
FINETUNED_MODEL_DIR = str(FINETUNE_DIR / "final_model")

# Legacy aliases (used by run_mock_judge / run_judge default args)
PREDICTIONS_JSONL = BASE_PREDICTIONS_JSONL
JUDGE_JSONL       = BASE_JUDGE_JSONL
REPORT_TXT        = BASE_REPORT_TXT

#  Split 
TEST_FRACTION     = 0.15
MIN_TEST_PER_TYPE = 3
SEED              = 42

#  Model 
MAX_NEW_TOKENS    = 512

# Judge — delegated to judge_groq.py

from judge_groq import run_judge_batched, GROQ_MODEL  



# UTILITIES
def create_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

def _write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]



# STEP 1 — STRATIFIED TRAIN / TEST SPLIT

def make_split() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    records = _load_jsonl(DATASET_JSONL)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r["qa_type"]].append(r)

    train_all: list[dict] = []
    test_all:  list[dict] = []

    print("\n[split] Stratified 85/15 split by qa_type:")
    print(f"  {'type':<22}  {'total':>5}  {'train':>5}  {'test':>5}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*5}  {'-'*5}")

    for qa_type, recs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        rng.shuffle(recs)
        n_test = max(MIN_TEST_PER_TYPE, int(len(recs) * TEST_FRACTION))
        if len(recs) > 1:
            n_test = min(n_test, len(recs) - 1)
        else:
            n_test = 0
        test_all.extend(recs[:n_test])
        train_all.extend(recs[n_test:])
        print(f"  {qa_type:<22}  {len(recs):>5}  {len(recs)-n_test:>5}  {n_test:>5}")

    total = len(train_all) + len(test_all)
    print(f"  {'TOTAL':<22}  {total:>5}  {len(train_all):>5}  {len(test_all):>5}")

    rng.shuffle(train_all)

    _write_jsonl(train_all, TRAIN_JSONL)
    _write_jsonl(test_all,  TEST_JSONL)
    print(f"\n[split] train.jsonl → {len(train_all)} records  (use 'text' field for SFT)")
    print(f"[split] test.jsonl  → {len(test_all)} records  (held-out for eval)")
    return train_all, test_all


# STEP 2 — MODEL INFERENCE

def load_model(model_type: str = "base"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if model_type == "finetuned":
        model_path = FINETUNED_MODEL_DIR
        label = f"fine-tuned ({model_path})"
    else:
        # Prefer local cache; fall back to HF Hub
        model_path = BASE_MODEL_DIR if Path(BASE_MODEL_DIR).exists() else BASE_MODEL_HF
        label = f"base ({model_path})"

    print(f"\n[model] Loading {label} …")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.eval()
    return model, tokenizer


def _inference_prompt(record: dict) -> str:
    SYSTEM = (
        "You are a precise and reliable legal assistant specialised in the "
        "Income-Tax Act, 2025 (as amended by the Finance Act, 2026). "
        "Answer questions based strictly on the provisions of the Act. "
        "Cite the relevant section number in your response. "
        "Do not speculate beyond what the text of the law states."
    )
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Based on the following legal provisions, answer the question:\n\n"
        f"---\n{record['context']}\n---\n\n{record['instruction']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def run_inference(
    test_records: list[dict],
    model,
    tokenizer,
    out_path: Path = PREDICTIONS_JSONL,
) -> list[dict]:
    """Run BASE model on test set, save predictions. Supports resuming."""
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    predictions: list[dict] = []
    if out_path.exists():
        existing = _load_jsonl(out_path)
        done_ids  = {r["id"] for r in existing}
        predictions.extend(existing)
        print(f"[inference] Resuming — {len(done_ids)} already done")

    pending = [r for r in test_records if r["id"] not in done_ids]
    print(f"[inference] Running {len(pending)} / {len(test_records)} test records …")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, record in enumerate(pending):
            prompt = _inference_prompt(record)
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,      # greedy decoding — deterministic
                    temperature=None,     # must be None when do_sample=False
                    top_p=None,           # override model's saved generation_config
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids    = out_ids[0][inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            pred_record = {
                "id":             record["id"],
                "section_num":    record["section_num"],
                "qa_type":        record["qa_type"],
                "instruction":    record["instruction"],
                "context":        record["context"],
                "gold_response":  record["response"],
                "model_response": prediction,
            }
            fout.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
            predictions.append(pred_record)

            if (i + 1) % 20 == 0 or (i + 1) == len(pending):
                print(f"  {i+1}/{len(pending)} done", end="\r")

    print(f"\n[inference] Saved → {out_path}")
    return predictions


# RAG INFERENCE HELPERS

def _get_rag_collection():
    """Load the ChromaDB collection built by step4_rag.py build."""
    import chromadb
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name=RAG_EMBED_MODEL)
    except Exception:
        ef = None

    client = chromadb.PersistentClient(path=str(RAG_CHROMA_DIR))
    if ef:
        return client.get_collection(RAG_COLLECTION, embedding_function=ef)
    return client.get_collection(RAG_COLLECTION)


def _rag_retrieve(question: str, collection, k: int = RAG_TOP_K) -> list[dict]:
    """Retrieve top-k chunks from ChromaDB for a question."""
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "section_num":   meta.get("section_num", "?"),
            "section_title": meta.get("section_title", ""),
            "text":          doc,
            "distance":      round(dist, 4),
        })
    return chunks


def _rag_prompt(record: dict, chunks: list[dict]) -> str:
    """Build a RAG-augmented prompt from retrieved chunks."""
    SYSTEM = (
        "You are a precise and reliable legal assistant specialised in the "
        "Income-Tax Act, 2025 (as amended by the Finance Act, 2026). "
        "Answer strictly based on the retrieved legal provisions below. "
        "Always cite the relevant section number. "
        "If the retrieved provisions do not contain the answer, state that clearly."
    )
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Provision {i} — Section {chunk['section_num']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Based on the following retrieved legal provisions from the Income-Tax Act, 2025, "
        f"answer the question accurately:\n\n"
        f"--- RETRIEVED PROVISIONS ---\n{context}\n--- END ---\n\n"
        f"Question: {record['instruction']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def run_inference_rag(
    test_records: list[dict],
    model,
    tokenizer,
    out_path: Path = RAG_PREDICTIONS_JSONL,
    k: int = RAG_TOP_K,
) -> list[dict]:

    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[rag-infer] Connecting to ChromaDB …")
    try:
        collection = _get_rag_collection()
        print(f"[rag-infer] Collection loaded — {collection.count()} chunks available")
    except Exception as e:
        raise RuntimeError(
            f"ChromaDB collection not found at {RAG_CHROMA_DIR}.\n"
            f"Run: python src/step4_rag.py build\nOriginal error: {e}"
        )

    done_ids: set[str] = set()
    predictions: list[dict] = []
    if out_path.exists():
        existing  = _load_jsonl(out_path)
        done_ids  = {r["id"] for r in existing}
        predictions.extend(existing)
        print(f"[rag-infer] Resuming — {len(done_ids)} already done")

    pending = [r for r in test_records if r["id"] not in done_ids]
    print(f"[rag-infer] Running {len(pending)} / {len(test_records)} test records …")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, record in enumerate(pending):
            # Retrieve context live from vector DB
            chunks = _rag_retrieve(record["instruction"], collection, k=k)
            rag_context = "\n\n".join(
                f"[Section {c['section_num']}] {c['text']}" for c in chunks
            )

            prompt = _rag_prompt(record, chunks)
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids    = out_ids[0][inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            pred_record = {
                "id":             record["id"],
                "section_num":    record["section_num"],
                "qa_type":        record["qa_type"],
                "instruction":    record["instruction"],
                # Store the RAG-retrieved context (not the gold context)
                "context":        rag_context,
                "retrieved_sections": [c["section_num"] for c in chunks],
                "gold_response":  record["response"],
                "model_response": prediction,
            }
            fout.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
            predictions.append(pred_record)

            if (i + 1) % 10 == 0 or (i + 1) == len(pending):
                print(f"  {i+1}/{len(pending)} done", end="\r")

    print(f"\n[rag-infer] Saved → {out_path}")
    return predictions




# STEP 3 — LLM-AS-JUDGE  (judge_groq.py)
# run_judge_batched() is imported at the top of this file.


# MOCK JUDGE  (no API key — heuristic rules, for pipeline testing)

def _mock_score(pred: dict) -> dict:
    response    = pred.get("model_response", "")
    gold        = pred.get("gold_response", "")
    section_num = pred.get("section_num", "")
    context     = pred.get("context", "")

    # Section citation
    cited = bool(re.search(rf"[Ss]ection\s+{re.escape(section_num)}(?:\s|\(|$)", response))
    section_citation = 2 if cited else 0

    # Word overlap with gold
    def tokens(t): return set(re.findall(r"\b\w{4,}\b", t.lower()))
    overlap = len(tokens(response) & tokens(gold)) / max(len(tokens(gold)), 1)
    factual_accuracy = min(4, math.floor(overlap * 6))

    # Completeness
    completeness = 2 if len(response) >= len(gold) * 0.5 else (1 if len(response) > 100 else 0)

    # Hallucination: numbers in response not in context
    resp_nums    = set(re.findall(r"\b\d{5,}\b|\d+%|₹[\d,]+", response))
    ctx_nums     = set(re.findall(r"\b\d{5,}\b|\d+%|₹[\d,]+", context))
    hallucinated     = bool(resp_nums - ctx_nums)
    no_hallucination = 0 if hallucinated else 2

    total = factual_accuracy + section_citation + completeness + no_hallucination
    return {
        "factual_accuracy":  factual_accuracy,
        "section_citation":  section_citation,
        "completeness":      completeness,
        "no_hallucination":  no_hallucination,
        "total_score":       total,
        "section_cited":     cited,
        "hallucinated":      hallucinated,
        "reasoning":         f"[mock] word_overlap={overlap:.2f} cited={cited} halluc={hallucinated}",
    }


def run_mock_judge(predictions: list[dict], out_path: Path = JUDGE_JSONL) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for pred in predictions:
            scores = _mock_score(pred)
            record = {
                "id":             pred["id"],
                "section_num":    pred["section_num"],
                "qa_type":        pred["qa_type"],
                "instruction":    pred["instruction"],
                "model_response": pred["model_response"],
                "gold_response":  pred["gold_response"],
                **scores,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)
    print(f"[mock_judge] {len(results)} records scored → {out_path}")
    return results


# STEP 4 — AGGREGATE REPORT

def compute_report(results: list[dict]) -> dict:
    valid = [r for r in results if r.get("total_score") is not None]
    if not valid:
        return {"error": "No valid scores"}

    def avg(vals): return round(sum(vals) / len(vals), 3) if vals else 0.0

    overall = {
        "n_evaluated":        len(valid),
        "n_errors":           len(results) - len(valid),
        "avg_total":          avg([r["total_score"] for r in valid]),
        "avg_factual":        avg([r.get("factual_accuracy", 0) for r in valid]),
        "avg_completeness":   avg([r.get("completeness", 0) for r in valid]),
        "avg_no_halluc":      avg([r.get("no_hallucination", 0) for r in valid]),
        "avg_cite":           avg([r.get("section_citation", 0) for r in valid]),
        "section_cite_rate":  avg([1 if r.get("section_cited") else 0 for r in valid]),
        "hallucination_rate": avg([1 if r.get("hallucinated") else 0 for r in valid]),
        "pct_ge_7":           round(100 * sum(1 for r in valid if r["total_score"] >= 7) / len(valid), 1),
        "pct_lt_4":           round(100 * sum(1 for r in valid if r["total_score"] < 4) / len(valid), 1),
    }

    by_type: dict[str, list] = defaultdict(list)
    for r in valid: by_type[r["qa_type"]].append(r)
    by_type_agg = {
        qt: {
            "n":         len(recs),
            "avg_score": avg([r["total_score"] for r in recs]),
            "halluc":    avg([1 if r.get("hallucinated") else 0 for r in recs]),
            "cite":      avg([1 if r.get("section_cited") else 0 for r in recs]),
        }
        for qt, recs in sorted(by_type.items(), key=lambda x: -len(x[1]))
    }

    by_section: dict[str, list] = defaultdict(list)
    for r in valid: by_section[r.get("section_num", "?")].append(r["total_score"])
    sec_avgs = {s: avg(v) for s, v in by_section.items() if len(v) >= 2}

    return {
        "overall":            overall,
        "by_type":            by_type_agg,
        "weakest_sections":   sorted(sec_avgs.items(), key=lambda x:  x[1])[:5],
        "strongest_sections": sorted(sec_avgs.items(), key=lambda x: -x[1])[:5],
    }

# COMPARE — base vs finetuned-rag

def compare_runs(
    base_path: Path = BASE_JUDGE_JSONL,
    rag_path: Path  = RAG_JUDGE_JSONL,
    out_path: Path  = COMPARE_REPORT_TXT,
):
    """Print and save a side-by-side delta table: base vs finetuned+RAG."""
    def load(path, label):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} results not found: {path}\n"
                f"Run: python src/base_eval.py --mode full --model-type "
                f"{'base' if 'base' in str(path) else 'finetuned-rag'}"
            )
        return {r["id"]: r for r in _load_jsonl(path) if r.get("total_score") is not None}

    base = load(base_path, "Base")
    rag  = load(rag_path,  "Finetuned+RAG")
    common = set(base) & set(rag)
    if not common:
        print("[compare] No common IDs — run both evals first.")
        return

    def avg(run, key, ids):
        vals = [run[i].get(key, 0) for i in ids if run[i].get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    lines = []
    def p(s=""): lines.append(s); print(s)

    p("=" * 72)
    p("ITA LEGAL ASSISTANT — BASE  vs  FINE-TUNED + RAG  COMPARISON")
    p(f"Judge: {GROQ_MODEL}    Matched records: {len(common)}")
    p("=" * 72)
    p()
    p(f"  {'Metric':<28}  {'Base':>7}  {'FT+RAG':>7}  {'Delta':>7}  {'FT+RAG wins':>11}")
    p(f"  {'-'*28}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*11}")

    metrics = [
        ("total_score",       "Total score       (0-10)"),
        ("factual_accuracy",  "Factual accuracy  (0-4) "),
        ("completeness",      "Completeness      (0-2) "),
        ("no_hallucination",  "No hallucination  (0-2) "),
        ("section_citation",  "Section citation  (0-2) "),
    ]
    for key, label in metrics:
        b    = avg(base, key, common)
        r    = avg(rag,  key, common)
        d    = r - b
        wins = sum(1 for i in common if rag[i].get(key, 0) > base[i].get(key, 0))
        sign = "+" if d >= 0 else ""
        p(f"  {label:<28}  {b:>7.3f}  {r:>7.3f}  {sign}{d:>6.3f}  {100*wins//len(common):>9}%")

    p()
    # Hallucination rate (lower = better, so show base - rag as improvement)
    b_h = sum(1 for i in common if base[i].get("hallucinated")) / len(common)
    r_h = sum(1 for i in common if rag[i].get("hallucinated"))  / len(common)
    d_h = r_h - b_h
    sign = "+" if d_h >= 0 else ""
    p(f"  {'Hallucination rate':<28}  {b_h:>7.3f}  {r_h:>7.3f}  {sign}{d_h:>6.3f}  {'↓ lower=better':>11}")

    b_c = sum(1 for i in common if base[i].get("section_cited")) / len(common)
    r_c = sum(1 for i in common if rag[i].get("section_cited"))  / len(common)
    d_c = r_c - b_c
    sign = "+" if d_c >= 0 else ""
    p(f"  {'Section citation rate':<28}  {b_c:>7.3f}  {r_c:>7.3f}  {sign}{d_c:>6.3f}  {'↑ higher=better':>11}")

    p()
    p("  Score distribution:")
    for threshold, tag, direction in [(7, "≥7 (good)", "↑"), (4, "<4 (failing)", "↓")]:
        if direction == "↑":
            b_pct = 100 * sum(1 for i in common if base[i]["total_score"] >= threshold) / len(common)
            r_pct = 100 * sum(1 for i in common if rag[i]["total_score"]  >= threshold) / len(common)
        else:
            b_pct = 100 * sum(1 for i in common if base[i]["total_score"] < threshold) / len(common)
            r_pct = 100 * sum(1 for i in common if rag[i]["total_score"]  < threshold) / len(common)
        sign = "+" if (r_pct - b_pct) >= 0 else ""
        p(f"    {tag:<16}  base={b_pct:5.1f}%  ft+rag={r_pct:5.1f}%  delta={sign}{r_pct-b_pct:.1f}%  {direction}")

    p()
    p("=" * 72)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[compare] Report saved → {out_path}")


def print_report(report: dict, out_path: Path = BASE_REPORT_TXT, label: str = "BASE MODEL"):
    lines = []
    def p(s=""): lines.append(s); print(s)

    p("=" * 64)
    p(f"ITA LEGAL ASSISTANT — EVALUATION REPORT  [{label}]")
    p(f"Judge: {GROQ_MODEL}")
    p("=" * 64)

    if "error" in report:
        p(f"ERROR: {report['error']}"); return

    ov = report["overall"]
    p()
    p("── OVERALL ──────────────────────────────────────────────────")
    p(f"  Records evaluated       : {ov['n_evaluated']}  (errors: {ov['n_errors']})")
    p(f"  Avg total score  (0-10) : {ov['avg_total']}")
    p(f"  Avg factual acc  (0-4)  : {ov['avg_factual']}")
    p(f"  Avg completeness (0-2)  : {ov['avg_completeness']}")
    p(f"  Avg no-halluc    (0-2)  : {ov['avg_no_halluc']}")
    p(f"  Avg section cite (0-2)  : {ov['avg_cite']}")
    p(f"  Section citation rate   : {ov['section_cite_rate']*100:.1f}%")
    p(f"  Hallucination rate      : {ov['hallucination_rate']*100:.1f}%")
    p(f"  % score ≥ 7  (good)     : {ov['pct_ge_7']}%")
    p(f"  % score < 4  (failing)  : {ov['pct_lt_4']}%")

    p()
    p("── BY QA TYPE ───────────────────────────────────────────────")
    p(f"  {'type':<22}  {'n':>4}  {'avg':>5}  {'cite%':>6}  {'halluc%':>8}")
    p(f"  {'-'*22}  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*8}")
    for qt, d in report["by_type"].items():
        p(f"  {qt:<22}  {d['n']:>4}  {d['avg_score']:>5.2f}  "
          f"{d['cite']*100:>5.1f}%  {d['halluc']*100:>7.1f}%")

    p()
    p("── WEAKEST SECTIONS  (min 2 test records) ───────────────────")
    for sec, score in report["weakest_sections"]:
        p(f"  Section {sec:<8}  avg: {score:.2f}")
    p()
    p("── STRONGEST SECTIONS ───────────────────────────────────────")
    for sec, score in report["strongest_sections"]:
        p(f"  Section {sec:<8}  avg: {score:.2f}")
    p()
    p("=" * 64)
    p()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[report] Saved → {out_path}")


# CLI

def main():
    create_dirs()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--mode", default="split",
        choices=["split", "infer", "judge", "mock", "report", "compare", "full"],
        help=(
            "split         — create train/test split only\n"
            "infer         — run model inference on test set\n"
            "judge         — score existing predictions with Groq llama3-70b-8192\n"
            "mock          — score with heuristic judge (no API key)\n"
            "report        — reprint report from existing judge results\n"
            "compare       — base vs finetuned-rag delta table\n"
            "full          — split + infer + judge (end-to-end)"
        ),
    )
    parser.add_argument(
        "--model-type",
        default="base",
        choices=["base", "finetuned", "finetuned-rag"],
        help=(
            "base           — un-fine-tuned LLaMA (models/llama_3_2_1b)\n"
            "finetuned      — fine-tuned model, gold context (finetune/final_model)\n"
            "finetuned-rag  — fine-tuned model, RAG context from ChromaDB  ← recommended post-FT eval"
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")

    #  Resolve paths and labels based on --model-type 
    if args.model_type == "finetuned-rag":
        pred_path    = RAG_PREDICTIONS_JSONL
        judge_path   = RAG_JUDGE_JSONL
        report_path  = RAG_REPORT_TXT
        report_label = "FINE-TUNED + RAG"
        mock_label   = "MOCK JUDGE (finetuned-rag)"
    elif args.model_type == "finetuned":
        pred_path    = AFTER_PREDICTIONS_JSONL
        judge_path   = AFTER_JUDGE_JSONL
        report_path  = AFTER_REPORT_TXT
        report_label = "FINE-TUNED MODEL"
        mock_label   = "MOCK JUDGE (finetuned)"
    else:
        pred_path    = BASE_PREDICTIONS_JSONL
        judge_path   = BASE_JUDGE_JSONL
        report_path  = BASE_REPORT_TXT
        report_label = "BASE MODEL"
        mock_label   = "MOCK JUDGE (base)"

    if args.mode != "compare":
        print(f"[eval] model-type={args.model_type}  mode={args.mode}")
        print(f"[eval] predictions → {pred_path}")
        print(f"[eval] judge out   → {judge_path}")
        print(f"[eval] report      → {report_path}")

    #  split 
    if args.mode in ("split", "full"):
        train, test = make_split()

    #  infer 
    if args.mode in ("infer", "full"):
        if args.mode == "infer":
            test = _load_jsonl(TEST_JSONL)

        model_key = "finetuned" if args.model_type == "finetuned-rag" else args.model_type
        model, tok = load_model(model_key)

        if args.model_type == "finetuned-rag":
            predictions = run_inference_rag(test, model, tok,
                                            out_path=pred_path, k=RAG_TOP_K)
        else:
            predictions = run_inference(test, model, tok, out_path=pred_path)
        del model

    #  judge 
    if args.mode in ("judge", "full"):
        if args.mode == "judge":
            predictions = _load_jsonl(pred_path)
        if not api_key:
            print("[judge] No GROQ_API_KEY found — falling back to mock judge")
            results = run_mock_judge(predictions, out_path=judge_path)
        else:
            results = run_judge_batched(
                predictions,
                api_key,
                out_path=judge_path,
                batch_size=10,
            )
        report = compute_report(results)
        print_report(report, out_path=report_path, label=report_label)

    #  mock 
    if args.mode == "mock":
        predictions = _load_jsonl(pred_path)
        results     = run_mock_judge(predictions, out_path=judge_path)
        print_report(compute_report(results), out_path=report_path, label=mock_label)

    #  report 
    if args.mode == "report":
        results = _load_jsonl(judge_path)
        print_report(compute_report(results), out_path=report_path, label=report_label)

    #  compare 
    if args.mode == "compare":
        compare_runs(
            base_path=BASE_JUDGE_JSONL,
            rag_path=RAG_JUDGE_JSONL,
            out_path=COMPARE_REPORT_TXT,
        )

if __name__ == "__main__":
    main()