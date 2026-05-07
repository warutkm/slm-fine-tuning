"""
Retrieval-Augmented Generation (RAG) Pipeline
Builds and queries a vector database of legal text chunks.

Components:
  Chunker   – splits structured sections into 300-800 token chunks with overlap
  Embedder  – sentence-transformers (BAAI/bge-small-en-v1.5; tiny, fast, CPU-ok)
  VectorDB  – ChromaDB (local, persistent, no server needed)
  Retriever – top-k MMR retrieval

Run once to build index:
    python rag.py build

Then use as a library:
    from rag import retrieve
    chunks = retrieve("What is the deduction limit under section 80C?", k=4)
"""

import json
import sys
import re
import hashlib
from pathlib import Path
from typing import Optional

ROOT_DIR     = Path(__file__).resolve().parent.parent   
PROCESSED    = ROOT_DIR / "data" / "processed"
RAG_DIR      = ROOT_DIR / "rag"

FLAT_JSONL   = PROCESSED / "flat_sections.jsonl"
CHUNKS_JSONL = RAG_DIR / "chunks.jsonl"
CHROMA_DIR   = RAG_DIR / "chroma_db"

MODELS_DIR   = ROOT_DIR / "models"
FINETUNE_DIR = ROOT_DIR / "finetune"

BASE_MODEL_DIR      = MODELS_DIR / "llama_3_2_1b"
BASE_MODEL_HF       = "meta-llama/Llama-3.2-1B-Instruct"   
FINETUNED_MODEL_DIR = FINETUNE_DIR / "final_model"

# Chunking
CHUNK_MAX_TOKENS   = 800
CHUNK_MIN_TOKENS   = 100
CHUNK_OVERLAP_TOKS = 80    

# Retrieval
TOP_K          = 6
EMBED_MODEL    = "BAAI/bge-base-en-v1.5"   
COLLECTION     = "ita_2025"


#  Token approximation 
def tok(text: str) -> int:
    return len(text) // 4

def hard_split(text: str, max_tokens: int) -> list[str]:

    # Split text respecting sentence boundaries.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buffer = [], ""

    for sent in sentences:
        candidate = (buffer + " " + sent).strip()
        if tok(candidate) > max_tokens and buffer:
            chunks.append(buffer.strip())
            buffer = sent
        else:
            buffer = candidate

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks

#  Chunker 
def chunk_section(sec: dict) -> list[dict]:

    snum   = sec.get("section_num", "?")
    title  = sec.get("section_title", "").strip() or f"Section {snum}"
    chap   = sec.get("chapter_num", "")
    chap_t = sec.get("chapter_title", "")
    text   = sec.get("full_text", "").strip()

    if tok(text) <= CHUNK_MAX_TOKENS:
        # Section fits in one chunk
        return [_make_chunk(snum, title, chap, chap_t, text, 0)]

    chunks = []
    # Try to split on subsection boundaries first
    subsec_parts = re.split(r"(?=\n\s*\(\d+\)\s)", text)
    if len(subsec_parts) > 1:
        parts = subsec_parts
    else:
        # Fall back to paragraph breaks
        parts = re.split(r"\n{2,}", text)

    buffer = ""
    chunk_idx = 0
    prev_tail = ""   # overlap: last CHUNK_OVERLAP_TOKS tokens of prev chunk

    for part in parts:
        candidate = (prev_tail + "\n" + buffer + "\n" + part).strip() if buffer else part.strip()
        if tok(candidate) > CHUNK_MAX_TOKENS and buffer:
            # Flush buffer
            if tok(buffer) >= CHUNK_MIN_TOKENS:
                chunks.append(_make_chunk(snum, title, chap, chap_t, buffer.strip(), chunk_idx))
                chunk_idx += 1
            # Save tail for overlap
            prev_tail = _tail(buffer, CHUNK_OVERLAP_TOKS)
            buffer = prev_tail + "\n" + part.strip()
        else:
            buffer = candidate

    if buffer.strip() and tok(buffer) >= CHUNK_MIN_TOKENS:
        chunks.append(_make_chunk(snum, title, chap, chap_t, buffer.strip(), chunk_idx))

    final_chunks = []

    for c in (chunks if chunks else [_make_chunk(snum, title, chap, chap_t, text, 0)]):
        if tok(c["text"]) > CHUNK_MAX_TOKENS:
        # Force split oversized chunk
            sub_parts = hard_split(c["text"], CHUNK_MAX_TOKENS)
            for j, part in enumerate(sub_parts):
                final_chunks.append(
                    _make_chunk(
                    snum,
                    title,
                    chap,
                    chap_t,
                    part,
                    j
                    )
                )
        else:
            final_chunks.append(c)

    return final_chunks

def _tail(text: str, n_tokens: int) -> str:
    chars = n_tokens * 4
    tail  = text[-chars:]
    # Start at sentence boundary
    m = re.search(r"[.!?]\s", tail)
    return tail[m.end():] if m else tail


def _make_chunk(snum, title, chap, chap_t, text, idx) -> dict:
    # Prepend metadata prefix for retrieval quality
    prefix = f"Section {snum} — {title} | Chapter {chap}: {chap_t}\n\n"
    full   = prefix + text

    chunk_id = hashlib.md5(full.encode()).hexdigest()  # use hash as ID

    return {
        "chunk_id":      chunk_id,                           
        "section_num":   snum,
        "section_title": title,
        "chapter_num":   chap,
        "chapter_title": chap_t,
        "text":          full,
        "approx_tokens": tok(full),
    }


#  Build chunks from all sections 
def build_chunks() -> list[dict]:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    print("[rag] Loading sections …")
    sections = []
    with open(FLAT_JSONL, encoding="utf-8") as f:
        for line in f:
            sections.append(json.loads(line))

    all_chunks = []
    for sec in sections:
        if len(sec.get("full_text", "")) < 50:
            continue
        all_chunks.extend(chunk_section(sec))

    print(f"[rag] {len(all_chunks)} chunks created from {len(sections)} sections")
    token_sizes = [c["approx_tokens"] for c in all_chunks]
    print(f"[rag] Token distribution — min:{min(token_sizes)} "
          f"avg:{sum(token_sizes)//len(token_sizes)} max:{max(token_sizes)}")

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[rag] Chunks written → {CHUNKS_JSONL}")
    return all_chunks


#  Embedder / Vector DB 
def get_embedding_fn():
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    except Exception:
        # Fallback: use chromadb's default
        return None


def build_index(chunks: list[dict]):
    import chromadb
    print(f"[rag] Building ChromaDB index at {CHROMA_DIR} …")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    ef = get_embedding_fn()
    if ef:
        collection = client.create_collection(COLLECTION, embedding_function=ef)
    else:
        collection = client.create_collection(COLLECTION)

    BATCH = 100
    total = len(chunks)
    for i in range(0, total, BATCH):
        batch = chunks[i:i + BATCH]
        collection.add(
            ids       = [c["chunk_id"] for c in batch],
            documents = [c["text"] for c in batch],
            metadatas = [
                {
                    "section_num":   c["section_num"],
                    "section_title": c["section_title"],
                    "chapter_num":   c["chapter_num"],
                }
                for c in batch
            ],
        )
        print(f"[rag] Indexed {min(i + BATCH, total)}/{total} chunks …", end="\r")

    print(f"\n[rag] Index built. Collection '{COLLECTION}' has {collection.count()} docs.")
    return collection


#  Retrieval 
def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = get_embedding_fn()
    if ef:
        return client.get_collection(COLLECTION, embedding_function=ef)
    return client.get_collection(COLLECTION)


def retrieve(query: str, k: int = TOP_K, collection=None) -> list[dict]:

    if collection is None:
        collection = get_collection()
    results = collection.query(
        query_texts=[query],
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
            "section_num":   meta.get("section_num"),
            "section_title": meta.get("section_title"),
            "chapter_num":   meta.get("chapter_num"),
            "text":          doc,
            "distance":      round(dist, 4),
        })
    return chunks


#  Prompt builder 
SYSTEM_PROMPT = (
    "You are a precise and reliable legal assistant specialised in the "
    "Income-Tax Act, 2025 (as amended by the Finance Act, 2026). "
    "Answer strictly based on the retrieved legal provisions below. "
    "Always cite the relevant section number. "
    "If the retrieved provisions do not contain the answer, state that clearly."
)

def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Provision {i} — Section {chunk['section_num']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Based on the following retrieved legal provisions from the Income-Tax Act, 2025, "
        f"answer the question accurately:\n\n"
        f"--- RETRIEVED PROVISIONS ---\n{context}\n--- END OF PROVISIONS ---\n\n"
        f"Question: {query}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


#  Inference with fine-tuned model 
def load_model(model_type: str = "finetuned"):
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if model_type == "finetuned":
        model_path = str(FINETUNED_MODEL_DIR)
        label = f"fine-tuned ({model_path})"

        print(f"[rag] Loading {label} …")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    else:
        model_path = str(BASE_MODEL_DIR) if BASE_MODEL_DIR.exists() else BASE_MODEL_HF
        label = f"base ({model_path})"

        print(f"[rag] Loading {label} …")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    model.eval()
    return model, tokenizer


def answer(
    query: str,
    model=None,
    tokenizer=None,
    collection=None,
    k: int = TOP_K,
    max_new_tokens: int = 512,
) -> dict:

    # 1. Retrieve
    chunks = retrieve(query, k=k, collection=collection)

    # 2. Build prompt
    prompt = build_rag_prompt(query, chunks)

    # 3. Generate (only if model provided)
    response_text = "[Model not loaded — prompt built successfully]"
    if model is not None and tokenizer is not None:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,      
                temperature=None,     
                top_p=None,           
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return {
        "query":            query,
        "retrieved_chunks": chunks,
        "prompt":           prompt,
        "response":         response_text,
    }

#  CLI 
def run():
    import argparse

    parser = argparse.ArgumentParser(
        prog="rag.py",
        description="RAG pipeline for the Income-Tax Act, 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Build the vector index (run once):\n"
            "  python src/rag.py build\n"
            "\n"
            "  # Retrieve chunks only (no model needed):\n"
            '  python src/rag.py query "What is the deduction under 80C?"\n'
            "\n"
            "  # Retrieve + generate answer with fine-tuned model (default):\n"
            '  python src/rag.py answer "What is the deduction under 80C?"\n'
            "\n"
            "  # Use base model instead:\n"
            '  python src/rag.py answer "What is TDS on salary?" --model-type base\n'
            "\n"
            "  # Show index statistics:\n"
            "  python src/rag.py stats"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # build
    sub.add_parser("build", help="Chunk flat_sections.jsonl and build ChromaDB index")

    # query — retrieve chunks only, no model needed
    p_query = sub.add_parser("query", help="Retrieve top-k chunks for a question (no model)")
    p_query.add_argument("question", help="Natural language question")
    p_query.add_argument("--k", type=int, default=TOP_K,
                         help=f"Number of chunks to retrieve (default: {TOP_K})")
    p_query.add_argument("--full", action="store_true",
                         help="Print full chunk text instead of first 600 chars")

    # answer — retrieve + generate
    p_ans = sub.add_parser("answer", help="Retrieve chunks and generate an answer with the model")
    p_ans.add_argument("question", help="Natural language question")
    p_ans.add_argument("--k", type=int, default=TOP_K,
                       help=f"Number of chunks to retrieve (default: {TOP_K})")
    p_ans.add_argument(
        "--model-type", default="finetuned", choices=["base", "finetuned"],
        help="base = models/llama_3_2_1b  |  finetuned = finetune/final_model  (default: finetuned)",
    )
    p_ans.add_argument("--max-tokens", type=int, default=512,
                       help="Max new tokens to generate (default: 512)")

    # stats
    sub.add_parser("stats", help="Show ChromaDB collection statistics")

    args = parser.parse_args()

    #  build 
    if args.command == "build":
        chunks = build_chunks()
        build_index(chunks)
        print("[rag] Index build complete.")

    #  query 
    elif args.command == "query":
        print(f"\n[rag] Retrieving top-{args.k} chunks for: {args.question!r}\n")
        chunks = retrieve(args.question, k=args.k)
        for i, c in enumerate(chunks, 1):
            print(f"── Chunk {i}  Section {c['section_num']} — {c['section_title']}  (dist={c['distance']}) ──")
            body = c["text"] if args.full else (c["text"][:600] + (" …" if len(c["text"]) > 600 else ""))
            print(body)
            print()

    #  answer 
    elif args.command == "answer":
        print(f"\n[rag] Loading model ({args.model_type}) …")
        model, tokenizer = load_model(args.model_type)
        col = get_collection()
        print(f"[rag] Answering: {args.question!r}\n")
        result = answer(
            args.question,
            model=model,
            tokenizer=tokenizer,
            collection=col,
            k=args.k,
            max_new_tokens=args.max_tokens,
        )
        print("── Retrieved Sections ──────────────────────────────────────────")
        for i, c in enumerate(result["retrieved_chunks"], 1):
            print(f"  {i}. Section {c['section_num']} — {c['section_title']}  (dist={c['distance']})")
        print()
        print("── Answer ──────────────────────────────────────────────────────")
        print(result["response"])
        print()

    #  stats 
    elif args.command == "stats":
        col = get_collection()
        print(f"[rag] ChromaDB at: {CHROMA_DIR}")
        print(f"[rag] Collection '{COLLECTION}' has {col.count()} chunks")

    else:
        parser.print_help()

if __name__ == "__main__":
    run()