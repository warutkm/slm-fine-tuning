"""
app/model_loader.py

Centralised model loading for the Streamlit app.

All heavy objects (model weights, tokenizer, ChromaDB collection) are loaded
once per server process via @st.cache_resource and reused across every user
interaction and tab.  Streamlit never reloads them on re-runs.

GPU handling

  CUDA available  → 4-bit NF4 quantisation, device_map="auto"
  CPU only        → 4-bit still works via bitsandbytes (slower)
"""

from __future__ import annotations
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATHS = {
    "base":      ROOT / "models"   / "llama_3_2_1b",
    "finetuned": ROOT / "finetune" / "final_model",
}
BASE_MODEL_HF   = "meta-llama/Llama-3.2-1B-Instruct"
CHROMA_DIR      = ROOT / "rag" / "chroma_db"
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "ita_2025"


def _bnb_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def _load_hf_model(path_str: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path_str)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path_str,
        quantization_config=_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


@st.cache_resource(show_spinner="Loading base model — one-time, please wait…")
def get_base_model():
    p = MODEL_PATHS["base"]
    return _load_hf_model(str(p) if p.exists() else BASE_MODEL_HF)


@st.cache_resource(show_spinner="Loading fine-tuned model — one-time, please wait…")
def get_finetuned_model():
    p = MODEL_PATHS["finetuned"]
    if not p.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {p}. "
            "Run the fine-tuning pipeline first."
        )
    return _load_hf_model(str(p))


@st.cache_resource(show_spinner="Connecting to ChromaDB…")
def get_rag_collection():
    import chromadb
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    except Exception:
        ef = None
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB index not found at {CHROMA_DIR}. "
            "Run:  python src/step4_rag.py build"
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return (
        client.get_collection(COLLECTION_NAME, embedding_function=ef)
        if ef else
        client.get_collection(COLLECTION_NAME)
    )


def device_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            return f" GPU  {name} ({mem} GB VRAM)"
        return " CPU only — inference will be slow"
    except Exception:
        return " Device info unavailable"
