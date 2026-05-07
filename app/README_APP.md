# LexiTune — Streamlit App


```
lexitune/
├── app.py                  ← Streamlit entry point
├── app/
│   ├── __init__.py
│   ├── model_loader.py     ← cached model/DB loading
│   ├── inference.py        ← pure inference helpers
│   ├── history.py          ← session + file query log
│   ├── tab_base.py         ← Base Model tab
│   ├── tab_finetuned.py    ← Fine-tuned tab
│   ├── tab_rag.py          ← Fine-tuned + RAG tab
│   └── tab_compare.py      ← Compare tab
└── logs/                   ← auto-created; stores query_history.jsonl
```

---

## Prerequisites

### Make sure your pipeline outputs exist

| What | Where | How to create |
|------|-------|---------------|
| Base model | `models/llama_3_2_1b/` | Already downloaded |
| Fine-tuned model | `finetune/final_model/` | Run `src/finetune.py` |
| RAG index | `rag/chroma_db/` | `python src/step4_rag.py build` |

The app will show a clear error message in the UI if any of these are missing —
it will not crash on startup.

---

## Running the app

Always run from the **project root** (`lexitune/`), never from inside `app/`:

```bash
cd E:\lexitune          # Windows
streamlit run app.py
```

```bash
cd ~/lexitune           # Linux / Mac
streamlit run app.py
```

The browser opens at **http://localhost:8501** automatically.

---

## Tabs

| Tab | Model | Context | Use for |
|-----|-------|---------|---------|
|  Base Model | LLaMA 3.2-1B (raw) | None | Pre-training baseline |
|  Fine-tuned | `finetune/final_model` | None | Isolate SFT effect |
|  Fine-tuned + RAG | `finetune/final_model` | ChromaDB live retrieval | **Best answers** |
|  Compare | All of the above | — | Side-by-side diff |
|  History | — | — | Session query log |

---

## Sidebar controls

- **Max new tokens** — controls response length (128–1024)
- **RAG top-k chunks** — how many chunks to retrieve per question (2–10)
- **Compute** — shows GPU name + VRAM if CUDA is available
- **Recent Queries** — last 10 queries with expandable responses

---

## Model loading behaviour

Models are loaded **once per server process** using `@st.cache_resource`.
Switching tabs does **not** reload weights.  Only restarting the server does.

Expected first-load times on a GPU machine:
- Base model:      ~20–40 s
- Fine-tuned:      ~20–40 s
- ChromaDB:        ~2–5 s

---

## Query history

Every query is logged to `logs/query_history.jsonl` (auto-created).
You can also download the session history as JSONL from the History tab.
