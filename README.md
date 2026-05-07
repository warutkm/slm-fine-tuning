# 1. lexitune

### Fine-Tuning Small Language Models (SLMs) for Indian Income Tax Compliance 

---

## 2. Project Overview

**lexitune** is a domain-specific AI assistant designed to answer **Indian Income Tax (IT)** questions with high factual accuracy and minimal hallucination.

---

## 3. Project Directory Structure

```
lexitune/
│
├── data/
│   ├── input/
│   │   └── it_2025.pdf
│   │
│   └── processed/
│       ├── ita_raw.txt
│       ├── structured_act.json
│       ├── flat_sections.jsonl
│       ├── qa_dataset.jsonl
│       ├── train.jsonl
│       ├── test.jsonl
│       └── dataset_stats.json
│
├── models/
│   ├── llama_3_2_1b/
│   └── phi_3_5_mini/
│
├── finetune/
│   ├── cpt_adapter/
│   ├── sft_adapter/
│   ├── final_model/
│   └── training_log.json
│
├── rag/
│   ├── chroma_db/
│   └── chunks.jsonl
│
├── scripts/
│   ├── download_llama_3_2_1b.py
│   └── README.md  
│   
├── eval/
│   ├── base_model_predictions.jsonl
│   ├── base_judge_results.jsonl
│   ├── base_eval_report.txt
│   ├── after_model_predictions.jsonl
│   ├── after_judge_results.jsonl
│   ├── after_eval_report.txt
│   ├── rag_model_predictions.jsonl
│   ├── rag_judge_results.jsonl
│   ├── rag_eval_report.txt
│   └── compare_report.txt
│
├── src/
│   ├── parse.py
│   ├── qa_gen.py
│   ├── finetune.py
│   ├── rag.py
│   ├── judge_groq.py
│   └── base_eval.py
│
├── demo/
│   └── ask_slm.py
│
├── app/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── inference.py
│   ├── history.py
│   ├── tab_base.py
│   ├── tab_finetuned.py
│   ├── tab_rag.py
│   ├── README_APP.MD
│   └── tab_compare.py
│
├── logs/
├── app.py
├── config.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Environment Setup (Python 3.10)

This project **strictly requires Python 3.10**. Please ensure Python 3.10 is installed before proceeding.

### 4.1 Create Virtual Environment (venv)

```bash
# Windows
py -3.10 -m venv venv

# Linux / macOS
python3.10 -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

---

### 4.2 Install Core Libraries

```bash
pip install -r requirements.txt
```

> PyTorch is installed separately to allow flexibility between CPU and GPU environments,
> while all other dependencies are pinned in `requirements.txt` for reproducibility.

---

### 4.3 Install CUDA-Enabled PyTorch

If an NVIDIA GPU with CUDA 12.1 support is available:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only systems:

```bash
pip install torch
```

---

## 5. Download Base Models (Local & Read-Only)

Refer to README.md inside the scripts folder for detailed instructions on installing LLaMA 3.2 locally.

---

## 6. Basic Question–Answer Setup (Single SLM Interaction)

```bash
python demo/ask_slm.py --config config.json
```

You will be prompted to enter a question, for example:

```
What is the standard deduction under the new tax regime?
```

The model response is printed directly to the console.

---

## 7. Data Parsing & Dataset Preparation

Before generating question–answer datasets, the raw Income Tax document must be parsed into structured formats.

### 7.1 Install Poppler (Required for PDF Parsing)

```bash
conda install -c conda-forge poppler
```

> This is required for PDF processing tools used in parsing.

---

### 7.2 Run Parsing Script

```bash
python src/parse.py --config config.json
```

---

### 7.3 Input

* `data/input/it_2025.pdf` → Raw Income Tax Act document

---

### 7.4 Outputs Generated

The parsing pipeline produces:

* `data/processed/ita_raw.txt`
  → Cleaned raw text extracted from the PDF

* `data/processed/structured_act.json`
  → Hierarchical structured representation of the Act

* `data/processed/flat_sections.jsonl`
  → Flattened sections (used for QA generation)

---

## 8. QA Dataset Generation

This step converts structured legal sections into a supervised fine-tuning (SFT) dataset.

It takes the parsed output (`flat_sections.jsonl`) and generates multiple types of question–answer pairs per section, such as:

* explanation
* definition
* eligibility
* scenario-based reasoning
* conditions and exceptions
* procedural queries
* multi-hop (cross-section reasoning)

Each record is formatted in **instruction-tuning format** compatible with LLaMA-style models.

### Input

* `data/processed/flat_sections.jsonl`
  → Output from `parse.py` containing section-wise structured data.

---

### Output

* `data/processed/qa_dataset.jsonl` → Final SFT dataset
* `data/processed/dataset_stats.json` → Summary statistics

---

From the project root:

```bash
python src/qa_gen.py --config config.json
```

---

## 9. Fine-Tuning (QLoRA + Optional CPT)

The fine-tuning pipeline (`src/finetune.py`) supports two stages:

* **CPT** (Continued Pre-Training) — adapts the model's language prior to ITA vocabulary using raw text
* **SFT** (Supervised Fine-Tuning) — instruction-tunes the model on the generated QA dataset

### 9.1 Modes

| Mode | Description |
|------|-------------|
| `sft` | SFT only (default) |
| `cpt` | CPT only |
| `both` | CPT then SFT (recommended) |

### 9.2 Run Fine-Tuning

From the project root:

```bash
# SFT only (default)
python src/finetune.py

# CPT only
python src/finetune.py --mode cpt

# CPT then SFT (recommended)
python src/finetune.py --mode both

# Resume SFT from a checkpoint
python src/finetune.py --mode sft --resume finetune/sft_adapter/checkpoint-400
```

### 9.3 Outputs Generated

* `finetune/cpt_adapter/` → Saved CPT LoRA adapter weights
* `finetune/sft_adapter/` → Saved SFT LoRA adapter weights
* `finetune/final_model/` → Merged final model (no PEFT dependency at inference)
* `finetune/training_log.json` → Per-stage loss history and VRAM snapshots

---

## 10. RAG Setup (Retrieval-Augmented Generation)

RAG grounds model responses in retrieved ITA section text at inference time, significantly reducing hallucination without requiring a larger model.

### 10.1 Build the Vector Store

```bash
python src/rag.py build
```

### 10.2 Input

* `data/processed/flat_sections.jsonl` → Section-wise parsed ITA content

### 10.3 Outputs Generated

* `rag/chroma_db/` → Persistent ChromaDB vector store
* `rag/chunks.jsonl` → Text chunks used for indexing

---

## 11. Evaluation

The evaluation pipeline runs model predictions through an LLM judge (Llama-3.3-70B) and scores each response on factual accuracy, completeness, hallucination, and section citation.

Scores are on a 0–10 scale across 7 QA types: `explain_section`, `multi_hop`, `scenario`, `condition`, `procedural`, `eligibility`, and `exception`.

### 11.1 Model Configurations
 
| Model Type | Description |
|---|---|
| `base` | Un-fine-tuned LLaMA (context from test record) |
| `finetuned` | Fine-tuned model (context from test record) |
| `finetuned-rag` | Fine-tuned model (context retrieved live from ChromaDB) |
 
### 11.2 Pipeline Modes
 
| Mode | Description |
|---|---|
| `split` | Create stratified 85/15 train/test split — **run once before training** |
| `infer` | Run model inference → predictions JSONL |
| `judge` | Score existing predictions with Groq LLM judge |
| `mock` | Score with heuristic judge (no API key required, for testing) |
| `report` | Reprint a previously saved eval report |
| `compare` | Side-by-side delta table: base vs finetuned-rag |
| `full` | `split + infer + judge` end-to-end |
 
### 11.3 Prerequisites
 
**Set your Groq API key** before running any judge step:
 
```bash
set GROQ_API_KEY=gsk_...
```
 
**Build the ChromaDB index** before using `finetuned-rag`:
 
```bash
python src/rag.py build
```
 
> Run `--mode split` exactly **once** before fine-tuning begins. Never re-run it — it would alter the test set the fine-tuned model was never trained on.
 
### 11.4 Run Evaluation
 
```bash
# Step 1 — Create train/test split (once, before fine-tuning)
python src/base_eval.py --mode split
 
# Step 2 — Evaluate base model end-to-end
python src/base_eval.py --mode full --model-type base
 
# Step 3 — Evaluate fine-tuned + RAG (recommended post-training eval)
python src/base_eval.py --mode full --model-type finetuned-rag
 
# Step 4 — Compare base vs fine-tuned+RAG side-by-side
python src/base_eval.py --mode compare
 
# Evaluate plain fine-tuned (no RAG) for a three-way comparison
python src/base_eval.py --mode full --model-type finetuned
 
# Inference only (skip judge)
python src/base_eval.py --mode infer --model-type finetuned-rag
 
# Re-judge existing predictions
python src/base_eval.py --mode judge --model-type finetuned-rag
 
# Mock judge (no API key, for local testing)
python src/base_eval.py --mode mock --model-type finetuned-rag
```


### 11.2 Outputs Generated

Each evaluation run produces the following files under `eval/`:

* `*_model_predictions.jsonl` → Raw model responses for each test record
* `*_judge_results.jsonl` → Per-record judge scores and reasoning
* `*_eval_report.txt` → Aggregated report with overall and per-type metrics
* `compare_report.txt` → Side-by-side comparison across all three modes

---

## 12. Streamlit App

The interactive Streamlit app lets you query and compare all three model configurations through a tabbed interface.

### 12.1 Structure

| File | Purpose |
|------|---------|
| `app/model_loader.py` | Loads base, fine-tuned, and RAG models |
| `app/inference.py` | Handles prompt construction and generation |
| `app/history.py` | Manages conversation history per tab |
| `app/tab_base.py` | Tab UI for the base model |
| `app/tab_finetuned.py` | Tab UI for the fine-tuned model |
| `app/tab_rag.py` | Tab UI for the RAG pipeline |
| `app/tab_compare.py` | Side-by-side comparison of all three |

### 12.2 Launch the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501` by default.

> See `README_APP.MD` for detailed instructions on app configuration and usage.

---

## 13. Configuration

All pipeline components (model paths, data paths, generation parameters) are controlled through `config.json` in the project root. 