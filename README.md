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
│
├── models/
│   └──llama_3_2_1b/
│   
├── scripts/
│   └──download_llama_3_2_1b.py
│   └──README.md
│
│── eval/
│   └── eval_score.py
│   └── config.json
│   └── golden_test_set.jsonl
│   └── run_evaluation.py
│   └── eval_output/
│
├── demo/
│   └── ask_slm.py
│
├── requirements.txt
│   
└── README.md
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
python demo/ask_slm.py
```

You will be prompted to enter a question, for example:

```
What is the standard deduction under the new tax regime?
```

The model response is printed directly to the console.

---

## 7. Golden Test Set

* File: `eval/golden_test_set.jsonl`
* **75 manually curated Indian Income Tax questions**
* Covers:

  * Deductions
  * Capital gains
  * Regime comparisons
  * Edge cases

**Strict rule**

* Never used for training
* Used only for evaluation

This ensures unbiased benchmarking.

---

## 8. Running Baseline Evaluation

```bash
python eval/run_evaluation.py --config eval/config.json --model llama
```

Runs the baseline evaluation for the specified model using the dataset defined in `config.json`.
Generates responses, evaluates them using the 0–4 scoring rubric, and computes aggregate metrics.
Saves detailed results (JSONL/CSV) and a summary report in the configured output directory.

---