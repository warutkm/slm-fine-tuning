## 1. Local Model Directory & Hugging Face Access (One-Time Setup)

This section prepares the local filesystem and enables authenticated access to gated models.

### 1.1 Create Local Model Directories (Run Once)

```bash
mkdir models
mkdir models\llama_3_2_1b
```

These directories store **read-only base models** downloaded from Hugging Face.

---

### 1.2 Meta LLaMA License Acceptance (Mandatory)

Before downloading LLaMA models, ensure you have completed this **once in the browser**:

1. Open the **LLaMA-3.2-1B-Instruct** model page on Hugging Face
2. Click **“Agree and Access”**
3. Accept the Meta license

Without this step, model downloads will fail even if authentication succeeds.

--- 

### 1.3 Hugging Face Authentication

* **Token type:** Read
* **Token source:** [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
* Copy token 

This step is required only once per environment.

---
### 1.4 Run the following command to download the model:

```bash
python scripts/download_llama_3_2_1b.py
```
* Paste copied token→ Press Enter → Done