"""
Stages

  1. CPT  — Continued Pre-Training on raw domain text
             Input : data/processed/ita_raw.txt
             Output: finetune/cpt_adapter/

  2. SFT  — Supervised Fine-Tuning (instruction tuning)
             Input : data/processed/train.jsonl
                     data/processed/test.jsonl
             Output: finetune/sft_adapter/
                     finetune/final_model/   (merged, ready for inference)

Usage

    python finetune.py              # SFT only (default)
    python finetune.py --mode cpt  # CPT only
    python finetune.py --mode both # CPT then SFT (recommended)
    python finetune.py --mode sft --resume finetune/sft_adapter/checkpoint-400
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lexitune")


# CONFIG  
class CFG:
    # Paths
    BASE_MODEL       = "meta-llama/Llama-3.2-1B-Instruct"
    CPT_RAW_TEXT     = "data/processed/ita_raw.txt"
    SFT_TRAIN_DATA   = "data/processed/train.jsonl"
    SFT_TEST_DATA    = "data/processed/test.jsonl"
    CPT_OUTPUT_DIR   = "finetune/cpt_adapter"
    SFT_OUTPUT_DIR   = "finetune/sft_adapter"
    FINAL_MODEL_DIR  = "finetune/final_model"
    LOG_FILE         = "finetune/training_log.json"

    # LoRA (shared by CPT and SFT)
    LORA_R           = 16
    LORA_ALPHA       = 32
    LORA_DROPOUT     = 0.05
    LORA_TARGETS     = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # CPT hypers
    CPT_EPOCHS       = 1
    CPT_BATCH_SIZE   = 1
    CPT_GRAD_ACCUM   = 8      
    CPT_MAX_SEQ_LEN  = 512      
    CPT_LR           = 5e-5
    CPT_PACKING      = True
    CPT_SAVE_STEPS   = 100
    CPT_EVAL_STEPS   = 100

    # SFT hypers
    SFT_EPOCHS       = 3
    SFT_BATCH_SIZE   = 1
    SFT_GRAD_ACCUM   = 16       
    SFT_MAX_SEQ_LEN  = 1024     
    SFT_LR           = 1e-4     
    SFT_PACKING      = False
    SFT_SAVE_STEPS   = 200
    SFT_EVAL_STEPS   = 200
    SFT_EARLY_STOP   = 2       

    # Shared
    LR_SCHEDULER     = "cosine"
    WARMUP_RATIO     = 0.05
    WEIGHT_DECAY     = 0.01
    MAX_GRAD_NORM    = 0.3      
    SEED             = 42

    # Precision — set at runtime by check_environment()
    COMPUTE_DTYPE    = "bfloat16"
    USE_BF16         = True
    USE_FP16         = False


# Reproducibility
def set_seed() -> None:
    import torch
    random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.SEED)


# GPU / environment check
def check_environment() -> None:
    import torch

    log.info("-" * 56)
    log.info(f"Python      : {sys.version.split()[0]}")
    log.info(f"PyTorch     : {torch.__version__}")

    # Log key package versions to help debug compatibility issues
    for pkg in ["transformers", "peft", "trl", "bitsandbytes", "accelerate"]:
        try:
            import importlib
            m = importlib.import_module(pkg)
            log.info(f"{pkg:<12}: {m.__version__}")
        except Exception:
            log.warning(f"{pkg:<12}: NOT FOUND")

    if not torch.cuda.is_available():
        log.warning("CUDA not detected — training on CPU will be extremely slow.")
        log.info("-" * 56)
        return

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024 ** 3
    log.info(f"GPU         : {props.name}")
    log.info(f"VRAM        : {vram_gb:.1f} GB")
    log.info(f"CUDA sm     : {props.major}.{props.minor}")

    # bf16 requires sm >= 8.0 (Ampere and newer)
    if props.major < 8:
        log.warning("GPU does not support native bf16 — switching to fp16.")
        CFG.COMPUTE_DTYPE = "float16"
        CFG.USE_BF16 = False
        CFG.USE_FP16 = True
    else:
        log.info(f"Precision   : bfloat16 (sm {props.major}.{props.minor} >= 8.0)")

    if vram_gb < 5.5:
        log.warning(
            f"Only {vram_gb:.1f} GB VRAM detected. "
            "Consider reducing SFT_MAX_SEQ_LEN or LORA_R."
        )
    log.info("-" * 56)


# Memory utilities
def free_memory() -> None:
    """Release GPU + CPU caches between training stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def vram_snapshot(label: str = "") -> None:
    """Log current / peak / reserved VRAM."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        cur  = torch.cuda.memory_allocated()     / 1024 ** 3
        peak = torch.cuda.max_memory_allocated()  / 1024 ** 3
        res  = torch.cuda.memory_reserved()      / 1024 ** 3
        tag  = f"[{label}] " if label else ""
        log.info(
            f"{tag}VRAM  allocated: {cur:.2f} GB | "
            f"reserved: {res:.2f} GB | peak: {peak:.2f} GB"
        )
    except Exception:
        pass


def reset_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# Model + tokenizer loader  (shared by CPT and SFT)
def load_model_and_tokenizer(
    base_model: str = CFG.BASE_MODEL,
    adapter_path: Optional[str] = None,
):

    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    #  Tokenizer 
    log.info(f"Loading tokenizer : {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    # LLaMA has no pad token — reuse EOS.
    # padding_side="right" avoids position-embedding overflow with bf16/fp16.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = getattr(torch, CFG.COMPUTE_DTYPE)

    #  BitsAndBytes config 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,        
    )

    #  Base model 

    log.info(f"Loading base model: {base_model}  [4-bit NF4, compute={CFG.COMPUTE_DTYPE}]")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0},            
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,        
    )
    model.config.use_cache = False     
    model.config.pretraining_tp = 1    

    #  Prepare for kbit training 
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    #  LoRA config 
    lora_cfg = LoraConfig(
        r=CFG.LORA_R,
        lora_alpha=CFG.LORA_ALPHA,
        lora_dropout=CFG.LORA_DROPOUT,
        target_modules=CFG.LORA_TARGETS,
        bias="none",                   
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    #  Adapter strategy 
    if adapter_path and Path(adapter_path).exists():
        # CPT -> SFT two-stage flow:
        # Load CPT adapter frozen, then add a fresh trainable SFT adapter.
        log.info(f"Loading CPT adapter (frozen) from: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            adapter_name="cpt",
            is_trainable=False,        
        )

        log.info("Adding fresh trainable SFT adapter on top of CPT ...")
        model.add_adapter("sft", lora_cfg)

        # Activate ONLY the SFT adapter for gradient updates.
        # CPT adapter contributes to the forward pass (stacked) but
        # receives no gradients because is_trainable=False.
        model.set_adapter("sft")

    else:
        # SFT-only or CPT-only: single fresh adapter.
        log.info("Initialising fresh LoRA adapter (no CPT base) ...")
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()
    vram_snapshot("after model load")
    return model, tokenizer


# SFTConfig builder  (CPT and SFT share this; differ in a few fields)
def build_sft_config(
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_seq_len: int,
    packing: bool,
    save_steps: int,
    eval_steps: int,
    do_eval: bool = False,
):

    from trl import SFTConfig

    return SFTConfig(
        #  Output / checkpointing 
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,             # keep only 2 checkpoints

        #  Evaluation 
        eval_strategy="steps" if do_eval else "no",
        eval_steps=eval_steps if do_eval else None,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False,

        #  Training loop 
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        #  Optimizer 
        # paged_adamw_8bit keeps Adam states in CPU RAM in 8-bit,
        # paged to GPU on demand — critical for 6 GB VRAM.
        optim="paged_adamw_8bit",
        learning_rate=lr,
        lr_scheduler_type=CFG.LR_SCHEDULER,
        warmup_ratio=CFG.WARMUP_RATIO,
        weight_decay=CFG.WEIGHT_DECAY,
        max_grad_norm=CFG.MAX_GRAD_NORM,

        #  Precision 
        bf16=CFG.USE_BF16,
        fp16=CFG.USE_FP16,

        #  SFT-specific (sequence / packing) 
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        packing=packing,

        #  Logging 
        logging_steps=10,
        report_to="none",               

        #  Misc 
        seed=CFG.SEED,
        data_seed=CFG.SEED,
        dataloader_num_workers=0,       
        dataloader_pin_memory=False,    
        remove_unused_columns=True,
        group_by_length=True,           
    )


# Dataset loaders
def load_jsonl_dataset(path: str):
    """Load a JSONL file.  Each line must contain a 'text' field."""
    from datasets import Dataset

    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                raise ValueError(f"Line {i} in '{path}' is missing the 'text' field.")
            records.append({"text": obj["text"]})

    log.info(f"Loaded {len(records):,} records from {path}")
    return Dataset.from_list(records)


def load_cpt_dataset(raw_text_path: str, tokenizer):
    """
    Chunk the raw legal text corpus into non-overlapping token windows.

    The chunks are decoded back to strings so that SFTTrainer can re-tokenise
    with its own padding/truncation logic, avoiding a custom data collator.
    """
    from datasets import Dataset

    log.info(f"Reading CPT corpus : {raw_text_path}")
    raw_text = Path(raw_text_path).read_text(encoding="utf-8")

    log.info("Tokenising corpus (full pass) ...")
    token_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    log.info(f"Total tokens : {len(token_ids):,}")

    chunks = [
        token_ids[i : i + CFG.CPT_MAX_SEQ_LEN]
        for i in range(0, len(token_ids) - CFG.CPT_MAX_SEQ_LEN + 1, CFG.CPT_MAX_SEQ_LEN)
    ]
    log.info(f"CPT chunks (seq_len={CFG.CPT_MAX_SEQ_LEN}) : {len(chunks):,}")

    records = [{"text": tokenizer.decode(c, skip_special_tokens=True)} for c in chunks]
    random.shuffle(records)
    return Dataset.from_list(records)


# Log saver
def save_training_log(stage: str, result, trainer) -> None:
    import torch

    Path(CFG.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(CFG.LOG_FILE)

    existing: dict = {}
    if log_path.exists():
        with open(log_path) as f:
            existing = json.load(f)

    existing[stage] = {
        "train_runtime_s": result.metrics.get("train_runtime"),
        "train_loss":      result.metrics.get("train_loss"),
        "total_flos":      result.metrics.get("total_flos"),
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else None
        ),
        "history": trainer.state.log_history,
    }
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    log.info(f"Training log -> {log_path}")


# STAGE 1 — Continued Pre-Training (CPT)
def run_cpt() -> None:
    """
    CPT adapts the model's language prior to Indian Income Tax vocabulary
    and legal phrasing BEFORE instruction tuning.

      - Lower LR (5e-5): nudge priors, don't overwrite general knowledge
      - 1 epoch: more passes risk catastrophic forgetting
      - Packing ON: raw text has no instruction/response boundaries
      - No eval dataset: CPT has no labelled reference; we skip eval
        to keep VRAM usage minimal during this stage
    """
    from trl import SFTTrainer

    log.info("=" * 56)
    log.info("STAGE 1 — Continued Pre-Training (CPT)")
    log.info("=" * 56)
    reset_peak_vram()

    model, tokenizer = load_model_and_tokenizer(CFG.BASE_MODEL)
    train_ds = load_cpt_dataset(CFG.CPT_RAW_TEXT, tokenizer)

    cfg = build_sft_config(
        output_dir=CFG.CPT_OUTPUT_DIR,
        epochs=CFG.CPT_EPOCHS,
        batch_size=CFG.CPT_BATCH_SIZE,
        grad_accum=CFG.CPT_GRAD_ACCUM,
        lr=CFG.CPT_LR,
        max_seq_len=CFG.CPT_MAX_SEQ_LEN,
        packing=CFG.CPT_PACKING,
        save_steps=CFG.CPT_SAVE_STEPS,
        eval_steps=CFG.CPT_EVAL_STEPS,
        do_eval=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=cfg,
        processing_class=tokenizer,
    )

    log.info("Starting CPT ...")
    result = trainer.train()

    Path(CFG.CPT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(CFG.CPT_OUTPUT_DIR)
    tokenizer.save_pretrained(CFG.CPT_OUTPUT_DIR)
    log.info(f"CPT adapter saved -> {CFG.CPT_OUTPUT_DIR}")

    vram_snapshot("CPT done")
    save_training_log("cpt", result, trainer)

    del trainer, model
    free_memory()
    vram_snapshot("after CPT cleanup")


# STAGE 2 — Supervised Fine-Tuning (SFT)
def run_sft(
    base_model_override: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> None:
    """
    SFT teaches the model to follow legal Q&A instructions via QLoRA.

    Two-stage workflow (--mode both):
        base_model_override=CFG.CPT_OUTPUT_DIR causes load_model_and_tokenizer
        to load the CPT adapter frozen, then adds a fresh SFT adapter on top.
        Only SFT adapter weights receive gradient updates.

    Post-training merge:
        merge_and_unload() dequantises the 4-bit backbone to bf16/fp16,
        folds all active adapter deltas into the weights, and returns a plain
        HF CausalLM model (~2.5 GB). Saved to FINAL_MODEL_DIR for inference
        with no PEFT dependency required at runtime.
    """
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer

    log.info("=" * 56)
    log.info("STAGE 2 — Supervised Fine-Tuning (SFT)")
    log.info("=" * 56)
    reset_peak_vram()

    #  1. Load model 
    model, tokenizer = load_model_and_tokenizer(
        base_model=CFG.BASE_MODEL,
        adapter_path=base_model_override,
    )

    #  2. Datasets 
    train_ds = load_jsonl_dataset(CFG.SFT_TRAIN_DATA)
    eval_ds  = load_jsonl_dataset(CFG.SFT_TEST_DATA)

    #  3. Training config 
    cfg = build_sft_config(
        output_dir=CFG.SFT_OUTPUT_DIR,
        epochs=CFG.SFT_EPOCHS,
        batch_size=CFG.SFT_BATCH_SIZE,
        grad_accum=CFG.SFT_GRAD_ACCUM,
        lr=CFG.SFT_LR,
        max_seq_len=CFG.SFT_MAX_SEQ_LEN,
        packing=CFG.SFT_PACKING,
        save_steps=CFG.SFT_SAVE_STEPS,
        eval_steps=CFG.SFT_EVAL_STEPS,
        do_eval=True,
    )

    #  4. Trainer 
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG.SFT_EARLY_STOP)],
    )

    #  5. Train 
    log.info("Starting SFT ...")
    result = trainer.train(resume_from_checkpoint=resume_from)

    #  6. Save ONLY SFT adapter 
    Path(CFG.SFT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Determine the name of the adapter we actually trained
    adapter_to_save = "sft" if base_model_override else "default"

    trainer.model.save_pretrained(
        CFG.SFT_OUTPUT_DIR,
        selected_adapters=[adapter_to_save]
    )
    tokenizer.save_pretrained(CFG.SFT_OUTPUT_DIR)
    log.info(f"SFT adapter saved -> {CFG.SFT_OUTPUT_DIR}")

    vram_snapshot("SFT done")
    save_training_log("sft", result, trainer)

    #  7. Merge all active adapters into the base model for inference 
    log.info("Merging adapters into base model for inference ...")
    merged = trainer.model.merge_and_unload()

    Path(CFG.FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(CFG.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(CFG.FINAL_MODEL_DIR)
    log.info(f"Final merged model -> {CFG.FINAL_MODEL_DIR}  (no PEFT needed at inference)")

    del trainer, model, merged
    free_memory()
    vram_snapshot("after SFT cleanup")


# OOM guard
def oom_guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.error("=" * 56)
            log.error("CUDA OUT OF MEMORY — try these fixes in order:")
            log.error("  1. Reduce SFT_MAX_SEQ_LEN   (biggest lever)")
            log.error("  2. Reduce LORA_R             (16 -> 8)")
            log.error("  3. Increase SFT_GRAD_ACCUM   (16 -> 32)")
            log.error("  4. Keep LORA_TARGETS to attention layers only")
            log.error("  5. Close GPU-heavy background apps")
            log.error("=" * 56)
            free_memory()
        raise


# Entry point
def parse_args():
    p = argparse.ArgumentParser(
        description="lexitune: QLoRA fine-tuning for Indian Income Tax"
    )
    p.add_argument(
        "--mode",
        choices=["cpt", "sft", "both"],
        default="sft",
        help="cpt=pre-train only | sft=instruction-tune only | both=CPT then SFT",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume SFT from a checkpoint (e.g. finetune/sft_adapter/checkpoint-400)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()
    check_environment()   

    t0 = time.time()

    if args.mode == "cpt":
        oom_guard(run_cpt)

    elif args.mode == "sft":
        oom_guard(run_sft, resume_from=args.resume)

    elif args.mode == "both":
        oom_guard(run_cpt)
        log.info("CPT complete — starting SFT on CPT adapter ...")
        oom_guard(run_sft, base_model_override=CFG.CPT_OUTPUT_DIR)

    elapsed = time.time() - t0
    log.info(f"Total wall time : {elapsed / 60:.1f} min")
    vram_snapshot("final")
    log.info("All done.")


if __name__ == "__main__":
    main()