import os
import argparse
import json
import torch
import transformers
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import YOUR RAG module
import src.rag as rag


# =========================
# UTILITIES
# =========================

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def create_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


# =========================
# MODEL LOADING (NON-RAG)
# =========================

def load_model(model_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True
    )

    model.eval()
    return tokenizer, model


# =========================
# PROMPT (NON-RAG)
# =========================

def build_prompt(conversation, system_prompt):
    prompt = system_prompt + "\n\n"

    for turn in conversation:
        if turn["role"] == "user":
            prompt += f"User: {turn['content']}\n"
        else:
            prompt += f"Assistant: {turn['content']}\n"

    prompt += "Assistant:"
    return prompt


# =========================
# CHAT LOOP
# =========================

def chat_loop(tokenizer, model, system_prompt, max_new_tokens):
    conversation = []

    print("\nEnter your tax question (type 'exit' or 'quit' to stop):")

    while True:
        user_input = input("> ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        prompt = build_prompt(conversation, system_prompt)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_reply = decoded.split("Assistant:")[-1].strip()

        conversation.append(
            {"role": "assistant", "content": assistant_reply}
        )

        print("\n--- MODEL RESPONSE ---")
        print(assistant_reply)
        print("-" * 40)


# =========================
# RAG CHAT LOOP (USING YOUR rag.py)
# =========================

def rag_chat_loop(max_new_tokens):
    print("\n[INFO] Using RAG pipeline (fine-tuned + retrieval)")
    print("\nEnter your tax question (type 'exit' or 'quit' to stop):")

    # Load collection once
    collection = rag.get_collection()

    # Load fine-tuned model (your rag.py already handles this properly)
    model, tokenizer = rag.load_model("finetuned")

    while True:
        user_input = input("> ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        if not user_input:
            continue

        result = rag.answer(
            query=user_input,
            model=model,
            tokenizer=tokenizer,
            collection=collection,
            max_new_tokens=max_new_tokens
        )

        print("\n--- RETRIEVED SECTIONS ---")
        for i, c in enumerate(result["retrieved_chunks"], 1):
            print(f"{i}. Section {c['section_num']} — {c['section_title']}")

        print("\n--- MODEL RESPONSE ---")
        print(result["response"])
        print("-" * 40)


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "finetuned", "rag"],
        default="rag"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    mode = args.mode
    max_new_tokens = config["model"]["max_new_tokens"]

    print(f"\n[INFO] Mode: {mode}")

    # =========================
    # MODE: RAG (DEFAULT)
    # =========================
    if mode == "rag":
        rag_chat_loop(max_new_tokens)
        return

    # =========================
    # NON-RAG MODES
    # =========================
    if mode == "base":
        model_path = config["model"]["base_model_dir"]

    elif mode == "finetuned":
        model_path = config["model"]["finetuned_model_dir"]

    print(f"[INFO] Loading model from: {model_path}")

    tokenizer, model = load_model(model_path)

    # SYSTEM PROMPT
    system_prompt = (
        "You are a precise and reliable legal assistant specialised in the "
        "Income-Tax Act, 2025 (as amended by the Finance Act, 2026). "
        "Answer strictly based on the law and cite relevant sections."
    )

    chat_loop(tokenizer, model, system_prompt, max_new_tokens)


if __name__ == "__main__":
    main()