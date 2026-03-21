import torch
import transformers
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

MODELS = {
    "name": "LLaMA 3.2 (1B)",
    "path": str(BASE_DIR / "models" / "llama_3_2_1b"),
}


# Load model + tokenizer
def load_model(model_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    model.eval()
    return tokenizer, model


# Build prompt manually
def build_prompt(conversation):
    prompt = (
        "You are an Indian income tax expert. "
        "Answer questions accurately based on Indian Income Tax laws. "
        # "If unsure, clearly say you do not know.\n"
    )

    for turn in conversation:
        if turn["role"] == "user":
            prompt += f"User: {turn['content']}\n"
        else:
            prompt += f"Assistant: {turn['content']}\n"

    prompt += "Assistant:"
    return prompt


# Chat loop
def chat_loop(tokenizer, model):
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

        prompt = build_prompt(conversation)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assistant_reply = decoded.split("Assistant:")[-1].strip()

        conversation.append(
            {"role": "assistant", "content": assistant_reply}
        )

        print("\n--- MODEL RESPONSE ---")
        print(assistant_reply)
        print("-" * 40)


def main():
    model_info = MODELS

    if not model_info:
        raise ValueError("Invalid choice")

    print(f"\nLoading {model_info['name']}...")
    tokenizer, model = load_model(model_info["path"])

    chat_loop(tokenizer, model)


if __name__ == "__main__":
    main()
