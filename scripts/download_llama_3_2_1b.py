import os
import torch
import huggingface_hub
import transformers

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SAVE_PATH = "lexitune/models/llama_3_2_1b"


def setup_environment():
    # Enable fast & resumable downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        # Will prompt only if not already logged in
        huggingface_hub.login()
    except Exception as e:
        print("Hugging Face authentication failed.")
        print('Run manually: python -c "import huggingface_hub; huggingface_hub.login()"')
        raise e


def model_exists(path: str) -> bool:
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def download_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer.save_pretrained(SAVE_PATH)
    model.save_pretrained(SAVE_PATH)

    print("LLaMA-3.2-1B downloaded successfully.")


def test_inference():
    pipe = transformers.pipeline(
        "text-generation",
        model=SAVE_PATH,
        tokenizer=SAVE_PATH,
        max_new_tokens=60,
    )

    output = pipe("What is standard deduction under new tax regime?")
    print("\nInference test output:\n")
    print(output[0]["generated_text"])


def main():
    setup_environment()

    if model_exists(SAVE_PATH):
        print(f"Model already exists at {SAVE_PATH}. Skipping download.")
    else:
        print("Model not found locally. Starting download...")
        download_model()

    test_inference()


if __name__ == "__main__":
    main()
