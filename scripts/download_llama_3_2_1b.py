import os
import argparse
import json
import torch
import huggingface_hub
import transformers


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setup_environment():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        huggingface_hub.login()
    except Exception:
        print("Hugging Face authentication failed.")
        print('Run manually: python -c "import huggingface_hub; huggingface_hub.login()"')
        raise


def model_exists(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def download_model(model_id, save_path):
    print(f"Downloading model: {model_id}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"Model saved at: {save_path}")

def test_inference(save_path, max_new_tokens):
    print("\nRunning inference test...\n")

    tokenizer = transformers.AutoTokenizer.from_pretrained(save_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        save_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    gen_config = transformers.GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt = """Explain in detail the standard deduction under the new tax regime in India.
Include:
- current amount
- eligibility
- comparison with old regime
"""

    output = pipe(prompt, generation_config=gen_config)

    print(output[0]["generated_text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    #  CONFIG MAPPING 
    model_id = config["model"]["model_id"]
    save_path = config["model"]["model_dir"]
    max_new_tokens = config["model"]["max_new_tokens"]

    create_dirs(save_path)
    setup_environment()

    if model_exists(save_path):
        print(f"Model already exists at {save_path}. Skipping download.")
    else:
        print("Model not found locally. Starting download...")
        download_model(model_id, save_path)

    test_inference(save_path, max_new_tokens)


if __name__ == "__main__":
    main()