import json
import torch
import pathlib
import argparse
from collections import defaultdict
from datetime import datetime

import transformers

from eval_score import score_response

def parse_args():
    parser = argparse.ArgumentParser(description="SLM Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--model",  type=str, required=True, help="Model key defined in config.json")
    return parser.parse_args()


# MODEL LOADER
def load_model(model_path: str, device: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        local_files_only=True
    )
    return tokenizer, model


# MAIN EVALUATION LOOP
def run_evaluation(model_key: str, config: dict):

    # Validate model key
    available_models = list(config["models"].keys())
    if model_key not in config["models"]:
        raise ValueError(
            f"Model '{model_key}' not found in config. "
            f"Available models: {available_models}"
        )

    model_info     = config["models"][model_key]
    input_file     = pathlib.Path(config["input_file"])
    output_dir     = pathlib.Path(config["output_dir"])
    max_new_tokens = config["max_new_tokens"]
    device         = config["device"]
    system_prompt  = config["system_prompt"]

    output_dir.mkdir(parents=True, exist_ok=True)
    datestamp = datetime.now().strftime("%d%m%y")

    print(f"\n{'='*50}")
    print(f"  Model  : {model_info['name']}")
    print(f"  Input  : {input_file}")
    print(f"  Device : {device}")
    print(f"{'='*50}\n")

    tokenizer, model = load_model(model_info["path"], device)

    results        = []
    section_scores = defaultdict(list)
    tag_counts     = defaultdict(int)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    total = len(lines)

    for idx, item in enumerate(lines, 1):
        q_id     = item["id"]
        section  = item["section"]
        question = item["question"]
        gt       = item["response"]
        source   = item.get("source", "")

        prompt = system_prompt + question
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated_text.startswith(prompt):
            model_response = generated_text[len(prompt):].strip()
        else:
            parts = generated_text.split("Question:")
            model_response = parts[-1].strip() if len(parts) > 1 else generated_text.strip()

        score, tags, overlap = score_response(model_response, gt)

        norm_section = section.replace("ITA", "").strip()
        section_scores[norm_section].append(score)
        for t in tags:
            tag_counts[t] += 1

        results.append({
            "id":             q_id,
            "section":        section,
            "section_label":  norm_section,
            "question":       question,
            "ground_truth":   gt,
            "source":         source,
            "model_response": model_response,
            "score":          score,
            "overlap_f1":     overlap,
            "error_tags":     "|".join(tags)
        })

        print(f"[{idx:>3}/{total}] {q_id} | section={norm_section:<14} | "
              f"score={score}/4 | tags={tags} | overlap={overlap:.2f}")

    # SAVE OUTPUTS
    file_stem    = f"{datestamp}_{model_info['name']}"
    jsonl_path   = output_dir / f"{file_stem}_eval.jsonl"
    summary_path = output_dir / f"{file_stem}_summary.txt"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # SUMMARY
    scores   = [r["score"] for r in results]
    overlaps = [r["overlap_f1"] for r in results]
    avg_score    = sum(scores) / total
    avg_overlap  = sum(overlaps) / total
    hall_rate    = sum(1 for s in scores if s == 0) / total * 100
    perfect_rate = sum(1 for s in scores if s == 4) / total * 100
    fail_rate    = sum(1 for s in scores if s <= 1) / total * 100

    summary_lines = [
        "=" * 50,
        f"  EVALUATION SUMMARY — {model_info['name']}",
        "=" * 50,
        f"  Total Questions     : {total}",
        f"  Average Score       : {avg_score:.2f} / 4",
        f"  Average Overlap F1  : {avg_overlap:.4f}",
        f"  Perfect (4/4) Rate  : {perfect_rate:.1f}%",
        f"  Failure (≤1) Rate   : {fail_rate:.1f}%",
        f"  Hallucination Rate  : {hall_rate:.1f}%",
        "",
        "  Error Tag Frequency:",
    ]
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        summary_lines.append(f"    {tag:<8} : {count} ({count/total*100:.1f}%)")

    summary_lines += ["", "  Section-wise Average Score:"]
    for sec in sorted(section_scores.keys()):
        sec_scores = section_scores[sec]
        avg        = sum(sec_scores) / len(sec_scores)
        summary_lines.append(f"    {sec:<16} : {avg:.2f}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\nOutputs saved:")
    print(f"  {jsonl_path}")
    print(f"  {summary_path}")


# ENTRY POINT
def main():
    args   = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    run_evaluation(args.model, config)


if __name__ == "__main__":
    main()