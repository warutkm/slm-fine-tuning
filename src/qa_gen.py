"""
Input:  data/flat_sections.jsonl
Output: data/qa_dataset.jsonl
        data/dataset_stats.json

QA types generated (per section, probabilistically):
  A. explain_section    - "Explain section X"
  B. definition         - "What is the definition of X?"
  C. eligibility        - "Can X claim Y deduction / exemption?"
  D. scenario           - "What happens if a taxpayer…?"
  E. condition          - "What are the conditions for…?"
  F. exception          - "What are the exceptions to…?"
  G. multi_hop          - Cross-section reasoning
  H. comparison         - Compare two provisions
  I. procedural         - "How does a taxpayer file / compute…?"
"""

import json
import random
import re
import hashlib
import argparse
import os
from pathlib import Path
from typing import Optional


#  CONFIG  #

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

MAX_CONTEXT_TOKENS = 1024      # approximate; 1 token ≈ 4 chars
MIN_TEXT_LEN       = 80       # skip trivially short sections
SEED               = 42

random.seed(SEED)

#  Token approximation 
def approx_tokens(text: str) -> int:
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    limit = max_tokens * 4
    if len(text) <= limit:
        return text
    # Cut at last sentence boundary
    cut = text[:limit]
    last_period = cut.rfind(".")
    return cut[:last_period + 1] if last_period > limit // 2 else cut


#  Text helpers 
def first_sentence(text: str) -> str:
    m = re.search(r"[.!?]", text)
    return text[:m.start() + 1].strip() if m else text[:200].strip()


def extract_section_title(sec: dict) -> str:
    title = sec.get("section_title", "").strip()
    if title and len(title) > 3:
        return title
    # Fall back to first meaningful phrase from full_text
    body = sec.get("full_text", "")
    # Remove section number prefix like "80C. (1)"
    body = re.sub(r"^\d+[A-Z]{0,3}\.\s+\(\d+\)\s+", "", body)
    return first_sentence(body)[:120]


def has_definitions(sec: dict) -> bool:
    text = sec.get("full_text", "")
    return bool(re.search(r'"[A-Za-z ]+"\s+means|shall mean|is defined as', text))


def has_conditions(sec: dict) -> bool:
    text = sec.get("full_text", "")
    return bool(re.search(
        r"subject to|condition|provided that|where|if|unless|only if|notwithstanding",
        text, re.I
    ))


def has_exceptions(sec: dict) -> bool:
    text = sec.get("full_text", "")
    return bool(re.search(
        r"except|exception|shall not apply|does not include|not be deemed|provided that nothing|nothing in this section",
        text, re.I
    ))


def has_deduction_or_exemption(sec: dict) -> bool:
    text = sec.get("full_text", "")
    return bool(re.search(
        r"deduction|exemption|rebate|allowance|relief|not included in|shall not be included",
        text, re.I
    ))


#  Instruction-response builders 
def fmt_context(sec: dict) -> str:
    """Formatted legal context block for injection."""
    snum  = sec.get("section_num", "?")
    title = extract_section_title(sec)
    chap  = sec.get("chapter_title", "")
    text  = truncate_to_tokens(sec.get("full_text", ""), MAX_CONTEXT_TOKENS)
    return (
        f"[Section {snum} — {title}]\n"
        f"Chapter: {chap}\n\n"
        f"{text}"
    )


def build_explain_section(sec: dict) -> Optional[dict]:
    snum  = sec.get("section_num")
    title = extract_section_title(sec)
    text  = sec.get("full_text", "")
    if len(text) < MIN_TEXT_LEN:
        return None

    templates = [
        f"Explain Section {snum} of the Income-Tax Act, 2025.",
        f"What does Section {snum} deal with under the Income-Tax Act, 2025?",
        f"Summarise the key provisions of Section {snum} ({title}).",
        f"What is the purpose and scope of Section {snum} of the Income-Tax Act?",
    ]
    instruction = random.choice(templates)
    # Construct answer from section text (truncated)
    response = (
        f"Section {snum} of the Income-Tax Act, 2025, titled '{title}', "
        f"falls under {sec.get('chapter_title', 'the Act')}.\n\n"
        + truncate_to_tokens(text, 400)
    )
    return _record(sec, "explain_section", instruction, response)


def build_definition(sec: dict) -> Optional[dict]:
    text = sec.get("full_text", "")
    # Find all quoted defined terms
    defined_terms = re.findall(r'"([A-Za-z ]{3,40})"(?:\s+means|\s+shall mean)', text)
    if not defined_terms:
        return None
    term = random.choice(defined_terms)
    # Extract the definition sentence
    pattern = re.compile(
        rf'"{re.escape(term)}"\s+(?:means|shall mean)\s+(.+?)(?=;\s*\(|\n\(|\Z)',
        re.S
    )
    m = pattern.search(text)
    definition = m.group(1).strip()[:600] if m else truncate_to_tokens(text, 200)

    templates = [
        f'What is the definition of "{term}" under the Income-Tax Act, 2025?',
        f'How is "{term}" defined in the Income-Tax Act, 2025?',
        f'Define "{term}" as per the Income-Tax Act, 2025.',
    ]
    instruction = random.choice(templates)
    response = (
        f'Under Section {sec.get("section_num")} of the Income-Tax Act, 2025, '
        f'"{term}" means {definition}'
    )
    return _record(sec, "definition", instruction, response)


def build_eligibility(sec: dict) -> Optional[dict]:
    if not has_deduction_or_exemption(sec):
        return None
    text  = sec.get("full_text", "")
    snum  = sec.get("section_num")
    title = extract_section_title(sec)

    # Detect the type of benefit
    if re.search(r"deduction", text, re.I):
        benefit = "claim a deduction"
    elif re.search(r"exemption|exempt", text, re.I):
        benefit = "claim an exemption"
    else:
        benefit = "avail the benefit"

    # Pick an assessable entity from text or default
    entities = ["an individual", "a Hindu Undivided Family (HUF)",
                 "a resident individual", "a company", "a partnership firm",
                 "a senior citizen", "a salaried employee"]
    entity = random.choice(entities)

    templates = [
        f"Can {entity} {benefit} under Section {snum} of the Income-Tax Act, 2025?",
        f"Who is eligible to {benefit} under Section {snum}?",
        f"What are the eligibility conditions to {benefit} under Section {snum} ({title})?",
    ]
    instruction = random.choice(templates)
    response = (
        f"Under Section {snum} ({title}) of the Income-Tax Act, 2025:\n\n"
        + truncate_to_tokens(text, 380)
        + "\n\nEligibility and conditions must be read with any relevant provisos and sub-sections."
    )
    return _record(sec, "eligibility", instruction, response)


def build_scenario(sec: dict) -> Optional[dict]:
    if not has_conditions(sec):
        return None
    text  = sec.get("full_text", "")
    snum  = sec.get("section_num")
    title = extract_section_title(sec)

    # Generate scenario from conditions present in text
    cond_words = re.findall(
        r"(?:where|if|when|in case|subject to)\s+([^,.;]{10,80})", text, re.I
    )
    scenario_hook = cond_words[0].strip() if cond_words else f"a taxpayer falls under Section {snum}"

    templates = [
        f"What happens if {scenario_hook} under the Income-Tax Act, 2025?",
        f"How does Section {snum} apply when {scenario_hook}?",
        f"A taxpayer faces a situation where {scenario_hook}. What are the tax implications under Section {snum}?",
    ]
    instruction = random.choice(templates)
    response = (
        f"Under Section {snum} ({title}) of the Income-Tax Act, 2025, the following provisions apply:\n\n"
        + truncate_to_tokens(text, 380)
    )
    return _record(sec, "scenario", instruction, response)


def build_conditions(sec: dict) -> Optional[dict]:
    if not has_conditions(sec):
        return None
    snum  = sec.get("section_num")
    title = extract_section_title(sec)
    text  = sec.get("full_text", "")

    templates = [
        f"What are the conditions to be satisfied under Section {snum} of the Income-Tax Act, 2025?",
        f"List the requirements under Section {snum} ({title}).",
        f"What must a taxpayer fulfil to invoke Section {snum}?",
    ]
    instruction = random.choice(templates)
    response = (
        f"Section {snum} ({title}) of the Income-Tax Act, 2025, prescribes the following conditions:\n\n"
        + truncate_to_tokens(text, 400)
    )
    return _record(sec, "condition", instruction, response)


def build_exception(sec: dict) -> Optional[dict]:
    if not has_exceptions(sec):
        return None
    snum  = sec.get("section_num")
    title = extract_section_title(sec)
    text  = sec.get("full_text", "")

    templates = [
        f"What are the exceptions under Section {snum} of the Income-Tax Act, 2025?",
        f"Are there any exclusions or exceptions provided in Section {snum}?",
        f"Which cases are excluded from the scope of Section {snum} ({title})?",
    ]
    instruction = random.choice(templates)
    response = (
        f"Section {snum} ({title}) of the Income-Tax Act, 2025, provides the following exceptions:\n\n"
        + truncate_to_tokens(text, 380)
    )
    return _record(sec, "exception", instruction, response)


def build_procedural(sec: dict) -> Optional[dict]:
    text = sec.get("full_text", "")
    if not re.search(r"shall be|must|required to|procedure|application|file|furnish|form|return", text, re.I):
        return None
    snum  = sec.get("section_num")
    title = extract_section_title(sec)

    templates = [
        f"What is the procedure prescribed under Section {snum} of the Income-Tax Act, 2025?",
        f"How should a taxpayer comply with Section {snum} ({title})?",
        f"Describe the procedural requirements under Section {snum}.",
    ]
    instruction = random.choice(templates)
    response = (
        f"The procedural requirements under Section {snum} ({title}) of the Income-Tax Act, 2025:\n\n"
        + truncate_to_tokens(text, 400)
    )
    return _record(sec, "procedural", instruction, response)


def build_multi_hop(sec: dict, all_sections: list[dict]) -> Optional[dict]:
    """Cross-reference question using the section's cross_refs."""
    xrefs = sec.get("cross_refs", [])
    if not xrefs:
        return None
    # Pick a cross-referenced section that exists
    ref_nums = set(s["section_num"] for s in all_sections)
    valid_xrefs = [x for x in xrefs if x in ref_nums]
    if not valid_xrefs:
        return None
    ref_num  = random.choice(valid_xrefs)
    snum     = sec.get("section_num")
    title    = extract_section_title(sec)

    templates = [
        f"How do Sections {snum} and {ref_num} of the Income-Tax Act, 2025 interact with each other?",
        f"What is the relationship between Section {snum} ({title}) and Section {ref_num}?",
        f"Explain how Section {ref_num} affects the application of Section {snum}.",
    ]
    instruction = random.choice(templates)
    response = (
        f"Section {snum} ({title}) refers to Section {ref_num} of the Income-Tax Act, 2025. "
        f"The interaction is as follows:\n\n"
        + truncate_to_tokens(sec.get("full_text", ""), 300)
        + f"\n\nReaders should also refer to Section {ref_num} for complete understanding of the cross-referenced provisions."
    )
    return _record(sec, "multi_hop", instruction, response)


#  Record factory ─
SYSTEM_PROMPT = (
    "You are a precise and reliable legal assistant specialised in the "
    "Income-Tax Act, 2025 (as amended by the Finance Act, 2026). "
    "Answer questions based strictly on the provisions of the Act. "
    "Do not speculate beyond the text of the law. "
    "Cite the relevant section number in your response."
)

def _record(sec: dict, qa_type: str, instruction: str, response: str) -> dict:
    ctx = fmt_context(sec)
    # Full prompt in LLaMA 3.1 Instruct format
    prompt = (
        f"<|begin_of_text|>\n"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Based on the following legal provisions, answer the question:\n\n"
        f"---\n{ctx}\n---\n\n{instruction}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    # Unique ID for dedup
    uid = hashlib.md5(f"{sec.get('section_num')}|{qa_type}|{instruction}".encode()).hexdigest()[:12]
    return {
        "id":          uid,
        "section_num": sec.get("section_num"),
        "chapter":     sec.get("chapter_num"),
        "qa_type":     qa_type,
        "instruction": instruction,
        "context":     ctx,
        "response":    response,
        # SFT-ready concatenation
        "text": prompt + response + "<|eot_id|>",
        "approx_tokens": approx_tokens(prompt + response),
    }


#  Main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # ---- CONFIG MAPPING (ONLY CHANGE) ----
    DATA_DIR   = Path(config["paths"]["data_dir"]) / "processed"
    FLAT_JSONL = DATA_DIR / "flat_sections.jsonl"
    OUT_JSONL  = DATA_DIR / "qa_dataset.jsonl"
    OUT_STATS  = DATA_DIR / "dataset_stats.json"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[qa_gen] Loading flat sections …")

    sections = []

    if not FLAT_JSONL.exists():
        print(f"[ERROR] Could not find {FLAT_JSONL}. Please run parse.py first.")
        return

    with open(FLAT_JSONL, encoding="utf-8") as f:
        for line in f:
            sections.append(json.loads(line))

    print(f"[qa_gen] {len(sections)} sections loaded")

    builders = [
        build_explain_section,
        build_definition,
        build_eligibility,
        build_scenario,
        build_conditions,
        build_exception,
        build_procedural,
    ]

    records = []
    seen_ids = set()
    type_counts: dict[str, int] = {}

    for sec in sections:
        if len(sec.get("full_text", "")) < MIN_TEXT_LEN:
            continue

        for builder in builders:
            rec = builder(sec)
            if rec and rec["id"] not in seen_ids:
                if rec["approx_tokens"] > 1200:
                    continue
                seen_ids.add(rec["id"])
                records.append(rec)
                type_counts[rec["qa_type"]] = type_counts.get(rec["qa_type"], 0) + 1

        rec = build_multi_hop(sec, sections)
        if rec and rec["id"] not in seen_ids:
            seen_ids.add(rec["id"])
            records.append(rec)
            type_counts["multi_hop"] = type_counts.get("multi_hop", 0) + 1

    random.shuffle(records)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[qa_gen] QA dataset written → {OUT_JSONL}  ({len(records)} records)")

    if len(records) > 0:
        avg_tokens = sum(r["approx_tokens"] for r in records) / max(len(records), 1)
        stats = {
            "total_records": len(records),
            "avg_approx_tokens": round(avg_tokens, 1),
            "by_type": type_counts,
            "sections_covered": len({r["section_num"] for r in records}),
        }

        with open(OUT_STATS, "w") as f:
            json.dump(stats, f, indent=2)

        print("[qa_gen] Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print("[qa_gen] No records were generated.")


if __name__ == "__main__":
    main()