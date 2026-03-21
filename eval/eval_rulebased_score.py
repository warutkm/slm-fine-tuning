import re


def extract_numbers(text: str) -> set:
    """Extract numeric tokens — handles ₹ amounts like 1,50,000 and percentages."""
    return set(re.findall(r"[\d,]+(?:\.\d+)?%?", text))


def extract_sections(text: str) -> set:
    """Extract section references like 80C, 112A, 115BAC etc."""
    return set(re.findall(
        r"\b(?:section\s*)?(\d{2,3}[a-z]{0,4}(?:\(\d+[a-z]?\))?)\b",
        text.lower()
    ))


def token_overlap(pred: str, ref: str) -> float:
    """Unigram F1 between prediction and reference."""
    pred_tokens = set(pred.lower().split())
    ref_tokens  = set(ref.lower().split())
    if not ref_tokens:
        return 0.0
    common    = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall    = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def detect_regime_confusion(response: str, ground_truth: str) -> bool:
    r  = response.lower()
    gt = ground_truth.lower()

    model_says_new_available = (
        "available under the new" in r or
        ("new regime" in r and "not available" not in r.split("new regime")[0][-30:])
    )
    gt_says_new_not_available = "not available under the new" in gt

    model_says_new_not_available = "not available under the new" in r
    gt_says_new_available = (
        "available under the new" in gt and
        "not available under the new" not in gt
    )

    if gt_says_new_not_available and model_says_new_available \
            and "not" not in r.split("new regime")[0][-10:]:
        return True
    if gt_says_new_available and model_says_new_not_available:
        return True
    return False


def score_response(response: str, ground_truth: str):
    """
    Scoring: 0–4 scale per question
      4  → Correct, complete, accurate numbers, regime-aware
      3  → Mostly correct, minor omission
      2  → Partially correct or vague
      1  → Outdated, confused, or heavily incomplete
      0  → Hallucination / refusal / fabricated section

    Error tags:
      OK     → No issues
      HALL   → Hallucination (refusal, fabricated section, nonsensical)
      OUT    → Outdated information detected
      VAGUE  → Vague, non-specific answer
      MISS   → Missing key numeric values present in ground truth
      CONF   → Incorrectly contradicts a regime distinction
      PART   → Partially correct (some points covered, some missed)

    Returns (score, tags, overlap_f1).
    """
    r   = response.lower()
    gt  = ground_truth.lower()
    tags  = []
    score = 4

    # Hallucination / refusal
    refusal_phrases = [
        "i don't know", "i do not know", "cannot answer",
        "not sure", "unsure", "i am unable", "i cannot provide",
        "no information", "beyond my knowledge"
    ]
    if any(x in r for x in refusal_phrases):
        return 0, ["HALL"], 0.0

    fake_sections = ["section 999", "section xyz", "section abc", "imaginary section"]
    if any(x in r for x in fake_sections):
        return 0, ["HALL"], 0.0

    # Outdated information
    outdated_phrases = [
        "before 2023", "old slabs apply", "fy 2021", "fy 2020",
        "₹3,00,000 exemption for all", "standard deduction of ₹40,000"
    ]
    if any(x in r for x in outdated_phrases):
        tags.append("OUT")
        score -= 2

    # Vague answer
    vague_phrases = [
        "depends on", "may vary", "as applicable",
        "consult a ca", "subject to conditions", "generally speaking"
    ]
    if sum(1 for x in vague_phrases if x in r) >= 2:
        tags.append("VAGUE")
        score -= 1

    # Missing key numbers from ground truth
    significant_gt_nums = {
        n for n in extract_numbers(gt)
        if len(n.replace(",", "").replace("%", "").replace(".", "")) >= 2
    }
    missing_nums = significant_gt_nums - extract_numbers(r)
    if significant_gt_nums and len(missing_nums) > len(significant_gt_nums) * 0.5:
        tags.append("MISS")
        score -= 1

    # Regime confusion
    if detect_regime_confusion(r, gt):
        tags.append("CONF")
        score -= 2

    # Token overlap
    overlap = token_overlap(response, ground_truth)
    if overlap < 0.10:
        tags.append("PART")
        score -= 1

    score = max(0, score)
    if not tags:
        tags.append("OK")

    return score, tags, round(overlap, 4)