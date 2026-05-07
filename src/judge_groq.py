"""
judge_groq.py  —  Batched LLM-as-judge for base_eval.py

Imported by base_eval.py — not intended to be run directly.

Provides:
  run_judge_batched()  — score predictions with Groq (llama3-70b-8192) in batches

  - Batching reduces API calls from ~200 → ~10-20, staying within RPM limits.
  - The judge prompt is context-grounded: the model is instructed to evaluate
    against the Legal Context field in each record, not its own training data.
    This prevents penalising the SLM for ITA-2025 answers that differ from
    the model's knowledge of the older ITA-1961.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import ssl
import time
import urllib.error
from pathlib import Path
from typing import Optional


#  Constants 

GROQ_MODEL      = "llama-3.3-70b-versatile"
GROQ_API_BASE   = "https://api.groq.com/openai/v1"

DEFAULT_BATCH_SIZE  = 10
JUDGE_MAX_TOKENS    = 4096        
INTER_BATCH_SLEEP   = 30.0        
MAX_RETRIES         = 4
RETRY_BASE_SLEEP    = 60  

# Context / response character limits per sample inside the batch prompt
CTX_LIMIT  = 600
GOLD_LIMIT = 500
PRED_LIMIT = 500


#  System prompt 

JUDGE_SYSTEM = """You are an expert evaluator for a legal QA system built on the
Income-Tax Act, 2025 (India) — a brand-new statute that replaced ITA-1961.

CRITICAL INSTRUCTION — CONTEXT-GROUNDED EVALUATION:
  The model under test (an SLM) was fine-tuned specifically on ITA-2025.
  ITA-2025 differs significantly from ITA-1961 in section numbers, thresholds,
  and definitions. You MUST evaluate each answer solely against:
    (a) the "Legal Context" field provided for that sample, AND
    (b) the "Gold Answer" field (which is also derived from ITA-2025).
  Do NOT penalise the model for stating values, section numbers, or rules that
  differ from ITA-1961 if they are consistent with the provided Legal Context.
  If the Gold Answer and Legal Context agree with the model's answer, score it
  highly — even if it contradicts your own training data about older tax law.

You will receive a JSON array of evaluation items. For EACH item, return one
JSON object scored with this rubric:

  factual_accuracy  (0-4)
      4 = fully correct per the Legal Context + Gold Answer
      3 = minor gap or imprecise phrasing, substance is right
      2 = partially correct (some right, some wrong)
      1 = mostly incorrect
      0 = completely wrong or contradicts the context

  section_citation  (0-2)
      2 = cites the correct section number
      1 = cites a related but not exact section
      0 = wrong section or no citation

  completeness      (0-2)
      2 = covers all key points from the Gold Answer
      1 = covers some key points
      0 = misses most key points

  no_hallucination  (0-2)
      2 = no claims beyond the Legal Context
      1 = minor unsupported claim
      0 = significant hallucination (facts not in the Legal Context)
      NOTE: facts consistent with the Legal Context are NOT hallucinations,
      even if they differ from ITA-1961 or your prior knowledge.

  total_score  = factual_accuracy + section_citation + completeness + no_hallucination  (0-10)

  section_cited   (bool) — true if a section number was cited
  hallucinated    (bool) — true if no_hallucination < 2

Return ONLY a valid JSON array — one object per input item, in the same order.
Each object must have EXACTLY these keys:
  id, section_num, qa_type,
  factual_accuracy, section_citation, completeness, no_hallucination,
  total_score, section_cited, hallucinated, reasoning

The "reasoning" value must be a single string with no internal double quotes.
No markdown fences. No extra keys. No preamble."""


def _build_batch_user_prompt(batch: list[dict]) -> str:
    items = []
    for i, pred in enumerate(batch, 1):
        items.append(
            f"--- ITEM {i} ---\n"
            f"id: {pred['id']}\n"
            f"section_num: {pred.get('section_num', '')}\n"
            f"qa_type: {pred.get('qa_type', '')}\n"
            f"Question: {pred['instruction']}\n"
            f"Legal Context (ITA-2025):\n{pred.get('context', '')[:CTX_LIMIT]}\n"
            f"Gold Answer:\n{pred.get('gold_response', '')[:GOLD_LIMIT]}\n"
            f"Model Answer:\n{pred.get('model_response', '')[:PRED_LIMIT]}"
        )
    return (
        "Evaluate each item below and return a JSON array with one object per item "
        f"(total: {len(batch)} objects).\n\n"
        + "\n\n".join(items)
    )


def _call_groq_batch(batch: list[dict], api_key: str) -> list[dict]:
    """
    Send one batch to Groq and return a list of parsed score dicts.
    """
    user_text = _build_batch_user_prompt(batch)

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_text},
        ],
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": 0.0,
    }

    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type":   "application/json",
        "Authorization":  f"Bearer {api_key}",
        "Content-Length": str(len(body_bytes)),
    }

    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection("api.groq.com", timeout=120, context=ctx)
    try:
        conn.request("POST", "/openai/v1/chat/completions", body=body_bytes, headers=headers)
        resp = conn.getresponse()
        resp_body = resp.read()
        if resp.status != 200:
            # Include Groq's error body in the exception message for easier debugging
            try:
                err_detail = json.loads(resp_body).get("error", {}).get("message", resp_body[:300])
            except Exception:
                err_detail = resp_body[:300]
            raise urllib.error.HTTPError(
                url="https://api.groq.com/openai/v1/chat/completions",
                code=resp.status,
                msg=f"{resp.reason} — {err_detail}",
                hdrs=None,
                fp=None,
            )
        body = json.loads(resp_body)
    finally:
        conn.close()

    raw_text = body["choices"][0]["message"]["content"]
    return _parse_batch_response(raw_text, batch)


def _sanitise_reasoning(text: str) -> str:
    text = re.sub(r'(?<!\\)"', "'", text)
    text = text.strip()
    if text and text[-1] not in ".!?)\"'":
        text = text.rsplit(" ", 1)[0] + "…"
    return text


def _repair_truncated_json(text: str) -> str:
    last_brace = text.rfind("}")
    if last_brace == -1:
        return text

    truncated = text[: last_brace + 1]

    opens = truncated.count("[") - truncated.count("]")
    if opens > 0:
        truncated += "]" * opens

    return truncated


def _parse_batch_response(text: str, batch: list[dict]) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    def try_parse(s: str):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return _validate_batch_scores(parsed, batch)
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        return _validate_batch_scores(v, batch)
        except (json.JSONDecodeError, Exception):
            return None

    #  attempt 1: direct parse 
    result = try_parse(text)
    if result is not None:
        return result

    #  attempt 2: repair truncation then re-parse 
    repaired = _repair_truncated_json(text)
    result = try_parse(repaired)
    if result is not None:
        print(f"\n  [parse] Recovered truncated response ({len(batch)} items)", flush=True)
        return result

    #  attempt 3: sanitise reasoning fields then re-parse 
    sanitised = re.sub(
        r'"reasoning"\s*:\s*"(.*?)"(\s*[,}])',
        lambda m: '"reasoning": "' + _sanitise_reasoning(m.group(1)) + '"' + m.group(2),
        repaired,
        flags=re.DOTALL,
    )
    result = try_parse(sanitised)
    if result is not None:
        print(f"\n  [parse] Recovered after reasoning sanitisation", flush=True)
        return result

    #  attempt 4: extract individual JSON objects with balanced-brace approach 
    objects = []
    depth, start = 0, None
    for i, ch in enumerate(sanitised):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = sanitised[start: i + 1]
                try:
                    obj = json.loads(chunk)
                    if "id" in obj or "factual_accuracy" in obj:
                        objects.append(obj)
                except json.JSONDecodeError:
                    chunk2 = re.sub(
                        r'"reasoning"\s*:\s*"(.*?)"(\s*[,}])',
                        lambda m: '"reasoning": "' + _sanitise_reasoning(m.group(1)) + '"' + m.group(2),
                        chunk,
                        flags=re.DOTALL,
                    )
                    try:
                        obj = json.loads(chunk2)
                        if "id" in obj or "factual_accuracy" in obj:
                            objects.append(obj)
                    except Exception:
                        pass
                start = None

    if objects:
        print(f"\n  [parse] Recovered {len(objects)}/{len(batch)} objects via extraction", flush=True)
        return _validate_batch_scores(objects, batch)

    raise ValueError(
        f"Could not parse batch response ({len(batch)} items).\n"
        f"First 400 chars:\n{text[:400]}"
    )


def _validate_batch_scores(raw_list: list[dict], batch: list[dict]) -> list[dict]:
    """
    Ensure every item has the required fields and sane numeric ranges.
    Falls back to id from the batch record when the model omits it.
    """
    required_int = {
        "factual_accuracy": (0, 4),
        "section_citation": (0, 2),
        "completeness":     (0, 2),
        "no_hallucination": (0, 2),
        "total_score":      (0, 10),
    }
    out = []
    for i, (item, pred) in enumerate(zip(raw_list, batch)):
        # Guarantee id/section_num/qa_type come from the source record
        item["id"]          = pred["id"]
        item["section_num"] = pred.get("section_num", item.get("section_num", ""))
        item["qa_type"]     = pred.get("qa_type",     item.get("qa_type", ""))

        # Clamp numeric fields
        for field, (lo, hi) in required_int.items():
            val = item.get(field)
            try:
                item[field] = max(lo, min(hi, int(val)))
            except (TypeError, ValueError):
                item[field] = 0

        # Recompute total to be safe
        item["total_score"] = (
            item["factual_accuracy"]
            + item["section_citation"]
            + item["completeness"]
            + item["no_hallucination"]
        )

        # Bool fields
        item["section_cited"] = bool(item.get("section_cited", item["section_citation"] > 0))
        item["hallucinated"]  = bool(item.get("hallucinated",  item["no_hallucination"] < 2))

        # Reasoning fallback
        if not isinstance(item.get("reasoning"), str) or not item["reasoning"].strip():
            item["reasoning"] = "no reasoning provided"

        out.append(item)

    return out


#  Retry  

def _call_with_retry(
    batch: list[dict],
    api_key: str,
    max_retries: int = MAX_RETRIES,
) -> list[dict]:
    
    # Call Groq with exponential back-off on rate-limit / transient errors.

    for attempt in range(max_retries + 1):
        try:
            return _call_groq_batch(batch, api_key)
        except urllib.error.HTTPError as e:
            status = e.code
            if status == 429 or status >= 500:
                wait = RETRY_BASE_SLEEP * (2 ** attempt)
                print(f"\n  [retry] HTTP {status} — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries:
                wait = RETRY_BASE_SLEEP * (2 ** attempt)
                print(f"\n  [retry] {e} — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"All {max_retries} retries exhausted for batch starting at id={batch[0]['id']}")


#  Main public function 

def run_judge_batched(
    predictions: list[dict],
    api_key: str,
    out_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep_between_batches: float = INTER_BATCH_SLEEP,
) -> list[dict]:

    # Score all predictions using Groq (llama3-70b-8192) in batches.

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-scored IDs
    done_ids: set[str] = set()
    results: list[dict] = []
    if out_path.exists():
        import json as _json
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = _json.loads(line)
                if r.get("total_score") is not None:
                    done_ids.add(r["id"])
                    results.append(r)
        print(f"[judge] Resuming — {len(done_ids)} already scored")

    pending = [p for p in predictions if p["id"] not in done_ids]
    n_batches = math.ceil(len(pending) / batch_size)
    print(
        f"[judge] {len(pending)} samples → {n_batches} batches of ≤{batch_size}  "
        f"(judge={GROQ_MODEL})"
    )

    errors = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for b_idx in range(0, len(pending), batch_size):
            batch = pending[b_idx: b_idx + batch_size]
            b_num = b_idx // batch_size + 1
            print(
                f"  Batch {b_num}/{n_batches}  "
                f"(ids {batch[0]['id']} … {batch[-1]['id']})  ",
                end="",
                flush=True,
            )

            try:
                scored = _call_with_retry(batch, api_key)

                for item in scored:
                    record = {
                        "id":               item["id"],
                        "section_num":      item["section_num"],
                        "qa_type":          item["qa_type"],
                        "factual_accuracy": item["factual_accuracy"],
                        "section_citation": item["section_citation"],
                        "completeness":     item["completeness"],
                        "no_hallucination": item["no_hallucination"],
                        "total_score":      item["total_score"],
                        "section_cited":    item["section_cited"],
                        "hallucinated":     item["hallucinated"],
                        "reasoning":        item["reasoning"],
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    results.append(record)

                avg_score = sum(s["total_score"] for s in scored) / len(scored)
                print(f"avg_score={avg_score:.2f}/10  ✓")

            except Exception as e:
                errors += 1
                print(f"  ERROR: {e}")
                # Write error placeholders so the run is resumable
                for pred in batch:
                    placeholder = {
                        "id":               pred["id"],
                        "section_num":      pred.get("section_num", ""),
                        "qa_type":          pred.get("qa_type", ""),
                        "factual_accuracy": None,
                        "section_citation": None,
                        "completeness":     None,
                        "no_hallucination": None,
                        "total_score":      None,
                        "section_cited":    None,
                        "hallucinated":     None,
                        "reasoning":        f"ERROR: {e}",
                    }
                    fout.write(json.dumps(placeholder, ensure_ascii=False) + "\n")
                    results.append(placeholder)

            # Sleep between batches to respect RPM limits
            if b_num < n_batches:
                time.sleep(sleep_between_batches)

    good = sum(1 for r in results if r.get("total_score") is not None)
    print(f"\n[judge] Done. Scored: {good}  Errors: {errors}  → {out_path}")
    return results