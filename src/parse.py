"""
Produces:
  data/processed/structured_act.json   – full hierarchical tree
  data/processed/flat_sections.jsonl   – one record per section (input to QA generator)
"""
import re
import json
import argparse
import os
from pathlib import Path
from collections import Counter
from typing import Optional


#  CONFIG  #

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


# TEXT EXTRACTION

def extract_pdf_text(pdf_path: Path, out_path: Path) -> str:
    import subprocess
    print(f"[parse] Extracting text from {pdf_path} …")
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), str(out_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    print(f"[parse] Text written to {out_path}")
    return out_path.read_text(encoding="utf-8")


def load_text(pdf_path: Path, cache: Path) -> str:
    if cache.exists():
        print(f"[parse] Using cached text at {cache}")
        return cache.read_text(encoding="utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    return extract_pdf_text(pdf_path, cache)


# CLEANING

# FIX 1: Strip form-feed (^L / \x0c) but keep newlines so line structure is intact
FF_RE         = re.compile(r"\x0c")
TRAILING_WS   = re.compile(r"[ \t]+$", re.M)
MULTI_BLANK   = re.compile(r"\n{3,}")

# Footnote / editorial annotation lines
FOOTNOTE_RE   = re.compile(
    r"^\s*\d+\.\s+(Substituted|Inserted|Omitted|Added|Prior to|Assented)",
    re.M,
)


def clean_text(raw: str) -> str:
    text = FF_RE.sub("", raw)           # remove form-feeds (not newlines)
    text = TRAILING_WS.sub("", text)
    lines = []
    skip = False
    for line in text.split("\n"):
        if FOOTNOTE_RE.match(line):
            skip = True
        elif skip and line.strip() == "":
            skip = False
            continue
        if not skip:
            lines.append(line)
    text = "\n".join(lines)
    text = MULTI_BLANK.sub("\n\n", text)
    return text.strip()


# CHAPTER SPLITTING
# FIX 2: Chapter headers are CENTRED — match with leading whitespace

# Matches:  "       CHAPTER I\n       PRELIMINARY"
# The title line (e.g. PRELIMINARY) immediately follows the chapter number line.
CHAPTER_LINE_RE = re.compile(
    r"^\s{5,}(CHAPTER\s+[IVXLCDM]+)\s*$",
    re.M,
)


def split_into_chapter_blocks(lines: list[str]) -> list[dict]:
    """
    Identify chapter boundaries from the line list.
    Returns [{chapter_num, chapter_title, start_line, end_line}].
    """
    positions = []  # (line_index, chapter_num, chapter_title)

    for i, line in enumerate(lines):
        m = CHAPTER_LINE_RE.match(line)
        if not m:
            continue
        chapter_num = m.group(1).strip()
        # Title is the NEXT non-blank line
        chapter_title = ""
        for j in range(i + 1, min(i + 4, len(lines))):
            candidate = lines[j].strip()
            # Title is all-uppercase (or all-caps with hyphens/commas)
            if candidate and re.match(r"^[A-Z][A-Z\s\-,\']+$", candidate):
                chapter_title = candidate
                break
        positions.append((i, chapter_num, chapter_title))

    # Build blocks with sentinel
    positions.append((len(lines), None, None))
    blocks = []
    for k, (start, cnum, ctitle) in enumerate(positions[:-1]):
        end = positions[k + 1][0]
        body = "\n".join(lines[start:end])
        blocks.append(
            {
                "chapter_num":   cnum,
                "chapter_title": ctitle,
                "body":          body,
                "start_line":    start,
                "end_line":      end,
            }
        )
    return blocks


# SECTION EXTRACTION
# FIX 3: Section title is on the PREVIOUS line; section number may have tabs

# Matches the NUMBER line: "1. (1) …"  or  "399.\t\t(1) …"  or  "80C. …"
# The section number starts at column 0 (no leading spaces) or after minimal indent.
SECTION_NUM_LINE_RE = re.compile(
    r"^(\d+[A-Z]{0,3})[.\t]+\s*(?:\(1\)\s+)?(.{5,})",
    re.M,
)

# A "title line" is a non-indented, Title-Case or mixed sentence ending in '.'
# It must NOT start with a digit (to avoid confusing table rows).
TITLE_LINE_RE = re.compile(
    r"^[A-Z][A-Za-z].*\.\s*$"
)


def is_title_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^\d", stripped):         # starts with digit → table row
        return False
    if len(stripped) > 200:               # unreasonably long
        return False
    return bool(TITLE_LINE_RE.match(stripped))


def extract_sections_from_lines(
    body_lines: list[str],
    chapter_num: str,
    chapter_title: str,
) -> list[dict]:
    """
    Walk chapter lines; find every (title_line, section_number_line) pair.
    Title line is the line IMMEDIATELY before the section number line.
    """
    sections = []
    # Collect (line_index, section_num, heading) for all section starts
    boundaries = []

    for i, line in enumerate(body_lines):
        m = SECTION_NUM_LINE_RE.match(line)
        if not m:
            continue
        sec_num = m.group(1)
        # Look back one line for the title
        heading = ""
        if i > 0 and is_title_line(body_lines[i - 1]):
            heading = body_lines[i - 1].strip().rstrip(".")
        boundaries.append((i, sec_num, heading))

    if not boundaries:
        return []

    # Build sections between consecutive boundaries
    # Include title line (i-1) in the body span if it was consumed as heading
    boundaries.append((len(body_lines), None, None))
    for k, (start, sec_num, heading) in enumerate(boundaries[:-1]):
        end = boundaries[k + 1][0]
        body_slice = body_lines[start:end]
        full_text = "\n".join(body_slice).strip()

        subsections = parse_subsections(full_text)
        sections.append(
            {
                "section_num":   sec_num,
                "section_title": heading,
                "chapter_num":   chapter_num,
                "chapter_title": chapter_title,
                "full_text":     full_text,
                "subsections":   subsections,
            }
        )
    return sections


# SUBSECTION / CLAUSE / SUB-CLAUSE PARSING

# (1) …  with optional leading whitespace
SUBSEC_RE     = re.compile(r"^\s*\((\d+)\)\s+(.+)", re.M)
CLAUSE_RE     = re.compile(r"^\s*\(([a-z])\)\s+(.+)", re.M)
SUBCLAUSE_RE  = re.compile(r"^\s*\(([ivxlcdm]+)\)\s+(.+)", re.M)


def parse_subsections(sec_text: str) -> list[dict]:
    matches = list(SUBSEC_RE.finditer(sec_text))
    if not matches:
        return []
    ends = [m.start() for m in matches[1:]] + [len(sec_text)]
    result = []
    for m, end in zip(matches, ends):
        body = sec_text[m.start():end].strip()
        result.append(
            {
                "subsection_num": m.group(1),
                "text":           body,
                "clauses":        parse_clauses(body),
            }
        )
    return result


def parse_clauses(text: str) -> list[dict]:
    matches = list(CLAUSE_RE.finditer(text))
    if not matches:
        return []
    ends = [m.start() for m in matches[1:]] + [len(text)]
    result = []
    for m, end in zip(matches, ends):
        body = text[m.start():end].strip()
        result.append(
            {
                "clause_letter": m.group(1),
                "text":          body,
                "sub_clauses":   parse_sub_clauses(body),
            }
        )
    return result


def parse_sub_clauses(text: str) -> list[dict]:
    return [
        {"sub_clause_num": m.group(1), "text": m.group(2).strip()}
        for m in SUBCLAUSE_RE.finditer(text)
    ]


# SCHEDULE EXTRACTION
# FIX 4: Schedule headers are CENTRED — match by stripping and checking core

SCHEDULE_HEADER_RE = re.compile(r"^SCHEDULE\s+[IVXLCDM]+$")


def extract_schedules(lines: list[str], schedule_start: int) -> list[dict]:
    """Extract schedule blocks starting from schedule_start line index."""
    boundaries = []
    for i in range(schedule_start, len(lines)):
        stripped = lines[i].replace("\x0c", "").strip()
        if SCHEDULE_HEADER_RE.match(stripped):
            boundaries.append((i, stripped))

    if not boundaries:
        return []

    boundaries.append((len(lines), None))
    schedules = []
    for k, (start, sch_id) in enumerate(boundaries[:-1]):
        end = boundaries[k + 1][0]
        full_text = "\n".join(lines[start:end]).strip()
        schedules.append({"schedule_id": sch_id, "full_text": full_text})
    return schedules

# CROSS-REFERENCE EXTRACTION

XREF_RE = re.compile(r"section\s+(\d+[A-Z]{0,3})(?:\s*\(\d+\))?", re.I)


def extract_cross_refs(text: str) -> list[str]:
    return sorted(set(XREF_RE.findall(text)))


# MAIN PIPELINE

def find_schedule_start(lines: list[str]) -> int:
    """Return line index where schedules begin (first standalone SCHEDULE header)."""
    for i, line in enumerate(lines):
        stripped = line.replace("\x0c", "").strip()
        if SCHEDULE_HEADER_RE.match(stripped):
            return i
    return len(lines)  # no schedules found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # ---- CONFIG MAPPING (ONLY CHANGE) ----
    DATA_DIR = Path(config["paths"]["data_dir"]) / "processed"
    PDF_PATH = Path(config["paths"]["input_files"]["it_act"])

    RAW_TXT  = DATA_DIR / "ita_raw.txt"
    OUT_JSON = DATA_DIR / "structured_act.json"
    OUT_FLAT = DATA_DIR / "flat_sections.jsonl"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    #  1. Load text  
    raw = load_text(PDF_PATH, RAW_TXT)

    #  2. Clean  
    print("[parse] Cleaning text …")
    clean = clean_text(raw)
    lines = clean.split("\n")
    print(f"[parse] Total lines after cleaning: {len(lines)}")

    #  3. Find where schedules begin 
    schedule_start = find_schedule_start(lines)
    main_lines = lines[:schedule_start]
    print(f"[parse] Main body: lines 0–{schedule_start}  |  "
          f"Schedule region: lines {schedule_start}–{len(lines)}")

    #  4. Split into chapters (main body only) 
    print("[parse] Splitting chapters …")
    chapter_blocks = split_into_chapter_blocks(main_lines)
    print(f"[parse] Found {len(chapter_blocks)} chapters")

    #  5. Extract sections from each chapter 
    all_sections = []
    for ch in chapter_blocks:
        ch_lines = ch["body"].split("\n")
        secs = extract_sections_from_lines(
            ch_lines, ch["chapter_num"], ch["chapter_title"]
        )
        all_sections.extend(secs)
    print(f"[parse] Found {len(all_sections)} sections")

    #  6. Enrich with cross-references 
    for sec in all_sections:
        sec["cross_refs"] = extract_cross_refs(sec["full_text"])

    #  7. Extract schedules 
    schedules = extract_schedules(lines, schedule_start)
    print(f"[parse] Found {len(schedules)} schedules")

    #  8. Build structured document 
    chapters_clean = [
        {k: v for k, v in ch.items() if k not in ("body",)}
        for ch in chapter_blocks
    ]
    structured = {
        "title":          "Income-Tax Act, 2025 (as amended by Finance Act, 2026)",
        "act_num":        "30 OF 2025",
        "total_chapters": len(chapter_blocks),
        "total_sections": len(all_sections),
        "chapters":       chapters_clean,
        "sections":       all_sections,
        "schedules":      schedules,
    }

    #  9. Write JSON  
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)
    print(f"[parse] structured_act.json → {OUT_JSON.stat().st_size / 1024:.1f} KB")

    #  10. Write flat JSONL 
    with open(OUT_FLAT, "w", encoding="utf-8") as f:
        for sec in all_sections:
            f.write(json.dumps(sec, ensure_ascii=False) + "\n")
    print(f"[parse] flat_sections.jsonl → {len(all_sections)} records")

    #  11. Stats  
    print("\n── Section coverage by chapter ──")
    counts = Counter(s["chapter_num"] for s in all_sections)
    for ch, n in counts.most_common():
        print(f"  {ch}: {n} sections")

    secs_with_subsections = sum(1 for s in all_sections if s["subsections"])
    secs_with_xrefs       = sum(1 for s in all_sections if s["cross_refs"])
    avg_subsecs = (
        sum(len(s["subsections"]) for s in all_sections) / len(all_sections)
        if all_sections else 0
    )
    print(f"\n── Quality stats ──")
    print(f"  Sections with subsections : {secs_with_subsections}/{len(all_sections)}")
    print(f"  Sections with cross-refs  : {secs_with_xrefs}/{len(all_sections)}")
    print(f"  Avg subsections/section   : {avg_subsecs:.1f}")


if __name__ == "__main__":
    main()