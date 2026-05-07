"""
app/history.py

Query history stored in st.session_state["history"] (list of dicts).
Also persisted to  logs/query_history.jsonl  so it survives page refreshes
within the same server session.

Each entry:
  ts          ISO timestamp
  question    str
  mode        "base" | "finetuned" | "rag" | "compare"
  response    str  (or dict for compare)
  latency_s   float
  tokens_out  int
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import streamlit as st

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "query_history.jsonl"


def _ensure_history():
    if "history" not in st.session_state:
        st.session_state["history"] = []


def add_entry(mode: str, question: str, result: dict | None = None, compare: dict | None = None):
    _ensure_history()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if compare:
        entry = {
            "ts":         datetime.now().isoformat(timespec="seconds"),
            "mode":       "compare",
            "question":   question,
            "compare":    compare,
        }
    else:
        entry = {
            "ts":         datetime.now().isoformat(timespec="seconds"),
            "mode":       mode,
            "question":   question,
            "response":   result.get("response", ""),
            "latency_s":  result.get("latency_s", 0),
            "tokens_out": result.get("tokens_out", 0),
        }

    st.session_state["history"].insert(0, entry)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_history() -> list[dict]:
    _ensure_history()
    return st.session_state["history"]


def render_history_sidebar():
    """Render a compact history panel in the sidebar."""
    history = get_history()
    if not history:
        st.sidebar.caption("No queries yet.")
        return

    st.sidebar.markdown("### Recent Queries")
    for i, e in enumerate(history[:10]):
        label = f"[{e['mode'].upper()}] {e['question'][:45]}…" if len(e['question']) > 45 else f"[{e['mode'].upper()}] {e['question']}"
        with st.sidebar.expander(label, expanded=False):
            st.caption(e["ts"])
            if e["mode"] == "compare":
                for m, r in e.get("compare", {}).items():
                    st.markdown(f"**{m}:** {str(r.get('response',''))[:200]}…")
            else:
                st.write(e.get("response", "")[:300])
                st.caption(f" {e.get('latency_s',0):.1f}s  |  {e.get('tokens_out',0)} tokens")
