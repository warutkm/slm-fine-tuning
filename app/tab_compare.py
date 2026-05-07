"""
app/tab_compare.py — Side-by-side comparison tab
Runs the same question through all three modes and shows results in columns.
"""
import streamlit as st
from app.model_loader import get_base_model, get_finetuned_model, get_rag_collection
from app.inference import run_base, run_finetuned, run_rag
from app.history import add_entry


def _mode_card(label: str, result: dict, show_chunks: bool = False):
    st.markdown(f"#### {label}")
    st.markdown(result["response"] if result.get("response") else "_No response_")
    st.caption(
        f" {result.get('latency_s', 0):.2f}s  "
        f"|  {result.get('tokens_out', 0)} tokens out"
    )
    if show_chunks and result.get("chunks"):
        with st.expander(f" Retrieved chunks ({len(result['chunks'])})"):
            for c in result["chunks"]:
                st.markdown(f"**Section {c['section_num']}** — {c['section_title']} (dist {c['distance']})")
                st.text(c["text"][:400] + "…")


def render(max_tokens: int, top_k: int):
    st.markdown(
        "Run the same question through **all three modes** side-by-side "
        "to see the effect of fine-tuning and RAG at a glance."
    )

    # Mode selector
    modes = st.multiselect(
        "Modes to compare",
        ["Base", "Fine-tuned", "Fine-tuned + RAG"],
        default=["Base", "Fine-tuned + RAG"],
    )
    if not modes:
        st.info("Select at least one mode.")
        return

    question = st.text_area(
        "Your question", height=100, key="cmp_q",
        placeholder="e.g. What is the basic exemption limit for a resident individual?",
    )

    if st.button("Compare ▶", key="cmp_run", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        results = {}
        errors  = {}

        with st.spinner("Running selected modes…"):
            if "Base" in modes:
                try:
                    m, t = get_base_model()
                    results["Base"] = run_base(question.strip(), m, t, max_tokens)
                except Exception as e:
                    errors["Base"] = str(e)

            if "Fine-tuned" in modes:
                try:
                    m, t = get_finetuned_model()
                    results["Fine-tuned"] = run_finetuned(question.strip(), m, t, max_tokens)
                except Exception as e:
                    errors["Fine-tuned"] = str(e)

            if "Fine-tuned + RAG" in modes:
                try:
                    m, t  = get_finetuned_model()
                    col   = get_rag_collection()
                    results["Fine-tuned + RAG"] = run_rag(
                        question.strip(), m, t, col, k=top_k, max_new_tokens=max_tokens
                    )
                except Exception as e:
                    errors["Fine-tuned + RAG"] = str(e)

        if errors:
            for mode_name, err in errors.items():
                st.error(f"**{mode_name}**: {err}")

        if not results:
            return

        add_entry("compare", question.strip(), compare=results)

        # ── Render side-by-side columns ───────────────────────────────────
        cols = st.columns(len(results))
        for col, (mode_name, res) in zip(cols, results.items()):
            with col:
                _mode_card(
                    mode_name, res,
                    show_chunks=(mode_name == "Fine-tuned + RAG"),
                )

        # ── Latency summary bar ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("####  Latency comparison")
        lat_cols = st.columns(len(results))
        for col, (mode_name, res) in zip(lat_cols, results.items()):
            col.metric(mode_name, f"{res.get('latency_s', 0):.2f}s")
