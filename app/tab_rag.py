"""
app/tab_rag.py — Fine-tuned + RAG tab
"""
import streamlit as st
from app.model_loader import get_finetuned_model, get_rag_collection
from app.inference import run_rag
from app.history import add_entry


def render(max_tokens: int, top_k: int):
    st.markdown(
        "The **recommended production mode**: the fine-tuned model answers "
        "using live-retrieved context from the ChromaDB vector index."
    )

    question = st.text_area(
        "Your question", height=100, key="rag_q",
        placeholder="e.g. What is the penalty for non-filing under section 271F?",
    )

    if st.button("Run ", key="rag_run", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving + generating…"):
            try:
                model, tokenizer = get_finetuned_model()
                collection       = get_rag_collection()
                result = run_rag(
                    question.strip(), model, tokenizer, collection,
                    k=top_k, max_new_tokens=max_tokens,
                )
            except FileNotFoundError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Error: {e}")
                return

        add_entry("rag", question.strip(), result)

        st.markdown("### Answer")
        st.markdown(result["response"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency",       f"{result['latency_s']:.2f}s")
        c2.metric("Retrieval",     f"{result['retrieval_ms']:.0f}ms")
        c3.metric("Tokens in",     result["tokens_in"])
        c4.metric("Tokens out",    result["tokens_out"])

        st.markdown("---")
        st.markdown(f"###  Retrieved Context  ({len(result['chunks'])} chunks)")
        for i, chunk in enumerate(result["chunks"], 1):
            with st.expander(
                f"Chunk {i} — Section {chunk['section_num']} "
                f"| {chunk['section_title']}  "
                f"(distance: {chunk['distance']})",
                expanded=(i == 1),
            ):
                st.text(chunk["text"])
