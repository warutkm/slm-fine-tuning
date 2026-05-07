"""
app/tab_base.py — Base Model tab
"""
import streamlit as st
from app.model_loader import get_base_model
from app.inference import run_base
from app.history import add_entry


def render(max_tokens: int):
    st.markdown(
        "Query the **un-fine-tuned** LLaMA 3.2-1B-Instruct model with no RAG. "
        "This is your pre-training baseline."
    )

    question = st.text_area(
        "Your question", height=100, key="base_q",
        placeholder="e.g. What is the deduction limit under section 80C?",
    )

    if st.button("Run ", key="base_run", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Loading model and generating…"):
            try:
                model, tokenizer = get_base_model()
                result = run_base(question.strip(), model, tokenizer, max_tokens)
            except Exception as e:
                st.error(f"Error: {e}")
                return

        add_entry("base", question.strip(), result)

        st.markdown("### Answer")
        st.markdown(result["response"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Latency", f"{result['latency_s']:.2f}s")
        c2.metric("Tokens in",  result["tokens_in"])
        c3.metric("Tokens out", result["tokens_out"])
