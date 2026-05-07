"""
app/tab_finetuned.py — Fine-tuned Model tab
"""
import streamlit as st
from app.model_loader import get_finetuned_model
from app.inference import run_finetuned
from app.history import add_entry


def render(max_tokens: int):
    st.markdown(
        "Query the **fine-tuned** model (`finetune/final_model`) without RAG. "
        "Same prompt format as base — lets you isolate the effect of fine-tuning alone."
    )

    question = st.text_area(
        "Your question", height=100, key="ft_q",
        placeholder="e.g. What are the TDS provisions under section 194?",
    )

    if st.button("Run ", key="ft_run", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Loading fine-tuned model and generating…"):
            try:
                model, tokenizer = get_finetuned_model()
                result = run_finetuned(question.strip(), model, tokenizer, max_tokens)
            except FileNotFoundError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Error: {e}")
                return

        add_entry("finetuned", question.strip(), result)

        st.markdown("### Answer")
        st.markdown(result["response"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Latency",    f"{result['latency_s']:.2f}s")
        c2.metric("Tokens in",  result["tokens_in"])
        c3.metric("Tokens out", result["tokens_out"])
