"""
app.py 
Run from the project root (lexitune/):
    streamlit run app.py

Tabs :-
  1. Base Model          — baseline LLaMA inference
  2. Fine-tuned Model    — SFT model, no RAG
  3. Fine-tuned + RAG    — recommended production mode
  4. Compare             — side-by-side all modes
  5. History             — query log for this session
"""

import streamlit as st

#  Page config (must be first Streamlit call) 
st.set_page_config(
    page_title="LexiTune — ITA Legal Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Sidebar 
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/scales.png", width=60
    )
    st.title("LexiTune")
    st.caption("Domain-specific AI for Indian Income Tax Act, 2025")
    st.divider()

    st.markdown("### Generation Settings")
    max_tokens = st.slider("Max new tokens", 128, 1024, 512, step=64)
    top_k      = st.slider("RAG top-k chunks", 2, 10, 6)
    st.divider()

    # Device info
    from app.model_loader import device_info
    st.markdown("###  Compute")
    st.info(device_info())
    st.divider()

    # History
    from app.history import render_history_sidebar
    render_history_sidebar()

#  Main tabs 
st.markdown("##  LexiTune — Income Tax Act Assistant")
st.caption(
    "Fine-tuned LLaMA 3.2-1B-Instruct on ITA 2025 · "
    "RAG via ChromaDB · "
    "LLM-as-judge evaluation"
)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⁂ Base Model",
    "⁂ Fine-tuned",
    "⁂ Fine-tuned + RAG",
    "⁂ Compare",
    "⁂ History",
])

from app import tab_base, tab_finetuned, tab_rag, tab_compare

with tab1:
    tab_base.render(max_tokens)

with tab2:
    tab_finetuned.render(max_tokens)

with tab3:
    tab_rag.render(max_tokens, top_k)

with tab4:
    tab_compare.render(max_tokens, top_k)

with tab5:
    from app.history import get_history
    import json, pandas as pd
    history = get_history()
    if not history:
        st.info("No queries yet. Run a question from any tab.")
    else:
        st.markdown(f"**{len(history)} queries this session**")
        rows = []
        for e in history:
            rows.append({
                "Time":    e["ts"],
                "Mode":    e["mode"],
                "Question": e["question"][:80],
                "Latency": e.get("latency_s", "—"),
                "Tokens":  e.get("tokens_out", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if st.button(" Download as JSONL"):
            blob = "\n".join(json.dumps(e, ensure_ascii=False) for e in history)
            st.download_button(
                "Download history.jsonl",
                data=blob,
                file_name="query_history.jsonl",
                mime="application/jsonl",
            )
