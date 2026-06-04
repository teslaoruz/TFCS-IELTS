from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference.ui_pipeline import (
    CASCADE_PRESETS,
    build_retriever,
    build_stage1,
    load_llm_scorer,
    load_splits,
    score_essay,
)
from src.rag.config import load_benchmark_config, resolve_path


st.set_page_config(
    page_title="TFCS-IELTS Scorer",
    page_icon="TF",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_splits() -> dict[str, pd.DataFrame]:
    return load_splits()


@st.cache_resource(show_spinner="Preparing TF-IDF Ridge stage...")
def cached_stage1(train_df: pd.DataFrame):
    return build_stage1(train_df)


@st.cache_resource(show_spinner="Building FAISS retriever...")
def cached_retriever(train_df: pd.DataFrame):
    return build_retriever(train_df)


@st.cache_resource(show_spinner="Loading local Qwen GGUF model...")
def cached_llm_scorer():
    return load_llm_scorer()


def word_count(text: str) -> int:
    return len(text.split())


def model_path_status() -> tuple[Path, bool]:
    config = load_benchmark_config()
    model_path = resolve_path(config["llm"]["model_name"])
    return model_path, model_path.exists()


st.title("TFCS-IELTS Offline Writing Scorer")

splits = cached_splits()
df_train = splits["train"]

with st.sidebar:
    st.header("Scoring")
    mode = st.selectbox(
        "Mode",
        ["Stage 1 only", "Lightweight", "Max accuracy", "Ultra-light"],
        index=1,
    )
    
    use_stage2 = st.checkbox("Enable DistilBERT Stage 2", value=True)
    use_stage3 = st.checkbox("Enable local Qwen Stage 3", value=True)
    device = st.selectbox("Torch device", ["auto", "cpu", "cuda:0"], index=0)

    model_path, has_model = model_path_status()
    st.divider()
    st.caption(f"Training split: {len(df_train):,} essays")
    st.caption(f"Qwen model: {'found' if has_model else 'missing'}")
    st.caption(str(model_path))

    if mode in CASCADE_PRESETS:
        preset = CASCADE_PRESETS[mode]
        st.caption(f"var < {preset['var']} / delta <= {preset['delta']}")

prompt = st.text_area(
    "IELTS Task 2 prompt",
    placeholder="Paste the essay question here...",
    height=90,
)

essay = st.text_area(
    "Student essay",
    placeholder="Paste the IELTS Writing Task 2 essay here...",
    height=280,
)

left, right = st.columns([1, 2])
with left:
    st.metric("Words", word_count(essay))
with right:
    st.caption("Stage 1 is fast. DistilBERT and Qwen are loaded only when enabled in the sidebar.")

score_clicked = st.button("Score essay", type="primary", disabled=not essay.strip())

if score_clicked:
    if word_count(essay) < 80 or word_count(essay) > 500:
        st.warning("The paper pipeline was validated on essays between 80 and 500 words.")

    stage1 = cached_stage1(df_train)

    retriever = None
    llm_scorer = None
    if mode != "Stage 1 only" and use_stage3:
        if has_model:
            retriever = cached_retriever(df_train)
            llm_scorer = cached_llm_scorer()
        else:
            st.info("Qwen GGUF model is missing, so Stage 3 will fall back when needed.")

    result = score_essay(
        essay=essay,
        prompt=prompt,
        mode=mode,
        stage1=stage1,
        df_train=df_train if use_stage2 else None,
        device=device,
        retriever=retriever,
        llm_scorer=llm_scorer,
    )

    st.divider()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Final band", f"{result.final_score:.1f}")
    metric_cols[1].metric("Stage 1", f"{result.stage1_score:.1f}")
    metric_cols[2].metric("Variance", f"{result.stage1_variance:.3f}")
    metric_cols[3].metric("Stage 2", "n/a" if result.stage2_score is None else f"{result.stage2_score:.1f}")

    st.subheader(result.route)

    details = {
        "final_score": result.final_score,
        "route": result.route,
        "stage1_score": result.stage1_score,
        "stage1_variance": result.stage1_variance,
        "stage2_score": result.stage2_score,
        "llm_score": result.llm_score,
        "llm_scores": result.llm_scores,
        "llm_error": result.llm_error,
        "var_threshold": result.threshold_var,
        "delta_threshold": result.threshold_delta,
    }
    st.download_button(
        "Download JSON result",
        data=json.dumps(details, indent=2),
        file_name="tfcs_ielts_score.json",
        mime="application/json",
    )

    with st.expander("Score details", expanded=True):
        st.json(details)

    if result.neighbors:
        with st.expander("Retrieved calibration essays"):
            for neighbor in result.neighbors[:3]:
                st.markdown(f"**Band {neighbor.band:.1f}**")
                st.write(neighbor.essay[:900] + ("..." if len(neighbor.essay) > 900 else ""))