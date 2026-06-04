from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.baselines.distilbert_baseline import predict_distilbert, train_distilbert
from src.baselines.tfidf_baseline import train_tfidf_ridge
from src.experiments.run_benchmark import _build_reference_retriever, _try_load_generator
from src.rag.config import load_benchmark_config, resolve_path, resolve_torch_device
from src.rag.llm_scorer import LLMScorer
from src.utils.evaluation_metrics import clip_and_round


@dataclass
class Stage1Bundle:
    pipeline: Any
    vectorizer: Any
    train_matrix: Any
    train_scores: np.ndarray


@dataclass
class Stage2Bundle:
    model: Any
    tokenizer: Any
    device: str


@dataclass
class ScoreResult:
    final_score: float
    route: str
    stage1_score: float
    stage1_variance: float
    stage2_score: float | None = None
    llm_score: float | None = None
    llm_scores: dict[str, float] | None = None
    llm_error: str | None = None
    neighbors: list[Any] | None = None
    threshold_var: float | None = None
    threshold_delta: float | None = None


CASCADE_PRESETS = {
    "Lightweight": {"var": 1.0, "delta": 1.5},
    "Max accuracy": {"var": 0.75, "delta": 0.5},
    "Ultra-light": {"var": 2.0, "delta": 1.5},
}


def load_splits(split_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    config = load_benchmark_config()
    split_root = Path(split_dir) if split_dir else Path(resolve_path(config["splits"]["split_dir"]))
    return {
        "train": pd.read_csv(split_root / "train.csv"),
        "val": pd.read_csv(split_root / "val.csv"),
        "test": pd.read_csv(split_root / "test.csv"),
    }


def build_stage1(df_train: pd.DataFrame) -> Stage1Bundle:
    pipeline = train_tfidf_ridge(
        df_train,
        band_column="overall",
        text_column="essay",
        alpha=1.0,
        max_features=5000,
        ngram_range=(1, 2),
    )
    vectorizer = pipeline.named_steps["tfidf"]
    train_texts = df_train["essay"].fillna("").astype(str).tolist()
    train_matrix = vectorizer.transform(train_texts)
    train_scores = df_train["overall"].to_numpy(dtype=float)
    return Stage1Bundle(pipeline, vectorizer, train_matrix, train_scores)


def build_stage2(df_train: pd.DataFrame, device: str = "auto") -> Stage2Bundle:
    resolved_device = resolve_torch_device(device)
    model, tokenizer = train_distilbert(
        df_train,
        band_column="overall",
        text_column="essay",
        device=resolved_device,
    )
    return Stage2Bundle(model=model, tokenizer=tokenizer, device=resolved_device)


def stage1_variance(text: str, bundle: Stage1Bundle, k: int = 20) -> float:
    query = bundle.vectorizer.transform([text])
    sims = cosine_similarity(query, bundle.train_matrix).flatten()
    top_k = np.argsort(sims)[-k:][::-1]
    neighbor_scores = bundle.train_scores[top_k]
    weights = sims[top_k]
    if weights.sum() > 0:
        mean = np.average(neighbor_scores, weights=weights)
        variance = np.average((neighbor_scores - mean) ** 2, weights=weights)
    else:
        variance = np.var(neighbor_scores)
    return float(variance)


def predict_stage1(text: str, bundle: Stage1Bundle) -> float:
    raw_score = float(bundle.pipeline.predict([text])[0])
    return float(clip_and_round([raw_score])[0])


def predict_stage2(text: str, bundle: Stage2Bundle) -> float:
    df = pd.DataFrame({"essay": [text]})
    pred = predict_distilbert(
        bundle.model,
        bundle.tokenizer,
        df,
        text_column="essay",
        device=bundle.device,
    )
    return float(pred[0])


def paper_fusion(stage2_score: float, llm_score: float | None) -> float:
    if llm_score is None or np.isnan(llm_score):
        return float(round(stage2_score * 2) / 2)
    delta = abs(llm_score - stage2_score)
    if delta > 1.5:
        fused = stage2_score
    elif delta > 1.0:
        fused = 0.9 * stage2_score + 0.1 * llm_score
    else:
        fused = 0.7 * stage2_score + 0.3 * llm_score
    return float(round(fused * 2) / 2)


def build_retriever(df_train: pd.DataFrame):
    config = load_benchmark_config()
    _, retriever = _build_reference_retriever(df_train, config)
    return retriever


def load_llm_scorer() -> LLMScorer:
    config = load_benchmark_config()
    model_path = resolve_path(config["llm"]["model_name"])
    if str(config["llm"]["model_name"]).lower().endswith(".gguf") and not model_path.exists():
        return LLMScorer(generator=None, prompt_template=resolve_path(config["llm"]["prompt_template"]))

    generator = _try_load_generator(config)
    return LLMScorer(
        generator=generator,
        prompt_template=resolve_path(config["llm"]["prompt_template"]),
        max_retries=config["llm"]["max_retries"],
        max_new_tokens=config["llm"].get("max_new_tokens", 80),
    )


def score_essay(
    essay: str,
    prompt: str | None,
    mode: str,
    stage1: Stage1Bundle,
    stage2: Stage2Bundle | None = None,
    retriever: Any | None = None,
    llm_scorer: LLMScorer | None = None,
) -> ScoreResult:
    text = essay.strip()
    s1 = predict_stage1(text, stage1)
    variance = stage1_variance(text, stage1)

    if mode == "Stage 1 only":
        return ScoreResult(
            final_score=s1,
            route="Stage 1 accepted",
            stage1_score=s1,
            stage1_variance=variance,
        )

    preset = CASCADE_PRESETS[mode]
    var_threshold = preset["var"]
    delta_threshold = preset["delta"]

    if variance < var_threshold:
        return ScoreResult(
            final_score=s1,
            route="Stage 1 accepted",
            stage1_score=s1,
            stage1_variance=variance,
            threshold_var=var_threshold,
            threshold_delta=delta_threshold,
        )

    if stage2 is None:
        return ScoreResult(
            final_score=s1,
            route="Stage 2 unavailable; returned Stage 1 fallback",
            stage1_score=s1,
            stage1_variance=variance,
            threshold_var=var_threshold,
            threshold_delta=delta_threshold,
        )

    s2 = predict_stage2(text, stage2)
    if abs(s2 - s1) <= delta_threshold:
        return ScoreResult(
            final_score=s2,
            route="Stage 2 accepted",
            stage1_score=s1,
            stage1_variance=variance,
            stage2_score=s2,
            threshold_var=var_threshold,
            threshold_delta=delta_threshold,
        )

    if retriever is None or llm_scorer is None or not llm_scorer.is_available():
        return ScoreResult(
            final_score=s2,
            route="Stage 3 unavailable; returned Stage 2 fallback",
            stage1_score=s1,
            stage1_variance=variance,
            stage2_score=s2,
            threshold_var=var_threshold,
            threshold_delta=delta_threshold,
            llm_error="Local LLM or retriever is not available.",
        )

    neighbors = retriever.retrieve(
        text,
        top_k=7,
        exclude_row_index=None,
        exclude_hashes=None,
        prompt_text=prompt or None,
    )
    result = llm_scorer.score(text, neighbors, distilbert_scores={"overall": s2})
    llm_scores = result.get("llm_scores")
    llm_score = llm_scores.get("overall") if llm_scores else None
    final = paper_fusion(s2, llm_score)
    return ScoreResult(
        final_score=final,
        route="Stage 3 fused with DistilBERT",
        stage1_score=s1,
        stage1_variance=variance,
        stage2_score=s2,
        llm_score=llm_score,
        llm_scores=llm_scores,
        llm_error=result.get("error"),
        neighbors=neighbors,
        threshold_var=var_threshold,
        threshold_delta=delta_threshold,
    )
