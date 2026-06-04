from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def threshold_gating(
    retrieval_score: float,
    llm_score: float | None,
    tau: float = 1.5,
    alpha: float = 0.7,
) -> float:
    if llm_score is None:
        return retrieval_score

    diff = abs(llm_score - retrieval_score)

    if diff > tau:
        return retrieval_score

    combined = alpha * retrieval_score + (1.0 - alpha) * llm_score
    return round(combined * 2) / 2


class LearnedFusion:
    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        if model_path:
            self.load(model_path)

    def load(self, model_path: str | Path) -> None:
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def save(self, model_path: str | Path) -> None:
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def is_ready(self) -> bool:
        return self.model is not None

    def _build_feature_vector(self, features: dict[str, float]) -> np.ndarray:
        keys = [
            "retrieval_score",
            "llm_score",
            "absolute_disagreement",
            "top_k_similarity_mean",
            "top_k_similarity_variance",
            "neighbor_score_variance",
            "neighbor_score_mean",
            "essay_length",
            "llm_valid",
        ]
        vec = []
        for k in keys:
            vec.append(features.get(k, 0.0))
        return np.array(vec).reshape(1, -1)

    def predict(self, features: dict[str, float]) -> float:
        if self.model is None:
            raise RuntimeError("LearnedFusion model not loaded. Call load() first or fall back.")
        X = self._build_feature_vector(features)
        pred = self.model.predict(X)[0]
        return round(float(pred) * 2) / 2

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)
        self.model.fit(X, y)

    def extract_features(
        self,
        retrieval_score: float,
        llm_score: float | None,
        retrieval_features: dict[str, float],
        llm_valid: bool = True,
    ) -> dict[str, float]:
        features = dict(retrieval_features)
        features["llm_score"] = llm_score if llm_score is not None else retrieval_score
        features["absolute_disagreement"] = (
            abs(llm_score - retrieval_score) if llm_score is not None else 0.0
        )
        features["llm_valid"] = 1.0 if llm_valid else 0.0
        return features


def fuse(
    retrieval_score: float,
    llm_score: float | None,
    strategy: str = "threshold_gating",
    retrieval_features: dict[str, float] | None = None,
    fusion_model: LearnedFusion | None = None,
    tau: float = 1.5,
    alpha: float = 0.7,
    llm_valid: bool = True,
) -> float:
    if strategy == "retrieval_only":
        return round(retrieval_score * 2) / 2

    if strategy == "llm_only":
        if llm_score is not None:
            return round(llm_score * 2) / 2
        raise ValueError("llm_only fusion requires a valid LLM score")

    if strategy == "threshold_gating":
        return threshold_gating(retrieval_score, llm_score, tau=tau, alpha=alpha)

    if strategy == "learned_fusion":
        if fusion_model is not None and fusion_model.is_ready() and retrieval_features is not None:
            features = fusion_model.extract_features(
                retrieval_score=retrieval_score,
                llm_score=llm_score,
                retrieval_features=retrieval_features,
                llm_valid=llm_valid,
            )
            return fusion_model.predict(features)
        return threshold_gating(retrieval_score, llm_score, tau=tau, alpha=alpha)

    warnings.warn(f"Unknown fusion strategy '{strategy}'. Falling back to retrieval_only.")
    return round(retrieval_score * 2) / 2
