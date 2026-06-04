from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedEssay:
    row_index: int
    rank: int
    distance: float
    similarity: float
    band: float
    essay: str
    band_category: str
    source: str | None = None
    prompt: str | None = None


class Retriever:
    """Configurable FAISS-based retriever for IELTS essays."""

    def __init__(
        self,
        df: pd.DataFrame,
        index,
        metadata: Any,
        embed_model: SentenceTransformer,
        band_column: str = "overall",
        prompt_aware: bool = False,
        diverse_exemplars: bool = False,
        max_distance: float | None = None,
    ):
        self.df = df
        self.index = index
        self.metadata = metadata
        self.embed_model = embed_model
        self.band_column = band_column
        self.prompt_aware = prompt_aware
        self.diverse_exemplars = diverse_exemplars
        self.max_distance = max_distance

    @staticmethod
    def _band_bucket(band: float) -> str:
        if band >= 7.0:
            return "high"
        if band >= 5.0:
            return "mid"
        return "low"

    def _map_to_row_index(self, faiss_index: int) -> int:
        if self.metadata is None:
            return faiss_index
        if isinstance(self.metadata, dict):
            row_map = self.metadata.get("row_map")
            if isinstance(row_map, list) and 0 <= faiss_index < len(row_map):
                candidate = row_map[faiss_index]
                if isinstance(candidate, (int, np.integer)):
                    return int(candidate)
            return faiss_index
        if isinstance(self.metadata, list):
            if 0 <= faiss_index < len(self.metadata):
                candidate = self.metadata[faiss_index]
                if isinstance(candidate, (int, np.integer)):
                    return int(candidate)
            return faiss_index
        return faiss_index

    def _resolve_prompt(self, row: pd.Series) -> str | None:
        for col in ("prompt", "question", "task"):
            if col in row and pd.notna(row[col]):
                return str(row[col])
        return None

    def retrieve(
        self,
        essay_text: str,
        top_k: int = 5,
        exclude_row_index: int | None = None,
        exclude_hashes: set[str] | None = None,
        prompt_text: str | None = None,
    ) -> list[RetrievedEssay]:
        query = essay_text
        if self.prompt_aware and prompt_text:
            query = f"{prompt_text} {essay_text}"

        query_emb = self.embed_model.encode([query]).astype("float32")
        if getattr(self.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(query_emb)

        search_k = min(max(top_k * 4, top_k), len(self.df))
        distances, indices = self.index.search(query_emb, search_k)

        all_neighbors: list[RetrievedEssay] = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if self.max_distance is not None and distance > self.max_distance:
                continue

            row_index = self._map_to_row_index(int(idx))
            if row_index < 0 or row_index >= len(self.df):
                continue

            if exclude_row_index is not None and row_index == exclude_row_index:
                continue

            row = self.df.iloc[row_index]
            row_hash = row.get("normalized_text_hash")
            if exclude_hashes and pd.notna(row_hash) and str(row_hash) in exclude_hashes:
                continue

            band_value = row.get(self.band_column)
            if pd.isna(band_value):
                continue

            band = float(band_value)
            similarity = float(1.0 / (1.0 + distance))
            all_neighbors.append(
                RetrievedEssay(
                    row_index=row_index,
                    rank=rank,
                    distance=float(distance),
                    similarity=similarity,
                    band=band,
                    essay=str(row.get("essay", "")),
                    band_category=self._band_bucket(band),
                    source=str(row.get("source", "")) if "source" in row else None,
                    prompt=self._resolve_prompt(row),
                )
            )

        if not all_neighbors:
            return []

        if self.diverse_exemplars and len(all_neighbors) >= 3:
            return self._select_diverse(all_neighbors, top_k)

        return all_neighbors[:top_k]

    def _select_diverse(
        self, neighbors: list[RetrievedEssay], top_k: int
    ) -> list[RetrievedEssay]:
        if not neighbors:
            return []

        query_band = neighbors[0].band if neighbors else 5.0

        selected: list[RetrievedEssay] = []
        seen_indices: set[int] = set()

        def _add_if_unique(n: RetrievedEssay) -> bool:
            if n.row_index not in seen_indices:
                selected.append(n)
                seen_indices.add(n.row_index)
                return True
            return False

        # 1. Same-level essay (closest to query band, within 0.5)
        same_level = [n for n in neighbors if abs(n.band - query_band) <= 0.5]
        if same_level:
            _add_if_unique(same_level[0])

        # 2. Slightly lower-band essay (0.5-1.5 below query)
        lower = [n for n in neighbors if query_band - n.band >= 0.5]
        if lower:
            _add_if_unique(lower[0])

        # 3. Slightly higher-band essay (0.5-1.5 above query)
        higher = [n for n in neighbors if n.band - query_band >= 0.5]
        if higher:
            _add_if_unique(higher[0])

        # 4. Fill remaining slots by proximity
        for n in neighbors:
            if len(selected) >= top_k:
                break
            _add_if_unique(n)

        return selected[:top_k]

    def predict_similarity_weighted(self, neighbors: list[RetrievedEssay]) -> float:
        if not neighbors:
            return 5.0
        similarities = np.array([n.similarity for n in neighbors], dtype=np.float32)
        bands = np.array([n.band for n in neighbors], dtype=np.float32)
        if similarities.sum() == 0:
            return float(np.mean(bands))
        weighted_avg = float(np.average(bands, weights=similarities))
        return round(weighted_avg * 2) / 2

    def get_retrieval_features(
        self, neighbors: list[RetrievedEssay], query_length: int | None = None
    ) -> dict[str, float]:
        if not neighbors:
            return {
                "retrieval_score": 5.0,
                "top_k_similarity_mean": 0.0,
                "top_k_similarity_variance": 0.0,
                "neighbor_score_variance": 0.0,
                "neighbor_score_mean": 5.0,
                "essay_length": float(query_length or 0),
            }

        similarities = [n.similarity for n in neighbors]
        bands = [n.band for n in neighbors]

        return {
            "retrieval_score": self.predict_similarity_weighted(neighbors),
            "top_k_similarity_mean": float(np.mean(similarities)),
            "top_k_similarity_variance": float(np.var(similarities)) if len(similarities) > 1 else 0.0,
            "neighbor_score_variance": float(np.var(bands)) if len(bands) > 1 else 0.0,
            "neighbor_score_mean": float(np.mean(bands)),
            "essay_length": float(query_length or 0),
        }
