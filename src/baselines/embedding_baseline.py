from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.rag.config import resolve_torch_device


class EmbeddingKNN:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        n_neighbors: int = 5,
        weights: str = "distance",
        metric: str = "cosine",
    ):
        self.model_name = model_name
        self.device = device
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model: SentenceTransformer | None = None
        self.knn: KNeighborsRegressor | None = None
        self._train_embeddings: np.ndarray | None = None

    def fit(
        self,
        df_train: pd.DataFrame,
        band_column: str = "overall",
        text_column: str = "essay",
    ) -> None:
        self.model = SentenceTransformer(
            self.model_name,
            device=resolve_torch_device(self.device),
        )
        texts = df_train[text_column].fillna("").astype(str).tolist()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        self._train_embeddings = np.array(embeddings).astype("float32")

        self.knn = KNeighborsRegressor(
            n_neighbors=min(self.n_neighbors, len(df_train)),
            weights=self.weights,
            metric=self.metric,
        )
        self.knn.fit(self._train_embeddings, df_train[band_column].values)

    def predict(
        self,
        df_test: pd.DataFrame,
        text_column: str = "essay",
        clip_range: tuple[float, float] = (0.0, 9.0),
    ) -> np.ndarray:
        if self.model is None or self.knn is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        texts = df_test[text_column].fillna("").astype(str).tolist()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        preds = self.knn.predict(embeddings)
        preds = np.clip(preds, clip_range[0], clip_range[1])
        return np.round(preds * 2) / 2


def train_embedding_lightgbm(
    df_train: pd.DataFrame,
    band_column: str = "overall",
    text_column: str = "essay",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "auto",
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
):
    try:
        import lightgbm as lgb
        backend = "lightgbm"
    except ImportError:
        lgb = None
        backend = "hist_gradient_boosting"

    model = SentenceTransformer(
        model_name,
        device=resolve_torch_device(device),
    )
    texts = df_train[text_column].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    X_train = np.array(embeddings).astype("float32")
    y_train = pd.to_numeric(df_train[band_column], errors="coerce").astype(float).values

    if backend == "lightgbm":
        regressor = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=42,
            verbose=-1,
        )
    else:
        regressor = HistGradientBoostingRegressor(
            max_depth=max_depth if max_depth and max_depth > 0 else None,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            random_state=42,
        )

    regressor.fit(X_train, y_train)
    setattr(regressor, "_backend_name", backend)
    return model, regressor


def predict_embedding_lightgbm(
    model: SentenceTransformer,
    lgb_model,
    df_test: pd.DataFrame,
    text_column: str = "essay",
    clip_range: tuple[float, float] = (0.0, 9.0),
) -> np.ndarray:
    texts = df_test[text_column].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    X_test = np.array(embeddings).astype("float32")
    preds = lgb_model.predict(X_test)
    preds = np.clip(preds, clip_range[0], clip_range[1])
    return np.round(preds * 2) / 2
