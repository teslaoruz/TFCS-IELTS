from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


def train_tfidf_ridge(
    df_train: pd.DataFrame,
    band_column: str = "overall",
    text_column: str = "essay",
    alpha: float = 1.0,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> Pipeline:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ("ridge", Ridge(alpha=alpha)),
    ])
    pipeline.fit(df_train[text_column].values, df_train[band_column].values)
    return pipeline


def train_tfidf_svr(
    df_train: pd.DataFrame,
    band_column: str = "overall",
    text_column: str = "essay",
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> Pipeline:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ("svr", SVR(kernel=kernel, C=C, gamma=gamma)),
    ])
    pipeline.fit(df_train[text_column].values, df_train[band_column].values)
    return pipeline


def predict_tfidf(
    pipeline: Pipeline,
    df_test: pd.DataFrame,
    text_column: str = "essay",
    clip_range: tuple[float, float] = (0.0, 9.0),
) -> np.ndarray:
    preds = pipeline.predict(df_test[text_column].values)
    preds = np.clip(preds, clip_range[0], clip_range[1])
    return np.round(preds * 2) / 2
