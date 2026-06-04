from __future__ import annotations

import numpy as np
import pandas as pd


def fit_mean(df_train: pd.DataFrame, band_column: str = "overall") -> float:
    return float(df_train[band_column].mean())


def fit_median(df_train: pd.DataFrame, band_column: str = "overall") -> float:
    return float(df_train[band_column].median())


def predict_constant(df_test: pd.DataFrame, constant: float) -> np.ndarray:
    return np.full(len(df_test), round(constant * 2) / 2)


def evaluate_mean_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    band_column: str = "overall",
) -> np.ndarray:
    mean_val = fit_mean(df_train, band_column)
    return predict_constant(df_test, mean_val)


def evaluate_median_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    band_column: str = "overall",
) -> np.ndarray:
    median_val = fit_median(df_train, band_column)
    return predict_constant(df_test, median_val)
