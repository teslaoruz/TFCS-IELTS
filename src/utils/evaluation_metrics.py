from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score


def compute_mae(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def compute_rmse(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def _to_ordinal(scores: np.ndarray) -> np.ndarray:
    return np.round(scores * 2).astype(int)


def _has_zero_variance(values: list[float] | np.ndarray) -> bool:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return True
    return bool(np.nanstd(arr) < 1e-12)


def compute_qwk(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    y_t = _to_ordinal(np.array(y_true))
    y_p = _to_ordinal(np.array(y_pred))
    return float(cohen_kappa_score(y_t, y_p, weights="quadratic"))


def compute_exact_accuracy(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    return float(np.mean(np.abs(y_t - y_p) < 0.01))


def compute_within_05(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    return float(np.mean(np.abs(y_t - y_p) <= 0.5 + 1e-9))


def compute_within_10(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    return float(np.mean(np.abs(y_t - y_p) <= 1.0 + 1e-9))


def compute_pearson(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    if _has_zero_variance(y_true) or _has_zero_variance(y_pred):
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_spearman(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    from scipy.stats import spearmanr
    if _has_zero_variance(y_true) or _has_zero_variance(y_pred):
        return float("nan")
    return float(spearmanr(y_true, y_pred)[0])


def compute_all_metrics(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
) -> dict[str, float]:
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)

    metrics = {
        "mae": compute_mae(y_t, y_p),
        "rmse": compute_rmse(y_t, y_p),
        "qwk": compute_qwk(y_t, y_p),
        "exact_accuracy": compute_exact_accuracy(y_t, y_p),
        "within_0.5": compute_within_05(y_t, y_p),
        "within_1.0": compute_within_10(y_t, y_p),
        "pearson": compute_pearson(y_t, y_p),
    }

    try:
        metrics["spearman"] = compute_spearman(y_t, y_p)
    except Exception:
        metrics["spearman"] = float("nan")

    return metrics


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    if n_bootstrap <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            if math.isfinite(score):
                scores.append(score)
        except Exception:
            continue
    if not scores:
        return (float("nan"), float("nan"))
    scores = np.array(scores)
    alpha = 1.0 - ci
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return (lower, upper)


def compute_all_metrics_with_ci(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
    n_bootstrap: int = 1000,
) -> dict[str, dict[str, float]]:
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)

    metric_fns = {
        "mae": compute_mae,
        "rmse": compute_rmse,
        "qwk": compute_qwk,
        "within_0.5": compute_within_05,
        "within_1.0": compute_within_10,
        "pearson": compute_pearson,
    }

    results = {}
    for name, fn in metric_fns.items():
        val = fn(y_t, y_p)
        lo, hi = bootstrap_confidence_interval(y_t, y_p, fn, n_bootstrap=n_bootstrap)
        results[name] = {"value": val, "ci_lower": lo, "ci_upper": hi}
    return results


def clip_and_round(
    scores: list[float] | np.ndarray,
    min_score: float = 0.0,
    max_score: float = 9.0,
) -> np.ndarray:
    arr = np.clip(np.array(scores, dtype=float), min_score, max_score)
    return np.round(arr * 2) / 2
