from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


HF_DATASET_ID = "chillies/IELTS-writing-task-2-evaluation"
EXPECTED_COLUMNS = [
    "essay", "prompt", "task_type", "overall",
    "task_response", "coherence", "lexical", "grammar",
    "feedback", "source",
]


def normalize_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = str(value).strip()
        if val.startswith("<"):
            val = val[1:]
        score = float(val)
        if score < 0 or score > 9:
            return None
        return round(score * 2) / 2
    except (ValueError, TypeError):
        return None


def load_hf_dataset(
    split: str = "train",
    sample: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if not _HF_AVAILABLE:
        raise ImportError(
            "The `datasets` library is required to load HuggingFace datasets. "
            "Install it with: pip install datasets"
        )
    ds = load_dataset(HF_DATASET_ID, split=split)
    df = ds.to_pandas()
    _normalize_hf_dataframe(df)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
    return df


def load_hf_dataset_full() -> dict[str, pd.DataFrame]:
    if not _HF_AVAILABLE:
        raise ImportError("The `datasets` library is required.")
    dataset = load_dataset(HF_DATASET_ID)
    result = {}
    for split_name in dataset:
        df = dataset[split_name].to_pandas()
        _normalize_hf_dataframe(df)
        result[split_name] = df
    return result


def load_hf_fallback_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        return load_hf_fallback_directory(path)
    if not path.exists():
        raise FileNotFoundError(
            f"HF dataset fallback CSV not found at {path}. "
            f"To use this, download the dataset manually from:\n"
            f"  https://huggingface.co/datasets/{HF_DATASET_ID}\n"
            f"and save as CSV at: {path}"
        )
    df = pd.read_csv(path)
    _normalize_hf_dataframe(df)
    return df


def load_hf_fallback_directory(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"HF dataset fallback directory not found at {path}. "
            f"Expected a folder containing split files such as train/test CSVs."
        )

    candidates = [
        path / "train.csv",
        path / "test.csv",
        path / "validation.csv",
        path / "val.csv",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        raise FileNotFoundError(
            f"No supported split CSVs found in {path}. "
            f"Expected one or more of: {[p.name for p in candidates]}"
        )

    frames: list[pd.DataFrame] = []
    for csv_path in found:
        df = pd.read_csv(csv_path)
        split_name = csv_path.stem.lower()
        df["split"] = split_name
        _normalize_hf_dataframe(df)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _normalize_hf_dataframe(df: pd.DataFrame) -> None:
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    score_cols = ["band", "overall", "task_response", "coherence", "lexical", "grammar"]
    for col in score_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_score)

    if "overall" in df.columns and df["overall"].isna().all() and "band" in df.columns:
        df["overall"] = df["band"]

    if "source" not in df.columns or df["source"].isna().all():
        df["source"] = "huggingface"

    if "task_type" not in df.columns or df["task_type"].isna().all():
        df["task_type"] = "task2"

    if "prompt" in df.columns:
        df.rename(columns={"prompt": "question"}, inplace=True)
    elif "question" not in df.columns:
        df["question"] = None

    if "id" not in df.columns:
        df["id"] = [f"hf_{i}" for i in range(len(df))]
