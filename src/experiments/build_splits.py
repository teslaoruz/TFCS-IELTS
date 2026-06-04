from __future__ import annotations

import argparse
import hashlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.rag.config import load_benchmark_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic train/val/test splits")
    parser.add_argument("--config", default=None, help="Path to benchmark config YAML")
    parser.add_argument("--input", default=None, help="Path to input CSV (overrides config)")
    parser.add_argument("--output-dir", default=None, help="Output directory for split CSVs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--ratio", type=float, nargs=2, default=None, metavar=("TRAIN", "VAL"),
                        help="Train/val ratio (test inferred)")
    return parser.parse_args()


def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_configured_dataset(ds_config: dict[str, Any]) -> pd.DataFrame:
    local_csv = resolve_path(ds_config["local_csv"])
    prefer_hf = bool(ds_config.get("prefer_hf", False))
    hf_dataset_id = ds_config.get("hf_dataset")
    hf_fallback = ds_config.get("hf_local_fallback")

    def _try_load_hf() -> pd.DataFrame:
        from src.data.hf_dataset import load_hf_dataset_full, load_hf_fallback_csv

        if hf_fallback:
            fallback_path = resolve_path(hf_fallback)
            if fallback_path.exists():
                warnings.warn(
                    f"Using local Hugging Face dataset fallback at {fallback_path}",
                    RuntimeWarning,
                )
                return load_hf_fallback_csv(fallback_path)

        try:
            dataset_splits = load_hf_dataset_full()
            frames = []
            for split_name, split_df in dataset_splits.items():
                split_copy = split_df.copy()
                split_copy["split"] = split_name
                frames.append(split_copy)
            return pd.concat(frames, ignore_index=True)
        except Exception as exc:
            if hf_fallback:
                fallback_path = resolve_path(hf_fallback)
                if fallback_path.exists():
                    warnings.warn(
                        f"HF dataset load failed ({exc}). Falling back to local HF data: {fallback_path}",
                        RuntimeWarning,
                    )
                    return load_hf_fallback_csv(fallback_path)
            raise

    if prefer_hf and hf_dataset_id:
        try:
            return _try_load_hf()
        except Exception as exc:
            if local_csv.exists():
                warnings.warn(
                    f"HF dataset load failed ({exc}). Falling back to local CSV: {local_csv}",
                    RuntimeWarning,
                )
                return pd.read_csv(local_csv)
            raise

    if local_csv.exists():
        return pd.read_csv(local_csv)

    if hf_dataset_id:
        return _try_load_hf()

    raise FileNotFoundError(
        f"No dataset source available. Local CSV not found: {local_csv}"
    )


def normalize_dataframe(
    df: pd.DataFrame,
    score_columns: list[str] | None = None,
    min_words: int = 80,
    max_words: int = 500,
    deduplicate: bool = True,
) -> pd.DataFrame:
    if score_columns is None:
        score_columns = ["overall", "band", "score"]

    cols_lower = {c.lower(): c for c in df.columns}

    essay_col = None
    for candidate in ("essay", "text", "essay_text"):
        if candidate in cols_lower:
            essay_col = cols_lower[candidate]
            break
    if essay_col is None:
        raise ValueError("No essay column found in dataset")

    score_col = None
    best_non_null = -1

    def _candidate_non_null_count(column_name: str) -> int:
        converted = pd.to_numeric(df[column_name], errors="coerce")
        return int(converted.notna().sum())

    for candidate in score_columns:
        if candidate in cols_lower:
            column_name = cols_lower[candidate]
            non_null = _candidate_non_null_count(column_name)
            if non_null > best_non_null:
                best_non_null = non_null
                score_col = column_name

    if score_col is None or best_non_null <= 0:
        for col in df.columns:
            lowered = col.lower()
            if "overall" in lowered or "band" in lowered or "score" in lowered:
                non_null = _candidate_non_null_count(col)
                if non_null > best_non_null:
                    best_non_null = non_null
                    score_col = col
    if score_col is None:
        raise ValueError(f"No score column found. Tried: {score_columns}")

    result = pd.DataFrame()
    result["essay"] = df[essay_col].astype(str)
    result["overall"] = pd.to_numeric(df[score_col], errors="coerce")

    prompt_col = None
    for candidate in ("prompt", "question", "task", "topic"):
        if candidate in cols_lower:
            prompt_col = cols_lower[candidate]
            break
    if prompt_col:
        result["question"] = df[prompt_col]

    task_col = None
    for candidate in ("task_type", "task"):
        if candidate in cols_lower:
            task_col = cols_lower[candidate]
            break
    if task_col:
        result["task_type"] = df[task_col].astype(str)
    else:
        result["task_type"] = "task2"

    source_col = None
    for candidate in ("source", "dataset", "origin"):
        if candidate in cols_lower:
            source_col = cols_lower[candidate]
            break
    if source_col:
        result["source"] = df[source_col].astype(str)
    else:
        result["source"] = "unknown"

    if "split" in cols_lower:
        result["split"] = df[cols_lower["split"]].astype(str)

    for sub_score in ("task_response", "coherence", "lexical", "grammar"):
        lower_key = sub_score.lower()
        if lower_key in cols_lower:
            result[sub_score] = pd.to_numeric(df[cols_lower[lower_key]], errors="coerce")

    evaluation_col = None
    for candidate in ("evaluation", "feedback"):
        if candidate in cols_lower:
            evaluation_col = cols_lower[candidate]
            break
    if evaluation_col is not None:
        missing_subscores = [s for s in ("task_response", "coherence", "lexical", "grammar") if s not in result.columns or result[s].isna().all()]
        if missing_subscores:
            from src.data.evaluation_parser import parse_subscores
            parsed_series = df[evaluation_col].apply(parse_subscores)
            for s in missing_subscores:
                parsed_vals = [p.get(s) for p in parsed_series]
                non_null = sum(1 for v in parsed_vals if v is not None)
                if non_null > 0:
                    result[s] = parsed_vals
                    print(f"  Parsed {non_null}/{len(parsed_vals)} values for '{s}' from evaluation text")

    result = result.dropna(subset=["essay", "overall"])
    result = result[(result["overall"] >= 0) & (result["overall"] <= 9)]
    result["overall"] = result["overall"].apply(lambda x: round(x * 2) / 2)

    result["essay"] = result["essay"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    result["word_count"] = result["essay"].apply(lambda x: len(x.split()))
    result = result[(result["word_count"] >= min_words) & (result["word_count"] <= max_words)]

    result["normalized_text_hash"] = result["essay"].apply(compute_text_hash)

    if deduplicate:
        if "split" in result.columns and result["split"].notna().any():
            split_priority = {"test": 0, "validation": 1, "val": 1, "train": 2}
            result["_split_priority"] = (
                result["split"]
                .astype(str)
                .str.lower()
                .map(lambda x: split_priority.get(x, 99))
            )
            result = result.sort_values(
                by=["_split_priority", "normalized_text_hash"]
            ).drop_duplicates(subset=["normalized_text_hash"], keep="first")
            result = result.drop(columns=["_split_priority"])
        else:
            result = result.drop_duplicates(subset=["normalized_text_hash"])

    result = result.reset_index(drop=True)
    result["id"] = result.index.astype(str)
    return result


def build_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    stratify_column: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    n = len(df)

    if "split" in df.columns:
        split_values = df["split"].astype(str).str.lower().str.strip()
        unique_splits = set(split_values.unique())
        if "train" in unique_splits and "test" in unique_splits:
            train_pool = df[split_values == "train"].copy()
            if "validation" in unique_splits:
                val = df[split_values == "validation"].copy()
            elif "val" in unique_splits:
                val = df[split_values == "val"].copy()
            else:
                train_pool, val = train_test_split(
                    train_pool,
                    test_size=val_ratio / max(train_ratio + val_ratio, 1e-9),
                    random_state=seed + 1,
                    stratify=train_pool[stratify_column].values if stratify_column and stratify_column in train_pool.columns else None,
                )
            train = train_pool.copy()
            test = df[split_values == "test"].copy()
            splits = {"train": train, "val": val, "test": test}

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                for name, split_df in splits.items():
                    path = output_dir / f"{name}.csv"
                    split_df.to_csv(path, index=False)
                    print(f"  Saved {name}: {len(split_df)} rows -> {path}")

            return splits

    stratify = None
    if stratify_column and stratify_column in df.columns:
        stratify = df[stratify_column].values

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(f"train_ratio + val_ratio must be <= 1.0, got {train_ratio + val_ratio}")

    if stratify is not None:
        train_val, test = train_test_split(
            df, test_size=test_ratio, random_state=seed, stratify=stratify
        )
        stratify_train_val = train_val[stratify_column].values if stratify_column else None
        train, val = train_test_split(
            train_val,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=seed + 1,
            stratify=stratify_train_val,
        )
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        train = df.iloc[train_idx].copy()
        val = df.iloc[val_idx].copy()
        test = df.iloc[test_idx].copy()

    splits = {"train": train, "val": val, "test": test}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, split_df in splits.items():
            path = output_dir / f"{name}.csv"
            split_df.to_csv(path, index=False)
            print(f"  Saved {name}: {len(split_df)} rows -> {path}")

    return splits


def print_split_stats(splits: dict[str, pd.DataFrame]) -> None:
    for name, split_df in splits.items():
        scores = split_df["overall"]
        print(
            f"  {name}: {len(split_df):>6} rows, "
            f"score mean={scores.mean():.2f}, std={scores.std():.2f}, "
            f"range=[{scores.min():.1f}, {scores.max():.1f}]"
        )


def main() -> None:
    args = parse_args()

    config = load_benchmark_config(args.config)
    ds_config = config["dataset"]
    split_config = config["splits"]

    input_path = args.input or ds_config["local_csv"]
    output_dir = args.output_dir or resolve_path(split_config["split_dir"])
    seed = args.seed or split_config["seed"]
    ratios = args.ratio or (split_config["train_ratio"], split_config["val_ratio"])

    if args.input:
        print(f"Loading data from explicit input CSV: {input_path}")
        df = pd.read_csv(input_path)
    else:
        print("Loading data from configured dataset source...")
        df = load_configured_dataset(ds_config)
    print(f"  Raw rows: {len(df)}")

    df_clean = normalize_dataframe(
        df,
        score_columns=ds_config["score_columns"],
        min_words=ds_config.get("min_words", 80),
        max_words=ds_config.get("max_words", 500),
        deduplicate=ds_config.get("deduplicate", True),
    )
    print(f"  After normalization: {len(df_clean)} rows")

    splits = build_splits(
        df_clean,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        seed=seed,
        stratify_column=split_config.get("stratify_column"),
        output_dir=output_dir,
    )

    print("\nSplit summary:")
    print_split_stats(splits)


if __name__ == "__main__":
    main()
