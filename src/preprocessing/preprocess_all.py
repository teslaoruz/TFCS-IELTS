from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IELTS essay datasets into unified benchmark table")
    parser.add_argument("--input-dir", default="data/raw", help="Directory with raw CSV files")
    parser.add_argument("--output", default="data/processed/ielts_clean.csv", help="Output CSV path")
    parser.add_argument("--hf-fallback", default=None, help="Optional HuggingFace dataset fallback CSV")
    parser.add_argument("--min-words", type=int, default=80, help="Minimum essay word count")
    parser.add_argument("--max-words", type=int, default=500, help="Maximum essay word count")
    parser.add_argument("--deduplicate", action="store_true", default=True, help="Remove exact duplicates")
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize_score(value: Any) -> float | None:
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


def process_file(filepath: str | Path) -> pd.DataFrame | None:
    filepath = Path(filepath)
    print(f"Processing: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Skipping (not valid CSV): {e}")
        return None

    cols = [c.lower() for c in df.columns]
    df.columns = cols

    essay_col = None
    for c in cols:
        if "essay" in c:
            essay_col = c
            break
    if essay_col is None:
        print(f"  Skipping (no essay column)")
        return None

    result = pd.DataFrame()
    result["essay"] = df[essay_col].astype(str)

    score_col = None
    for c in cols:
        if "overall" in c:
            score_col = c
            break
    if score_col is None:
        for c in cols:
            if "score" in c or "band" in c:
                score_col = c
                break

    if score_col:
        result["overall"] = df[score_col].apply(normalize_score)
        result = result.dropna(subset=["overall"])
    else:
        print(f"  Skipping (no score column)")
        return None

    prompt_col = None
    for c in cols:
        if c in ("prompt", "question", "topic"):
            prompt_col = c
            break
    if prompt_col:
        result["question"] = df[prompt_col].astype(str)
    else:
        result["question"] = None

    task_col = None
    for c in cols:
        if c in ("task_type", "task"):
            task_col = c
            break
    if task_col:
        result["task_type"] = df[task_col].astype(str)
    else:
        result["task_type"] = "task2"

    source_col = None
    for c in cols:
        if c in ("source", "dataset"):
            source_col = c
            break
    if source_col:
        result["source"] = df[source_col].astype(str)
    else:
        result["source"] = filepath.stem

    for sub_score in ("task_response", "coherence", "lexical", "grammar"):
        for c in cols:
            if sub_score in c:
                result[sub_score] = df[c].apply(normalize_score)
                break
        if sub_score not in result.columns:
            result[sub_score] = None

    feedback_col = None
    for c in cols:
        if "feedback" in c:
            feedback_col = c
            break
    if feedback_col:
        result["feedback"] = df[feedback_col].astype(str)
    else:
        result["feedback"] = None

    return result


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    all_dfs = []

    if input_dir.exists():
        for file in input_dir.glob("*.csv"):
            df = process_file(file)
            if df is not None:
                all_dfs.append(df)

    if args.hf_fallback:
        hf_path = Path(args.hf_fallback)
        if hf_path.exists():
            print(f"Processing HF fallback: {hf_path}")
            df = process_file(hf_path)
            if df is not None:
                all_dfs.append(df)

    if not all_dfs:
        print("No valid data found!")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nBefore cleaning: {len(df)} rows")

    df = df.dropna(subset=["essay", "overall"])
    df["essay"] = df["essay"].apply(clean_text)
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df = df.dropna(subset=["overall"])
    df = df[(df["overall"] >= 0) & (df["overall"] <= 9)]
    df["overall"] = df["overall"].apply(lambda x: round(x * 2) / 2)

    df["word_count"] = df["essay"].apply(lambda x: len(x.split()))
    df = df[(df["word_count"] >= args.min_words) & (df["word_count"] <= args.max_words)]

    df["normalized_text_hash"] = df["essay"].apply(compute_text_hash)
    if args.deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=["normalized_text_hash"])
        print(f"Removed {before - len(df)} duplicates")

    for col in ["question", "task_type", "source", "feedback"]:
        if col not in df.columns:
            df[col] = None

    for col in ("task_response", "coherence", "lexical", "grammar"):
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(str)

    print(f"After cleaning: {len(df)} rows")
    print(f"Score range: [{df['overall'].min():.1f}, {df['overall'].max():.1f}]")
    print(f"Score mean: {df['overall'].mean():.2f}")
    print(f"Sources: {df['source'].unique()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
