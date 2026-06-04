from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.rag.config import load_benchmark_config, load_models_config, resolve_path, resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from processed essays")
    parser.add_argument("--config", default=None, help="Path to benchmark config YAML")
    parser.add_argument("--data-path", default=None, help="Input CSV path (overrides config)")
    parser.add_argument("--index-path", default=None, help="Output FAISS index path")
    parser.add_argument("--meta-path", default=None, help="Output metadata path")
    parser.add_argument("--embed-model", default=None, help="Embedding model name")
    parser.add_argument("--index-type", default=None, help="FAISS index type: FlatL2 or FlatIP")
    return parser.parse_args()


def _compose_retrieval_text(row: pd.Series, text_column: str = "essay", prompt_aware: bool = False) -> str:
    essay = str(row.get(text_column, "") or "")
    if not prompt_aware:
        return essay

    prompt = ""
    for candidate in ("prompt", "question", "task"):
        value = row.get(candidate)
        if pd.notna(value):
            prompt = str(value)
            break

    return f"{prompt} {essay}".strip()


def build_index(
    df: pd.DataFrame,
    text_column: str = "essay",
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_type: str = "FlatL2",
    prompt_aware: bool = False,
    device: str = "auto",
    index_path: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    texts = [_compose_retrieval_text(row, text_column=text_column, prompt_aware=prompt_aware) for _, row in df.iterrows()]
    resolved_device = resolve_torch_device(device)
    print(f"Loading embedding model: {embed_model_name} on {resolved_device}")
    model = SentenceTransformer(embed_model_name, device=resolved_device)

    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    if index_type == "FlatIP":
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    print(f"Index size: {index.ntotal} vectors")

    row_map = list(range(len(df)))
    metadata = {
        "num_rows": len(df),
        "columns": list(df.columns),
        "embed_model": embed_model_name,
        "index_type": index_type,
        "dimension": dim,
        "prompt_aware": prompt_aware,
        "device": resolved_device,
    }

    if index_path:
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        print(f"Index saved to: {index_path}")

    if meta_path:
        meta_path = Path(meta_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "wb") as f:
            pickle.dump({"row_map": row_map, "metadata": metadata}, f)
        print(f"Metadata saved to: {meta_path}")

    return index, model, row_map


def main() -> None:
    args = parse_args()

    config = load_benchmark_config(args.config)
    models_config = load_models_config()
    ds_config = config["dataset"]
    ret_config = config["retrieval"]
    emb_config = models_config.get("embedding", {}).get("primary", {})

    data_path = Path(args.data_path or resolve_path(ds_config["local_csv"]))
    embed_model = args.embed_model or ret_config["embed_model"]
    index_type = args.index_type or ret_config.get("index_type", "FlatL2")

    if args.index_path:
        index_path = Path(args.index_path)
    elif ret_config.get("index_path"):
        index_path = resolve_path(ret_config["index_path"])
    else:
        index_path = resolve_path("data/embeddings/faiss.index")

    if args.meta_path:
        meta_path = Path(args.meta_path)
    elif ret_config.get("meta_path"):
        meta_path = resolve_path(ret_config["meta_path"])
    else:
        meta_path = resolve_path("data/embeddings/metadata.pkl")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")

    build_index(
        df=df,
        text_column="essay",
        embed_model_name=embed_model,
        index_type=index_type,
        prompt_aware=ret_config.get("prompt_aware", False),
        device=ret_config.get("device", emb_config.get("device", "auto")),
        index_path=index_path,
        meta_path=meta_path,
    )


if __name__ == "__main__":
    main()
