from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.baselines.distilbert_baseline import train_distilbert, predict_distilbert
from src.baselines.embedding_baseline import EmbeddingKNN, train_embedding_lightgbm, predict_embedding_lightgbm
from src.baselines.mean_baseline import evaluate_mean_baseline, evaluate_median_baseline
from src.baselines.tfidf_baseline import train_tfidf_ridge, train_tfidf_svr, predict_tfidf
from src.experiments.build_splits import build_splits, load_configured_dataset, normalize_dataframe
from src.rag.build_index import build_index as build_faiss_index
from src.rag.config import load_benchmark_config, load_models_config, resolve_path, resolve_torch_device
from src.rag.fusion import fuse
from src.rag.llm_scorer import LLMScorer
from src.rag.retriever import Retriever
from src.utils.evaluation_metrics import (
    clip_and_round,
    compute_all_metrics_with_ci,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark evaluations")
    parser.add_argument("--config", default=None, help="Path to benchmark config YAML")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run: mean median tfidf_ridge tfidf_svr "
                             "embedding_knn embedding_lightgbm distilbert "
                             "retrieval_only llm_only retrieval_plus_llm")
    parser.add_argument("--output-dir", default=None, help="Results output directory")
    parser.add_argument("--split-dir", default=None, help="Split directory (overrides config)")
    parser.add_argument("--sample", type=int, default=None, help="Sample test set size")
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline methods")
    return parser.parse_args()


def load_data_and_splits(config: dict[str, Any], split_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    ds_config = config["dataset"]
    split_config = config["splits"]

    if split_dir is None:
        split_dir = resolve_path(split_config["split_dir"])

    split_dir = Path(split_dir)

    if (
        split_dir.exists()
        and all((split_dir / f"{n}.csv").exists() for n in ("train", "val", "test"))
    ):
        print(f"Loading existing splits from {split_dir}")
        splits = {}
        for name in ("train", "val", "test"):
            splits[name] = pd.read_csv(split_dir / f"{name}.csv")
        return splits

    print("Splits not found or configured dataset source changed. Building from scratch...")
    df = load_configured_dataset(ds_config)
    df_clean = normalize_dataframe(
        df,
        score_columns=ds_config["score_columns"],
        min_words=ds_config.get("min_words", 80),
        max_words=ds_config.get("max_words", 500),
        deduplicate=ds_config.get("deduplicate", True),
    )
    return build_splits(
        df_clean,
        train_ratio=split_config["train_ratio"],
        val_ratio=split_config["val_ratio"],
        seed=split_config["seed"],
        stratify_column=split_config.get("stratify_column"),
        output_dir=split_dir,
    )


def _run_baseline(
    method: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    models_config: dict[str, Any],
) -> np.ndarray:
    band_col = "overall"
    text_col = "essay"

    if method == "mean":
        return evaluate_mean_baseline(df_train, df_test, band_column=band_col)

    elif method == "median":
        return evaluate_median_baseline(df_train, df_test, band_column=band_col)

    elif method == "tfidf_ridge":
        params = models_config.get("tfidf", {}).get("ridge", {})
        pipe = train_tfidf_ridge(
            df_train, band_column=band_col, text_column=text_col,
            alpha=params.get("alpha", 1.0),
            max_features=params.get("max_features", 5000),
            ngram_range=tuple(params.get("ngram_range", [1, 2])),
        )
        return predict_tfidf(pipe, df_test, text_column=text_col)

    elif method == "tfidf_svr":
        params = models_config.get("tfidf", {}).get("svr", {})
        pipe = train_tfidf_svr(
            df_train, band_column=band_col, text_column=text_col,
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            max_features=params.get("max_features", 5000),
            ngram_range=tuple(params.get("ngram_range", [1, 2])),
        )
        return predict_tfidf(pipe, df_test, text_column=text_col)

    elif method == "embedding_knn":
        params = models_config.get("embedding_baseline", {}).get("knn", {})
        emb_model_config = models_config.get("embedding", {}).get("primary", {})
        knn = EmbeddingKNN(
            model_name=emb_model_config.get("name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=emb_model_config.get("device", "auto"),
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "distance"),
            metric=params.get("metric", "cosine"),
        )
        knn.fit(df_train, band_column=band_col, text_column=text_col)
        return knn.predict(df_test, text_column=text_col)

    elif method == "embedding_lightgbm":
        params = models_config.get("embedding_baseline", {}).get("lightgbm", {})
        emb_model_config = models_config.get("embedding", {}).get("primary", {})
        model, lgb = train_embedding_lightgbm(
            df_train, band_column=band_col, text_column=text_col,
            model_name=emb_model_config.get("name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=emb_model_config.get("device", "auto"),
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            num_leaves=params.get("num_leaves", 31),
        )
        return predict_embedding_lightgbm(model, lgb, df_test, text_column=text_col)

    elif method == "distilbert":
        params = models_config.get("distilbert", {})
        device = resolve_torch_device(params.get("device", "auto"))
        model, tokenizer = train_distilbert(
            df_train, band_column=band_col, text_column=text_col,
            model_name=params.get("model_name", "distilbert-base-uncased"),
            max_length=params.get("max_length", 256),
            batch_size=params.get("batch_size", 16),
            learning_rate=params.get("learning_rate", 2e-5),
            num_epochs=params.get("num_epochs", 3),
            hidden_dim=params.get("hidden_dim", 128),
            dropout=params.get("dropout", 0.1),
            device=device,
        )
        return predict_distilbert(
            model, tokenizer, df_test, text_column=text_col,
            max_length=params.get("max_length", 256),
            batch_size=params.get("batch_size", 16),
            device=device,
        )

    else:
        raise ValueError(f"Unknown baseline method: {method}")


def _build_reference_retriever(
    reference_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Retriever]:
    ret_config = config["retrieval"]
    models_config = load_models_config()
    emb_config = models_config.get("embedding", {}).get("primary", {})
    reference_df = reference_df.reset_index(drop=True).copy()
    index, embed_model, row_map = build_faiss_index(
        reference_df,
        text_column="essay",
        embed_model_name=ret_config["embed_model"],
        index_type=ret_config.get("index_type", "FlatL2"),
        prompt_aware=ret_config.get("prompt_aware", False),
        device=ret_config.get("device", emb_config.get("device", "auto")),
    )
    retriever = Retriever(
        df=reference_df,
        index=index,
        metadata=row_map,
        embed_model=embed_model,
        band_column="overall",
        prompt_aware=ret_config.get("prompt_aware", False),
        diverse_exemplars=ret_config.get("diverse_exemplars", False),
        max_distance=ret_config.get("max_distance"),
    )
    return reference_df, retriever


def _retrieve_and_score(
    essay_text: str,
    retriever: Retriever,
    llm_scorer: LLMScorer | None,
    config: dict[str, Any],
    prompt_text: str | None = None,
    query_hash: str | None = None,
) -> tuple[float, float | None, dict[str, float]]:
    neighbors = retriever.retrieve(
        essay_text=essay_text,
        top_k=config["retrieval"]["top_k"],
        exclude_row_index=None,
        exclude_hashes={query_hash} if query_hash else None,
        prompt_text=prompt_text,
    )

    retrieval_score = retriever.predict_similarity_weighted(neighbors)
    retrieval_features = retriever.get_retrieval_features(neighbors, len(essay_text.split()))

    llm_score = None
    if llm_scorer and llm_scorer.is_available():
        result = llm_scorer.score(essay_text, neighbors)
        if result["llm_scores"] is not None:
            scores = result["llm_scores"]
            llm_score = scores.get("overall", np.mean(list(scores.values())))

    return retrieval_score, llm_score, retrieval_features


def run_system_method(
    method: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    config: dict[str, Any],
    llm_scorer: LLMScorer | None = None,
) -> np.ndarray:
    import traceback as _tb
    print(f"  [DEBUG] run_system_method called with method='{method}'")
    fusion_config = config["fusion"]
    retriever = None
    if method in ("retrieval_only", "retrieval_plus_llm"):
        reference_df = pd.concat([df_train, df_val], ignore_index=True)
        _, retriever = _build_reference_retriever(reference_df, config)

    predictions = []
    invalid_llm_outputs = 0
    for idx, row in df_test.iterrows():
        essay_text = str(row["essay"])
        prompt_text = None
        if "question" in df_test.columns and pd.notna(row.get("question")):
            prompt_text = str(row.get("question"))
        query_hash = None
        if "normalized_text_hash" in df_test.columns and pd.notna(row.get("normalized_text_hash")):
            query_hash = str(row.get("normalized_text_hash"))

        if method == "llm_only":
            if llm_scorer is None or not llm_scorer.is_available():
                raise RuntimeError("LLM scorer is not available for llm_only evaluation.")
            result = llm_scorer.score(essay_text, neighbors=None)
            llm_scores = result.get("llm_scores")
            if llm_scores is None:
                invalid_llm_outputs += 1
                predictions.append(np.nan)
                continue
            pred = llm_scores.get("overall", np.mean(list(llm_scores.values())))
            predictions.append(pred)
            continue

        retrieval_score, llm_score, retrieval_features = _retrieve_and_score(
            essay_text,
            retriever,
            llm_scorer,
            config,
            prompt_text=prompt_text,
            query_hash=query_hash,
        )

        if method == "retrieval_only":
            pred = fuse(retrieval_score, None, strategy="retrieval_only")
        elif method == "retrieval_plus_llm":
            pred = fuse(
                retrieval_score, llm_score,
                strategy=fusion_config.get("strategy", "threshold_gating"),
                retrieval_features=retrieval_features,
                tau=fusion_config.get("tau", 1.5),
                alpha=fusion_config.get("alpha", 0.7),
                llm_valid=llm_score is not None,
            )
        else:
            pred = retrieval_score

        predictions.append(pred)

    if method == "llm_only":
        if invalid_llm_outputs:
            print(f"  [DEBUG] Raising error: method={method}, invalid={invalid_llm_outputs}")
            _tb.print_stack()
            raise RuntimeError(
                f"LLM-only evaluation produced {invalid_llm_outputs} invalid outputs; "
                "the method is skipped instead of silently backfilling with retrieval scores."
            )
    elif invalid_llm_outputs:
        print(f"  Warning: {invalid_llm_outputs}/{len(df_test)} LLM scores invalid during {method}")

    return np.array(predictions)


def run_benchmark(
    config: dict[str, Any],
    methods: list[str] | None = None,
    output_dir: str | Path | None = None,
    sample_size: int | None = None,
    skip_baselines: bool = False,
) -> dict[str, dict[str, float]]:
    models_config = load_models_config()
    eval_config = config["evaluation"]

    splits = load_data_and_splits(config)
    df_train, df_val, df_test = splits["train"], splits["val"], splits["test"]

    if sample_size and sample_size > 0 and sample_size < len(df_test):
        df_test = df_test.sample(n=sample_size, random_state=42)
        print(f"  Sampled test set: {len(df_test)} rows")

    ground_truth = df_test["overall"].values

    if methods is None:
        methods = ["mean", "median", "tfidf_ridge", "embedding_knn"]

    output_dir = Path(output_dir or resolve_path(eval_config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: dict[str, np.ndarray] = {}
    all_metrics: dict[str, dict[str, Any]] = {}

    llm_scorer = None
    if config["llm"]["enabled"]:
        generator = _try_load_generator(config)
        if generator:
            llm_scorer = LLMScorer(
                generator=generator,
                prompt_template=resolve_path(config["llm"]["prompt_template"]),
                max_retries=config["llm"]["max_retries"],
                max_new_tokens=config.get("llm", {}).get("max_new_tokens", 80),
            )

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running: {method}")
        print(f"{'='*60}")

        t0 = time.time()

        if method in ("retrieval_only", "llm_only", "retrieval_plus_llm"):
            if not skip_baselines:
                preds = run_system_method(
                    method, df_train, df_val, df_test, config, llm_scorer
                )
            else:
                print(f"  Skipping {method} (--no-baselines)")
                continue
        else:
            try:
                preds = _run_baseline(method, df_train, df_val, df_test, models_config)
            except Exception as exc:
                print(f"  Error running {method}: {exc}")
                warnings.warn(f"Skipping {method} due to error: {exc}")
                continue

        elapsed = time.time() - t0
        preds = clip_and_round(preds, eval_config["min_score"], eval_config["max_score"])
        all_predictions[method] = preds

        metrics = compute_all_metrics_with_ci(
            ground_truth, preds,
            n_bootstrap=eval_config.get("bootstrap_samples", 1000),
        )
        metrics["elapsed_seconds"] = elapsed
        all_metrics[method] = metrics

        print(f"  MAE: {metrics['mae']['value']:.4f} "
              f"[{metrics['mae']['ci_lower']:.4f}, {metrics['mae']['ci_upper']:.4f}]")
        print(f"  RMSE: {metrics['rmse']['value']:.4f}")
        print(f"  QWK: {metrics['qwk']['value']:.4f}")
        print(f"  Within 0.5: {metrics['within_0.5']['value']:.4f}")
        print(f"  Within 1.0: {metrics['within_1.0']['value']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    predictions_df = pd.DataFrame({"true": ground_truth})
    for method, preds in all_predictions.items():
        predictions_df[f"pred_{method}"] = preds
    pred_path = output_dir / "predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    summary = {}
    for method, m in all_metrics.items():
        summary[method] = {k: v["value"] for k, v in m.items() if isinstance(v, dict)}
    summary_df = pd.DataFrame(summary).T
    summary_path = output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")

    return all_metrics


def _try_load_generator(config: dict[str, Any]):
    llm_config = config["llm"]
    model_name = llm_config["model_name"]

    if model_name.lower().endswith(".gguf"):
        model_path = str(resolve_path(model_name))
        return _load_gguf_generator(model_path, llm_config)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
    except ImportError:
        print("  transformers not available. LLM scoring disabled.")
        return None

    device_map = llm_config.get("device_map", "auto")
    dtype_str = llm_config.get("torch_dtype", "float16")
    load_in_4bit = llm_config.get("load_in_4bit", False)

    dtype = torch.float16 if dtype_str == "float16" else torch.float32

    model_kwargs = dict(
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=True,
    )

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    import gc
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print(f"  Loading LLM: {model_name}" + (" (4-bit)" if load_in_4bit else ""))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        model.eval()

        lora_path = llm_config.get("lora_path")
        if lora_path:
            from peft import PeftModel
            lora_path = str(resolve_path(lora_path))
            if Path(lora_path).exists():
                print(f"  Loading LoRA adapter from {lora_path}")
                model = PeftModel.from_pretrained(model, lora_path)
                model = model.merge_and_unload()
                model.eval()
            else:
                print(f"  Warning: LoRA path not found: {lora_path}")

        max_input_tokens = llm_config.get("max_input_tokens", 300)

        def generate_text(prompt: str, max_new_tokens: int = 20) -> str:
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are an IELTS Writing examiner."},
                    {"role": "user", "content": prompt},
                ]
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                rendered = prompt

            model_device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = tokenizer(
                rendered, return_tensors="pt", truncation=True, max_length=max_input_tokens
            ).to(model_device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or 20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)

            for artifact in ("<tool_call>", "Human:", "Assistant:"):
                response = response.split(artifact)[0]

            return response.strip()

        return generate_text

    except Exception as exc:
        print(f"  LLM load failed: {exc}")
        import traceback
        traceback.print_exc()
        return None


def _load_gguf_generator(model_path: str, llm_config: dict[str, Any]):
    import os as _os
    try:
        import torch as _torch
        _torch_lib = _os.path.join(_os.path.dirname(_torch.__file__), "lib")
        if _os.path.isdir(_torch_lib):
            _os.environ["PATH"] = _torch_lib + _os.pathsep + _os.environ.get("PATH", "")
    except ImportError:
        pass

    try:
        from llama_cpp import Llama
    except (ImportError, RuntimeError, OSError) as _e:
        print(f"  llama-cpp-python not available: {_e}. GGUF LLM scoring disabled.")
        return None

    max_input_tokens = llm_config.get("max_input_tokens", 1024)

    print(f"  Loading GGUF model: {model_path}")
    print(f"  Context: {max_input_tokens} tokens")

    llm = Llama(
        model_path=model_path,
        n_ctx=max_input_tokens,
        n_gpu_layers=-1,
        verbose=False,
    )

    def generate_text(prompt: str, max_new_tokens: int = 20) -> str:
        max_tokens = max_new_tokens or 20
        messages = [
            {"role": "system", "content": "You are an IELTS Writing examiner."},
            {"role": "user", "content": prompt},
        ]
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"]
        return content.strip()

    return generate_text


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config)
    output_dir = args.output_dir or resolve_path(config["evaluation"]["output_dir"])
    sample_size = args.sample or config["experiment"].get("sample_size", -1)
    if sample_size <= 0:
        sample_size = None

    if args.methods:
        methods = args.methods
    else:
        methods = ["mean", "median", "tfidf_ridge", "embedding_knn",
                    "retrieval_only", "llm_only", "retrieval_plus_llm"]

    run_benchmark(
        config,
        methods=methods,
        output_dir=output_dir,
        sample_size=sample_size,
        skip_baselines=args.no_baselines,
    )


if __name__ == "__main__":
    main()
