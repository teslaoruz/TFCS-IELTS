"""TFCS v2: Full pipeline with caching + plots for all configs."""

import gc
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.baselines.distilbert_baseline import train_distilbert, predict_distilbert
from src.baselines.tfidf_baseline import train_tfidf_ridge
from src.experiments.run_benchmark import _build_reference_retriever, _try_load_generator
from src.rag.config import load_benchmark_config, resolve_path, resolve_torch_device
from src.rag.llm_scorer import LLMScorer
from src.utils.evaluation_metrics import clip_and_round

# ── Style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "lines.linewidth": 1.5,
})

CACHE = Path("results/tfcs_v2/cache")
CACHE.mkdir(parents=True, exist_ok=True)
FIG = Path("results/tfcs_v2/figures")
FIG.mkdir(parents=True, exist_ok=True)
bc, tc = "overall", "essay"


def tfidf_variance(text, vf, X_ref, ref_scores, k=20):
    xq = vf.transform([text])
    sims = cosine_similarity(xq, X_ref).flatten()
    top_k = np.argsort(sims)[-k:][::-1]
    ns = ref_scores[top_k]
    nw = sims[top_k]
    if nw.sum() > 0:
        m = np.average(ns, weights=nw)
        v = np.average((ns - m)**2, weights=nw)
    else:
        v = np.var(ns)
    return float(v)


def paper_fusion(S_rag, S_llm):
    if S_llm is None or np.isnan(S_llm):
        return round(S_rag * 2) / 2
    d = abs(S_llm - S_rag)
    if d > 1.5:
        return round(S_rag * 2) / 2
    elif d > 1.0:
        return round((0.90 * S_rag + 0.10 * S_llm) * 2) / 2
    else:
        return round((0.70 * S_rag + 0.30 * S_llm) * 2) / 2


def evaluate_cascade(s1, s2, y, var, v_thresh, d_thresh, get_llm_fn, texts, prompts, indices):
    """Apply cascade given thresholds. Returns (cascade_preds, n_s1, n_s2, n_s3, n_fail)."""
    cascade = np.copy(s1)
    n_s1 = int((var < v_thresh).sum())
    uncertain = ~(var < v_thresh)
    s2_used = uncertain
    cascade[s2_used] = s2[s2_used]
    disagree = s2_used & (np.abs(s2 - s1) > d_thresh)
    n_s3 = int(disagree.sum())
    n_fail = 0
    for j in np.where(disagree)[0]:
        s3 = get_llm_fn(texts[j], prompts[j], indices[j], s2[j])
        if s3 is None:
            n_fail += 1
            s3 = s2[j]
        cascade[j] = paper_fusion(s2[j], s3)
    n_s2 = int(uncertain.sum()) - n_s3
    return cascade, n_s1, n_s2, n_s3, n_fail


def main():
    config = load_benchmark_config()
    eval_cfg = config["evaluation"]
    split_dir = Path(resolve_path(config["splits"]["split_dir"]))
    t_start = time.time()

    print("Loading splits...")
    df_train = pd.read_csv(split_dir / "train.csv")
    df_val = pd.read_csv(split_dir / "val.csv")
    df_test = pd.read_csv(split_dir / "test.csv")
    print(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # ── STAGE 1: TF-IDF Ridge ───────────────────────────────
    cache_s1 = CACHE / "stage1.pkl"
    if cache_s1.exists():
        print("\n[Cache] Loading Stage 1...")
        with open(cache_s1, "rb") as f:
            data = pickle.load(f)
        pipe, vf, X_train_tfidf, train_scores = data["pipe"], data["vf"], data["X_train_tfidf"], data["train_scores"]
        val_s1, val_var, test_s1, test_var = data["val_s1"], data["val_var"], data["test_s1"], data["test_var"]
        y_val = data["y_val"]
        y_test = data["y_test"]
        val_texts = data["val_texts"]
        test_texts = data["test_texts"]
        val_prompts = data["val_prompts"]
        test_prompts = data["test_prompts"]
        print(f"  Loaded (val={len(val_s1)}, test={len(test_s1)})")
    else:
        print("\n[Stage 1] TF-IDF Ridge...")
        t0 = time.time()
        pipe = train_tfidf_ridge(df_train, band_column=bc, text_column=tc, alpha=1.0)
        vf = pipe.named_steps["tfidf"]
        train_texts = df_train[tc].fillna("").astype(str).tolist()
        X_train_tfidf = vf.transform(train_texts)
        train_scores = df_train[bc].values

        def s1(text):
            s = float(pipe.predict([text])[0])
            return clip_and_round(s, eval_cfg["min_score"], eval_cfg["max_score"])

        y_val = df_val[bc].values
        y_test = df_test[bc].values
        val_texts = df_val[tc].fillna("").astype(str).tolist()
        test_texts = df_test[tc].fillna("").astype(str).tolist()
        val_prompts = [str(r.get("question")) if "question" in df_val.columns and pd.notna(r.get("question")) else None for _, r in df_val.iterrows()]
        test_prompts = [str(r.get("question")) if "question" in df_test.columns and pd.notna(r.get("question")) else None for _, r in df_test.iterrows()]

        print("  Computing val...")
        val_s1 = np.array([s1(t) for t in val_texts])
        val_var = np.array([tfidf_variance(t, vf, X_train_tfidf, train_scores) for t in val_texts])
        print("  Computing test...")
        test_s1 = np.array([s1(t) for t in test_texts])
        test_var = np.array([tfidf_variance(t, vf, X_train_tfidf, train_scores) for t in test_texts])

        with open(cache_s1, "wb") as f:
            pickle.dump({
                "pipe": pipe, "vf": vf, "X_train_tfidf": X_train_tfidf, "train_scores": train_scores,
                "val_s1": val_s1, "val_var": val_var, "test_s1": test_s1, "test_var": test_var,
                "y_val": y_val, "y_test": y_test, "val_texts": val_texts, "test_texts": test_texts,
                "val_prompts": val_prompts, "test_prompts": test_prompts,
            }, f)
        print(f"  Done ({time.time()-t0:.0f}s)")

    # ── STAGE 2: DistilBERT ─────────────────────────────────
    cache_s2 = CACHE / "stage2.pkl"
    if cache_s2.exists():
        print("\n[Cache] Loading Stage 2 (DistilBERT)...")
        with open(cache_s2, "rb") as f:
            data = pickle.load(f)
        val_db, test_db = data["val_db"], data["test_db"]
        print(f"  Loaded (val={len(val_db)}, test={len(test_db)})")
    else:
        print("\n[Stage 2] DistilBERT...")
        t0 = time.time()
        device = resolve_torch_device("auto")
        model, tokenizer = train_distilbert(df_train, band_column=bc, text_column=tc, device=device)
        print(f"  Training done ({time.time()-t0:.0f}s)")

        print("  Predicting val...")
        val_db = predict_distilbert(model, tokenizer, df_val, text_column=tc, device=device)
        print("  Predicting test...")
        test_db = predict_distilbert(model, tokenizer, df_test, text_column=tc, device=device)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        with open(cache_s2, "wb") as f:
            pickle.dump({"val_db": val_db, "test_db": test_db}, f)
        print(f"  Done ({time.time()-t0:.0f}s)")

    # ── STAGE 3: Retriever + LLM cache ──────────────────────
    # Build retriever (cached)
    cache_ret = CACHE / "retriever.pkl"
    if cache_ret.exists():
        print("\n[Cache] Loading retriever...")
        with open(cache_ret, "rb") as f:
            retriever = pickle.load(f)
    else:
        print("\n[Stage 3] Building dense retriever...")
        t0 = time.time()
        _, retriever = _build_reference_retriever(df_train, config)
        with open(cache_ret, "wb") as f:
            pickle.dump(retriever, f)
        print(f"  Done ({time.time()-t0:.0f}s)")

    # LLM cache
    cache_llm = CACHE / "llm_scores.pkl"
    if cache_llm.exists():
        print("\n[Cache] Loading LLM scores...")
        with open(cache_llm, "rb") as f:
            llm_cache = pickle.load(f)
    else:
        llm_cache = {}

    def get_llm(text, prompt, idx, distilbert_score=None):
        if idx in llm_cache:
            return llm_cache[idx]
        nn = retriever.retrieve(text, top_k=config["retrieval"]["top_k"],
            exclude_row_index=None, exclude_hashes=None, prompt_text=prompt)
        db_scores = {"overall": float(distilbert_score)} if distilbert_score is not None else None
        result = llm_scorer.score(text, nn, distilbert_scores=db_scores)
        if result["llm_scores"] is not None:
            s = result["llm_scores"].get("overall", float(np.mean(list(result["llm_scores"].values()))))
        else:
            s = None
        llm_cache[idx] = s
        # Save incrementally
        with open(cache_llm, "wb") as f:
            pickle.dump(llm_cache, f)
        return s

    # Load LLM only if cache is incomplete
    if len(llm_cache) < len(df_val) + len(df_test):
        print("\n[Stage 3] Loading LLM (3B GGUF)...")
        generator = _try_load_generator(config)
        llm_scorer = LLMScorer(
            generator=generator,
            prompt_template=resolve_path(config["llm"]["prompt_template"]),
            max_retries=config["llm"]["max_retries"],
            max_new_tokens=config.get("llm", {}).get("max_new_tokens", 80),
        )

    # ── TUNE on VAL ─────────────────────────────────────────
    var_thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    delta_thresholds = [0.5, 0.75, 1.0, 1.25, 1.5]

    print(f"\n{'='*60}")
    print("Tuning on VAL...")
    print(f"{'='*60}")

    val_log = []
    for d_thresh in delta_thresholds:
        for v_thresh in var_thresholds:
            cascade, n_s1, n_s2, n_s3, n_fail = evaluate_cascade(
                val_s1, val_db, y_val, val_var, v_thresh, d_thresh,
                get_llm, val_texts, val_prompts, [f"v{i}" for i in range(len(val_texts))])
            mae = float(np.abs(cascade - y_val).mean())
            p1 = n_s1 / len(y_val) * 100
            p3 = n_s3 / len(y_val) * 100
            val_log.append({"var": v_thresh, "delta": d_thresh, "mae": round(mae, 4),
                           "pct_S1": round(p1, 1), "pct_S3": round(p3, 1), "llm_fail": n_fail})
            print(f"  var<{v_thresh:<5.2f} delta<={d_thresh:.2f}: S1={p1:5.1f}% S3={p3:4.1f}% MAE={mae:.4f}")

    # Save LLM cache
    with open(cache_llm, "wb") as f:
        pickle.dump(llm_cache, f)

    # ── Best configs ────────────────────────────────────────
    best_max_acc = min(val_log, key=lambda x: x["mae"])
    # Best lightweight: minimize S3% while keeping MAE < 0.95
    lightweight_candidates = [v for v in val_log if v["mae"] < 0.95]
    best_lightweight = min(lightweight_candidates, key=lambda x: x["pct_S3"]) if lightweight_candidates else best_max_acc
    # Ultra-light: minimize S3% regardless
    best_ultra = min(val_log, key=lambda x: x["pct_S3"])

    print(f"\n  >> Max Accuracy:  var<{best_max_acc['var']:.2f} delta<={best_max_acc['delta']:.2f} | "
          f"MAE={best_max_acc['mae']:.4f} | S3={best_max_acc['pct_S3']:.1f}%")
    print(f"  >> Lightweight:   var<{best_lightweight['var']:.2f} delta<={best_lightweight['delta']:.2f} | "
          f"MAE={best_lightweight['mae']:.4f} | S3={best_lightweight['pct_S3']:.1f}%")
    print(f"  >> Ultra-light:   var<{best_ultra['var']:.2f} delta<={best_ultra['delta']:.2f} | "
          f"MAE={best_ultra['mae']:.4f} | S3={best_ultra['pct_S3']:.1f}%")

    # ── EVALUATE on TEST for all 3 configs ──────────────────
    print(f"\n{'='*60}")
    print("Testing all configs...")
    print(f"{'='*60}")
    test_configs = {
        "max_accuracy": best_max_acc,
        "lightweight": best_lightweight,
        "ultra_light": best_ultra,
    }
    test_results = {}
    for cfg_name, cfg in test_configs.items():
        cascade, n_s1, n_s2, n_s3, n_fail = evaluate_cascade(
            test_s1, test_db, y_test, test_var, cfg["var"], cfg["delta"],
            get_llm, test_texts, test_prompts, [f"t{i}" for i in range(len(test_texts))])
        mae = float(np.abs(cascade - y_test).mean())
        mae_s1 = float(np.abs(test_s1 - y_test).mean())
        mae_s2 = float(np.abs(test_db - y_test).mean())
        test_results[cfg_name] = {
            "mae_stage1": round(mae_s1, 4),
            "mae_stage2": round(mae_s2, 4),
            "mae_cascade": round(mae, 4),
            "n_stage1": n_s1, "n_stage2": n_s2, "n_stage3": n_s3,
            "pct_stage1": round(n_s1/len(y_test)*100, 1),
            "pct_stage2": round(n_s2/len(y_test)*100, 1),
            "pct_stage3": round(n_s3/len(y_test)*100, 1),
            "llm_failures": n_fail,
        }
        print(f"\n  {cfg_name}:")
        print(f"    MAE cascade = {mae:.4f}")
        print(f"    Stage1: {n_s1:4d} ({n_s1/len(y_test)*100:.0f}%) | Stage2: {n_s2:4d} ({n_s2/len(y_test)*100:.0f}%) | "
              f"Stage3: {n_s3:4d} ({n_s3/len(y_test)*100:.1f}%)")

        # Save predictions for this config
        np.savez(CACHE / f"predictions_{cfg_name}.npz",
                 y_true=y_test, s1=test_s1, s2=test_db, cascade=cascade)

    results = {
        "method": "TFCS v2 (TFCS Cascade)",
        "val_tuning": val_log,
        "configs": {k: {**v, "var": test_configs[k]["var"], "delta": test_configs[k]["delta"]}
                    for k, v in test_results.items()},
        "total_time_s": round(time.time() - t_start, 1),
    }
    with open(CACHE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── PLOTS ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")

    # Figure 1: Bar chart — all methods
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["TF-IDF\nRidge", "DistilBERT", "Cascade\n(Max Acc)", "Cascade\n(Lightweight)", "Cascade\n(Ultra-light)"]
    maes_plot = [
        test_results["max_accuracy"]["mae_stage1"],
        test_results["max_accuracy"]["mae_stage2"],
        test_results["max_accuracy"]["mae_cascade"],
        test_results["lightweight"]["mae_cascade"],
        test_results["ultra_light"]["mae_cascade"],
    ]
    colors_bar = ["#4C72B0", "#55A868", "#C44E52", "#C44E52", "#C44E52"]
    bars = ax.bar(names, maes_plot, color=colors_bar, width=0.5, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, maes_plot):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MAE")
    ax.set_ylim(0, max(maes_plot) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("TFCS v2: Per-Method MAE", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG / "fig1_bar_mae.png")
    plt.close(fig)
    print("  fig1_bar_mae.png")

    # Figure 2: Scatter — predicted vs actual (lightweight config)
    d_test = np.load(CACHE / "predictions_lightweight.npz")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, preds), color in zip(axes,
        [("TF-IDF Ridge", d_test["s1"]), ("DistilBERT", d_test["s2"]), ("Cascade", d_test["cascade"])],
        ["#4C72B0", "#55A868", "#C44E52"]):
        ax.scatter(d_test["y_true"], preds, alpha=0.4, s=15, color=color, edgecolors="none")
        ax.plot([0, 9], [0, 9], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(3, 9)
        ax.set_ylim(3, 9)
        ax.set_xlabel("True Score")
        ax.set_ylabel("Predicted Score")
        m = float(np.abs(preds - d_test["y_true"]).mean())
        ax.set_title(f"{name}\nMAE={m:.4f}")
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG / "fig2_scatter.png")
    plt.close(fig)
    print("  fig2_scatter.png")

    # Figure 3: Error distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(-3, 3.1, 0.5)
    for name, preds, color in [
        ("TF-IDF Ridge", d_test["s1"], "#4C72B0"),
        ("DistilBERT", d_test["s2"], "#55A868"),
        ("Cascade (Lightweight)", d_test["cascade"], "#C44E52"),
    ]:
        errors = preds - d_test["y_true"]
        m = float(np.abs(errors).mean())
        ax.hist(errors, bins=bins, alpha=0.5, label=f"{name} (MAE={m:.4f})",
                color=color, edgecolor="black", lw=0.3)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution (Lightweight Config)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    plt.tight_layout()
    fig.savefig(FIG / "fig3_error_dist.png")
    plt.close(fig)
    print("  fig3_error_dist.png")

    # Figure 4: Efficiency-Accuracy Trade-off
    fig, ax = plt.subplots(figsize=(7, 5))
    for d_thresh in [0.5, 0.75, 1.0, 1.25, 1.5]:
        subset = [v for v in val_log if abs(v["delta"] - d_thresh) < 0.01]
        subset.sort(key=lambda x: x["var"])
        xv = [s["pct_S3"] for s in subset]
        yv = [s["mae"] for s in subset]
        ax.plot(xv, yv, "o-", label=f"|Δ| ≤ {d_thresh}", markersize=5, alpha=0.7)

    # Mark key configs
    for cfg_name, cfg_color, cfg_marker, cfg in [
        ("Max Acc", "green", "s", best_max_acc),
        ("Lightweight", "blue", "D", best_lightweight),
        ("Ultra-light", "red", "^", best_ultra),
    ]:
        ax.plot(cfg["pct_S3"], cfg["mae"], marker=cfg_marker, color=cfg_color,
                markersize=10, label=f"{cfg_name}", zorder=5)
        offset = (-60, 10) if cfg["pct_S3"] > 30 else (10, -20)
        ax.annotate(f"{cfg_name}\nMAE={cfg['mae']}\nS3={cfg['pct_S3']:.0f}%",
                    (cfg["pct_S3"], cfg["mae"]), textcoords="offset points",
                    xytext=offset, fontsize=9, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    ax.set_xlabel("% Essays Requiring LLM (Stage 3)")
    ax.set_ylabel("Val MAE")
    ax.set_title("Efficiency-Accuracy Trade-off")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(FIG / "fig4_tradeoff.png")
    plt.close(fig)
    print("  fig4_tradeoff.png")

    # Figure 5: Confusion matrix
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    y_binned = np.round(d_test["y_true"] * 2).astype(int)
    present_bins = sorted(set(y_binned))
    label_map = {b: f"{b/2:.1f}" for b in present_bins}

    for ax, (name, preds), color in zip(axes,
        [("TF-IDF", d_test["s1"]), ("DistilBERT", d_test["s2"]), ("Cascade", d_test["cascade"])],
        ["#4C72B0", "#55A868", "#C44E52"]):
        pred_binned = np.clip(np.round(preds * 2).astype(int), min(present_bins), max(present_bins))
        cm = confusion_matrix(y_binned, pred_binned, labels=present_bins)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        n = len(present_bins)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([label_map[b] for b in present_bins], fontsize=7)
        ax.set_yticklabels([label_map[b] for b in present_bins], fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
    plt.tight_layout()
    fig.savefig(FIG / "fig5_confusion.png")
    plt.close(fig)
    print("  fig5_confusion.png")

    # Figure 6: Cascade flow diagram (text-based summary)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    cfg = test_results["lightweight"]
    txt = (
        "TFCS v2 Cascade Flow (Lightweight Config)\n"
        "=================================================\n\n"
        f"[Input] Test Essay (n={len(y_test)})\n"
        f"    |\n"
        f"    v\n"
        f"[Stage 1: TF-IDF Ridge]  MAE={cfg['mae_stage1']:.4f}\n"
        f"    |-- var < {best_lightweight['var']:.2f}? --> Accept ({cfg['n_stage1']} essays, {cfg['pct_stage1']:.0f}%)  [No GPU]\n"
        f"    |-- var >= {best_lightweight['var']:.2f}? -->\n"
        f"         v\n"
        f"[Stage 2: DistilBERT]   MAE={cfg['mae_stage2']:.4f}\n"
        f"    |-- |S2-S1| <= {best_lightweight['delta']:.1f}? --> Accept ({cfg['n_stage2']} essays, {cfg['pct_stage2']:.0f}%)  [Light GPU]\n"
        f"    |-- |S2-S1| > {best_lightweight['delta']:.1f}? -->\n"
        f"         v\n"
        f"[Stage 3: LLM+RAG]      ({cfg['n_stage3']} essays, {cfg['pct_stage3']:.1f}%)  [Heavy GPU]\n"
        f"    |\n"
        f"    v\n"
        f"[Output] Cascade MAE = {cfg['mae_cascade']:.4f}\n\n"
        f"GPU compute saved: {cfg['pct_stage1']:.0f}% of essays use ZERO GPU\n"
        f"LLM calls saved:   {100-cfg['pct_stage3']:.0f}% vs all-LLM baseline"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontfamily="monospace",
            fontsize=9, va="top", linespacing=1.5)
    fig.savefig(FIG / "fig6_cascade_flow.png")
    plt.close(fig)
    print("  fig6_cascade_flow.png")

    print(f"\nAll figures saved to {FIG}/")
    print(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
