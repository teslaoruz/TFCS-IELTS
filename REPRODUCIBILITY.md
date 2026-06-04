# Reproducibility Audit

## Exact Workflow

1. Install Python 3.11 dependencies from `requirements.txt`.
2. Ensure `data/models/qwen2.5-3b-instruct-q4_k_m.gguf` exists.
3. Run `python -m src.experiments.build_splits`.
4. Clear old caches with `rm -rf results/tfcs_v2/cache`.
5. Run `python tfcs_v2_full.py`.
6. Compare `results/tfcs_v2/cache/results.json` against the expected metrics in `README.md`.

## Paper Alignment Checklist

- Dataset: `chillies/IELTS-writing-task-2-evaluation`.
- Cleaning: exact deduplication, invalid score removal, 80-500 word filter.
- Split: 6,782 train, 1,454 validation, 486 test.
- Stage 1: TF-IDF Ridge, unigram/bigram, `max_features=5000`, `alpha=1.0`, k-NN variance with `k=20`.
- Stage 2: DistilBERT regressor, 3 epochs, learning rate `2e-5`, batch size `16`, `max_length=256`, seed `42`.
- Stage 3: MiniLM-L6-v2 retriever, FAISS `FlatIP`, top 7 retrieval, top 3 prompt exemplars.
- LLM: local Qwen2.5-3B GGUF Q4_K_M, deterministic decoding, `max_new_tokens=80`, two retries.
- Prompt: strict five-key JSON output.
- Threshold grid: 9 variance thresholds x 5 delta thresholds.

## Files Kept

- `tfcs_v2_full.py`
- `configs/benchmark.yaml`
- `configs/models.yaml`
- `configs/prompts/scoring_prompt.txt`
- `src/baselines/`
- `src/data/`
- `src/experiments/`
- `src/rag/`
- `src/utils/`
- `data/splits/`
- `paper/camera.md`
- `results/tfcs_v2/figures/`

## Files Removed Or Ignored

- Local virtual environments and build outputs.
- Debug scripts named `debug_*`, `tmp_*`, `_check_data.py`, and `_results_table.py`.
- Old paper drafts, reviewer-response artifacts, and compiled PDFs/DOCX files.
- Kaggle packaging directories.
- Generated raw/processed datasets and FAISS indexes.
- Local model binaries, LoRA checkpoints, and cache files.
- Historical result folders from exploratory runs.

## Known Risks

- The Qwen GGUF model must be downloaded manually because it is too large for normal GitHub hosting.
- Full reproduction requires network access for Hugging Face models and dataset unless all artifacts are pre-cached.
- Exact DistilBERT metrics may vary slightly across CUDA, PyTorch, and driver versions despite fixed seeds.
- The paper evaluates a single fixed split; no cross-validation confidence intervals are claimed.
- Dataset labels are not double-marked by certified IELTS examiners.

## Final Assessment

The cleaned package is aligned with the final paper methodology and can reproduce the main claims when the public dataset and local Qwen GGUF model are available. Remaining replication risk is primarily external artifact availability and GPU/runtime variability, not undocumented project logic.
