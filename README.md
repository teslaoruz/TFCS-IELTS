# TFCS-IELTS

Three-Tier Cascaded Scoring System for Offline IELTS Writing Assessment.

TFCS-IELTS reproduces the final paper pipeline: a tunable cascade that scores most IELTS Writing Task 2 essays with lightweight models and escalates only uncertain cases to a local retrieval-augmented LLM.

## Paper Abstract

Automated IELTS Writing assessment requires reliable scoring under privacy and resource constraints. TFCS-IELTS uses TF-IDF Ridge regression with k-nearest-neighbor variance estimation, a fine-tuned DistilBERT regressor, and a local Qwen2.5-3B GGUF model conditioned on retrieved IELTS reference essays. On the Hugging Face `chillies/IELTS-writing-task-2-evaluation` corpus, 10,324 raw essays are cleaned to 8,722 valid samples and split into 6,782 train, 1,454 validation, and 486 test essays. The lightweight cascade reaches MAE 0.9033 and QWK 0.4659 while invoking the LLM for 5.3% of test essays with 0% parsing failures.

## Method Summary

Stage 1 fits TF-IDF unigram/bigram features with `max_features=5000` and Ridge `alpha=1.0`. Confidence is estimated with weighted k-nearest-neighbor score variance using `k=20`.

Stage 2 fine-tunes `distilbert-base-uncased` for overall IELTS band regression with `max_length=256`, batch size `16`, learning rate `2e-5`, `3` epochs, and seed `42`.

Stage 3 builds a FAISS `FlatIP` index over MiniLM-L6-v2 essay embeddings. It retrieves top 7 neighbors and injects the top 3 as calibration examples into a local Qwen2.5-3B GGUF Q4_K_M prompt. The LLM must return exactly five JSON keys: `task_response`, `coherence`, `lexical`, `grammar`, and `overall`.

## Repository Structure

```text
TFCS-IELTS/
├── configs/                 # Model, data, retrieval, LLM, and prompt settings
├── data/
│   ├── models/              # Place qwen2.5-3b-instruct-q4_k_m.gguf here
│   └── splits/              # Fixed train/val/test CSV splits used by the paper
├── paper/                   # Final submitted paper markdown
├── results/tfcs_v2/figures/ # Paper figure outputs
├── scripts/                 # Reproduction helpers
├── src/
│   ├── baselines/           # TF-IDF Ridge and DistilBERT regressors
│   ├── data/                # Hugging Face dataset loading
│   ├── experiments/         # Split builder and benchmark support
│   ├── inference/           # Interactive UI scoring helpers
│   ├── rag/                 # Retriever, LLM scorer, config utilities
│   └── utils/               # Metrics and rounding helpers
├── streamlit_app.py         # Local scoring UI
├── tfcs_v2_full.py          # Main paper reproduction entry point
├── reproduce.sh             # Linux/macOS reproduction script
└── requirements.txt
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset Preparation

The default source is the public Hugging Face dataset:

```text
chillies/IELTS-writing-task-2-evaluation
```

Build the deterministic paper splits:

```bash
python -m src.experiments.build_splits
```

Expected split sizes:

```text
train: 6782
val:   1454
test:   486
```

If Hugging Face access is unavailable, download the dataset manually and place CSV split files under `data/huggingface_chillies/`; the loader will use them as a fallback.

## Model Preparation

Download the Qwen2.5-3B Instruct GGUF Q4_K_M file separately and place it at:

```text
data/models/qwen2.5-3b-instruct-q4_k_m.gguf
```

The model binary is intentionally not committed to GitHub.

## Local UI

Run the Streamlit interface from the repository root:

```bash
streamlit run streamlit_app.py
```

Windows PowerShell:

```powershell
.\scripts\run_ui.ps1
```

Then open:

```text
http://localhost:8501
```

The UI supports four modes:

| Mode | Behavior |
| --- | --- |
| Stage 1 only | Fast TF-IDF Ridge score and uncertainty estimate |
| Lightweight | Paper default cascade, `var=1.0`, `delta=1.5` |
| Max accuracy | More Stage 3 use, `var=0.75`, `delta=0.5` |
| Ultra-light | Minimal Stage 3 use, `var=2.0`, `delta=1.5` |

Stage 1 runs immediately after the first cached setup. DistilBERT and Qwen are optional toggles in the sidebar because their first run is slower and requires model availability. If the Qwen GGUF file is missing, the UI keeps working and falls back before Stage 3.

## Reproduction

From the repository root:

```bash
bash reproduce.sh
```

or on Windows:

```powershell
.\scripts\reproduce.ps1
```

Manual equivalent:

```bash
python -m src.experiments.build_splits
rm -rf results/tfcs_v2/cache
python tfcs_v2_full.py
```

## Expected Outputs

The main run writes:

```text
results/tfcs_v2/cache/results.json
results/tfcs_v2/cache/predictions_lightweight.npz
results/tfcs_v2/figures/fig1_bar_mae.png
results/tfcs_v2/figures/fig2_scatter.png
results/tfcs_v2/figures/fig3_error_dist.png
results/tfcs_v2/figures/fig4_tradeoff.png
results/tfcs_v2/figures/fig5_confusion.png
results/tfcs_v2/figures/fig6_cascade_flow.png
```

Paper-aligned headline results:

| Method | Test MAE | Test QWK | LLM calls |
| --- | ---: | ---: | ---: |
| TF-IDF Ridge | 0.9599 | 0.3577 | 0% |
| Fine-tuned DistilBERT | 0.9105 | 0.4670 | 0% |
| Cascade Max Accuracy | 0.8909 | 0.4654 | 49.6% |
| Cascade Lightweight | 0.9033 | 0.4659 | 5.3% |
| Cascade Ultra-light | 0.9506 | 0.3752 | 0.6% |

## Hardware

Reported experiments used an Intel i5-12500H CPU, 16 GB RAM, and optional RTX 3050 4 GB GDDR6 acceleration. Stage 1 runs on CPU. Stage 2 uses about 1.3 GB GPU memory. Stage 3 uses about 2.3 GB GPU memory for the quantized local LLM.

## Citation

```bibtex
@article{rana2026tfcsielts,
  title={TFCS-IELTS: A Three-Tier Cascaded Scoring System for Offline IELTS Writing Assessment},
  author={Rana, Md Jahidul Islam and Oruzgani, Irshad Ahmad and Kunicina, Nadezhda},
  year={2026}
}
```

## License

Code is released under the MIT License. Dataset and model files are governed by their original upstream licenses.
