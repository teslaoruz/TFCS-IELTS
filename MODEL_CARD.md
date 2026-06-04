# Model Card: TFCS-IELTS Cascade

## Model Details

TFCS-IELTS is a three-stage automated essay scoring cascade for IELTS Writing Task 2 overall band prediction.

Stage 1: TF-IDF Ridge regression.  
Stage 2: fine-tuned DistilBERT regression.  
Stage 3: MiniLM retrieval plus local Qwen2.5-3B GGUF scoring with strict JSON output.

## Intended Use

Research reproduction and offline IELTS-style writing assessment experiments. It is not a certified IELTS examiner and should not be used as the sole basis for high-stakes decisions.

## Training Data

The system uses the public Hugging Face `chillies/IELTS-writing-task-2-evaluation` dataset after deduplication, score validation, and 80-500 word filtering.

## Evaluation

The held-out test split contains 486 essays. The lightweight cascade reports MAE 0.9033, QWK 0.4659, 5.3% LLM calls, and 0% parsing failures.

## Limitations

The system predicts overall band only. It has not been validated against double-marked certified examiner scores, prompt-domain transfer, cross-validation, or unquantized Qwen variants.
