"""
Local RAG Feedback (optional GPTQ LLM)
--------------------------------------
Retrieves similar essays from FAISS and optionally generates feedback
with a local GPTQ model. If GPTQ dependencies are missing, the script
still runs and prints a useful retrieval summary.
"""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[2]
FAISS_INDEX_PATH = ROOT_DIR / "data" / "embeddings" / "faiss.index"
METADATA_PATH = ROOT_DIR / "data" / "embeddings" / "metadata.pkl"
PROCESSED_CSV_PATH = ROOT_DIR / "data" / "processed" / "ielts_clean.csv"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
QWEN_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_REFERENCE_ESSAYS = 3
MAX_WORDS_PER_REFERENCE = 170
ENABLE_LLM_GENERATION = True
STRICT_SECTION_FORMAT = True
SHOW_RAW_OUTPUT_ON_FALLBACK = True
RAW_OUTPUT_PREVIEW_CHARS = 700


def truncate_words(text: str, max_words: int) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words]) + " ..."


def sanitize_query(raw_query: str) -> str:
    query = str(raw_query).strip()

    # If user pasted a full CSV row, prefer the first text field as essay input.
    if (query.count(",") >= 8 and '"' in query) or '","' in query:
        try:
            fields = next(csv.reader([query]))
            if fields:
                best = max(fields, key=lambda f: len(str(f).split()))
                if len(best.split()) >= 20:
                    query = best
        except Exception:
            pass

    query = query.replace("\r", " ").replace("\n", " ")
    query = re.sub(r"\s+", " ", query).strip(" \"'")
    if query.startswith(". "):
        query = query[2:].strip()
    return query


def normalize_for_dedup(text: str) -> str:
    lowered = str(text).lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def is_degenerate_output(text: str) -> bool:
    cleaned = str(text).strip()

    if not cleaned:
        return True

    if len(cleaned) < 10:
        return True

    if re.search(r"([!?.])\1{20,}", cleaned):
        return True

    if cleaned.count("�") > 5:
        return True

    return False

def is_valid_response(text: str) -> bool:
    if not has_expected_feedback_sections(text):
        return False

    if "Mistakes and Corrections:" not in text:
        return False

    # check paragraph length (rough)
    if "One Improved Sample Paragraph:" in text:
        part = text.split("One Improved Sample Paragraph:")[-1]
        words = len(part.split())
        if words < 60:   # strict threshold
            return False

    return True

def has_expected_feedback_sections(text: str) -> bool:
    t = str(text).lower()

    checks = [
        re.search(r"band|score", t),
        re.search(r"strength", t),
        re.search(r"weakness", t),
        re.search(r"improvement|suggestion", t),
        re.search(r"paragraph", t),
    ]

    return sum(bool(c) for c in checks) >= 3


def select_prompt_references(retrieved: list[dict], limit: int) -> list[dict]:
    chosen: list[dict] = []
    seen: set[str] = set()
    for item in retrieved:
        signature = normalize_for_dedup(item.get("essay", ""))[:600]
        if signature in seen:
            continue
        seen.add(signature)
        chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


def build_structured_fallback(query: str, retrieved: list[dict], score_label: str) -> str:
    weighted_scores = []

    # 🔥 STEP 1: collect valid scores
    scores = []
    for item in retrieved:
        try:
            scores.append(float(item["score"]))
        except (TypeError, ValueError):
            continue

    mean_score = sum(scores) / len(scores) if scores else 0

    # 🔥 STEP 2: weighted scoring with filtering
    for item in retrieved:
        try:
            score = float(item["score"])
        except (TypeError, ValueError):
            continue

        distance = float(item["distance"])

        # 🚀 FILTER BAD NEIGHBORS
        if distance > 1.2:
            continue

        similarity = 1.0 / (1.0 + distance)

        # 🚀 SCORE-AWARE PENALTY
        score_penalty = 1 / (1 + 0.5 * abs(score - mean_score))

        final_weight = similarity * score_penalty

        weighted_scores.append((score, final_weight))

    # 🚀 FALLBACK IF TOO FEW NEIGHBORS
    if len(weighted_scores) < 3:
        weighted_scores = []
        for item in retrieved:
            try:
                score = float(item["score"])
            except:
                continue

            similarity = 1.0 / (1.0 + float(item["distance"]))
            weighted_scores.append((score, similarity))

    # 🔥 STEP 3: final prediction
    if weighted_scores:
        numerator = sum(score * w for score, w in weighted_scores)
        denominator = sum(w for _, w in weighted_scores) or 1.0
        predicted = round((numerator / denominator) * 2) / 2
    else:
        predicted = "N/A"

    return (
        "Estimated Band:\n"
        f"- {predicted} (retrieval-weighted estimate from nearest essays by {score_label})\n\n"
        "Strengths:\n"
        "- Main position is present and relevant to the prompt.\n"
        "- Core ideas are understandable and supported to some extent.\n\n"
        "Weaknesses:\n"
        "- Grammar accuracy and sentence control reduce clarity in parts.\n"
        "- Lexical repetition limits precision and style.\n"
        "- Cohesion can be improved with clearer paragraph links.\n\n"
        "Top 3 Improvements:\n"
        "1. Use one clear topic sentence per body paragraph.\n"
        "2. Add one concrete example after each main claim.\n"
        "3. Proofread for articles, verb forms, and punctuation.\n\n"
        "One Improved Sample Paragraph:\n"
        "Overall, a stronger response should begin with a clear overview, followed by direct "
        "comparisons of the most important features. In each body paragraph, present one key point, "
        "support it with precise detail, and explain why that detail matters. This approach improves "
        "coherence, makes your argument easier to follow, and helps you achieve a more consistent "
        "academic style."
    )


def resolve_score_column(df: pd.DataFrame) -> str | None:
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in ("band", "overall", "score"):
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    for column in df.columns:
        lowered = column.lower()
        if "band" in lowered or "overall" in lowered or "score" in lowered:
            return column

    return None

def clean_markdown(text: str) -> str:
    text = text.replace("**", "")
    text = text.replace("*", "")
    return text


def clean_corrections(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        if "→" in line:
            parts = line.split("→")
            if len(parts) == 2:
                left = parts[0].strip().strip('- ').strip('"')
                right = parts[1].strip().strip('"')

                if left == right:
                    continue

        cleaned.append(line)

    return "\n".join(cleaned)

def trim_after_section(text: str) -> str:
    if "Mistakes and Corrections:" in text:
        parts = text.split("Mistakes and Corrections:")
        before = parts[0]
        after = parts[1]

        lines = after.split("\n")
        cleaned = []

        for line in lines:
            if "→" in line:
                cleaned.append(line)
            elif line.strip() == "":
                continue
            else:
                break

        return before + "Mistakes and Corrections:\n" + "\n".join(cleaned)

    return text

def load_optional_generator():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

    except ModuleNotFoundError:
        return None

    print("Loading Qwen2.5-1.5B 4-bit model... (may take a minute)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True   # 🔥 ADD THIS
        )
        
        model.eval()

        # 🔥 FORCE RESET ENTIRE GENERATION CONFIG
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig(
            max_new_tokens=20,
            min_new_tokens=0,
            min_length=0,
            do_sample=False,
        )

        # 🔥 FIX: override bad default generation config
        model.generation_config.min_new_tokens = 0
        model.generation_config.min_length = 0
        model.generation_config.max_new_tokens = 20

        def generate_text(prompt: str, max_new_tokens: int = 20) -> str:
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an IELTS Writing examiner. Provide clear, structured, "
                            "practical feedback in plain English."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                rendered_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                rendered_prompt = prompt

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            max_model_len = getattr(tokenizer, "model_max_length", 4096)
            if not isinstance(max_model_len, int) or max_model_len <= 0 or max_model_len > 32768:
                max_model_len = 4096
            max_model_len = min(max_model_len, 4096)

            inputs = tokenizer(
                rendered_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_model_len,
            ).to(device)

            # 🔥 HARD LIMIT INPUT LENGTH (THIS FIXES YOUR WARNINGS)
            # 🔥 HARD LIMIT INPUT SIZE (THIS IS THE REAL FIX)
            if inputs["input_ids"].shape[1] > 300:
                inputs["input_ids"] = inputs["input_ids"][:, -300:]
                inputs["attention_mask"] = inputs["attention_mask"][:, -300:]

            eos_token_id = tokenizer.eos_token_id
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,

                    # 🔥 FORCE EVERYTHING HERE (THIS IS THE REAL FINAL FIX)
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=0,
                    min_length=0,
                    max_length=inputs["input_ids"].shape[1] + 20,

                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 🔥 CUT OFF weird chat artifacts
            response = response.split("<tool_call>")[0]
            response = response.split("Human:")[0]
            response = response.split("Assistant:")[0]

            response = response.strip()

            return response

        return generate_text
    except Exception as exc:
        print(f"Warning: local GPTQ model load failed ({exc}). Using fallback retrieval mode.")
        return None


def load_metadata(meta_path: Path):
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def map_to_row_index(faiss_index: int, metadata) -> int:
    if metadata is None:
        return faiss_index
    if isinstance(metadata, list) and 0 <= faiss_index < len(metadata):
        mapped = metadata[faiss_index]
        if isinstance(mapped, (int, np.integer)):
            return int(mapped)
    return faiss_index


def retrieve_essays(
    query: str,
    top_k: int,
    embed_model: SentenceTransformer,
    index,
    metadata,
    essays_df: pd.DataFrame,
    score_column: str | None,
) -> list[dict]:
    query_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    retrieved = []
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if distance > 0.8:
            continue
        row_idx = map_to_row_index(int(idx), metadata)
        if row_idx < 0 or row_idx >= len(essays_df):
            continue

        row = essays_df.iloc[row_idx]
        score_value = row.get(score_column, "N/A") if score_column else "N/A"
        retrieved.append(
            {
                "rank": rank,
                "row_index": row_idx,
                "distance": float(distance),
                "score": score_value,
                "essay": str(row.get("essay", "")),
            }
        )
    return retrieved


def print_retrieved_essays(retrieved: list[dict], score_label: str) -> None:
    print("\nRetrieved Essays:")
    for item in retrieved:
        header = (
            f"--- Rank {item['rank']} | Row {item['row_index'] + 1} | "
            f"{score_label}: {item['score']} | Distance: {item['distance']:.4f} ---"
        )
        print(header)
        print(item["essay"][:500] + "...\n")


def print_fallback_feedback(retrieved: list[dict]) -> None:
    numeric_scores = []
    for item in retrieved:
        try:
            numeric_scores.append(float(item["score"]))
        except (TypeError, ValueError):
            continue

    print("\nFallback feedback (no local GPTQ model loaded):")
    print("- Retrieved essays can still be used as quality references.")
    if numeric_scores:
        avg_score = sum(numeric_scores) / len(numeric_scores)
        print(f"- Average reference score from top results: {avg_score:.2f}")
        print(f"- Reference score range: {min(numeric_scores):.1f} to {max(numeric_scores):.1f}")
    else:
        print("- No numeric score field detected in retrieved rows.")
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        torch_cuda = str(torch.version.cuda)
    except Exception:
        cuda_available = False
        torch_cuda = "unknown"

    if cuda_available:
        print("- CUDA is available. You can try GPTQ with compatible CUDA torch + auto-gptq.")
        print("- Install command:")
        print("  pip install --no-build-isolation auto-gptq")
    else:
        print("- GPU GPTQ is currently unavailable in this environment (CPU-only torch detected).")
        print(f"- Detected torch CUDA version: {torch_cuda}")
        print("- If you want GPTQ, install a CUDA-enabled torch build first, then install auto-gptq.")


def main():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = load_metadata(METADATA_PATH)
    essays_df = pd.read_csv(PROCESSED_CSV_PATH)
    score_column = resolve_score_column(essays_df)
    score_label = score_column if score_column else "score"

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    generator = None
    if ENABLE_LLM_GENERATION:
        generator = load_optional_generator()
        if generator is None:
            print("Note: local GPTQ generation is unavailable. Using structured fallback mode.")
    
    if generator:
        print("\n=== SANITY TEST ===")
        print(generator("Hello, how are you?"))
        print("====================\n")

    print("IELTS RAG Feedback System")
    while True:
        query = input("\nEnter your essay topic or question (or 'exit' to quit):\n> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            print("Please enter a non-empty query.")
            continue
        lower_query = query.lower()
        if "python.exe" in lower_query or "rag_feedback_local.py" in lower_query:
            print("That looks like a shell command. Please paste your essay text instead.")
            continue
        query = sanitize_query(query)
        if len(query.split()) < 20:
            print("Please paste a full essay response (at least ~20 words).")
            continue

        retrieved = retrieve_essays(
            query=query,
            top_k=10,
            embed_model=embed_model,
            index=index,
            metadata=metadata,
            essays_df=essays_df,
            score_column=score_column,
        )
        if not retrieved:
            print("No essays were retrieved. Check index and metadata alignment.")
            continue

        print_retrieved_essays(retrieved, score_label=score_label)
        prompt_references = select_prompt_references(retrieved, MAX_REFERENCE_ESSAYS)
        retrieved_text = "\n\n".join(
            [
                (
                    f"Rank {item['rank']} | {score_label}: {item['score']}\n"
                    f"{truncate_words(item['essay'], MAX_WORDS_PER_REFERENCE)}"
                )
                for item in prompt_references
            ]
        )

        if generator is None:
            response = build_structured_fallback(query=query, retrieved=retrieved, score_label=score_label)
            print("\nRAG Feedback:\n")
            print(response)
            continue

        prompt = f"""
You are a strict IELTS Writing examiner.

Give realistic band scoring (do NOT be overly positive).
Be strict. Penalize grammar mistakes heavily.

Output MUST be plain text only. No bold, no markdown, no symbols.

You MUST include at least 3 items in "Mistakes and Corrections".
If there are fewer, create more based on grammar errors in the text.

Do NOT wrap the paragraph in quotation marks.

Each strength and weakness MUST quote a specific phrase from the student's writing.

You MUST follow the format EXACTLY.

- Do NOT use markdown (no **, no bold, no symbols).
- Do NOT add any extra sections.
- Do NOT rename any headings.
- Output must be plain text only.
- If you fail to follow format, the answer is incorrect.
- If you add anything outside the required format, your answer is WRONG.

Estimated Band:
(only a number from 4.0 to 9.0)

Strengths (based ONLY on student's writing):
- ...

Weaknesses (based ONLY on student's writing):
- ...

Top 3 Improvements (based ONLY on student's writing):
1. ...
2. ...
3. ...

One Improved Sample Paragraph:
(Write a FULL paragraph of 80-120 words. Do not cut off mid-sentence.)

Mistakes and Corrections:
- "wrong word" → "correct word"

Only include REAL mistakes from the student's text.

- The original word MUST be incorrect.
- The corrected word MUST be different.
- NEVER include identical pairs like "word" → "word".
- Each correction must fix a clear spelling or grammar error.

If the phrase is already correct, DO NOT include it.
Each correction must clearly fix an error.

This is IELTS Academic Task 1.

- Do NOT suggest personal opinions, stories, or anecdotes.
- Focus only on data description, comparisons, and trends.

Student submission:
{query}

Reference essays (DO NOT copy from these. Use only for general quality comparison):

{retrieved_text}

IMPORTANT:
- Only evaluate the STUDENT submission.
- Do NOT quote or correct reference essays.
- All mistakes MUST come from the student text only.
- If a sentence is not in the student submission, DO NOT use it.
"""
        
        response = generator(prompt)
        
        response = clean_markdown(response)
        response = clean_corrections(response)
        response = trim_after_section(response)

        low_quality = is_degenerate_output(response) or not is_valid_response(response)

        if low_quality:
            if SHOW_RAW_OUTPUT_ON_FALLBACK:
                print("\nRaw LLM output preview (debug):")
                if response:
                    preview = response[:RAW_OUTPUT_PREVIEW_CHARS]
                    if len(response) > RAW_OUTPUT_PREVIEW_CHARS:
                        preview += " ..."
                    print(preview)
                else:
                    print("[empty output from model]")

            print("Warning: model output was low-quality; using structured fallback.")
            response = build_structured_fallback(
                query=query,
                retrieved=retrieved,
                score_label=score_label
            )

        print("\nRAG Feedback:\n")
        print(response if response else "[No output generated]")


if __name__ == "__main__":
    main()

