"""Parse criterion subscores from HuggingFace dataset evaluation field."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


_CRITERION_PATTERNS = {
    "task_response": [
        r"Suggested Band Score\s*\(Task Achievement\):\s*([\d.]+)",
        r"Suggested Band Score\s*\(Task Response\):\s*([\d.]+)",
        r"(?:Task Achievement|Task Response)[^:]*:\s*([\d.]+)",
        r"(?:Score for|score for)\s*(?:Task Achievement|Task Response)[^:]*:\s*([\d.]+)",
        r"(?:Task Achievement|Task Response)\s*(?:score|Score)\s*:?\s*([\d.]+)",
    ],
    "coherence": [
        r"Suggested Band Score\s*\(Coherence and Cohesion\):\s*([\d.]+)",
        r"Suggested Band Score\s*\(Cohesion[^)]*\):\s*([\d.]+)",
        r"Coherence and Cohesion[^:]*:\s*([\d.]+)",
        r"(?:Score for|score for)\s*(?:Coherence|Cohesion)[^:]*:\s*([\d.]+)",
        r"(?:Coherence and Cohesion|Cohesion)\s*(?:score|Score)\s*:?\s*([\d.]+)",
    ],
    "lexical": [
        r"Suggested Band Score\s*\(Lexical Resource[^)]*\):\s*([\d.]+)",
        r"Suggested Band Score\s*\(Vocabulary[^)]*\):\s*([\d.]+)",
        r"(?:Lexical Resource|Vocabulary)[^:]*:\s*([\d.]+)",
        r"(?:Score for|score for)\s*(?:Lexical|Vocabulary)[^:]*:\s*([\d.]+)",
        r"(?:Lexical Resource|Vocabulary)\s*(?:score|Score)\s*:?\s*([\d.]+)",
    ],
    "grammar": [
        r"Suggested Band Score\s*\(Grammatical[^)]*\):\s*([\d.]+)",
        r"Suggested Band Score\s*\(Grammar[^)]*\):\s*([\d.]+)",
        r"(?:Grammatical Range and Accuracy|Grammar)[^:]*:\s*([\d.]+)",
        r"(?:Score for|score for)\s*(?:Grammatical|Grammar)[^:]*:\s*([\d.]+)",
        r"(?:Grammatical Range and Accuracy|Grammar)\s*(?:score|Score)\s*:?\s*([\d.]+)",
    ],
}

_OVERALL_PATTERNS = [
    r"Suggested Overall Band Score:\s*([\d.]+)",
    r"suggested overall band score would be\s*([\d.]+)",
    r"suggested overall band score for this essay is\s*([\d.]+)",
    r"overall band score for the essay is\s*([\d.]+)",
    r"essay deserves an overall band score of\s*([\d.]+)",
    r"overall band score of\s*([\d.]+)",
    r"## Overall Band Score:\s*([\d.]+)",
    r"\*\*Overall Band Score:\*\*\s*([\d.]+)",
    r"### Overall Band Score:\s*([\d.]+)",
    r"(?:Overall|overall)\s*(?:Band|band)?\s*(?:Score|score)\s*:?\s*([\d.]+)",
]


def _extract_first(text: str, patterns: list[str]) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if 0 <= val <= 9:
                    return round(val * 2) / 2
            except (ValueError, IndexError):
                continue
    return None


def parse_subscores(evaluation_text: str | None) -> dict[str, float | None]:
    if not evaluation_text or not isinstance(evaluation_text, str):
        return {"task_response": None, "coherence": None, "lexical": None, "grammar": None, "overall": None}

    result = {}
    for criterion, patterns in _CRITERION_PATTERNS.items():
        result[criterion] = _extract_first(evaluation_text, patterns)

    result["overall"] = _extract_first(evaluation_text, _OVERALL_PATTERNS)

    return result


def add_parsed_subscores(df: pd.DataFrame, evaluation_column: str = "evaluation") -> pd.DataFrame:
    parsed = df[evaluation_column].apply(parse_subscores)
    parsed_df = pd.json_normalize(parsed)

    for col in ["task_response", "coherence", "lexical", "grammar", "overall"]:
        if col in parsed_df.columns and parsed_df[col].notna().any():
            df[col] = parsed_df[col].where(parsed_df[col].notna(), df.get(col, None))

    return df


def get_subscore_coverage(df: pd.DataFrame) -> dict[str, float]:
    coverage = {}
    for col in ["task_response", "coherence", "lexical", "grammar", "overall"]:
        sub_col = f"subscore_{col}"
        if sub_col in df.columns:
            coverage[col] = df[sub_col].notna().sum()
        elif col in df.columns:
            coverage[col] = df[col].notna().sum()
        else:
            coverage[col] = 0
    return coverage
