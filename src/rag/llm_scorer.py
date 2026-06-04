from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable


_FAILURE_CATEGORIES = {
    "malformed_output": "Output could not be parsed as JSON",
    "missing_fields": "JSON missing required score fields",
    "out_of_range": "Score value outside valid IELTS range (0-9)",
    "extra_prose": "Non-JSON text detected before/after JSON block",
    "inference_failure": "Model inference returned no output",
}


class LLMScorer:
    def __init__(
        self,
        generator: Callable | None = None,
        prompt_template: str | Path | None = None,
        max_retries: int = 1,
        max_new_tokens: int = 20,
    ):
        self.generator = generator
        self.max_retries = max_retries
        self.max_new_tokens = max_new_tokens
        self.failure_log: list[dict[str, Any]] = []

        if prompt_template is not None:
            with open(prompt_template, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        else:
            _default = Path(__file__).resolve().parents[2] / "configs" / "prompts" / "scoring_prompt.txt"
            if _default.exists():
                with open(_default, "r", encoding="utf-8") as f:
                    self.prompt_template = f.read()
            else:
                self.prompt_template = self._builtin_prompt()

    @staticmethod
    def _builtin_prompt() -> str:
        return (
            "You are a strict IELTS Writing examiner. Evaluate the student essay below "
            "and return a JSON object with criterion scores.\n\n"
            'Rate each criterion on the IELTS scale (0-9, in half-band increments):\n'
            '- "task_response": How well the essay addresses the prompt.\n'
            '- "coherence": How logically the essay is organized.\n'
            '- "lexical": The range and precision of vocabulary.\n'
            '- "grammar": The range and accuracy of grammatical structures.\n\n'
            "Return ONLY a valid JSON object with these four keys and numeric values.\n\n"
            "Examples:\n{examples}\n\n"
            "Student Essay:\n{essay}\n\n"
            "Think step by step, then output the JSON."
        )

    def format_prompt(
        self, essay: str, neighbors: list[Any] | None = None,
        distilbert_scores: dict[str, float] | None = None,
    ) -> str:
        examples_text = ""
        if neighbors:
            for i, n in enumerate(neighbors[:3]):
                ex_essay = n.essay[:200] if hasattr(n, "essay") else str(n)[:200]
                ex_band = n.band if hasattr(n, "band") else "N/A"
                examples_text += (
                    f"Example {i+1} (Band {ex_band}):\n{ex_essay}\n\n"
                )

        db_scores_text = ""
        if distilbert_scores:
            parts = []
            for k in ("task_response", "coherence", "lexical", "grammar", "overall"):
                if k in distilbert_scores:
                    parts.append(f"{k}: {distilbert_scores[k]:.1f}")
            db_scores_text = ", ".join(parts)

        prompt = self.prompt_template.replace("{essay}", essay)
        prompt = prompt.replace("{examples}", examples_text.strip())
        prompt = prompt.replace("{distilbert_scores}", db_scores_text)
        return prompt

    def score(
        self, essay: str, neighbors: list[Any] | None = None,
        distilbert_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if self.generator is None:
            return {"error": "No generator available", "llm_scores": None}

        prompt = self.format_prompt(essay, neighbors, distilbert_scores)

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.generator(prompt, self.max_new_tokens)
            except Exception as exc:
                self._log_failure("inference_failure", str(exc), prompt)
                return {"error": str(exc), "llm_scores": None}

            if not raw or not raw.strip():
                self._log_failure("inference_failure", "Empty output", prompt)
                return {"error": "Empty output", "llm_scores": None}

            parsed = self._parse_response(raw)
            if parsed is not None:
                scores = self._validate_scores(parsed)
                if scores is not None:
                    return {"llm_scores": scores, "raw_output": raw, "error": None}

        return {"error": "Max retries exceeded", "llm_scores": None, "raw_output": raw}

    def _parse_response(self, raw: str) -> dict[str, float] | None:
        raw = raw.strip()
        has_prose_before = False
        has_prose_after = False

        json_start = raw.find("{")
        json_end = raw.rfind("}")

        if json_start == -1 or json_end == -1:
            self._log_failure("malformed_output", "No JSON found", raw[:200])
            return None

        if json_start > 0:
            has_prose_before = True

        if json_end < len(raw) - 1:
            trailing = raw[json_end + 1:].strip()
            if trailing:
                has_prose_after = True

        if has_prose_before or has_prose_after:
            self._log_failure("extra_prose", "Text outside JSON block", raw[:200])

        json_str = raw[json_start: json_end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            self._log_failure("malformed_output", "JSON parse error", json_str[:200])
            return None

        if not isinstance(data, dict):
            self._log_failure("malformed_output", "JSON not a dict", json_str[:200])
            return None

        return data

    def _validate_scores(self, data: dict[str, Any]) -> dict[str, float] | None:
        expected_keys = ["task_response", "coherence", "lexical", "grammar", "overall"]
        actual_keys = set(data.keys())
        expected_key_set = set(expected_keys)
        if actual_keys != expected_key_set:
            missing = sorted(expected_key_set - actual_keys)
            extra = sorted(actual_keys - expected_key_set)
            detail = []
            if missing:
                detail.append(f"missing={missing}")
            if extra:
                detail.append(f"extra={extra}")
            self._log_failure("missing_fields", ", ".join(detail), str(data))
            return None

        scores: dict[str, float] = {}

        for key in expected_keys:
            if key not in data:
                self._log_failure("missing_fields", f"Missing key: {key}", str(data))
                return None
            try:
                val = float(data[key])
            except (TypeError, ValueError):
                self._log_failure("missing_fields", f"Non-numeric {key}: {data[key]}", str(data))
                return None
            if val < 0.0 or val > 9.0:
                self._log_failure("out_of_range", f"{key}={val}", str(data))
                return None
            scores[key] = round(val * 2) / 2

        return scores

    def _log_failure(self, category: str, detail: str, snippet: str) -> None:
        self.failure_log.append({
            "category": category,
            "detail": detail,
            "snippet": snippet[:300],
        })

    def get_failure_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for entry in self.failure_log:
            cat = entry["category"]
            summary[cat] = summary.get(cat, 0) + 1
        return summary

    def is_available(self) -> bool:
        return self.generator is not None
