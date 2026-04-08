from __future__ import annotations

import re
from dataclasses import dataclass

from cgbv.data.base import DataSample

_SYSTEM_PROMPT = (
    "You are a careful logical reasoner. "
    "Use zero-shot chain-of-thought reasoning based only on the given premises. "
    "If the premises entail the conclusion, answer true. "
    "If they contradict the conclusion, answer false. "
    "If the conclusion is neither proved nor disproved, answer uncertain. "
    "End your response with exactly one line in this format: "
    "Final answer: <true|false|uncertain>"
)

_DATASET_HINTS = {
    "folio": "Natural-language first-order reasoning.",
    "prontoqa": "Rule-style commonsense logical reasoning.",
    "proofwriter": "Multi-hop deductive reasoning from stated facts and rules.",
    "proverqa": "Deductive reasoning with a three-way label space.",
}

_ANSWER_PATTERNS = (
    r"final answer\s*[:：-]\s*(.+)",
    r"final verdict\s*[:：-]\s*(.+)",
    r"answer\s*[:：-]\s*(.+)",
    r"label\s*[:：-]\s*(.+)",
)


@dataclass(frozen=True)
class ParsedPrediction:
    verdict: str | None
    parse_status: str
    parse_error: str | None = None


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    if not text:
        return None
    if re.search(
        r"\b(uncertain|unknown|undetermined)\b|"
        r"not enough information|insufficient information|cannot be determined",
        text,
    ):
        return "uncertain"
    if re.search(r"\b(false|refuted|contradicted)\b", text):
        return "false"
    if re.search(r"\b(true|entailed|supported)\b", text):
        return "true"
    return None


def _label_mentions(text: str) -> list[str]:
    lowered = text.strip().lower()
    matches: list[str] = []
    if re.search(
        r"\b(uncertain|unknown|undetermined)\b|"
        r"not enough information|insufficient information|cannot be determined",
        lowered,
    ):
        matches.append("uncertain")
    if re.search(r"\b(false|refuted|contradicted)\b", lowered):
        matches.append("false")
    if re.search(r"\b(true|entailed|supported)\b", lowered):
        matches.append("true")
    return list(dict.fromkeys(matches))


def _extract_label(line: str) -> str | None:
    unique = _label_mentions(line)
    if len(unique) == 1:
        return unique[0]
    return None


def build_prompt_text(sample: DataSample) -> str:
    hint = _DATASET_HINTS.get(sample.dataset, "Logical reasoning task.")
    premises = "\n".join(
        f"{idx}. {premise}" for idx, premise in enumerate(sample.premises, start=1)
    ) or "1. (no premises provided)"
    options = ""
    if sample.options:
        options = "\nOptions:\n" + "\n".join(sample.options)

    return (
        f"Dataset: {sample.dataset}\n"
        f"Task hint: {hint}\n\n"
        "Premises:\n"
        f"{premises}\n\n"
        f"Conclusion:\n{sample.conclusion}\n"
        f"{options}\n\n"
        "Think step by step before deciding.\n"
        "Return only one final label from {true, false, uncertain}.\n"
        "Use uncertain when the premises are insufficient to prove or disprove the conclusion."
    )


def build_messages(sample: DataSample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt_text(sample)},
    ]


def parse_prediction(response: str) -> ParsedPrediction:
    text = (response or "").strip()
    if not text:
        return ParsedPrediction(
            verdict=None,
            parse_status="empty_response",
            parse_error="LLM returned an empty response.",
        )

    for pattern in _ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        for match in reversed(matches):
            verdict = _extract_label(match.group(1))
            if verdict is not None:
                return ParsedPrediction(verdict=verdict, parse_status="parsed")

    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        verdict = _extract_label(line)
        if verdict is not None:
            return ParsedPrediction(verdict=verdict, parse_status="parsed_fallback")

    return ParsedPrediction(
        verdict=None,
        parse_status="missing_label",
        parse_error="Could not extract a final label from the LLM response.",
    )
