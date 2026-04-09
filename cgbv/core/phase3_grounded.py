from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from cgbv.core.grounded_template_ir import (
    IRValidationError,
    evaluate_grounded_template_ir,
    parse_grounded_template_ir,
    render_grounded_template_ir,
    validate_grounded_template_ir,
)
from cgbv.core.logic_compiler import SymbolTable, extract_logic_predicates
from cgbv.llm.base import LLMClient
from cgbv.prompts.prompt_engine import PromptEngine

logger = logging.getLogger(__name__)


@dataclass
class GroundingAttempt:
    """One Phase 3 attempt for a single sentence."""

    attempt_num: int
    messages: list[dict[str, str]] = field(default_factory=list)
    raw_output: str = ""
    extracted_formula: str = ""
    validation_error: str | None = None
    accepted: bool = False


@dataclass
class GroundedFormula:
    sentence_index: int
    nl_sentence: str
    # Keep compatibility with Phase 4/result schema; this is debug_render now.
    formula_code: str
    # New IR payload for deterministic local execution.
    template_ir: dict[str, Any] | None = None
    failed: bool = False
    attempts: list[GroundingAttempt] = field(default_factory=list)
    error: str | None = None


@dataclass
class Phase3Result:
    grounded: list[GroundedFormula]


@dataclass
class GroundingTemplate:
    sentence_index: int
    nl_sentence: str
    template_ir: dict[str, Any] | None
    debug_render: str
    failed: bool = False
    attempts: list[GroundingAttempt] = field(default_factory=list)
    error: str | None = None


@dataclass
class Phase3TemplateResult:
    templates: list[GroundingTemplate]


async def reground_with_hint(
    idx: int,
    sentence: str,
    domain_desc_str: str,
    domain: dict,
    current_formula: str,
    expected_truth: bool,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int = 2,
    world_assumption: str = "owa",
    solver: "Z3Solver | None" = None,
) -> GroundedFormula:
    """
    Backward-compatible wrapper.

    This route now regenerates a grounded-template IR and returns a Phase4-compatible
    GroundedFormula (formula_code=debug_render, template_ir=IR).
    """
    hint = (
        f"Previous grounded template `{current_formula}` mismatched expected truth "
        f"value {expected_truth} on a witness. Re-derive the template IR from sentence meaning."
    )
    tmpl = await retemplate_with_hint(
        idx=idx,
        sentence=sentence,
        domain_schema_str=domain_desc_str,
        domain=domain,
        current_template=current_formula,
        hint=hint,
        llm=llm,
        prompt_engine=prompt_engine,
        max_retries=max_retries,
        world_assumption=world_assumption,
        solver=solver,
        sentence_logic=None,
        symbol_context={},
    )
    return GroundedFormula(
        sentence_index=idx,
        nl_sentence=sentence,
        formula_code=tmpl.debug_render,
        template_ir=tmpl.template_ir,
        failed=tmpl.failed,
        attempts=tmpl.attempts,
        error=tmpl.error,
    )


async def retemplate_with_hint(
    idx: int,
    sentence: str,
    domain_schema_str: str,
    domain: dict,
    current_template: str,
    hint: str,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int = 2,
    world_assumption: str = "owa",
    solver: "Z3Solver | None" = None,
    sentence_logic: Any | None = None,
    symbol_context: dict[str, object] | None = None,
) -> GroundingTemplate:
    """Re-generate one grounded-template IR with semantic hint feedback."""
    symbol_context = symbol_context or {}
    messages = _build_template_messages(
        sentence=sentence,
        domain_schema_str=domain_schema_str,
        prompt_engine=prompt_engine,
        world_assumption=world_assumption,
        sentence_logic=sentence_logic,
        symbol_context=symbol_context,
        sentence_index=idx,
    )
    messages = messages + [
        {"role": "assistant", "content": current_template},
        _build_template_retry_message(
            sentence=sentence,
            domain_schema_str=domain_schema_str,
            previous_output=current_template,
            last_error=hint,
            attempt_num=1,
            max_attempts=max_retries + 1,
            prompt_engine=prompt_engine,
            world_assumption=world_assumption,
            sentence_logic=sentence_logic,
            symbol_context=symbol_context,
            sentence_index=idx,
        ),
    ]
    return await _retry_template_generation(
        idx=idx,
        sentence=sentence,
        domain=domain,
        llm=llm,
        max_retries=max_retries,
        messages=messages,
        prompt_engine=prompt_engine,
        domain_schema_str=domain_schema_str,
        world_assumption=world_assumption,
        solver=solver,
        sentence_logic=sentence_logic,
        symbol_context=symbol_context,
    )


async def generate_templates(
    sentences: list[str],
    domain_schema_str: str,
    domain: dict,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int = 2,
    world_assumption: str = "owa",
    solver: "Z3Solver | None" = None,
    dsl_payload: dict[str, object] | None = None,
    symbol_table: SymbolTable | None = None,
    batch_size: int = 0,
) -> Phase3TemplateResult:
    """Generate witness-independent grounded-template IR (Template-Once)."""
    n = len(sentences)
    if batch_size <= 0:
        batch_size = n

    all_templates: list[GroundingTemplate] = []
    for i in range(0, n, batch_size):
        batch_sentences = sentences[i : i + batch_size]
        templates = await _generate_batch(
            sentences=batch_sentences,
            start_index=i,
            domain_schema_str=domain_schema_str,
            domain=domain,
            llm=llm,
            prompt_engine=prompt_engine,
            max_retries=max_retries,
            world_assumption=world_assumption,
            solver=solver,
            dsl_payload=dsl_payload,
            symbol_table=symbol_table,
        )
        all_templates.extend(templates)

    return Phase3TemplateResult(templates=all_templates)


async def generate_templates_partial(
    indices: set[int],
    sentences: list[str],
    domain_schema_str: str,
    domain: dict,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int = 2,
    world_assumption: str = "owa",
    solver: "Z3Solver | None" = None,
    dsl_payload: dict[str, object] | None = None,
    symbol_table: SymbolTable | None = None,
) -> list[GroundingTemplate]:
    """Generate templates only for selected sentence indices."""
    templates: list[GroundingTemplate] = []
    for idx in sorted(indices):
        if idx < 0 or idx >= len(sentences):
            continue
        tmpl = await _template_one(
            idx=idx,
            sentence=sentences[idx],
            domain_schema_str=domain_schema_str,
            domain=domain,
            llm=llm,
            prompt_engine=prompt_engine,
            max_retries=max_retries,
            world_assumption=world_assumption,
            solver=solver,
            sentence_logic=_logic_for_index(dsl_payload, idx),
            symbol_context=_symbol_context(dsl_payload, symbol_table),
        )
        templates.append(tmpl)
    return templates


async def _generate_batch(
    sentences: list[str],
    start_index: int,
    domain_schema_str: str,
    domain: dict,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int,
    world_assumption: str,
    solver: "Z3Solver | None",
    dsl_payload: dict[str, object] | None = None,
    symbol_table: SymbolTable | None = None,
) -> list[GroundingTemplate]:
    # Reliability-first mode:
    # Batch generation is currently unstable for the active model family and
    # repeatedly falls back after structural parse failures. Run per sentence
    # directly to avoid paying batch-failure token/latency tax.
    symbol_context = _symbol_context(dsl_payload, symbol_table)
    fallback: list[GroundingTemplate] = []
    for i, sentence in enumerate(sentences):
        idx = start_index + i
        fallback.append(
            await _template_one(
                idx=idx,
                sentence=sentence,
                domain_schema_str=domain_schema_str,
                domain=domain,
                llm=llm,
                prompt_engine=prompt_engine,
                max_retries=max_retries,
                world_assumption=world_assumption,
                solver=solver,
                sentence_logic=_logic_for_index(dsl_payload, idx),
                symbol_context=symbol_context,
            )
        )
    return fallback


async def _template_one(
    idx: int,
    sentence: str,
    domain_schema_str: str,
    domain: dict,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int,
    world_assumption: str,
    solver: "Z3Solver | None",
    sentence_logic: Any | None,
    symbol_context: dict[str, object],
) -> GroundingTemplate:
    messages = _build_template_messages(
        sentence=sentence,
        domain_schema_str=domain_schema_str,
        prompt_engine=prompt_engine,
        world_assumption=world_assumption,
        sentence_logic=sentence_logic,
        symbol_context=symbol_context,
        sentence_index=idx,
    )
    return await _retry_template_generation(
        idx=idx,
        sentence=sentence,
        domain=domain,
        llm=llm,
        max_retries=max_retries,
        messages=messages,
        prompt_engine=prompt_engine,
        domain_schema_str=domain_schema_str,
        world_assumption=world_assumption,
        solver=solver,
        sentence_logic=sentence_logic,
        symbol_context=symbol_context,
    )


async def _retry_template_generation(
    idx: int,
    sentence: str,
    domain: dict[str, Any],
    llm: LLMClient,
    max_retries: int,
    messages: list[dict[str, str]],
    prompt_engine: PromptEngine,
    domain_schema_str: str,
    world_assumption: str,
    solver: "Z3Solver | None",
    sentence_logic: Any | None,
    symbol_context: dict[str, object],
) -> GroundingTemplate:
    attempts: list[GroundingAttempt] = []
    last_error: str | None = None
    raw_output = ""
    max_attempts = max_retries + 1

    for attempt in range(max_attempts):
        if attempt > 0:
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                _build_template_retry_message(
                    sentence=sentence,
                    domain_schema_str=domain_schema_str,
                    previous_output=_normalise_output_for_prompt(raw_output),
                    last_error=last_error or "Unknown error",
                    attempt_num=attempt + 1,
                    max_attempts=max_attempts,
                    prompt_engine=prompt_engine,
                    world_assumption=world_assumption,
                    sentence_logic=sentence_logic,
                    symbol_context=symbol_context,
                    sentence_index=idx,
                ),
            ]

        attempt_record = GroundingAttempt(
            attempt_num=attempt + 1,
            messages=_snapshot_messages(messages),
        )
        raw_output = await llm.complete_with_retry(messages)
        attempt_record.raw_output = raw_output

        payload, extracted_text, extract_error = _extract_template_json(raw_output)
        attempt_record.extracted_formula = extracted_text
        if payload is None:
            last_error = extract_error or "Model output did not contain a JSON object."
            attempt_record.validation_error = last_error
            attempts.append(attempt_record)
            continue

        try:
            template = _materialise_template(
                idx=idx,
                sentence=sentence,
                payload=payload,
                domain=domain,
                symbol_context=symbol_context,
                sentence_logic=sentence_logic,
                solver=solver,
            )
        except ValueError as e:
            last_error = str(e)
            attempt_record.validation_error = last_error
            attempts.append(attempt_record)
            continue

        attempt_record.accepted = True
        attempts.append(attempt_record)
        template.attempts = attempts
        return template

    return GroundingTemplate(
        sentence_index=idx,
        nl_sentence=sentence,
        template_ir=None,
        debug_render="",
        failed=True,
        attempts=attempts,
        error=(
            f"Template generation failed after {max_attempts} attempts. "
            f"Last error: {last_error}"
        ),
    )


def _materialise_template(
    idx: int,
    sentence: str,
    payload: dict[str, Any],
    domain: dict[str, Any],
    symbol_context: dict[str, object],
    sentence_logic: Any | None,
    solver: "Z3Solver | None",
) -> GroundingTemplate:
    try:
        ir = parse_grounded_template_ir(payload, symbol_context)
        required_predicates = extract_logic_predicates(sentence_logic) if sentence_logic else None
        validate_grounded_template_ir(
            ir,
            symbol_context=symbol_context,
            required_predicates=required_predicates,
        )
        runtime_err = _validate_formula_runtime(ir, domain, solver)
        if runtime_err:
            raise IRValidationError(runtime_err)
    except (IRValidationError, KeyError, TypeError, ValueError) as e:
        raise ValueError(str(e)) from e

    debug_render = render_grounded_template_ir(ir)
    return GroundingTemplate(
        sentence_index=idx,
        nl_sentence=sentence,
        template_ir=ir,
        debug_render=debug_render,
    )


def _build_template_messages(
    sentence: str,
    domain_schema_str: str,
    prompt_engine: PromptEngine,
    world_assumption: str = "owa",
    sentence_logic: Any | None = None,
    symbol_context: dict[str, object] | None = None,
    sentence_index: int | None = None,
) -> list[dict[str, str]]:
    if sentence_index is None:
        sentence_index = 0
    user_content = prompt_engine.render(
        "phase3_template.j2",
        sentence=sentence,
        sentence_index=sentence_index,
        domain_schema=domain_schema_str,
        world_assumption=world_assumption,
        sentence_logic_json=json.dumps(sentence_logic or {}, ensure_ascii=False, indent=2),
        symbol_context_json=json.dumps(symbol_context or {}, ensure_ascii=False, indent=2),
        core_predicates=sorted(extract_logic_predicates(sentence_logic)) if sentence_logic else [],
    )
    return [{"role": "user", "content": user_content}]


def _build_template_batch_messages(
    sentences: list[str],
    domain_schema_str: str,
    prompt_engine: PromptEngine,
    world_assumption: str = "owa",
    sentence_logic_pairs: list[dict[str, object]] | None = None,
    symbol_context: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    user_content = prompt_engine.render(
        "phase3_template_batch.j2",
        sentences=sentences,
        domain_schema=domain_schema_str,
        world_assumption=world_assumption,
        sentence_logic_pairs=sentence_logic_pairs or [],
        symbol_context_json=json.dumps(symbol_context or {}, ensure_ascii=False, indent=2),
    )
    return [{"role": "user", "content": user_content}]


def _build_template_retry_message(
    sentence: str,
    domain_schema_str: str,
    previous_output: str,
    last_error: str,
    attempt_num: int,
    max_attempts: int,
    prompt_engine: PromptEngine,
    world_assumption: str = "owa",
    sentence_logic: Any | None = None,
    symbol_context: dict[str, object] | None = None,
    sentence_index: int | None = None,
) -> dict[str, str]:
    if sentence_index is None:
        sentence_index = 0
    user_content = prompt_engine.render(
        "phase3_retry.j2",
        sentence=sentence,
        sentence_index=sentence_index,
        domain_schema=domain_schema_str,
        previous_output=previous_output,
        last_error=last_error,
        attempt_num=attempt_num,
        max_attempts=max_attempts,
        world_assumption=world_assumption,
        sentence_logic_json=json.dumps(sentence_logic or {}, ensure_ascii=False, indent=2),
        symbol_context_json=json.dumps(symbol_context or {}, ensure_ascii=False, indent=2),
        core_predicates=sorted(extract_logic_predicates(sentence_logic)) if sentence_logic else [],
    )
    return {"role": "user", "content": user_content}


def _build_batch_retry_message(
    errors: dict[int, str],
    sentences: list[str],
    sentence_logic_pairs: list[dict[str, object]] | None,
    attempt_num: int,
    max_attempts: int,
) -> dict[str, str]:
    error_lines = "\n".join(f"[{i + 1}] {err}" for i, err in sorted(errors.items()))
    sentence_lines = []
    logic_lines = []
    for i in sorted(errors.keys()):
        pair = (sentence_logic_pairs or [])[i] if sentence_logic_pairs and i < len(sentence_logic_pairs) else {}
        sentence_idx = pair.get("sentence_index", i)
        core_predicates = pair.get("core_predicates", [])
        logic_json = str(pair.get("logic_json", "{}"))
        sentence_lines.append(
            f"[{i + 1}] sentence_index={sentence_idx}, sentence=\"{sentences[i]}\""
        )
        logic_lines.append(
            f"[{i + 1}] core_predicates={core_predicates}\n{logic_json}"
        )
    content = (
        f"Attempt {attempt_num}/{max_attempts}. Fix only errored entries.\n\n"
        f"Errors:\n{error_lines}\n\n"
        f"Sentences to fix:\n{'\n'.join(sentence_lines)}\n\n"
        f"Sentence DSL + required predicate/function names:\n{'\n\n'.join(logic_lines)}\n\n"
        "Output ONLY `[N] {JSON_OBJECT}` lines."
    )
    return {"role": "user", "content": content}


def _should_early_batch_fallback(errors: dict[int, str], batch_size: int) -> bool:
    if batch_size <= 0 or not errors:
        return False

    classes = [_classify_structural_batch_error(err) for err in errors.values()]
    classes = [cls for cls in classes if cls]
    if not classes:
        return False
    if len(classes) / batch_size < 0.7:
        return False

    counts: dict[str, int] = {}
    for cls in classes:
        counts[cls] = counts.get(cls, 0) + 1
    dominant = max(counts.values(), default=0)
    return dominant / batch_size >= 0.7


def _classify_structural_batch_error(error: str) -> str | None:
    err = error.strip()
    if err.startswith("Missing ["):
        return "missing_slot"
    if "Unsupported boolean op" in err:
        return "unsupported_bool_op"
    if "must be a JSON object" in err or "Invalid JSON IR" in err:
        return "json_shape"
    if "requires object field" in err or "requires object fields" in err:
        return "node_shape"
    if "requires non-empty" in err:
        return "node_shape"
    return None


def _snapshot_messages(messages: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "role": str(m.get("role", "")),
            "content": str(m.get("content", "")),
        }
        for m in messages
    ]


def _extract_template_json(raw: str) -> tuple[dict[str, Any] | None, str, str | None]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    text = text.strip().strip("`").strip()

    # 1) direct JSON object
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, json.dumps(obj, ensure_ascii=False), None
    except json.JSONDecodeError:
        pass

    # 2) first {...} block heuristic
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidate = text[first : last + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, candidate, None
        except json.JSONDecodeError as e:
            return None, candidate, f"Invalid JSON object: {e}"

    return None, text[:500], "Model output did not contain a JSON object."


def _normalise_output_for_prompt(raw: str) -> str:
    _, extracted, _ = _extract_template_json(raw)
    return extracted if extracted else raw.strip()


def _parse_batch_output(raw: str, num_sentences: int) -> dict[int, dict[str, Any]]:
    text = raw.strip()
    text = re.sub(r"^```(?:json|text)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    text = text.strip()

    # format A: JSON array
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            parsed: dict[int, dict[str, Any]] = {}
            for i, item in enumerate(obj):
                if i >= num_sentences:
                    break
                if isinstance(item, dict):
                    parsed[i] = item
            if parsed:
                return parsed
        if isinstance(obj, dict):
            parsed_obj = _parse_batch_output_dict(obj, num_sentences)
            if parsed_obj:
                return parsed_obj
    except json.JSONDecodeError:
        pass

    # format B: lines like [N] {...}
    result: dict[int, dict[str, Any]] = {}
    marker = re.compile(r"^\[(\d+)\]\s*", re.MULTILINE)
    matches = list(marker.finditer(text))
    if not matches:
        return result
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        one_based = int(match.group(1))
        if one_based < 1 or one_based > num_sentences:
            continue
        chunk = text[start:end].strip().strip("`").strip()
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            result[one_based - 1] = payload
    return result


def _parse_batch_output_dict(obj: dict[str, Any], num_sentences: int) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}

    # format C1: {"1": {...}, "2": {...}} / {"[1]": {...}}
    for key, value in obj.items():
        if not isinstance(value, dict):
            continue
        idx = _parse_batch_slot_key(str(key), num_sentences)
        if idx is None:
            continue
        result[idx] = value
    if result:
        return result

    # format C2: {"results":[{"index":1,"expr":...}, ...]}
    items = obj.get("results")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            idx_raw = item.get("index", item.get("slot", item.get("sentence_index")))
            idx = _parse_batch_index(idx_raw, num_sentences)
            if idx is None:
                continue
            payload = item.get("payload")
            if isinstance(payload, dict):
                result[idx] = payload
            elif "expr" in item:
                result[idx] = {"expr": item.get("expr")}
            elif isinstance(item.get("ir"), dict):
                result[idx] = item["ir"]
    return result


def _parse_batch_slot_key(key: str, num_sentences: int) -> int | None:
    m = re.fullmatch(r"\[(\d+)\]", key.strip())
    if m:
        return _parse_batch_index(int(m.group(1)), num_sentences, one_based=True)
    if key.strip().isdigit():
        return _parse_batch_index(int(key.strip()), num_sentences)
    return None


def _parse_batch_index(idx_raw: Any, num_sentences: int, *, one_based: bool = False) -> int | None:
    if isinstance(idx_raw, bool):
        return None
    if isinstance(idx_raw, int):
        idx = idx_raw
    elif isinstance(idx_raw, str) and idx_raw.strip().lstrip("-").isdigit():
        idx = int(idx_raw.strip())
    else:
        return None

    if one_based:
        idx -= 1
    else:
        # tolerate both 0-based and 1-based integer keys
        if 1 <= idx <= num_sentences:
            idx -= 1

    if idx < 0 or idx >= num_sentences:
        return None
    return idx


def _logic_for_index(dsl_payload: dict[str, object] | None, idx: int) -> Any | None:
    if not dsl_payload:
        return None
    sentences = dsl_payload.get("sentences", [])
    if isinstance(sentences, list) and idx < len(sentences):
        item = sentences[idx]
        if isinstance(item, dict):
            return item.get("logic")
    query = dsl_payload.get("query")
    if isinstance(query, dict) and isinstance(sentences, list) and idx == len(sentences):
        return query.get("logic")
    return None


def _symbol_context(
    dsl_payload: dict[str, object] | None,
    symbol_table: SymbolTable | None = None,
) -> dict[str, object]:
    payload = dsl_payload or {}
    ctx: dict[str, object] = {
        "sorts": payload.get("sorts", []),
        "functions": payload.get("functions", []),
        "constants": payload.get("constants", {}),
        "variables": payload.get("variables", []),
    }
    if symbol_table is None:
        return ctx

    if not ctx["sorts"] and getattr(symbol_table, "sorts", None):
        ctx["sorts"] = [
            {"name": name, "type": str(sort)}
            for name, sort in symbol_table.sorts.items()
        ]
    if not ctx["functions"] and getattr(symbol_table, "functions", None):
        functions: list[dict[str, object]] = []
        for name, fn in symbol_table.functions.items():
            domain = [str(fn.domain(i)) for i in range(fn.arity())]
            functions.append(
                {"name": name, "domain": domain, "range": str(fn.range())}
            )
        ctx["functions"] = functions
    if not ctx["constants"] and getattr(symbol_table, "constants", None):
        constants_by_sort: dict[str, dict[str, object]] = {}
        for name, const in symbol_table.constants.items():
            sort_name = str(const.sort())
            entry = constants_by_sort.setdefault(
                sort_name, {"sort": sort_name, "members": []}
            )
            entry["members"].append(name)
        ctx["constants"] = constants_by_sort
    if not ctx["variables"] and getattr(symbol_table, "variables", None):
        ctx["variables"] = [
            {"name": name, "sort": str(var.sort())}
            for name, var in symbol_table.variables.items()
        ]
    return ctx


def _validate_formula(
    template_ir: dict[str, Any] | str,
    domain: dict[str, Any],
    required_predicates: set[str] | None = None,
    symbol_context: dict[str, object] | None = None,
) -> str | None:
    """Backward-compatible static validator wrapper for tests/internal calls."""
    del domain
    symbol_context = symbol_context or {}
    try:
        ir = parse_grounded_template_ir(template_ir, symbol_context)
        validate_grounded_template_ir(
            ir,
            symbol_context=symbol_context,
            required_predicates=required_predicates,
        )
    except (IRValidationError, ValueError, TypeError, KeyError) as e:
        return str(e)
    return None


def _validate_formula_runtime(
    template_ir: dict[str, Any] | str,
    domain: dict[str, Any],
    solver: "Z3Solver | None",
) -> str | None:
    """Reject IR formulas that fail deterministic local execution."""
    del solver  # runtime validation now uses IR evaluator directly.
    if isinstance(template_ir, str):
        try:
            template_ir = json.loads(template_ir)
        except json.JSONDecodeError:
            return "Grounded template IR is not valid JSON."
    if not isinstance(template_ir, dict):
        return "Grounded template IR must be a JSON object."
    actual_truth = evaluate_grounded_template_ir(template_ir, domain)
    if actual_truth is None:
        return (
            "Grounded template IR is syntactically valid but could not be executed "
            "on the witness domain."
        )
    return None
