from __future__ import annotations

import ast
import asyncio
import logging
import re
from dataclasses import dataclass, field

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
    # Python boolean expression using truth["PredName(entity)"] lookups, e.g.:
    #   truth["allows_pets(olive_garden)"] and not truth["is_managed(pine_court)"]
    formula_code: str
    failed: bool = False
    attempts: list[GroundingAttempt] = field(default_factory=list)
    error: str | None = None


@dataclass
class Phase3Result:
    grounded: list[GroundedFormula]  # one per NL sentence (premises + conclusion)


async def run_phase3(
    sentences: list[str],       # [P1, P2, ..., Pn, C]  — premises + conclusion
    domain_desc_str: str,
    domain: dict,               # structured domain (for validation)
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int = 2,
    world_assumption: str = "owa",
) -> Phase3Result:
    """
    Phase 3: Grounded Re-Formalization.

    For each NL sentence, LLM writes a quantifier-free propositional formula
    on the finite domain of the boundary witness.

    All sentences are processed in parallel via asyncio.gather.
    """
    tasks = [
        _formalize_one(idx, sentence, domain_desc_str, domain, llm, prompt_engine, max_retries, world_assumption)
        for idx, sentence in enumerate(sentences)
    ]
    results: list[GroundedFormula] = await asyncio.gather(*tasks)
    return Phase3Result(grounded=list(results))


async def _formalize_one(
    idx: int,
    sentence: str,
    domain_desc_str: str,
    domain: dict,
    llm: LLMClient,
    prompt_engine: PromptEngine,
    max_retries: int,
    world_assumption: str = "owa",
) -> GroundedFormula:
    messages = _build_messages(sentence, domain_desc_str, prompt_engine, world_assumption)
    last_error: str | None = None
    raw_output = ""
    attempts: list[GroundingAttempt] = []
    max_attempts = max_retries + 1

    for attempt in range(max_attempts):
        if attempt > 0:
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                _build_retry_message(
                    sentence=sentence,
                    domain_desc_str=domain_desc_str,
                    previous_output=_normalise_output_for_prompt(raw_output),
                    last_error=last_error or "Unknown error",
                    attempt_num=attempt + 1,
                    max_attempts=max_attempts,
                    prompt_engine=prompt_engine,
                    world_assumption=world_assumption,
                ),
            ]
        attempt_record = GroundingAttempt(
            attempt_num=attempt + 1,
            messages=_snapshot_messages(messages),
        )
        raw_output = await llm.complete_with_retry(messages)
        attempt_record.raw_output = raw_output
        formula_code = _extract_formula(raw_output)
        attempt_record.extracted_formula = formula_code

        err = _validate_formula(formula_code, domain)
        if err:
            last_error = err
            attempt_record.validation_error = err
            attempts.append(attempt_record)
            logger.debug("Phase 3 idx=%d attempt %d: validation error: %s", idx, attempt + 1, err)
            continue

        attempt_record.accepted = True
        attempts.append(attempt_record)
        return GroundedFormula(
            sentence_index=idx,
            nl_sentence=sentence,
            formula_code=formula_code,
            attempts=attempts,
        )

    return GroundedFormula(
        sentence_index=idx,
        nl_sentence=sentence,
        formula_code="",
        failed=True,
        attempts=attempts,
        error=f"Grounding failed after {max_retries + 1} attempts. Last error: {last_error}",
    )


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
) -> GroundedFormula:
    """
    Re-ground a sentence when Phase 4 detected a structural Phase 3 semantic error.

    Unlike _formalize_one (which starts blind), this function pre-seeds the LLM
    conversation with the wrong formula and a semantic error explaining what the
    formula should evaluate to.  This gives the LLM direct signal about the
    comparison direction or quantifier structure error without consuming an extra
    "blind" generation.

    Args:
        current_formula: The Phase 3 formula that Phase 4 found to be incorrect.
        expected_truth:  What the formula should evaluate to on this witness
                         (False for ¬q witness, True for q witness).
    """
    semantic_hint = (
        f"Your formula evaluated to {not expected_truth} but must evaluate to "
        f"{expected_truth} on this domain. "
        f"The most likely cause is an incorrect comparison direction (e.g. < vs >) "
        f"or an inverted quantifier. "
        f"Rewrite the formula so it evaluates to {expected_truth}."
    )

    # Seed the conversation: initial prompt → current (wrong) formula → semantic error.
    # This is structurally identical to a Phase 3 retry but with a Phase 4-sourced hint.
    base_messages = _build_messages(sentence, domain_desc_str, prompt_engine, world_assumption)
    messages = base_messages + [
        {"role": "assistant", "content": current_formula},
        _build_retry_message(
            sentence=sentence,
            domain_desc_str=domain_desc_str,
            previous_output=current_formula,
            last_error=semantic_hint,
            attempt_num=1,
            max_attempts=max_retries + 1,
            prompt_engine=prompt_engine,
            world_assumption=world_assumption,
        ),
    ]

    last_error: str | None = semantic_hint
    raw_output: str = ""
    attempts: list[GroundingAttempt] = []

    for attempt in range(max_retries + 1):
        if attempt > 0:
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                _build_retry_message(
                    sentence=sentence,
                    domain_desc_str=domain_desc_str,
                    previous_output=_normalise_output_for_prompt(raw_output),
                    last_error=last_error or "Unknown error",
                    attempt_num=attempt + 1,
                    max_attempts=max_retries + 1,
                    prompt_engine=prompt_engine,
                    world_assumption=world_assumption,
                ),
            ]

        attempt_record = GroundingAttempt(
            attempt_num=attempt + 1,
            messages=_snapshot_messages(messages),
        )
        raw_output = await llm.complete_with_retry(messages)
        attempt_record.raw_output = raw_output
        formula_code = _extract_formula(raw_output)
        attempt_record.extracted_formula = formula_code

        err = _validate_formula(formula_code, domain)
        if err:
            last_error = err
            attempt_record.validation_error = err
            attempts.append(attempt_record)
            logger.debug(
                "reground_with_hint idx=%d attempt %d: validation error: %s",
                idx, attempt + 1, err,
            )
            continue

        attempt_record.accepted = True
        attempts.append(attempt_record)
        return GroundedFormula(
            sentence_index=idx,
            nl_sentence=sentence,
            formula_code=formula_code,
            attempts=attempts,
        )

    return GroundedFormula(
        sentence_index=idx,
        nl_sentence=sentence,
        formula_code="",
        failed=True,
        attempts=attempts,
        error=f"Phase 3 re-grounding failed after {max_retries + 1} attempts. Last error: {last_error}",
    )


def _build_messages(sentence: str, domain_desc_str: str, prompt_engine: PromptEngine, world_assumption: str = "owa") -> list[dict]:
    user_content = prompt_engine.render(
        "phase3_grounded.j2",
        sentence=sentence,
        domain_desc=domain_desc_str,
        world_assumption=world_assumption,
    )
    return [{"role": "user", "content": user_content}]


def _build_retry_message(
    sentence: str,
    domain_desc_str: str,
    previous_output: str,
    last_error: str,
    attempt_num: int,
    max_attempts: int,
    prompt_engine: PromptEngine,
    world_assumption: str = "owa",
) -> dict[str, str]:
    user_content = prompt_engine.render(
        "phase3_retry.j2",
        sentence=sentence,
        domain_desc=domain_desc_str,
        previous_output=previous_output,
        last_error=last_error,
        attempt_num=attempt_num,
        max_attempts=max_attempts,
        world_assumption=world_assumption,
    )
    return {"role": "user", "content": user_content}


def _snapshot_messages(messages: list[dict]) -> list[dict[str, str]]:
    """Copy chat messages so retry traces keep the exact prompt history."""
    return [
        {
            "role": str(m.get("role", "")),
            "content": str(m.get("content", "")),
        }
        for m in messages
    ]


def _extract_formula(raw: str) -> str:
    """Strip markdown fences and single backticks, return the formula expression."""
    raw = re.sub(r'^```(?:python)?\s*\n', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\n```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    # Strip surrounding single backticks (inline code: `expression`)
    if raw.startswith('`') and raw.endswith('`') and len(raw) > 1:
        raw = raw[1:-1].strip()
    return raw


def _normalise_output_for_prompt(raw: str) -> str:
    formula = _extract_formula(raw)
    if formula:
        return formula
    return raw.strip()


def _build_valid_keys(domain: dict) -> set[str]:
    """Build the set of valid truth[] keys from the domain predicates dict."""
    valid: set[str] = set()
    for pred_name, interp in domain.get("predicates", {}).items():
        for args in interp.keys():
            key = f"{pred_name}({', '.join(args)})"
            valid.add(key)
    return valid


def _build_valid_value_keys(domain: dict) -> set[str]:
    """Build the set of valid value[] keys from the domain function_values dict."""
    return set(domain.get("function_values", {}).keys())


def _validate_formula(formula_code: str, domain: dict) -> str | None:
    """
    Validate a grounded formula (truth-table or value-comparison expression).

    Checks (in order):
    1. Not empty.
    2. AST structural check: must parse as a pure expression (no imports, assignments,
       no direct predicate calls other than all/any).
    3. Static truth["..."] key validation against known domain atoms.
    4. Static value["..."] key validation against known function values.
    5. f-string entity iterators only use entities present in the domain.
    6. Sort-aware check: entity loops use entities from the correct sort.
    7. At least one truth["..."] or value["..."] reference exists (unless True/False).
    """
    if not formula_code:
        return "Empty formula"

    # --- Check 2: AST structural parse ---
    ast_err = _check_ast_structure(formula_code)
    if ast_err:
        return ast_err

    # --- Check 3: Static truth["..."] key validation ---
    used_keys = re.findall(r'truth\["([^"]+)"\]', formula_code)
    used_keys += re.findall(r"truth\['([^']+)'\]", formula_code)
    has_truth_fstring = bool(re.search(r'truth\[f["\']', formula_code))

    if used_keys:
        valid_keys = _build_valid_keys(domain)
        unknown = [k for k in used_keys if k not in valid_keys]
        if unknown:
            valid_sample = sorted(valid_keys)[:5]
            return (
                f"Unknown truth table keys: {unknown[:3]}. "
                f"Valid keys include: {valid_sample}"
            )

    # --- Check 4: Static value["..."] key validation ---
    used_value_keys = re.findall(r'value\["([^"]+)"\]', formula_code)
    used_value_keys += re.findall(r"value\['([^']+)'\]", formula_code)
    has_value_fstring = bool(re.search(r'value\[f["\']', formula_code))

    if used_value_keys:
        valid_value_keys = _build_valid_value_keys(domain)
        unknown_v = [k for k in used_value_keys if k not in valid_value_keys]
        if unknown_v:
            valid_v_sample = sorted(valid_value_keys)[:5]
            return (
                f"Unknown value[] keys: {unknown_v[:3]}. "
                f"Valid keys include: {valid_v_sample}"
            )

    # --- Check 7: At least one reference required ---
    has_any_ref = used_keys or has_truth_fstring or used_value_keys or has_value_fstring
    if not has_any_ref and formula_code not in ("True", "False"):
        return (
            'Formula must use truth["PredName(entity)"] or value["fname(entity)"] '
            f'references. Got: {formula_code[:80]!r}'
        )

    # --- Check 5+6: f-string entity and sort validation ---
    if has_truth_fstring or has_value_fstring:
        fstring_err = _check_fstring_entities(formula_code, domain)
        if fstring_err:
            return fstring_err

    return None


def _check_ast_structure(formula_code: str) -> str | None:
    """
    Verify the formula is a pure boolean expression with no unsafe constructs.

    Allowed AST node types cover boolean ops, comparisons, subscripts (truth[...]),
    all/any calls, generator expressions, f-strings, name/constant references.
    Disallows: Import, FunctionDef, ClassDef, Assign, Delete, Exec, etc.
    """
    try:
        tree = ast.parse(formula_code, mode='eval')
    except SyntaxError as e:
        return f"Syntax error in grounded formula: {e}"

    # Walk the expression tree looking for disallowed constructs
    _ALLOWED_EXPR_TYPES = (
        ast.BoolOp, ast.UnaryOp, ast.Compare, ast.BinOp,
        ast.Call, ast.Subscript, ast.Index,
        ast.GeneratorExp, ast.ListComp, ast.comprehension,
        ast.JoinedStr, ast.FormattedValue,   # f-strings
        ast.List, ast.Tuple,
        ast.Name, ast.Attribute,
        ast.Constant,
        ast.Expression,                       # root node for mode='eval'
        # Ops
        ast.And, ast.Or, ast.Not,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.In, ast.NotIn,
        ast.Load, ast.Store,
    )
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_EXPR_TYPES):
            return (
                f"Formula contains disallowed construct: {type(node).__name__}. "
                "Only boolean expressions using truth[...] / all() / any() are allowed."
            )

    # Disallow every function call except all() and any().
    # truth["..."] / truth[f"..."] are subscripts, not calls — unaffected.
    # The check rejects all non-whitelisted forms: predicate calls like
    # moved_to(u, x), method calls like obj.method(), and call-expression
    # forms like (lambda: ...)() — anything whose callee isn't Name('all')
    # or Name('any') is refused.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_allowed = isinstance(func, ast.Name) and func.id in {"all", "any"}
            if not is_allowed:
                call_repr = func.id if isinstance(func, ast.Name) else type(func).__name__
                return (
                    f"Formula contains a disallowed call ({call_repr!r}). "
                    "Only 'all(...)' and 'any(...)' are permitted as function calls. "
                    "Access predicates via truth[\"PredName(entity)\"] subscripts; "
                    "access function values via value[\"fname(entity)\"] subscripts."
                )

    return None


def _check_fstring_entities(formula_code: str, domain: dict) -> str | None:
    """
    AST-level validation for f-string grounded formulas.

    Handles both truth[f"pred(...)"] and value[f"fname(...)"] subscripts.

    Checks:
    1. Every entity string literal inside a `for var in [...]` generator loop is a
       known domain entity.
    2. For truth[f"pred({var})"]: {var} iterates over entities of the correct sort,
       matched by argument position via predicate_signatures.
    3. For value[f"fname({var})"]: {var} iterates over entities of the correct sort,
       matched by argument position via raw_function_signatures.
    """
    all_entities: set[str] = set(domain.get("entities", []))
    for sort_ents in domain.get("sorts", {}).values():
        all_entities.update(sort_ents)
    entity_to_sort: dict[str, str] = {
        ent: srt
        for srt, ents in domain.get("sorts", {}).items()
        for ent in ents
    }
    pred_sigs: dict[str, list[str]] = domain.get("predicate_signatures", {})
    raw_func_sigs: dict[str, list[str]] = domain.get("raw_function_signatures", {})

    try:
        tree = ast.parse(formula_code, mode='eval')
    except SyntaxError:
        return None  # already caught by _check_ast_structure

    # --- Pass 1: collect for-loop variable → [entities] from comprehension nodes ---
    var_entities: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.comprehension):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        var = node.target.id
        if not isinstance(node.iter, ast.List):
            continue
        ents = [
            elt.value
            for elt in node.iter.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
        if ents:
            var_entities[var] = ents

    # Check 1: all loop entities are known domain entities
    for var, ents in var_entities.items():
        for ent in ents:
            if ent not in all_entities:
                return (
                    f"Entity {ent!r} in loop for variable '{var}' is not a "
                    f"known domain entity. Valid entities: {sorted(all_entities)[:8]}"
                )

    # --- Pass 2: sort-aware check via AST Subscript → JoinedStr nodes ---
    # Handles both truth[f"..."] (uses pred_sigs) and value[f"..."] (uses raw_func_sigs).
    if not entity_to_sort:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        lookup_name = node.value.id   # 'truth' or 'value'
        if lookup_name not in ("truth", "value"):
            continue
        joined = node.slice
        if not isinstance(joined, ast.JoinedStr):
            continue

        parsed = _parse_fstring_pred_args(joined)
        if parsed is None:
            continue
        func_name, arg_vars = parsed

        # Select signature source based on lookup type
        if lookup_name == "truth":
            sigs = pred_sigs
        else:  # "value"
            sigs = raw_func_sigs

        if func_name not in sigs:
            continue
        expected_sorts = sigs[func_name]
        if len(arg_vars) != len(expected_sorts):
            continue  # arity mismatch — skip

        for var, expected_sort in zip(arg_vars, expected_sorts):
            if var is None:
                continue  # complex expression — skip
            loop_ents = var_entities.get(var)
            if not loop_ents:
                continue
            expected_ents = set(domain.get("sorts", {}).get(expected_sort, []))
            if not expected_ents:
                continue
            for ent in loop_ents:
                ent_sort = entity_to_sort.get(ent)
                if ent_sort and ent_sort != expected_sort:
                    return (
                        f"Entity {ent!r} (sort: {ent_sort}) in loop for variable "
                        f"'{var}' of {lookup_name}[f\"{func_name}(...)\"] but expected sort "
                        f"'{expected_sort}'. Use entities: {sorted(expected_ents)}"
                    )

    return None


def _parse_fstring_pred_args(
    joined: ast.JoinedStr,
) -> tuple[str, list[str | None]] | None:
    """
    Extract (pred_name, [arg_var_or_none, ...]) from a JoinedStr node.

    Handles all argument shapes the LLM prompt encourages:
      f"pred({v})"              → ("pred", ["v"])
      f"pred({v1}, {v2})"      → ("pred", ["v1", "v2"])
      f"pred(alice, {v})"      → ("pred", [None, "v"])   # None = constant, no loop
      f"pred({v}, alice)"      → ("pred", ["v", None])
      f"pred(alice, bob)"      → ("pred", [None, None])  # all constants
      f"pred({complex_expr})"  → ("pred", [None])        # non-Name expr → None

    Returns None if the f-string doesn't match a simple predicate call at all.
    """
    values = joined.values
    if not values:
        return None

    # First chunk must be a string Constant containing "predname("
    if not isinstance(values[0], ast.Constant) or not isinstance(values[0].value, str):
        return None
    first_str: str = values[0].value
    paren_idx = first_str.find('(')
    if paren_idx < 0:
        return None
    pred_name = first_str[:paren_idx]
    if not pred_name.isidentifier():
        return None

    # Build a flat token list representing the argument zone:
    #   ('s', string_value)  — literal text (may contain constants and commas)
    #   ('v', var_or_None)   — a {var} or {complex_expr} placeholder
    # Start with whatever follows "predname(" in the first chunk.
    tokens: list[tuple[str, str | None]] = []
    after_paren = first_str[paren_idx + 1:]
    if after_paren:
        tokens.append(('s', after_paren))

    for val in values[1:]:
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            tokens.append(('s', val.value))
        elif isinstance(val, ast.FormattedValue):
            tokens.append(('v', val.value.id if isinstance(val.value, ast.Name) else None))

    # Strip the trailing ")" from the last string token (the predicate call close)
    if not tokens:
        return None
    if tokens[-1][0] == 's':
        tail = tokens[-1][1].rstrip()
        if not tail.endswith(')'):
            return None   # unexpected structure
        tail = tail[:-1].rstrip()
        if tail:
            tokens[-1] = ('s', tail)
        else:
            tokens.pop()
    else:
        return None  # last token is a variable — no closing ')' string found

    # Parse comma-separated arguments from the token list.
    # Each argument is one or more consecutive tokens; commas only appear inside
    # string ('s') tokens, never inside variable ('v') tokens.
    arg_token_lists: list[list[tuple[str, str | None]]] = [[]]
    for tok_type, tok_val in tokens:
        if tok_type == 's':
            parts = tok_val.split(',')
            # First part continues the current argument
            stripped = parts[0].strip()
            if stripped:
                arg_token_lists[-1].append(('s', stripped))
            # Each subsequent part belongs to a new argument
            for part in parts[1:]:
                arg_token_lists.append([])
                stripped = part.strip()
                if stripped:
                    arg_token_lists[-1].append(('s', stripped))
        else:
            arg_token_lists[-1].append((tok_type, tok_val))

    # Reduce each argument's token list to a single value:
    #   single ('v', name) token → name string (loop variable to validate)
    #   single ('v', None) token → None (complex expr — skip)
    #   single ('s', ...) token → None (constant entity — no loop, already validated elsewhere)
    #   empty or multi-token     → None (complex / ambiguous — skip)
    arg_vars: list[str | None] = []
    for toks in arg_token_lists:
        if len(toks) == 1 and toks[0][0] == 'v':
            arg_vars.append(toks[0][1])   # variable name or None for complex expr
        else:
            arg_vars.append(None)          # constant or ambiguous — skip loop check

    return pred_name, arg_vars
