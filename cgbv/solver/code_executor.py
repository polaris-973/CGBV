from __future__ import annotations

import ast
import logging
import queue
import re
import textwrap
import threading
from typing import Any

logger = logging.getLogger(__name__)


class CodeExecutionError(Exception):
    pass


# ---------------------------------------------------------------------------
# NameError auto-correction helpers
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], (
                prev if a[i - 1] == b[j - 1]
                else 1 + min(prev, dp[j], dp[j - 1])
            )
    return dp[n]


def _extract_declared_names(code: str) -> list[str]:
    """
    Return all top-level assignment target names from *code*.

    These are the variable names the LLM declared (sorts, constants,
    predicates, bound variables).  Used to find the closest match when a
    NameError occurs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.iter_child_nodes(tree):   # top-level statements only
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
    return names


def build_name_error_hint(code: str, error_msg: str) -> str | None:
    """
    When a NameError occurs, try to identify the exact spelling error and
    return a human-readable diagnosis string suitable for injection into the
    retry prompt.

    Returns None if the error is not a NameError or no close match is found.
    """
    m = re.search(r"name '(\w+)' is not defined", str(error_msg))
    if not m:
        return None
    undefined = m.group(1)
    declared = _extract_declared_names(code)
    if not declared:
        return None

    close = [(name, _edit_distance(undefined, name)) for name in declared]
    close_enough = [(name, dist) for name, dist in close if 0 < dist <= 2]
    if not close_enough:
        return None

    best, dist = min(close_enough, key=lambda x: x[1])

    # Locate declaration and usage line numbers for context
    lines = code.split("\n")
    decl_linenos = [
        i + 1 for i, line in enumerate(lines)
        if re.search(r"\b" + re.escape(best) + r"\b", line)
    ]
    use_linenos = [
        i + 1 for i, line in enumerate(lines)
        if re.search(r"\b" + re.escape(undefined) + r"\b", line)
    ]

    char_desc = "1 character" if dist == 1 else f"{dist} characters"
    parts = [
        f"Spelling error: you used `{undefined}` but the declared name is `{best}` (differ by {char_desc}).",
    ]
    if decl_linenos:
        parts.append(f"`{best}` is declared on line {decl_linenos[0]}.")
    if use_linenos:
        parts.append(f"`{undefined}` is used on line(s) {', '.join(str(l) for l in use_linenos)}.")
    parts.append(f"Fix: replace every occurrence of `{undefined}` with `{best}`.")
    return " ".join(parts)


def build_runtime_error_hint(code: str, error_msg: str) -> str | None:
    """Diagnose common Z3 runtime errors beyond NameError.

    Returns a human-readable hint or None if no pattern matches.
    """
    err = str(error_msg)

    # IndexError / sequence errors
    if "index out of bounds" in err or "IndexError" in err:
        return (
            "A function or sequence was indexed beyond its declared arity. "
            "Check that each Function(...) declaration has the right number "
            "of argument sorts and that all call sites pass the correct "
            "number of arguments."
        )

    # Sort mismatch
    if "sort mismatch" in err.lower():
        return (
            "Z3 sort mismatch: a predicate or function received an argument "
            "of the wrong sort. Verify that ForAll/Exists bound variables "
            "use the correct DeclareSort type and that constants match "
            "their declared sorts."
        )

    # Arity mismatch
    m = re.search(r'takes (\d+) positional argument', err)
    if m:
        return (
            f"Function arity error: a function was called with the wrong "
            f"number of arguments (expected {m.group(1)}). Check Function(...) "
            f"declarations and all call sites."
        )

    # Bool vs non-Bool confusion
    if "BoolRef" in err and ("ArithRef" in err or "SeqRef" in err):
        return (
            "Type confusion between Bool and non-Bool expressions. "
            "Ensure boolean predicates return BoolSort() and are not "
            "used in arithmetic or string contexts."
        )

    return None


def attempt_name_error_autocorrect(code: str, error_msg: str) -> tuple[str, str] | None:
    """
    Attempt to auto-correct a NameError caused by a trivial spelling typo.

    Strategy: if exactly ONE declared name is within Levenshtein distance ≤ 2
    of the undefined name, replace all word-boundary occurrences of the
    undefined name with the declared name and return the corrected code.

    Returns ``(corrected_code, description)`` on success, ``None`` otherwise.
    The caller should log the description and re-execute the corrected code.

    Safety invariant: only applies when there is exactly one close match, so
    the correction is unambiguous.  Multiple close matches → return None and
    let the LLM retry with the hint message instead.
    """
    m = re.search(r"name '(\w+)' is not defined", str(error_msg))
    if not m:
        return None
    undefined = m.group(1)
    declared = _extract_declared_names(code)
    if not declared:
        return None

    close_enough = [
        (name, _edit_distance(undefined, name))
        for name in declared
        if 0 < _edit_distance(undefined, name) <= 2
    ]
    if len(close_enough) != 1:
        return None   # ambiguous or no match — do not auto-correct

    best, dist = close_enough[0]
    corrected = re.sub(r"\b" + re.escape(undefined) + r"\b", best, code)
    description = (
        f"NameError auto-correction: `{undefined}` → `{best}` "
        f"(edit distance {dist})"
    )
    return corrected, description


# ---------------------------------------------------------------------------
# Background-thread accounting
# ---------------------------------------------------------------------------
# We allow at most this many daemon execution threads to be alive simultaneously.
# If the cap is reached (all prior threads are hung on infinite loops), new
# execute_z3_code calls fail immediately rather than accumulating more workers.
_MAX_LIVE_WORKERS = 8
_live_workers: set[threading.Thread] = set()
_live_workers_lock = threading.Lock()


def configure_max_workers(n: int) -> None:
    """Set the maximum number of concurrent code-execution threads."""
    global _MAX_LIVE_WORKERS
    _MAX_LIVE_WORKERS = n


def _register_worker(t: threading.Thread) -> None:
    """Prune finished threads, then register a new one (or raise if cap reached)."""
    with _live_workers_lock:
        # Remove threads that have already finished
        finished = {w for w in _live_workers if not w.is_alive()}
        _live_workers.difference_update(finished)
        if len(_live_workers) >= _MAX_LIVE_WORKERS:
            raise CodeExecutionError(
                f"Too many hung code-execution threads ({len(_live_workers)} / "
                f"{_MAX_LIVE_WORKERS}). This usually means the LLM keeps generating "
                "infinite-loop code. Aborting to prevent resource exhaustion."
            )
        _live_workers.add(t)


def _unregister_worker(t: threading.Thread) -> None:
    with _live_workers_lock:
        _live_workers.discard(t)


def _extract_bound_var_names(code: str) -> set[str]:
    """
    Extract names of variables that should be excluded from the Unique Name
    Assumption (UNA) because they are quantifier-bound variables, not named
    domain entities.

    Two sources:
    1. Variables that explicitly appear in ForAll([x, y, ...], ...) or
       Exists([x, y, ...], ...) lists.
    2. Single-lowercase-letter Const declarations (e.g. ``c = Const('c', Sort)``).
       By universal convention in FOL code, single-letter names are quantifier
       variables.  An LLM may declare such a constant intending it as a bound
       variable but accidentally omit it from a ForAll/Exists list; treating it
       as a named entity and applying UNA (Distinct) would then force it to be
       distinct from every other named constant, causing vacuous entailment.
    """
    bound_names: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bound_names

    # Pass 1: explicit ForAll/Exists quantifier variable lists
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name: str | None = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name in ("ForAll", "Exists") and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.List):
                for elt in first_arg.elts:
                    if isinstance(elt, ast.Name):
                        bound_names.add(elt.id)

    # Pass 2: single-letter Const declarations  (e.g. `c = Const('c', Channel)`)
    # Named domain entities always have multi-character names; single letters
    # are always intended as quantifier variables.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        py_name = node.targets[0].id
        if len(py_name) != 1 or not py_name.islower():
            continue
        val = node.value
        if isinstance(val, ast.Call):
            fn = val.func
            fn_name = (
                fn.id if isinstance(fn, ast.Name)
                else fn.attr if isinstance(fn, ast.Attribute)
                else None
            )
            if fn_name == "Const":
                bound_names.add(py_name)

    return bound_names


def _strip_fences(code: str) -> str:
    """Remove markdown code fences if LLM wraps output in them."""
    code = re.sub(r'^```(?:python)?\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
    return code.strip()


def _exec_in_daemon_thread(code: str, result_queue: "queue.Queue[tuple]") -> None:
    """
    Run exec() and put (ok, namespace) or (error, exception) into result_queue.

    Designed to run inside a daemon thread so that if the code enters an
    infinite loop it will not block the calling thread beyond the timeout, and
    will be cleaned up automatically when the Python process exits.
    """
    try:
        namespace: dict[str, Any] = {}
        compiled = compile(code, "<llm_generated>", "exec")
        exec(compiled, namespace)  # noqa: S102
        result_queue.put(("ok", namespace))
    except Exception as exc:
        result_queue.put(("error", exc))


def execute_z3_code(code: str, timeout_seconds: int = 30) -> dict[str, Any]:
    """
    Execute LLM-generated Z3-Python code and extract the formalization objects.

    The code is expected to define:
        premises: list[z3.ExprRef]   - list of FOL formulas
        q: z3.ExprRef                - conclusion formula

    Runs in a daemon thread with a wall-clock timeout.  Because the thread is
    a daemon, it will be reaped automatically when the Python process exits,
    so an infinite-loop code cannot prevent process shutdown.  Within a run,
    a timed-out thread continues executing in the background but the calling
    code is unblocked immediately — this is the best hard-isolation available
    inside CPython without crossing a process boundary (which would break Z3
    object return).

    Returns a dict with keys: premises, q, namespace, raw_code, bound_var_names
    Raises CodeExecutionError on syntax/runtime/timeout errors.
    """
    code = _strip_fences(code)
    code = textwrap.dedent(code)

    try:
        compile(code, "<llm_generated>", "exec")   # syntax check before spawning thread
    except SyntaxError as e:
        raise CodeExecutionError(f"Syntax error in generated code: {e}") from e

    result_queue: queue.Queue[tuple] = queue.Queue()
    worker = threading.Thread(
        target=_exec_in_daemon_thread,
        args=(code, result_queue),
        daemon=True,    # dies with the process; won't block interpreter shutdown
    )
    _register_worker(worker)   # raises CodeExecutionError if cap exceeded
    worker.start()
    worker.join(timeout=timeout_seconds)

    if worker.is_alive():
        # Thread is still running (infinite loop or blocked Z3 call).
        # It stays registered in _live_workers so the cap accounts for it.
        # It will be pruned on the next call once it eventually finishes.
        live_count = sum(1 for w in _live_workers if w.is_alive())
        logger.warning(
            "Code execution timed out after %ds (%d/%d background threads live)",
            timeout_seconds, live_count, _MAX_LIVE_WORKERS,
        )
        raise CodeExecutionError(
            f"Code execution timed out after {timeout_seconds}s "
            "(possible infinite loop or blocking Z3 call)"
        )

    _unregister_worker(worker)
    tag, val = result_queue.get_nowait()
    if tag == "error":
        exc = val
        if isinstance(exc, SyntaxError):
            raise CodeExecutionError(f"Syntax error in generated code: {exc}") from exc

        error_msg = str(exc)
        # Attempt to auto-correct trivial NameError typos (e.g. 'brandford_college'
        # used where 'branford_college' was declared — edit distance 1).
        # Only fires when exactly one declared name is within edit distance ≤ 2,
        # so the correction is unambiguous.  This avoids burning an LLM call for
        # a pure spelling inconsistency that the model keeps reproducing.
        if isinstance(exc, NameError) or "is not defined" in error_msg:
            correction = attempt_name_error_autocorrect(code, error_msg)
            if correction is not None:
                corrected_code, description = correction
                logger.info("execute_z3_code: %s — retrying", description)
                # Re-execute the corrected code (one extra attempt, no LLM call)
                corrected_result_queue: queue.Queue[tuple] = queue.Queue()
                corrected_worker = threading.Thread(
                    target=_exec_in_daemon_thread,
                    args=(corrected_code, corrected_result_queue),
                    daemon=True,
                )
                try:
                    _register_worker(corrected_worker)
                    corrected_worker.start()
                    corrected_worker.join(timeout=timeout_seconds)
                    if not corrected_worker.is_alive():
                        _unregister_worker(corrected_worker)
                        c_tag, c_val = corrected_result_queue.get_nowait()
                        if c_tag == "ok":
                            # Correction succeeded — substitute the corrected code
                            code = corrected_code
                            val = c_val
                            tag = "ok"
                            logger.debug(
                                "execute_z3_code: auto-correction succeeded (%s)", description
                            )
                        # else: corrected code still errors — fall through to original error
                    else:
                        _unregister_worker(corrected_worker)
                except CodeExecutionError:
                    pass  # worker cap exceeded; fall through to original error

        if tag == "error":
            raise CodeExecutionError(f"Runtime error in generated code: {exc}") from exc
    namespace = val

    if "premises" not in namespace:
        raise CodeExecutionError("Generated code did not define 'premises' list")
    if "q" not in namespace:
        raise CodeExecutionError("Generated code did not define conclusion 'q'")

    bound_var_names = _extract_bound_var_names(code)
    return {
        "premises": namespace["premises"],
        "q": namespace["q"],
        "namespace": namespace,
        "raw_code": code,
        "bound_var_names": bound_var_names,
    }
