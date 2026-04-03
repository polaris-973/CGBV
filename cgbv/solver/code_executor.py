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
# Background-thread accounting
# ---------------------------------------------------------------------------
# We allow at most this many daemon execution threads to be alive simultaneously.
# If the cap is reached (all prior threads are hung on infinite loops), new
# execute_z3_code calls fail immediately rather than accumulating more workers.
_MAX_LIVE_WORKERS = 8
_live_workers: set[threading.Thread] = set()
_live_workers_lock = threading.Lock()


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
    Extract names of variables declared as ForAll/Exists quantifier variables.

    These are Const declarations that appear as the first argument list of
    ForAll([x, y, ...], ...) or Exists([x, y, ...], ...) and should NOT
    be treated as named entity constants in model extraction.
    """
    bound_names: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bound_names
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
