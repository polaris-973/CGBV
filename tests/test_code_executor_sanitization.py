from cgbv.core.phase1_formalize import _normalise_code_for_prompt
from cgbv.solver.code_executor import _strip_fences


def test_code_executor_extracts_first_fenced_block_from_mixed_output() -> None:
    raw = """
Here is the formalization.

```python
from z3 import *
premises = []
q = BoolVal(True)
```

✓ Checked for syntax.
"""

    cleaned = _strip_fences(raw)

    assert cleaned == "from z3 import *\npremises = []\nq = BoolVal(True)"


def test_phase1_prompt_normalizer_prefers_fenced_code_block() -> None:
    raw = """
I fixed the bug.

```python
from z3 import *
premises = []
q = BoolVal(False)
```

Notes: no extra imports.
"""

    cleaned = _normalise_code_for_prompt(raw)

    assert cleaned == "from z3 import *\npremises = []\nq = BoolVal(False)"
