from __future__ import annotations

from pathlib import Path

from cgbv.prompts.prompt_engine import PromptEngine


def test_prompt_engine_supports_structured_few_shot_bundle() -> None:
    fixtures_root = Path("tests")
    engine = PromptEngine(
        str(fixtures_root),
        str(fixtures_root),
    )

    rendered = engine.render("prompt_engine_structured_t.j2", dataset="prompt_engine_structured_demo")
    assert rendered == "example-one|example-two|"
    assert engine.get_excluded_ids("prompt_engine_structured_demo", "validation") == ["a1", "a2"]
    assert engine.get_excluded_ids("prompt_engine_structured_demo", "test") == []


def test_prompt_engine_backwards_compatible_with_list_few_shot() -> None:
    fixtures_root = Path("tests")
    engine = PromptEngine(
        str(fixtures_root),
        str(fixtures_root),
    )

    assert engine.render("prompt_engine_list_t.j2", dataset="prompt_engine_list_demo") == "2"
    assert engine.get_excluded_ids("prompt_engine_list_demo", "validation") == []
