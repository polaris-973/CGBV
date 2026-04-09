from __future__ import annotations

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptEngine:
    """Load and render Jinja2 prompt templates with optional few-shot injection."""

    def __init__(self, templates_dir: str, few_shot_dir: str):
        self.templates_dir = Path(templates_dir)
        self.few_shot_dir = Path(few_shot_dir)
        self._env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._few_shot_cache: dict[str, dict[str, object]] = {}

    def _load_few_shot(self, dataset: str) -> dict[str, object]:
        if dataset not in self._few_shot_cache:
            path = self.few_shot_dir / f"{dataset}.yaml"
            if path.exists():
                with open(path) as f:
                    raw = yaml.safe_load(f) or {}
                    self._few_shot_cache[dataset] = self._normalise_few_shot(raw)
            else:
                self._few_shot_cache[dataset] = {
                    "examples": [],
                    "exclude_ids_by_split": {},
                }
        return self._few_shot_cache[dataset]

    def _normalise_few_shot(self, raw: object) -> dict[str, object]:
        if isinstance(raw, list):
            return {
                "examples": raw,
                "exclude_ids_by_split": {},
            }
        if not isinstance(raw, dict):
            return {
                "examples": [],
                "exclude_ids_by_split": {},
            }
        examples = raw.get("examples")
        if not isinstance(examples, list):
            examples = []
        exclude_ids_by_split = raw.get("exclude_ids_by_split")
        if not isinstance(exclude_ids_by_split, dict):
            exclude_ids_by_split = {}
        normalised_excludes: dict[str, list[str]] = {}
        for split, ids in exclude_ids_by_split.items():
            if isinstance(ids, list):
                normalised_excludes[str(split)] = [str(x) for x in ids]
        return {
            "examples": examples,
            "exclude_ids_by_split": normalised_excludes,
        }

    def get_excluded_ids(self, dataset: str, split: str) -> list[str]:
        few_shot = self._load_few_shot(dataset)
        exclude_ids_by_split = few_shot.get("exclude_ids_by_split", {})
        if not isinstance(exclude_ids_by_split, dict):
            return []
        ids = exclude_ids_by_split.get(split, [])
        if not isinstance(ids, list):
            return []
        return [str(x) for x in ids]

    def render(self, template_name: str, dataset: str = "", **kwargs) -> str:
        """Render a Jinja2 template.

        Args:
            template_name: filename relative to templates_dir (e.g. "phase1_formalize.j2")
            dataset: dataset name for few-shot lookup (empty = no few-shot)
            **kwargs: template variables
        """
        few_shot = self._load_few_shot(dataset) if dataset else {
            "examples": [],
            "exclude_ids_by_split": {},
        }
        template = self._env.get_template(template_name)
        return template.render(
            few_shot_examples=few_shot.get("examples", []),
            **kwargs,
        )
