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
        self._few_shot_cache: dict[str, list[dict]] = {}

    def _load_few_shot(self, dataset: str) -> list[dict]:
        if dataset not in self._few_shot_cache:
            path = self.few_shot_dir / f"{dataset}.yaml"
            if path.exists():
                with open(path) as f:
                    self._few_shot_cache[dataset] = yaml.safe_load(f) or []
            else:
                self._few_shot_cache[dataset] = []
        return self._few_shot_cache[dataset]

    def render(self, template_name: str, dataset: str = "", **kwargs) -> str:
        """Render a Jinja2 template.

        Args:
            template_name: filename relative to templates_dir (e.g. "phase1_formalize.j2")
            dataset: dataset name for few-shot lookup (empty = no few-shot)
            **kwargs: template variables
        """
        few_shot = self._load_few_shot(dataset) if dataset else []
        template = self._env.get_template(template_name)
        return template.render(few_shot_examples=few_shot, **kwargs)
