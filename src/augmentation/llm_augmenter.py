from __future__ import annotations

import json
from pathlib import Path

from .base_augmenter import AugmentedSample, BaseAugmenter


class LLMAugmenter(BaseAugmenter):
    method_name = "llm"

    def __init__(
        self,
        prompt_template: str,
        prompt_log_file: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prompt_template = prompt_template
        self.prompt_log_file = Path(prompt_log_file) if prompt_log_file else None

    def augment_one(self, text: str, label: str) -> list[AugmentedSample]:
        prompt = self.build_prompt(text, label)
        if self.prompt_log_file is not None:
            self.prompt_log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.prompt_log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"text": text, "label": label, "prompt": prompt}, ensure_ascii=False) + "\n")
        return []

    def build_prompt(self, text: str, label: str) -> str:
        return self.prompt_template.format(text=text, label=label)

    @staticmethod
    def from_generated_pairs(rows: list[dict[str, str]]) -> list[AugmentedSample]:
        samples: list[AugmentedSample] = []
        for row in rows:
            samples.append(
                AugmentedSample(
                    original_text=row["original_text"],
                    text=row["generated_text"],
                    label=row["label"],
                    method=LLMAugmenter.method_name,
                    metadata=row.get("metadata", "external_generation"),
                )
            )
        return samples
