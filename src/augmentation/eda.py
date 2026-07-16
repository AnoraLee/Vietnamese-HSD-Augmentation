"""EDA Augmentation - Easy Data Augmentation"""
from __future__ import annotations
from .synonyms import build_synonym_dict

import random
from typing import Any

from underthesea import word_tokenize

from .base_augmenter import BaseAugmenter, AugmentedSample


class EDAAugmenter(BaseAugmenter):
    method_name = "eda"

    def __init__(
        self,
        alpha: float = 0.1,
        num_aug: int = 2,
        operations: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.num_aug = num_aug
        self.operations = operations or ["SR", "RI", "RS", "RD"]
        self.synonyms = build_synonym_dict()

    def _segment(self, text: str) -> list[str]:
        return word_tokenize(text, format='text').split()

    def _rejoin(self, tokens: list[str]) -> str:
        return ' '.join(t.replace('_', ' ') for t in tokens)

    def _sr(self, tokens: list[str], n: int) -> list[str]:
        new = tokens.copy()
        candidates = [t for t in tokens if t in self.synonyms]
        random.shuffle(candidates)
        for word in candidates[:n]:
            if word in new:
                idx = new.index(word)
                new[idx] = random.choice(self.synonyms[word])
        return new

    def _ri(self, tokens: list[str], n: int) -> list[str]:
        new = tokens.copy()
        candidates = [t for t in tokens if t in self.synonyms]
        for _ in range(n):
            if not candidates:
                break
            word = random.choice(candidates)
            syn = random.choice(self.synonyms[word])
            new.insert(random.randint(0, len(new)), syn)
        return new

    def _rs(self, tokens: list[str], n: int) -> list[str]:
        new = tokens.copy()
        for _ in range(n):
            if len(new) >= 2:
                i, j = random.sample(range(len(new)), 2)
                new[i], new[j] = new[j], new[i]
        return new

    def _rd(self, tokens: list[str], p: float) -> list[str]:
        new = [t for t in tokens if random.random() > p]
        return new if new else [random.choice(tokens)]

    def augment_one(self, text: str, label: str) -> list[AugmentedSample]:
        tokens = self._segment(text)
        if len(tokens) < 3:
            return []

        n = max(1, int(self.alpha * len(tokens)))
        results = []
        used_texts = {text}

        for _ in range(self.num_aug * 2):
            op = random.choice(self.operations)

            if op == 'SR':
                aug = self._sr(tokens, n)
            elif op == 'RI':
                aug = self._ri(tokens, n)
            elif op == 'RS':
                aug = self._rs(tokens, n)
            elif op == 'RD':
                aug = self._rd(tokens, self.alpha)
            else:
                continue

            out = self._rejoin(aug)

            if out.strip() != text.strip() and out not in used_texts:
                results.append(AugmentedSample(
                    original_text=text,
                    text=out,
                    label=label,
                    method=self.method_name,
                    metadata=f"op={op}",
                ))
                used_texts.add(out)

                if len(results) >= self.num_aug:
                    break

        return results