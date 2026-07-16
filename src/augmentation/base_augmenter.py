from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.utils.preprocess import clean_hsd_frame


@dataclass
class AugmentedSample:
    original_text: str
    text: str
    label: str
    method: str
    metadata: str = ""


class BaseAugmenter(ABC):

    method_name = "base"

    def __init__(
        self,
        text_column: str = "text",
        label_column: str = "label",
        random_state: int = 42,
    ) -> None:
        self.text_column = text_column
        self.label_column = label_column
        self.random_state = random_state

    def augment(
        self,
        df: pd.DataFrame,
        max_samples: int | None = None,
        labels: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        source = clean_hsd_frame(df, self.text_column, self.label_column)
        if labels is not None:
            wanted = set(labels)
            source = source[source["label"].isin(wanted)]
        if max_samples is not None:
            source = source.head(max_samples)

        rows: list[dict[str, str]] = []
        for record in source.to_dict(orient="records"):
            for sample in self.augment_one(record["text"], record["label"]):
                if sample.text and sample.text != record["text"]:
                    rows.append(
                        {
                            "text": sample.text,
                            "label": sample.label,
                            "original_text": sample.original_text,
                            "augmentation_method": sample.method,
                            "metadata": sample.metadata,
                        }
                    )
        return pd.DataFrame(rows)

    @abstractmethod
    def augment_one(self, text: str, label: str) -> list[AugmentedSample]:
        """Return augmented variants for one input sample."""
