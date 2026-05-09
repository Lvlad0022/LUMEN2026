"""Recommender backed by a wide CSV of precomputed item scores.

Expected CSV shape:
- one row per customer
- ``CustomerID`` column
- one column per item, with numeric recommendation scores
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


class WideCsvScoreRecommender:
    """Serve recommendations from a precomputed wide score CSV."""

    def __init__(self, csv_path: str | Path, customer_col: str = "CustomerID") -> None:
        self.csv_path = str(csv_path)
        self.customer_col = customer_col
        self.recommendations_by_customer: dict[str, list[dict[str, Any]]] = {}
        self._load_csv(Path(csv_path))

    def recommend(self, customer_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Return top-k scored items for one customer."""

        customer_key = self._customer_key(customer_id)
        if customer_key not in self.recommendations_by_customer:
            raise KeyError(customer_id)
        return self.recommendations_by_customer[customer_key][: int(top_k)]

    def _load_csv(self, csv_path: Path) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Score CSV does not exist: {csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Score CSV has no header: {csv_path}")
            if self.customer_col not in reader.fieldnames:
                raise ValueError(f"Score CSV is missing customer column: {self.customer_col}")

            item_columns = [column for column in reader.fieldnames if column != self.customer_col]
            for row in reader:
                raw_customer_id = row.get(self.customer_col)
                if raw_customer_id is None or raw_customer_id == "":
                    continue

                scored_items: list[dict[str, Any]] = []
                for item_id in item_columns:
                    score = self._parse_score(row.get(item_id))
                    if score is None or score <= 0.0:
                        continue
                    scored_items.append({"item_id": item_id, "score": score})

                scored_items.sort(key=lambda item: item["score"], reverse=True)
                self.recommendations_by_customer[self._customer_key(raw_customer_id)] = scored_items

    @staticmethod
    def _parse_score(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            score = float(value)
        except ValueError:
            return None
        if not math.isfinite(score):
            return None
        return score

    @staticmethod
    def _customer_key(customer_id: int | str) -> str:
        text = str(customer_id).strip()
        try:
            return str(int(float(text)))
        except ValueError:
            return text


__all__ = ["WideCsvScoreRecommender"]
