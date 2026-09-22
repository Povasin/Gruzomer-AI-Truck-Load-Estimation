from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Prediction:
    transport_id: str
    load_pct: float
    cargo_type: str
    models_count: int
    backbone: str
    created_at: str


class MemoryPredictionRepository:
    """MVP repository; replace with Supabase once project credentials are supplied."""

    def __init__(self) -> None:
        self._items: dict[str, Prediction] = {}

    def save(
        self,
        transport_id: str,
        load_pct: float,
        cargo_type: str,
        models_count: int,
        backbone: str,
    ) -> Prediction:
        prediction = Prediction(
            transport_id=transport_id,
            load_pct=float(load_pct),
            cargo_type=cargo_type,
            models_count=int(models_count),
            backbone=backbone,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._items[transport_id] = prediction
        return prediction

    def get(self, transport_id: str) -> Prediction | None:
        return self._items.get(transport_id)


class SupabasePredictionRepository:
    """Persists the latest prediction for each transport in Supabase."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def save(
        self,
        transport_id: str,
        load_pct: float,
        cargo_type: str,
        models_count: int,
        backbone: str,
    ) -> Prediction:
        prediction = Prediction(
            transport_id=transport_id,
            load_pct=float(load_pct),
            cargo_type=cargo_type,
            models_count=int(models_count),
            backbone=backbone,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._client.table("predictions").upsert(
            prediction.__dict__, on_conflict="transport_id"
        ).execute()
        return prediction

    def get(self, transport_id: str) -> Prediction | None:
        response = (
            self._client.table("predictions")
            .select("*")
            .eq("transport_id", transport_id)
            .maybe_single()
            .execute()
        )
        if response.data is None:
            return None
        return Prediction(**response.data)
