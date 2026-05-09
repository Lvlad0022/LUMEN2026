"""Small REST API wrapper around a recommender object.

The API intentionally assumes only one model contract:
the loaded object must expose ``recommend(customer_id, top_k)``.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class RecommenderProtocol(Protocol):
    """Minimal model interface expected by the API."""

    def recommend(self, customer_id: int, top_k: int) -> list[Any]:
        """Return ranked recommendations for one customer."""


class RecommendationRequest(BaseModel):
    """Request body for recommendation calls."""

    customer_id: int = Field(..., description="Customer identifier.")
    top_k: int = Field(10, ge=1, le=100, description="Number of items to recommend.")


class RecommendationItem(BaseModel):
    """One ranked recommendation."""

    rank: int
    item_id: int | str
    score: float | None = None


class RecommendationResponse(BaseModel):
    """Response body returned by the recommender API."""

    customer_id: int
    top_k: int
    recommendations: list[RecommendationItem]


_recommender: RecommenderProtocol | None = None


def set_recommender(model: RecommenderProtocol) -> None:
    """Set the recommender instance used by API handlers."""

    global _recommender
    if not hasattr(model, "recommend") or not callable(getattr(model, "recommend")):
        raise TypeError("Model must expose a callable recommend(customer_id, top_k) method.")
    _recommender = model


def _load_pickled_recommender(path: str | Path) -> RecommenderProtocol:
    """Load a pickled recommender object and validate the expected interface."""

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    if not hasattr(model, "recommend") or not callable(getattr(model, "recommend")):
        raise TypeError("Loaded model must expose recommend(customer_id, top_k).")
    return model


def _normalize_recommendation(raw_item: Any, rank: int) -> RecommendationItem:
    """Convert common recommender outputs into the API response schema."""

    if isinstance(raw_item, dict):
        item_id = raw_item.get("item_id", raw_item.get("item_idx", raw_item.get("id")))
        if item_id is None:
            raise ValueError(f"Recommendation at rank {rank} is missing item_id/item_idx/id.")
        raw_score = raw_item.get("score")
        return RecommendationItem(
            rank=rank,
            item_id=item_id,
            score=None if raw_score is None else float(raw_score),
        )

    if isinstance(raw_item, tuple):
        if not raw_item:
            raise ValueError(f"Empty tuple recommendation at rank {rank}.")
        item_id = raw_item[0]
        score = float(raw_item[1]) if len(raw_item) > 1 and raw_item[1] is not None else None
        return RecommendationItem(rank=rank, item_id=item_id, score=score)

    return RecommendationItem(rank=rank, item_id=raw_item, score=None)


def create_app(model: RecommenderProtocol | None = None) -> FastAPI:
    """Create the FastAPI app.

    A model can be injected directly or loaded from the MODEL_PATH environment variable.
    """

    if model is not None:
        set_recommender(model)
    elif os.getenv("MODEL_PATH"):
        set_recommender(_load_pickled_recommender(os.environ["MODEL_PATH"]))

    app = FastAPI(
        title="LUMEN Recommendation API",
        version="0.1.0",
        description="REST API for returning top-k product recommendations for one customer.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": _recommender is not None,
        }

    @app.post("/recommend", response_model=RecommendationResponse)
    def recommend(request: RecommendationRequest) -> RecommendationResponse:
        if _recommender is None:
            raise HTTPException(
                status_code=503,
                detail="Recommendation model is not loaded. Set MODEL_PATH or inject a model in create_app().",
            )

        try:
            raw_recommendations = _recommender.recommend(request.customer_id, request.top_k)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown customer_id: {request.customer_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        recommendations = [
            _normalize_recommendation(item, rank=rank)
            for rank, item in enumerate(raw_recommendations[: request.top_k], start=1)
        ]
        return RecommendationResponse(
            customer_id=request.customer_id,
            top_k=request.top_k,
            recommendations=recommendations,
        )

    return app


app = create_app()


__all__ = [
    "RecommendationItem",
    "RecommendationRequest",
    "RecommendationResponse",
    "RecommenderProtocol",
    "app",
    "create_app",
    "set_recommender",
]
