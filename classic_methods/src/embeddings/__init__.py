"""Embedding utilities."""

from .customer_features import CustomerFeatureConfig, CustomerFeatureMatrixBuilder, build_customer_feature_matrix
from .reductions import (
    PCAReducer,
    RecommendationAwareVAEConfig,
    TORCH_AVAILABLE,
    VAEEmbeddingReducer,
    build_recommendation_targets,
    evaluate_cross_sell_predictions,
    evaluate_latent_space,
    fit_pca_embedding,
    preprocess_feature_matrix,
    train_recommendation_aware_vae,
    train_vae,
)
from .simple_embeddings import Function1Embedding, SimpleEmbeddingFunction1, function1

__all__ = [
    "CustomerFeatureConfig",
    "CustomerFeatureMatrixBuilder",
    "Function1Embedding",
    "PCAReducer",
    "RecommendationAwareVAEConfig",
    "SimpleEmbeddingFunction1",
    "TORCH_AVAILABLE",
    "VAEEmbeddingReducer",
    "build_recommendation_targets",
    "build_customer_feature_matrix",
    "evaluate_cross_sell_predictions",
    "evaluate_latent_space",
    "fit_pca_embedding",
    "function1",
    "preprocess_feature_matrix",
    "train_recommendation_aware_vae",
    "train_vae",
]
