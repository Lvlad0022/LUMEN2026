"""Shared pipeline contracts for classic methods.

The pipeline is intentionally lightweight:
- each stage declares the artifact types it consumes and produces
- the pipeline checks compatibility before execution
- stages can expose multiple named inputs and outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ARTIFACT_DATAFRAME = "dataframe"
ARTIFACT_MATRIX = "matrix"
ARTIFACT_RELATION_MATRICES = "relation_matrices"
ARTIFACT_SIMILARITY_MATRIX = "similarity_matrix"
ARTIFACT_CLUSTER_LABELS = "cluster_labels"
ARTIFACT_RECOMMENDATION_SCORES = "recommendation_scores"
ARTIFACT_INDEX_ARRAY = "index_array"
ARTIFACT_MAPPING = "mapping"
ARTIFACT_MODEL = "model"


@dataclass(frozen=True)
class ArtifactSpec:
    """Declared artifact type for a named pipeline input or output."""

    name: str
    kind: str
    dense: bool | None = None
    description: str | None = None


@dataclass(frozen=True)
class StageContract:
    """Declared compatibility contract for a pipeline stage."""

    input_type: str | None = None
    output_type: str | None = None
    input_artifacts: dict[str, ArtifactSpec] = field(default_factory=dict)
    output_artifacts: dict[str, ArtifactSpec] = field(default_factory=dict)
    dense: bool | None = None
    description: str | None = None


@dataclass
class PipelineArtifact:
    """Runtime artifact stored by the pipeline."""

    name: str
    kind: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageConfig:
    """Configuration for one stage loaded from YAML."""

    name: str
    import_path: str
    method: str = "fit"
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    config_path: str | None = None


def get_single_output_type(contract: StageContract | None) -> str | None:
    if contract is None:
        return None
    if contract.output_type is not None:
        return contract.output_type
    if len(contract.output_artifacts) == 1:
        return next(iter(contract.output_artifacts.values())).kind
    return None


__all__ = [
    "ARTIFACT_CLUSTER_LABELS",
    "ARTIFACT_DATAFRAME",
    "ARTIFACT_INDEX_ARRAY",
    "ARTIFACT_MATRIX",
    "ARTIFACT_MODEL",
    "ARTIFACT_MAPPING",
    "ARTIFACT_RECOMMENDATION_SCORES",
    "ARTIFACT_RELATION_MATRICES",
    "ARTIFACT_SIMILARITY_MATRIX",
    "ArtifactSpec",
    "PipelineArtifact",
    "StageConfig",
    "StageContract",
    "get_single_output_type",
]
