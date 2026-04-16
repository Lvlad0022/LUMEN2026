"""Configuration utilities for resolving the config tree from the base config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


try:
    from .pipeline import _deep_merge, _load_simple_yaml, _resolve_placeholders
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline import _deep_merge, _load_simple_yaml, _resolve_placeholders  # type: ignore


@dataclass(frozen=True)
class ResolvedPipelineConfig:
    """Resolved config components for one pipeline execution mode."""

    base_config_path: Path
    project_dir: Path
    root_config: dict[str, Any]
    paths_config: dict[str, Any]
    pipeline_config: dict[str, Any]
    merged_config: dict[str, Any]


@dataclass(frozen=True)
class ResolvedValidationConfig:
    """Resolved validation config plus resolved recommendation-pipeline config."""

    base_config_path: Path
    project_dir: Path
    root_config: dict[str, Any]
    paths_config: dict[str, Any]
    validation_config: dict[str, Any]
    merged_validation_config: dict[str, Any]
    recommendation: ResolvedPipelineConfig


def project_dir_from_config_path(config_path: str | Path) -> Path:
    """Inputs: config path. Outputs: project directory above the nearest config folder."""

    path = Path(config_path)
    for parent in path.parents:
        if parent.name == "config":
            return parent.parent
    return path.parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    """Inputs: config file path. Outputs: parsed YAML-like mapping."""

    loaded = _load_simple_yaml(Path(path))
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping.")
    return loaded


def resolve_config_reference(project_dir: Path, section: Any) -> tuple[Path | None, dict[str, Any]]:
    """Inputs: project dir and a config section. Outputs: resolved referenced path and loaded mapping or inline mapping."""

    if not isinstance(section, Mapping):
        return None, {}
    config_path = section.get("config_path")
    if not config_path:
        return None, dict(section)
    path = Path(str(config_path))
    if not path.is_absolute():
        path = (project_dir / path).resolve()
    return path, load_config(path)


def resolve_pipeline_config(
    base_config_path: str | Path,
    *,
    pipeline_config_path: str | Path | None = None,
) -> ResolvedPipelineConfig:
    """Inputs: base config path and optional explicit pipeline config path. Outputs: resolved pipeline config bundle."""

    base_path = Path(base_config_path).resolve()
    project_dir = project_dir_from_config_path(base_path)
    root_config = load_config(base_path)
    _, paths_config = resolve_config_reference(project_dir, root_config.get("paths"))

    if pipeline_config_path is None:
        _, pipeline_config = resolve_config_reference(project_dir, root_config.get("pipeline"))
    else:
        explicit_pipeline_path = Path(pipeline_config_path)
        if not explicit_pipeline_path.is_absolute():
            explicit_pipeline_path = (project_dir / explicit_pipeline_path).resolve()
        pipeline_config = load_config(explicit_pipeline_path)

    config_context = _deep_merge(_deep_merge(root_config, paths_config), pipeline_config)
    resolved_root = _resolve_placeholders(root_config, config_context)
    resolved_paths = _resolve_placeholders(paths_config, config_context)
    resolved_pipeline = _resolve_placeholders(pipeline_config, config_context)
    merged = _deep_merge(_deep_merge(resolved_root, resolved_paths), resolved_pipeline)
    return ResolvedPipelineConfig(
        base_config_path=base_path,
        project_dir=project_dir,
        root_config=resolved_root,
        paths_config=resolved_paths,
        pipeline_config=resolved_pipeline,
        merged_config=merged,
    )


def resolve_validation_config(base_config_path: str | Path) -> ResolvedValidationConfig:
    """Inputs: base config path. Outputs: resolved validation bundle plus resolved recommendation-pipeline bundle."""

    base_path = Path(base_config_path).resolve()
    project_dir = project_dir_from_config_path(base_path)
    root_config = load_config(base_path)
    _, paths_config = resolve_config_reference(project_dir, root_config.get("paths"))
    _, validation_config = resolve_config_reference(project_dir, root_config.get("validation"))

    config_context = _deep_merge(_deep_merge(root_config, paths_config), validation_config)
    resolved_root = _resolve_placeholders(root_config, config_context)
    resolved_paths = _resolve_placeholders(paths_config, config_context)
    resolved_validation = _resolve_placeholders(validation_config, config_context)
    merged_validation = _deep_merge(_deep_merge(resolved_root, resolved_paths), resolved_validation)

    pipeline_section = dict(resolved_validation.get("pipeline", {}) or {})
    recommendation_config_path = pipeline_section.get("recommendation_config_path")
    if not recommendation_config_path:
        raise ValueError("validation.pipeline.recommendation_config_path is required.")

    recommendation = resolve_pipeline_config(
        base_path,
        pipeline_config_path=str(recommendation_config_path),
    )
    return ResolvedValidationConfig(
        base_config_path=base_path,
        project_dir=project_dir,
        root_config=resolved_root,
        paths_config=resolved_paths,
        validation_config=resolved_validation,
        merged_validation_config=merged_validation,
        recommendation=recommendation,
    )


__all__ = [
    "ResolvedPipelineConfig",
    "ResolvedValidationConfig",
    "load_config",
    "project_dir_from_config_path",
    "resolve_config_reference",
    "resolve_pipeline_config",
    "resolve_validation_config",
]
