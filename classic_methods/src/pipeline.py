"""Config-driven numbered-stage pipeline for classic methods.

The root config provides the CSV source and numbered stage references:
- ``stage1``
- ``stage2``
- ``stage3``

Each stage points to its own config file, which contains the target callable
and the stage parameters. The pipeline loads the stages in order, validates
artifact compatibility, and stores intermediate artifacts.
"""

from __future__ import annotations

import re
import json
import pickle
from dataclasses import dataclass
from importlib import import_module
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import scipy.sparse as sp

try:
    from .pipeline_contracts import (
        ARTIFACT_DATAFRAME,
        ARTIFACT_MAPPING,
        ARTIFACT_MODEL,
        ARTIFACT_RECOMMENDATION_SCORES,
        ArtifactSpec,
        PipelineArtifact,
        StageConfig,
        StageContract,
    )
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import (  # type: ignore
        ARTIFACT_DATAFRAME,
        ARTIFACT_MAPPING,
        ARTIFACT_MODEL,
        ARTIFACT_RECOMMENDATION_SCORES,
        ArtifactSpec,
        PipelineArtifact,
        StageConfig,
        StageContract,
    )


def _load_object(import_path: str) -> Any:
    """Load a Python object from a dotted import path."""

    module_path, _, attr = import_path.rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"Invalid import path: {import_path}")
    module = import_module(module_path)
    return getattr(module, attr)


def _strip_comment(line: str) -> str:
    """Remove an inline YAML comment from a line of text."""

    if "#" not in line:
        return line
    hash_index = line.find("#")
    if hash_index == -1:
        return line
    return line[:hash_index]


def _parse_scalar(text: str) -> Any:
    """Parse a YAML scalar into a Python value."""

    value = text.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if value in {"{}", "{ }"}:
        return {}
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _clean_lines(path: Path) -> list[str]:
    """Read a YAML-like file and remove blank lines plus comments."""

    cleaned: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = _strip_comment(raw_line).rstrip()
        if stripped.strip():
            cleaned.append(stripped)
    return cleaned


def _indent_of(line: str) -> int:
    """Return the leading-space indent of a line."""

    return len(line) - len(line.lstrip(" "))


def _next_meaningful_indent(lines: list[str], start: int) -> int | None:
    """Find the indent level of the next non-empty line."""

    for idx in range(start, len(lines)):
        if lines[idx].strip():
            return _indent_of(lines[idx])
    return None


def _parse_mapping(lines: list[str], start: int, indent: int) -> tuple[dict[str, Any], int]:
    """Parse a YAML-like mapping block."""

    result: dict[str, Any] = {}
    i = start
    while i < len(lines):
        line = lines[i]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation at line: {line!r}")
        stripped = line.strip()
        if stripped.startswith("- "):
            break
        if ":" not in stripped:
            raise ValueError(f"Invalid mapping line: {line!r}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        i += 1
        if raw_value:
            result[key] = _parse_scalar(raw_value)
            continue
        next_indent = _next_meaningful_indent(lines, i)
        if next_indent is None or next_indent <= indent:
            result[key] = None
            continue
        if next_indent == indent + 2:
            parsed, i = _parse_node(lines, i, indent + 2)
            result[key] = parsed
        else:
            raise ValueError(f"Unsupported indentation structure near key '{key}'.")
    return result, i


def _parse_sequence(lines: list[str], start: int, indent: int) -> tuple[list[Any], int]:
    """Parse a YAML-like sequence block."""

    result: list[Any] = []
    i = start
    while i < len(lines):
        line = lines[i]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation at line: {line!r}")
        stripped = line.strip()
        if not stripped.startswith("- "):
            break
        item_text = stripped[2:].strip()
        i += 1
        if not item_text:
            next_indent = _next_meaningful_indent(lines, i)
            if next_indent is None:
                result.append(None)
                continue
            if next_indent < indent + 2:
                result.append(None)
                continue
            item, i = _parse_node(lines, i, indent + 2)
            result.append(item)
            continue
        if ":" in item_text:
            key, raw_value = item_text.split(":", 1)
            item: dict[str, Any] = {key.strip(): _parse_scalar(raw_value.strip())}
            next_indent = _next_meaningful_indent(lines, i)
            if next_indent is not None and next_indent >= indent + 2:
                extra, i = _parse_mapping(lines, i, indent + 2)
                item.update(extra)
            result.append(item)
            continue
        result.append(_parse_scalar(item_text))
    return result, i


def _parse_node(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    """Parse either a mapping or a sequence block."""

    next_line = None
    for idx in range(start, len(lines)):
        if lines[idx].strip():
            next_line = lines[idx]
            break
    if next_line is None:
        return {}, start
    if _indent_of(next_line) != indent:
        raise ValueError(f"Unexpected indentation near line: {next_line!r}")
    if next_line.strip().startswith("- "):
        return _parse_sequence(lines, start, indent)
    return _parse_mapping(lines, start, indent)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the repo's small YAML subset without requiring PyYAML."""

    lines = _clean_lines(path)
    if not lines:
        return {}
    data, index = _parse_mapping(lines, 0, 0)
    if index < len(lines):
        remaining = lines[index].strip()
        if remaining:
            raise ValueError(f"Unexpected trailing content near: {remaining!r}")
    return data


_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _deep_merge(base: Any, override: Any) -> Any:
    """Deep-merge two YAML-like structures, preferring values from ``override``."""

    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _lookup_path(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    """Look up a dotted path like ``paths.output_dir`` inside a mapping."""

    current: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Unknown config reference: {dotted_path}")
        current = current[part]
    return current


def _resolve_placeholders(value: Any, context: Mapping[str, Any]) -> Any:
    """Recursively resolve ``${...}`` placeholders using the provided context."""

    if isinstance(value, dict):
        return {key: _resolve_placeholders(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(item, context) for item in value]
    if isinstance(value, str):
        matches = list(_PLACEHOLDER_PATTERN.finditer(value))
        if not matches:
            return value
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return _lookup_path(context, matches[0].group(1))

        resolved = value
        for match in matches:
            replacement = _lookup_path(context, match.group(1))
            resolved = resolved.replace(match.group(0), str(replacement))
        return resolved
    return value


def _coerce_contract(obj: Any) -> StageContract:
    """Normalize class- or instance-level stage contracts."""

    contract = getattr(obj, "contract", None)
    if isinstance(contract, StageContract):
        return contract

    input_type = getattr(obj, "input_type", None)
    output_type = getattr(obj, "output_type", None)
    input_artifacts = getattr(obj, "input_artifacts", {}) or {}
    output_artifacts = getattr(obj, "output_artifacts", {}) or {}

    if input_artifacts or output_artifacts or input_type or output_type:
        normalized_inputs = {
            name: spec if isinstance(spec, ArtifactSpec) else ArtifactSpec(name=name, kind=str(spec))
            for name, spec in input_artifacts.items()
        }
        normalized_outputs = {
            name: spec if isinstance(spec, ArtifactSpec) else ArtifactSpec(name=name, kind=str(spec))
            for name, spec in output_artifacts.items()
        }
        return StageContract(
            input_type=input_type,
            output_type=output_type,
            input_artifacts=normalized_inputs,
            output_artifacts=normalized_outputs,
        )

    return StageContract()


def _resolve_stage_instance(import_path: str, params: dict[str, Any]) -> Any:
    """Instantiate or return the callable referenced by ``_target_``."""

    obj = _load_object(import_path)
    if isinstance(obj, type):
        return obj(**params)
    if callable(obj):
        return obj
    raise TypeError(f"Imported object is not callable: {import_path}")


def _coerce_stage_config(name: str, raw: Mapping[str, Any], config_path: Path) -> StageConfig:
    """Convert a stage config file into a runtime ``StageConfig`` object."""

    import_path = raw.get("_target_", raw.get("target", raw.get("import_path")))
    if not import_path:
        raise ValueError(f"Stage config '{config_path}' is missing a '_target_' field.")

    params = raw.get("params", {}) or {}
    inputs = raw.get("inputs", {}) or {}
    if not isinstance(params, Mapping):
        raise ValueError(f"Stage config '{config_path}' field 'params' must be a mapping.")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"Stage config '{config_path}' field 'inputs' must be a mapping.")

    return StageConfig(
        name=name,
        import_path=str(import_path),
        method=str(raw.get("method", "fit")),
        params=dict(params),
        inputs=dict(inputs),
        config_path=str(config_path),
    )


def _stage_number(name: str) -> int | None:
    """Return the numeric suffix for keys named ``stage{number}``."""

    match = re.fullmatch(r"stage(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


@dataclass(frozen=True)
class StageRef:
    """Reference to a numbered stage config from the root config."""

    name: str
    config_path: Path


@dataclass
class ResolvedStage:
    """Fully resolved stage ready for execution."""

    config: StageConfig
    instance: Any
    contract: StageContract


class Pipeline:
    """Config-driven, compatibility-checked stage pipeline."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.project_dir = self._project_dir_from_config_path(self.config_path)
        self.root_config = self._load_config(self.config_path)
        self.paths_config = self._load_config_if_ref(self.root_config.get("paths"))
        self.pipeline_config = self._load_config_if_ref(self.root_config.get("pipeline"))
        if not self.paths_config:
            default_paths_path = self.config_path.parent / "paths.yaml"
            if default_paths_path.exists():
                self.paths_config = self._load_config(default_paths_path)
        self.config_context = _deep_merge(_deep_merge(self.root_config, self.paths_config), self.pipeline_config)
        self.root_config = _resolve_placeholders(self.root_config, self.config_context)
        self.paths_config = _resolve_placeholders(self.paths_config, self.config_context)
        self.pipeline_config = _resolve_placeholders(self.pipeline_config, self.config_context)
        self.config_context = _deep_merge(_deep_merge(self.root_config, self.paths_config), self.pipeline_config)
        self.pipeline_root = self.pipeline_config or self.root_config
        self.data_config = dict(self.pipeline_root.get("data", {}) or {})
        self.output_config = dict(self.pipeline_root.get("output", {}) or {})
        self.save_intermediates = bool(self.pipeline_root.get("save_intermediates", True))
        self.csv_path = self._resolve_csv_path(self.data_config)
        self.output_root = self._resolve_output_dir(self.output_config)
        self.output_dir = self._prepare_run_dir(self.output_root)
        self.stage_refs = self._collect_stage_refs(self.pipeline_root, self.project_dir)
        self.stage_configs = [
            self._load_stage_config(stage_ref.name, stage_ref.config_path)
            for stage_ref in self.stage_refs
        ]
        self.artifacts: dict[str, PipelineArtifact] = {}
        self.resolved_stages: list[ResolvedStage] = []
        self.validate()

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        """Load a root or stage config from disk."""

        loaded = _load_simple_yaml(path)
        if not isinstance(loaded, dict):
            raise ValueError("Pipeline config must be a mapping at the top level.")
        return loaded

    @staticmethod
    def _project_dir_from_config_path(config_path: Path) -> Path:
        """Infer the project directory above the nearest ``config`` folder."""

        for parent in config_path.parents:
            if parent.name == "config":
                return parent.parent
        return config_path.parent.parent

    def _load_config_if_ref(self, section: Any) -> dict[str, Any]:
        """Load a config file referenced by a section with ``config_path``."""

        if not isinstance(section, Mapping):
            return {}
        config_path = section.get("config_path")
        if not config_path:
            return dict(section)
        path = Path(str(config_path))
        if not path.is_absolute():
            path = (self.project_dir / path).resolve()
        loaded = self._load_config(path)
        return loaded

    def _resolve_csv_path(self, data_config: Mapping[str, Any]) -> Path:
        """Resolve the CSV path declared in the root config."""

        csv_path = data_config.get("csv_path", data_config.get("path"))
        if not csv_path:
            raise ValueError("Root config must declare data.csv_path.")
        path = Path(str(csv_path))
        if not path.is_absolute():
            path = (self.project_dir / path).resolve()
        return path

    def _resolve_output_dir(self, output_config: Mapping[str, Any]) -> Path:
        """Resolve the output directory declared in the root config."""

        output_path = output_config.get("dir", output_config.get("path"))
        if not output_path:
            output_path = "output"
        path = Path(str(output_path))
        if not path.is_absolute():
            path = (self.project_dir / path).resolve()
        return path

    def _read_dataframe(self, csv_path: Path) -> pd.DataFrame:
        """Read a CSV with a small encoding fallback chain."""

        attempts = (
            {"encoding": "utf-8", "sep": ",", "low_memory": False},
            {"encoding": "utf-8-sig", "sep": ",", "low_memory": False},
            {"encoding": "utf-16", "sep": "|", "low_memory": False},
            {"encoding": "utf-16", "sep": ",", "low_memory": False},
            {"encoding": "cp1250", "sep": ",", "low_memory": False},
            {"encoding": "latin1", "sep": ",", "low_memory": False},
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return pd.read_csv(csv_path, **kwargs)
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return pd.read_csv(csv_path)

    def _prepare_run_dir(self, output_root: Path) -> Path:
        """Return the next numbered run directory under ``output_root``."""

        output_root.mkdir(parents=True, exist_ok=True)
        run_pattern = re.compile(r"run(\d+)$")
        run_numbers: list[int] = []
        for child in output_root.iterdir():
            if not child.is_dir():
                continue
            match = run_pattern.fullmatch(child.name)
            if match is not None:
                run_numbers.append(int(match.group(1)))
        next_run = max(run_numbers, default=0) + 1
        return output_root / f"run{next_run}"

    def _collect_stage_refs(self, root_config: Mapping[str, Any], base_dir: Path) -> list[StageRef]:
        """Collect and validate contiguous ``stage{n}`` references."""

        numbered: list[tuple[int, str, Any]] = []
        for key, value in root_config.items():
            number = _stage_number(str(key))
            if number is not None:
                numbered.append((number, str(key), value))

        if not numbered:
            return []

        numbered.sort(key=lambda item: item[0])
        expected = 1
        for number, key, _ in numbered:
            if number != expected:
                raise ValueError(
                    f"Missing stage{expected} in root config. Found '{key}' instead."
                )
            expected += 1

        refs: list[StageRef] = []
        for _, key, raw in numbered:
            if not isinstance(raw, Mapping):
                raise ValueError(f"Root config entry '{key}' must be a mapping.")
            config_path = raw.get("config_path", raw.get("path"))
            if not config_path:
                raise ValueError(f"Root config entry '{key}' must declare 'config_path'.")
            path = Path(str(config_path))
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            refs.append(StageRef(name=key, config_path=path))
        return refs

    def _load_stage_config(self, name: str, path: Path) -> StageConfig:
        """Load a numbered stage config file."""

        raw = self._load_config(path)
        raw = _resolve_placeholders(raw, self.config_context)
        return _coerce_stage_config(name, raw, path)

    def validate(self) -> None:
        """Validate stage compatibility without running the pipeline."""

        self.resolved_stages = []
        produced_kinds: dict[str, str] = {"dataframe": ARTIFACT_DATAFRAME}

        for stage_config in self.stage_configs:
            instance = _resolve_stage_instance(stage_config.import_path, stage_config.params)
            contract = _coerce_contract(instance)

            if contract.input_artifacts:
                for input_name, expected_spec in contract.input_artifacts.items():
                    if input_name not in stage_config.inputs:
                        raise ValueError(
                            f"Stage '{stage_config.name}' is missing required input '{input_name}'."
                        )
                    source_name = stage_config.inputs[input_name]
                    if source_name not in produced_kinds:
                        raise ValueError(
                            f"Stage '{stage_config.name}' expects artifact '{source_name}' "
                            f"for input '{input_name}', but it has not been produced yet."
                        )
                    actual_kind = produced_kinds[source_name]
                    if actual_kind != expected_spec.kind:
                        raise ValueError(
                            f"Stage '{stage_config.name}' input '{input_name}' expects kind "
                            f"'{expected_spec.kind}' but got '{actual_kind}' from artifact '{source_name}'."
                        )
            elif contract.input_type:
                if not stage_config.inputs:
                    raise ValueError(
                        f"Stage '{stage_config.name}' declares input_type '{contract.input_type}' "
                        "but does not provide an inputs mapping."
                    )
                source_name = next(iter(stage_config.inputs.values()))
                if source_name not in produced_kinds:
                    raise ValueError(
                        f"Stage '{stage_config.name}' expects artifact '{source_name}', but it has not been produced yet."
                    )
                actual_kind = produced_kinds[source_name]
                if actual_kind != contract.input_type:
                    raise ValueError(
                        f"Stage '{stage_config.name}' expects input type '{contract.input_type}' "
                        f"but got '{actual_kind}'."
                    )

            self.resolved_stages.append(ResolvedStage(stage_config, instance, contract))

            if contract.output_artifacts:
                for artifact_name, artifact_spec in contract.output_artifacts.items():
                    produced_kinds[artifact_name] = artifact_spec.kind
            elif contract.output_type:
                produced_kinds[stage_config.name] = contract.output_type

    def run(
        self,
        initial_artifacts: dict[str, Any] | None = None,
        *,
        save_artifacts: bool | None = None,
        return_recommendations: bool = False,
        recommendation_k: int = 10,
    ) -> dict[str, PipelineArtifact]:
        """Run the configured pipeline and store intermediate artifacts."""

        self.artifacts = {}
        should_save_artifacts = self.save_intermediates if save_artifacts is None else bool(save_artifacts)
        if initial_artifacts and "dataframe" in initial_artifacts:
            dataframe_value = initial_artifacts["dataframe"]
        else:
            dataframe_value = self._read_dataframe(self.csv_path)

        self.artifacts["dataframe"] = PipelineArtifact(
            name="dataframe",
            kind=ARTIFACT_DATAFRAME,
            value=dataframe_value,
            metadata={"source": str(self.csv_path)},
        )

        if initial_artifacts:
            for name, value in initial_artifacts.items():
                if name == "dataframe":
                    continue
                declared_kind = self._infer_kind(value)
                self.artifacts[name] = PipelineArtifact(name=name, kind=declared_kind, value=value)

        for resolved in self.resolved_stages:
            stage = resolved.instance
            stage_config = resolved.config

            kwargs: dict[str, Any] = {}
            for input_name, source_name in stage_config.inputs.items():
                if source_name not in self.artifacts:
                    raise KeyError(
                        f"Stage '{stage_config.name}' requires artifact '{source_name}' for '{input_name}'."
                    )
                kwargs[input_name] = self.artifacts[source_name].value

            method = getattr(stage, stage_config.method, None)
            if method is None or not callable(method):
                raise AttributeError(
                    f"Stage '{stage_config.name}' has no callable method '{stage_config.method}'."
                )

            result = method(**kwargs)

            if hasattr(stage, "export_artifacts") and callable(getattr(stage, "export_artifacts")):
                exported = dict(stage.export_artifacts())
            elif isinstance(result, dict):
                exported = dict(result)
            else:
                exported = {stage_config.name: result}

            for artifact_name, value in exported.items():
                declared_kind = None
                if artifact_name in resolved.contract.output_artifacts:
                    declared_kind = resolved.contract.output_artifacts[artifact_name].kind
                elif artifact_name == stage_config.name and resolved.contract.output_type:
                    declared_kind = resolved.contract.output_type
                kind = declared_kind or self._infer_kind(value)
                self.artifacts[artifact_name] = PipelineArtifact(
                    name=artifact_name,
                    kind=kind,
                    value=value,
                    metadata={"stage": stage_config.name},
                )
                if self.save_intermediates:
                    alias = f"{stage_config.name}.{artifact_name}"
                    self.artifacts[alias] = PipelineArtifact(
                        name=alias,
                        kind=kind,
                        value=value,
                        metadata={"stage": stage_config.name, "alias_of": artifact_name},
                    )

        if return_recommendations:
            recommendations = self._build_recommendations(recommendation_k)
            self.artifacts["recommendations"] = PipelineArtifact(
                name="recommendations",
                kind=ARTIFACT_RECOMMENDATION_SCORES,
                value=recommendations,
                metadata={"k": int(recommendation_k)},
            )

        if should_save_artifacts:
            self._save_artifacts()
        return self.artifacts

    def _build_recommendations(self, recommendation_k: int) -> dict[int, list[int]]:
        """Build ranked recommendations for every known customer from the fitted model."""

        if recommendation_k <= 0:
            raise ValueError("recommendation_k must be a positive integer.")
        if "model" not in self.artifacts:
            raise KeyError("Cannot build recommendations because the pipeline did not produce a 'model' artifact.")
        if "customer_idx" not in self.artifacts:
            raise KeyError(
                "Cannot build recommendations because the pipeline did not produce a 'customer_idx' artifact."
            )

        model = self.artifacts["model"].value
        predict = getattr(model, "predict", None)
        if predict is None or not callable(predict):
            raise AttributeError("Cannot build recommendations because the fitted model has no callable predict().")

        customer_values = self.artifacts["customer_idx"].value
        recommendations: dict[int, list[int]] = {}
        for customer_id in customer_values:
            normalized_customer_id = int(customer_id)
            ranked_items = predict(normalized_customer_id, num_prediction=int(recommendation_k))
            recommendations[normalized_customer_id] = [int(item_id) for item_id in ranked_items]
        return recommendations

    def _save_artifacts(self) -> None:
        """Persist the canonical artifacts to the configured output directory."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, str] = {}

        for name, artifact in self.artifacts.items():
            if "." in name:
                continue
            file_path = self._artifact_path(name, artifact.value)
            self._save_artifact_value(name, file_path, artifact.value)
            manifest[name] = str(file_path)

        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _artifact_path(self, name: str, value: Any) -> Path:
        """Return the file path used to persist one artifact."""

        if name == "dataframe":
            return self.output_dir / "dataframe.csv"
        if name == "idx2item":
            return self.output_dir / "idx2item.json"
        if name == "embedding":
            return self.output_dir / "embedding.npy"
        if name == "clusters":
            return self.output_dir / "clusters.npy"
        if name == "customer_idx":
            return self.output_dir / "customer_idx.npy"
        if name == "item_idx":
            return self.output_dir / "item_idx.npy"
        if name in {"user_user_matrix", "user_item_matrix", "similarity_matrix"}:
            return self.output_dir / f"{name}.npz"
        if name == "model":
            return self.output_dir / "model.pkl"
        if name == "recommendations":
            return self.output_dir / "recommendations.json"
        return self.output_dir / f"{name}.bin"

    def _save_artifact_value(self, name: str, path: Path, value: Any) -> None:
        """Persist a single artifact to disk based on its value type."""

        suffix = path.suffix.lower()
        if suffix == ".csv":
            if not isinstance(value, pd.DataFrame):
                value = pd.DataFrame(value)
            value.to_csv(path, index=False)
            return
        if suffix == ".json":
            if name == "idx2item" and isinstance(value, dict) and all(isinstance(key, int) for key in value):
                ordered = [value[index] for index in sorted(value)]
                path.write_text(json.dumps(ordered, indent=2, default=str), encoding="utf-8")
            else:
                path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
            return
        if suffix == ".npy":
            import numpy as np

            np.save(path, value)
            return
        if suffix == ".npz":
            if not sp.issparse(value):
                value = sp.csr_matrix(value)
            sp.save_npz(path, value)
            return
        if suffix == ".pkl":
            with path.open("wb") as handle:
                pickle.dump(value, handle)
            return
        with path.open("wb") as handle:
            pickle.dump(value, handle)

    @staticmethod
    def _infer_kind(value: Any) -> str:
        """Infer a coarse artifact kind from a Python object."""

        if hasattr(value, "toarray") or hasattr(value, "tocsr"):
            return "matrix"
        if hasattr(value, "columns") and hasattr(value, "shape"):
            return ARTIFACT_DATAFRAME
        if hasattr(value, "ndim") and hasattr(value, "shape"):
            try:
                if int(value.ndim) == 1:
                    return "index_array"
                if int(value.ndim) >= 2:
                    return "matrix"
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            return "index_array"
        if isinstance(value, dict):
            return ARTIFACT_MAPPING
        return ARTIFACT_MODEL


__all__ = ["Pipeline", "ResolvedStage", "StageRef"]
