# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data-only StarVLA Runtime packages with explicit robot I/O and locked provenance."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import environments

if TYPE_CHECKING:
    from collections.abc import Sequence

BACKEND = "starvla"
ARTIFACT = "policy.starvla"
RUN_INFO = "physicalai_run.json"
BACKBONE_DIR = "Qwen3-VL"
_SLICE_BOUNDS = 2


@dataclass(frozen=True)
class IoSpec:
    """Map ordered StarVLA video/state/action keys to Runtime cameras and vector slices.

    State/action slices refer to the Runtime vectors, not normalized model values.
    Empty state explicitly disables proprioception. Actions cover the output vector
    exactly once; their names must match the checkpoint's training DataConfig.
    """

    cameras: dict[str, str]
    state: dict[str, tuple[int, int]]
    action: dict[str, tuple[int, int]]

    def __post_init__(self) -> None:
        """Reject ambiguous or incomplete robot mappings.

        Raises:
            ValueError: If keys or vector slices are invalid.
        """
        if not isinstance(self.cameras, dict) or not self.cameras:
            msg = "At least one explicit camera mapping is required."
            raise ValueError(msg)
        if any(
            not isinstance(k, str) or not k.startswith("video.") or not isinstance(v, str) or not v
            for k, v in self.cameras.items()
        ):
            msg = "Cameras must map StarVLA video.<key> names to nonempty Runtime camera names."
            raise ValueError(msg)
        for label, mapping in (("state", self.state), ("action", self.action)):
            if not isinstance(mapping, dict) or (label == "action" and not mapping):
                msg = f"Invalid {label} mapping."
                raise ValueError(msg)
            spans = []
            for key, bounds in mapping.items():
                if not isinstance(key, str) or not key.startswith(f"{label}."):
                    msg = f"Invalid {label} key {key!r}."
                    raise ValueError(msg)
                if (
                    not isinstance(bounds, (list, tuple))
                    or len(bounds) != _SLICE_BOUNDS
                    or any(type(v) is not int for v in bounds)
                    or not 0 <= bounds[0] < bounds[1]
                ):
                    msg = f"Invalid {label} slice for {key!r}."
                    raise ValueError(msg)
                spans.append(tuple(bounds))
            end = 0
            for start, stop in sorted(spans):
                if start < end or (label == "action" and start != end):
                    msg = f"Overlapping or incomplete {label} slices."
                    raise ValueError(msg)
                end = stop

    @property
    def action_dim(self) -> int:
        """Width of the Runtime action vector."""
        return max(end for _, end in self.action.values())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IoSpec:
        """Validate the JSON form of an I/O mapping.

        Returns:
            Validated mapping.

        Raises:
            ValueError: If the mapping is malformed.
        """
        if not isinstance(value, dict) or set(value) != {"cameras", "state", "action"}:
            msg = "I/O mapping needs cameras, state, and action entries."
            raise ValueError(msg)
        return cls(**value)


def framework_from_config(config_yaml: str | Path | None, overrides: Sequence[str] = ()) -> str | None:
    """Return ``framework.name`` from a StarVLA YAML file, letting ``framework.name=...`` overrides win.

    Returns:
        The framework name, or ``None`` when neither the overrides nor the file set one.
    """
    for token in reversed(overrides):
        key, _, value = token.removeprefix("--").partition("=")
        if key == "framework.name" and value.strip():
            return value.strip().strip("'\"")
    if config_yaml is None:
        return None
    import yaml  # ruff: ignore[import-outside-top-level]  # Only needed when reading training configs.

    data = yaml.safe_load(Path(config_yaml).expanduser().read_text(encoding="utf-8"))
    framework = data.get("framework") if isinstance(data, dict) else None
    name = framework.get("name") if isinstance(framework, dict) else None
    return name if isinstance(name, str) and name else None


def contained_file(root: Path, relative: object) -> Path:
    """Resolve a package file, rejecting absolute paths and symlink escapes.

    Returns:
        Resolved file path inside the package.

    Raises:
        ValueError: If the path is invalid or missing.
    """
    if not isinstance(relative, str) or Path(relative).is_absolute():
        msg = "Package paths must be relative strings."
        raise ValueError(msg)
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        msg = "Package file is missing or escapes the artifact."
        raise ValueError(msg)
    return path


def load_descriptor(artifact: Path) -> dict[str, Any]:
    """Read and validate a package before starting its isolated model worker.

    Returns:
        Validated package metadata.

    Raises:
        ValueError: If metadata, paths, or environment pins are invalid.
    """
    descriptor = json.loads(contained_file(artifact, "package.json").read_text())
    if (
        not isinstance(descriptor, dict)
        or type(descriptor.get("format_version")) is not int
        or descriptor["format_version"] != 1
        or descriptor.get("policy") not in environments.POLICY_ENVIRONMENTS
    ):
        msg = "Unsupported StarVLA package descriptor."
        raise ValueError(msg)
    if descriptor.get("environment") != environments.for_policy(descriptor["policy"]).provenance():
        msg = "StarVLA environment pins differ; install the plugin version that exported this package."
        raise ValueError(msg)
    if not isinstance(descriptor.get("unnorm_key"), str) or not descriptor["unnorm_key"]:
        msg = "An explicit normalization key is required."
        raise ValueError(msg)
    IoSpec.from_dict(descriptor.get("io"))
    checkpoint = contained_file(artifact, descriptor.get("checkpoint"))
    if checkpoint.suffix != ".safetensors":
        msg = "Only safetensors checkpoints are supported."
        raise ValueError(msg)
    contained_file(artifact, "model/config.yaml")
    contained_file(artifact, "model/dataset_statistics.json")
    contained_file(artifact, f"{BACKBONE_DIR}/config.json")
    return descriptor


def write_package(
    checkpoint: Path,
    output: Path,
    *,
    backbone: Path,
    io: IoSpec,
    unnorm_key: str,
) -> Path:
    """Atomically package a local safetensors checkpoint and local backbone for Runtime.

    Existing exports are never overwritten. Only model data is copied; package
    configuration cannot introduce Python files into the worker environment.

    Returns:
        Completed Runtime package directory.

    Raises:
        ValueError: If model files, normalization key, or paths are invalid.
        FileExistsError: If the destination already exists.
    """
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if checkpoint.suffix != ".safetensors" or not checkpoint.is_file():
        msg = "Export needs a local .safetensors checkpoint, not a pickle/.pt file."
        raise ValueError(msg)
    run_dir = checkpoint.parents[1]
    for name in ("config.yaml", "dataset_statistics.json"):
        contained_file(run_dir, name)
    spec = environments.for_policy(framework_from_config(run_dir / "config.yaml") or "")
    stats = json.loads((run_dir / "dataset_statistics.json").read_text())
    if not isinstance(stats, dict) or unnorm_key not in stats:
        msg = "unnorm_key is not present in the checkpoint's dataset statistics."
        raise ValueError(msg)
    backbone = backbone.expanduser().resolve(strict=True)
    config = json.loads(contained_file(backbone, "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_vl":
        msg = "This plugin requires a Qwen3-VL backbone."
        raise ValueError(msg)
    if not any(backbone.glob("*.safetensors")):
        msg = "The local backbone must contain safetensors weights."
        raise ValueError(msg)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        msg = "Export destination already exists."
        raise FileExistsError(msg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".starvla-", dir=output.parent) as temp:
        staged = Path(temp) / "package"
        artifact = staged / ARTIFACT
        model = artifact / "model"
        (model / "checkpoints").mkdir(parents=True)
        shutil.copy2(checkpoint, model / "checkpoints" / "model.safetensors")
        for name in ("config.yaml", "dataset_statistics.json"):
            shutil.copy2(run_dir / name, model / name)
        # Upstream selects the VLM implementation by a substring of its local path.
        _copy_backbone(backbone, artifact / BACKBONE_DIR)
        descriptor = {
            "format_version": 1,
            "policy": spec.policy,
            "environment": spec.provenance(),
            "checkpoint": "model/checkpoints/model.safetensors",
            "unnorm_key": unnorm_key,
            "io": asdict(io),
        }
        (artifact / "package.json").write_text(json.dumps(descriptor, indent=2) + "\n")
        load_descriptor(artifact)
        _write_manifest(staged, spec.policy)
        staged.rename(output)
    return output


def _copy_backbone(source: Path, destination: Path) -> None:
    destination.mkdir()
    for path in source.rglob("*"):
        if path.suffix not in {".json", ".safetensors", ".txt", ".model", ".jinja"}:
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source) or not resolved.is_file():
            msg = "Materialize backbone symlinks inside the source directory before export."
            raise ValueError(msg)
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, target)


def _write_manifest(package: Path, policy: str) -> None:
    from physicalai.inference.manifest import (  # ruff: ignore[import-outside-top-level]  # Packaging-only Runtime imports.
        ComponentSpec,
        Manifest,
        MetadataSpec,
        ModelSpec,
        PolicySource,
        PolicySpec,
    )

    Manifest(
        policy=PolicySpec(name=policy, source=PolicySource()),
        model=ModelSpec(
            artifacts={BACKEND: ARTIFACT},
            callbacks=[
                ComponentSpec(class_path="physicalai_starvla_policy_plugin.runtime.StarVLAResetCallback", init_args={}),
            ],
        ),
        metadata=MetadataSpec(created_by="physicalai-starvla-policy-plugin"),
    ).save(package / "manifest.json")
