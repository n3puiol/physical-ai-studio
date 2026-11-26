# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data-only StarVLA Runtime packages with explicit robot I/O and locked provenance."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import environments
from ._worker.starvla_worker_common import BACKBONES, DEFAULT_BACKBONE, FAST_TOKENIZER_FILES, trained_with_state

if TYPE_CHECKING:
    from collections.abc import Sequence

BACKEND = "starvla"
ARTIFACT = "policy.starvla"
RUN_INFO = "physicalai_run.json"
# StarVLA picks its VLM implementation by a substring of the backbone path (BACKBONES); packages name
# the backbone folder after it and record that name as "backbone" in package.json. Packages without
# the field hold Qwen3-VL.
BACKBONE_DIR = BACKBONES[DEFAULT_BACKBONE]
# StarVLA also reads this marker in the backbone path: it enables the action tokens of StarVLA's
# "-Action" backbones, which QwenFast predicts. Packages keep it in the backbone folder's name.
ACTION_MARKER = "-Action"
# QwenFast packages carry their FAST action tokenizer's data files (never its code) in this folder.
FAST_DIR = "FAST"
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
    def from_dict(cls, value: object) -> IoSpec:
        """Validate the JSON form of an I/O mapping, which may come from an untrusted package.

        Returns:
            Validated mapping.

        Raises:
            ValueError: If the mapping is malformed.
        """
        if not isinstance(value, dict) or set(value) != {"cameras", "state", "action"}:
            msg = "I/O mapping needs cameras, state, and action entries."
            raise ValueError(msg)
        return cls(**value)


def config_value(config_yaml: str | Path | None, overrides: Sequence[str], key: str) -> str | None:
    """Return a dotted string setting from a StarVLA YAML file, letting ``KEY=VALUE`` overrides win.

    Returns:
        The value, or ``None`` when neither the overrides nor the file set a non-empty string.
    """
    for token in reversed(overrides):
        name, _, value = token.removeprefix("--").partition("=")
        if name == key and value.strip():
            return value.strip().strip("'\"")
    if config_yaml is None:
        return None
    import yaml  # ruff: ignore[import-outside-top-level]  # Only needed when reading training configs.

    data: object = yaml.safe_load(Path(config_yaml).expanduser().read_text(encoding="utf-8"))
    for part in key.split("."):
        data = data.get(part) if isinstance(data, dict) else None
    return data if isinstance(data, str) and data else None


def framework_from_config(config_yaml: str | Path | None, overrides: Sequence[str] = ()) -> str | None:
    """Return ``framework.name`` from a StarVLA YAML file, letting ``framework.name=...`` overrides win.

    Returns:
        The framework name, or ``None`` when neither the overrides nor the file set one.
    """
    return config_value(config_yaml, overrides, "framework.name")


def backbone_from_config(config_yaml: str | Path | None, overrides: Sequence[str] = ()) -> str:
    """Return the backbone type StarVLA loads for a training config's ``framework.qwenvl.base_vlm``.

    Returns:
        A key of ``environments.BACKBONE_ENVIRONMENTS``.
    """
    return environments.backbone_for_path(config_value(config_yaml, overrides, "framework.qwenvl.base_vlm"))


def package_environment(descriptor: dict[str, Any]) -> environments.PolicyEnvironment:
    """Return the worker environment a validated package runs in: its framework on its backbone type.

    Returns:
        The environment declaration.
    """
    folder = str(descriptor.get("backbone", BACKBONE_DIR)).removesuffix(ACTION_MARKER)
    kind = next((model_type for model_type, name in BACKBONES.items() if name == folder), DEFAULT_BACKBONE)
    return environments.for_policy(descriptor["policy"], kind)


def checkpoint_uses_state(config_yaml: Path) -> bool:
    """Whether a run's checkpoint was trained with proprioceptive state (``include_state``).

    Returns:
        True when StarVLA's loader packed state into the run's training samples.
    """
    import yaml  # ruff: ignore[import-outside-top-level]  # Only needed when reading training configs.

    data = yaml.safe_load(Path(config_yaml).read_text(encoding="utf-8"))
    datasets = data.get("datasets") if isinstance(data, dict) else None
    vla_data = datasets.get("vla_data") if isinstance(datasets, dict) else None
    return trained_with_state(vla_data if isinstance(vla_data, dict) else {})


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
    backbone = descriptor.get("backbone", BACKBONE_DIR)
    if backbone not in {name + suffix for name in BACKBONES.values() for suffix in ("", ACTION_MARKER)}:
        msg = "Unsupported StarVLA package backbone."
        raise ValueError(msg)
    if descriptor.get("environment") != package_environment(descriptor).provenance():
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
    contained_file(artifact, f"{backbone}/config.json")
    if descriptor["policy"] == "QwenFast":
        if descriptor.get("fast_tokenizer") != FAST_DIR:
            msg = "QwenFast packages carry their FAST tokenizer."
            raise ValueError(msg)
        for name in FAST_TOKENIZER_FILES:
            contained_file(artifact, f"{FAST_DIR}/{name}")
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
    configuration cannot introduce Python files into the worker environment. A checkpoint
    trained without proprioceptive state (StarVLA's ``include_state``, off by default) gets a
    package that maps none, so Runtime neither requires nor sends it.

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
    backbone = backbone.expanduser().resolve(strict=True)
    kind, folder = _backbone_folder(backbone)
    spec = environments.for_policy(framework_from_config(run_dir / "config.yaml") or "", kind)
    if not checkpoint_uses_state(run_dir / "config.yaml"):
        io = replace(io, state={})
    stats = json.loads((run_dir / "dataset_statistics.json").read_text())
    if not isinstance(stats, dict) or unnorm_key not in stats:
        msg = "unnorm_key is not present in the checkpoint's dataset statistics."
        raise ValueError(msg)
    fast = _fast_tokenizer(run_dir / "config.yaml") if spec.policy == "QwenFast" else None
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
        _copy_backbone(backbone, artifact / folder)
        descriptor = {
            "format_version": 1,
            "policy": spec.policy,
            "environment": spec.provenance(),
            "checkpoint": "model/checkpoints/model.safetensors",
            "unnorm_key": unnorm_key,
            "io": asdict(io),
            "backbone": folder,
        }
        if fast is not None:
            (artifact / FAST_DIR).mkdir()
            for name in FAST_TOKENIZER_FILES:
                shutil.copy2(fast / name, artifact / FAST_DIR / name)
            descriptor["fast_tokenizer"] = FAST_DIR
        (artifact / "package.json").write_text(json.dumps(descriptor, indent=2) + "\n")
        load_descriptor(artifact)
        _write_manifest(staged, spec.policy)
        staged.rename(output)
    return output


def _backbone_folder(backbone: Path) -> tuple[str, str]:
    """Return a backbone's type and the package folder named for it.

    StarVLA selects the VLM implementation, and enables action tokens, by substrings of its path.

    Returns:
        The backbone's ``model_type`` and the folder name, keeping StarVLA's "-Action" marker.

    Raises:
        ValueError: If the backbone is not a supported type.
    """
    config = json.loads(contained_file(backbone, "config.json").read_text())
    kind = config.get("model_type") if isinstance(config, dict) else None
    if kind not in BACKBONES:
        msg = "This plugin requires a Qwen3-VL or Qwen3.5 backbone."
        raise ValueError(msg)
    return kind, BACKBONES[kind] + (ACTION_MARKER if ACTION_MARKER in backbone.name else "")


def _fast_tokenizer(config_yaml: Path) -> Path:
    """Return the FAST tokenizer folder a QwenFast run was trained with.

    Raises:
        ValueError: If the run's config does not name an absolute folder with the tokenizer files.
    """
    import yaml  # ruff: ignore[import-outside-top-level]  # Only needed when reading training configs.

    data = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    framework = data.get("framework") if isinstance(data, dict) else None
    action_model = framework.get("action_model") if isinstance(framework, dict) else None
    path = action_model.get("fast_tokenizer_path") if isinstance(action_model, dict) else None
    folder = Path(str(path)) if path else None
    if (
        folder is None
        or not folder.is_absolute()
        or not all((folder / name).is_file() for name in FAST_TOKENIZER_FILES)
    ):
        msg = "QwenFast packages carry the FAST tokenizer: set the run's framework.action_model.fast_tokenizer_path."
        raise ValueError(msg)
    return folder


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
