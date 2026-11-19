# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO Runtime packages: converted in the locked export worker, run without StarVLA or torch."""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Fixed worker argv, shell disabled.
import tempfile
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import environments
from ._worker import starvla_openvino_pipeline as pipeline
from .package import ARTIFACT, IoSpec, contained_file, write_package

if TYPE_CHECKING:
    from collections.abc import Sequence

BACKEND = "starvla_openvino"
OPENVINO_ARTIFACT = "policy.starvla_ov"
WEIGHT_FORMATS = pipeline.WEIGHT_FORMATS


def load_openvino_descriptor(artifact: Path) -> dict[str, Any]:
    """Validate an OpenVINO package before compiling its graphs.

    Returns:
        Validated package metadata.

    Raises:
        ValueError: If metadata, files, or the robot mapping are invalid.
    """
    artifact = artifact.resolve(strict=True)
    contained_file(artifact, pipeline.DESCRIPTOR)
    descriptor = pipeline.load_descriptor(artifact)
    if descriptor.get("policy") not in environments.OPENVINO_EXPORT_ENVIRONMENTS or "validation" not in descriptor:
        msg = "Unsupported or unvalidated StarVLA OpenVINO package."
        raise ValueError(msg)
    for name in pipeline.MODELS:
        contained_file(artifact, f"{name}.xml")
        contained_file(artifact, f"{name}.bin")
    contained_file(artifact, pipeline.EMBEDDINGS)
    contained_file(artifact, pipeline.TOKENIZER)
    io = IoSpec.from_dict(descriptor.get("io"))
    for label, mapping in (("state", io.state), ("action", io.action)):
        widths = dict(descriptor[label]["keys"])
        if set(mapping) != set(widths) or any(end - start != widths[key] for key, (start, end) in mapping.items()):
            msg = f"Package {label} mapping differs from the exported {label} keys."
            raise ValueError(msg)
    if list(io.cameras) and len(io.cameras) != descriptor["image"]["cameras"]:
        msg = "Package camera mapping differs from the exported camera count."
        raise ValueError(msg)
    return descriptor


def write_openvino_package(
    checkpoint: Path,
    output: Path,
    *,
    backbone: Path,
    io: IoSpec,
    unnorm_key: str,
    weights: str = "fp16",
    image_size: Sequence[int] | None = None,
    interpreter: str | Path | None = None,
) -> Path:
    """Convert a checkpoint to an OpenVINO Runtime package in the locked export environment.

    The native package is staged first (same validation as a native export), converted on CPU
    by the export worker, and checked there against StarVLA's PyTorch implementation. Only the
    OpenVINO graphs, token embeddings, tokenizer and metadata are kept.

    Returns:
        Completed Runtime package directory.

    Raises:
        ValueError: If options are invalid.
        FileExistsError: If the destination already exists.
        RuntimeError: If conversion or validation fails.
    """
    if weights not in WEIGHT_FORMATS:
        msg = f"weights must be one of {', '.join(WEIGHT_FORMATS)}."
        raise ValueError(msg)
    if image_size is not None and (len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size)):  # ruff: ignore[magic-value-comparison]
        msg = "image_size must be (height, width) in pixels."
        raise ValueError(msg)
    if not io.state:
        msg = "The OpenVINO export requires a proprioceptive state mapping."
        raise ValueError(msg)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        msg = "Export destination already exists."
        raise FileExistsError(msg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".starvla-openvino-", dir=output.parent) as temp:
        native = write_package(checkpoint, Path(temp) / "native", backbone=backbone, io=io, unnorm_key=unnorm_key)
        policy = json.loads((native / ARTIFACT / "package.json").read_text())["policy"]
        spec = environments.for_openvino_export(policy)
        staged = Path(temp) / "package"
        artifact = staged / OPENVINO_ARTIFACT
        staged.mkdir()
        python = environments.interpreter_path(spec, interpreter)
        with resources.as_file(environments.OPENVINO_EXPORT_WORKER) as worker:
            argv = [str(python), "-I", str(worker), "--artifact", str(native / ARTIFACT), "--output", str(artifact)]
            argv += ["--weights", weights]
            if image_size is not None:
                argv += ["--image-size", *(str(v) for v in image_size)]
            completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # No shell expansion.
                argv,
                shell=False,
                check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
        if completed.returncode != 0:
            msg = f"StarVLA OpenVINO export failed with exit code {completed.returncode}; see the log above."
            raise RuntimeError(msg)
        descriptor_path = artifact / pipeline.DESCRIPTOR
        descriptor = json.loads(descriptor_path.read_text())
        descriptor["export_environment"] = spec.provenance()
        descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n")
        load_openvino_descriptor(artifact)
        shutil.rmtree(native)
        _write_manifest(staged, policy)
        staged.rename(output)
    return output


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
            artifacts={BACKEND: OPENVINO_ARTIFACT},
            callbacks=[
                ComponentSpec(class_path="physicalai_starvla_policy_plugin.runtime.StarVLAResetCallback", init_args={}),
            ],
        ),
        metadata=MetadataSpec(created_by="physicalai-starvla-policy-plugin"),
    ).save(package / "manifest.json")
