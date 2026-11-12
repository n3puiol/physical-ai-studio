# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Studio training entry point; all model imports live in the locked worker."""

from __future__ import annotations

import json
import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Native CLI argv, shell disabled.
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from physicalai.engine import Run, TrainingEngine, WorkerEnv, run_worker

from . import environments
from .package import POLICY, RUN_INFO, IoSpec, write_package

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from physicalai.engine import TrainHooks

WORKER_ENV = WorkerEnv(
    framework="starvla",
    project=environments.SOURCE,
    worker=environments.TRAIN_WORKER,
    distribution="physicalai-starvla-policy-plugin",
    package="starvla",
    accelerators=("cu128",),
    python="3.10",
)


class StarVLAEngine(TrainingEngine):
    """Train StarVLA QwenGR00T and export its native checkpoints for Runtime."""

    framework: ClassVar[str] = "starvla"
    env: ClassVar[WorkerEnv] = WORKER_ENV
    training_accelerators: ClassVar[tuple[str, ...]] = ("cu128",)
    stock_export_formats: ClassVar[tuple[str, ...]] = ("native",)

    @classmethod
    def cli(cls, argv: Sequence[str], *, interpreter: str | Path | None = None) -> int:
        """Pass native --config_yaml and OmegaConf dotlist arguments to the worker.

        Returns:
            Native process exit code.
        """
        python = environments.interpreter_path(interpreter)
        with resources.as_file(environments.TRAIN_WORKER) as worker:
            return subprocess.call(  # ruff: ignore[subprocess-without-shell-equals-true]  # No shell expansion.
                [str(python), str(worker), *argv],
                shell=False,
                env={**os.environ, "PHYSICALAI_STARVLA_PROVENANCE": json.dumps(environments.provenance())},
            )

    @classmethod
    def run(
        cls,
        settings: Mapping[str, Any],
        *,
        dataset: Path,
        output_dir: Path,
        device: str,
        hooks: TrainHooks | None = None,
        interpreter: str | Path | None = None,
    ) -> Run:
        """Train with Studio-owned dataset root, output directory, and CUDA device.

        Settings contain config_yaml and optional native KEY=VALUE overrides.
        The dataset root contains the datasets named by the config's data_mix.

        Returns:
            Native training output and final checkpoint.

        Raises:
            ValueError: If settings, device, or output name are invalid.
        """
        unknown = set(settings) - {"config_yaml", "overrides"}
        if unknown or not isinstance(settings.get("config_yaml"), (str, Path)):
            msg = "StarVLA settings require config_yaml and optional overrides only."
            raise ValueError(msg)
        overrides = settings.get("overrides", [])
        if not isinstance(overrides, (list, tuple)) or any(not isinstance(v, str) or "=" not in v for v in overrides):
            msg = "overrides must be a list of KEY=VALUE strings."
            raise ValueError(msg)
        output = output_dir.expanduser().absolute()
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", output.name):
            msg = "The output directory name must use 1-64 letters, digits, underscores, or hyphens."
            raise ValueError(msg)
        variables = {**_device_env(device), "PHYSICALAI_STARVLA_PROVENANCE": json.dumps(environments.provenance())}
        argv = [
            "--config_yaml",
            str(Path(settings["config_yaml"]).expanduser().resolve(strict=True)),
            *overrides,
            f"datasets.vla_data.data_root_dir={json.dumps(str(dataset.expanduser().resolve(strict=True)))}",
            f"run_root_dir={json.dumps(str(output.parent))}",
            f"run_id={output.name}",
        ]
        return run_worker(
            WORKER_ENV,
            argv,
            device="cuda",
            accelerator="cu128",
            hooks=hooks,
            interpreter=interpreter,
            extra_env=variables,
        )

    @classmethod
    def export(cls, source: Run | str | Path, output_path: str | Path, backend: str | None = "native") -> Path:
        """Export using physicalai_run.json and an explicit io.json beside the run.

        Use export_native for checkpoints trained outside the plugin or to supply
        a different robot mapping or materialized local backbone.

        Returns:
            Runtime package directory.

        Raises:
            ValueError: If the training record has different environment pins.
        """
        checkpoint = _checkpoint(source)
        root = checkpoint.parents[1]
        record = json.loads((root / RUN_INFO).read_text())
        if record.get("environment") != environments.provenance():
            msg = "Training record pins differ from the installed StarVLA plugin."
            raise ValueError(msg)
        return cls.export_native(
            checkpoint,
            output_path,
            backend=backend,
            backbone_path=record["backbone_path"],
            io=IoSpec.from_dict(json.loads((root / "io.json").read_text())),
            unnorm_key=record["unnorm_key"],
        )

    @classmethod
    def export_native(
        cls,
        source: Run | str | Path,
        output_path: str | Path,
        *,
        backbone_path: str | Path,
        io: IoSpec,
        unnorm_key: str,
        backend: str | None = "native",
    ) -> Path:
        """Package native weights, normalization, robot mapping, and the local backbone.

        Returns:
            Runtime package directory.

        Raises:
            ValueError: If the requested backend is not native.
        """
        if backend not in {None, "native"}:
            msg = "StarVLA supports native Runtime packages only."
            raise ValueError(msg)
        return write_package(
            _checkpoint(source),
            Path(output_path),
            backbone=Path(backbone_path),
            io=io,
            unnorm_key=unnorm_key,
        )

    @classmethod
    def policies(cls) -> list[str]:
        """Return the supported upstream framework names."""
        return [POLICY]

    @classmethod
    def config_schema(cls, policy: str) -> dict[str, Any]:
        """Describe the small host facade over the upstream YAML configuration.

        Returns:
            JSON schema for headless training settings.

        Raises:
            ValueError: If the policy is not provided by this engine.
        """
        if policy != POLICY:
            msg = f"Unknown StarVLA policy {policy!r}."
            raise ValueError(msg)
        return {
            "type": "object",
            "required": ["config_yaml"],
            "additionalProperties": False,
            "properties": {
                "config_yaml": {"type": "string"},
                "overrides": {"type": "array", "items": {"type": "string"}},
            },
        }


def _checkpoint(source: Run | str | Path) -> Path:
    if isinstance(source, Run):
        if source.framework != "starvla" or source.checkpoint is None:
            msg = "Export needs a StarVLA run with a checkpoint."
            raise ValueError(msg)
        return source.checkpoint
    return Path(source).expanduser().resolve(strict=True)


def _device_env(device: str) -> dict[str, str]:
    if device in {"auto", "cuda", "gpu"}:
        return {}
    if device.startswith("cuda:") and device.removeprefix("cuda:").isdigit():
        return {"CUDA_VISIBLE_DEVICES": device.removeprefix("cuda:")}
    msg = "StarVLA training requires a CUDA device."
    raise ValueError(msg)
