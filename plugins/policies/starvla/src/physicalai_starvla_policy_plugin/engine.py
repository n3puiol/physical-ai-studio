# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Studio training entry point; all model imports live in the locked worker."""

from __future__ import annotations

import json
import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Native CLI argv, shell disabled.  # nosec B404
from importlib import resources  # nosemgrep  # Python 3.10+ only.
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from physicalai.engine import Enablement, Run, TrainingEngine, WorkerEnv, run_worker

from . import environments
from ._worker.starvla_worker_common import BACKBONE_VARIABLE
from .openvino_package import write_openvino_package
from .package import RUN_INFO, IoSpec, backbone_from_config, framework_from_config, write_package

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from physicalai.engine import TrainHooks


# The framework the host chose the worker environment for; the training worker refuses any other.
FRAMEWORK_VARIABLE = "PHYSICALAI_STARVLA_FRAMEWORK"
# Devices that leave the worker every GPU the caller exposes.
_ALL_VISIBLE_DEVICES = frozenset({"auto", "cuda", "gpu"})

# StarVLA has no OpenVINO path; the plugin adds one, converting in its own locked environment.
OPENVINO_ENABLEMENTS = tuple(
    Enablement(
        name=spec.name,
        provides={policy: {"export": ("openvino",)}},
        upstream=f"https://github.com/starVLA/starVLA@{spec.revision} + OpenVINO and NNCF",
        env=spec.worker_env(),
    )
    for policy, spec in environments.OPENVINO_EXPORT_ENVIRONMENTS.items()
)
# Qwen3.5 backbones train and run in their own environment (transformers 5); the engine picks it
# when StarVLA would load Qwen3.5 for framework.qwenvl.base_vlm.
QWEN35_ENABLEMENT = Enablement(
    name=environments.QWENGR00T_QWEN35.name,
    provides={policy: {"backbone": (environments.QWEN35_BACKBONE,)} for policy in environments.QWEN35_ENVIRONMENTS},
    upstream=f"https://github.com/starVLA/starVLA@{environments.QWENGR00T_QWEN35.revision} on transformers 5",
    env=environments.QWENGR00T_QWEN35.worker_env(),
)


class StarVLAEngine(TrainingEngine):
    """Train StarVLA frameworks in per-framework worker environments and export them for Runtime.

    Each framework (``framework.name`` in the training YAML) maps to its own locked environment in
    ``policy_envs``; a Qwen3.5 backbone (``framework.qwenvl.base_vlm``) uses the ``qwen35``
    enablement's environment instead. There is no default environment: an unlisted framework is
    rejected rather than run with another framework's dependencies.
    """

    framework: ClassVar[str] = environments.FRAMEWORK
    env: ClassVar[WorkerEnv | None] = None
    policy_envs: ClassVar[Mapping[str, WorkerEnv]] = MappingProxyType(
        {policy: spec.worker_env() for policy, spec in environments.POLICY_ENVIRONMENTS.items()},
    )
    enablements: ClassVar[tuple[Enablement, ...]] = (*OPENVINO_ENABLEMENTS, QWEN35_ENABLEMENT)
    training_accelerators: ClassVar[tuple[str, ...]] = (environments.ACCELERATOR,)
    stock_export_formats: ClassVar[tuple[str, ...]] = ("native",)

    @classmethod
    def cli(cls, argv: Sequence[str], *, interpreter: str | Path | None = None) -> int:
        """Pass native --config_yaml and OmegaConf dotlist arguments to the framework's worker.

        Returns:
            Native process exit code.
        """
        config, overrides = _config_from_cli(argv)
        policy = framework_from_config(config, overrides)
        spec = environments.for_policy(policy, backbone_from_config(config, overrides))
        python = environments.interpreter_path(spec, interpreter)
        variables = {"PHYSICALAI_STARVLA_PROVENANCE": json.dumps(spec.provenance()), BACKBONE_VARIABLE: spec.backbone}
        if policy is not None:
            variables[FRAMEWORK_VARIABLE] = spec.policy
        with resources.as_file(environments.TRAIN_WORKER) as worker:
            return subprocess.call(  # ruff: ignore[subprocess-without-shell-equals-true]  # No shell expansion.  # nosec B603
                [str(python), str(worker), *argv],
                shell=False,
                env={**os.environ, **variables},
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

        Settings contain config_yaml, optional native KEY=VALUE overrides, and optional
        ``all_devices``: train on every GPU the caller exposes (``CUDA_VISIBLE_DEVICES``), one
        data-parallel process per GPU, with ``device="cuda"``. The per-device batch size stays as
        configured, so each step uses that many samples per GPU.
        The dataset root contains the datasets named by the config's data_mix.

        Returns:
            Native training output and final checkpoint.

        Raises:
            ValueError: If settings, device, or output name are invalid.
            TypeError: If ``all_devices`` is not a boolean.
        """
        unknown = set(settings) - {"config_yaml", "overrides", "all_devices"}
        if unknown or not isinstance(settings.get("config_yaml"), (str, Path)):
            msg = "StarVLA settings require config_yaml, and allow overrides and all_devices only."
            raise ValueError(msg)
        all_devices = settings.get("all_devices", False)
        if not isinstance(all_devices, bool):
            msg = "all_devices must be true or false."
            raise TypeError(msg)
        if all_devices and device not in _ALL_VISIBLE_DEVICES:
            msg = (
                "all_devices trains on every visible GPU: use device='cuda' and choose GPUs with CUDA_VISIBLE_DEVICES."
            )
            raise ValueError(msg)
        overrides = settings.get("overrides", [])
        if not isinstance(overrides, (list, tuple)) or any(not isinstance(v, str) or "=" not in v for v in overrides):
            msg = "overrides must be a list of KEY=VALUE strings."
            raise ValueError(msg)
        output = output_dir.expanduser().absolute()
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", output.name):
            msg = "The output directory name must use 1-64 letters, digits, underscores, or hyphens."
            raise ValueError(msg)
        device_variables = _device_env(device)
        config_yaml = Path(settings["config_yaml"]).expanduser().resolve(strict=True)
        spec = environments.for_policy(
            framework_from_config(config_yaml, overrides) or "",
            backbone_from_config(config_yaml, overrides),
        )
        variables = {
            **device_variables,
            "PHYSICALAI_STARVLA_PROVENANCE": json.dumps(spec.provenance()),
            FRAMEWORK_VARIABLE: spec.policy,
            BACKBONE_VARIABLE: spec.backbone,
        }
        argv = [
            "--config_yaml",
            str(config_yaml),
            *overrides,
            f"datasets.vla_data.data_root_dir={json.dumps(str(dataset.expanduser().resolve(strict=True)))}",
            f"run_root_dir={json.dumps(str(output.parent))}",
            f"run_id={output.name}",
            *(["--all_devices"] if all_devices else []),
        ]
        return run_worker(
            spec.worker_env(),
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

        ``backend="openvino"`` converts on CPU in the locked OpenVINO export environment with
        fp16 weights and the camera resolution of the training dataset. Use export_native for
        checkpoints trained outside the plugin, a different robot mapping, backbone, camera
        resolution, or OpenVINO weight format.

        Returns:
            Runtime package directory.

        Raises:
            ValueError: If the training record has different environment pins.
        """
        checkpoint = _checkpoint(source)
        root = checkpoint.parents[1]
        record = json.loads((root / RUN_INFO).read_text())
        spec = environments.for_policy(
            framework_from_config(root / "config.yaml") or "",
            environments.backbone_for_path(record.get("backbone_path")),
        )
        if record.get("environment") != spec.provenance():
            msg = f"Training record pins differ from the installed StarVLA {spec.policy} environment."
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
        openvino_weights: str = "fp16",
        image_size: tuple[int, int] | None = None,
        interpreter: str | Path | None = None,
    ) -> Path:
        """Package native weights, or convert them to OpenVINO, with normalization and robot mapping.

        ``native`` copies the checkpoint and local backbone for the isolated CUDA worker.
        ``openvino`` converts in the locked export environment (``interpreter`` overrides it) and
        validates against PyTorch; the package then runs in-process on any OpenVINO device.
        ``openvino_weights`` is one of ``fp32``, ``fp16`` or ``int8``. ``image_size`` (height,
        width) fixes the camera resolution; by default it is read from the training dataset.

        Returns:
            Runtime package directory.

        Raises:
            ValueError: If the requested backend is unsupported.
        """
        if backend == "openvino":
            return write_openvino_package(
                _checkpoint(source),
                Path(output_path),
                backbone=Path(backbone_path),
                io=io,
                unnorm_key=unnorm_key,
                weights=openvino_weights,
                image_size=image_size,
                interpreter=interpreter,
            )
        if backend not in {None, "native"}:
            msg = "StarVLA exports native or openvino Runtime packages."
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
        """Return the StarVLA frameworks that have a worker environment."""
        return sorted(environments.POLICY_ENVIRONMENTS)

    @classmethod
    def config_schema(cls, policy: str) -> dict[str, Any]:
        """Describe the small host facade over the upstream YAML configuration.

        Returns:
            JSON schema for headless training settings.

        Raises:
            ValueError: If the policy is not provided by this engine.
        """
        if policy not in environments.POLICY_ENVIRONMENTS:
            msg = f"Unknown StarVLA policy {policy!r}."
            raise ValueError(msg)
        return {
            "type": "object",
            "required": ["config_yaml"],
            "additionalProperties": False,
            "properties": {
                "config_yaml": {"type": "string"},
                "overrides": {"type": "array", "items": {"type": "string"}},
                "all_devices": {"type": "boolean"},
            },
        }


def _policy_from_cli(arguments: Sequence[str]) -> str | None:
    """Read the framework a native command line trains: its ``--config_yaml`` plus dotlist overrides.

    Returns:
        The framework name, or ``None`` without a readable config (e.g. ``--help``).
    """
    return framework_from_config(*_config_from_cli(arguments))


def _config_from_cli(arguments: Sequence[str]) -> tuple[str | None, list[str]]:
    """Split a native command line into its readable ``--config_yaml`` and its dotlist overrides.

    Returns:
        The config path (``None`` if missing or unreadable) and the ``KEY=VALUE`` overrides.
    """
    config: str | None = None
    overrides: list[str] = []
    remaining = iter(arguments)
    for argument in remaining:
        if argument == "--config_yaml":
            config = next(remaining, None)
        elif argument.startswith("--config_yaml="):
            config = argument.partition("=")[2]
        elif "=" in argument:
            overrides.append(argument)
    if config is not None and not Path(config).expanduser().is_file():
        config = None
    return config, overrides


def _checkpoint(source: Run | str | Path) -> Path:
    if isinstance(source, Run):
        if source.framework != "starvla" or source.checkpoint is None:
            msg = "Export needs a StarVLA run with a checkpoint."
            raise ValueError(msg)
        return source.checkpoint
    return Path(source).expanduser().resolve(strict=True)


def _device_env(device: str) -> dict[str, str]:
    """Pin the worker to one GPU, treating ``cuda:N`` as an index into the caller's visible devices.

    ``cuda:N`` is logical: with ``CUDA_VISIBLE_DEVICES=2,3`` already set, ``cuda:1`` is physical
    GPU 3. Replacing the mask with ``N`` would move the worker onto a GPU the caller excluded.

    Returns:
        Worker environment overrides; empty to inherit the caller's devices.

    Raises:
        ValueError: If the device is not CUDA or the index is outside the caller's visible devices.
    """
    if device in _ALL_VISIBLE_DEVICES:
        return {}
    index = device.removeprefix("cuda:")
    if not device.startswith("cuda:") or not index.isdigit():
        msg = "StarVLA training requires a CUDA device."
        raise ValueError(msg)
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return {"CUDA_VISIBLE_DEVICES": index}
    visible = [entry.strip() for entry in mask.split(",") if entry.strip()]
    if int(index) >= len(visible):
        msg = f"{device} is outside CUDA_VISIBLE_DEVICES={mask!r}, which exposes {len(visible)} device(s)."
        raise ValueError(msg)
    return {"CUDA_VISIBLE_DEVICES": visible[int(index)]}
