# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pinned per-policy worker declarations, importable without physicalai-train or StarVLA.

Each StarVLA framework (``framework.name`` in its training YAML) trains and runs in a locked
worker project under ``envs/<name>/``, chosen by the framework and the backbone type StarVLA loads
for ``framework.qwenvl.base_vlm``. Frameworks with the same dependencies share a project; one that
needs different dependencies gets its own, so they never share a resolution. A framework and
backbone without an entry here are not supported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources  # nosemgrep  # Python 3.10+ only.
from pathlib import Path
from typing import TYPE_CHECKING

from physicalai.isolate import BundledProject, Environment, SyncProfile, prepare

from ._worker.starvla_worker_common import (
    BACKBONES,
    DEFAULT_BACKBONE,
    WORLD_MODEL_FRAMEWORKS,
    WORLD_MODELS,
    dispatched_backbone,
    dispatched_world_model,
)

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable  # nosemgrep  # Python 3.10+ only.

    from physicalai.engine import WorkerEnv

FRAMEWORK = "starvla"
PACKAGE = resources.files("physicalai_starvla_policy_plugin")
TRAIN_WORKER = PACKAGE / "_worker" / "starvla_train_worker.py"
INFER_WORKER = PACKAGE / "_worker" / "starvla_infer_worker.py"
OPENVINO_EXPORT_WORKER = PACKAGE / "_worker" / "starvla_openvino_export.py"
PYTHON = "3.11"
ACCELERATOR = "cu128"
DISTRIBUTION = "physicalai-starvla-policy-plugin"


@dataclass(frozen=True)
class PolicyEnvironment:
    """One StarVLA framework's locked worker project and the pins its packages record.

    ``revision`` and ``version`` mirror the project's ``pyproject.toml`` and ``uv.lock``;
    ``test_duplicated_pins_agree`` keeps them in sync.
    """

    policy: str
    name: str
    revision: str
    version: str
    worker: Traversable = TRAIN_WORKER
    backbone: str = DEFAULT_BACKBONE

    @property
    def project(self) -> Traversable:
        """Bundled uv project (``pyproject.toml`` + ``uv.lock``) for this framework."""
        return PACKAGE / "envs" / self.name

    @property
    def source(self) -> BundledProject:
        """Project source consumed by Runtime's isolate provider."""
        return BundledProject(self.project)

    @property
    def environment(self) -> Environment:
        """Isolate environment: the project synced for CUDA with the pinned Python."""
        return Environment(source=self.source, profile=SyncProfile(python=PYTHON, extras=(ACCELERATOR,)))

    def worker_env(self) -> WorkerEnv:
        """Declare this environment to the engine, which then prepares the same cached environment.

        Needs ``physicalai-train`` (the ``train`` extra); Runtime-only installs never call it.

        Returns:
            The engine's worker declaration for this environment.
        """
        from physicalai.engine import WorkerEnv  # ruff: ignore[import-outside-top-level]  # Train extra only.

        return WorkerEnv(
            framework=FRAMEWORK,
            name=self.name,
            project=self.source,
            worker=self.worker,
            distribution=DISTRIBUTION,
            package="starvla",
            accelerators=(ACCELERATOR,),
            python=PYTHON,
        )

    def provenance(self) -> dict[str, str]:
        """Return the pins required to interpret this framework's Runtime packages."""
        return {
            "revision": self.revision,
            "lock_sha256": hashlib.sha256((self.project / "uv.lock").read_bytes()).hexdigest(),
        }


QWENGR00T = PolicyEnvironment(
    policy="QwenGR00T",
    name="qwengr00t",
    revision="13d8ad43eb4128554c57227507055ecf9c3a62a2",
    version="1.0.1",
)
# QwenPI, QwenOFT and QwenFast ship in the same pinned StarVLA wheel and need nothing more, so they share
# QwenGR00T's locked environment; the engine lists a shared environment once.
QWENPI = PolicyEnvironment(
    policy="QwenPI",
    name=QWENGR00T.name,
    revision=QWENGR00T.revision,
    version=QWENGR00T.version,
)
QWENOFT = PolicyEnvironment(
    policy="QwenOFT",
    name=QWENGR00T.name,
    revision=QWENGR00T.revision,
    version=QWENGR00T.version,
)
QWENFAST = PolicyEnvironment(
    policy="QwenFast",
    name=QWENGR00T.name,
    revision=QWENGR00T.revision,
    version=QWENGR00T.version,
)
VLM_ENVIRONMENTS: dict[str, PolicyEnvironment] = {spec.policy: spec for spec in (QWENGR00T, QWENPI, QWENOFT, QWENFAST)}
# World-model frameworks build on a diffusers world model instead of a VLM; the pinned wheel runs them
# with the same dependencies, so they share QwenGR00T's environment too.
WORLD_MODEL_ENVIRONMENTS: dict[str, dict[str, PolicyEnvironment]] = {
    kind: {
        policy: PolicyEnvironment(policy, QWENGR00T.name, QWENGR00T.revision, QWENGR00T.version, backbone=kind)
        for policy, built_for in WORLD_MODEL_FRAMEWORKS.items()
        if built_for == kind
    }
    for kind in WORLD_MODELS
}
# Every framework's own environment, on the backbone type it is built for by default.
POLICY_ENVIRONMENTS: dict[str, PolicyEnvironment] = {
    **VLM_ENVIRONMENTS,
    **{policy: spec for specs in WORLD_MODEL_ENVIRONMENTS.values() for policy, spec in specs.items()},
}

# Qwen3.5 backbones: the same StarVLA commit on transformers 5. QwenFast is not declared: it needs a
# Qwen3.5 backbone with StarVLA's action tokens ("-Action"), which StarVLA does not publish.
QWEN35_BACKBONE = "qwen3_5"
QWEN35_ENVIRONMENTS: dict[str, PolicyEnvironment] = {
    policy: PolicyEnvironment(policy, "qwen35", QWENGR00T.revision, QWENGR00T.version, backbone=QWEN35_BACKBONE)
    for policy in ("QwenGR00T", "QwenPI", "QwenOFT")
}
QWENGR00T_QWEN35 = QWEN35_ENVIRONMENTS["QwenGR00T"]
BACKBONE_ENVIRONMENTS: dict[str, dict[str, PolicyEnvironment]] = {
    DEFAULT_BACKBONE: VLM_ENVIRONMENTS,
    QWEN35_BACKBONE: QWEN35_ENVIRONMENTS,
    **WORLD_MODEL_ENVIRONMENTS,
}
# Display names of the backbone types, for messages.
BACKBONE_NAMES = {**BACKBONES, **{kind: folder for kind, (folder, _pipeline) in WORLD_MODELS.items()}}

# Conversion only: the training project plus OpenVINO and NNCF, with every shared package at the
# same locked version. Kept separate so adding the converter never changes training pins.
QWENGR00T_OPENVINO = PolicyEnvironment(
    policy="QwenGR00T",
    name="qwengr00t-openvino",
    revision=QWENGR00T.revision,
    version=QWENGR00T.version,
    worker=OPENVINO_EXPORT_WORKER,
)
# QwenPI and QwenOFT train in QwenGR00T's environment, so they convert in its OpenVINO one too.
OPENVINO_EXPORT_ENVIRONMENTS: dict[str, PolicyEnvironment] = {
    QWENGR00T_OPENVINO.policy: QWENGR00T_OPENVINO,
    **{
        policy: PolicyEnvironment(
            policy,
            QWENGR00T_OPENVINO.name,
            QWENGR00T_OPENVINO.revision,
            QWENGR00T_OPENVINO.version,
            worker=OPENVINO_EXPORT_WORKER,
        )
        for policy in ("QwenPI", "QwenOFT")
    },
}


def for_policy(policy: str | None, backbone: str | None = None) -> PolicyEnvironment:
    """Return the worker environment for a StarVLA framework on a backbone type.

    Without a backbone type, the framework's own default is used. With ``policy=None`` (e.g.
    ``--help`` without a config), the environment every framework shares is used.

    Returns:
        The framework's environment declaration.

    Raises:
        ValueError: If the framework has no environment for the backbone, or ``None`` is ambiguous.
    """
    if backbone is None:
        backbone = POLICY_ENVIRONMENTS[policy].backbone if policy in POLICY_ENVIRONMENTS else DEFAULT_BACKBONE
    if backbone not in BACKBONE_ENVIRONMENTS:
        msg = f"StarVLA backbone type {backbone!r} is not supported; supported: {', '.join(BACKBONE_ENVIRONMENTS)}."
        raise ValueError(msg)
    environments = BACKBONE_ENVIRONMENTS[backbone]
    if policy is None and len({spec.name for spec in environments.values()}) == 1:
        return next(iter(environments.values()))
    if policy in environments:
        return environments[policy]
    if policy in POLICY_ENVIRONMENTS:
        supported = ", ".join(sorted(environments))
        name = BACKBONE_NAMES[backbone]
        msg = f"StarVLA framework {policy!r} has no {name} environment; {name} supports: {supported}."
        raise ValueError(msg)
    supported = ", ".join(sorted(POLICY_ENVIRONMENTS))
    msg = f"StarVLA framework {policy!r} has no worker environment; supported: {supported}."
    raise ValueError(msg)


def backbone_for(policy: str | None, path: str | Path | None) -> str:
    """Return the backbone type StarVLA loads for a framework's backbone path.

    World-model frameworks take ``framework.world_model.base_wm``, the others
    ``framework.qwenvl.base_vlm``. A path StarVLA would load as another model gets the framework's
    default type; the training worker then refuses it.

    Returns:
        A key of ``BACKBONE_ENVIRONMENTS``.
    """
    if policy in WORLD_MODEL_FRAMEWORKS:
        kind = dispatched_world_model(str(path)) if path else None
        return kind if kind in BACKBONE_ENVIRONMENTS else WORLD_MODEL_FRAMEWORKS[policy]
    return backbone_for_path(path)


def backbone_for_path(base_vlm: str | Path | None) -> str:
    """Return the backbone type StarVLA loads for ``framework.qwenvl.base_vlm``, as its VLM dispatch does.

    Paths StarVLA dispatches to another VLM get the default type; the worker then refuses them.

    Returns:
        A key of ``BACKBONE_ENVIRONMENTS``.
    """
    kind = dispatched_backbone(str(base_vlm)) if base_vlm else None
    return kind if kind in BACKBONE_ENVIRONMENTS else DEFAULT_BACKBONE


def for_openvino_export(policy: str) -> PolicyEnvironment:
    """Return the OpenVINO conversion environment for a StarVLA framework.

    Returns:
        The framework's export environment declaration.

    Raises:
        ValueError: If the framework has no OpenVINO export.
    """
    if policy in OPENVINO_EXPORT_ENVIRONMENTS:
        return OPENVINO_EXPORT_ENVIRONMENTS[policy]
    supported = ", ".join(sorted(OPENVINO_EXPORT_ENVIRONMENTS))
    msg = f"StarVLA framework {policy!r} has no OpenVINO export; supported: {supported}."
    raise ValueError(msg)


def interpreter_path(spec: PolicyEnvironment, interpreter: str | Path | None = None) -> Path:
    """Resolve an explicitly managed interpreter, or provision the framework's locked CUDA worker.

    With ``physicalai-train`` installed, the engine prepares the environment, with the uv installed
    alongside Studio and the labels ``physicalai engine list`` and ``prune`` rely on. A Runtime-only
    install prepares the same cached environment through isolate directly.

    Returns:
        Worker Python executable.
    """
    if interpreter is not None:
        return Path(interpreter).expanduser().resolve(strict=True)
    try:
        from physicalai.engine import prepare_environment  # ruff: ignore[import-outside-top-level]  # Train extra only.
    except ImportError:
        return prepare(spec.environment).python
    return prepare_environment(spec.worker_env(), ACCELERATOR).python
