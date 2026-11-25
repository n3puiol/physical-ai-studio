# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pinned per-policy worker declarations, importable without physicalai-train or StarVLA.

Each StarVLA framework (``framework.name`` in its training YAML) trains and runs in a locked
worker project under ``envs/<name>/``. Frameworks with the same dependencies share a project;
one that needs different dependencies gets its own, so they never share a resolution. A framework
without an entry here is not supported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources  # nosemgrep  # Python 3.10+ only.
from pathlib import Path
from typing import TYPE_CHECKING

from physicalai.isolate import BundledProject, Environment, SyncProfile, prepare

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
# QwenPI ships in the same pinned StarVLA wheel and needs nothing more, so it shares QwenGR00T's
# locked environment; the engine lists a shared environment once.
QWENPI = PolicyEnvironment(
    policy="QwenPI",
    name=QWENGR00T.name,
    revision=QWENGR00T.revision,
    version=QWENGR00T.version,
)
POLICY_ENVIRONMENTS: dict[str, PolicyEnvironment] = {spec.policy: spec for spec in (QWENGR00T, QWENPI)}

# Conversion only: the training project plus OpenVINO and NNCF, with every shared package at the
# same locked version. Kept separate so adding the converter never changes training pins.
QWENGR00T_OPENVINO = PolicyEnvironment(
    policy="QwenGR00T",
    name="qwengr00t-openvino",
    revision=QWENGR00T.revision,
    version=QWENGR00T.version,
    worker=OPENVINO_EXPORT_WORKER,
)
OPENVINO_EXPORT_ENVIRONMENTS: dict[str, PolicyEnvironment] = {QWENGR00T_OPENVINO.policy: QWENGR00T_OPENVINO}


def for_policy(policy: str | None) -> PolicyEnvironment:
    """Return the worker environment for a StarVLA framework.

    With ``policy=None`` (e.g. ``--help`` without a config), the environment every framework
    shares is used.

    Returns:
        The framework's environment declaration.

    Raises:
        ValueError: If the framework has no environment, or ``None`` is ambiguous.
    """
    if policy is None and len({spec.name for spec in POLICY_ENVIRONMENTS.values()}) == 1:
        return next(iter(POLICY_ENVIRONMENTS.values()))
    if policy in POLICY_ENVIRONMENTS:
        return POLICY_ENVIRONMENTS[policy]
    supported = ", ".join(sorted(POLICY_ENVIRONMENTS))
    msg = f"StarVLA framework {policy!r} has no worker environment; supported: {supported}."
    raise ValueError(msg)


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
