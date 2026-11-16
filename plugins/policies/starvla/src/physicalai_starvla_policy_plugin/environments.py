# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pinned per-policy worker declarations, importable without physicalai-train or StarVLA.

Each StarVLA framework (``framework.name`` in its training YAML) trains and runs in its own
locked worker project under ``envs/<name>/``, so frameworks with different dependencies never
share a resolution. A framework without an entry here is not supported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from physicalai.isolate import BundledProject, Environment, SyncProfile, prepare

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

FRAMEWORK = "starvla"
PACKAGE = resources.files("physicalai_starvla_policy_plugin")
TRAIN_WORKER = PACKAGE / "_worker" / "starvla_train_worker.py"
INFER_WORKER = PACKAGE / "_worker" / "starvla_infer_worker.py"
PYTHON = "3.10"
ACCELERATOR = "cu128"


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
POLICY_ENVIRONMENTS: dict[str, PolicyEnvironment] = {QWENGR00T.policy: QWENGR00T}


def for_policy(policy: str | None) -> PolicyEnvironment:
    """Return the worker environment for a StarVLA framework.

    With ``policy=None`` (e.g. ``--help`` without a config) the only declared framework is used.

    Returns:
        The framework's environment declaration.

    Raises:
        ValueError: If the framework has no environment, or ``None`` is ambiguous.
    """
    if policy is None and len(POLICY_ENVIRONMENTS) == 1:
        return next(iter(POLICY_ENVIRONMENTS.values()))
    if policy in POLICY_ENVIRONMENTS:
        return POLICY_ENVIRONMENTS[policy]
    supported = ", ".join(sorted(POLICY_ENVIRONMENTS))
    msg = f"StarVLA framework {policy!r} has no worker environment; supported: {supported}."
    raise ValueError(msg)


def interpreter_path(spec: PolicyEnvironment, interpreter: str | Path | None = None) -> Path:
    """Resolve an explicitly managed interpreter, or provision the framework's locked CUDA worker.

    Returns:
        Worker Python executable.
    """
    if interpreter is not None:
        return Path(interpreter).expanduser().resolve(strict=True)
    return prepare(spec.environment).python
