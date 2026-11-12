# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pinned worker declarations, importable without physicalai-train or StarVLA."""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path

from physicalai.isolate import BundledProject, Environment, SyncProfile, prepare

FRAMEWORK = "starvla"
REVISION = "13d8ad43eb4128554c57227507055ecf9c3a62a2"
PACKAGE = resources.files("physicalai_starvla_policy_plugin")
SOURCE = BundledProject(PACKAGE / "env")
ENVIRONMENT = Environment(source=SOURCE, profile=SyncProfile(python="3.10", extras=("cu128",)))
TRAIN_WORKER = PACKAGE / "_worker" / "starvla_train_worker.py"
INFER_WORKER = PACKAGE / "_worker" / "starvla_infer_worker.py"


def provenance() -> dict[str, str]:
    """Return the pins required to interpret this version's Runtime packages."""
    return {
        "revision": REVISION,
        "lock_sha256": hashlib.sha256((PACKAGE / "env" / "uv.lock").read_bytes()).hexdigest(),
    }


def interpreter_path(interpreter: str | Path | None = None) -> Path:
    """Resolve an explicitly managed interpreter, or provision the locked CUDA worker.

    Returns:
        Worker Python executable.
    """
    if interpreter is not None:
        return Path(interpreter).expanduser().resolve(strict=True)
    return prepare(ENVIRONMENT).python
