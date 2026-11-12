# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Register the StarVLA backend without importing its framework or Studio."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.inference.adapters.registry import AdapterRegistry


def register(registry: AdapterRegistry) -> None:
    """Install the lazy Runtime adapter entry point."""
    registry.register_lazy_module("starvla", "physicalai_starvla_policy_plugin.runtime", extensions=(".starvla",))
