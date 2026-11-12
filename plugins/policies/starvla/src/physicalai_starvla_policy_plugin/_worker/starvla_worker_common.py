# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Worker-only event, cancellation, and model-loading guards (no Studio imports)."""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import threading
from importlib.metadata import version
from typing import Any


class CancelledError(Exception):
    """Raised at a training step boundary on host cancellation."""


class EventChannel:
    """Send the engine runner's tagged JSON events on its protocol descriptor."""

    def __init__(self) -> None:
        """Bind to the runner, or disable events for CLI passthrough."""
        configured = os.environ.get("PHYSICALAI_WORKER_PROTOCOL_FD")
        self._fd = int(configured) if configured else None

    def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-compatible event without writing into training logs."""
        if self._fd is None:
            return
        payload = b"\x1ephysicalai-worker " + (json.dumps(message, allow_nan=False) + "\n").encode()
        while payload:
            payload = payload[os.write(self._fd, payload) :]


def identity() -> dict[str, str]:
    """Return the installed framework identity for the host handshake."""
    return {"framework": "starvla", "package": "starvla", "version": version("starvla")}


def start_cancel_listener() -> threading.Event:
    """Listen for the runner's cancellation messages on stdin.

    Returns:
        Event set by a cancellation request.
    """
    cancelled = threading.Event()

    def listen() -> None:
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict) and message.get("type") == "cancel":
                cancelled.set()
                return

    threading.Thread(target=listen, name="starvla-cancel", daemon=True).start()
    return cancelled


def finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Filter native log values to the engine's finite scalar metric contract.

    Returns:
        Finite scalar metrics only.
    """
    return {
        key: float(value) for key, value in metrics.items() if isinstance(value, (int, float)) and math.isfinite(value)
    }


def disable_remote_code() -> None:
    """Refuse Transformers dynamic modules, including upstream's trust_remote_code defaults."""
    dynamic = importlib.import_module("transformers.dynamic_module_utils")
    original = dynamic.get_class_from_dynamic_module
    for name in (
        "configuration_auto",
        "auto_factory",
        "tokenization_auto",
        "processing_auto",
        "image_processing_auto",
        "feature_extraction_auto",
        "video_processing_auto",
    ):
        importlib.import_module(f"transformers.models.auto.{name}")

    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = "Model repository Python code is disabled by the Physical AI StarVLA plugin."
        raise RuntimeError(msg)

    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", {})
        if (
            str(namespace.get("__name__", "")).startswith("transformers")
            and namespace.get(
                "get_class_from_dynamic_module",
            )
            is original
        ):
            namespace["get_class_from_dynamic_module"] = refuse
