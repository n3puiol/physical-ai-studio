# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Worker-only event, cancellation, and model-loading guards (no Studio imports)."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import select
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

_ACK_TIMEOUT_SECONDS = 60.0


class CancelledError(Exception):
    """Raised at a training step boundary on host cancellation."""


class WorkerHandshakeError(RuntimeError):
    """Raised when the host does not accept the worker's version handshake."""


class EventChannel:
    """Send the engine runner's tagged JSON events on its protocol descriptor."""

    def __init__(self) -> None:
        """Bind to the runner, or disable events for CLI passthrough."""
        configured = os.environ.get("PHYSICALAI_WORKER_PROTOCOL_FD")
        self._fd = int(configured) if configured else None

    @property
    def enabled(self) -> bool:
        """Whether an engine runner is listening, rather than a native CLI passthrough."""
        return self._fd is not None

    def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-compatible event without writing into training logs."""
        if self._fd is None:
            return
        payload = b"\x1ephysicalai-worker " + (json.dumps(message, allow_nan=False) + "\n").encode()
        while payload:
            payload = payload[os.write(self._fd, payload) :]


def trained_with_state(vla_data: Mapping[str, object]) -> bool:
    """Whether StarVLA's data loader packs proprioceptive state into training samples.

    Mirrors ``starVLA.dataloader.gr00t_lerobot.datasets``: state is packed unless
    ``include_state`` is ``False`` or ``"False"``, and the setting defaults to ``False``.
    ``vla_data`` is the training config's ``datasets.vla_data`` (a dict or an OmegaConf node).

    Returns:
        True when the checkpoint was trained with proprioceptive state.
    """
    return vla_data.get("include_state", False) not in ("False", False)  # ruff: ignore[literal-membership]  # Same test as upstream, any value.


def identity() -> dict[str, str]:
    """Return the installed framework identity for the host handshake."""
    return {"framework": "starvla", "package": "starvla", "version": version("starvla")}


def wait_for_host_acknowledgement() -> str:
    """Block until the host accepts the handshake, so a rejected worker never starts training.

    Returns:
        ``"ack"`` when accepted, ``"cancel"`` when the host cancelled, or ``"closed"`` when the
        host closed standard input or did not answer in time.
    """
    deadline = time.monotonic() + _ACK_TIMEOUT_SECONDS
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            readable, _, _ = select.select([sys.stdin], [], [], min(remaining, 0.5))
        except (OSError, ValueError):
            return "closed"
        if not readable:
            continue
        line = sys.stdin.buffer.readline()
        if not line:
            return "closed"
        try:
            control = json.loads(line)
        except ValueError:
            continue
        if isinstance(control, dict) and control.get("type") in {"ack", "cancel"}:
            return str(control["type"])
    return "closed"


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


# Hugging Face "remote code" the plugin has reviewed and ships in remote_code/, by module file: the
# vendored package and the SHA-256 of the upstream file at its pinned revision (remote_code/README.md).
REVIEWED_REMOTE_CODE = {
    "processing_action_tokenizer": ("fast", "6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20"),
}
# Files of a FAST action tokenizer folder (physical-intelligence/fast) that QwenFast needs at run time.
FAST_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "processor_config.json")
# Backbones the plugin trains, by config model_type: the substring of the backbone path that makes
# StarVLA load that VLM (its get_vlm_model), which packages also use as the backbone folder's name.
# Qwen3.5 needs transformers 5, so it trains in its own worker environment.
BACKBONES = {"qwen3_vl": "Qwen3-VL", "qwen3_5": "Qwen3.5"}
DEFAULT_BACKBONE = "qwen3_vl"
# The backbone type the host chose this worker environment for; the training worker refuses others.
BACKBONE_VARIABLE = "PHYSICALAI_STARVLA_BACKBONE"


def dispatched_backbone(path: str) -> str | None:
    """Return the backbone type StarVLA's get_vlm_model loads for a ``base_vlm`` path.

    Upstream picks the VLM by substrings of the path, in a fixed order; this follows that order.

    Returns:
        A key of ``BACKBONES``, or ``None`` when StarVLA would load another VLM.
    """
    if "Qwen2.5-VL" in path or "nora" in path.lower():  # Upstream checks these first.
        return None
    return next((model_type for model_type, name in BACKBONES.items() if name in path), None)


def use_fast_tokenizer(folder: Path) -> None:
    """Point QwenFast's FAST action tokenizer at ``folder``.

    StarVLA builds it from ``playground/Pretrained_models/fast``, relative to the working directory,
    with no config key; the plugin replaces that default.

    Raises:
        ValueError: If the folder lacks the FAST tokenizer files.
    """
    missing = [name for name in FAST_TOKENIZER_FILES if not (folder / name).is_file()]
    if not folder.is_absolute() or missing:
        msg = f"QwenFast needs an absolute FAST tokenizer folder (physical-intelligence/fast); missing {missing}."
        raise ValueError(msg)
    fast = importlib.import_module(
        "starVLA.model.modules.action_model.fast_ActionHeader",
    )  # nosemgrep  # Fixed module name.
    fast.Fast_Action_Tokenizer.__init__.__defaults__ = (str(folder),)


def load_reviewed_class(
    class_reference: str,
    pretrained_model_name_or_path: object = None,
    *_args: object,
    **_kwargs: object,
) -> type:
    """Answer a Transformers request for a model's own Python class from the plugin's reviewed copy.

    Returns:
        The requested class from the vendored module.

    Raises:
        RuntimeError: If the code is not among the reviewed files, or the model folder's own copy
            differs from the reviewed revision.
    """
    module_name, _, class_name = class_reference.rpartition(".")
    module_name = module_name.rpartition("--")[2]
    if module_name not in REVIEWED_REMOTE_CODE:
        msg = (
            f"Model repository Python code ({module_name}.py) has not been reviewed for the Physical AI StarVLA plugin."
        )
        raise RuntimeError(msg)
    package, digest = REVIEWED_REMOTE_CODE[module_name]
    if pretrained_model_name_or_path is not None:
        local = Path(str(pretrained_model_name_or_path)) / f"{module_name}.py"
        if local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            msg = (
                f"{local} differs from the reviewed {module_name}.py; use the pinned revision (remote_code/README.md)."
            )
            raise RuntimeError(msg)
    module = importlib.import_module(f"remote_code.{package}.{module_name}")  # nosemgrep  # Names from the table above.
    return getattr(module, class_name)


def restrict_remote_code() -> None:
    """Serve a model's own Python code only from the plugin's reviewed copies; refuse everything else.

    Transformers loads ``trust_remote_code`` classes through ``get_class_from_dynamic_module``. Every
    reference to it is replaced with ``load_reviewed_class``, including upstream's
    ``trust_remote_code=True`` defaults, so no Python file from a model folder or the Hub runs.
    """
    dynamic = importlib.import_module("transformers.dynamic_module_utils")  # nosemgrep  # Fixed module name.
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
        importlib.import_module(f"transformers.models.auto.{name}")  # nosemgrep  # Fixed Transformers module names.
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", {})
        if (
            str(namespace.get("__name__", "")).startswith("transformers")
            and namespace.get("get_class_from_dynamic_module") is original
        ):
            namespace["get_class_from_dynamic_module"] = load_reviewed_class
