# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime-only StarVLA adapter using bounded, pickle-free array messages."""

from __future__ import annotations

import contextlib
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Fixed worker argv, shell disabled.
import threading
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.registry import adapter_registry
from physicalai.inference.callbacks.base import Callback

from . import environments
from ._worker import starvla_worker_protocol as protocol
from .package import IoSpec, load_descriptor

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel

_VECTOR_BATCH_RANK = 2
_IMAGE_RANK = 3
_BATCHED_IMAGE_RANK = 4


@adapter_registry.register("starvla", extensions=(".starvla",))
class StarVLARuntimeAdapter(RuntimeAdapter):
    """Run packaged QwenGR00T checkpoints in the isolated StarVLA CUDA environment."""

    def __init__(
        self,
        device: str = "auto",
        *,
        interpreter: str | Path | None = None,
        seed: int | None = None,
        **kwargs: Any,  # ruff: ignore[any-type]  # Runtime forwards backend-specific options.
    ) -> None:
        """Configure the worker; an explicit interpreter bypasses lock provisioning."""
        super().__init__(device=device, **kwargs)
        self._interpreter = interpreter
        self._seed = seed
        self._calls = 0
        self._worker: _Worker | None = None
        self._io: IoSpec | None = None

    def default_device(self) -> str:  # ruff: ignore[no-self-use]
        """Use CUDA, as required by the upstream inference implementation.

        Returns:
            CUDA device name.
        """
        return "cuda"

    def load(self, model_path: Path | str) -> None:
        """Validate the package and robot mapping before accepting predictions.

        Raises:
            RuntimeError: If the worker reports a different framework or version.
        """
        artifact = Path(model_path).expanduser().resolve(strict=True)
        descriptor = load_descriptor(artifact)
        io = IoSpec.from_dict(descriptor["io"])
        worker = _Worker(environments.interpreter_path(self._interpreter))
        try:
            hello = worker.request({"op": "hello"})
            if (
                hello.get("framework") != "starvla"
                or hello.get("package") != "starvla"
                or hello.get("version") != "1.0.1"
            ):
                msg = "Worker is not the pinned StarVLA distribution."
                raise RuntimeError(msg)  # ruff: ignore[raise-within-try]  # Failed handshakes must close the worker.
            worker.request({"op": "load", "artifact": str(artifact), "descriptor": descriptor, "device": self.device})
        except BaseException:
            worker.close()
            raise
        self.close()
        self._worker, self._io = worker, io
        self._calls = 0

    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return unnormalized actions shaped (1, horizon, action_dim).

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._worker is None or self._io is None:
            msg = "Load a StarVLA package before predicting."
            raise RuntimeError(msg)
        request = {"op": "predict", **to_observation(inputs, self._io)}
        if self._seed is not None:
            request["seed"] = self._seed + self._calls
        response = self._worker.request(request)
        action = flatten_actions(response["actions"], self._io)
        self._calls += 1
        return {"action": action}

    def reset(self) -> None:
        """Reset the deterministic prediction sequence between episodes."""
        self._calls = 0
        if self._worker is not None:
            self._worker.request({"op": "reset"})

    def close(self) -> None:
        """Stop the isolated process and release its pipes."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    @property
    def input_names(self) -> list[str]:
        """Pass Runtime's structured observation through unchanged."""
        return []

    @property
    def output_names(self) -> list[str]:
        """Expose the Runtime action chunk."""
        return ["action"]


class StarVLAResetCallback(Callback):
    """Forward Runtime episode resets to the StarVLA adapter."""

    def __init__(self) -> None:
        """Create an unbound callback."""
        self._adapter: StarVLARuntimeAdapter | None = None

    def on_load(self, model: InferenceModel) -> None:
        """Bind only to this plugin's adapter.

        Raises:
            TypeError: If Runtime loaded a different adapter.
        """
        if not isinstance(model.adapter, StarVLARuntimeAdapter):
            msg = "StarVLA reset callback requires the StarVLA Runtime adapter."
            raise TypeError(msg)
        self._adapter = model.adapter

    def on_reset(self) -> None:
        """Restart the adapter's episode state."""
        if self._adapter is not None:
            self._adapter.reset()


def to_observation(inputs: Mapping[str, Any], io: IoSpec) -> dict[str, Any]:
    """Convert explicit Runtime camera and state mappings into named raw model inputs.

    Returns:
        Named uint8 images, raw float32 states, and task text.

    Raises:
        KeyError: If a camera or state is missing.
        ValueError: If state, image, or task values are invalid.
    """
    nested = inputs.get("images", {})
    images = dict(nested) if isinstance(nested, Mapping) else {}
    for key, value in inputs.items():
        for prefix in ("images.", "observation.images."):
            if key.startswith(prefix):
                images[key.removeprefix(prefix)] = value
    video = {key: _image(np.asarray(images[camera])) for key, camera in io.cameras.items()}
    state = {}
    if io.state:
        raw = inputs.get("state", inputs.get("observation.state"))
        if raw is None:
            msg = "Observation is missing state."
            raise KeyError(msg)
        vector = np.asarray(raw, dtype=np.float32)
        if vector.ndim == _VECTOR_BATCH_RANK and vector.shape[0] == 1:
            vector = vector[0]
        width = max(end for _, end in io.state.values())
        if vector.ndim != 1 or vector.size < width or not np.isfinite(vector).all():
            msg = "Observation state must be a finite single vector covering the mapped slices."
            raise ValueError(msg)
        state = {key: vector[start:end][None].copy() for key, (start, end) in io.state.items()}
    task = inputs.get("task")
    if isinstance(task, np.ndarray):
        task = task.tolist()
    if isinstance(task, (list, tuple)) and len(task) == 1:
        task = task[0]
    if not isinstance(task, str):
        msg = "Observation requires a task instruction string."
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]  # Invalid observation content.
    return {"video": video, "state": state, "task": task}


def _image(image: np.ndarray) -> np.ndarray:
    if image.ndim == _BATCHED_IMAGE_RANK and image.shape[0] == 1:
        image = image[0]
    if image.ndim != _IMAGE_RANK:
        msg = "Camera images must be HWC or CHW (optionally with batch size one)."
        raise ValueError(msg)
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] not in {1, 3, 4} or not image.size or not np.isfinite(image).all():
        msg = "Camera images need finite pixel values and 1, 3, or 4 channels."
        raise ValueError(msg)
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.floating) or image.min() < 0 or image.max() > 1:
            msg = "Images must be uint8 [0,255] or floating point [0,1]."
            raise ValueError(msg)
        image = np.rint(image * 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return np.ascontiguousarray(image[..., :3])


def flatten_actions(actions: Mapping[str, Any], io: IoSpec) -> np.ndarray:
    """Reorder named native action groups into the explicitly mapped Runtime vector.

    Returns:
        Finite float32 action chunk of shape (1, horizon, action_dim).

    Raises:
        ValueError: If action keys, shapes, or values are invalid.
    """
    if set(actions) != set(io.action):
        msg = "Worker action keys differ from the package mapping."
        raise ValueError(msg)
    chunk = None
    for key, (start, end) in io.action.items():
        value = np.asarray(actions[key], dtype=np.float32)
        if (
            value.ndim != _VECTOR_BATCH_RANK
            or value.shape[1] != end - start
            or not value.shape[0]
            or not np.isfinite(value).all()
        ):
            msg = f"Invalid worker action shape or values for {key!r}."
            raise ValueError(msg)
        if chunk is None:
            chunk = np.empty((1, value.shape[0], io.action_dim), dtype=np.float32)
        if value.shape[0] != chunk.shape[1]:
            msg = "Worker action groups have different horizons."
            raise ValueError(msg)
        chunk[0, :, start:end] = value
    return chunk


class _Worker:
    """One serialized request stream; a failed exchange permanently closes it."""

    def __init__(self, python: Path) -> None:
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        try:
            self.process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]  # No shell expansion.
                [str(python), str(environments.INFER_WORKER)],
                stdin=subprocess.DEVNULL,
                shell=False,
                env={
                    **os.environ,
                    "PHYSICALAI_STARVLA_REQUEST_FD": str(request_read),
                    "PHYSICALAI_STARVLA_RESPONSE_FD": str(response_write),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                pass_fds=(request_read, response_write),
            )
        except BaseException:
            os.close(request_write)
            os.close(response_read)
            raise
        finally:
            os.close(request_read)
            os.close(response_write)
        self._requests = os.fdopen(request_write, "wb")
        self._responses = os.fdopen(response_read, "rb")
        self._lock = threading.Lock()
        self._closed = False
        self._finalizer = weakref.finalize(self, _stop, self.process, self._requests, self._responses)

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Exchange one message; never reuse a stream after an incomplete request.

        Returns:
            Successful worker response.

        Raises:
            RuntimeError: If the worker is closed or reports failure.
        """
        with self._lock:
            if self._closed:
                msg = "StarVLA worker is closed."
                raise RuntimeError(msg)
            try:
                protocol.write_frame(self._requests, message)
                response = protocol.read_frame(self._responses)
                if response is None or response.get("ok") is not True:
                    msg = f"StarVLA worker failed: {response}"
                    raise RuntimeError(msg)  # ruff: ignore[raise-within-try]  # A failed exchange invalidates the stream.
            except BaseException:
                self.close()
                raise
        return response

    def close(self) -> None:
        """Stop the process before closing pipes, including after interrupted reads."""
        self._closed = True
        self._finalizer()


def _stop(process: subprocess.Popen[bytes], requests: BinaryIO, responses: BinaryIO) -> None:
    if process.poll() is None:
        process.kill()
    process.wait()
    for stream in (requests, responses):
        with contextlib.suppress(OSError):
            stream.close()
