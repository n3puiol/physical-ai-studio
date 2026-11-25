# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serve a StarVLA Runtime package to StarVLA's own benchmark clients.

StarVLA's evaluation clients (``examples/simBenchmarks/*/eval_files``) talk to a websocket policy
server. Frames are msgpack, with numpy arrays encoded as ``{__ndarray__, data, dtype, shape}``. The
server first sends handshake metadata; the client then sends
``{"type": "infer", "payload": {"examples": [...], "unnorm_key": ...}}`` and reads unnormalized
actions from ``{"status": "ok", "data": {"actions": (batch, horizon, action_dim)}}``.

This module answers that protocol from a Runtime package, native or OpenVINO, so a model trained
and exported with this plugin runs StarVLA's benchmarks unchanged. Each example's ``image`` list is
mapped onto the package's cameras in order, ``lang`` is the task, and ``state`` is used only by a
package that maps state. Actions come back in the package's action layout.

Usage::

    python -m physicalai_starvla_policy_plugin.serve exports/starvla --device cuda --port 10093
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .openvino_package import OPENVINO_ARTIFACT
from .package import ARTIFACT, IoSpec, framework_from_config

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

_ARRAY_KINDS = frozenset("biuf")
_BATCH_RANK = 2


# msgpack with numpy arrays, compatible with StarVLA's deployment/model_server/tools/msgpack_numpy.py.
def _encode(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in _ARRAY_KINDS:
            msg = f"Unsupported array dtype {value.dtype}."
            raise ValueError(msg)
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _decode(value: dict[Any, Any]) -> object:
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in _ARRAY_KINDS:  # Never object, void or string arrays from the network.
            msg = f"Unsupported array dtype {dtype}."
            raise ValueError(msg)
        if b"__npgeneric__" in value:
            return dtype.type(value[b"data"])
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
    return value


def pack(value: object) -> bytes:
    """Encode a message the way StarVLA's clients and servers do.

    Returns:
        The msgpack frame.
    """
    import msgpack  # ruff: ignore[import-outside-top-level]  # Only with the serve extra.

    return msgpack.packb(value, default=_encode)


def unpack(frame: bytes) -> object:
    """Decode a StarVLA msgpack frame, refusing object, void and string arrays.

    Returns:
        The decoded message.
    """
    import msgpack  # ruff: ignore[import-outside-top-level]  # Only with the serve extra.

    return msgpack.unpackb(frame, object_hook=_decode)


class RuntimePolicy:
    """Answer StarVLA ``predict_action`` requests from a Runtime package."""

    def __init__(self, model: Any, package: Path) -> None:  # ruff: ignore[any-type]  # InferenceModel.
        """Read the package's robot mapping and the metadata StarVLA's clients expect."""
        self.model = model
        self.descriptor, self.metadata = package_metadata(package)
        self.io = IoSpec.from_dict(self.descriptor["io"])

    def predict_action(
        self,
        examples: Sequence[Mapping[str, Any]],
        unnorm_key: str | None = None,
        **_options: object,
    ) -> dict[str, np.ndarray]:
        """Predict one unnormalized action chunk per example.

        Sampling options StarVLA's clients send (``do_sample``, ``use_ddim``, ...) are ignored, as
        StarVLA's own frameworks ignore them.

        Returns:
            ``{"actions": float32 (batch, horizon, action_dim)}``.

        Raises:
            ValueError: If the normalization key or an example does not match the package.
        """
        if unnorm_key is not None and unnorm_key != self.descriptor["unnorm_key"]:
            msg = f"This package was exported for unnorm_key {self.descriptor['unnorm_key']!r}, not {unnorm_key!r}."
            raise ValueError(msg)
        if not examples:
            msg = "predict_action needs at least one example."
            raise ValueError(msg)
        chunks = [np.asarray(self.model.predict_action_chunk(self._observation(example))) for example in examples]
        return {"actions": np.stack(chunks).astype(np.float32)}

    def _observation(self, example: Mapping[str, Any]) -> dict[str, Any]:
        images = list(example.get("image") or [])
        cameras = list(self.io.cameras.values())
        if len(images) != len(cameras):
            msg = f"The package maps {len(cameras)} camera(s), in order {cameras}; the example has {len(images)}."
            raise ValueError(msg)
        observation: dict[str, Any] = {
            "images": {camera: np.asarray(image) for camera, image in zip(cameras, images, strict=True)},
            "task": example.get("lang"),
        }
        if self.io.state:
            if example.get("state") is None:
                msg = "This package was trained with proprioceptive state; the example has none."
                raise ValueError(msg)
            state = np.asarray(example["state"], dtype=np.float32)
            observation["state"] = state[0] if state.ndim == _BATCH_RANK and state.shape[0] == 1 else state
        return observation

    def reset(self) -> None:
        """Restart the package's seeded noise sequence."""
        self.model.reset()


def package_metadata(package: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a native or OpenVINO package's descriptor and StarVLA's handshake metadata for it.

    Returns:
        The package descriptor and the metadata sent at the websocket handshake.

    Raises:
        ValueError: If the folder is not a StarVLA package or its action horizon is unknown.
    """
    package = Path(package)
    native, openvino = package / ARTIFACT / "package.json", package / OPENVINO_ARTIFACT / "package.json"
    if openvino.is_file():
        descriptor = json.loads(openvino.read_text())
        horizon = descriptor["action"]["horizon"]
        image_size = descriptor["image"]["size"]
        data_mix = None
        backend = "starvla_openvino"
    elif native.is_file():
        import yaml  # ruff: ignore[import-outside-top-level]  # Native packages only.

        descriptor = json.loads(native.read_text())
        config = yaml.safe_load((package / ARTIFACT / "model" / "config.yaml").read_text(encoding="utf-8"))
        action_model = config.get("framework", {}).get("action_model") or {}
        horizon = action_model.get("action_horizon")
        if horizon is None and type(action_model.get("future_action_window_size")) is int:
            # StarVLA's older field counts the steps after the current one, as its own server does.
            horizon = action_model["future_action_window_size"] + 1
        vla_data = config.get("datasets", {}).get("vla_data") or {}
        image_size, data_mix = vla_data.get("obs_image_size"), vla_data.get("data_mix")
        backend = "starvla"
    else:
        msg = f"{package} is not a StarVLA Runtime package."
        raise ValueError(msg)
    if type(horizon) is not int:
        msg = "The package does not record its action horizon."
        raise ValueError(msg)
    io = descriptor["io"]

    def by_start(mapping: Mapping[str, Sequence[int]]) -> list[str]:
        return sorted(mapping, key=lambda key: mapping[key][0])

    metadata = {
        "env": "physicalai_starvla_runtime_server",
        "backend": backend,
        "policy": descriptor.get("policy") or framework_from_config(package / ARTIFACT / "model" / "config.yaml"),
        "ckpt_path": str(package),
        "action_chunk_size": horizon,
        "available_unnorm_keys": [descriptor["unnorm_key"]],
        "default_unnorm_key": descriptor["unnorm_key"],
        "training_data_mix": data_mix,
        "training_obs_image_size": image_size,
        "cameras": list(io["cameras"].values()),
        "action_keys": by_start(io["action"]),
        "state_keys": by_start(io["state"]),
        "eval_image_contract": (
            "Send one image per package camera, in the order of `cameras`; actions come back in the "
            "order of `action_keys`. State is used only if `state_keys` is not empty."
        ),
    }
    return descriptor, metadata


def respond(policy: RuntimePolicy, message: object) -> dict[str, Any]:
    """Answer one StarVLA request; errors are returned to the client, as StarVLA's server does.

    Returns:
        The response message.
    """
    if not isinstance(message, dict):
        return _error("unknown", "default", "Messages must be maps.")
    request_id = message.get("request_id", "default")
    kind = message.get("type", "infer")
    payload = message.get("payload", message)
    if kind in {"ping", "reset"}:
        if kind == "reset":
            policy.reset()
        return {"status": "ok", "ok": True, "type": kind, "request_id": request_id}
    if kind not in {"infer", "predict_action"}:
        return _error("unknown", request_id, f"Unsupported message type {kind!r}.")
    if not isinstance(payload, dict):
        return _error("inference_result", request_id, "Payload must be a map.")
    options = {key: value for key, value in payload.items() if key not in {"type", "request_id"}}
    try:
        data = policy.predict_action(**options)
    except Exception as error:  # The client gets the error, as from StarVLA's server.
        logger.exception("Inference failed (request %s)", request_id)
        return _error("inference_result", request_id, f"{type(error).__name__}: {error}")
    return {"status": "ok", "ok": True, "type": "inference_result", "request_id": request_id, "data": data}


def _error(kind: str, request_id: object, message: str) -> dict[str, Any]:
    return {"status": "error", "ok": False, "type": kind, "request_id": request_id, "error": {"message": message}}


async def serve(policy: RuntimePolicy, host: str, port: int, ready: Callable[[int], None] | None = None) -> None:
    """Serve StarVLA clients until cancelled; one request is answered at a time.

    ``ready`` is called with the bound port once the server listens.
    """
    from websockets.asyncio.server import serve as websocket_serve  # ruff: ignore[import-outside-top-level]

    async def handle(connection: Any) -> None:  # ruff: ignore[any-type]  # websockets connection.
        await connection.send(pack(policy.metadata))
        async for frame in connection:
            response = respond(policy, unpack(frame) if isinstance(frame, bytes) else None)
            await connection.send(pack(response))

    async with websocket_serve(handle, host, port, compression=None, max_size=None) as server:
        bound = next(iter(server.sockets)).getsockname()[1]
        logger.info("Serving %s to StarVLA clients on %s:%s", policy.metadata["ckpt_path"], host, bound)
        if ready is not None:
            ready(bound)
        await server.serve_forever()


def main(argv: Sequence[str] | None = None) -> None:
    """Load a package with Runtime and serve it to StarVLA's benchmark clients."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("package", type=Path, help="Runtime package exported by this plugin")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to listen on (no authentication)")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="auto", help="Runtime device, e.g. cuda, CPU or GPU")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the package's noise sequence")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from physicalai.inference import InferenceModel  # ruff: ignore[import-outside-top-level]

    with InferenceModel(args.package, device=args.device, seed=args.seed) as model:
        policy = RuntimePolicy(model, args.package)
        asyncio.run(serve(policy, args.host, args.port, functools.partial(print, "READY", flush=True)))


if __name__ == "__main__":
    main()
