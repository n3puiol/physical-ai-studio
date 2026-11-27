# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The StarVLA-protocol policy server over Runtime packages, without a model or GPU."""

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import pytest
from websockets.sync.client import connect

from physicalai_starvla_policy_plugin import serve
from physicalai_starvla_policy_plugin.openvino_package import OPENVINO_ARTIFACT
from physicalai_starvla_policy_plugin.package import IoSpec, write_package


class FakeModel:
    """Records Runtime observations; each chunk is filled with the call count."""

    def __init__(self, horizon: int = 2, width: int = 3) -> None:
        """Return chunks of ``(horizon, width)``."""
        self.observations: list[dict[str, Any]] = []
        self.resets = 0
        self.shape = (horizon, width)

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        """Record the observation.

        Returns:
            A chunk filled with the call count.
        """
        self.observations.append(observation)
        return np.full(self.shape, len(self.observations), np.float32)

    def reset(self) -> None:
        """Count resets."""
        self.resets += 1


@pytest.fixture
def native(tmp_path: Path, checkpoint: Path, backbone: Path, io: IoSpec) -> Path:
    # A native package from a run trained with state, recording an 8-step horizon.
    (checkpoint.parents[1] / "config.yaml").write_text(
        "framework:\n  name: QwenGR00T\n  action_model:\n    action_horizon: 8\n"
        "datasets:\n  vla_data:\n    include_state: true\n    data_mix: libero_goal\n",
    )
    return write_package(checkpoint, tmp_path / "native", backbone=backbone, io=io, unnorm_key="franka")


def test_frames_match_starvlas_msgpack_numpy_encoding() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    frame = serve.pack({"examples": [{"image": [image], "state": np.float32(1.5)}]})
    raw = msgpack.unpackb(frame)
    encoded = raw["examples"][0]["image"][0]
    assert encoded[b"__ndarray__"] is True
    assert encoded[b"dtype"] == "|u1"
    assert encoded[b"shape"] == [2, 2, 3]
    decoded = serve.unpack(frame)
    np.testing.assert_array_equal(decoded["examples"][0]["image"][0], image)
    assert decoded["examples"][0]["state"] == np.float32(1.5)


def test_object_arrays_are_refused_both_ways() -> None:
    with pytest.raises(ValueError, match="dtype"):
        serve.pack(np.array([object()], dtype=object))
    frame = msgpack.packb({b"__ndarray__": True, b"data": b"\x00" * 8, b"dtype": "|O", b"shape": [1]})
    with pytest.raises(ValueError, match="dtype"):
        serve.unpack(frame)


def test_metadata_describes_a_native_package(native: Path) -> None:
    _, metadata = serve.package_metadata(native)

    assert metadata["backend"] == "starvla"
    assert metadata["policy"] == "QwenGR00T"
    assert metadata["action_chunk_size"] == 8
    assert metadata["available_unnorm_keys"] == ["franka"]
    assert metadata["training_data_mix"] == "libero_goal"
    assert metadata["cameras"] == ["front", "wrist"]
    # In the package's action layout: action.gripper occupies slice 0:1, action.arm 1:3.
    assert metadata["action_keys"] == ["action.gripper", "action.arm"]
    assert metadata["state_keys"] == ["state.arm"]


def test_metadata_reads_starvlas_older_horizon_field(
    tmp_path: Path,
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
) -> None:
    """Released checkpoints record ``future_action_window_size`` (steps after the current one) instead."""
    (checkpoint.parents[1] / "config.yaml").write_text(
        "framework:\n  name: QwenOFT\n  action_model:\n    future_action_window_size: 7\n",
    )
    package = write_package(checkpoint, tmp_path / "older", backbone=backbone, io=io, unnorm_key="franka")

    assert serve.package_metadata(package)[1]["action_chunk_size"] == 8


def test_metadata_describes_an_openvino_package(tmp_path: Path, io: IoSpec) -> None:
    artifact = tmp_path / "ov" / OPENVINO_ARTIFACT
    artifact.mkdir(parents=True)
    descriptor = {
        "policy": "QwenGR00T",
        "unnorm_key": "franka",
        "io": {"cameras": io.cameras, "state": {}, "action": io.action},
        "action": {"horizon": 8},
        "image": {"size": [224, 224]},
    }
    (artifact / "package.json").write_text(json.dumps(descriptor))

    _, metadata = serve.package_metadata(tmp_path / "ov")

    assert metadata["backend"] == "starvla_openvino"
    assert metadata["training_obs_image_size"] == [224, 224]
    assert metadata["state_keys"] == []


def test_examples_map_onto_the_package_cameras_task_and_state(native: Path) -> None:
    model = FakeModel()
    policy = serve.RuntimePolicy(model, native)
    front, wrist = np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)

    result = policy.predict_action(
        examples=[
            {"image": [front, wrist], "lang": "pick", "state": np.array([[0.5, 1.0]], np.float32)},
            {"image": [front, wrist], "lang": "place", "state": np.array([0.1, 0.2], np.float32)},
        ],
        unnorm_key="franka",
        do_sample=False,
        use_ddim=True,
    )

    assert result["actions"].shape == (2, 2, 3)
    assert result["actions"].dtype == np.float32
    first, second = model.observations
    np.testing.assert_array_equal(first["images"]["wrist"], wrist)
    assert (first["task"], second["task"]) == ("pick", "place")
    np.testing.assert_array_equal(first["state"], [0.5, 1.0])


def test_requests_that_do_not_match_the_package_are_refused(native: Path) -> None:
    policy = serve.RuntimePolicy(FakeModel(), native)
    images = [np.zeros((4, 4, 3), np.uint8)] * 2
    state = np.zeros(2, np.float32)

    with pytest.raises(ValueError, match="unnorm_key 'franka'"):
        policy.predict_action([{"image": images, "lang": "x", "state": state}], unnorm_key="bridge")
    with pytest.raises(ValueError, match="2 camera"):
        policy.predict_action([{"image": images[:1], "lang": "x", "state": state}])
    with pytest.raises(ValueError, match="trained with proprioceptive state"):
        policy.predict_action([{"image": images, "lang": "x"}])


def test_a_package_without_state_ignores_it(tmp_path: Path, checkpoint: Path, backbone: Path, io: IoSpec) -> None:
    (checkpoint.parents[1] / "config.yaml").write_text(
        "framework:\n  name: QwenGR00T\n  action_model:\n    action_horizon: 8\n",
    )
    package = write_package(checkpoint, tmp_path / "stateless", backbone=backbone, io=io, unnorm_key="franka")
    model = FakeModel()

    serve.RuntimePolicy(model, package).predict_action(
        [{"image": [np.zeros((4, 4, 3), np.uint8)] * 2, "lang": "x", "state": np.zeros(2, np.float32)}],
    )

    assert "state" not in model.observations[0]


def test_messages_are_routed_like_starvlas_server(native: Path) -> None:
    model = FakeModel()
    policy = serve.RuntimePolicy(model, native)
    example = {"image": [np.zeros((4, 4, 3), np.uint8)] * 2, "lang": "x", "state": np.zeros(2, np.float32)}

    assert serve.respond(policy, {"type": "ping", "request_id": "a"})["status"] == "ok"
    assert serve.respond(policy, {"type": "reset"})["ok"] is True
    assert model.resets == 1
    # A flat message is the payload, as StarVLA's clients send it.
    flat = serve.respond(policy, {"examples": [example]})
    assert flat["type"] == "inference_result"
    assert flat["data"]["actions"].shape == (1, 2, 3)
    failed = serve.respond(policy, {"type": "infer", "payload": {"examples": [{"image": [], "lang": "x"}]}})
    assert failed["status"] == "error"
    assert "camera" in failed["error"]["message"]
    assert serve.respond(policy, {"type": "train"})["error"]["message"] == "Unsupported message type 'train'."
    assert serve.respond(policy, "not a map")["status"] == "error"


@contextlib.contextmanager
def running(policy: serve.RuntimePolicy) -> Iterator[int]:
    """Serve on a free local port in a background thread.

    Yields:
        The bound port.
    """
    loop = asyncio.new_event_loop()
    ready, ports, tasks = threading.Event(), [], []

    def run() -> None:
        def listening(port: int) -> None:
            ports.append(port)
            ready.set()

        task = loop.create_task(serve.serve(policy, "127.0.0.1", 0, listening))
        tasks.append(task)
        with contextlib.suppress(asyncio.CancelledError):
            loop.run_until_complete(task)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(10)
    try:
        yield ports[0]
    finally:
        loop.call_soon_threadsafe(tasks[0].cancel)
        thread.join(10)


def test_a_starvla_style_client_gets_metadata_then_actions(native: Path) -> None:
    policy = serve.RuntimePolicy(FakeModel(), native)
    with running(policy) as port, connect(f"ws://127.0.0.1:{port}", max_size=None, compression=None) as client:
        metadata = serve.unpack(client.recv())
        assert metadata["action_chunk_size"] == 8
        example = {"image": [np.zeros((4, 4, 3), np.uint8)] * 2, "lang": "x", "state": np.zeros(2, np.float32)}
        client.send(serve.pack({"examples": [example], "unnorm_key": None, "do_sample": False}))
        response = serve.unpack(client.recv())
    assert response["status"] == "ok"
    assert response["data"]["actions"].shape == (1, 2, 3)


def test_a_client_leaving_without_a_close_frame_is_not_an_error(native: Path, caplog: pytest.LogCaptureFixture) -> None:
    """StarVLA's LIBERO client exits without closing; StarVLA's server logs that as a closed connection."""
    policy = serve.RuntimePolicy(FakeModel(), native)
    caplog.set_level(logging.INFO)
    with running(policy) as port, connect(f"ws://127.0.0.1:{port}", max_size=None, compression=None) as client:
        serve.unpack(client.recv())
        client.socket.close()  # Drop the TCP connection without a close frame.
        deadline = time.monotonic() + 10
        while not any("closed" in record.getMessage() for record in caplog.records) and time.monotonic() < deadline:
            time.sleep(0.05)

    assert any(record.levelno == logging.INFO and "closed" in record.getMessage() for record in caplog.records)
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
