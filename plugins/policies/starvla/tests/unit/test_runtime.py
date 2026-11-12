# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Observation mappings and Runtime adapter lifecycle with a real tiny worker process."""

import sys
from pathlib import Path

import numpy as np
import pytest
from physicalai.inference import InferenceModel

from physicalai_starvla_policy_plugin import environments, runtime
from physicalai_starvla_policy_plugin.package import IoSpec


@pytest.fixture
def observation() -> dict:
    return {
        "images": {"front": np.zeros((8, 9, 3), np.uint8)},
        "observation.images.wrist": np.ones((3, 8, 9), np.float32),
        "state": np.array([0.1, 0.2], np.float32),
        "task": ["pick"],
    }


def test_observation(io: IoSpec, observation: dict) -> None:
    result = runtime.to_observation(observation, io)
    assert result["video"]["video.wrist"].shape == (8, 9, 3)
    assert result["video"]["video.wrist"].max() == 255
    np.testing.assert_allclose(result["state"]["state.arm"], [[0.1, 0.2]])
    assert result["task"] == "pick"


@pytest.mark.parametrize("field", ["state", "task", "images"])
def test_missing_observation(io: IoSpec, observation: dict, field: str) -> None:
    del observation[field]
    with pytest.raises((KeyError, ValueError)):
        runtime.to_observation(observation, io)


@pytest.mark.parametrize("image", [np.zeros((2, 2)), np.full((2, 2, 3), np.nan), np.full((2, 2, 3), 2.0)])
def test_bad_image(io: IoSpec, observation: dict, image: np.ndarray) -> None:
    observation["images"]["front"] = image
    with pytest.raises(ValueError, match=r"images|Images"):
        runtime.to_observation(observation, io)


def test_action_mapping(io: IoSpec) -> None:
    result = runtime.flatten_actions({"action.arm": np.ones((2, 2)), "action.gripper": np.full((2, 1), 9)}, io)
    assert result.shape == (1, 2, 3)
    np.testing.assert_array_equal(result[0, 0], [9, 1, 1])
    with pytest.raises(ValueError, match="horizons"):
        runtime.flatten_actions({"action.arm": np.zeros((1, 2)), "action.gripper": np.zeros((2, 1))}, io)
    with pytest.raises(ValueError, match="shape or values"):
        runtime.flatten_actions({"action.arm": np.full((2, 2), np.nan), "action.gripper": np.zeros((2, 1))}, io)


@pytest.fixture
def fake_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the actual inference server and transport, replacing only the large model.
    script = tmp_path / "worker.py"
    script.write_text(f"""
import sys
sys.path.insert(0, {str(Path(str(environments.INFER_WORKER)).parent)!r})
import numpy as np
import starvla_infer_worker as worker
class TinyPolicy:
    def __init__(self, artifact, descriptor, device):
        self.mapping = descriptor["io"]["action"]
    def predict(self, message):
        seed = message.get("seed", 0)
        return {{key: np.full((2, end-start), seed, dtype=np.float32)
                for key, (start, end) in self.mapping.items()}}
worker.Policy = TinyPolicy
worker.identity = lambda: {{"framework": "starvla", "package": "starvla", "version": "1.0.1"}}
worker.main()
""")
    monkeypatch.setattr(environments, "INFER_WORKER", script)
    monkeypatch.setattr(environments, "interpreter_path", lambda _: Path(sys.executable))


def test_runtime_manifest_load_predict_reset(package: Path, fake_worker: None, observation: dict) -> None:
    del fake_worker
    model = InferenceModel(package, device="cuda", seed=10)
    try:
        first = model(observation)["action"]
        second = model(observation)["action"]
        assert first.shape == (1, 2, 3)
        assert first.max() == 10
        assert second.max() == 11
        model.reset()
        np.testing.assert_array_equal(first, model(observation)["action"])
        model.reset()
        assert model.predict_action_chunk(observation).shape == (2, 3)
    finally:
        model.close()


def test_failed_exchange_is_not_reused(fake_worker: None) -> None:
    del fake_worker
    worker = runtime._Worker(Path(sys.executable))
    with pytest.raises(RuntimeError, match="failed"):
        worker.request({"op": "unknown"})
    with pytest.raises(RuntimeError, match="closed"):
        worker.request({"op": "hello"})
    assert worker.process.poll() is not None
    worker.close()


def test_close_kills_a_stuck_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "stuck.py"
    script.write_text("import time; time.sleep(600)")
    monkeypatch.setattr(environments, "INFER_WORKER", script)
    worker = runtime._Worker(Path(sys.executable))
    worker.close()
    assert worker.process.poll() is not None
    assert worker._requests.closed
