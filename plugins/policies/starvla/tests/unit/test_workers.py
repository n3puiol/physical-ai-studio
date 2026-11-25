# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Worker callbacks, cancellation, and model normalization without framework imports."""

import importlib
import io
import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Fixed interpreter and worker path.
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from physicalai_starvla_policy_plugin import environments


@pytest.fixture
def train_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.syspath_prepend(str(Path(str(environments.TRAIN_WORKER)).parent))
    worker = importlib.import_module("starvla_train_worker")
    events = []
    cancelled = threading.Event()
    monkeypatch.setattr(worker, "EventChannel", lambda: SimpleNamespace(send=events.append, enabled=False))
    monkeypatch.setattr(worker, "identity", lambda: {"framework": "starvla", "package": "starvla", "version": "1.0.1"})
    monkeypatch.setattr(worker, "disable_remote_code", lambda: None)
    monkeypatch.setattr(worker, "start_cancel_listener", lambda: cancelled)
    monkeypatch.setenv("PHYSICALAI_STARVLA_PROVENANCE", json.dumps(environments.QWENGR00T.provenance()))
    monkeypatch.setenv("PHYSICALAI_STARVLA_FRAMEWORK", "QwenGR00T")
    backbone = tmp_path / "Qwen3-VL"
    backbone.mkdir()
    (backbone / "config.json").write_text('{"model_type": "qwen3_vl"}')
    cfg = SimpleNamespace(
        framework=SimpleNamespace(name="QwenGR00T", qwenvl=SimpleNamespace(base_vlm=str(backbone))),
        trainer=SimpleNamespace(max_train_steps=2),
        datasets=SimpleNamespace(vla_data=SimpleNamespace(data_root_dir="data", data_mix="libero_goal")),
        run_root_dir=str(tmp_path),
        run_id="run",
    )

    class Trainer:
        completed_steps = 1
        config = cfg
        _train_step = Mock(return_value={"loss": 0.1})
        _log_metrics = Mock()

    def native_main(config: SimpleNamespace) -> None:
        output = Path(config.run_root_dir) / config.run_id
        (output / "final_model").mkdir(parents=True)
        (output / "final_model" / "model.safetensors").write_bytes(b"fake")
        (output / "config.full.yaml").write_text("full config")
        (output / "dataset_statistics.json").write_text('{"franka": {}}')
        trainer = Trainer()
        trainer._train_step([])
        trainer._log_metrics({"loss": 0.1, "nan": float("nan")})

    native = SimpleNamespace(VLATrainer=Trainer, main=native_main, _get_state_dict=lambda *_: {})
    for module, value in {
        "omegaconf": SimpleNamespace(
            OmegaConf=SimpleNamespace(load=lambda _: cfg, from_dotlist=Mock(return_value={}), merge=lambda *_: cfg),
        ),
        "starVLA": SimpleNamespace(),
        "starVLA.model": SimpleNamespace(),
        "starVLA.model.framework": SimpleNamespace(),
        "starVLA.model.framework.share_tools": SimpleNamespace(apply_config_compat=lambda c: c),
        "starVLA.training": SimpleNamespace(train_starvla=native),
        "starVLA.training.trainer_utils": SimpleNamespace(),
    }.items():
        monkeypatch.setitem(sys.modules, module, value)
    return SimpleNamespace(worker=worker, events=events, cancelled=cancelled, cfg=cfg, trainer=Trainer)


def test_training_events_and_record(train_worker: SimpleNamespace) -> None:
    original_step = train_worker.trainer._train_step
    output, checkpoint = train_worker.worker.train(["--config_yaml", "unused.yaml"])
    assert checkpoint.is_file()
    assert train_worker.events[0]["type"] == "hello"
    assert train_worker.events[1] == {"type": "progress", "step": 1, "total": 2, "metrics": {"loss": 0.1}}
    assert train_worker.trainer._train_step is original_step
    assert train_worker.cfg.trainer.save_format == "safetensors"
    record = json.loads((output / "physicalai_run.json").read_text())
    assert record["environment"] == environments.QWENGR00T.provenance()
    assert record["unnorm_key"] == "franka"
    assert (output / "config.yaml").read_text() == "full config"


def test_dotlist_overrides_are_not_silently_discarded(train_worker: SimpleNamespace) -> None:
    train_worker.worker.train([
        "--config_yaml",
        "unused.yaml",
        "run_id=studio",
        "--trainer.max_train_steps=20",
        'run_root_dir="/path with spaces"',
    ])
    sys.modules["omegaconf"].OmegaConf.from_dotlist.assert_called_once_with([
        "run_id=studio",
        "trainer.max_train_steps=20",
        'run_root_dir="/path with spaces"',
    ])


def test_invalid_override_fails_instead_of_being_ignored(train_worker: SimpleNamespace) -> None:
    with pytest.raises(SystemExit) as exc:
        train_worker.worker.train(["--config_yaml", "unused.yaml", "--run_id", "studio"])
    assert exc.value.code == 2


def test_cancel_restores_native_hooks(train_worker: SimpleNamespace) -> None:
    original_step = train_worker.trainer._train_step
    train_worker.cancelled.set()
    with pytest.raises(train_worker.worker.CancelledError):
        train_worker.worker.train(["--config_yaml", "unused.yaml"])
    assert train_worker.trainer._train_step is original_step


def test_config_for_another_framework_is_rejected(
    train_worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host chose the environment for one framework; the worker refuses to train another."""
    train_worker.cfg.framework.name = "QwenPI"
    with pytest.raises(ValueError, match="differs from 'QwenGR00T'"):
        train_worker.worker.train(["--config_yaml", "unused.yaml"])
    monkeypatch.delenv("PHYSICALAI_STARVLA_FRAMEWORK")
    with pytest.raises(ValueError, match="through the StarVLA engine"):
        train_worker.worker.train(["--config_yaml", "unused.yaml"])


def test_qwenpi_trains_when_the_host_chose_it(train_worker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    train_worker.cfg.framework.name = "QwenPI"
    monkeypatch.setenv("PHYSICALAI_STARVLA_FRAMEWORK", "QwenPI")

    _, checkpoint = train_worker.worker.train(["--config_yaml", "unused.yaml"])

    assert checkpoint.is_file()
    assert [event["type"] for event in train_worker.events] == ["hello", "progress"]


def test_inference_refuses_a_checkpoint_of_another_framework(
    train_worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del train_worker
    worker = importlib.import_module("starvla_infer_worker")
    monkeypatch.setattr(worker, "disable_remote_code", lambda: None)
    config = SimpleNamespace(framework=SimpleNamespace(name="QwenPI"))
    for module, value in {
        "omegaconf": SimpleNamespace(OmegaConf=SimpleNamespace(load=lambda _: config)),
        "deployment": SimpleNamespace(),
        "deployment.model_server": SimpleNamespace(),
        "deployment.model_server.policy_wrapper": SimpleNamespace(PolicyServerWrapper=Mock()),
    }.items():
        monkeypatch.setitem(sys.modules, module, value)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "model.safetensors").write_bytes(b"fake")

    with pytest.raises(ValueError, match="different StarVLA frameworks"):
        worker.Policy(tmp_path, {"checkpoint": "model/model.safetensors", "policy": "QwenGR00T"}, "cuda")


def test_host_acknowledgement_reads_ack_cancel_and_closed(
    train_worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = importlib.import_module("starvla_worker_common")

    def answer(lines: bytes) -> str:
        read_fd, write_fd = os.pipe()
        os.write(write_fd, lines)
        os.close(write_fd)
        with io.TextIOWrapper(os.fdopen(read_fd, "rb"), encoding="utf-8") as stdin:
            monkeypatch.setattr(common.sys, "stdin", stdin)
            return common.wait_for_host_acknowledgement()

    del train_worker
    assert answer(b'not json\n{"type":"other"}\n{"type":"ack"}\n') == "ack"
    assert answer(b'{"type":"cancel"}\n') == "cancel"
    assert answer(b"") == "closed"


@pytest.mark.parametrize(
    ("answer", "error"),
    [("closed", "WorkerHandshakeError"), ("cancel", "CancelledError")],
)
def test_training_waits_for_the_host_to_accept_the_handshake(
    train_worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    error: str,
) -> None:
    """A rejected or cancelled worker stops after its hello, before StarVLA is imported."""
    worker = train_worker.worker
    monkeypatch.setattr(worker, "EventChannel", lambda: SimpleNamespace(send=train_worker.events.append, enabled=True))
    monkeypatch.setattr(worker, "wait_for_host_acknowledgement", lambda: answer)
    monkeypatch.setattr(worker, "start_cancel_listener", Mock(side_effect=AssertionError("training started")))

    with pytest.raises(getattr(worker, error)):
        worker.train(["--config_yaml", "unused.yaml"])

    assert [event["type"] for event in train_worker.events] == ["hello"]


def test_tied_weights_keep_both_keys_in_safetensors(tmp_path: Path, train_worker: SimpleNamespace) -> None:
    tensor = torch.arange(6).reshape(2, 3)
    state = {"embedding": tensor, "head": tensor, "transposed": tensor.T}
    result = train_worker.worker._saveable_state_dict(state)
    path = tmp_path / "model.safetensors"
    save_file(result, path)
    restored = load_file(path)
    assert set(restored) == set(state)
    for key, value in state.items():
        torch.testing.assert_close(restored[key], value)
    assert state["embedding"] is state["head"]


def test_inference_applies_state_transforms_once(
    train_worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del train_worker
    worker = importlib.import_module("starvla_infer_worker")

    class ToTensor:
        apply_to = ("state.a", "state.b")

        def __call__(self, state: dict) -> dict:
            assert all(isinstance(value, np.ndarray) for value in state.values())
            return {key: torch.as_tensor(value) for key, value in state.items()}

    class Normalize:
        apply_to = ("state.a", "state.b")

        def __call__(self, state: dict) -> dict:
            return {key: value * 2 for key, value in state.items()}

    monkeypatch.setitem(
        sys.modules,
        "starVLA.dataloader.gr00t_lerobot.transform.state_action",
        SimpleNamespace(
            StateActionToTensor=ToTensor,
            StateActionTransform=Normalize,
        ),
    )
    policy = worker.Policy.__new__(worker.Policy)
    policy.uses_state = True
    policy.io = {"cameras": {"video.main": "front"}, "state": {"state.a": [0, 1], "state.b": [1, 2]}}
    policy.processor = SimpleNamespace(
        transform=SimpleNamespace(transforms=[ToTensor(), Normalize()]),
        state_keys=["state.b", "state.a"],
        action_keys=["action.arm"],
        action_key_dims={"action.arm": 2},
    )
    predict = Mock(return_value={"actions": np.ones((1, 3, 2), np.float32)})
    policy.wrapper = SimpleNamespace(predict_action=predict)
    result = policy.predict({
        "video": {"video.main": np.zeros((8, 9, 3), np.uint8)},
        "state": {"state.a": np.array([[1]], np.float32), "state.b": np.array([[2]], np.float32)},
        "task": "pick",
    })
    np.testing.assert_array_equal(predict.call_args.kwargs["examples"][0]["state"], [[4, 2]])
    assert result["action.arm"].shape == (3, 2)


def test_inference_leaves_out_state_a_checkpoint_was_not_trained_on(
    train_worker: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older packages still map state; a checkpoint trained without include_state never receives it."""
    del train_worker
    worker = importlib.import_module("starvla_infer_worker")
    monkeypatch.setitem(
        sys.modules,
        "starVLA.dataloader.gr00t_lerobot.transform.state_action",
        SimpleNamespace(StateActionToTensor=object, StateActionTransform=object),
    )
    policy = worker.Policy.__new__(worker.Policy)
    policy.uses_state = False
    policy.io = {"cameras": {"video.main": "front"}, "state": {"state.a": [0, 1]}}
    policy.processor = SimpleNamespace(action_keys=["action.arm"], action_key_dims={"action.arm": 2})
    predict = Mock(return_value={"actions": np.ones((1, 3, 2), np.float32)})
    policy.wrapper = SimpleNamespace(predict_action=predict)

    policy.predict({
        "video": {"video.main": np.zeros((8, 9, 3), np.uint8)},
        "state": {"state.a": np.array([[1]], np.float32)},
        "task": "pick",
    })

    assert "state" not in predict.call_args.kwargs["examples"][0]


def test_mapping_check_follows_the_state_the_checkpoint_was_trained_on(train_worker: SimpleNamespace) -> None:
    del train_worker
    worker = importlib.import_module("starvla_infer_worker")
    processor = SimpleNamespace(action_key_dims={"action.arm": 2}, state_key_dims={"state.a": 1})
    io = {"action": {"action.arm": [0, 2]}, "state": {"state.a": [0, 1]}}

    worker.check_mapping(io, processor, uses_state=True)
    # Without state in training, a stale or empty state mapping is ignored.
    worker.check_mapping({**io, "state": {"state.other": [0, 5]}}, processor, uses_state=False)
    worker.check_mapping({**io, "state": {}}, processor, uses_state=False)
    with pytest.raises(ValueError, match="trained with proprioceptive state"):
        worker.check_mapping({**io, "state": {}}, processor, uses_state=True)
    with pytest.raises(ValueError, match="state keys or widths"):
        worker.check_mapping({**io, "state": {"state.a": [0, 2]}}, processor, uses_state=True)
    with pytest.raises(ValueError, match="action keys or widths"):
        worker.check_mapping({**io, "action": {"action.arm": [0, 3]}}, processor, uses_state=False)


@pytest.mark.parametrize(
    "worker",
    [environments.TRAIN_WORKER, environments.INFER_WORKER, environments.OPENVINO_EXPORT_WORKER],
    ids=lambda path: Path(str(path)).stem,
)
def test_worker_finds_its_sibling_modules_without_its_folder_on_sys_path(worker: object, tmp_path: Path) -> None:
    """Isolated launches keep the worker's folder off ``sys.path``; its own imports must still resolve."""
    script = str(Path(str(worker)))
    load = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='worker_import_check')"
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # No untrusted arguments.
        [sys.executable, "-I", "-c", load, script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
