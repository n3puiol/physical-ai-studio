# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine discovery and native training/export routing without provisioning CUDA."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from physicalai.engine import Engine, Run

from physicalai_starvla_policy_plugin import engine, environments
from physicalai_starvla_policy_plugin.package import ARTIFACT, RUN_INFO, IoSpec, load_descriptor


def test_entry_point_and_pins() -> None:
    assert Engine.get("starvla") is engine.StarVLAEngine
    assert engine.StarVLAEngine.policies() == ["QwenGR00T"]
    assert engine.WORKER_ENV.pinned().base_version == "1.0.1"
    assert engine.WORKER_ENV.accelerators == ("cu128",)
    assert len(environments.provenance()["lock_sha256"]) == 64
    assert len(environments.REVISION) == 40


def test_headless_settings_are_overridden_by_studio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("{}")
    runner = Mock(return_value=Run("starvla", tmp_path, None))
    monkeypatch.setattr(engine, "run_worker", runner)
    result = engine.StarVLAEngine.run(
        {"config_yaml": str(config), "overrides": ["run_id=ignored", "trainer.max_train_steps=20"]},
        dataset=tmp_path,
        output_dir=tmp_path / "studio",
        device="cuda:2",
    )
    assert result.framework == "starvla"
    args, kwargs = runner.call_args
    assert args[1][-1] == "run_id=studio"
    assert "trainer.max_train_steps=20" in args[1]
    assert kwargs["extra_env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert kwargs["accelerator"] == "cu128"


@pytest.mark.parametrize(
    "settings",
    [{}, {"config_yaml": "x", "unknown": True}, {"config_yaml": "x", "overrides": "x=y"}],
)
def test_bad_settings(tmp_path: Path, settings: dict) -> None:
    with pytest.raises(ValueError, match=r"settings|overrides"):
        engine.StarVLAEngine.run(settings, dataset=tmp_path, output_dir=tmp_path / "out", device="cuda")


def test_unsupported_device(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CUDA"):
        engine.StarVLAEngine.run(
            {"config_yaml": "unused"},
            dataset=tmp_path,
            output_dir=tmp_path / "out",
            device="cpu",
        )


def test_export_from_run(checkpoint: Path, backbone: Path, io: IoSpec, tmp_path: Path) -> None:
    root = checkpoint.parents[1]
    (root / "io.json").write_text(json.dumps({"cameras": io.cameras, "state": io.state, "action": io.action}))
    (root / RUN_INFO).write_text(
        json.dumps({
            "environment": environments.provenance(),
            "backbone_path": str(backbone),
            "unnorm_key": "franka",
        }),
    )
    exported = engine.StarVLAEngine.export(Run("starvla", root, checkpoint), tmp_path / "out")
    assert load_descriptor(exported / ARTIFACT)["unnorm_key"] == "franka"
    with pytest.raises(ValueError, match="native"):
        engine.StarVLAEngine.export_native(
            checkpoint,
            tmp_path / "ov",
            backbone_path=backbone,
            io=io,
            unnorm_key="franka",
            backend="openvino",
        )


def test_cli_inherits_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environments, "interpreter_path", lambda _: Path("/worker/python"))
    call = Mock(return_value=7)
    monkeypatch.setattr(engine.subprocess, "call", call)
    assert engine.StarVLAEngine.cli(["--help"]) == 7
    assert call.call_args.args[0][-1] == "--help"
    assert call.call_args.kwargs["shell"] is False
