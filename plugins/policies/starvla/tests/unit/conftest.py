# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiny data-only checkpoint fixtures; no model downloads or GPU required."""

from pathlib import Path

import pytest

from physicalai_starvla_policy_plugin.package import IoSpec, write_package


@pytest.fixture
def io() -> IoSpec:
    return IoSpec(
        cameras={"video.primary": "front", "video.wrist": "wrist"},
        state={"state.arm": (0, 2)},
        action={"action.arm": (1, 3), "action.gripper": (0, 1)},
    )


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    (root / "final_model").mkdir(parents=True)
    (root / "config.yaml").write_text("framework:\n  name: QwenGR00T\n")
    (root / "dataset_statistics.json").write_text('{"franka": {}}')
    path = root / "final_model" / "model.safetensors"
    path.write_bytes(b"tiny fake model")
    return path


@pytest.fixture
def backbone(tmp_path: Path) -> Path:
    root = tmp_path / "backbone"
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "qwen3_vl"}')
    (root / "model.safetensors").write_bytes(b"tiny fake backbone")
    (root / "tokenizer.json").write_text("{}")
    (root / "untrusted.py").write_text("raise RuntimeError('must not ship')")
    return root


@pytest.fixture
def package(tmp_path: Path, checkpoint: Path, backbone: Path, io: IoSpec) -> Path:
    return write_package(checkpoint, tmp_path / "export", backbone=backbone, io=io, unnorm_key="franka")
