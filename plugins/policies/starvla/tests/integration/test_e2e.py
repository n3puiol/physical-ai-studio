# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real CUDA training, packaging, and Runtime inference (local data/weights)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from physicalai.engine import Engine
from physicalai.inference import InferenceModel

from physicalai_starvla_policy_plugin.package import IoSpec

pytestmark = pytest.mark.skipif(
    os.environ.get("PHYSICALAI_RUN_STARVLA_E2E") != "1",
    reason="Requires CUDA, local backbone, and a StarVLA dataset",
)


def test_twenty_steps_export_and_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    root = Path(__file__).parents[2]
    backbone = Path(os.environ["PHYSICALAI_STARVLA_BACKBONE"])
    io_path = Path(os.environ.get("PHYSICALAI_STARVLA_IO", root / "configs" / "libero_io.json"))
    io = IoSpec.from_dict(json.loads(io_path.read_text(encoding="utf-8")))
    engine = Engine.get("starvla")
    run = engine.run(
        {
            "config_yaml": os.environ.get("PHYSICALAI_STARVLA_CONFIG", str(root / "configs" / "qwengroot_libero.yaml")),
            "overrides": [
                f"framework.qwenvl.base_vlm={backbone}",
                "trainer.max_train_steps=20",
                "trainer.save_interval=20",
            ],
        },
        dataset=Path(os.environ["PHYSICALAI_STARVLA_DATASETS"]),
        output_dir=tmp_path / "run",
        device="cuda",
    )
    assert run.checkpoint is not None
    assert run.checkpoint.is_file()
    package = engine.export_native(
        run,
        tmp_path / "export",
        backbone_path=backbone,
        io=io,
        unnorm_key=os.environ.get("PHYSICALAI_STARVLA_UNNORM_KEY", "franka"),
    )
    # Use a representative observation, not an all-zero robot state.
    with np.load(os.environ["PHYSICALAI_STARVLA_OBSERVATION"], allow_pickle=False) as arrays:
        observation = {
            "images": {camera: arrays[camera] for camera in io.cameras.values()},
            "task": str(arrays["task"].item()),
        }
        if io.state:
            observation["state"] = arrays["state"]
        model = InferenceModel(package, device="cuda", seed=42)
        try:
            first = model(observation)["action"]
            model.reset()
            repeated = model(observation)["action"]
            assert first.shape[0] == 1
            assert first.shape[-1] == io.action_dim
            assert np.isfinite(first).all()
            np.testing.assert_allclose(first, repeated)
        finally:
            model.close()
