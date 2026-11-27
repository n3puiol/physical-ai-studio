# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Handing runs to RLinf and importing its results, with tiny real safetensors checkpoints."""

import json
from pathlib import Path

import pytest

from physicalai_starvla_policy_plugin import environments, rlinf
from physicalai_starvla_policy_plugin.package import ARTIFACT, RUN_INFO, IoSpec, load_descriptor

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


@pytest.fixture
def run(tmp_path: Path, backbone: Path, io: IoSpec) -> Path:
    """A plugin run of QwenOFT on Qwen3-VL: bf16 VLM weights and an fp32 head, as StarVLA saves them.

    Returns:
        The run directory.
    """
    root = tmp_path / "run"
    (root / "final_model").mkdir(parents=True)
    weights = {
        "qwen_vl_interface.model.embed.weight": torch.arange(6, dtype=torch.bfloat16).reshape(3, 2),
        "qwen_vl_interface.model.lm_head.weight": torch.arange(6, dtype=torch.bfloat16).reshape(3, 2),
        "action_model.proj.weight": torch.full((2, 2), 0.1),
    }
    safetensors_torch.save_file(weights, str(root / "final_model" / "model.safetensors"))
    (root / "config.yaml").write_text("framework:\n  name: QwenOFT\n")
    (root / "dataset_statistics.json").write_text('{"franka": {}}')
    (root / "io.json").write_text(json.dumps({"cameras": io.cameras, "state": io.state, "action": io.action}))
    record = {"environment": environments.QWENGR00T.provenance(), "backbone_path": str(backbone)}
    record["unnorm_key"] = "franka"
    (root / RUN_INFO).write_text(json.dumps(record))
    return root


def test_a_run_is_handed_to_rlinf_in_starvlas_layout(run: Path, tmp_path: Path) -> None:
    checkpoint = rlinf.export_for_rlinf(run, tmp_path / "rlinf")

    # StarVLA's loader reads config.yaml and dataset_statistics.json next to checkpoints/.
    assert checkpoint == tmp_path / "rlinf" / "checkpoints" / "steps_0_model.safetensors"
    assert (checkpoint.parents[1] / "config.yaml").read_text() == (run / "config.yaml").read_text()
    assert (checkpoint.parents[1] / "dataset_statistics.json").is_file()
    assert checkpoint.read_bytes() == (run / "final_model" / "model.safetensors").read_bytes()
    with pytest.raises(FileExistsError):
        rlinf.export_for_rlinf(run, tmp_path / "rlinf")


@pytest.mark.parametrize(
    ("config", "backbone_path", "match"),
    [
        ("framework:\n  name: QwenFast\n", "/m/Qwen3-VL-4B-Instruct-Action", "this run is QwenFast"),
        ("framework:\n  name: QwenOFT\n", "/m/Qwen3.5-0.8B", "on qwen3_5"),
        ("framework:\n  name: CosmoPredict2GR00T\n", "/m/Cosmos-Predict2-2B", "CosmoPredict2GR00T on cosmos_predict2"),
    ],
)
def test_runs_rlinf_cannot_run_are_refused(
    run: Path,
    tmp_path: Path,
    config: str,
    backbone_path: str,
    match: str,
) -> None:
    (run / "config.yaml").write_text(config)
    record = json.loads((run / RUN_INFO).read_text())
    (run / RUN_INFO).write_text(json.dumps({**record, "backbone_path": backbone_path}))

    with pytest.raises(ValueError, match=match):
        rlinf.export_for_rlinf(run, tmp_path / "rlinf")


def _rlinf_policy(run: Path, path: Path, *, extra: dict | None = None) -> Path:
    """RLinf's trained policy as safetensors: the StarVLA weights under its prefix in bf16, plus its own.

    Returns:
        The safetensors file.
    """
    base = safetensors_torch.load_file(str(run / "final_model" / "model.safetensors"))
    state = {f"starvla_model.{key}": (value.float() * 2).to(torch.bfloat16) for key, value in base.items()}
    state["actor_logstd"] = torch.zeros(7)
    state["value_head.weight"] = torch.zeros(1, 2)
    state.update(extra or {})
    safetensors_torch.save_file(state, str(path))
    return path


def test_rlinfs_policy_becomes_a_run_that_exports(run: Path, tmp_path: Path) -> None:
    trained = _rlinf_policy(run, tmp_path / "rl.safetensors")

    output = rlinf.import_from_rlinf(trained, run, tmp_path / "rl-run")

    weights = safetensors_torch.load_file(str(output / "final_model" / "model.safetensors"))
    assert set(weights) == {
        "qwen_vl_interface.model.embed.weight",
        "qwen_vl_interface.model.lm_head.weight",
        "action_model.proj.weight",
    }
    head = weights["action_model.proj.weight"]
    assert head.dtype == torch.float32  # The starting run's dtype, holding RLinf's bf16 value.
    assert torch.equal(head, torch.full((2, 2), 0.2).to(torch.bfloat16).float())
    record = json.loads((output / RUN_INFO).read_text())
    assert record["post_training"]["framework"] == "rlinf"
    assert len(record["post_training"]["weights_sha256"]) == 64
    assert record["environment"] == environments.QWENGR00T.provenance()

    from physicalai.engine import Engine, Run  # ruff: ignore[import-outside-top-level]

    package = Engine.get("starvla").export(
        Run("starvla", output, output / "final_model" / "model.safetensors"),
        tmp_path / "pkg",
    )
    assert load_descriptor(package / ARTIFACT)["policy"] == "QwenOFT"


def test_a_bare_starvla_state_dict_is_accepted(run: Path, tmp_path: Path) -> None:
    bare = tmp_path / "bare.safetensors"
    bare.write_bytes((run / "final_model" / "model.safetensors").read_bytes())

    output = rlinf.import_from_rlinf(bare, run, tmp_path / "bare-run")

    assert (output / "final_model" / "model.safetensors").read_bytes() == bare.read_bytes()


def test_weights_that_do_not_match_the_starting_run_are_refused(run: Path, tmp_path: Path) -> None:
    extra = _rlinf_policy(run, tmp_path / "extra.safetensors", extra={"starvla_model.new.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="1 unexpected"):
        rlinf.import_from_rlinf(extra, run, tmp_path / "a")
    wider = {"starvla_model.action_model.proj.weight": torch.ones(3, 2)}
    reshaped = _rlinf_policy(run, tmp_path / "shape.safetensors", extra=wider)
    with pytest.raises(ValueError, match="changed shape"):
        rlinf.import_from_rlinf(reshaped, run, tmp_path / "b")


def test_pickled_checkpoints_are_refused_unread(run: Path, tmp_path: Path) -> None:
    pickled = tmp_path / "rl.pt"
    pickled.write_bytes(b"not read")

    with pytest.raises(ValueError, match="safetensors only"):
        rlinf.import_from_rlinf(pickled, run, tmp_path / "pt-run")
    assert not (tmp_path / "pt-run").exists()
