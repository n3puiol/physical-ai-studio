# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Package structure, allowlists, and untrusted descriptor validation."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from physicalai.inference.manifest import Manifest

from physicalai_starvla_policy_plugin import environments
from physicalai_starvla_policy_plugin._worker.starvla_worker_common import FAST_TOKENIZER_FILES, trained_with_state
from physicalai_starvla_policy_plugin.package import (
    ARTIFACT,
    BACKBONE_DIR,
    IoSpec,
    load_descriptor,
    package_environment,
    write_package,
)


def test_package_roundtrip(package: Path, io: IoSpec) -> None:
    descriptor = load_descriptor(package / ARTIFACT)
    # The fixture run was trained without proprioceptive state (StarVLA's default), so the package maps none.
    assert asdict(IoSpec.from_dict(descriptor["io"])) == json.loads(json.dumps(asdict(replace(io, state={}))))
    assert Manifest.load(package / "manifest.json").model.artifacts == {"starvla": ARTIFACT}
    assert (package / ARTIFACT / BACKBONE_DIR / "tokenizer.json").is_file()
    assert not list(package.rglob("*.py"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", 2),
        ("format_version", True),
        ("policy", "other"),
        ("checkpoint", "../../escape.safetensors"),
        ("checkpoint", "/etc/passwd"),
        ("checkpoint", None),
        ("unnorm_key", 2),
        ("environment", {}),
        ("io", []),
    ],
)
def test_bad_descriptor(package: Path, field: str, value: object) -> None:
    path = package / ARTIFACT / "package.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"Unsupported|Package|paths|normalization|pins|mapping"):
        load_descriptor(path.parent)


def test_descriptor_symlink_escape(package: Path, tmp_path: Path) -> None:
    config = package / ARTIFACT / BACKBONE_DIR / "config.json"
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    config.unlink()
    config.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        load_descriptor(package / ARTIFACT)


def test_export_does_not_overwrite(package: Path, checkpoint: Path, backbone: Path, io: IoSpec) -> None:
    with pytest.raises(FileExistsError):
        write_package(checkpoint, package, backbone=backbone, io=io, unnorm_key="franka")
    assert load_descriptor(package / ARTIFACT)["policy"] == "QwenGR00T"


def test_failed_export_is_atomic(tmp_path: Path, checkpoint: Path, backbone: Path, io: IoSpec) -> None:
    (backbone / "bad.json").symlink_to(tmp_path / "run" / "dataset_statistics.json")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="symlinks"):
        write_package(checkpoint, output, backbone=backbone, io=io, unnorm_key="franka")
    assert not output.exists()
    assert not list(tmp_path.glob(".starvla-*"))


def test_pickle_checkpoint_is_rejected(tmp_path: Path, backbone: Path, io: IoSpec) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not safe")
    with pytest.raises(ValueError, match="safetensors"):
        write_package(checkpoint, tmp_path / "out", backbone=backbone, io=io, unnorm_key="franka")


@pytest.mark.parametrize(
    "action",
    [
        {},
        {"action.a": [1, 2]},
        {"action.a": [0, 2], "action.b": [1, 3]},
        {"action.a": [False, 1]},
        {"action.a": [-1, 2]},
    ],
)
def test_invalid_action_mapping(action: dict) -> None:
    with pytest.raises(ValueError, match=r"mapping|slices|slice"):
        IoSpec(cameras={"video.main": "front"}, state={}, action=action)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("missing", False),
        (False, False),
        ("False", False),
        (0, False),
        (True, True),
        ("true", True),
        ("false", True),  # Upstream only treats False and "False" as off.
        (None, True),
    ],
)
def test_state_rule_mirrors_starvla(value: object, *, expected: bool) -> None:
    assert trained_with_state({} if value == "missing" else {"include_state": value}) is expected


@pytest.mark.parametrize(
    ("vla_data", "keeps_state"),
    [
        ("", False),
        ("datasets:\n  vla_data:\n    include_state: false\n", False),
        ("datasets:\n  vla_data:\n    include_state: true\n", True),
    ],
)
def test_package_maps_state_only_if_training_used_it(
    tmp_path: Path,
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
    vla_data: str,
    *,
    keeps_state: bool,
) -> None:
    (checkpoint.parents[1] / "config.yaml").write_text("framework:\n  name: QwenGR00T\n" + vla_data)
    exported = write_package(checkpoint, tmp_path / "state-export", backbone=backbone, io=io, unnorm_key="franka")
    state = load_descriptor(exported / ARTIFACT)["io"]["state"]
    assert state == ({"state.arm": [0, 2]} if keeps_state else {})


def test_unsupported_backbones_and_package_folders_are_refused(
    tmp_path: Path,
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
    package: Path,
) -> None:
    (backbone / "config.json").write_text('{"model_type": "florence2"}')
    with pytest.raises(ValueError, match=r"requires a Qwen3-VL or Qwen3\.5 backbone"):
        write_package(checkpoint, tmp_path / "florence-export", backbone=backbone, io=io, unnorm_key="franka")
    path = package / ARTIFACT / "package.json"
    data = json.loads(path.read_text())
    data["backbone"] = "../escape"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="backbone"):
        load_descriptor(path.parent)


@pytest.fixture
def fast(tmp_path: Path) -> Path:
    # A FAST tokenizer folder as downloaded from physical-intelligence/fast, code included.
    root = tmp_path / "fast"
    root.mkdir()
    for name in FAST_TOKENIZER_FILES:
        (root / name).write_text("{}")
    (root / "processing_action_tokenizer.py").write_text("raise RuntimeError('must not ship')")
    return root


def test_a_qwenfast_package_carries_its_fast_tokenizer_data(
    tmp_path: Path,
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
    fast: Path,
) -> None:
    config = checkpoint.parents[1] / "config.yaml"
    config.write_text("framework:\n  name: QwenFast\n")
    with pytest.raises(ValueError, match="fast_tokenizer_path"):
        write_package(checkpoint, tmp_path / "no-fast", backbone=backbone, io=io, unnorm_key="franka")

    config.write_text(f"framework:\n  name: QwenFast\n  action_model:\n    fast_tokenizer_path: {fast}\n")
    action = backbone.rename(backbone.with_name("Qwen3-VL-4B-Instruct-Action"))
    package = write_package(checkpoint, tmp_path / "fast-export", backbone=action, io=io, unnorm_key="franka")
    artifact = package / ARTIFACT

    descriptor = load_descriptor(artifact)
    assert descriptor["fast_tokenizer"] == "FAST"
    # StarVLA enables the action tokens only when the backbone path says -Action.
    assert descriptor["backbone"] == "Qwen3-VL-Action"
    assert (artifact / "Qwen3-VL-Action" / "config.json").is_file()
    assert sorted(path.name for path in (artifact / "FAST").iterdir()) == sorted(FAST_TOKENIZER_FILES)
    (artifact / "FAST" / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_descriptor(artifact)


def test_a_qwen35_package_runs_in_the_qwen35_environment(
    tmp_path: Path,
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
) -> None:
    (backbone / "config.json").write_text('{"model_type": "qwen3_5"}')
    package = write_package(checkpoint, tmp_path / "qwen35", backbone=backbone, io=io, unnorm_key="franka")
    artifact = package / ARTIFACT
    descriptor = load_descriptor(artifact)

    assert descriptor["backbone"] == "Qwen3.5"
    assert package_environment(descriptor) is environments.QWENGR00T_QWEN35
    # A package claiming the Qwen3-VL environment for a Qwen3.5 backbone is refused.
    data = json.loads((artifact / "package.json").read_text())
    (artifact / "package.json").write_text(json.dumps({**data, "environment": environments.QWENGR00T.provenance()}))
    with pytest.raises(ValueError, match="pins differ"):
        load_descriptor(artifact)


def test_qwenfast_is_not_packaged_on_qwen35(tmp_path: Path, checkpoint: Path, backbone: Path, io: IoSpec) -> None:
    (backbone / "config.json").write_text('{"model_type": "qwen3_5"}')
    (checkpoint.parents[1] / "config.yaml").write_text("framework:\n  name: QwenFast\n")

    with pytest.raises(ValueError, match=r"no Qwen3\.5 environment"):
        write_package(checkpoint, tmp_path / "fast35", backbone=backbone, io=io, unnorm_key="franka")


def test_world_model_frameworks_need_their_world_model(
    tmp_path: Path,
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
) -> None:
    (checkpoint.parents[1] / "config.yaml").write_text("framework:\n  name: CosmoPredict2GR00T\n")
    with pytest.raises(ValueError, match="needs a Cosmos-Predict2 world model"):
        write_package(checkpoint, tmp_path / "vlm", backbone=backbone, io=io, unnorm_key="franka")
    (backbone / "model_index.json").write_text('{"_class_name": "WanPipeline"}')
    with pytest.raises(ValueError, match="Cosmos2VideoToWorldPipeline"):
        write_package(checkpoint, tmp_path / "wan", backbone=backbone, io=io, unnorm_key="franka")


def test_wan_runs_are_packaged_with_their_wan22_folder(
    tmp_path: Path,
    checkpoint: Path,
    io: IoSpec,
) -> None:
    (checkpoint.parents[1] / "config.yaml").write_text("framework:\n  name: WanOFT\n")
    wan = tmp_path / "Wan2.2-TI2V-5B-Diffusers"
    for part in ("vae", "transformer", "text_encoder"):
        (wan / part).mkdir(parents=True)
        (wan / part / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    (wan / "model_index.json").write_text('{"_class_name": "WanPipeline"}')
    (wan / "vae" / "config.json").write_text('{"z_dim": 16}')
    with pytest.raises(ValueError, match=r"needs a Wan2\.2-TI2V world model"):
        write_package(checkpoint, tmp_path / "wan21", backbone=wan, io=io, unnorm_key="franka")
    (wan / "vae" / "config.json").write_text('{"z_dim": 48}')

    package = write_package(checkpoint, tmp_path / "wan22", backbone=wan, io=io, unnorm_key="franka")

    descriptor = load_descriptor(package / ARTIFACT)
    assert descriptor["backbone"] == "Wan2.2-TI2V"
    assert package_environment(descriptor) is environments.POLICY_ENVIRONMENTS["WanOFT"]
    assert (package / ARTIFACT / "Wan2.2-TI2V" / "vae" / "config.json").is_file()
