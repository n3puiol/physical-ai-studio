# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Package structure, allowlists, and untrusted descriptor validation."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from physicalai.inference.manifest import Manifest

from physicalai_starvla_policy_plugin._worker.starvla_worker_common import trained_with_state
from physicalai_starvla_policy_plugin.package import ARTIFACT, BACKBONE_DIR, IoSpec, load_descriptor, write_package


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
