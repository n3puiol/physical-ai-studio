# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-free OpenVINO pipeline pieces, packaging, and the Runtime adapter with fake graphs.

The real conversion is checked against PyTorch by the export worker itself on every export;
these tests pin the numpy ports to hand-computed values and the glue around the graphs.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from physicalai.inference import InferenceModel

from physicalai_starvla_policy_plugin import engine, environments, openvino_package, openvino_runtime
from physicalai_starvla_policy_plugin._worker import starvla_openvino_pipeline as pipeline
from physicalai_starvla_policy_plugin.package import ARTIFACT, IoSpec

VOCAB = {"[UNK]": 0, "<vs>": 1, "<img>": 2, "<ve>": 3, "pick": 4, "up": 5, "go": 6}


@pytest.mark.parametrize(
    ("size", "expected"),
    [((256, 256), (256, 256)), ((224, 224), (256, 256)), ((480, 640), (480, 640)), ((250, 333), (256, 320))],
)
def test_smart_resize_matches_qwen_vl(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert pipeline.smart_resize(*size, 32, 65536, 16777216) == expected


def test_image_patches_use_merge_window_order() -> None:
    spec = {"resized": [8, 8], "mean": [0.0] * 3, "std": [1 / 255] * 3, "patch_size": 2, "temporal_patch_size": 2}
    spec["merge_size"] = 2
    image = np.arange(8 * 8 * 3, dtype=np.int64).reshape(8, 8, 3).astype(np.uint8)
    patches = pipeline.image_patches(image, spec)
    assert patches.shape == (16, 3 * 2 * 2 * 2)
    # Row for patch (y=1, x=2): merge block (0, 1), position (1, 0) inside it -> 1*4 + 1*2 + 0.
    # Columns are (channel, temporal, dy, dx); the frame is repeated over the temporal axis.
    row = patches[6].reshape(3, 2, 2, 2)
    np.testing.assert_array_equal(row[:, 0], row[:, 1])
    np.testing.assert_array_equal(row[:, 0], image[2:4, 4:6].transpose(2, 0, 1).astype(np.float32))


def test_rope_index_offsets_image_grid_after_text() -> None:
    ids = np.array([[9, 9, 1, 2, 2, 2, 2, 3, 9]])
    positions = pipeline.rope_index(ids, [[1, 4, 4]], {"merge_size": 2, "image_token_id": 2})
    assert positions.shape == (3, 1, 9)
    np.testing.assert_array_equal(
        positions[:, 0],
        [[0, 1, 2, 3, 3, 3, 3, 5, 6], [0, 1, 2, 3, 3, 4, 4, 5, 6], [0, 1, 2, 3, 4, 3, 4, 5, 6]],
    )
    with pytest.raises(ValueError, match="image"):
        pipeline.rope_index(ids, [[1, 4, 4], [1, 4, 4]], {"merge_size": 2, "image_token_id": 2})


@pytest.mark.parametrize(
    ("mode", "stats", "raw", "normalized"),
    [
        ("min_max", {"min": [0, 1], "max": [2, 1]}, [1.5, 7.0], [0.5, 0.0]),
        ("q99", {"q01": [0, 1], "q99": [1, 1]}, [5.0, 7.0], [2.2, 2.2]),
        ("q99", {"q01": [0, 0], "q99": [4, 4]}, [1.0, 3.0], [-0.5, 0.5]),
        ("mean_std", {"mean": [1, 2], "std": [2, 0]}, [5.0, 7.0], [2.0, 7.0]),
        ("binary", {}, [0.2, 0.9], [0.0, 1.0]),
    ],
)
def test_normalization_matches_starvla(mode: str, stats: dict, raw: list, normalized: list) -> None:
    spec = {"mode": mode, "stats": stats}
    np.testing.assert_allclose(pipeline.normalize(np.array(raw), spec, 0.5), normalized, rtol=1e-6)


def test_unnormalization_inverts_starvla_modes() -> None:
    x = np.array([[-1.0, 0.0, 1.0]], np.float32)
    np.testing.assert_allclose(
        pipeline.unnormalize(x, {"mode": "min_max", "stats": {"min": [0] * 3, "max": [2] * 3}}, 0.5),
        [[0, 1, 2]],
    )
    np.testing.assert_allclose(
        pipeline.unnormalize(x, {"mode": "mean_std", "stats": {"mean": [1] * 3, "std": [2] * 3}}, 0.5),
        [[-1, 1, 3]],
    )
    np.testing.assert_allclose(pipeline.unnormalize(x, {"mode": "binary", "stats": {}}, 0.5), [[0, 0, 1]])
    with pytest.raises(ValueError, match="mode"):
        pipeline.unnormalize(x, {"mode": "scale", "stats": {}}, 0.5)


def test_bfloat16_embeddings_are_exact(tmp_path: Path) -> None:
    values = np.array([[1.0, -2.5], [0.15625, 3.0e-5]], np.float32)
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    np.save(tmp_path / "e.npy", bits)
    table = pipeline.EmbeddingTable(tmp_path / "e.npy", "bfloat16")
    expected = (bits.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(table(np.array([1, 0])), expected[[1, 0]])
    np.testing.assert_array_equal(table(np.array([0]))[0, :1], [1.0])


# A tiny OpenVINO package: real descriptor, tokenizer and embeddings, fake compiled graphs. ------
@pytest.fixture
def descriptor(io: IoSpec) -> dict[str, Any]:
    return {
        "format_version": 1,
        "backend": "openvino",
        "policy": "QwenGR00T",
        "weights": "fp16",
        "unnorm_key": "franka",
        "io": {"cameras": io.cameras, "state": io.state, "action": io.action},
        "image": {
            "cameras": 2,
            "size": [4, 4],
            "resized": [4, 4],
            "grid": [1, 2, 2],
            "tokens": 4,
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 1,
        },
        "prompt": {"prefix": "<vs><img><ve><vs><img><ve>", "suffix": " go", "image_token": "<img>"},
        "tokens": {"image_token_id": 2, "vision_start_token_id": 1},
        "embedding": {"dtype": "float32", "shape": [len(VOCAB), 3]},
        "state": {"keys": [["state.arm", 2]]},
        "action": {"keys": [["action.arm", 2], ["action.gripper", 1]], "horizon": 2, "dim": 3},
        "normalization": {
            "binary_threshold": 0.5,
            "keys": {
                "state.arm": {"mode": "min_max", "stats": {"min": [0, 0], "max": [1, 2]}},
                "action.arm": {"mode": "mean_std", "stats": {"mean": [10, 20], "std": [1, 2]}},
                "action.gripper": {"mode": "binary", "stats": {}},
            },
        },
        "validation": {"actions": {"max_abs": 0.0}},
    }


def _write_artifact(artifact: Path, descriptor: dict[str, Any]) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers  # ruff: ignore[import-outside-top-level]

    artifact.mkdir(parents=True)
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))  # ruff: ignore[hardcoded-password-func-arg]
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<vs>", "<img>", "<ve>"])
    tokenizer.save(str(artifact / pipeline.TOKENIZER))
    np.save(artifact / pipeline.EMBEDDINGS, np.arange(len(VOCAB) * 3, dtype=np.float32).reshape(-1, 3))
    for name in pipeline.MODELS:
        (artifact / f"{name}.xml").write_text("<net/>")
        (artifact / f"{name}.bin").write_bytes(b"")
    (artifact / pipeline.DESCRIPTOR).write_text(json.dumps(descriptor))
    return artifact


class FakePipeline(pipeline.QwenGR00TPipeline):
    """Real glue, preprocessing and normalization; the three graphs are recorded fakes."""

    calls: list[dict[str, Any]]

    def __init__(self, artifact: Path, device: str = "CPU", config: dict | None = None) -> None:
        """Load the real descriptor, tokenizer and embeddings; record graph calls."""
        from tokenizers import Tokenizer  # ruff: ignore[import-outside-top-level]

        self.artifact, self.device, self.config = artifact, device, config
        self.descriptor = pipeline.load_descriptor(artifact)
        self.embeddings = pipeline.EmbeddingTable(artifact / pipeline.EMBEDDINGS, "float32")
        self.tokenizer = Tokenizer.from_file(str(artifact / pipeline.TOKENIZER))
        self.calls = []
        FakePipeline.last = self

        def vision(inputs: list[np.ndarray]) -> dict[str, np.ndarray]:
            pixels = inputs[0]
            return {"embeds": pixels[:, :3] * 0 + 100, "deepstack": np.ones((len(pixels), 3), np.float32)}

        def llm(inputs: list[np.ndarray]) -> list[np.ndarray]:
            embeds, mask, positions, deepstack = inputs
            self.calls.append({"embeds": embeds, "mask": mask, "positions": positions, "deepstack": deepstack})
            return [embeds + deepstack.sum(0)]

        def head(inputs: list[np.ndarray]) -> list[np.ndarray]:
            hidden, mask, state, noise = inputs
            self.calls.append({"hidden": hidden, "state": state, "noise": noise, "head_mask": mask})
            return [noise * 0 + state.sum()]

        self.models = {"vision": vision, "llm": llm, "head": head}


@pytest.fixture
def ov_package(tmp_path: Path, descriptor: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Path:
    package = tmp_path / "ov-export"
    _write_artifact(package / openvino_package.OPENVINO_ARTIFACT, descriptor)
    openvino_package._write_manifest(package, "QwenGR00T")
    monkeypatch.setattr(openvino_runtime.pipeline, "QwenGR00TPipeline", FakePipeline)
    return package


@pytest.fixture
def observation() -> dict:
    return {
        "images": {"front": np.zeros((4, 4, 3), np.uint8), "wrist": np.full((4, 4, 3), 255, np.uint8)},
        "state": np.array([0.5, 1.0], np.float32),
        "task": "pick up",
    }


def test_runtime_adapter_glue_and_normalization(ov_package: Path, observation: dict) -> None:
    model = InferenceModel(ov_package, seed=3)
    assert model.backend == "starvla_openvino"
    first = model(observation)["action"]
    fake = FakePipeline.last
    assert fake.device == "CPU"
    llm, head = fake.calls[0], fake.calls[1]
    # "<vs><img x4><ve>" twice, then "pick up go".
    assert llm["embeds"].shape == (1, 15, 3)
    image_rows = llm["embeds"][0, [1, 2, 3, 4, 7, 8, 9, 10]]
    np.testing.assert_array_equal(image_rows, 100)
    np.testing.assert_array_equal(llm["embeds"][0, 0], [3, 4, 5])  # "<vs>" row of the embedding table.
    np.testing.assert_array_equal(llm["deepstack"][0, 0, [0, 5, 6]], 0)
    np.testing.assert_array_equal(llm["mask"], 1)
    # Temporal mRoPE axis: image tokens share t; the next text token follows the grid's max.
    np.testing.assert_array_equal(llm["positions"][0, 0, :6], [0, 1, 1, 1, 1, 3])
    # min_max state normalization: [0.5, 1.0] over [0..1, 0..2] -> [0, 0].
    np.testing.assert_allclose(head["state"], [[[0.0, 0.0]]])
    assert head["noise"].shape == (1, 2, 3)
    assert head["head_mask"].all()
    # Head returns 0 normalized; arm mean_std -> mean (Runtime slice 1:3), gripper binary -> 0 (slice 0:1).
    np.testing.assert_allclose(first, [[[0, 10, 20], [0, 10, 20]]])

    model(observation)
    model.reset()
    model(observation)
    noises = [call["noise"] for call in fake.calls if "noise" in call]
    np.testing.assert_array_equal(noises[0], noises[2])
    assert not np.array_equal(noises[0], noises[1])
    model.close()


def test_runtime_adapter_rejects_wrong_camera_size(ov_package: Path, observation: dict) -> None:
    model = InferenceModel(ov_package)
    observation["images"]["front"] = np.zeros((5, 4, 3), np.uint8)
    with pytest.raises(ValueError, match="4x4x3"):
        model(observation)


def test_device_names_map_to_openvino() -> None:
    adapter = openvino_runtime.StarVLAOpenVINOAdapter(device="xpu")
    assert adapter.output_names == ["action"]
    assert openvino_runtime._DEVICES[adapter.device.lower()] == "GPU"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("validation"),
        lambda d: d.update(backend="native"),
        lambda d: d.update(weights="int4"),
        lambda d: d["state"].update(keys=[["state.arm", 3]]),
        lambda d: d["normalization"]["keys"]["state.arm"].update(mode="scale"),
    ],
)
def test_descriptor_rejects_unvalidated_or_mismatched_packages(
    tmp_path: Path,
    descriptor: dict,
    change: Callable[[dict], object],
) -> None:
    change(descriptor)
    artifact = _write_artifact(tmp_path / "artifact", descriptor)
    with pytest.raises(ValueError, match=r"Unsupported|mapping|modes"):
        openvino_package.load_openvino_descriptor(artifact)


def _fake_export_worker(
    descriptor: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exit_code: int = 0,
) -> list:
    calls = []

    class Completed:
        returncode = exit_code

    def run(argv: list[str], **kwargs: object) -> Completed:
        calls.append((argv, kwargs))
        _write_artifact(Path(argv[argv.index("--output") + 1]), descriptor)
        return Completed()

    monkeypatch.setattr(openvino_package.subprocess, "run", run)
    monkeypatch.setattr(environments, "interpreter_path", lambda *_: Path(sys.executable))
    return calls


def test_openvino_export_keeps_only_graphs_and_records_provenance(
    tmp_path: Path,
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
    descriptor: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_export_worker(descriptor, monkeypatch)
    output = engine.StarVLAEngine.export_native(
        checkpoint,
        tmp_path / "ov",
        backbone_path=backbone,
        io=io,
        unnorm_key="franka",
        backend="openvino",
        openvino_weights="int8",
        image_size=(4, 4),
    )
    argv, kwargs = calls[0]
    assert argv[argv.index("--weights") + 1] == "int8"
    assert argv[argv.index("--image-size") + 1 :][:2] == ["4", "4"]
    assert Path(argv[argv.index("--artifact") + 1]).name == ARTIFACT
    assert not kwargs["env"]["CUDA_VISIBLE_DEVICES"]
    assert kwargs["shell"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"]["artifacts"] == {"starvla_openvino": "policy.starvla_ov"}
    artifact = output / openvino_package.OPENVINO_ARTIFACT
    saved = json.loads((artifact / "package.json").read_text())
    assert saved["export_environment"] == environments.QWENGR00T_OPENVINO.provenance()
    assert sorted(path.name for path in output.iterdir()) == ["manifest.json", "policy.starvla_ov"]
    assert not list(output.rglob("*.safetensors"))
    assert not list(tmp_path.glob(".starvla-openvino-*"))


def test_failed_openvino_export_leaves_nothing(
    tmp_path: Path,
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
    descriptor: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_export_worker(descriptor, monkeypatch, exit_code=3)
    with pytest.raises(RuntimeError, match="exit code 3"):
        openvino_package.write_openvino_package(
            checkpoint,
            tmp_path / "ov",
            backbone=backbone,
            io=io,
            unnorm_key="franka",
        )
    assert not (tmp_path / "ov").exists()
    assert not list(tmp_path.glob(".starvla-openvino-*"))


@pytest.mark.parametrize(
    ("options", "message"),
    [({"weights": "int4"}, "weights"), ({"image_size": (4,)}, "image_size"), ({"image_size": (4, 0)}, "image_size")],
)
def test_openvino_export_options_are_validated(
    tmp_path: Path,
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
    options: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        openvino_package.write_openvino_package(
            checkpoint,
            tmp_path / "ov",
            backbone=backbone,
            io=io,
            unnorm_key="franka",
            **options,
        )


def test_openvino_export_requires_state(tmp_path: Path, checkpoint: Path, backbone: Path) -> None:
    io = IoSpec(cameras={"video.primary": "front"}, state={}, action={"action.arm": (0, 1)})
    with pytest.raises(ValueError, match="state"):
        openvino_package.write_openvino_package(
            checkpoint,
            tmp_path / "ov",
            backbone=backbone,
            io=io,
            unnorm_key="franka",
        )
