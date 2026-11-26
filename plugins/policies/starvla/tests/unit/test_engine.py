# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine discovery and native training/export routing without provisioning CUDA."""

import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from physicalai import isolate
from physicalai.engine import Engine, Run, prune, to_isolate_environment

from physicalai_starvla_policy_plugin import engine, environments, openvino_package
from physicalai_starvla_policy_plugin.package import ARTIFACT, RUN_INFO, IoSpec, load_descriptor


def test_entry_point_and_pins() -> None:
    env = engine.StarVLAEngine.environment_for_policy("QwenGR00T")
    assert Engine.get("starvla") is engine.StarVLAEngine
    assert engine.StarVLAEngine.policies() == ["QwenFast", "QwenGR00T", "QwenOFT", "QwenPI"]
    assert env.pinned().base_version == "1.0.1"
    assert env.accelerators == ("cu128",)
    assert len(environments.QWENGR00T.provenance()["lock_sha256"]) == 64
    assert len(environments.QWENGR00T.revision) == 40


@pytest.mark.parametrize(
    "spec",
    [
        *environments.POLICY_ENVIRONMENTS.values(),
        *environments.OPENVINO_EXPORT_ENVIRONMENTS.values(),
        *environments.QWEN35_ENVIRONMENTS.values(),
    ],
    ids=lambda spec: f"{spec.name}-{spec.policy}",
)
def test_duplicated_pins_agree(spec: environments.PolicyEnvironment) -> None:
    """Each framework's commit, version and environment are declared in several places; keep them in sync."""
    project = Path(str(spec.project))
    pyproject = tomllib.loads((project / "pyproject.toml").read_text())
    lock = tomllib.loads((project / "uv.lock").read_text())
    (starvla,) = [package for package in lock["package"] if package["name"] == "starvla"]
    assert project.name == spec.name
    assert pyproject["tool"]["uv"]["sources"]["starvla"]["rev"] == spec.revision
    assert starvla["source"]["git"].endswith(f"#{spec.revision}")
    assert starvla["version"] == spec.version
    if spec.policy in engine.StarVLAEngine.policy_envs and spec is environments.POLICY_ENVIRONMENTS[spec.policy]:
        env = engine.StarVLAEngine.environment_for_policy(spec.policy)
        assert env.name == spec.name
        assert str(env.pinned()) == spec.version
        # Training (engine) and Runtime (isolate) must resolve to one cached worker environment.
        assert to_isolate_environment(env, "cu128").key() == spec.environment.key()


def test_openvino_export_environment_only_adds_the_converter() -> None:
    """Every package shared with the training environment keeps its locked version."""
    for policy, export in environments.OPENVINO_EXPORT_ENVIRONMENTS.items():
        train = environments.POLICY_ENVIRONMENTS[policy]
        locks = [
            {p["name"]: p["version"] for p in tomllib.loads((Path(str(s.project)) / "uv.lock").read_text())["package"]}
            for s in (train, export)
        ]
        shared = set(locks[0]) & set(locks[1])
        assert {name for name in shared if locks[0][name] != locks[1][name]} == set()
        assert {"openvino", "nncf"} <= set(locks[1]) - set(locks[0])
        assert set(locks[0]) - set(locks[1]) == {"physicalai-starvla-worker"}


def test_frameworks_with_the_same_dependencies_share_one_environment_and_there_is_no_default() -> None:
    assert engine.StarVLAEngine.env is None
    # QwenPI shares QwenGR00T's locked project, so the engine lists (and prepares) it once.
    assert [env.name for env in engine.StarVLAEngine.environments()] == ["qwengr00t", "qwengr00t-openvino", "qwen35"]
    assert engine.StarVLAEngine.environment_for_policy("QwenPI") == engine.StarVLAEngine.environment_for_policy(
        "QwenGR00T",
    )
    assert environments.QWENPI.environment.key() == environments.QWENGR00T.environment.key()
    assert environments.QWENPI.provenance() == environments.QWENGR00T.provenance()
    assert environments.QWENOFT.environment.key() == environments.QWENGR00T.environment.key()
    with pytest.raises(ValueError, match="no worker environment"):
        environments.for_policy("QwenAdapter")
    assert environments.for_policy(None) is environments.QWENGR00T


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--help"], None),
        (["--config_yaml", "CONFIG"], "QwenGR00T"),
        (["--config_yaml=CONFIG", "framework.name=QwenOFT"], "QwenOFT"),
        (["--config_yaml", "missing.yaml", "--framework.name=QwenPI"], "QwenPI"),
    ],
)
def test_framework_is_read_from_the_config_and_overrides(tmp_path: Path, argv: list[str], expected: str | None) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("framework:\n  name: QwenGR00T\n")

    assert engine._policy_from_cli([token.replace("CONFIG", str(config)) for token in argv]) == expected


def test_unsupported_framework_is_rejected_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("framework:\n  name: QwenAdapter\n")
    runner = Mock()
    monkeypatch.setattr(engine, "run_worker", runner)
    call = Mock(return_value=0)
    monkeypatch.setattr(engine.subprocess, "call", call)

    with pytest.raises(ValueError, match="'QwenAdapter' has no worker environment"):
        engine.StarVLAEngine.run(
            {"config_yaml": str(config)},
            dataset=tmp_path,
            output_dir=tmp_path / "out",
            device="cuda",
        )
    with pytest.raises(ValueError, match="'QwenAdapter' has no worker environment"):
        engine.StarVLAEngine.cli(["--config_yaml", str(config)])
    runner.assert_not_called()
    call.assert_not_called()


@pytest.mark.parametrize(
    ("mask", "device", "expected"),
    [
        (None, "cuda:2", {"CUDA_VISIBLE_DEVICES": "2"}),
        ("1", "cuda:0", {"CUDA_VISIBLE_DEVICES": "1"}),
        ("2,3", "cuda:1", {"CUDA_VISIBLE_DEVICES": "3"}),
        ("GPU-aaa, GPU-bbb", "cuda:1", {"CUDA_VISIBLE_DEVICES": "GPU-bbb"}),
        ("1", "cuda", {}),
        ("1", "auto", {}),
    ],
)
def test_cuda_index_selects_within_an_existing_device_mask(
    monkeypatch: pytest.MonkeyPatch,
    mask: str | None,
    device: str,
    expected: dict[str, str],
) -> None:
    """``cuda:N`` is a logical index, so it must index the caller's mask, not replace it."""
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    assert engine._device_env(device) == expected


@pytest.mark.parametrize(("mask", "device"), [("1", "cuda:1"), ("", "cuda:0"), ("2,3", "cuda:2")])
def test_cuda_index_outside_the_device_mask_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    mask: str,
    device: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        engine._device_env(device)


@pytest.mark.parametrize("framework", ["QwenGR00T", "QwenPI", "QwenOFT", "QwenFast"])
def test_headless_settings_are_overridden_by_studio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config = tmp_path / "train.yaml"
    config.write_text(f"framework:\n  name: {framework}\n")
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
    assert args[0].name == "qwengr00t"
    assert args[1][-1] == "run_id=studio"
    assert "trainer.max_train_steps=20" in args[1]
    assert kwargs["extra_env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert kwargs["extra_env"]["PHYSICALAI_STARVLA_FRAMEWORK"] == framework
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
            "environment": environments.QWENGR00T.provenance(),
            "backbone_path": str(backbone),
            "unnorm_key": "franka",
        }),
    )
    exported = engine.StarVLAEngine.export(Run("starvla", root, checkpoint), tmp_path / "out")
    assert load_descriptor(exported / ARTIFACT)["unnorm_key"] == "franka"
    with pytest.raises(ValueError, match="native or openvino"):
        engine.StarVLAEngine.export_native(
            checkpoint,
            tmp_path / "onnx",
            backbone_path=backbone,
            io=io,
            unnorm_key="franka",
            backend="onnx",
        )


def test_cli_inherits_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environments, "interpreter_path", lambda *_: Path("/worker/python"))
    monkeypatch.delenv("PHYSICALAI_STARVLA_FRAMEWORK", raising=False)
    call = Mock(return_value=7)
    monkeypatch.setattr(engine.subprocess, "call", call)
    assert engine.StarVLAEngine.cli(["--help"]) == 7
    assert call.call_args.args[0][-1] == "--help"
    assert call.call_args.kwargs["shell"] is False
    assert "PHYSICALAI_STARVLA_FRAMEWORK" not in call.call_args.kwargs["env"]


def test_cli_tells_the_worker_which_framework_it_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("framework:\n  name: QwenPI\n")
    prepared = Mock(return_value=Path("/worker/python"))
    monkeypatch.setattr(environments, "interpreter_path", prepared)
    call = Mock(return_value=0)
    monkeypatch.setattr(engine.subprocess, "call", call)

    assert engine.StarVLAEngine.cli(["--config_yaml", str(config)]) == 0

    assert prepared.call_args.args[0] is environments.QWENPI
    assert call.call_args.kwargs["env"]["PHYSICALAI_STARVLA_FRAMEWORK"] == "QwenPI"


@pytest.mark.parametrize("framework", ["QwenPI", "QwenOFT"])
def test_frameworks_without_a_conversion_export_native_packages_only(
    checkpoint: Path,
    backbone: Path,
    io: IoSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
) -> None:
    capabilities = engine.StarVLAEngine.capabilities(framework)
    assert capabilities.export_formats == ("native",)
    assert capabilities.enablements == ("qwen35",)
    (checkpoint.parents[1] / "config.yaml").write_text(f"framework:\n  name: {framework}\n")

    native = engine.StarVLAEngine.export_native(
        checkpoint,
        tmp_path / "native",
        backbone_path=backbone,
        io=io,
        unnorm_key="franka",
    )
    assert load_descriptor(native / ARTIFACT)["policy"] == framework

    # Refused before staging a native copy (the whole backbone) or preparing the conversion environment.
    prepared, staged = Mock(), Mock()
    monkeypatch.setattr(environments, "interpreter_path", prepared)
    monkeypatch.setattr(openvino_package, "write_package", staged)
    with pytest.raises(ValueError, match=f"'{framework}' has no OpenVINO export"):
        engine.StarVLAEngine.export_native(
            checkpoint,
            tmp_path / "openvino",
            backbone_path=backbone,
            io=io,
            unnorm_key="franka",
            backend="openvino",
        )
    prepared.assert_not_called()
    staged.assert_not_called()


def test_openvino_export_is_an_enablement_with_its_own_environment() -> None:
    capabilities = engine.StarVLAEngine.capabilities("QwenGR00T")
    export = engine.StarVLAEngine.environment(environments.QWENGR00T_OPENVINO.name)

    assert set(capabilities.export_formats) == {"native", "openvino"}
    assert set(capabilities.enablements) == {environments.QWENGR00T_OPENVINO.name, "qwen35"}
    assert export.worker == environments.OPENVINO_EXPORT_WORKER
    assert to_isolate_environment(export, "cu128").key() == environments.QWENGR00T_OPENVINO.environment.key()


@pytest.mark.parametrize(
    "spec",
    [environments.QWENGR00T, environments.QWENGR00T_OPENVINO],
    ids=lambda spec: spec.name,
)
def test_environments_are_prepared_by_the_engine_with_its_labels(
    spec: environments.PolicyEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Training, export and Runtime prepare one labelled environment, so engine list and prune see it."""
    prepared: list[isolate.PreparedEnvironment] = []

    def fake_prepare(
        environment: isolate.Environment,
        *,
        provider: isolate.UvProvider,
        offline: bool,
        labels: dict[str, str],
    ) -> isolate.PreparedEnvironment:
        del provider, offline
        root = tmp_path / environment.key()
        (root / "venv" / "bin").mkdir(parents=True, exist_ok=True)
        (root / "venv" / "bin" / "python").write_bytes(b"python")
        prepared.append(
            isolate.PreparedEnvironment(path=root, key=environment.key(), profile=environment.profile, labels=labels),
        )
        return prepared[-1]

    monkeypatch.setattr(isolate, "prepare", fake_prepare)
    declared = {env.name: env for env in engine.StarVLAEngine.environments()}

    python = environments.interpreter_path(spec)

    (entry,) = prepared
    assert python == entry.python
    # The same cached environment a Runtime-only install prepares, labelled as the engine declares it.
    assert entry.key == spec.environment.key()
    assert entry.labels["engine.framework"] == "starvla"
    assert entry.labels["engine.name"] == spec.name
    assert entry.labels["engine.key"] == declared[spec.name].key()

    # engine prune keeps it: the installed StarVLA engine still declares it.
    removed: list[isolate.PreparedEnvironment] = []
    monkeypatch.setattr(isolate, "list_prepared", lambda: list(prepared))
    monkeypatch.setattr(isolate, "remove_prepared", removed.append)
    assert prune() == []
    assert removed == []


def test_runtime_only_installs_prepare_through_isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without physicalai-train, the Runtime adapter prepares the same environment through isolate."""
    calls: list[isolate.Environment] = []
    monkeypatch.setitem(sys.modules, "physicalai.engine", None)
    monkeypatch.setattr(
        environments,
        "prepare",
        lambda environment: calls.append(environment) or SimpleNamespace(python=tmp_path / "python"),
    )

    assert environments.interpreter_path(environments.QWENGR00T) == tmp_path / "python"
    assert calls == [environments.QWENGR00T.environment]


def test_qwen35_environment_only_moves_transformers() -> None:
    """The Qwen3.5 project is the Qwen3-VL one on transformers 5; every other shared package keeps its version."""
    locks = [
        {p["name"]: p["version"] for p in tomllib.loads((Path(str(s.project)) / "uv.lock").read_text())["package"]}
        for s in (environments.QWENGR00T, environments.QWENGR00T_QWEN35)
    ]
    shared = set(locks[0]) & set(locks[1])
    assert {name for name in shared if locks[0][name] != locks[1][name]} == {"transformers", "huggingface-hub"}
    assert locks[1]["transformers"] == "5.10.4"
    assert set(locks[0]) - set(locks[1]) == {"physicalai-starvla-worker"}
    assert set(locks[1]) - set(locks[0]) <= {
        "physicalai-starvla-qwen35-worker",
        "typer",
        "shellingham",
        "annotated-doc",
    }


def test_qwen35_is_an_enablement_with_its_own_environment() -> None:
    env = engine.StarVLAEngine.environment("qwen35")
    assert env.name == "qwen35"
    assert to_isolate_environment(env, "cu128").key() == environments.QWENGR00T_QWEN35.environment.key()
    assert env.pinned() == engine.StarVLAEngine.environment_for_policy("QwenGR00T").pinned()
    # QwenPI and QwenOFT share it; QwenFast needs a Qwen3.5 "-Action" backbone StarVLA does not publish.
    assert {spec.name for spec in environments.QWEN35_ENVIRONMENTS.values()} == {"qwen35"}
    assert engine.StarVLAEngine.capabilities("QwenFast").enablements == ()
    with pytest.raises(ValueError, match=r"'QwenFast' has no Qwen3\.5 environment; Qwen3\.5 supports"):
        environments.for_policy("QwenFast", "qwen3_5")
    with pytest.raises(ValueError, match="backbone type 'molmo2'"):
        environments.for_policy("QwenGR00T", "molmo2")


@pytest.mark.parametrize(
    ("base_vlm", "expected"),
    [
        ("/models/Qwen3-VL-4B-Instruct", "qwen3_vl"),
        ("/models/Qwen3.5-0.8B", "qwen3_5"),
        ("/models/Qwen3-VL-2B/Qwen3.5-0.8B", "qwen3_vl"),  # StarVLA checks Qwen3-VL first.
        ("/models/Qwen2.5-VL-3B", "qwen3_vl"),  # Another VLM: the default, which the worker refuses.
        ("/home/nora/Qwen3.5-0.8B", "qwen3_vl"),  # StarVLA loads nora's Qwen2.5-VL wrapper first.
        (None, "qwen3_vl"),
    ],
)
def test_the_backbone_type_follows_starvlas_vlm_dispatch(base_vlm: str | None, expected: str) -> None:
    assert environments.backbone_for_path(base_vlm) == expected


@pytest.mark.parametrize("framework", ["QwenGR00T", "QwenPI", "QwenOFT"])
def test_qwen35_backbones_train_in_their_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(f"framework:\n  name: {framework}\n  qwenvl:\n    base_vlm: ./Qwen3-VL-4B-Instruct\n")
    runner = Mock(return_value=Run("starvla", tmp_path, None))
    monkeypatch.setattr(engine, "run_worker", runner)
    call = Mock(return_value=0)
    monkeypatch.setattr(engine.subprocess, "call", call)
    prepared = Mock(return_value=Path("/worker/python"))
    monkeypatch.setattr(environments, "interpreter_path", prepared)
    override = "framework.qwenvl.base_vlm=/models/Qwen3.5-0.8B"

    engine.StarVLAEngine.run(
        {"config_yaml": str(config), "overrides": [override]},
        dataset=tmp_path,
        output_dir=tmp_path / "studio",
        device="cuda",
    )
    engine.StarVLAEngine.cli(["--config_yaml", str(config), override])

    args, kwargs = runner.call_args
    assert args[0].name == "qwen35"
    assert kwargs["extra_env"]["PHYSICALAI_STARVLA_BACKBONE"] == "qwen3_5"
    assert kwargs["extra_env"]["PHYSICALAI_STARVLA_PROVENANCE"] == json.dumps(
        environments.QWENGR00T_QWEN35.provenance(),
    )
    assert prepared.call_args.args[0] is environments.QWEN35_ENVIRONMENTS[framework]
    assert call.call_args.kwargs["env"]["PHYSICALAI_STARVLA_BACKBONE"] == "qwen3_5"
    # Without the override, the config's Qwen3-VL backbone keeps the default environment.
    engine.StarVLAEngine.run({"config_yaml": str(config)}, dataset=tmp_path, output_dir=tmp_path / "vl", device="cuda")
    assert runner.call_args.args[0].name == "qwengr00t"
    assert runner.call_args.kwargs["extra_env"]["PHYSICALAI_STARVLA_BACKBONE"] == "qwen3_vl"


def test_qwenfast_on_qwen35_is_refused_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("framework:\n  name: QwenFast\n  qwenvl:\n    base_vlm: /models/Qwen3.5-0.8B-Action\n")
    runner = Mock()
    monkeypatch.setattr(engine, "run_worker", runner)

    with pytest.raises(ValueError, match=r"no Qwen3\.5 environment"):
        engine.StarVLAEngine.run(
            {"config_yaml": str(config)},
            dataset=tmp_path,
            output_dir=tmp_path / "o",
            device="cuda",
        )
    runner.assert_not_called()


def test_export_from_a_qwen35_run(checkpoint: Path, io: IoSpec, tmp_path: Path) -> None:
    backbone = tmp_path / "Qwen3.5-0.8B"
    backbone.mkdir()
    (backbone / "config.json").write_text('{"model_type": "qwen3_5"}')
    (backbone / "model.safetensors-00001-of-00001.safetensors").write_bytes(b"tiny fake backbone")
    root = checkpoint.parents[1]
    (root / "io.json").write_text(json.dumps({"cameras": io.cameras, "state": io.state, "action": io.action}))
    record = {
        "environment": environments.QWENGR00T.provenance(),
        "backbone_path": str(backbone),
        "unnorm_key": "franka",
    }
    (root / RUN_INFO).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="pins differ"):
        engine.StarVLAEngine.export(Run("starvla", root, checkpoint), tmp_path / "stale")

    (root / RUN_INFO).write_text(json.dumps({**record, "environment": environments.QWENGR00T_QWEN35.provenance()}))
    exported = engine.StarVLAEngine.export(Run("starvla", root, checkpoint), tmp_path / "out")

    descriptor = load_descriptor(exported / ARTIFACT)
    assert descriptor["backbone"] == "Qwen3.5"
    assert descriptor["environment"] == environments.QWENGR00T_QWEN35.provenance()
    assert (exported / ARTIFACT / "Qwen3.5" / "model.safetensors-00001-of-00001.safetensors").is_file()


def test_all_devices_trains_on_every_gpu_the_caller_exposes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config = tmp_path / "train.yaml"
    config.write_text("framework:\n  name: QwenGR00T\n")
    runner = Mock(return_value=Run("starvla", tmp_path, None))
    monkeypatch.setattr(engine, "run_worker", runner)
    settings = {"config_yaml": str(config), "all_devices": True}

    engine.StarVLAEngine.run(settings, dataset=tmp_path, output_dir=tmp_path / "ddp", device="cuda")

    args, kwargs = runner.call_args
    assert args[1][-1] == "--all_devices"
    assert "CUDA_VISIBLE_DEVICES" not in kwargs["extra_env"]
    assert engine.StarVLAEngine.config_schema("QwenGR00T")["properties"]["all_devices"] == {"type": "boolean"}
    with pytest.raises(ValueError, match="all_devices trains on every visible GPU"):
        engine.StarVLAEngine.run(settings, dataset=tmp_path, output_dir=tmp_path / "one", device="cuda:1")
    with pytest.raises(TypeError, match="true or false"):
        engine.StarVLAEngine.run(
            {**settings, "all_devices": "yes"},
            dataset=tmp_path,
            output_dir=tmp_path / "bad",
            device="cuda",
        )
    engine.StarVLAEngine.run({"config_yaml": str(config)}, dataset=tmp_path, output_dir=tmp_path / "one", device="cuda")
    assert "--all_devices" not in runner.call_args.args[1]
