# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reviewed remote code and the FAST tokenizer, as the StarVLA workers use them."""

import hashlib
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from physicalai_starvla_policy_plugin import environments
from physicalai_starvla_policy_plugin._worker import starvla_worker_common as common

WORKER = Path(str(environments.TRAIN_WORKER)).parent
REMOTE_CODE = WORKER / "remote_code"


@pytest.fixture
def worker_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The workers import the vendored packages as ``remote_code``, from their own folder.
    monkeypatch.syspath_prepend(str(WORKER))


def test_vendored_files_are_the_reviewed_upstream_files() -> None:
    readme = (REMOTE_CODE / "README.md").read_text()
    for module, (package, digest) in common.REVIEWED_REMOTE_CODE.items():
        source = REMOTE_CODE / package / f"{module}.py"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, source
        assert f"`{package}/{module}.py`" in readme
        assert digest in readme


def test_unreviewed_remote_code_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"modeling_molmo2\.py\) has not been reviewed"):
        common.load_reviewed_class("modeling_molmo2.Molmo2ForCausalLM", tmp_path)


def test_a_model_folder_copy_must_match_the_reviewed_revision(tmp_path: Path) -> None:
    (tmp_path / "processing_action_tokenizer.py").write_text("# Another revision.\n")
    with pytest.raises(RuntimeError, match=r"differs from the reviewed processing_action_tokenizer\.py"):
        common.load_reviewed_class("processing_action_tokenizer.UniversalActionProcessor", tmp_path)


@pytest.mark.usefixtures("worker_path")
@pytest.mark.parametrize("copy", [False, True], ids=["package-without-code", "pinned-download"])
def test_reviewed_classes_come_from_the_vendored_copies(tmp_path: Path, *, copy: bool) -> None:
    for module in ("transformers", "tokenizers", "scipy"):
        pytest.importorskip(module)
    if copy:  # A downloaded tokenizer folder carries the identical upstream file.
        shutil.copy2(REMOTE_CODE / "fast" / "processing_action_tokenizer.py", tmp_path)

    cls = common.load_reviewed_class(
        "physical-intelligence/fast--processing_action_tokenizer.UniversalActionProcessor",
        tmp_path,
    )

    assert cls.__name__ == "UniversalActionProcessor"
    assert Path(sys.modules[cls.__module__].__file__).parent == REMOTE_CODE / "fast"


@pytest.mark.usefixtures("worker_path")
def test_transformers_asks_the_reviewed_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("transformers")
    dynamic = importlib.import_module("transformers.dynamic_module_utils")
    auto_factory = importlib.import_module("transformers.models.auto.auto_factory")
    monkeypatch.setattr(dynamic, "get_class_from_dynamic_module", dynamic.get_class_from_dynamic_module)
    monkeypatch.setattr(auto_factory, "get_class_from_dynamic_module", auto_factory.get_class_from_dynamic_module)

    common.restrict_remote_code()

    assert auto_factory.get_class_from_dynamic_module is common.load_reviewed_class
    assert dynamic.get_class_from_dynamic_module is common.load_reviewed_class


def _fast_folder(root: Path) -> Path:
    root.mkdir(parents=True)
    for name in common.FAST_TOKENIZER_FILES:
        (root / name).write_text("{}")
    return root


def test_fast_tokenizer_replaces_starvlas_hard_coded_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Tokenizer:  # ruff: ignore[class-as-data-structure]  # Mirrors upstream's default argument.
        def __init__(self, fast_tokenizer_name: str = "playground/Pretrained_models/fast") -> None:
            self.name = fast_tokenizer_name

    module = SimpleNamespace(Fast_Action_Tokenizer=Tokenizer)
    monkeypatch.setitem(sys.modules, "starVLA.model.modules.action_model.fast_ActionHeader", module)
    folder = _fast_folder(tmp_path / "fast")

    common.use_fast_tokenizer(folder)

    assert Tokenizer().name == str(folder)
    with pytest.raises(ValueError, match="missing"):
        common.use_fast_tokenizer(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        common.use_fast_tokenizer(Path("fast"))
