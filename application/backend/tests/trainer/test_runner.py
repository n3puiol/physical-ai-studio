# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trainer runner's job plumbing.

The training run itself lives in ``training.run_training_job`` and is
tested there; what the runner owns is where the snapshot and outputs go, how a
canceled run is signalled to the queue worker, and the model archive.
"""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from trainer.runner import JobCanceledError, TrainerRunner

if TYPE_CHECKING:
    from pathlib import Path

    from trainer.schemas import SubmitJobRequest

RUNNER = "trainer.runner"
_JOB_ID = "3f6c1c1e-0e5a-4f10-9d0f-2b8b3a4c5d6e"


@pytest.fixture
def storage(tmp_path: Path):
    """Point the runner at a throwaway storage directory."""
    settings = MagicMock()
    settings.storage_dir = tmp_path
    settings.datasets_dir = tmp_path / "datasets"
    settings.models_dir = tmp_path / "models"
    settings.archives_dir = tmp_path / "archives"
    with patch(f"{RUNNER}.get_settings", return_value=settings):
        yield settings


def _completed_run(model_files: dict[str, bytes] | None = None):
    """Fake a successful run_training_job by writing the model output it would."""

    def _run(_spec, *, output_dir, **_kwargs) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (model_files or {"model.ckpt": b"weights"}).items():
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    return _run


def test_run_passes_the_submitted_spec_and_job_paths(storage, sample_request: SubmitJobRequest) -> None:
    with patch("training.run_training_job", side_effect=_completed_run()) as mock_run:
        TrainerRunner().run(_JOB_ID, sample_request, should_stop=lambda: False, report=MagicMock())

    kwargs = mock_run.call_args.kwargs
    assert mock_run.call_args.args[0] == sample_request.spec
    assert kwargs["dataset_root"] == storage.datasets_dir / _JOB_ID
    assert kwargs["output_dir"] == storage.models_dir / _JOB_ID
    assert kwargs["cache_dir"] == storage.storage_dir / "cache" / _JOB_ID


def test_run_archives_the_trained_model(storage, sample_request: SubmitJobRequest) -> None:
    files = {"model.ckpt": b"weights", "exports/openvino/model.xml": b"<net/>"}

    with patch("training.run_training_job", side_effect=_completed_run(files)):
        archive = TrainerRunner().run(_JOB_ID, sample_request, should_stop=lambda: False, report=MagicMock())

    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == ["exports/openvino/model.xml", "model.ckpt"]


def test_run_raises_job_canceled_when_a_stop_was_requested(storage, sample_request: SubmitJobRequest) -> None:
    """``run_training_job`` returns on cancel; the queue worker needs an exception."""
    with (
        patch("training.run_training_job"),
        pytest.raises(JobCanceledError),
    ):
        TrainerRunner().run(_JOB_ID, sample_request, should_stop=lambda: True, report=MagicMock())


def test_run_removes_the_uploaded_dataset(storage, sample_request: SubmitJobRequest) -> None:
    dataset_dir = storage.datasets_dir / _JOB_ID
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "meta.json").write_bytes(b"{}")

    with patch("training.run_training_job", side_effect=_completed_run()):
        TrainerRunner().run(_JOB_ID, sample_request, should_stop=lambda: False, report=MagicMock())

    assert not dataset_dir.exists()


def test_run_removes_the_uploaded_dataset_after_a_failure(storage, sample_request: SubmitJobRequest) -> None:
    """A failed job must not leave its uploaded dataset on the trainer's disk."""
    dataset_dir = storage.datasets_dir / _JOB_ID
    dataset_dir.mkdir(parents=True)

    with (
        patch("training.run_training_job", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        TrainerRunner().run(_JOB_ID, sample_request, should_stop=lambda: False, report=MagicMock())

    assert not dataset_dir.exists()


def test_cleanup_job_outputs_removes_the_model_and_cache(storage) -> None:
    model_dir = storage.models_dir / _JOB_ID
    cache_dir = storage.storage_dir / "cache" / _JOB_ID
    for path in (model_dir, cache_dir):
        path.mkdir(parents=True)

    TrainerRunner.cleanup_job_outputs(_JOB_ID)

    assert not model_dir.exists()
    assert not cache_dir.exists()


def _interrupted_export(model_dir: Path) -> None:
    """Leave a model directory as a trainer killed mid-export would: torch done, openvino half-written."""
    from training.export import EXPORTS_DIRNAME, ExportReport, ExportStatus

    model_dir.mkdir(parents=True)
    (model_dir / "model.ckpt").write_bytes(b"weights")
    for name in ("torch", "openvino"):
        (model_dir / EXPORTS_DIRNAME / name).mkdir(parents=True)
        (model_dir / EXPORTS_DIRNAME / name / "model.bin").write_bytes(b"x")
    report = ExportReport()
    report.plan(["torch", "openvino"])
    report.record("torch", ExportStatus.SUCCEEDED)
    report.save(model_dir)


def test_salvage_archives_a_model_whose_training_finished(storage) -> None:
    """A job that died while exporting still delivers its checkpoint and finished exports."""
    from training.export import ExportReport

    model_dir = storage.models_dir / _JOB_ID
    _interrupted_export(model_dir)

    archive = TrainerRunner.salvage(_JOB_ID, "Trainer process exited unexpectedly (-11)", report=MagicMock())

    assert archive is not None
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert {"model.ckpt", "exports/torch/model.bin", "export_report.json"} <= names
    assert not any(name.startswith("exports/openvino") for name in names)
    report = ExportReport.load(model_dir)
    assert report is not None and report.finished
    assert report.failures[0].backend == "openvino"
    assert report.failures[0].error == "export did not finish (Trainer process exited unexpectedly (-11))"
    assert TrainerRunner.completion_message(_JOB_ID) == "Training finished; OpenVINO export failed"


def test_salvage_leaves_a_finished_export_report_untouched(storage) -> None:
    """Dying after export (e.g. while archiving) just archives again."""
    from training.export import ExportReport, ExportStatus

    model_dir = storage.models_dir / _JOB_ID
    model_dir.mkdir(parents=True)
    (model_dir / "model.ckpt").write_bytes(b"weights")
    report = ExportReport(planned=True, finished=True)
    report.record("torch", ExportStatus.SUCCEEDED)
    report.save(model_dir)

    assert TrainerRunner.salvage(_JOB_ID, "boom", report=MagicMock()) is not None
    assert ExportReport.load(model_dir) == report


def test_salvage_declines_a_run_that_never_finished_training(storage) -> None:
    """No export report means training never published a checkpoint: nothing to deliver."""
    model_dir = storage.models_dir / _JOB_ID
    model_dir.mkdir(parents=True)
    (model_dir / "model.ckpt").write_bytes(b"weights")

    assert TrainerRunner.salvage(_JOB_ID, "boom", report=MagicMock()) is None


def test_salvage_declines_a_model_without_a_checkpoint(storage) -> None:
    from training.export import ExportReport

    model_dir = storage.models_dir / _JOB_ID
    model_dir.mkdir(parents=True)
    ExportReport().save(model_dir)

    assert TrainerRunner.salvage(_JOB_ID, "boom", report=MagicMock()) is None
