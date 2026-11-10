# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for what a model reports about its own exports."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from schemas.model import Model
from training.export import EXPORTS_DIRNAME, ExportReport, ExportStatus

if TYPE_CHECKING:
    from pathlib import Path


def _model(path: Path) -> Model:
    return Model(
        id=uuid4(),
        project_id=uuid4(),
        path=str(path),
        name="m",
        policy="smolvla",
        properties={},
    )


def test_a_model_without_an_export_report_has_no_export_issues(tmp_path: Path) -> None:
    assert _model(tmp_path).export_issues == []


def test_each_failed_backend_is_an_export_issue(tmp_path: Path) -> None:
    (tmp_path / EXPORTS_DIRNAME / "torch").mkdir(parents=True)
    (tmp_path / EXPORTS_DIRNAME / "torch" / "model.pt").write_text("x")
    report = ExportReport(planned=True, finished=True)
    report.record("torch", ExportStatus.SUCCEEDED)
    report.record("openvino", ExportStatus.FAILED, "export process crashed (SIGSEGV) while exporting")
    report.record("executorch", ExportStatus.SKIPPED, "optional dependency missing")
    report.save(tmp_path)
    model = _model(tmp_path)

    [issue] = model.export_issues

    assert issue.id == f"model-{model.id}-export-openvino"
    assert issue.title == "OpenVINO export failed"
    assert issue.copy_text == "export process crashed (SIGSEGV) while exporting"
    assert model.available_backends == ["torch"]
    assert "export_issues" in model.model_dump(mode="json")


def test_an_export_that_never_started_is_one_issue(tmp_path: Path) -> None:
    report = ExportReport()
    report.abort("could not reload the checkpoint (RuntimeError: bad)")
    report.save(tmp_path)

    [issue] = _model(tmp_path).export_issues

    assert issue.title == "Export failed"
    assert issue.detail == "Export failed: could not reload the checkpoint (RuntimeError: bad)"
