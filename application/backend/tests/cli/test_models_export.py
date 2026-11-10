# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for re-exporting a persisted model from the CLI."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from physicalai.export import ExportablePolicyMixin, ExportBackend

from cli.models import _export_requested_backends
from training.export import EXPORTS_DIRNAME, ExportReport, ExportStatus


class _Policy(ExportablePolicyMixin):
    def __init__(self, failing: set[ExportBackend] | None = None) -> None:
        self._failing = failing or set()

    def get_supported_export_backends(self) -> list[ExportBackend]:  # type: ignore[override]
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]

    def export(self, output_path, backend, input_sample=None, **export_kwargs) -> None:  # type: ignore[override]
        Path(output_path).mkdir(parents=True, exist_ok=True)
        (Path(output_path) / "model").write_text("x")
        if backend in self._failing:
            msg = "converter crashed"
            raise RuntimeError(msg)


def _failed_openvino_report(model_dir: Path) -> None:
    report = ExportReport(planned=True, finished=True)
    report.record("torch", ExportStatus.SUCCEEDED)
    report.record("openvino", ExportStatus.FAILED, "export process crashed (SIGSEGV) while exporting")
    report.save(model_dir)


def test_a_successful_re_export_clears_the_failure(tmp_path: Path) -> None:
    _failed_openvino_report(tmp_path)

    _export_requested_backends(_Policy(), tmp_path, ["openvino"])

    report = ExportReport.load(tmp_path)
    assert report is not None and report.failures == []
    assert (tmp_path / EXPORTS_DIRNAME / "openvino" / "model").is_file()


def test_a_failed_re_export_is_recorded_and_its_output_removed(tmp_path: Path) -> None:
    _failed_openvino_report(tmp_path)

    with pytest.raises(click.ClickException, match="openvino: converter crashed"):
        _export_requested_backends(_Policy(failing={ExportBackend.OPENVINO}), tmp_path, ["openvino", "onnx"])

    report = ExportReport.load(tmp_path)
    assert report is not None
    assert [(r.backend, r.error) for r in report.failures] == [("openvino", "converter crashed")]
    # An unsupported backend was never meant to exist, so it is not an issue on the model.
    assert "onnx" not in {r.backend for r in report.backends}
    assert not (tmp_path / EXPORTS_DIRNAME / "openvino").exists()
