# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for crash-isolated export and the export report.

The workers below run in real spawned children: what is under test is that a
child dying (with a signal, an exit code, or before it planned anything) costs
only the backend it was exporting, never the caller or the other backends.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from training.export import (
    EXPORT_REPORT_NAME,
    EXPORTS_DIRNAME,
    ExportReport,
    ExportStatus,
    discard_unfinished_exports,
    export_completion_message,
    run_exports_isolated,
)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

BACKENDS = ["torch", "openvino", "onnx"]

needs_signals = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


def _write_export(model_dir: str, name: str) -> None:
    target = Path(model_dir) / EXPORTS_DIRNAME / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.bin").write_text(name)


def _exporting_worker(conn: Connection, skip: list[str], model_dir: str, crash: str | None, how: str) -> None:
    """Export ``BACKENDS``, dying while exporting ``crash`` the way ``how`` says."""
    conn.send(("plan", BACKENDS))
    for name in BACKENDS:
        if name in skip:
            continue
        conn.send(("start", name))
        _write_export(model_dir, name)
        if name == crash:
            if how == "segfault":
                os.kill(os.getpid(), signal.SIGSEGV)
            elif how == "exit":
                os._exit(3)
            elif how == "hang":
                time.sleep(60)
        conn.send(("result", name, "succeeded", None, None))
    conn.send(("done",))
    conn.close()


def _crash_before_planning(conn: Connection, skip: list[str]) -> None:
    os.kill(os.getpid(), signal.SIGSEGV)


def _export(tmp_path: Path, crash: str | None = None, how: str = "segfault", **kwargs) -> ExportReport:
    kwargs.setdefault("report", MagicMock())
    kwargs.setdefault("should_stop", lambda: False)
    return run_exports_isolated(_exporting_worker, (str(tmp_path), crash, how), tmp_path, **kwargs)


class TestRunExportsIsolated:
    def test_every_backend_succeeds(self, tmp_path: Path) -> None:
        report = MagicMock()

        result = _export(tmp_path, report=report)

        assert result.finished and result.summary() is None
        assert [r.status for r in result.backends] == [ExportStatus.SUCCEEDED] * 3
        assert ExportReport.load(tmp_path) == result
        assert [call.args for call in report.call_args_list] == [
            (99, f"Exporting to {name} format", {}) for name in BACKENDS
        ]

    @needs_signals
    def test_a_segfault_fails_only_the_backend_being_exported(self, tmp_path: Path) -> None:
        """The SmolVLA case: OpenVINO's converter segfaults, Torch and ONNX still ship."""
        result = _export(tmp_path, crash="openvino", how="segfault")

        statuses = {r.backend: r.status for r in result.backends}
        assert statuses == {
            "torch": ExportStatus.SUCCEEDED,
            "openvino": ExportStatus.FAILED,
            "onnx": ExportStatus.SUCCEEDED,
        }
        assert result.failures[0].error == "export process crashed (SIGSEGV) while exporting"
        assert result.finished
        assert result.summary() == "OpenVINO export failed"
        exports = tmp_path / EXPORTS_DIRNAME
        assert sorted(path.name for path in exports.iterdir()) == ["onnx", "torch"]
        assert ExportReport.load(tmp_path) == result

    def test_a_worker_exiting_mid_backend_fails_that_backend(self, tmp_path: Path) -> None:
        result = _export(tmp_path, crash="torch", how="exit")

        assert result.failures[0].backend == "torch"
        assert result.failures[0].error == "export process exited with code 3 while exporting"
        assert [r.status for r in result.backends[1:]] == [ExportStatus.SUCCEEDED] * 2

    @needs_signals
    def test_a_crash_before_planning_is_reported_without_retrying(self, tmp_path: Path) -> None:
        result = run_exports_isolated(
            _crash_before_planning, (), tmp_path, report=MagicMock(), should_stop=lambda: False
        )

        assert result.finished and result.backends == []
        assert result.error == "export process crashed (SIGSEGV) before exporting"
        assert result.summary() == "export failed: export process crashed (SIGSEGV) before exporting"

    def test_cancellation_terminates_the_worker(self, tmp_path: Path) -> None:
        stop = False

        def report(_progress: int, message: str, _extra: dict) -> None:
            nonlocal stop
            stop = message == "Exporting to openvino format"

        started = time.monotonic()
        result = _export(tmp_path, crash="openvino", how="hang", report=report, should_stop=lambda: stop)

        assert time.monotonic() - started < 30
        assert not result.finished
        assert result.pending == ["openvino", "onnx"]
        assert not (tmp_path / EXPORTS_DIRNAME / "openvino").exists()

    def test_the_real_worker_reports_an_unloadable_checkpoint(self, tmp_path: Path) -> None:
        """``training.job._export_worker`` must be importable by a spawned child and contain its failures."""
        from training import TrainingJobSpec
        from training.job import CHECKPOINT_NAME, _export_worker

        (tmp_path / CHECKPOINT_NAME).write_text("not a checkpoint")

        result = run_exports_isolated(
            _export_worker,
            (TrainingJobSpec(policy="act"), str(tmp_path)),
            tmp_path,
            report=MagicMock(),
            should_stop=lambda: False,
        )

        assert result.finished
        assert result.error is not None and result.error.startswith("could not reload the checkpoint")


class TestExportReport:
    def test_a_model_without_a_report_has_none(self, tmp_path: Path) -> None:
        assert ExportReport.load(tmp_path) is None
        assert export_completion_message(tmp_path) == "Training finished"

    def test_an_unreadable_report_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / EXPORT_REPORT_NAME).write_text("{not json")

        assert ExportReport.load(tmp_path) is None

    def test_abort_fails_every_pending_backend(self) -> None:
        report = ExportReport()
        report.plan(["torch", "openvino"])
        report.record("torch", ExportStatus.SUCCEEDED)

        report.abort("trainer process exited unexpectedly (-11)")

        assert report.finished and report.error is None
        assert report.failures[0].backend == "openvino"
        assert report.failures[0].error == "trainer process exited unexpectedly (-11)"

    def test_abort_before_planning_records_a_report_wide_error(self) -> None:
        report = ExportReport()

        report.abort("boom")

        assert report.error == "boom"
        assert report.summary() == "export failed: boom"

    def test_the_completion_message_names_failed_backends(self, tmp_path: Path) -> None:
        report = ExportReport(planned=True, finished=True)
        report.record("torch", ExportStatus.SUCCEEDED)
        report.record("openvino", ExportStatus.FAILED, "crashed")
        report.record("onnx", ExportStatus.FAILED, "crashed")
        report.save(tmp_path)

        assert export_completion_message(tmp_path) == "Training finished; OpenVINO, ONNX export failed"
        assert export_completion_message(tmp_path, "Model uploaded").startswith("Model uploaded; ")

    def test_discard_keeps_succeeded_and_skipped_exports(self, tmp_path: Path) -> None:
        for name in ("torch", "openvino", "executorch"):
            _write_export(str(tmp_path), name)
        report = ExportReport(planned=True)
        report.record("torch", ExportStatus.SUCCEEDED)
        report.record("openvino", ExportStatus.FAILED, "x")
        report.record("executorch", ExportStatus.SKIPPED, "missing")

        discard_unfinished_exports(tmp_path, report)

        assert sorted(path.name for path in (tmp_path / EXPORTS_DIRNAME).iterdir()) == ["executorch", "torch"]
