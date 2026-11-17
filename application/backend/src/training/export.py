# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Crash-isolated export and the per-backend report it leaves beside a model.

Export runs after training has already produced and published its checkpoint,
so its failures must never cost the checkpoint. Python exceptions are easy to
contain, but export backends also run native code (OpenVINO's TorchScript
frontend, ONNX, ExecuTorch) that can take the whole process down with a signal.
:func:`run_exports_isolated` therefore runs the exports in a child process,
records each backend's outcome in :data:`EXPORT_REPORT_NAME` as it goes, and
when the child dies mid-backend it marks that backend failed, discards its
partial output, and starts a fresh child for the backends still pending.

The report is the hand-off to whoever delivers the model: it travels in the
model directory (and therefore in the trainer's model archive), and Studio reads
it to report failed backends on the job and model instead of failing the run.

This module deliberately imports neither torch nor ``physicalai`` so Studio can
read reports in a recording-only install.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
import signal
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.connection import Connection

    from physicalai.train.callbacks import ReportFn, StopFn

logger = logging.getLogger(__name__)

EXPORTS_DIRNAME = "exports"
"""Subdirectory of a model directory holding one directory per export backend."""

EXPORT_REPORT_NAME = "export_report.json"
"""Per-backend export outcomes, written beside the checkpoint in a model directory."""

_POLL_INTERVAL_S = 0.5
_TERMINATE_GRACE_S = 10.0

_BACKEND_LABELS = {"openvino": "OpenVINO", "onnx": "ONNX", "torch": "Torch", "executorch": "ExecuTorch"}


def backend_label(name: str) -> str:
    """Return the display name of an export backend, e.g. ``"openvino"`` -> ``"OpenVINO"``."""
    return _BACKEND_LABELS.get(name, name)


class ExportStatus(StrEnum):
    """Outcome of exporting a model to one backend."""

    PENDING = "pending"
    """Planned but not finished. A report left with pending backends means the export was interrupted."""
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Not attempted because an optional dependency of the backend is not installed."""


class BackendExportResult(BaseModel):
    """What happened when exporting to one backend."""

    backend: str
    status: ExportStatus
    error: str | None = None


class ExportReport(BaseModel):
    """Per-backend outcome of exporting one trained model.

    Written to :data:`EXPORT_REPORT_NAME` as soon as training has published its
    checkpoint and updated after every backend, so the file on disk always says
    how far export got, even if the process writing it dies.
    """

    model_config = ConfigDict(extra="ignore")

    planned: bool = Field(default=False, description="Whether the backends to export have been decided.")
    finished: bool = Field(default=False, description="Whether export ran to the end (successfully or not).")
    error: str | None = Field(default=None, description="Failure that prevented exporting any backend.")
    backends: list[BackendExportResult] = Field(default_factory=list)

    @classmethod
    def load(cls, model_dir: Path | str) -> ExportReport | None:
        """Read the report from a model directory.

        Returns:
            The report, or ``None`` when the directory has none (a model trained
            before reports existed) or it cannot be parsed.
        """
        path = Path(model_dir) / EXPORT_REPORT_NAME
        try:
            return cls.model_validate_json(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValidationError, ValueError):
            logger.warning("Ignoring unreadable export report %s", path, exc_info=True)
            return None

    def save(self, model_dir: Path | str) -> None:
        """Write the report into a model directory atomically."""
        path = Path(model_dir) / EXPORT_REPORT_NAME
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.replace(path)

    def plan(self, names: Sequence[str]) -> None:
        """Record the backends about to be exported; later plans are ignored."""
        if self.planned:
            return
        self.planned = True
        self.backends = [BackendExportResult(backend=name, status=ExportStatus.PENDING) for name in names]

    def record(self, name: str, status: ExportStatus, error: str | None = None) -> None:
        """Set one backend's outcome."""
        for result in self.backends:
            if result.backend == name:
                result.status, result.error = status, error
                return
        self.backends.append(BackendExportResult(backend=name, status=status, error=error))

    @property
    def pending(self) -> list[str]:
        """Backends planned but not finished."""
        return [result.backend for result in self.backends if result.status is ExportStatus.PENDING]

    @property
    def failures(self) -> list[BackendExportResult]:
        """Backends whose export failed."""
        return [result for result in self.backends if result.status is ExportStatus.FAILED]

    def abort(self, reason: str) -> None:
        """Finish an interrupted export: every pending backend failed for ``reason``."""
        for name in self.pending:
            self.record(name, ExportStatus.FAILED, reason)
        if not self.planned and self.error is None:
            self.error = reason
        self.finished = True

    def summary(self) -> str | None:
        """One line describing what failed, or ``None`` when nothing did."""
        if self.error is not None:
            return f"export failed: {self.error}"
        failed = [backend_label(result.backend) for result in self.failures]
        if not failed:
            return None
        return f"{', '.join(failed)} export failed"


def discard_unfinished_exports(model_dir: Path | str, report: ExportReport) -> None:
    """Remove export directories of backends that failed or never finished.

    A backend that failed or crashed part-way can leave a half-written
    directory behind (an OpenVINO ``.xml`` without its ``.bin``); a non-empty
    directory reads as an available backend, so it must not survive.
    """
    exports_dir = Path(model_dir) / EXPORTS_DIRNAME
    for result in report.backends:
        if result.status in {ExportStatus.FAILED, ExportStatus.PENDING}:
            shutil.rmtree(exports_dir / result.backend, ignore_errors=True)


def export_completion_message(model_dir: Path | str, prefix: str = "Training finished") -> str:
    """Return a job's completion message, naming any export that failed.

    Args:
        model_dir: The finished model's directory.
        prefix: The message for a run whose exports all succeeded.

    Returns:
        ``prefix``, followed by the export report's summary when a backend failed.
    """
    report = ExportReport.load(model_dir)
    summary = report.summary() if report is not None else None
    return prefix if summary is None else f"{prefix}; {summary}"


ExportWorker = Callable[..., None]
"""Child-process entry point, called as ``worker(conn, skip, *worker_args)``.

It narrates progress over ``conn`` and must send, in order: ``("plan",
[backend, ...])`` once the backends are known; ``("start", backend)`` and then
``("result", backend, status, error, details)`` for each backend not in
``skip``; and finally ``("done",)``. A failure that prevents planning is sent as
``("error", message, details)``. ``details`` is a traceback or other log-only
text.
"""


def run_exports_isolated(
    worker: ExportWorker,
    worker_args: tuple[Any, ...],
    model_dir: Path | str,
    *,
    report: ReportFn,
    should_stop: StopFn,
) -> ExportReport:
    """Export a model in a child process so a native crash cannot affect the caller.

    Never raises for an export failure: every outcome, including the child
    dying with a signal, ends up in the returned report (also saved to
    ``model_dir``). A crash while exporting one backend fails only that
    backend; the remaining ones are exported by a fresh child.

    Args:
        worker: Child entry point following the :data:`ExportWorker` protocol.
            Must be importable by a spawned interpreter (a module-level function).
        worker_args: Picklable arguments for ``worker``.
        model_dir: Model directory holding the checkpoint; receives the report.
        report: Telemetry sink, called with ``(progress, message, extra_info)``.
        should_stop: Cooperative cancellation probe. When it returns True the
            child is terminated and the report is returned unfinished.

    Returns:
        The export report.
    """
    model_dir = Path(model_dir)
    export_report = ExportReport()
    export_report.save(model_dir)
    try:
        # Each attempt resolves at least one backend or gives up, so a bound
        # only guards against a child that keeps dying between backends.
        attempts = 0
        while not export_report.finished and not should_stop():
            attempts += 1
            if attempts > len(export_report.backends) + 2:
                export_report.abort("export process kept exiting before finishing")
                break
            _run_attempt(worker, worker_args, model_dir, export_report, report=report, should_stop=should_stop)
    except Exception as exc:
        # Supervision itself failing (e.g. the child cannot be spawned) is still
        # an export failure, not a training failure.
        logger.exception("Export could not run")
        export_report.abort(f"export could not run: {exc}")
    discard_unfinished_exports(model_dir, export_report)
    export_report.save(model_dir)
    return export_report


def _run_attempt(
    worker: ExportWorker,
    worker_args: tuple[Any, ...],
    model_dir: Path,
    export_report: ExportReport,
    *,
    report: ReportFn,
    should_stop: StopFn,
) -> None:
    """Run one export child until it finishes, dies, or is told to stop; update ``export_report``."""
    skip = [result.backend for result in export_report.backends if result.status is not ExportStatus.PENDING]
    process, conn = _launch(worker, (skip, *worker_args))
    current: str | None = None
    try:
        while True:
            if should_stop():
                logger.info("Export canceled")
                _stop(process)
                return
            if not conn.poll(_POLL_INTERVAL_S):
                if process.is_alive():
                    continue
                if not conn.poll():
                    break
            try:
                message = conn.recv()
            except EOFError:
                break
            current = _apply_message(message, current, model_dir, export_report, report)
            export_report.save(model_dir)
            if export_report.finished:
                return
    finally:
        conn.close()
        process.join(_TERMINATE_GRACE_S)
        if process.is_alive():
            _stop(process)

    # The child went away without saying it was done.
    reason = f"export process {_describe_exit(process.exitcode)}"
    if current is not None:
        logger.error("Export to %s crashed: %s", current, reason)
        export_report.record(current, ExportStatus.FAILED, f"{reason} while exporting")
        discard_unfinished_exports(model_dir, export_report)
    elif not export_report.planned:
        logger.error("Export crashed before exporting any backend: %s", reason)
        export_report.abort(f"{reason} before exporting")
    else:
        logger.warning("Export %s between backends; retrying the remaining ones", reason)
    export_report.save(model_dir)


def _apply_message(
    message: tuple[Any, ...],
    current: str | None,
    model_dir: Path,
    export_report: ExportReport,
    report: ReportFn,
) -> str | None:
    """Fold one worker message into the report.

    Returns:
        The backend now being exported, or ``None`` between backends.
    """
    kind = message[0]
    if kind == "plan":
        export_report.plan(message[1])
    elif kind == "start":
        current = message[1]
        logger.info("Exporting model to %s format", current)
        report(99, f"Exporting to {current} format", {})
    elif kind == "result":
        _, name, status, error, details = message
        status = ExportStatus(status)
        if status is ExportStatus.FAILED:
            logger.error("Failed exporting model to %s format: %s\n%s", name, error, details or "")
        elif status is ExportStatus.SKIPPED:
            logger.warning("Skipping %s export: %s", name, error)
        export_report.record(name, status, error)
        if status is ExportStatus.FAILED:
            discard_unfinished_exports(model_dir, export_report)
        current = None
    elif kind == "error":
        _, error, details = message
        logger.error("Export failed: %s\n%s", error, details or "")
        export_report.abort(error)
    elif kind == "done":
        export_report.finished = True
    else:
        logger.warning("Ignoring unknown export worker message %r", kind)
    return current


def _launch(worker: ExportWorker, args: tuple[Any, ...]) -> tuple[Any, Connection]:
    """Start ``worker(conn, *args)`` in a spawned child.

    Spawned rather than forked: the parent has just trained on an accelerator,
    and a forked CUDA/XPU context is unusable in the child.

    Returns:
        The process handle and the receiving end of its message pipe.
    """
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(sender, *args), name="ModelExport", daemon=True)
    process.start()
    # Drop the parent's copy of the sending end so the pipe reports EOF once the child exits.
    sender.close()
    return process, receiver


def _stop(process: Any) -> None:
    """Terminate a child, escalating to SIGKILL if it ignores SIGTERM."""
    process.terminate()
    process.join(_TERMINATE_GRACE_S)
    if process.is_alive():
        process.kill()
        process.join()


def _describe_exit(exitcode: int | None) -> str:
    """Describe how a child process ended, e.g. ``"crashed (SIGSEGV)"``."""
    if exitcode is None:
        return "did not report an exit status"
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            name = f"signal {-exitcode}"
        hint = ", likely out of memory" if -exitcode == getattr(signal, "SIGKILL", None) else ""
        return f"crashed ({name}{hint})"
    return f"exited with code {exitcode}"
