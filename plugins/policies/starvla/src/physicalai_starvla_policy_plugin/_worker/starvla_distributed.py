# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel StarVLA training on every visible GPU, supervised by the engine's worker.

The engine starts one worker; it keeps the engine protocol (handshake, events, cancellation) and
launches ``torchrun`` with one rank per GPU. Each rank runs StarVLA's own trainer, which uses
``accelerate``'s DistributedDataParallel under ``torchrun``. Rank 0 appends its progress events to
a file the supervisor relays; a cancellation is a flag file the ranks agree on at a step boundary,
so none of them waits forever for another. ``torchrun`` stays in the worker's process group, which
the engine (or isolate, if the host dies) kills as a whole. ``torchrun`` starts each rank in a
session of its own, out of that group's reach, so every rank asks Linux to kill it when ``torchrun``
dies.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Fixed torchrun argv, shell disabled.  # nosec B404
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from threading import Event

# Set in each rank's environment: the folder holding rank 0's events and the cancellation flag.
DISTRIBUTED_DIR_VARIABLE = "PHYSICALAI_STARVLA_DISTRIBUTED_DIR"
EVENTS = "events.jsonl"
CANCEL = "cancel"
# The engine terminates a worker 5 s after cancelling it; stop the ranks well before that.
CANCEL_GRACE_SECONDS = 3.0
TERMINATE_GRACE_SECONDS = 10.0
_POLL_SECONDS = 0.1
_PR_SET_PDEATHSIG = 1


class FileChannel:
    """Rank 0's events, appended to the file the supervisor relays; other ranks send nothing."""

    def __init__(self, directory: Path, rank: int) -> None:
        """Write events only on rank 0."""
        self._path = directory / EVENTS if rank == 0 else None

    @property
    def enabled(self) -> bool:
        """Whether this rank reports events."""
        return self._path is not None

    def send(self, message: dict[str, Any]) -> None:
        """Append one JSON event line."""
        if self._path is None:
            return
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(message, allow_nan=False) + "\n")


class SharedCancellation:
    """A cancellation every rank sees at the same training step.

    Ranks may notice the supervisor's flag file at different moments; stopping on different steps
    would leave the others waiting in a collective. Each check therefore takes the maximum of the
    ranks' observations, so all of them stop together.
    """

    def __init__(self, directory: Path) -> None:
        """Watch the supervisor's flag file."""
        self._flag = directory / CANCEL

    def is_set(self) -> bool:
        """Whether any rank has seen the cancellation flag; a collective call on every rank.

        Returns:
            True on every rank once one of them sees the flag.
        """
        import torch
        from torch import distributed

        seen = self._flag.exists()
        if not distributed.is_available() or not distributed.is_initialized():
            return seen
        value = torch.tensor([int(seen)], device=torch.device("cuda", torch.cuda.current_device()))
        distributed.all_reduce(value, op=distributed.ReduceOp.MAX)
        return bool(value.item())


def die_with_parent() -> None:
    """Have Linux kill this process with SIGKILL when its parent, ``torchrun``, dies.

    ``torchrun`` runs each rank in a new session, so killing the worker's process group stops
    ``torchrun`` but not the ranks; without this, a rank outlives a killed worker and keeps its GPU
    memory. Other platforms are unaffected (the engine's managed workers are POSIX-only, and Linux is
    the only POSIX platform the worker environments are locked for).

    Raises:
        OSError: If the kernel refuses the request.
    """
    if not sys.platform.startswith("linux"):
        return
    import ctypes

    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent:  # The parent died before the request took effect.
        os._exit(1)


def rank_hooks(directory: Path) -> tuple[FileChannel, SharedCancellation]:
    """Return the event channel and cancellation of the ``torchrun`` rank this process is.

    Returns:
        Rank 0's event file (disabled on other ranks) and the shared cancellation.
    """
    return FileChannel(directory, int(os.environ.get("RANK", "0"))), SharedCancellation(directory)


def leave_process_group() -> None:
    """Leave the ranks' process group before a cancelled rank exits, as upstream does after training."""
    from torch import distributed

    if distributed.is_available() and distributed.is_initialized():
        distributed.destroy_process_group()


def visible_gpus() -> int:
    """Count the CUDA devices this worker sees (its ``CUDA_VISIBLE_DEVICES``).

    Returns:
        The number of visible CUDA devices.
    """
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def torchrun_command(worker: Path, arguments: Sequence[str], processes: int) -> list[str]:
    """Return the ``torchrun`` command line starting one rank per GPU on this machine.

    Returns:
        The argv, run with this worker environment's Python.
    """
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={processes}",
        str(worker),
        *arguments,
    ]


def supervise(
    command: Sequence[str],
    *,
    send: Callable[[dict[str, Any]], None],
    cancelled: Event,
    cancel_grace: float = CANCEL_GRACE_SECONDS,
    terminate_grace: float = TERMINATE_GRACE_SECONDS,
) -> bool:
    """Run the ranks, relay rank 0's events, and pass on a cancellation.

    The ranks' output goes to this worker's stdout, never to the engine's protocol stream, and they
    do not read its stdin.

    Returns:
        True if the host cancelled training, whatever the ranks' exit status.

    Raises:
        RuntimeError: If a rank failed.
    """
    with tempfile.TemporaryDirectory(prefix="starvla-ranks-") as temporary:
        directory = Path(temporary)
        events = directory / EVENTS
        events.touch()
        environment = {key: value for key, value in os.environ.items() if key != "PHYSICALAI_WORKER_PROTOCOL_FD"}
        environment[DISTRIBUTED_DIR_VARIABLE] = str(directory)
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]  # No shell.  # nosec B603
            list(command),
            env=environment,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            shell=False,
        )
        requested_at: float | None = None
        terminated_at: float | None = None
        try:
            with events.open("rb") as stream:
                pending = b""
                while True:
                    finished = process.poll() is not None
                    pending = _relay(stream, pending, send)
                    if finished:
                        break
                    now = time.monotonic()
                    if requested_at is None and cancelled.is_set():
                        (directory / CANCEL).touch()
                        requested_at = now
                    if requested_at is not None and terminated_at is None and now - requested_at >= cancel_grace:
                        process.send_signal(signal.SIGTERM)  # torchrun stops its ranks.
                        terminated_at = now
                    elif terminated_at is not None and now - terminated_at >= terminate_grace:
                        process.kill()
                    time.sleep(_POLL_SECONDS)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    if requested_at is not None:
        return True
    if process.returncode != 0:
        msg = f"Distributed training exited with status {process.returncode}; the rank logs above give the cause."
        raise RuntimeError(msg)
    return False


def _relay(stream: Any, pending: bytes, send: Callable[[dict[str, Any]], None]) -> bytes:  # ruff: ignore[any-type]
    """Forward complete event lines; keep a partially written one for the next poll.

    Returns:
        The incomplete tail.
    """
    pending += stream.read()
    *lines, pending = pending.split(b"\n")
    for line in lines:
        if line.strip():
            send(json.loads(line))
    return pending
