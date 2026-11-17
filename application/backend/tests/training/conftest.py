# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the training runner tests."""

from __future__ import annotations

import multiprocessing as mp
import threading
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


class _ThreadProcess:
    """Stand-in for a spawned export child that runs the worker on a thread.

    Lets runner tests patch the policy loader in-process; the real spawn path
    is covered separately in ``test_export.py``.
    """

    def __init__(self, target: Callable[..., None], args: tuple[Any, ...]) -> None:
        self.exitcode: int | None = None
        self._thread = threading.Thread(target=self._run, args=(target, args), daemon=True)

    def _run(self, target: Callable[..., None], args: tuple[Any, ...]) -> None:
        try:
            target(*args)
        except BaseException:
            self.exitcode = 1
            raise
        self.exitcode = 0

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def terminate(self) -> None:
        """Threads cannot be killed; the tests that use this never cancel mid-export."""

    kill = terminate


def _launch_in_thread(worker: Callable[..., None], args: tuple[Any, ...]) -> tuple[_ThreadProcess, Any]:
    receiver, sender = mp.Pipe(duplex=False)
    process = _ThreadProcess(worker, (sender, *args))
    process.start()
    return process, receiver


@pytest.fixture
def exports_in_thread():
    """Run the export worker on a thread of the test process instead of a spawned child."""
    with patch("training.export._launch", _launch_in_thread):
        yield
