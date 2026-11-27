# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The multi-GPU supervisor with stand-in ranks: event relay, failures and cancellation, without GPUs."""

import json
import os
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Fixed interpreter and test scripts.
import sys
import threading
import time
from pathlib import Path

import pytest

from physicalai_starvla_policy_plugin._worker import starvla_distributed as distributed

# A stand-in for torchrun: writes rank 0's events (one of them in two pieces) and checks its inputs.
RANKS = """
import json, os, sys, time
from pathlib import Path
directory = Path(os.environ["PHYSICALAI_STARVLA_DISTRIBUTED_DIR"])
assert "PHYSICALAI_WORKER_PROTOCOL_FD" not in os.environ
assert sys.stdin.read() == ""
with (directory / "events.jsonl").open("a") as events:
    events.write(json.dumps({"type": "progress", "step": 1}) + "\\n")
    events.flush()
    events.write('{"type": "progress",')
    events.flush()
    time.sleep(0.3)
    events.write(' "step": 2}\\n')
mode = sys.argv[1]
if mode == "fail":
    sys.exit(3)
if mode in {"cooperate", "stubborn"}:
    if mode == "stubborn":
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        if mode == "cooperate" and (directory / "cancel").exists():
            sys.exit(1)
        time.sleep(0.05)
"""


def _run(mode: str, *, cancel_after: float | None = None, **grace: float) -> tuple[bool, list[dict]]:
    sent: list[dict] = []
    cancelled = threading.Event()
    if cancel_after is not None:
        threading.Timer(cancel_after, cancelled.set).start()
    result = distributed.supervise(
        [sys.executable, "-c", RANKS, mode],
        send=sent.append,
        cancelled=cancelled,
        **grace,
    )
    return result, sent


def test_rank_zero_events_are_relayed_in_order_even_when_written_in_pieces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSICALAI_WORKER_PROTOCOL_FD", "2")  # The ranks must not get the engine's protocol stream.

    cancelled, sent = _run("succeed")

    assert cancelled is False
    assert sent == [{"type": "progress", "step": 1}, {"type": "progress", "step": 2}]


def test_a_failing_rank_fails_the_run() -> None:
    with pytest.raises(RuntimeError, match="status 3"):
        _run("fail")


def test_ranks_stop_together_when_the_host_cancels() -> None:
    started = time.monotonic()

    cancelled, sent = _run("cooperate", cancel_after=0.5, cancel_grace=30.0)

    assert cancelled is True
    assert sent[-1] == {"type": "progress", "step": 2}
    assert time.monotonic() - started < 10  # Stopped by the flag, not by the grace period.


def test_ranks_that_ignore_the_cancellation_are_killed() -> None:
    started = time.monotonic()

    cancelled, _ = _run("stubborn", cancel_after=0.5, cancel_grace=0.2, terminate_grace=0.5)

    assert cancelled is True
    assert time.monotonic() - started < 10


def test_only_rank_zero_writes_events(tmp_path: Path) -> None:
    distributed.FileChannel(tmp_path, rank=1).send({"type": "progress"})
    assert not (tmp_path / distributed.EVENTS).exists()
    assert distributed.FileChannel(tmp_path, rank=1).enabled is False

    distributed.FileChannel(tmp_path, rank=0).send({"type": "progress", "step": 1})

    lines = (tmp_path / distributed.EVENTS).read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"type": "progress", "step": 1}]


def test_without_a_process_group_the_cancellation_is_the_flag(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    cancellation = distributed.SharedCancellation(tmp_path)
    assert cancellation.is_set() is False
    (tmp_path / distributed.CANCEL).touch()
    assert cancellation.is_set() is True


def test_torchrun_starts_one_rank_per_gpu_with_this_python(tmp_path: Path) -> None:
    command = distributed.torchrun_command(tmp_path / "worker.py", ["--config_yaml", "c.yaml", "a=1"], 2)

    assert command[:5] == [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    assert command[5:] == [str(tmp_path / "worker.py"), "--config_yaml", "c.yaml", "a=1"]


# A parent that starts a rank in a session of its own (as torchrun does), then waits to be killed.
PARENT = """
import subprocess, sys, time
rank = subprocess.Popen([sys.executable, "-c", sys.argv[1]], start_new_session=True)
print(rank.pid, flush=True)
time.sleep(60)
"""
RANK = """
import sys, time
sys.path.insert(0, {worker!r})
from starvla_distributed import die_with_parent
die_with_parent()
time.sleep(60)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Parent-death signals are Linux-only.")
def test_a_rank_dies_with_torchrun_even_in_a_session_of_its_own() -> None:
    worker = str(Path(distributed.__file__).parent)
    parent = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]  # No untrusted arguments.
        [sys.executable, "-c", PARENT, RANK.format(worker=worker)],
        stdout=subprocess.PIPE,
        text=True,
    )
    rank = int(parent.stdout.readline())
    time.sleep(1.0)  # Let the rank make its request.
    os.kill(rank, 0)  # Running.

    parent.send_signal(signal.SIGKILL)
    parent.wait()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(rank, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(rank, signal.SIGKILL)
    pytest.fail("The rank outlived its parent.")
