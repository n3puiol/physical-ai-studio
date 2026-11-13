# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Try multi-GPU training (all_devices + rank-0 guards) on a CUDA host, without the trainer service.

Runs ``training.run_training_job`` exactly as the trainer would, one process per GPU
under ``torchrun``. Throwaway helper for the multi-GPU plan (Steps 2-3); not used by Studio.

All GPUs (DDP):
    uv run --no-sync torchrun --standalone --nproc_per_node=gpu scripts/try_multi_gpu.py \
        --dataset ~/datasets/my-dataset --out /tmp/mgpu-all

One GPU, for comparison:
    uv run --no-sync python scripts/try_multi_gpu.py --gpu 0 \
        --dataset ~/datasets/my-dataset --out /tmp/mgpu-one
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training import TrainingJobSpec, run_training_job

RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


def _log(message: str) -> None:
    print(f"[rank {RANK}/{WORLD_SIZE}] {message}", flush=True)


def main() -> None:
    """Train once with the given settings and list what ended up in the output dir."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Local LeRobot dataset root (has meta/, data/).")
    parser.add_argument("--out", type=Path, required=True, help="Output dir; replaced if it exists.")
    parser.add_argument("--gpu", type=int, default=None, help="Train on this one GPU instead of all of them.")
    parser.add_argument("--policy", default="act")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8, help="Per GPU under DDP.")
    parser.add_argument("--export", nargs="*", default=["torch"], help="Export backends; pass none to skip export.")
    args = parser.parse_args()

    spec = TrainingJobSpec(
        policy=args.policy,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device_type="cuda",
        device_index=args.gpu,
        all_devices=args.gpu is None,
        export_backends=args.export,
    )
    _log(f"spec: all_devices={spec.all_devices} device_index={spec.device_index} batch_size={spec.batch_size}")

    out = args.out.expanduser().resolve()
    # Every rank must agree on the cache dir; torchrun gives all ranks the same run id.
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
    cache = out.parent / f"{out.name}-cache-{run_id}"

    last_progress = -1

    def report(progress: int, message: str | None, extra: dict | None) -> None:
        # run_training_job wraps this with rank_zero_only: only rank 0 should ever print it.
        nonlocal last_progress
        if message or progress != last_progress:
            _log(f"report {progress:3d}% {message or ''} {extra or ''}")
            last_progress = progress

    start = time.monotonic()
    run_training_job(
        spec,
        dataset_root=args.dataset.expanduser().resolve(),
        output_dir=out,
        cache_dir=cache,
        report=report,
        should_stop=lambda: False,
    )
    _log(f"run_training_job returned after {time.monotonic() - start:.0f}s; output exists: {out.exists()}")
    if out.exists():
        for path in sorted(out.rglob("*")):
            if path.is_file():
                _log(f"  {path.relative_to(out)}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
