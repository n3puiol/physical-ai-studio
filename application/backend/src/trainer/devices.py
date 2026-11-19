# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hardware discovery for the trainer service.

Reports the compute devices this trainer can use for training so the studio
backend can surface the real (often GPU) hardware instead of the studio host's
local CPU. Mirrors the studio backend's device enumeration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from trainer.schemas import DeviceInfo
from trainer.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Sequence


def gpu_busy(device_type: str, index: int) -> bool | None:
    """Best-effort host-wide GPU memory check; None means telemetry unavailable."""
    try:
        import torch

        backend = torch.cuda if device_type == "cuda" else torch.xpu if device_type == "xpu" else None
        if backend is None:
            return None
        free, total = backend.mem_get_info(index)
        # Ignore this trainer's cached allocations (single-job mode trains in-process).
        # ponytail: Memory is a heuristic, not a reservation. Tune for desktop/driver overhead.
        return total - free - backend.memory_reserved(index) >= get_settings().gpu_busy_memory_mb * 1024 * 1024
    except Exception as exc:
        logger.warning("Cannot read {} GPU {} memory: {}", device_type, index, exc)
        return None


def get_training_devices() -> list[DeviceInfo]:
    """Enumerate Intel XPU and NVIDIA CUDA devices available for training."""
    devices: list[DeviceInfo] = []

    try:
        import torch
    except Exception as exc:
        logger.warning("torch unavailable; no accelerator devices reported: {}", exc)
        return devices

    try:
        if torch.xpu.is_available():
            for device_idx in range(torch.xpu.device_count()):
                xpu_props = torch.xpu.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type="xpu",
                        name=xpu_props.name,
                        memory=xpu_props.total_memory,
                        index=device_idx,
                        busy=gpu_busy("xpu", device_idx),
                    ),
                )
    except Exception as exc:
        logger.warning("Failed to enumerate XPU devices: {}", exc)

    try:
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                cuda_props = torch.cuda.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type="cuda",
                        name=cuda_props.name,
                        memory=cuda_props.total_memory,
                        index=device_idx,
                        busy=gpu_busy("cuda", device_idx),
                    ),
                )
    except Exception as exc:
        logger.warning("Failed to enumerate CUDA devices: {}", exc)

    return devices


def visible_devices_for(indices: Sequence[int], parent: str | None) -> str:
    """Return the ``CUDA_VISIBLE_DEVICES`` value that exposes exactly ``indices`` to a child process.

    ``indices`` are the indices this trainer reports (torch's numbering in this
    process). When the trainer itself runs with ``CUDA_VISIBLE_DEVICES`` set,
    those indices point into that list, so they are mapped through it; its
    entries may be numbers or ``GPU-…``/``MIG-…`` UUIDs and are passed on as is.

    Args:
        indices: The devices the child may use, in the order it should number them.
        parent: This process's ``CUDA_VISIBLE_DEVICES``, or None when unset.

    Returns:
        A comma-separated device list for the child's environment.

    Raises:
        ValueError: If an index is outside the parent's visible devices.

    Example:
        >>> visible_devices_for([0, 2], None)
        '0,2'
        >>> visible_devices_for([0, 1], "4,5,6,7")
        '4,5'
    """
    if parent is None:
        return ",".join(str(index) for index in indices)
    entries = [entry.strip() for entry in parent.split(",") if entry.strip()]
    missing = [index for index in indices if not 0 <= index < len(entries)]
    if missing:
        msg = f"CUDA device {missing[0]} is not among this trainer's visible devices ({parent!r})"
        raise ValueError(msg)
    return ",".join(entries[index] for index in indices)
