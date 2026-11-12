# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run upstream StarVLA training with native arguments and engine protocol events."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

from starvla_worker_common import (
    CancelledError,
    EventChannel,
    disable_remote_code,
    finite_metrics,
    identity,
    start_cancel_listener,
)


def train(argv: list[str] | None = None) -> tuple[Path, Path]:  # ruff: ignore[too-many-statements]  # Native CLI lifecycle.
    """Run the native trainer, preserving its normalization and checkpoint format.

    Returns:
        Run directory and final checkpoint.

    Raises:
        ValueError: If the framework, run name, or backbone is unsupported.
        FileNotFoundError: If training does not produce its final checkpoint.
    """
    parser = argparse.ArgumentParser(description="StarVLA QwenGR00T training (native OmegaConf KEY=VALUE overrides)")
    parser.add_argument("--config_yaml", required=True, help="Local StarVLA training YAML")
    args, overrides = parser.parse_known_args(argv)
    if any("=" not in token for token in overrides):
        parser.error("Overrides must use KEY=VALUE or --KEY=VALUE tokens.")
    channel = EventChannel()
    channel.send({"type": "hello", "protocol": 1, **identity()})
    cancelled = start_cancel_listener()
    os.environ.setdefault("STARVLA_DISABLE_DEEPSPEED", "1")
    # Model code and weights must already be local; no mutable remote revisions.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    disable_remote_code()
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.training import train_starvla as native

    cfg = OmegaConf.merge(
        OmegaConf.load(args.config_yaml),
        OmegaConf.from_dotlist([token.removeprefix("--") for token in overrides]),
    )
    cfg = apply_config_compat(cfg)
    if cfg.framework.name != "QwenGR00T":
        msg = "This plugin currently supports framework.name=QwenGR00T only."
        raise ValueError(msg)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", str(cfg.run_id)):
        msg = "run_id must use 1-64 letters, digits, underscores, or hyphens."
        raise ValueError(msg)
    backbone = Path(cfg.framework.qwenvl.base_vlm).expanduser().resolve(strict=True)
    if not (backbone / "config.json").is_file():
        msg = "framework.qwenvl.base_vlm must point to a local materialized backbone."
        raise ValueError(msg)
    if (
        "Qwen3-VL" not in str(backbone)
        or json.loads((backbone / "config.json").read_text()).get("model_type") != "qwen3_vl"
    ):
        msg = "Use a local Qwen3-VL backbone with Qwen3-VL in its directory name (upstream VLM dispatch)."
        raise ValueError(msg)
    cfg.framework.qwenvl.base_vlm = str(backbone)
    cfg.run_root_dir = str(Path(cfg.run_root_dir).expanduser().resolve())
    cfg.trainer.save_format = "safetensors"
    cfg.config_yaml = str(Path(args.config_yaml).resolve())
    output = Path(cfg.run_root_dir).expanduser().resolve() / str(cfg.run_id)
    step_method = native.VLATrainer._train_step
    log_method = native.VLATrainer._log_metrics
    state_method = native._get_state_dict

    def state_dict(accelerator: object, model: object) -> dict[str, Tensor]:
        return _saveable_state_dict(state_method(accelerator, model))

    def step(trainer: native.VLATrainer, batch_vla: list[dict[str, Any]], batch_vlm: object = None) -> dict[str, float]:
        if cancelled.is_set():
            raise CancelledError
        return step_method(trainer, batch_vla, batch_vlm)

    def log(trainer: native.VLATrainer, metrics: dict[str, Any]) -> None:
        log_method(trainer, metrics)
        channel.send({
            "type": "progress",
            "step": trainer.completed_steps,
            "total": int(trainer.config.trainer.max_train_steps),
            "metrics": finite_metrics(metrics),
        })

    native.VLATrainer._train_step = step
    native.VLATrainer._log_metrics = log
    native._get_state_dict = state_dict
    try:
        native.main(cfg)
    finally:
        native.VLATrainer._train_step = step_method
        native.VLATrainer._log_metrics = log_method
        native._get_state_dict = state_method
    checkpoint = output / "final_model" / "model.safetensors"
    if not checkpoint.is_file():
        msg = "StarVLA finished without a final safetensors checkpoint."
        raise FileNotFoundError(msg)
    shutil.copy2(output / "config.full.yaml", output / "config.yaml")
    stats = json.loads((output / "dataset_statistics.json").read_text())
    record = {
        "environment": json.loads(os.environ["PHYSICALAI_STARVLA_PROVENANCE"]),
        "backbone_path": str(backbone),
        "unnorm_key": next(iter(stats)) if len(stats) == 1 else None,
        "dataset_root": str(cfg.datasets.vla_data.data_root_dir),
        "data_mix": str(cfg.datasets.vla_data.data_mix),
    }
    (output / "physicalai_run.json").write_text(json.dumps(record, indent=2) + "\n")
    return output, checkpoint


def _saveable_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    # Upstream save_file rejects Qwen3's tied embedding/head storage; retain both state_dict keys.
    seen = set()
    result = {}
    for key, tensor in state.items():
        pointer = tensor.untyped_storage().data_ptr()
        result[key] = tensor.contiguous().clone() if pointer in seen else tensor.contiguous()
        seen.add(pointer)
    return result


def main() -> int:
    """Emit one terminal event and retain a nonzero status on failures.

    Returns:
        Zero on success, one on cancellation.
    """
    channel = EventChannel()
    try:
        output, checkpoint = train()
    except CancelledError:
        channel.send({"type": "cancelled", "step": None})
        return 1
    except Exception as exc:
        channel.send({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        raise
    channel.send({"type": "result", "output_dir": str(output), "checkpoint": str(checkpoint)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
