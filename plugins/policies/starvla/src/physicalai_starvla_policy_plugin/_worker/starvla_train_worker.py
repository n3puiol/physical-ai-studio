# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run upstream StarVLA training with native arguments and engine protocol events."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor

# Isolated workers run without their folder on sys.path; the sibling modules live there.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from starvla_distributed import (
    DISTRIBUTED_DIR_VARIABLE,
    die_with_parent,
    leave_process_group,
    rank_hooks,
    supervise,
    torchrun_command,
    visible_gpus,
)
from starvla_worker_common import (
    BACKBONE_VARIABLE,
    BACKBONES,
    DEFAULT_BACKBONE,
    WORLD_MODEL_FRAMEWORKS,
    WORLD_MODELS,
    CancelledError,
    EventChannel,
    WorkerHandshakeError,
    dispatched_backbone,
    dispatched_world_model,
    finite_metrics,
    identity,
    restrict_remote_code,
    start_cancel_listener,
    use_fast_tokenizer,
    wait_for_host_acknowledgement,
)

_MIN_DATA_PARALLEL_GPUS = 2


def train(argv: list[str] | None = None) -> tuple[Path, Path]:
    """Run the native trainer, preserving its normalization and checkpoint format.

    With ``--all_devices`` this worker supervises one data-parallel ``torchrun`` rank per visible
    GPU; each rank runs this function again, as rank, without the engine protocol.

    Returns:
        Run directory and final checkpoint.

    Raises:
        ValueError: If ``--all_devices`` is given for a co-training config.
        FileNotFoundError: If training does not produce its final checkpoint.
    """
    parser = argparse.ArgumentParser(description="StarVLA training (native OmegaConf KEY=VALUE overrides)")
    parser.add_argument("--config_yaml", required=True, help="Local StarVLA training YAML")
    parser.add_argument(
        "--all_devices",
        action="store_true",
        help="Train on every visible GPU, one data-parallel process per GPU",
    )
    args, overrides = parser.parse_known_args(argv)
    if any("=" not in token for token in overrides):
        parser.error("Overrides must use KEY=VALUE or --KEY=VALUE tokens.")
    rank_directory = os.environ.get(DISTRIBUTED_DIR_VARIABLE)
    if rank_directory is not None:
        die_with_parent()
    channel, cancelled = _start(rank_directory)
    cfg, backbone, cotraining = _load_config(args.config_yaml, overrides)
    output = Path(cfg.run_root_dir) / str(cfg.run_id)
    checkpoint = output / "final_model" / "model.safetensors"
    if rank_directory is not None:  # A torchrun rank: the supervisor reports the run.
        _train_natively(cfg, channel, cancelled, cotraining=cotraining, data_parallel=True)
        return output, checkpoint
    if args.all_devices:
        if cotraining:
            msg = (
                "Co-training runs on one GPU: StarVLA's co-training step runs two backward passes per step, which "
                "DistributedDataParallel does not support; upstream co-trains on several GPUs with DeepSpeed."
            )
            raise ValueError(msg)
        _train_on_all_gpus(sys.argv[1:] if argv is None else argv, channel, cancelled)
    else:
        _train_natively(cfg, channel, cancelled, cotraining=cotraining)
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


def _train_on_all_gpus(arguments: list[str], channel: EventChannel, cancelled: Any) -> None:  # ruff: ignore[any-type]
    """Supervise one torchrun rank per visible GPU, each running this worker as a rank.

    Raises:
        ValueError: If fewer than two GPUs are visible.
        CancelledError: If the host cancels training.
    """
    processes = visible_gpus()
    if processes < _MIN_DATA_PARALLEL_GPUS:
        msg = f"--all_devices trains on every visible GPU and needs at least two; this worker sees {processes}."
        raise ValueError(msg)
    command = torchrun_command(
        Path(__file__).resolve(),
        [item for item in arguments if item != "--all_devices"],
        processes,
    )
    if supervise(command, send=channel.send, cancelled=cancelled):
        raise CancelledError


def _start(rank_directory: str | None) -> tuple[Any, Any]:
    """Open the engine protocol, or a rank's channel and cancellation under a supervising worker.

    Returns:
        The event channel and the cancellation to poll.

    Raises:
        CancelledError: If the host cancels before accepting the handshake.
        WorkerHandshakeError: If the host does not accept the handshake.
    """
    if rank_directory is not None:
        return rank_hooks(Path(rank_directory))
    channel = EventChannel()
    channel.send({"type": "hello", "protocol": 1, **identity()})
    if channel.enabled:
        # The host checks the version first; a rejected or cancelled worker never starts training.
        answer = wait_for_host_acknowledgement()
        if answer == "cancel":
            raise CancelledError
        if answer != "ack":
            msg = "The host did not accept the handshake."
            raise WorkerHandshakeError(msg)
    return channel, start_cancel_listener()


def _load_config(config_yaml: str, overrides: list[str]) -> tuple[Any, Path, bool]:
    """Load and check the training config, and prepare upstream for it in this process.

    Returns:
        The config, the resolved backbone folder, and whether it co-trains with VLM data.

    Raises:
        ValueError: If the framework, run name, or backbone is unsupported.
    """
    os.environ.setdefault("STARVLA_DISABLE_DEEPSPEED", "1")
    # Model code and weights must already be local; no mutable remote revisions.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    restrict_remote_code()
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.merge(
        OmegaConf.load(config_yaml),
        OmegaConf.from_dotlist([token.removeprefix("--") for token in overrides]),
    )
    cfg = apply_config_compat(cfg)
    cotraining = getattr(cfg.datasets, "vlm_data", None) is not None
    # The host chose this worker environment for one framework; refuse a config for another.
    expected = os.environ.get("PHYSICALAI_STARVLA_FRAMEWORK")
    if expected is None:
        msg = "Start StarVLA training through the StarVLA engine, which chooses the framework's environment."
        raise ValueError(msg)
    if cfg.framework.name != expected:
        msg = f"framework.name={cfg.framework.name!r} differs from {expected!r}, the framework this worker runs."
        raise ValueError(msg)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", str(cfg.run_id)):
        msg = "run_id must use 1-64 letters, digits, underscores, or hyphens."
        raise ValueError(msg)
    if cfg.framework.name in WORLD_MODEL_FRAMEWORKS:
        backbone = _check_world_model(cfg, cotraining=cotraining)
    else:
        backbone = _check_backbone(Path(cfg.framework.qwenvl.base_vlm), cotraining=cotraining)
        cfg.framework.qwenvl.base_vlm = str(backbone)
    # These register upstream state in this process, so every torchrun rank repeats them.
    if cfg.framework.name == "QwenFast":
        _prepare_qwenfast(cfg, backbone)
    if cotraining:
        _prepare_cotraining(cfg, backbone)
    cfg.run_root_dir = str(Path(cfg.run_root_dir).expanduser().resolve())
    cfg.trainer.save_format = "safetensors"
    cfg.config_yaml = str(Path(config_yaml).resolve())
    return cfg, backbone, cotraining


def _train_natively(
    cfg: Any,  # ruff: ignore[any-type]  # OmegaConf node.
    channel: Any,  # ruff: ignore[any-type]  # EventChannel, or a rank's FileChannel.
    cancelled: Any,  # ruff: ignore[any-type]  # threading.Event, or a rank's SharedCancellation.
    *,
    cotraining: bool,
    data_parallel: bool = False,
) -> None:
    """Run upstream's SFT or co-training trainer with progress, cancellation and safetensors hooks.

    ``data_parallel`` marks a torchrun rank. Upstream's multi-GPU recipes use DeepSpeed, which
    tolerates trainable parameters that a step leaves unused (an unused state encoder, for one);
    DistributedDataParallel stops at the second step unless it looks for them.
    """
    # A config with VLM data co-trains, with upstream's own co-training trainer.
    if cotraining:
        native = _import_cotrainer()
        trainer_class = native.VLAMTrainer
    else:
        from starVLA.training import train_starvla

        native, trainer_class = train_starvla, train_starvla.VLATrainer
    step_method = trainer_class._train_step
    log_method = trainer_class._log_metrics

    def step(trainer: Any, batch_vla: list[dict[str, Any]], batch_vlm: object = None) -> dict[str, float]:  # ruff: ignore[any-type]
        if cancelled.is_set():
            raise CancelledError
        return step_method(trainer, batch_vla, batch_vlm)

    def log(trainer: Any, metrics: dict[str, Any]) -> None:  # ruff: ignore[any-type]
        log_method(trainer, metrics)
        channel.send({
            "type": "progress",
            "step": trainer.completed_steps,
            "total": int(trainer.config.trainer.max_train_steps),
            "metrics": finite_metrics(metrics),
        })

    trainer_class._train_step = step
    trainer_class._log_metrics = log
    # Both trainers save safetensors, which reject Qwen3's tied embedding and head storage.
    if cotraining:
        accelerator = native.accelerator
        get_state_dict = accelerator.get_state_dict
        accelerator.get_state_dict = lambda model, *args, **kwargs: _saveable_state_dict(
            get_state_dict(model, *args, **kwargs),
        )
    else:
        state_method = native._get_state_dict
        native._get_state_dict = lambda accelerator, model: _saveable_state_dict(state_method(accelerator, model))
    try:
        with (
            _vlm_loader_workers(cfg) if cotraining else contextlib.nullcontext(),
            _find_unused_parameters(native.accelerator) if data_parallel else contextlib.nullcontext(),
        ):
            native.main(cfg)
    finally:
        trainer_class._train_step = step_method
        trainer_class._log_metrics = log_method
        if cotraining:
            del accelerator.get_state_dict
        else:
            native._get_state_dict = state_method


@contextlib.contextmanager
def _find_unused_parameters(accelerator: Any) -> Iterator[None]:  # ruff: ignore[any-type]  # accelerate.Accelerator.
    """Make the accelerator's DistributedDataParallel look for trainable parameters a step left unused.

    Yields:
        Nothing; the accelerator's previous setting is restored on exit.
    """
    from accelerate.utils import DistributedDataParallelKwargs

    previous = accelerator.ddp_handler
    accelerator.ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
    try:
        yield
    finally:
        accelerator.ddp_handler = previous


@contextlib.contextmanager
def _vlm_loader_workers(cfg: Any) -> Iterator[None]:  # ruff: ignore[any-type]  # OmegaConf node.
    """Build upstream's VLM data loader with ``datasets.vlm_data.num_workers`` worker processes.

    Upstream hard-codes four. Forking them after the model was on the GPU stalled training on the
    first step, as forking video decoders does for the action data (``vla_data.num_workers``).
    Leaving the setting unset keeps upstream's four.

    Yields:
        Nothing; upstream's loader factory is restored on exit.
    """
    workers = getattr(cfg.datasets.vlm_data, "num_workers", None)
    if workers is None:
        yield
        return
    from torch.utils.data import DataLoader

    dataloaders: Any = importlib.import_module("starVLA.dataloader")  # nosemgrep  # Fixed module name.

    make = dataloaders.make_vlm_dataloader

    def make_vlm_dataloader(config: Any) -> dict[str, Any]:  # ruff: ignore[any-type]
        module = make(config)
        loader = module["train_dataloader"]
        module["train_dataloader"] = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            collate_fn=loader.collate_fn,
            num_workers=int(workers),
        )
        return module

    dataloaders.make_vlm_dataloader = make_vlm_dataloader
    try:
        yield
    finally:
        dataloaders.make_vlm_dataloader = make


def _import_cotrainer() -> Any:  # ruff: ignore[any-type]  # Upstream module.
    """Import upstream's co-training module, without DeepSpeed in the plugin's single-GPU mode.

    ``train_starvla`` builds no DeepSpeed plugin when ``STARVLA_DISABLE_DEEPSPEED=1``;
    ``train_starvla_cotrain`` builds one at import regardless, and DeepSpeed is not in the worker
    environment. Apply the same switch while the module is imported.

    Returns:
        The ``train_starvla_cotrain`` module.
    """
    import accelerate

    original = accelerate.DeepSpeedPlugin
    if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1":
        accelerate.DeepSpeedPlugin = lambda *_args, **_kwargs: None
    try:
        from starVLA.training import train_starvla_cotrain
    finally:
        accelerate.DeepSpeedPlugin = original
    return train_starvla_cotrain


def _check_world_model(cfg: Any, *, cotraining: bool) -> Path:  # ruff: ignore[any-type]  # OmegaConf node.
    """Check that StarVLA will load ``framework.world_model.base_wm`` as the world model chosen for it.

    Returns:
        The resolved world-model folder, also written back to the config.

    Raises:
        ValueError: If the folder is not a local diffusers world model of that type, StarVLA would load
            another model, or the config co-trains.
    """
    kind = os.environ.get(BACKBONE_VARIABLE, WORLD_MODEL_FRAMEWORKS[cfg.framework.name])
    folder_name, pipeline = WORLD_MODELS.get(kind, (kind, "?"))
    base_wm = getattr(getattr(cfg.framework, "world_model", None), "base_wm", None)
    if not base_wm:
        msg = f"{cfg.framework.name} reads its backbone from framework.world_model.base_wm; set a local {folder_name}."
        raise ValueError(msg)
    backbone = Path(str(base_wm)).expanduser().resolve(strict=True)
    index = backbone / "model_index.json"
    found = json.loads(index.read_text()).get("_class_name") if index.is_file() else None
    # StarVLA picks the world model by a substring of the path; model_index.json says what it holds.
    if found != pipeline or dispatched_world_model(str(backbone)) != kind:
        msg = (
            f"Use a local, materialized {folder_name} diffusers folder ({pipeline}) with {folder_name} in its "
            "directory name (upstream world-model dispatch)."
        )
        raise ValueError(msg)
    if cotraining:
        msg = "Co-training with VLM data supports Qwen3-VL backbones; StarVLA's VLM data loader needs one."
        raise ValueError(msg)
    cfg.framework.world_model.base_wm = str(backbone)
    return backbone


def _check_backbone(path: Path, *, cotraining: bool) -> Path:
    """Check that StarVLA will load a local backbone as the type the host chose this environment for.

    Returns:
        The resolved backbone folder.

    Raises:
        ValueError: If the folder is not a local backbone of that type, or StarVLA would load another VLM.
    """
    backbone = path.expanduser().resolve(strict=True)
    if not (backbone / "config.json").is_file():
        msg = "framework.qwenvl.base_vlm must point to a local materialized backbone."
        raise ValueError(msg)
    # StarVLA picks the VLM by substrings of the path; the config says what the folder holds.
    kind = os.environ.get(BACKBONE_VARIABLE, DEFAULT_BACKBONE)
    name = BACKBONES.get(kind, kind)
    if (
        dispatched_backbone(str(backbone)) != kind
        or json.loads((backbone / "config.json").read_text()).get("model_type") != kind
    ):
        msg = f"Use a local {name} backbone with {name} in its directory name (upstream VLM dispatch)."
        raise ValueError(msg)
    if cotraining and kind != DEFAULT_BACKBONE:
        msg = "Co-training with VLM data supports Qwen3-VL backbones; StarVLA's VLM data loader needs one."
        raise ValueError(msg)
    return backbone


def _prepare_qwenfast(cfg: Any, backbone: Path) -> None:  # ruff: ignore[any-type]  # OmegaConf node.
    """Check QwenFast's backbone and point its FAST tokenizer at the configured folder.

    Raises:
        ValueError: If the backbone lacks StarVLA's action tokens or the tokenizer folder is invalid.
    """
    # StarVLA enables the action-token range only for backbones whose path says "-Action".
    if "-Action" not in str(backbone):
        msg = "QwenFast needs StarVLA's Qwen3-VL-4B-Instruct-Action backbone, in a folder named with -Action."
        raise ValueError(msg)
    # A plugin key: StarVLA hard-codes the FAST tokenizer's location (see use_fast_tokenizer).
    fast = Path(str(getattr(cfg.framework.action_model, "fast_tokenizer_path", "") or "")).expanduser()
    use_fast_tokenizer(fast)
    cfg.framework.action_model.fast_tokenizer_path = str(fast)


def _prepare_cotraining(cfg: Any, backbone: Path) -> None:  # ruff: ignore[any-type]  # OmegaConf node.
    """Check a co-training config and register its VLM datasets with StarVLA.

    Upstream names VLM datasets in a Python table with paths relative to the working directory.
    The plugin instead reads them from ``datasets.vlm_data.sources`` (name -> absolute
    ``annotation_path`` and ``data_path``) and registers them before training starts.

    Upstream's VLM loader also overwrites the backbone's video-processor sizes, frame counts and
    fps from ``video_*`` settings, even for image-only data, and fails when they are unset. Unset
    ones keep the backbone's own values.

    Raises:
        ValueError: If the datasets, backbone type, VLM freezing or loss scale cannot co-train.
    """
    vlm_data = cfg.datasets.vlm_data
    sources = getattr(vlm_data, "sources", None) or {}
    names = [re.sub(r"%\d+$", "", name.strip()) for name in str(getattr(vlm_data, "dataset_use", "")).split(",")]
    if not any(names) or not set(names) <= set(sources):
        msg = (
            "List every datasets.vlm_data.dataset_use name under datasets.vlm_data.sources, "
            "each with an absolute annotation_path and data_path."
        )
        raise ValueError(msg)
    registered = {}
    for name in names:
        annotation = Path(str(sources[name]["annotation_path"]))
        images = Path(str(sources[name]["data_path"]))
        if not (annotation.is_absolute() and annotation.is_file() and images.is_absolute() and images.is_dir()):
            msg = f"VLM dataset {name!r} needs an existing absolute annotation_path file and data_path folder."
            raise ValueError(msg)
        registered[name] = {"annotation_path": str(annotation), "data_path": f"{images}/"}
    if getattr(vlm_data, "model_type", None) is None:
        vlm_data.model_type = "qwen3vl"
    if vlm_data.model_type != "qwen3vl":
        msg = "Co-training a Qwen3-VL backbone needs datasets.vlm_data.model_type=qwen3vl."
        raise ValueError(msg)
    frozen = [part.strip() for part in str(getattr(cfg.trainer, "freeze_modules", "") or "").split(",")]
    if "qwen_vl_interface" in frozen:
        msg = "Co-training trains the VLM on VLM data: freeze_modules must leave part of qwen_vl_interface trainable."
        raise ValueError(msg)
    workers = getattr(vlm_data, "num_workers", None)
    if workers is not None and (type(workers) is not int or workers < 0):
        msg = "datasets.vlm_data.num_workers must be a non-negative integer."
        raise ValueError(msg)
    if getattr(getattr(cfg.trainer, "loss_scale", None), "vlm", None) is None:
        msg = "Co-training weighs the VLM loss by trainer.loss_scale.vlm; set it."
        raise ValueError(msg)
    from starVLA.dataloader.qwenvl_llavajson import qwen_data_config
    from transformers import AutoProcessor

    # The validated local backbone, with the Hub offline: nothing is downloaded.
    video = AutoProcessor.from_pretrained(str(backbone)).video_processor  # nosec B615
    defaults = {
        "video_min_pixels": video.size["shortest_edge"],
        "video_max_pixels": video.size["longest_edge"],
        "video_min_frames": video.min_frames,
        "video_max_frames": video.max_frames,
        "video_fps": video.fps,
    }
    for key, value in defaults.items():
        if getattr(vlm_data, key, None) is None:
            setattr(vlm_data, key, value)
    qwen_data_config.data_dict.update(registered)


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
        Zero on success, one on cancellation; zero for a torchrun rank that stopped with the others
        for a cancellation, so torchrun does not report it as a failed rank.
    """
    channel = EventChannel()
    try:
        output, checkpoint = train()
    except CancelledError:
        if os.environ.get(DISTRIBUTED_DIR_VARIABLE) is not None:
            leave_process_group()
            return 0
        channel.send({"type": "cancelled", "step": None})
        return 1
    except Exception as exc:
        channel.send({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        raise
    channel.send({"type": "result", "output_dir": str(output), "checkpoint": str(checkpoint)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
