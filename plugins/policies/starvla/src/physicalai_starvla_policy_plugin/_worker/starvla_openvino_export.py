# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Convert a native QwenGR00T package to OpenVINO and check it against StarVLA's PyTorch code.

Runs in the locked ``qwengr00t-openvino`` environment on CPU:

    python starvla_openvino_export.py --artifact <native policy.starvla> --output <dir>
        [--weights fp32|fp16|int8] [--image-size H W]

``fp16`` stores fp16 weights (computed in fp32 on CPU); ``int8`` additionally compresses the text
model's weights to INT8 (asymmetric, per channel).

The output directory receives ``vision.xml``, ``llm.xml``, ``head.xml`` (+ ``.bin``), the token
embedding table, ``tokenizer.json`` and ``package.json``. The package is complete only once
``package.json`` has a ``validation`` entry: the torch-free pipeline reproduced PyTorch on a
synthetic observation within the weight format's tolerance.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import starvla_openvino_pipeline as pipeline

# Largest acceptable |OpenVINO - PyTorch| on normalized actions, and smallest cosine, per weight format.
TOLERANCE = {"fp32": (1e-3, 0.99999), "fp16": (5e-2, 0.9999), "int8": (1e-1, 0.999)}
# The staged PyTorch model must reproduce StarVLA's own predict_action this closely.
SPLIT_TOLERANCE = 1e-4
STATE_TOLERANCE = 1e-5
UNNORMALIZE_TOLERANCE = 1e-4
SENTINEL = "\x00INSTRUCTION\x00"
VALIDATION_TASK = "put the bowl on the plate"
POLICY = "QwenGR00T"


class ExportError(RuntimeError):
    """The checkpoint cannot be exported, or the export does not reproduce PyTorch."""


def log(message: str) -> None:
    """Report progress on stderr; stdout carries only the final JSON result."""
    print(f"[starvla-openvino {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def stats(actual: Any, expected: Any) -> dict[str, float]:
    """Compare two arrays.

    Returns:
        Max and mean absolute difference, and cosine similarity.
    """
    a = np.asarray(actual, dtype=np.float64).ravel()
    b = np.asarray(expected, dtype=np.float64).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {"max_abs": float(np.max(np.abs(a - b))), "mean_abs": float(np.mean(np.abs(a - b))), "cosine": cosine}


# PyTorch modules traced into the three graphs ------------------------------------------------
def build_modules(torch: Any, framework: Any, grid: Any) -> dict[str, Any]:
    """Wrap StarVLA's vision tower, text model and action head as traceable modules.

    Patches the text model in place (identity final norm, dense DeepStack), so StarVLA's own
    ``predict_action`` must not be called afterwards.

    Returns:
        ``vision``, ``llm`` and ``head`` modules, and ``head_fn`` for the PyTorch reference.
    """
    head = framework.action_model
    text = framework.qwen_vl_interface.model.model.language_model
    visual = framework.qwen_vl_interface.model.model.visual

    def head_with_noise(vl_embs: Any, mask: Any, state: Any, noise: Any) -> Any:
        """``FlowmatchingActionHead.predict_action`` with the sampled noise as an input.

        Returns:
            Normalized actions of shape (batch, horizon, action_dim).
        """
        batch = vl_embs.shape[0]
        actions = noise
        steps = head.num_inference_timesteps
        state_features = head.state_encoder(state)
        future = head.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
        for step in range(steps):
            bucket = int(step / float(steps) * head.num_timestep_buckets)
            timesteps = torch.full((batch,), bucket, dtype=torch.long)
            features = head.action_encoder(actions, timesteps)
            if head.config.add_pos_embed:
                features += head.position_embedding(torch.arange(features.shape[1])).unsqueeze(0)
            hidden = torch.cat((state_features, future, features), dim=1)
            out = head.model(
                hidden_states=hidden,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=mask,
                timestep=timesteps,
            )
            # Not ``+=``: the first step would update the caller's noise tensor in place.
            actions = actions + (1.0 / steps) * head.action_decoder(out)[:, -head.action_horizon :]  # ruff: ignore[non-augmented-assignment]
        return actions

    class Head(torch.nn.Module):
        """Four unrolled Euler steps of the flow-matching head."""

        def __init__(self) -> None:
            super().__init__()
            self.head = head

        def forward(self, vl_embs: Any, attention_mask: Any, state: Any, noise: Any) -> Any:  # ruff: ignore[no-self-use]
            return head_with_noise(vl_embs, attention_mask, state, noise)

    class Vision(torch.nn.Module):
        """Vision tower with the exported camera grid baked in."""

        def __init__(self) -> None:
            super().__init__()
            self.visual = visual
            self.register_buffer("grid", grid, persistent=False)

        def forward(self, pixel_values: Any) -> Any:
            embeds, deepstack = self.visual(pixel_values, grid_thw=self.grid)
            return (embeds, *deepstack)

    # StarVLA conditions on hidden_states[-1]: the last decoder layer before the final RMSNorm.
    text.norm = torch.nn.Identity()
    # DeepStack features arrive dense (zero off image tokens) instead of boolean-index assignment.
    text._deepstack_process = lambda hidden, _mask, dense: hidden + dense

    class Text(torch.nn.Module):
        """Text model on prepared embeddings, returning the pre-norm last hidden state."""

        def __init__(self) -> None:
            super().__init__()
            self.text = text

        def forward(self, inputs_embeds: Any, attention_mask: Any, position_ids: Any, deepstack: Any) -> Any:
            # transformers builds masks with vmap, which TorchScript cannot trace; pass a 4D mask.
            length = inputs_embeds.shape[1]
            causal = torch.ones((length, length), dtype=torch.bool).tril()
            allowed = causal[None, None] & attention_mask[:, None, None, :].bool()
            mask = torch.zeros(allowed.shape, dtype=inputs_embeds.dtype).masked_fill(
                ~allowed,
                torch.finfo(inputs_embeds.dtype).min,
            )
            out = self.text(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                position_ids=position_ids,
                visual_pos_masks=None,
                deepstack_visual_embeds=list(deepstack.unbind(0)),
                use_cache=False,
            )
            return out.last_hidden_state

    return {"head": Head().eval(), "vision": Vision().eval(), "llm": Text().eval(), "head_fn": head_with_noise}


# Package metadata ------------------------------------------------------------------------------
def normalization_spec(processor: Any) -> dict[str, Any]:
    """Extract StarVLA's state/action normalizers as data.

    Returns:
        Binary threshold and per-key normalization mode and statistics.

    Raises:
        ExportError: If a state/action transform has no numpy implementation.
    """
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform

    keys: dict[str, Any] = {}
    threshold = 0.5
    for transform in processor.transform.transforms:
        if not any(key.startswith(("state.", "action.")) for key in transform.apply_to):
            continue
        if isinstance(transform, StateActionToTensor):
            continue
        if not isinstance(transform, StateActionTransform) or transform._rotation_transformers:
            msg = f"State/action transform {type(transform).__name__} has no OpenVINO implementation yet."
            raise ExportError(msg)
        threshold = float(transform.binary_threshold)
        for key, normalizer in transform._normalizers.items():
            if normalizer.mode not in pipeline.NORMALIZATION_MODES:
                msg = f"Normalization mode {normalizer.mode!r} for {key} has no OpenVINO implementation yet."
                raise ExportError(msg)
            keys[key] = {
                "mode": normalizer.mode,
                "stats": {
                    name: np.asarray(value, dtype=np.float32).tolist() for name, value in normalizer.statistics.items()
                },
            }
    return {"binary_threshold": threshold, "keys": keys}


def dataset_image_size(cfg: Any, cameras: list[str]) -> tuple[int, int]:
    """Read the training cameras' resolution from the LeRobot dataset metadata.

    Returns:
        Camera (height, width).

    Raises:
        ExportError: If the metadata is missing or the cameras differ in resolution.
    """
    from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES

    vla = cfg.datasets.vla_data
    sizes = set()
    try:
        for name, _weight, _robot in DATASET_NAMED_MIXTURES[vla.data_mix]:
            meta = Path(vla.data_root_dir) / name / "meta"
            modality = json.loads((meta / "modality.json").read_text())["video"]
            features = json.loads((meta / "info.json").read_text())["features"]
            for camera in cameras:
                shape = features[modality[camera.removeprefix("video.")]["original_key"]]["shape"]
                sizes.add((int(shape[0]), int(shape[1])))
    except (OSError, KeyError) as error:
        msg = f"Cannot read the training camera resolution ({error}); pass the image size explicitly."
        raise ExportError(msg) from error
    if len(sizes) != 1:
        msg = f"Training cameras have different resolutions {sorted(sizes)}; pass the image size explicitly."
        raise ExportError(msg)
    return sizes.pop()


def prompt_template(framework: Any, cameras: int) -> dict[str, str]:
    """Render StarVLA's chat prompt around a sentinel instruction.

    Returns:
        Prompt prefix, suffix and the image placeholder token.

    Raises:
        ExportError: If the instruction cannot be located in the rendered template.
    """
    interface = framework.qwen_vl_interface
    instruction = SENTINEL
    vla = framework.config.datasets.vla_data
    if "CoT_prompt" in vla:
        instruction = vla.get("CoT_prompt", "").replace("{instruction}", SENTINEL)  # ruff: ignore[missing-f-string-syntax]  # StarVLA's literal placeholder.
    content = [{"type": "image", "image": None} for _ in range(cameras)] + [{"type": "text", "text": instruction}]
    rendered = interface.processor.apply_chat_template(
        [[{"role": "user", "content": content}]],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = rendered if isinstance(rendered, str) else rendered[0]
    if text.count(SENTINEL) != 1:
        msg = "Could not locate the instruction in the chat template."
        raise ExportError(msg)
    prefix, suffix = text.split(SENTINEL)
    return {"prefix": prefix, "suffix": suffix, "image_token": interface.processor.image_token}


# Export steps ----------------------------------------------------------------------------------
@dataclass
class Export:
    """State shared by the export steps."""

    artifact: Path
    output: Path
    weights: str
    image_size: tuple[int, int] | None
    descriptor: dict[str, Any] = field(default_factory=dict)
    framework: Any = None
    processor: Any = None
    embedding: Any = None
    tensors: dict[str, Any] = field(default_factory=dict)
    images: list[np.ndarray] = field(default_factory=list)
    state_groups: dict[str, np.ndarray] = field(default_factory=dict)
    modules: dict[str, Any] = field(default_factory=dict)
    package: dict[str, Any] = field(default_factory=dict)


def load(export: Export) -> None:
    """Check the package against the training DataConfig and load the checkpoint in float32.

    Raises:
        ExportError: If the checkpoint or mapping has no OpenVINO export.
    """
    from deployment.model_server.policy_norm_processor import _resolve_robot_type
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from omegaconf import OmegaConf
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP

    export.descriptor = json.loads((export.artifact / "package.json").read_text())
    cfg = OmegaConf.load(export.artifact / "model" / "config.yaml")
    io = export.descriptor["io"]
    cameras = list(io["cameras"])
    data_config = ROBOT_TYPE_CONFIG_MAP[
        _resolve_robot_type(OmegaConf.to_container(cfg), export.descriptor["unnorm_key"])
    ]
    problems = {
        "Only QwenGR00T has an OpenVINO export.": cfg.framework.name != POLICY,
        "Checkpoints trained with obs_image_size have no OpenVINO export yet.": bool(
            getattr(cfg.datasets.vla_data, "obs_image_size", None),
        ),
        "Package camera order differs from the training DataConfig.": cameras != list(data_config.video_keys),
        "The OpenVINO action head is exported with proprioceptive state; this package has none.": not io["state"],
    }
    for message, failed in problems.items():
        if failed:
            raise ExportError(message)
    if export.image_size is None:
        export.image_size = dataset_image_size(cfg, cameras)
    log(f"cameras {cameras} at {export.image_size[0]}x{export.image_size[1]}, weights {export.weights}")

    started = time.time()
    wrapper = PolicyServerWrapper(
        str(export.artifact / export.descriptor["checkpoint"]),
        device="cpu",
        unnorm_key=export.descriptor["unnorm_key"],
        config_overrides=[f"framework.qwenvl.base_vlm={export.artifact / 'Qwen3-VL'}"],
    )
    export.framework = wrapper._framework
    export.embedding = export.framework.qwen_vl_interface.model.model.get_input_embeddings().weight.detach()
    # The checkpoint loads a bf16 VLM and fp32 head that StarVLA reconciles with CUDA autocast,
    # which does nothing on CPU. Export (and the reference) run in float32.
    export.framework.float().eval()
    export.processor = wrapper.get_norm_processor()
    for label, dims in (("action", export.processor.action_key_dims), ("state", export.processor.state_key_dims)):
        mapping = io[label]
        if set(mapping) != set(dims) or any(end - start != dims[key] for key, (start, end) in mapping.items()):
            msg = f"Package {label} keys or widths differ from the checkpoint DataConfig."
            raise ExportError(msg)
    log(f"loaded checkpoint in {time.time() - started:.0f}s")


def reference(export: Export) -> None:  # ruff: ignore[too-many-locals]  # One linear reference computation.
    """Run StarVLA's predict_action and the staged PyTorch split on a synthetic observation.

    Raises:
        ExportError: If the staged split no longer reproduces predict_action.
    """
    import torch
    from PIL import Image

    framework, processor, io = export.framework, export.processor, export.descriptor["io"]
    rng = np.random.default_rng(0)
    export.images = [rng.integers(0, 256, (*export.image_size, 3), dtype=np.uint8) for _ in io["cameras"]]
    raw_state = rng.normal(0, 0.2, max(end for _, end in io["state"].values())).astype(np.float32)
    export.state_groups = {key: raw_state[None, start:end] for key, (start, end) in io["state"].items()}
    normalized_state = {key: value.copy() for key, value in export.state_groups.items()}
    for transform in processor.transform.transforms:
        if any(key.startswith("state.") for key in transform.apply_to):
            normalized_state = transform(normalized_state)
    example_state = np.concatenate([np.asarray(normalized_state[key]) for key in processor.state_keys], axis=-1)
    example = {
        "image": [Image.fromarray(image) for image in export.images],
        "lang": VALIDATION_TASK,
        "state": example_state,
    }

    with torch.inference_mode():
        interface = framework.qwen_vl_interface
        inputs = interface.build_qwenvl_inputs(images=[example["image"]], instructions=[example["lang"]])
        vlm = interface.model.model
        grid = inputs["image_grid_thw"]
        if len({tuple(row) for row in grid.tolist()}) != 1:
            msg = "Cameras resolve to different image grids."
            raise ExportError(msg)
        native_hidden = interface(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
        image_embeds, deepstack = vlm.visual(inputs["pixel_values"], grid_thw=grid)
        ids = inputs["input_ids"]
        embeds = vlm.get_input_embeddings()(ids)
        image_mask = (ids == vlm.config.image_token_id).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(image_mask, image_embeds)
        dense = torch.zeros((len(deepstack), *embeds.shape))
        for index, features in enumerate(deepstack):
            dense[index] = dense[index].masked_scatter(image_mask, features)
        position_ids, _ = vlm.get_rope_index(ids, grid, None, attention_mask=inputs["attention_mask"])
        head = framework.action_model
        torch.manual_seed(1)
        native_actions = framework.predict_action(examples=[example])["normalized_actions"]
        torch.manual_seed(1)
        noise = torch.randn((1, head.action_horizon, head.action_dim))

        # Patches the text model in place, so the native path above must already have run.
        export.modules = build_modules(torch, framework, grid.clone())
        hidden = export.modules["llm"](embeds, inputs["attention_mask"], position_ids, dense)
        state = torch.from_numpy(np.array([example_state])).float()
        mask = inputs["attention_mask"].bool()
        actions = export.modules["head_fn"](hidden, mask, state, noise)
    split = stats(actions, native_actions)
    log(f"torch split vs predict_action: hidden {stats(hidden, native_hidden)}, actions {split}")
    if split["max_abs"] > SPLIT_TOLERANCE:
        msg = "The staged PyTorch model no longer reproduces predict_action; upstream changed."
        raise ExportError(msg)
    export.tensors = {
        **inputs,
        "embeds": embeds,
        "dense": dense,
        "position_ids": position_ids,
        "hidden": hidden,
        "mask": mask,
        "state": state,
        "noise": noise,
        "actions": actions,
        "native_actions": native_actions,
    }


def convert(export: Export) -> None:
    """Trace, optionally compress, and save the three OpenVINO graphs."""
    import openvino as ov
    import torch

    t = export.tensors
    examples = {
        "vision": ((t["pixel_values"],), ["pixel_values"]),
        "llm": (
            (t["embeds"], t["attention_mask"], t["position_ids"], t["dense"]),
            ["inputs_embeds", "attention_mask", "position_ids", "deepstack"],
        ),
        "head": ((t["hidden"], t["mask"], t["state"], t["noise"]), ["vl_embs", "attention_mask", "state", "noise"]),
    }
    for name in pipeline.MODELS:
        started = time.time()
        tensors, names = examples[name]
        # Inference-mode tensors cannot be traced; clone them as ordinary tensors.
        with torch.no_grad():
            model = ov.convert_model(export.modules[name], example_input=tuple(tensor.clone() for tensor in tensors))
        for port, port_name in zip(model.inputs, names, strict=True):
            port.get_tensor().set_names({port_name})
        # INT8 only for the text model, where the size is (14 GB fp32). INT8 vision weights were
        # accurate on real frames but drifted on out-of-distribution inputs (cosine 0.975).
        if export.weights == "int8" and name == "llm":
            import nncf

            model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
        ov.save_model(model, export.output / f"{name}.xml", compress_to_fp16=export.weights != "fp32")
        del model
        gc.collect()
        log(f"saved {name}.xml in {time.time() - started:.0f}s")


def describe(export: Export) -> None:
    """Write the embedding table, tokenizer and package metadata the torch-free pipeline reads."""
    import torch

    framework, processor = export.framework, export.processor
    bfloat16 = export.embedding.dtype == torch.bfloat16
    table = export.embedding.view(torch.int16).numpy().view(np.uint16) if bfloat16 else export.embedding.float().numpy()
    np.save(export.output / pipeline.EMBEDDINGS, table)
    shutil.copy2(export.artifact / "Qwen3-VL" / pipeline.TOKENIZER, export.output / pipeline.TOKENIZER)

    image_processor = framework.qwen_vl_interface.processor.image_processor
    merge = image_processor.merge_size
    resized = pipeline.smart_resize(
        *export.image_size,
        image_processor.patch_size * merge,
        image_processor.size["shortest_edge"],
        image_processor.size["longest_edge"],
    )
    grid = export.tensors["image_grid_thw"][0].tolist()
    vlm_config = framework.qwen_vl_interface.model.model.config
    head = framework.action_model
    io = export.descriptor["io"]
    export.package = {
        "format_version": pipeline.FORMAT_VERSION,
        "backend": pipeline.BACKEND,
        "policy": POLICY,
        "source_environment": export.descriptor["environment"],
        "weights": export.weights,
        "unnorm_key": export.descriptor["unnorm_key"],
        "io": io,
        "image": {
            "cameras": len(io["cameras"]),
            "size": list(export.image_size),
            "resized": list(resized),
            "grid": grid,
            "tokens": grid[0] * grid[1] * grid[2] // merge**2,
            "mean": [float(v) for v in image_processor.image_mean],
            "std": [float(v) for v in image_processor.image_std],
            "patch_size": image_processor.patch_size,
            "temporal_patch_size": image_processor.temporal_patch_size,
            "merge_size": merge,
        },
        "prompt": prompt_template(framework, len(io["cameras"])),
        "tokens": {
            "image_token_id": int(vlm_config.image_token_id),
            "vision_start_token_id": int(vlm_config.vision_start_token_id),
        },
        "embedding": {"dtype": "bfloat16" if bfloat16 else "float32", "shape": list(export.embedding.shape)},
        "state": {"keys": [[key, processor.state_key_dims[key]] for key in processor.state_keys]},
        "action": {
            "keys": [[key, processor.action_key_dims[key]] for key in processor.action_keys],
            "horizon": int(head.action_horizon),
            "dim": int(head.action_dim),
        },
        "normalization": normalization_spec(processor),
    }
    (export.output / pipeline.DESCRIPTOR).write_text(json.dumps(export.package, indent=2) + "\n")


def validate(export: Export) -> dict[str, Any]:
    """Run the torch-free pipeline on the synthetic observation and compare it with PyTorch.

    Returns:
        Validation measurements, also recorded in ``package.json``.

    Raises:
        ExportError: If any stage differs beyond tolerance.
    """
    t = export.tensors
    ov_pipeline = pipeline.QwenGR00TPipeline(export.output, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    prepared = ov_pipeline.prepare(export.images, VALIDATION_TASK)
    unnormalized = ov_pipeline.unnormalize_actions(t["native_actions"][0])
    checks: dict[str, Any] = {
        "input_ids_equal": bool(np.array_equal(prepared["input_ids"], t["input_ids"].numpy())),
        "position_ids_equal": bool(np.array_equal(prepared["position_ids"], t["position_ids"].numpy())),
        "pixel_values": stats(prepared["pixel_values"], t["pixel_values"]),
        "state": stats(ov_pipeline.normalize_state(export.state_groups), t["state"].numpy()),
        "unnormalize": stats(
            np.concatenate([unnormalized[key] for key, _ in export.package["action"]["keys"]], axis=-1),
            export.processor.unapply_actions(t["native_actions"][0]),
        ),
        "actions": stats(
            ov_pipeline.predict_normalized(export.images, VALIDATION_TASK, t["state"].numpy(), t["noise"].numpy()),
            t["actions"],
        ),
    }
    log(f"validation: {json.dumps(checks)}")
    max_abs, cosine = TOLERANCE[export.weights]
    failures = {
        "Torch-free prompt or position ids differ from transformers.": not (
            checks["input_ids_equal"] and checks["position_ids_equal"]
        ),
        "Numpy normalization differs from StarVLA's transforms.": checks["state"]["max_abs"] > STATE_TOLERANCE
        or checks["unnormalize"]["max_abs"] > UNNORMALIZE_TOLERANCE,
        f"OpenVINO actions differ from PyTorch beyond the {export.weights} tolerance.": checks["actions"]["max_abs"]
        > max_abs
        or checks["actions"]["cosine"] < cosine,
    }
    for message, failed in failures.items():
        if failed:
            raise ExportError(message)
    return checks


def main() -> int:
    """Export, validate and mark the package complete.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=pipeline.WEIGHT_FORMATS, default="fp16")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    args = parser.parse_args()
    export = Export(
        artifact=args.artifact.resolve(strict=True),
        output=args.output.absolute(),
        weights=args.weights,
        image_size=tuple(args.image_size) if args.image_size else None,
    )
    export.output.mkdir(parents=True, exist_ok=False)
    load(export)
    reference(export)
    convert(export)
    describe(export)
    export.package["validation"] = validate(export)
    (export.output / pipeline.DESCRIPTOR).write_text(json.dumps(export.package, indent=2) + "\n")
    print(json.dumps({"ok": True, "validation": export.package["validation"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
