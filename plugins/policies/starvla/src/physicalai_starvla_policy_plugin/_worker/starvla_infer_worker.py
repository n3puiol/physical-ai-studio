# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""StarVLA inference worker, importable only inside its isolated framework environment."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Isolated workers run without their folder on sys.path; the sibling modules live there.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import starvla_worker_protocol as protocol
from starvla_worker_common import (
    WORLD_MODELS,
    identity,
    restrict_remote_code,
    trained_with_state,
    use_fast_tokenizer,
)

_ACTION_RANK = 3


class Policy:
    """Use upstream model construction and training-time normalization, not host reimplementations."""

    def __init__(self, artifact: Path, descriptor: dict[str, Any], device: str) -> None:
        """Load a data-only checkpoint and check the complete robot mapping.

        Raises:
            ValueError: If model metadata, device, or robot mapping is unsupported.
        """
        restrict_remote_code()
        import torch
        from deployment.model_server.policy_wrapper import PolicyServerWrapper
        from omegaconf import OmegaConf

        checkpoint = (artifact / descriptor["checkpoint"]).resolve(strict=True)
        if not checkpoint.is_relative_to(artifact.resolve()) or checkpoint.suffix != ".safetensors":
            msg = "Invalid checkpoint path or format."
            raise ValueError(msg)
        cfg = OmegaConf.load(artifact / "model" / "config.yaml")
        # The host validated the descriptor's framework and started its environment.
        if cfg.framework.name != descriptor["policy"]:
            msg = "The package's checkpoint config and descriptor name different StarVLA frameworks."
            raise ValueError(msg)
        if device == "auto":
            device = "cuda"
        if not device.startswith("cuda") or not torch.cuda.is_available():
            msg = "StarVLA inference requires CUDA."
            raise ValueError(msg)
        # Upstream picks its VLM, and enables action tokens, by substrings of this folder name.
        folder = descriptor.get("backbone", "Qwen3-VL")
        backbone = (artifact / folder).resolve(strict=True)
        # World-model frameworks load their backbone from framework.world_model.base_wm.
        world_models = {name for name, _pipeline in WORLD_MODELS.values()}
        key = "framework.world_model.base_wm" if folder in world_models else "framework.qwenvl.base_vlm"
        if descriptor.get("fast_tokenizer"):
            use_fast_tokenizer((artifact / descriptor["fast_tokenizer"]).resolve(strict=True))
        self.wrapper = PolicyServerWrapper(
            str(checkpoint),
            device=device,
            unnorm_key=descriptor["unnorm_key"],
            config_overrides=[f"{key}={backbone}"],
        )
        self.processor = self.wrapper.get_norm_processor()
        self.io = descriptor["io"]
        # Feed state only if training did: StarVLA packs it only with include_state.
        self.uses_state = trained_with_state(cfg.datasets.vla_data) and bool(self.processor.state_keys)
        check_mapping(self.io, self.processor, uses_state=self.uses_state)
        # The wrapper deliberately does not infer image order. Check it against training.
        from deployment.model_server.policy_norm_processor import (
            _resolve_robot_type,  # ruff: ignore[import-private-name]  # Pinned wrapper resolution.
        )
        from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP

        data_config = ROBOT_TYPE_CONFIG_MAP[_resolve_robot_type(OmegaConf.to_container(cfg), descriptor["unnorm_key"])]
        if list(self.io["cameras"]) != list(data_config.video_keys):
            msg = "Package camera count/order differs from the training DataConfig."
            raise ValueError(msg)
        check_single_step(data_config)

    def predict(self, message: dict[str, Any]) -> dict[str, np.ndarray]:
        """Normalize raw states with training transforms and unnormalize native actions.

        Returns:
            Unnormalized action groups in training DataConfig order.

        Raises:
            TypeError: If a state transform needs an unsupported inference path.
            ValueError: If the model returns invalid action dimensions.
        """
        import torch
        from PIL import Image
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform

        if "seed" in message:
            seed = int(message["seed"])
            random.seed(seed)
            np.random.seed(seed % 2**32)  # ruff: ignore[numpy-legacy-random]  # Seed upstream's global RNG.
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        example = {
            "image": [Image.fromarray(message["video"][key]) for key in self.io["cameras"]],
            "lang": message["task"],
        }
        if self.uses_state:
            state = {key: np.asarray(value, dtype=np.float32) for key, value in message["state"].items()}
            for transform in self.processor.transform.transforms:
                if not any(key.startswith("state.") for key in transform.apply_to):
                    continue
                if not isinstance(transform, (StateActionToTensor, StateActionTransform)):
                    msg = "Unsupported state transform; add an explicit inference implementation before deploying."
                    raise TypeError(msg)
                state = transform(state)
            example["state"] = np.concatenate([np.asarray(state[key]) for key in self.processor.state_keys], axis=-1)
        with torch.inference_mode():
            actions = self.wrapper.predict_action(examples=[example])["actions"]
        expected_width = sum(self.processor.action_key_dims.values())
        if actions.ndim != _ACTION_RANK or actions.shape[0] != 1 or actions.shape[2] != expected_width:
            msg = "Model returned an invalid action chunk."
            raise ValueError(msg)
        result = {}
        cursor = 0
        for key in self.processor.action_keys:
            width = self.processor.action_key_dims[key]
            result[key] = actions[0, :, cursor : cursor + width]
            cursor += width
        return result


def check_single_step(data_config: Any) -> None:  # ruff: ignore[any-type]  # Upstream DataConfig.
    """Refuse a training DataConfig that feeds the model more than the current observation and state.

    Raises:
        ValueError: If the DataConfig uses an observation or state history.
    """
    for label in ("observation_indices", "state_indices"):
        indices = getattr(data_config, label, None) or [0]  # Some DataConfigs (oxe_bridge) set None.
        if list(indices) != [0]:
            msg = "Observation-history checkpoints are not supported."
            raise ValueError(msg)


def check_mapping(io: dict[str, Any], processor: Any, *, uses_state: bool) -> None:  # ruff: ignore[any-type]
    """Check the package's robot mapping against the checkpoint's training DataConfig.

    A checkpoint trained without proprioceptive state ignores any state mapping; packages
    exported before the plugin followed ``include_state`` still carry one.

    Raises:
        ValueError: If action keys or widths, or the state the checkpoint was trained on, differ.
    """
    labels = [("action", processor.action_key_dims)]
    if uses_state:
        if not io["state"]:
            msg = "The checkpoint was trained with proprioceptive state (include_state); the package maps none."
            raise ValueError(msg)
        labels.append(("state", processor.state_key_dims))
    for label, dimensions in labels:
        widths = {key: end - start for key, (start, end) in io[label].items()}
        if widths != dict(dimensions):
            msg = f"Package {label} keys or widths differ from the checkpoint DataConfig."
            raise ValueError(msg)


def main() -> None:
    """Serve framed messages on private descriptors; framework logs keep their own streams.

    Raises:
        ValueError: If a request names an unsupported operation.
    """
    policy = None
    with (
        os.fdopen(int(os.environ["PHYSICALAI_STARVLA_REQUEST_FD"]), "rb") as requests,
        os.fdopen(int(os.environ["PHYSICALAI_STARVLA_RESPONSE_FD"]), "wb") as responses,
    ):
        while (message := protocol.read_frame(requests)) is not None:
            try:  # ruff: ignore[too-many-statements-in-try-clause]  # Every operation shares the protocol error boundary.
                operation = message.get("op")
                if operation == "hello":
                    result = identity()
                elif operation == "load":
                    policy = Policy(Path(message["artifact"]), message["descriptor"], message["device"])
                    result = {}
                elif operation == "predict" and policy is not None:
                    result = {"actions": policy.predict(message)}
                elif operation == "reset" and policy is not None:
                    # predict_action is stateless; the host resets the per-call seed sequence.
                    result = {}
                else:
                    msg = f"Unknown operation or model not loaded: {operation!r}."
                    raise ValueError(msg)  # ruff: ignore[raise-within-try]  # Send protocol errors to the host.
                protocol.write_frame(responses, {"ok": True, **result})
            except Exception as exc:  # ruff: ignore[blind-except]  # Convert model errors to protocol responses.
                protocol.write_frame(responses, {"ok": False, "error": type(exc).__name__, "message": str(exc)})


if __name__ == "__main__":
    main()
