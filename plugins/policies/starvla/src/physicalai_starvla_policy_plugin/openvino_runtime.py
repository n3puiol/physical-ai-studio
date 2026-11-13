# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""In-process OpenVINO Runtime adapter for exported QwenGR00T packages (no StarVLA, torch or CUDA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.registry import adapter_registry

from ._worker import starvla_openvino_pipeline as pipeline
from .openvino_package import BACKEND, load_openvino_descriptor
from .package import IoSpec
from .runtime import flatten_actions, to_observation

_DEVICES = {"auto": "CPU", "cpu": "CPU", "gpu": "GPU", "xpu": "GPU", "npu": "NPU"}


@adapter_registry.register(BACKEND, extensions=(".starvla_ov",))
class StarVLAOpenVINOAdapter(RuntimeAdapter):
    """Run an OpenVINO QwenGR00T package: vision, text and action-head graphs plus numpy glue."""

    def __init__(
        self,
        device: str = "CPU",
        *,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,  # ruff: ignore[any-type]  # Runtime forwards backend-specific options.
    ) -> None:
        """Configure the OpenVINO device, compile options and the noise seed."""
        super().__init__(device=device, **kwargs)
        self._seed = seed
        self._config = dict(config or {})
        self._calls = 0
        self._rng = np.random.default_rng(seed)
        self._pipeline: pipeline.QwenGR00TPipeline | None = None
        self._io: IoSpec | None = None

    def default_device(self) -> str:  # ruff: ignore[no-self-use]
        """Use the CPU unless another OpenVINO device is requested.

        Returns:
            OpenVINO device name.
        """
        return "CPU"

    def load(self, model_path: Path | str) -> None:
        """Validate the package and compile its graphs on the configured device."""
        artifact = Path(model_path).expanduser().resolve(strict=True)
        descriptor = load_openvino_descriptor(artifact)
        device = _DEVICES.get(str(self.device).lower(), str(self.device))
        self._pipeline = pipeline.QwenGR00TPipeline(artifact, device, self._config)
        self._io = IoSpec.from_dict(descriptor["io"])
        self.reset()

    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return unnormalized actions shaped (1, horizon, action_dim).

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._pipeline is None or self._io is None:
            msg = "Load a StarVLA OpenVINO package before predicting."
            raise RuntimeError(msg)
        observation = to_observation(inputs, self._io)
        images = [observation["video"][key] for key in self._io.cameras]
        state = self._pipeline.normalize_state(observation["state"])
        rng = np.random.default_rng(self._seed + self._calls) if self._seed is not None else self._rng
        normalized = self._pipeline.predict_normalized(images, observation["task"], state, self._pipeline.noise(rng))
        self._calls += 1
        return {"action": flatten_actions(self._pipeline.unnormalize_actions(normalized[0]), self._io)}

    def reset(self) -> None:
        """Restart the deterministic noise sequence between episodes."""
        self._calls = 0

    def close(self) -> None:
        """Release the compiled graphs."""
        self._pipeline = None

    @property
    def input_names(self) -> list[str]:
        """Pass Runtime's structured observation through unchanged."""
        return []

    @property
    def output_names(self) -> list[str]:
        """Expose the Runtime action chunk."""
        return ["action"]
