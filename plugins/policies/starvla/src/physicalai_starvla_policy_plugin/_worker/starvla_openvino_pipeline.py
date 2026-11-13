# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-free QwenGR00T inference over exported OpenVINO graphs.

Shared by the export worker, which checks it against StarVLA's PyTorch implementation, and
by the in-process Runtime adapter. Only numpy, OpenVINO and ``tokenizers`` are imported; the
module must stay importable on the worker's Python 3.10 and Runtime's Python 3.12+.

The pipeline reproduces ``QwenGR00T.predict_action`` for one observation:

1. Qwen3-VL image preprocessing (``Qwen2VLImageProcessorFast``) and the chat prompt.
2. ``vision.xml``: the vision tower with the exported camera grid baked in.
3. Host glue: token embedding lookup, image-token scatter, dense DeepStack features, mRoPE ids.
4. ``llm.xml``: the text model, returning the *pre-norm* last hidden state StarVLA conditions on.
5. ``head.xml``: the flow-matching action head with the sampled noise as an input.
6. StarVLA's state normalization and action unnormalization.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

FORMAT_VERSION = 1
BACKEND = "openvino"
DESCRIPTOR = "package.json"
MODELS = ("vision", "llm", "head")
EMBEDDINGS = "embed_tokens.npy"
TOKENIZER = "tokenizer.json"
WEIGHT_FORMATS = ("fp32", "fp16", "int8")
NORMALIZATION_MODES = ("q99", "mean_std", "min_max", "binary")
_MAX_ASPECT_RATIO = 200


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Return Qwen-VL's resized image size: multiples of ``factor`` within the pixel budget.

    Returns:
        Resized (height, width).

    Raises:
        ValueError: If the aspect ratio is extreme.
    """
    if max(height, width) / min(height, width) > _MAX_ASPECT_RATIO:
        msg = "Image aspect ratio must be smaller than 200."
        raise ValueError(msg)
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def image_patches(image: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Turn one HWC uint8 RGB image into Qwen-VL's flattened patch rows.

    Matches ``Qwen2VLImageProcessorFast`` (fused rescale/normalize, frame repeated to the
    temporal patch size, merge-window patch order). A resize uses OpenCV bicubic, which is close
    to but not bit-identical with torchvision; the export records the measured difference.

    Returns:
        float32 array of shape (grid_h * grid_w, channels * temporal * patch * patch).
    """
    resized = tuple(spec["resized"])
    if tuple(image.shape[:2]) != resized:
        import cv2  # Only needed when the camera size is not a patch multiple.

        image = cv2.resize(image, (resized[1], resized[0]), interpolation=cv2.INTER_CUBIC)
    # Fused rescale/normalize, as in transformers: (x - 255 * mean) / (255 * std).
    pixels = (image.astype(np.float32) - np.float32(255.0) * np.asarray(spec["mean"], dtype=np.float32)) / (
        np.float32(255.0) * np.asarray(spec["std"], dtype=np.float32)
    )
    pixels = pixels.transpose(2, 0, 1)
    patch, temporal, merge = spec["patch_size"], spec["temporal_patch_size"], spec["merge_size"]
    channels, height, width = pixels.shape
    grid_h, grid_w = height // patch, width // patch
    frames = np.repeat(pixels[None], temporal, axis=0)
    frames = frames.reshape(temporal, channels, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    # (grid_h/m, grid_w/m, m_h, m_w, channel, temporal, patch_h, patch_w), as in transformers.
    frames = frames.transpose(2, 5, 3, 6, 1, 0, 4, 7)
    return np.ascontiguousarray(frames.reshape(grid_h * grid_w, channels * temporal * patch * patch))


def rope_index(  # ruff: ignore[too-many-locals]  # Mirrors transformers' get_rope_index.
    input_ids: np.ndarray,
    grids: list[list[int]],
    spec: dict[str, Any],
) -> np.ndarray:
    """Qwen3-VL multimodal RoPE position ids for one unpadded image+text sequence.

    Text tokens advance all three axes together; each image's tokens take (t, h, w) grid
    coordinates offset by the preceding text, as in ``Qwen3VLModel.get_rope_index``.

    Returns:
        int64 array of shape (3, 1, sequence_length).

    Raises:
        ValueError: If image tokens do not match the image grids.
    """
    tokens = input_ids.reshape(-1).tolist()
    merge = spec["merge_size"]
    image_token = spec["image_token_id"]
    parts: list[np.ndarray] = []
    start = 0
    next_position = 0
    for grid in grids:
        try:
            end = tokens.index(image_token, start)
        except ValueError:
            msg = "Prompt has fewer image tokens than images."
            raise ValueError(msg) from None
        frames, height, width = grid[0], grid[1] // merge, grid[2] // merge
        text = end - start
        parts.append(np.broadcast_to(np.arange(text, dtype=np.int64) + next_position, (3, text)))
        t_index = np.repeat(np.arange(frames, dtype=np.int64), height * width)
        h_index = np.tile(np.repeat(np.arange(height, dtype=np.int64), width), frames)
        w_index = np.tile(np.arange(width, dtype=np.int64), frames * height)
        parts.append(np.stack([t_index, h_index, w_index]) + text + next_position)
        next_position = int(parts[-1].max()) + 1
        start = end + frames * height * width
    if image_token in tokens[start:]:
        msg = "Prompt has more image tokens than images."
        raise ValueError(msg)
    text = len(tokens) - start
    if text:
        parts.append(np.broadcast_to(np.arange(text, dtype=np.int64) + next_position, (3, text)))
    positions = np.concatenate(parts, axis=1)
    if positions.shape[1] != len(tokens):
        msg = "Image tokens do not match the image grids."
        raise ValueError(msg)
    return positions[:, None, :]


def normalize(values: np.ndarray, spec: dict[str, Any], threshold: float) -> np.ndarray:
    """StarVLA ``Normalizer.forward`` in numpy (float32).

    Returns:
        Normalized values with the input's shape.

    Raises:
        ValueError: If the mode is unknown.
    """
    x = np.asarray(values, dtype=np.float32)
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in spec["stats"].items()}
    mode = spec["mode"]
    if mode == "binary":
        return (x > threshold).astype(np.float32)
    if mode == "mean_std":
        mask = stats["std"] != 0
        return np.where(mask, (x - stats["mean"]) / np.where(mask, stats["std"], 1), x).astype(np.float32)
    if mode in {"q99", "min_max"}:
        low, high = (stats["q01"], stats["q99"]) if mode == "q99" else (stats["min"], stats["max"])
        mask = low != high
        scaled = 2 * ((x - low) / np.where(mask, high - low, 1)) - 1
        if mode == "q99":
            return np.clip(np.where(mask, scaled, x), -2.2, 2.2).astype(np.float32)
        return np.where(mask, scaled, 0).astype(np.float32)
    msg = f"Unsupported normalization mode {mode!r}."
    raise ValueError(msg)


def unnormalize(values: np.ndarray, spec: dict[str, Any], threshold: float) -> np.ndarray:
    """StarVLA ``Normalizer.inverse`` in numpy (float32).

    Returns:
        Unnormalized values with the input's shape.

    Raises:
        ValueError: If the mode is unknown.
    """
    x = np.asarray(values, dtype=np.float32)
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in spec["stats"].items()}
    mode = spec["mode"]
    if mode == "binary":
        return (x > threshold).astype(np.float32)
    if mode == "mean_std":
        return (x * stats["std"] + stats["mean"]).astype(np.float32)
    if mode in {"q99", "min_max"}:
        low, high = (stats["q01"], stats["q99"]) if mode == "q99" else (stats["min"], stats["max"])
        return ((x + 1) / 2 * (high - low) + low).astype(np.float32)
    msg = f"Unsupported normalization mode {mode!r}."
    raise ValueError(msg)


def load_descriptor(artifact: Path) -> dict[str, Any]:
    """Read an OpenVINO package descriptor and check the fields the pipeline relies on.

    Returns:
        Descriptor dictionary.

    Raises:
        ValueError: If the descriptor is not a supported OpenVINO StarVLA package.
    """
    descriptor = json.loads((artifact / DESCRIPTOR).read_text())
    required = {"image", "prompt", "tokens", "embedding", "state", "action", "normalization", "weights"}
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("format_version") != FORMAT_VERSION
        or descriptor.get("backend") != BACKEND
        or not required <= set(descriptor)
        or descriptor["weights"] not in WEIGHT_FORMATS
    ):
        msg = "Unsupported StarVLA OpenVINO package descriptor."
        raise ValueError(msg)
    modes = {spec.get("mode") for spec in descriptor["normalization"]["keys"].values()}
    if not modes <= set(NORMALIZATION_MODES):
        msg = f"Unsupported normalization modes {sorted(modes - set(NORMALIZATION_MODES))}."
        raise ValueError(msg)
    return descriptor


class EmbeddingTable:
    """Token embeddings stored as float32, or as raw bfloat16 bits (exact, half the size)."""

    def __init__(self, path: Path, dtype: str) -> None:
        """Memory-map the table; rows are converted on lookup.

        Raises:
            ValueError: If the stored dtype is unsupported.
        """
        if dtype not in {"float32", "bfloat16"}:
            msg = f"Unsupported embedding dtype {dtype!r}."
            raise ValueError(msg)
        self._table = np.load(path, mmap_mode="r", allow_pickle=False)
        self._bfloat16 = dtype == "bfloat16"

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        """Return float32 embeddings for token ids of shape (L,)."""
        rows = np.asarray(self._table[ids])
        if self._bfloat16:
            return (rows.astype(np.uint32) << 16).view(np.float32)
        return rows.astype(np.float32, copy=False)


class QwenGR00TPipeline:
    """Run an exported QwenGR00T package with OpenVINO on one observation."""

    def __init__(self, artifact: Path, device: str = "CPU", config: dict[str, Any] | None = None) -> None:
        """Compile the three graphs and load the tokenizer and embedding table."""
        import openvino as ov  # Keep the module importable for pure-numpy helpers.
        from tokenizers import Tokenizer

        self.artifact = artifact
        self.descriptor = load_descriptor(artifact)
        core = ov.Core()
        self.models = {name: core.compile_model(artifact / f"{name}.xml", device, config or {}) for name in MODELS}
        embedding = self.descriptor["embedding"]
        self.embeddings = EmbeddingTable(artifact / EMBEDDINGS, embedding["dtype"])
        self.tokenizer = Tokenizer.from_file(str(artifact / TOKENIZER))

    # Preprocessing -----------------------------------------------------------------------
    def input_ids(self, task: str, cameras: int) -> np.ndarray:
        """Tokenize the chat prompt with each image placeholder expanded to its token count.

        Returns:
            int64 token ids of shape (1, L).

        Raises:
            ValueError: If the prompt's image placeholders differ from the camera count.
        """
        prompt, image = self.descriptor["prompt"], self.descriptor["image"]
        text = prompt["prefix"] + task + prompt["suffix"]
        placeholder = prompt["image_token"]
        if text.count(placeholder) != cameras:
            msg = "Prompt template image placeholders differ from the camera count."
            raise ValueError(msg)
        text = text.replace(placeholder, "\0").replace("\0", placeholder * image["tokens"])
        return np.asarray([self.tokenizer.encode(text).ids], dtype=np.int64)

    def prepare(self, images: list[np.ndarray], task: str) -> dict[str, np.ndarray]:
        """Build the vision and text inputs for HWC uint8 RGB images in training camera order.

        Returns:
            ``pixel_values``, ``input_ids`` and ``position_ids``.

        Raises:
            ValueError: If an image does not have the exported camera size.
        """
        spec = self.descriptor["image"]
        if len(images) != spec["cameras"]:
            msg = f"Expected {spec['cameras']} camera images, got {len(images)}."
            raise ValueError(msg)
        for image in images:
            if tuple(image.shape) != (*spec["size"], 3) or image.dtype != np.uint8:
                msg = f"Camera images must be uint8 {spec['size'][0]}x{spec['size'][1]}x3 for this export."
                raise ValueError(msg)
        pixel_values = np.concatenate([image_patches(image, spec) for image in images])
        input_ids = self.input_ids(task, len(images))
        grids = [list(spec["grid"])] * len(images)
        position_ids = rope_index(input_ids, grids, {**spec, **self.descriptor["tokens"]})
        return {"pixel_values": pixel_values, "input_ids": input_ids, "position_ids": position_ids}

    def normalize_state(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Normalize raw state groups in training order.

        Returns:
            float32 state of shape (1, 1, state_dim), or an empty (1, 1, 0) array without state.
        """
        normalization = self.descriptor["normalization"]
        parts = []
        for key, width in self.descriptor["state"]["keys"]:
            value = np.asarray(state[key], dtype=np.float32).reshape(1, width)
            spec = normalization["keys"].get(key)
            parts.append(normalize(value, spec, normalization["binary_threshold"]) if spec else value)
        return np.concatenate(parts, axis=-1)[None] if parts else np.zeros((1, 1, 0), np.float32)

    def unnormalize_actions(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        """Split a normalized (horizon, action_dim) chunk into unnormalized action groups.

        Returns:
            Action groups keyed like the training DataConfig.
        """
        normalization = self.descriptor["normalization"]
        result = {}
        cursor = 0
        for key, width in self.descriptor["action"]["keys"]:
            value = actions[:, cursor : cursor + width]
            spec = normalization["keys"].get(key)
            result[key] = unnormalize(value, spec, normalization["binary_threshold"]) if spec else value
            cursor += width
        return result

    # Model -------------------------------------------------------------------------------
    def vision(self, pixel_values: np.ndarray) -> list[np.ndarray]:
        """Return image embeddings followed by the DeepStack features."""
        return list(self.models["vision"]([pixel_values]).values())

    def language(self, input_ids: np.ndarray, position_ids: np.ndarray, vision: list[np.ndarray]) -> np.ndarray:
        """Scatter image features into the token embeddings and run the text model.

        Returns:
            Pre-norm last hidden state of shape (1, L, hidden).
        """
        ids = input_ids[0]
        embeds = self.embeddings(ids)[None].copy()
        positions = ids == self.descriptor["tokens"]["image_token_id"]
        embeds[0, positions] = vision[0]
        deepstack = np.zeros((len(vision) - 1, *embeds.shape), dtype=np.float32)
        for index, features in enumerate(vision[1:]):
            deepstack[index, 0, positions] = features
        mask = np.ones_like(input_ids)
        return self.models["llm"]([embeds, mask, position_ids, deepstack])[0]

    def head(self, hidden: np.ndarray, state: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Run the unrolled flow-matching action head on normalized state and explicit noise.

        Returns:
            Normalized actions of shape (1, horizon, action_dim).
        """
        mask = np.ones(hidden.shape[:2], dtype=bool)
        return self.models["head"]([hidden, mask, state, noise.astype(np.float32)])[0]

    def predict_normalized(
        self,
        images: list[np.ndarray],
        task: str,
        state: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        """Return normalized actions of shape (1, horizon, action_dim)."""
        inputs = self.prepare(images, task)
        hidden = self.language(inputs["input_ids"], inputs["position_ids"], self.vision(inputs["pixel_values"]))
        return self.head(hidden, state, noise)

    def noise(self, rng: np.random.Generator) -> np.ndarray:
        """Sample the head's initial noise.

        Returns:
            float32 noise of shape (1, horizon, action_dim).
        """
        action = self.descriptor["action"]
        return rng.standard_normal((1, action["horizon"], action["dim"]), dtype=np.float32)
