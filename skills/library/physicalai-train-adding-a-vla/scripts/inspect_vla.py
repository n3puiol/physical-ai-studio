# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check a LeRobot VLA checkpoint against a dataset before training it through the engine.

Prints the policy type and the worker environment the LeRobot engine will use, the parameter
count and a memory estimate, the checkpoint's and the dataset's camera keys, and a suggested
``--rename_map``. Reads only metadata (config.json, safetensors headers, dataset info).

Usage (from library/):
    .venv/bin/python ../skills/library/physicalai-train-adding-a-vla/scripts/inspect_vla.py \
        lerobot/smolvla_base lerobot/pusht
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_BYTES_PER_PARAM_FULL_FP32 = 16  # weights + gradients + AdamW moments, float32
_GIB = 1024**3


def _checkpoint_config(checkpoint: str) -> dict:
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return json.loads((path / "config.json").read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(checkpoint, "config.json")).read_text(encoding="utf-8"))


def _parameter_count(checkpoint: str) -> int | None:
    path = Path(checkpoint).expanduser()
    try:
        if path.is_dir():
            from safetensors import safe_open

            with safe_open(str(path / "model.safetensors"), framework="np") as handle:
                return sum(math.prod(handle.get_slice(key).get_shape()) for key in handle.keys())
        from huggingface_hub import get_safetensors_metadata

        return sum(get_safetensors_metadata(checkpoint).parameter_count.values())
    except Exception:  # noqa: BLE001  # Size is informative only.
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Hugging Face repo id or local pretrained_model directory")
    parser.add_argument("dataset", nargs="?", help="LeRobot dataset repo id, e.g. lerobot/pusht")
    args = parser.parse_args()

    config = _checkpoint_config(args.checkpoint)
    policy_type = config.get("type")
    print(f"policy type:        {policy_type}")

    from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

    env = LeRobotEngine.environment_for_policy(policy_type)
    print(f"worker environment: {env.name}" + ("  (default: no family environment mapped)" if env.name == "default" else ""))

    params = _parameter_count(args.checkpoint)
    if params is not None:
        print(
            f"parameters:         {params / 1e6:,.0f} M  "
            f"(full float32 fine-tune needs ~{params * _BYTES_PER_PARAM_FULL_FP32 / _GIB:,.0f} GiB before activations)",
        )

    features = config.get("input_features", {})
    cameras = [key for key, feature in features.items() if feature.get("type") == "VISUAL"]
    print(f"checkpoint cameras: {cameras}")
    print(f"empty_cameras:      {config.get('empty_cameras', 0)}")
    if args.dataset is None:
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    dataset = LeRobotDatasetMetadata(args.dataset)
    dataset_cameras = [key for key, value in dataset.features.items() if value["dtype"] in {"video", "image"}]
    print(f"dataset cameras:    {dataset_cameras}")
    unmatched = [camera for camera in dataset_cameras if camera not in cameras]
    if not unmatched:
        print("rename_map:         not needed")
        return
    free_slots = [camera for camera in cameras if camera not in dataset_cameras]
    mapping = dict(zip(unmatched, free_slots, strict=False))
    print(f"suggested:          '--rename_map={json.dumps(mapping)}'  (check the order matches the cameras' roles)")
    if len(unmatched) > len(free_slots):
        print(f"warning:            no checkpoint slot for {unmatched[len(free_slots):]}")


if __name__ == "__main__":
    main()
