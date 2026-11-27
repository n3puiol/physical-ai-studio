# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hand StarVLA runs to RLinf for reinforcement-learning post-training, and import its results back.

RLinf (https://github.com/RLinf/RLinf) fine-tunes StarVLA policies with reinforcement learning in
simulation (for example GRPO on LIBERO) from its own environment and installer. It loads a StarVLA
policy only as a ``.pt`` file in StarVLA's run layout, and writes its trained policy as ``.pt`` files
too, with the StarVLA weights under ``starvla_model.``. The plugin reads and writes safetensors only
and never unpickles a file: the two conversions between ``.pt`` and safetensors run in RLinf's own
environment (see the README).

- :func:`export_for_rlinf` writes a plugin run in StarVLA's run layout
  (``<out>/checkpoints/steps_0_model.safetensors`` next to ``config.yaml`` and
  ``dataset_statistics.json``); converted to a ``.pt`` beside it, that is RLinf's
  ``actor.model.model_path``.
- :func:`import_from_rlinf` takes RLinf's trained policy converted to safetensors, checks every
  weight against the run RLinf started from, and writes a plugin run that
  ``Engine.get("starvla").export`` packages as usual.

Both need ``torch`` and ``safetensors`` (the plugin's ``train`` extra). Usage::

    python -m physicalai_starvla_policy_plugin.rlinf export <run_dir> <rlinf_dir>
    python -m physicalai_starvla_policy_plugin.rlinf import <rlinf_policy.safetensors> <run_dir> <new_run_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from . import environments
from ._worker.starvla_worker_common import DEFAULT_BACKBONE
from .package import RUN_INFO, framework_from_config

if TYPE_CHECKING:
    from collections.abc import Sequence

# StarVLA frameworks RLinf's StarVLA wrapper runs (its OFT and flow-matching handlers, on a VLM).
RLINF_FRAMEWORKS = frozenset({"QwenGR00T", "QwenOFT", "QwenPI"})
# Where RLinf's policy wrapper keeps the StarVLA model; its other parameters (``actor_logstd``, a
# value head) are RLinf's own.
RLINF_PREFIX = "starvla_model."
_CHECKPOINT = Path("final_model") / "model.safetensors"
_RUN_FILES = ("config.yaml", "config.full.yaml", "dataset_statistics.json", "io.json")


def export_for_rlinf(run_dir: Path, output: Path) -> Path:
    """Write a plugin run in StarVLA's run layout, for RLinf to start from.

    Returns:
        The safetensors checkpoint; its ``.pt`` conversion beside it is RLinf's model path.

    Raises:
        ValueError: If RLinf cannot run the run's framework or backbone.
        FileExistsError: If the output exists.
    """
    run_dir = run_dir.expanduser().resolve(strict=True)
    framework = framework_from_config(run_dir / "config.yaml")
    record = json.loads((run_dir / RUN_INFO).read_text())
    backbone = environments.backbone_for(framework, record.get("backbone_path"))
    if framework not in RLINF_FRAMEWORKS or backbone != DEFAULT_BACKBONE:
        supported = ", ".join(sorted(RLINF_FRAMEWORKS))
        msg = f"RLinf's StarVLA wrapper runs {supported} on Qwen3-VL; this run is {framework} on {backbone}."
        raise ValueError(msg)
    output = output.expanduser().absolute()
    if output.exists():
        msg = f"{output} already exists."
        raise FileExistsError(msg)
    (output / "checkpoints").mkdir(parents=True)
    for name in ("config.yaml", "dataset_statistics.json"):
        shutil.copy2(run_dir / name, output / name)
    checkpoint = output / "checkpoints" / "steps_0_model.safetensors"
    shutil.copy2(run_dir / _CHECKPOINT, checkpoint)
    return checkpoint


def import_from_rlinf(weights: Path, run_dir: Path, output: Path) -> Path:
    """Write a plugin run holding the policy RLinf trained from ``run_dir``.

    ``weights`` is RLinf's trained policy converted to safetensors in RLinf's environment. Its
    StarVLA weights must match the starting run's in name and shape, and they keep the starting
    run's dtypes; RLinf's own parameters are left out.

    Returns:
        The new run directory.

    Raises:
        ValueError: If the file is not safetensors or does not hold the starting run's StarVLA weights.
        FileExistsError: If the output exists.
    """
    from safetensors.torch import load_file, save_file  # ruff: ignore[import-outside-top-level]  # Train extra.

    weights = Path(weights).expanduser().resolve(strict=True)
    if weights.suffix != ".safetensors":
        msg = f"{weights.name}: the plugin imports safetensors only; convert RLinf's .pt in RLinf's environment."
        raise ValueError(msg)
    run_dir = run_dir.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if output.exists():
        msg = f"{output} already exists."
        raise FileExistsError(msg)
    loaded = load_file(str(weights))
    # RLinf's policy wrapper keeps the StarVLA model under a prefix; a bare StarVLA state dict also works.
    prefixed = any(key.startswith(RLINF_PREFIX) for key in loaded)
    trained = {
        key.removeprefix(RLINF_PREFIX): value
        for key, value in loaded.items()
        if key.startswith(RLINF_PREFIX) or not prefixed
    }
    base = load_file(str(run_dir / _CHECKPOINT))
    missing, unexpected = sorted(set(base) - set(trained)), sorted(set(trained) - set(base))
    if missing or unexpected:
        msg = (
            f"{weights.name} does not hold the starting run's StarVLA weights: "
            f"{len(missing)} missing (e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})."
        )
        raise ValueError(msg)
    reshaped = [key for key, value in base.items() if tuple(trained[key].shape) != tuple(value.shape)]
    if reshaped:
        msg = f"{len(reshaped)} weights changed shape, e.g. {reshaped[:3]}."
        raise ValueError(msg)
    (output / _CHECKPOINT.parent).mkdir(parents=True)
    save_file({key: trained[key].to(base[key].dtype).contiguous() for key in base}, str(output / _CHECKPOINT))
    for name in _RUN_FILES:
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, output / name)
    record = json.loads((run_dir / RUN_INFO).read_text())
    record["post_training"] = {
        "framework": "rlinf",
        "started_from": str(run_dir / _CHECKPOINT),
        "weights": str(weights),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    (output / RUN_INFO).write_text(json.dumps(record, indent=2) + "\n")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    """Command line for :func:`export_for_rlinf` and :func:`import_from_rlinf`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Write a plugin run in StarVLA's layout, for RLinf")
    export.add_argument("run_dir", type=Path)
    export.add_argument("output", type=Path)
    imported = commands.add_parser("import", help="Write a plugin run from RLinf's policy, as safetensors")
    imported.add_argument("weights", type=Path)
    imported.add_argument("run_dir", type=Path, help="The run RLinf started from")
    imported.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        print(export_for_rlinf(args.run_dir, args.output))  # ruff: ignore[print]
    else:
        print(import_from_rlinf(args.weights, args.run_dir, args.output))  # ruff: ignore[print]


if __name__ == "__main__":
    main()
