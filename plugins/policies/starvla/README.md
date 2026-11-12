# StarVLA engine

Train StarVLA **QwenGR00T** with its upstream trainer and run exported checkpoints
through Physical AI Runtime. StarVLA, PyTorch, Transformers, and its dataset code
live in a separate locked Python 3.10/CUDA 12.8 environment. Neither the host nor
a Runtime-only install imports `physicalai-train` to perform inference.

This initial integration supports single-GPU QwenGR00T, upstream registered
LeRobot dataset mixtures, one observation per camera, and native safetensors
packages. Other StarVLA frameworks, custom executable dataset registrations,
DeepSpeed/multi-GPU launching, and ONNX/OpenVINO conversion are not enabled.
Training and inference require CUDA; CPU/XPU are not advertised.

## Install and prepare

From `library/`:

```bash
uv sync --extra starvla
uv run physicalai engine prepare starvla --accelerator cu128
uv run physicalai train starvla --help
```

The worker locks StarVLA to
`13d8ad43eb4128554c57227507055ecf9c3a62a2`. Upstream's wheel includes its
benchmark dataset registries and deployment wrapper. No downstream code patches
are applied. Transformers 4.57.1 and PyTorch 2.7.1/torchvision 0.22.1 provide the
Qwen3-VL/CUDA 12.8 combination; these differ from upstream's older requirements.

Prepare a **local, materialized** `Qwen/Qwen3-VL-4B-Instruct` backbone, including
its tokenizer, processor, and safetensors weights. Use a specific Hugging Face
commit when downloading; do not use a floating revision. Qwen's standard
Transformers implementation needs no remote Python code. The plugin refuses
dynamic remote model code and runs model loading offline. Its directory path
must contain `Qwen3-VL`: upstream dispatches the VLM by this substring. For example,
run with the library's Python and `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-VL-4B-Instruct",
    revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md", "*.jinja", "*.model"],
    local_dir="/absolute/path/to/Qwen3-VL-4B-Instruct",
)
```

This revision is ungated and contains 4,437,815,808 BF16 parameters (~71 GB for float32 full-fine-tuning
parameters/gradients/optimizer state alone). The example freezes
`qwen_vl_interface` and uses batch size one. The LIBERO smoke run used about
12.5 GiB on an RTX 3090 (sampled usage, not a measured peak); larger images or
batches need more memory. Check checkpoint access/license terms before downloading.

## Train a 20-step smoke run

The example uses LIBERO goal: the dataset root must contain
`libero_goal_no_noops_1.0.0_lerobot/` with StarVLA's LeRobot data and modality
metadata. Use only trusted local datasets: upstream reads serialized dataset
caches. Camera/state/action order comes from its registered
`libero_franka` DataConfig, not guessed from camera names. The verified dataset
was `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` at revision
`222cf888ed360fad0a5f983748c1cc40743d43e7`, with `meta/modality.json` copied from
the pinned StarVLA repository's
`examples/simBenchmarks/LIBERO/train_files/modality.json`.

From the repository root, after installation:

```bash
WANDB_MODE=disabled library/.venv/bin/physicalai train starvla \
  --config_yaml plugins/policies/starvla/configs/qwengroot_libero.yaml \
  framework.qwenvl.base_vlm=/absolute/path/to/Qwen3-VL-4B-Instruct \
  datasets.vla_data.data_root_dir=/absolute/path/to/datasets \
  run_root_dir=/absolute/path/to/runs run_id=starvla
```

Model overrides use `KEY=VALUE` or `--KEY=VALUE`; `--KEY VALUE` is not supported.
The smoke config uses `num_workers: 0` to avoid a video-decoder fork stall observed
after model initialization. Increase it only after checking your loader setup.

The plugin forces safetensors output. It preserves the native trainer, emits
step metrics/cancellation events for the headless engine API, and writes
`final_model/model.safetensors`, full `config.yaml`, `dataset_statistics.json`,
and `physicalai_run.json` in the run directory. Native resume settings pass
through as `trainer.is_resume=true`; upstream restores model weights and step
count, **not an exact optimizer/RNG-state resume**.

Headless Studio/Python callers use:

```python
from pathlib import Path
from physicalai.engine import Engine

run = Engine.get("starvla").run(
    {
        "config_yaml": "plugins/policies/starvla/configs/qwengroot_libero.yaml",
        "overrides": ["framework.qwenvl.base_vlm=/absolute/path/to/Qwen3-VL-4B-Instruct"],
    },
    dataset=Path("/absolute/path/to/datasets"),  # mixture root, not one dataset
    output_dir=Path("/absolute/path/to/runs/starvla"),
    device="cuda:0",
)
```

## Longer action-head training

Use `configs/qwengroot_libero_long.yaml` for a 10,000-step run: batch size four,
500 warmup steps, and checkpoints every 1,000 steps. The Qwen backbone remains
frozen, and the video loader stays single-process. Supply the same local
backbone and dataset-root overrides as for the smoke run, and choose a new
`run_id` so its outputs remain separate.

To warm-start from an existing run, add
`trainer.pretrained_checkpoint=/absolute/path/to/model.safetensors`. This loads
the weights but starts fresh AdamW and scheduler state; it is not an exact
resume. The configured 10,000 steps are additional updates. Allow roughly
130 GiB of extra disk space for ten full checkpoints, the final model, and a
Runtime export. Judge the resulting policy by simulation task success, not
training loss alone.

## Export with explicit robot mappings

Review `configs/libero_io.json` against your actual observations. It specifies
ordered camera views, eight state coordinates (including LIBERO's pad), and
seven action coordinates. State inputs use **raw training units**; the worker
applies the training state transforms and unnormalizes actions with StarVLA's
`PolicyNormProcessor`. No client-side action normalization is needed. The worker
rejects camera order or state/action widths that differ from the checkpoint.
An empty state mapping is accepted only when the training DataConfig has no state
keys. State transforms other than upstream's `StateActionToTensor` and
`StateActionTransform` are rejected rather than silently bypassed.

```python
import json
from pathlib import Path
from physicalai.engine import Engine
from physicalai_starvla_policy_plugin.package import IoSpec

io = IoSpec.from_dict(json.loads(Path("plugins/policies/starvla/configs/libero_io.json").read_text()))
package = Engine.get("starvla").export_native(
    "/absolute/path/to/runs/starvla/final_model/model.safetensors",
    "exports/starvla",
    backbone_path="/absolute/path/to/Qwen3-VL-4B-Instruct",
    io=io,
    unnorm_key="franka",  # use the actual top-level key in dataset_statistics.json
)
```

Alternatively, copy your reviewed mapping to `<run>/io.json`, then call
`Engine.get("starvla").export(run, "exports/starvla")`. This uses the local
backbone and single normalization key recorded by training. Multi-key statistics
require explicit `export_native(..., unnorm_key=...)`.

Exports contain a Runtime manifest, trained safetensors, config/statistics, I/O
mapping, environment pins, and the complete local backbone/processor. This
intentionally duplicates backbone weights because upstream constructs the base
model before restoring the trained state dict. Packages are offline-loadable
once the worker environment is prepared, and existing exports are never
overwritten. `.pt`/pickle checkpoints and executable package files are refused.
Materialize Hugging Face cache symlinks before export.

## Runtime-only inference

Install `physicalai` and this plugin (without its `[train]` extra) on the Runtime
host. On first load, Runtime's `physicalai.isolate` provisions the same worker;
this requires network access for locked dependencies but not model downloads.
After preparing it, inference runs offline. Run from a writable cache location
with a CUDA-compatible NVIDIA driver.

```python
import numpy as np
from physicalai.inference import InferenceModel

model = InferenceModel("exports/starvla", device="cuda", seed=42)
try:
    actions = model({
        "images": {
            "front": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "state": np.zeros(8, dtype=np.float32),
        "task": "pick up the object",
    })
    model.reset()  # restarts the seeded prediction sequence
finally:
    model.close()
```

The adapter returns `{"action": float32[1, horizon, action_dim]}` to Runtime.
Images may be HWC/CHW uint8 or normalized floating point, batch size one.
Use `model.predict_action_chunk(observation)` for a `(horizon, action_dim)` array,
or Runtime's `model.select_action(observation)` to consume the chunk one action
at a time. This plugin adds no robot-specific safety checks: validate action
units, limits, and control rate before hardware use.

## Verification

```bash
cd library
uv run --no-sync pytest ../plugins/policies/starvla/tests -q
```

Tests exercise engine routing, package validation, I/O conversion, framing,
worker failures, and Runtime loading with a tiny fake worker. They do not prove
model quality, numerical parity, CUDA compatibility, or real robot safety.
Verified with 55 CPU contract tests and a real GPU smoke run on one RTX 3090
24 GB (driver 580.173.02, isolated PyTorch 2.7.1/CUDA 12.8):

- Qwen3-VL-4B backbone frozen; 161.5M action-head parameters trained for 20 steps
  on LIBERO goal, batch size one, with progress events and safetensors checkpoints.
- Native Runtime package exported and loaded; finite action shape `(1, 8, 7)`.
- Native StarVLA wrapper and Runtime predictions matched exactly on one real
  dataset observation (maximum absolute difference `0.0`). Seeded reset also
  reproduced the prediction exactly.
- A separate Runtime-only host environment, with no `physicalai-train`, PyTorch,
  or StarVLA installed, ran the same real GPU prediction through the managed
  isolate. Its output also matched the native reference exactly. Runtime's
  `predict_action_chunk`, `select_action`, reset, and close APIs worked.
- Training, checkpointing, export, native reference, and Runtime validation took
  69 seconds, excluding environment preparation and model/dataset downloads.

Both wheel and source distributions include the locked worker project. The plugin
handles Qwen3's tied embedding/head tensors when writing safetensors, keeping
both keys. The CPU suite also covers override parsing, state normalization,
transport errors, and Runtime-only imports.

A subsequent simulator evaluation of this same **20-step smoke checkpoint**
scored **0/30 task successes**: all 10 LIBERO-goal tasks, three fixed initial
states each, seed 7, 10 settling steps, and 300 policy steps per episode. It used
`hf-libero==0.1.3`, robosuite 1.4.0, MuJoCo 3.2.3, both cameras at the training
resolution of 256 pixels with LIBERO's 180-degree image rotation, raw 8D state,
and Runtime's 8-action chunk execution. Gripper commands used StarVLA's LIBERO
conversion, `1 - 2 * (value > 0.5)`; OSC controls were clipped to `[-1, 1]`.
Success came from `env.check_success()`, not reward or timeout. All episodes
completed without simulator/inference errors and reached the step limit;
63.9% of steps needed control clipping. Failure videos confirmed robot motion
but no task completion. This checkpoint validates execution, not a task-capable
policy, and its score does not represent a fully trained StarVLA model.

Longer training, a larger success-rate benchmark, real-robot safety, and parity
across a representative observation set remain unchecked. Run the opt-in
execution-path test on your CUDA host:

```bash
export PHYSICALAI_RUN_STARVLA_E2E=1
export PHYSICALAI_STARVLA_BACKBONE=/absolute/path/to/Qwen3-VL-4B-Instruct
export PHYSICALAI_STARVLA_DATASETS=/absolute/path/to/datasets
export PHYSICALAI_STARVLA_OBSERVATION=/absolute/path/to/observation.npz
uv run --no-sync pytest ../plugins/policies/starvla/tests/integration -q
```

The observation NPZ contains `front`, `wrist`, `state`, and scalar string `task`
arrays. Optional `PHYSICALAI_STARVLA_CONFIG`, `PHYSICALAI_STARVLA_IO`, and
`PHYSICALAI_STARVLA_UNNORM_KEY` select a different registered dataset setup.
