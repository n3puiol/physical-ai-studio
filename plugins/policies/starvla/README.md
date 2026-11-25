# StarVLA engine

Train StarVLA **QwenGR00T**, **QwenPI** and **QwenOFT** with the upstream trainers, including
co-training with VLM data, and run exported checkpoints through Physical AI Runtime. StarVLA, PyTorch, Transformers, and its dataset code
live in a separate locked Python 3.11/CUDA 12.8 environment. Neither the host nor
a Runtime-only install imports `physicalai-train` to perform inference.

This integration supports single-GPU QwenGR00T, QwenPI and QwenOFT on Qwen3-VL, supervised
training and co-training with VLM data, upstream registered LeRobot dataset mixtures, one observation per camera, native safetensors
packages (CUDA inference in the isolated worker), and, for QwenGR00T, OpenVINO packages
(in-process inference on CPU or another OpenVINO device, without StarVLA or PyTorch). Other
StarVLA frameworks, custom executable dataset registrations, DeepSpeed/multi-GPU
launching, and ONNX conversion are not enabled. Training requires CUDA.

## Install and prepare

From `library/`:

```bash
uv sync --extra starvla
uv run physicalai engine prepare starvla --policy QwenGR00T --accelerator cu128  # QwenPI and QwenOFT share it
uv run physicalai engine prepare starvla --enablement qwengr00t-openvino  # OpenVINO conversion
uv run physicalai train starvla --help
```

Each StarVLA framework maps to a locked worker project under
`src/physicalai_starvla_policy_plugin/envs/<name>/`, declared in `environments.POLICY_ENVIRONMENTS`
and mapped through `StarVLAEngine.policy_envs`. Training, export and Runtime pick the environment
from `framework.name` in the training YAML (a `framework.name=...` override wins). Frameworks with
the same dependencies share a project: QwenPI and QwenOFT use QwenGR00T's `envs/qwengr00t/`, so the engine
lists and prepares it once. A framework that needs other dependencies gets its own project, so
frameworks never share a resolution they were not locked for. There is no default environment: a
framework without an entry is rejected, and the training worker refuses a config whose
`framework.name` differs from the framework the engine chose its environment for.

To add a framework that the pinned StarVLA wheel already runs, add a `PolicyEnvironment` entry with
the shared `name`. For other dependencies, copy `envs/qwengr00t/`, change them, lock the copy with
`library/.venv/bin/uv lock --project <dir>`, and add an entry for it. `test_duplicated_pins_agree`
checks each entry against its project and lock.

The shared worker (`envs/qwengr00t/`) locks StarVLA to
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

StarVLA feeds proprioceptive state to the model only when `datasets.vla_data.include_state=true`.
It is off by default, in StarVLA's own LIBERO recipe and in the bundled configs, and packages follow
the checkpoint: one trained without state maps none, so Runtime neither needs nor sends it, and one
trained with state needs it in every observation. LIBERO's `libero_franka` DataConfig normalizes
only actions, so with `include_state=true` the eight state values reach the model in raw units.

Verified on one RTX 3090. For the 20-step smoke checkpoint, trained without state:

- The native package maps no state, and Runtime matches StarVLA's own `predict_action` without
  state exactly.
- A package exported earlier, which still maps state, now gives the same actions.
- The OpenVINO export validates with a head that has no state input.

For a 20-step Qwen3-VL-2B run with `include_state=true`:

- The package keeps the state mapping, and Runtime matches StarVLA with state exactly.
- Runtime refuses an observation without state, and the OpenVINO head has a state input.

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

## QwenPI

QwenPI predicts actions with a layer-wise flow-matching DiT: one block per VLM layer, each
cross-attending to that layer's hidden states at the VLM's width. Its action head therefore grows
with the backbone (counted from the pinned StarVLA code; training state is fp32 weights, gradients
and AdamW moments, without activations):

| Backbone    | VLM layers × width | Action head | Full training state |
| ----------- | ------------------ | ----------- | ------------------- |
| Qwen3-VL-2B | 28 × 2048          | 1.68B       | ~25 GiB             |
| Qwen3-VL-4B | 36 × 2560          | 3.35B       | ~50 GiB             |
| Qwen3-VL-8B | 36 × 4096          | 8.57B       | ~128 GiB            |

`configs/qwenpi_libero.yaml` is a 20-step LIBERO smoke run sized for one 24 GB GPU: Qwen3-VL-2B,
with the backbone and the DiT blocks frozen (`trainer.freeze_modules:
qwen_vl_interface,action_model.model.transformer_blocks`), so only the head's 32M-parameter
encoders, decoders and embeddings train. The DiT blocks start from random weights, so this run checks
the integration and cannot produce a useful policy. To train QwenPI properly, use a GPU with room for
the whole head and drop `action_model.model.transformer_blocks` from `freeze_modules`.

Download the backbone as above, with `Qwen/Qwen3-VL-2B-Instruct` at revision
`89644892e4d85e24eaac8bacfd4f463576704203` (ungated, 4.3 GB of safetensors), then:

```bash
WANDB_MODE=disabled library/.venv/bin/physicalai train starvla \
  --config_yaml plugins/policies/starvla/configs/qwenpi_libero.yaml \
  framework.qwenvl.base_vlm=/absolute/path/to/Qwen3-VL-2B-Instruct \
  datasets.vla_data.data_root_dir=/absolute/path/to/datasets \
  run_root_dir=/absolute/path/to/runs run_id=starvla-qwenpi
```

Runs and packages record the shared environment's pins. Export and Runtime inference work as for
QwenGR00T, but only as native packages: `backend="openvino"` is refused, because the OpenVINO
conversion is written for QwenGR00T's graphs.

Verified on one RTX 3090 with the smoke config: 20 steps through the engine took 40 seconds with
progress events, using at most 12.2 GB of GPU memory. The loss fell from 8,784 to 3,379; QwenGR00T's
smoke run starts at 1.31. The native package (11.6 GB checkpoint) exported in 18
seconds. On a real LIBERO-goal frame, Runtime's actions matched StarVLA's own `predict_action`
exactly (maximum absolute difference `0.0`), at 0.47 seconds per chunk and 11.2 GB of GPU memory, and
a seeded replay after `reset()` was exact. The engine listed no new environment.

QwenPI's initial loss is high because of upstream's head, not this plugin. The DiT cross-attends to
raw Qwen3-VL hidden states, which from the third layer on carry a massive activation of about 14,800
(RMS about 32). The head also calls the DiT with `return_pre_output=True`, which skips the DiT's
final LayerNorm, so nothing bounds the residual stream: its RMS grows from 0.5 to 320 over the 28
blocks, and the decoder predicts velocities with RMS 74 against a target of 1.1. On real LIBERO-goal
batches at initialisation, QwenPI's loss is about 5,200 and QwenGR00T's about 1.35. QwenGR00T reads
the same raw states but normalises the DiT output. Normalising either the DiT output or the VLM
states brings QwenPI's initial loss to 1.2–1.4. The plugin runs upstream unchanged, so trained
packages stay faithful to StarVLA; whether the scale slows or harms full QwenPI training was not
measured.

## QwenOFT

QwenOFT appends eight copies of a query token to the prompt and regresses the action chunk from
their last hidden states with a small MLP and an L1 loss. Its predictions involve no noise, so they
are deterministic. When a checkpoint was trained with state, QwenOFT writes the state into the
prompt as discretized text.

`configs/qwenoft_libero.yaml` is a 20-step LIBERO smoke run on Qwen3-VL-4B with the VLM frozen, so
only the MLP trains. Upstream's OFT recipes also train the VLM, which needs more than 24 GB, so this
run checks the integration rather than producing a useful policy. QwenOFT exports native packages
only.

Verified on one RTX 3090: 20 steps through the engine took 23 seconds at 11.0 GB of GPU memory, and
the loss fell from 0.87 to 0.47. On a real LIBERO-goal frame, Runtime matched StarVLA's own
`predict_action` exactly (maximum absolute difference `0.0`) and repeated it exactly, and an
OpenVINO export was refused.

## Co-training with VLM data

A training config with `datasets.vlm_data` runs StarVLA's co-training trainer instead of the
supervised one. Each step adds a language-modelling loss on VLM question-answering data, weighted by
`trainer.loss_scale.vlm`, to the action loss. That loss trains the VLM, so `freeze_modules` must
leave part of `qwen_vl_interface` trainable.

StarVLA names VLM datasets in a Python table with paths relative to its working directory. The
plugin reads them from the config instead. Every name in `dataset_use` (optionally with a `%NN`
sampling suffix) must be listed under `sources`, with an absolute path to a LLaVA-format JSON file
and to its image folder:

```yaml
datasets:
  vlm_data:
    dataset_py: vlm_datasets
    dataset_use: libero_goal_vqa
    sources:
      libero_goal_vqa:
        annotation_path: /absolute/path/to/vqa.json
        data_path: /absolute/path/to/images
    model_type: qwen3vl
```

Each JSON record has an `image` (relative to `data_path`) and `conversations` of `human` and `gpt`
turns, with `<image>` marking where the image goes. Training refuses unknown names, missing or
relative paths, a `model_type` other than `qwen3vl`, a fully frozen VLM and a missing
`trainer.loss_scale.vlm`.

`configs/qwengroot_libero_cotrain.yaml` co-trains QwenGR00T on Qwen3-VL-2B for 20 steps with
upstream's loss weights (VLA 1.0, VLM 0.1). To fit a 24 GB GPU it freezes the vision tower, the
embeddings and the first 26 of 28 LLM layers; upstream's recipes train the whole VLM.

```bash
WANDB_MODE=disabled library/.venv/bin/physicalai train starvla \
  --config_yaml plugins/policies/starvla/configs/qwengroot_libero_cotrain.yaml \
  framework.qwenvl.base_vlm=/absolute/path/to/Qwen3-VL-2B-Instruct \
  datasets.vla_data.data_root_dir=/absolute/path/to/datasets \
  datasets.vlm_data.sources.libero_goal_vqa.annotation_path=/absolute/path/to/vqa.json \
  datasets.vlm_data.sources.libero_goal_vqa.data_path=/absolute/path/to/images \
  run_root_dir=/absolute/path/to/runs run_id=starvla-cotrain
```

Co-trained checkpoints export and run like any other run of their framework.

Three upstream gaps for single-GPU co-training are bridged in the worker:

- StarVLA's co-training trainer builds a DeepSpeed plugin unconditionally. DeepSpeed is not in the
  worker environment, so the worker builds none, as the supervised trainer does in single-GPU mode.
- The VLM loader overwrites the backbone's video-processor sizes, frame counts and fps from
  `video_*` settings, even for image data, and fails when they are unset. Unset ones keep the
  backbone's own values.
- The VLM loader hard-codes four worker processes, which stalled training after model
  initialization. `datasets.vlm_data.num_workers` sets the count; the bundled config uses 0.

Verified on one RTX 3090 with a 40-example question-answering set built from LIBERO-goal frames
("What task is the robot performing?", answered with the episode's instruction). 20 steps took 20
seconds at 8.9 GB of GPU memory and reported the action loss (1.28) and the weighted VLM loss
(0.46) every step. Runtime matched StarVLA's own `predict_action` exactly, and the co-trained
QwenGR00T also exported to OpenVINO (actions within 3.1e-3 of PyTorch).

## Export with explicit robot mappings

Review `configs/libero_io.json` against your actual observations. It specifies
ordered camera views, eight state coordinates (including LIBERO's pad), and
seven action coordinates. State inputs use **raw training units**; the worker
applies the training state transforms and unnormalizes actions with StarVLA's
`PolicyNormProcessor`. No client-side action normalization is needed. The worker
rejects camera order or state/action widths that differ from the checkpoint.
The state mapping is used only if the checkpoint was trained with state (`include_state`);
otherwise the export drops it. Packages exported before the plugin followed `include_state` still
map state, and the worker ignores it for such checkpoints. State transforms other than upstream's
`StateActionToTensor` and
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

## OpenVINO export

`export_native(..., backend="openvino")` (or `export(run, out, backend="openvino")`) converts the
checkpoint on CPU in a second locked environment, `envs/qwengr00t-openvino/`: the training
project plus OpenVINO 2026.4.1 and NNCF, with every shared package at the same locked version.
Adding the converter never changes the training pins.

Runs and native packages record the hash of the training lock. Those made before the workers moved
to Python 3.11 are refused by `export(run)` and by Runtime; re-export their checkpoint with
`export_native(...)`. OpenVINO packages run without a worker and are not affected.

```python
package = Engine.get("starvla").export_native(
    "/absolute/path/to/runs/starvla/final_model/model.safetensors",
    "exports/starvla-openvino",
    backbone_path="/absolute/path/to/Qwen3-VL-4B-Instruct",
    io=io,
    unnorm_key="franka",
    backend="openvino",
    openvino_weights="fp16",  # fp32 | fp16 | int8 (INT8 text model, fp16 vision and head)
    image_size=(256, 256),  # optional: default is the training dataset's camera resolution
)
```

The package holds three graphs and the data needed to run them without StarVLA:

- `vision.xml`: the Qwen3-VL vision tower, with the camera grid for `image_size` baked in.
- `llm.xml`: the text model, returning the **pre-norm** last hidden state (StarVLA conditions on
  `hidden_states[-1]`, before the final RMSNorm).
- `head.xml`: the 4-step flow-matching action head, unrolled, with the noise as an input.
- The token embedding table, `tokenizer.json`, and `package.json` with the prompt template, image
  preprocessing, mRoPE settings, and the state/action normalization statistics.

The checkpoint, backbone weights and Python code are not included. Numpy glue
(`_worker/starvla_openvino_pipeline.py`) reproduces the Qwen3-VL image preprocessing, chat prompt,
multimodal RoPE ids, image-token scatter and StarVLA's normalization. Before writing `validation`
into `package.json`, the export worker runs that pipeline on a synthetic observation and requires it
to match StarVLA's PyTorch `predict_action`:

- Token ids, position ids, pixel values and normalization must match exactly.
- Normalized actions must match within the weight format's tolerance.

The export fails rather than writing an unvalidated package.

| Weights | Package | Actions vs PyTorch (validation) | CPU latency per chunk¹ |
| ------- | ------- | ------------------------------- | ---------------------- |
| fp32    | 17 GB   | max 3e-5                        | 1.09 s                 |
| fp16    | 8.6 GB  | max 9e-3, cosine 0.99999        | 1.08 s                 |
| int8    | 5.2 GB  | max 5e-2, cosine 0.9995         | 0.86 s                 |

¹ i9-10920X, 12 cores, 10,000-step LIBERO checkpoint; PyTorch fp32 on the same CPU takes 1.9 s.

INT8 vision weights were accurate on real frames but drifted on out-of-distribution inputs, so
`int8` compresses only the text model.

On the same 30-episode LIBERO-goal benchmark as the native package, the fp16 OpenVINO package scored
**7/30** on CPU, against **8/30** for native CUDA bf16. The checkpoint, seeds, initial states,
observation handling and action conversion were identical. The noise stream (numpy vs torch) was not,
so episodes differ individually; per task the scores were within one success, and the totals are
statistically indistinguishable. The run took 23 minutes on the CPU, against 6 minutes for native on
an RTX 3090. INT8 and fp32 packages have not been benchmarked closed-loop. Current limitations:

- Fixed camera resolution per export; images of another size are rejected.
- The head takes proprioceptive state only if the checkpoint was trained with it. OpenVINO packages
  exported earlier from checkpoints trained without state still feed an untrained state token;
  re-export them.
- Checkpoints trained with `obs_image_size`, rotation state transforms, or batch size above one
  are not supported.

```python
model = InferenceModel("exports/starvla-openvino", device="CPU", seed=42)  # or GPU / NPU
```

On an Intel GPU (PCI device `0xe223`), the fp16 package of the 10,000-step checkpoint took 0.043 s
per chunk, against 1.7 s on the same machine's CPU. On three LIBERO-goal frames, GPU and CPU actions
differed by 7.9e-4 on average (0.4% of the mean action) and by at most 7.4e-3; both were 0.083 from
the recorded actions on average. GPU inference is not bit-reproducible: a seeded replay after
`reset()` differed by up to 1e-3.

The OpenVINO adapter (`starvla_openvino` backend) runs in the Runtime process. It needs only
`physicalai` and this plugin, imports no PyTorch or StarVLA, and samples noise with numpy: a seed
gives a reproducible sequence that `reset()` restarts. Its predictions are not bit-identical to the
native CUDA worker, which samples noise with PyTorch and runs the VLM in bf16.

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
        "state": np.zeros(8, dtype=np.float32),  # used only if trained with include_state
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
Verified with 140 CPU contract tests and a real GPU smoke run on one RTX 3090
24 GB (driver 580.173.02, isolated PyTorch 2.7.1/CUDA 12.8). After the workers moved to
Python 3.11, the 20-step training, native and OpenVINO export, native/Runtime match and seeded
replay were rerun with the same results. The other measurements below predate that move:

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

Both wheel and source distributions include the locked worker projects. The plugin
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
