# StarVLA engine

Train StarVLA **QwenGR00T**, **QwenPI**, **QwenOFT** and **QwenFast** with the upstream trainers, including
co-training with VLM data, and run exported checkpoints through Physical AI Runtime. StarVLA, PyTorch, Transformers, and its dataset code
live in a separate locked Python 3.11/CUDA 12.8 environment. Neither the host nor
a Runtime-only install imports `physicalai-train` to perform inference.

This integration supports QwenGR00T, QwenPI, QwenOFT and QwenFast on Qwen3-VL (QwenGR00T, QwenPI and
QwenOFT also on Qwen3.5), StarVLA's world-model frameworks on Cosmos-Predict2 (CosmoPredict2GR00T,
CosmoPredict2OFT and CosmoPredict2PI), on one GPU or data-parallel on all visible GPUs of one machine, supervised
training and co-training with VLM data, upstream registered LeRobot dataset mixtures, one observation per camera, native safetensors
packages (CUDA inference in the isolated worker), and, for QwenGR00T, OpenVINO packages
(in-process inference on CPU or another OpenVINO device, without StarVLA or PyTorch). Other
StarVLA frameworks, custom executable dataset registrations, DeepSpeed, multi-machine training
and ONNX conversion are not enabled. Training requires CUDA.

## Install and prepare

From `library/`:

```bash
uv sync --extra starvla
uv run physicalai engine prepare starvla --policy QwenGR00T --accelerator cu128  # shared by every framework but on Qwen3.5
uv run physicalai engine prepare starvla --enablement qwengr00t-openvino  # OpenVINO conversion
uv run physicalai engine prepare starvla --enablement qwen35  # Qwen3.5 backbones (transformers 5)
uv run physicalai train starvla --help
```

Each StarVLA framework maps to a locked worker project under
`src/physicalai_starvla_policy_plugin/envs/<name>/`, declared in `environments.POLICY_ENVIRONMENTS`
and mapped through `StarVLAEngine.policy_envs`. Training, export and Runtime pick the environment
from `framework.name` in the training YAML and the backbone StarVLA loads for
`framework.qwenvl.base_vlm` (overrides win); Qwen3.5 backbones use their own environment (see
[Qwen3.5 backbones](#qwen35-backbones)). Frameworks with
the same dependencies share a project: QwenPI, QwenOFT, QwenFast and the Cosmos-Predict2 frameworks use QwenGR00T's `envs/qwengr00t/`, so the engine
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
Transformers implementation needs no remote Python code. The workers never run Python code from a
model folder or the Hub (see [Reviewed remote code](#reviewed-remote-code)) and load models
offline. Its directory path
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

## QwenFast

QwenFast trains the VLM itself to predict actions. StarVLA encodes each action chunk with the FAST
tokenizer (a discrete cosine transform, then byte-pair encoding) into `<robot_action_N>` tokens and
trains the language model on them; at inference the model generates those tokens, which are decoded
back into actions. It needs:

- StarVLA's `StarVLA/Qwen3-VL-4B-Instruct-Action` backbone (verified at revision
  `c41500cf1d287cd79e9f6602f419937b05048bd8`), which adds the action tokens. StarVLA enables them
  only when the backbone path contains `-Action`, so keep that in the folder name; training refuses
  a QwenFast backbone without it, and packages name their backbone folder `Qwen3-VL-Action`.
- The FAST tokenizer, `physical-intelligence/fast` at revision
  `ec4d7aa71691cac0b8bed6942be45684db2110f4`. StarVLA loads it from a fixed path relative to its
  working directory, with no config key. The plugin reads `framework.action_model.fast_tokenizer_path`
  (an absolute folder) instead, and QwenFast packages carry the tokenizer's JSON files in `FAST/`.

`configs/qwenfast_libero.yaml` is a 20-step LIBERO smoke run. Because the actions come out of the
language model, part of it must train: to fit a 24 GB GPU the config trains only the last two of its
36 layers, where upstream's recipes train the whole VLM. QwenFast exports native packages only.

```bash
WANDB_MODE=disabled library/.venv/bin/physicalai train starvla \
  --config_yaml plugins/policies/starvla/configs/qwenfast_libero.yaml \
  framework.qwenvl.base_vlm=/absolute/path/to/Qwen3-VL-4B-Instruct-Action \
  framework.action_model.fast_tokenizer_path=/absolute/path/to/fast \
  datasets.vla_data.data_root_dir=/absolute/path/to/datasets \
  run_root_dir=/absolute/path/to/runs run_id=starvla-qwenfast
```

When the generated text holds no action tokens, StarVLA's FAST decoder prints `Error decoding tokens`
and returns an all-zero normalized chunk instead of failing; Runtime passes that chunk on like any
other. Check that a QwenFast policy emits action tokens before running it on a robot.

Verified on one RTX 3090: 20 steps through the engine took 21 seconds at 11.6 GB of GPU memory, and
the loss fell from 16.1 to 14.2. The package held the FAST tokenizer's data files and no Python, and
an OpenVINO export was refused. After 20 steps the model does not yet generate action tokens, so on a
real LIBERO-goal frame both StarVLA's own `predict_action` and Runtime returned the zero-chunk
fallback, after 4.8 seconds of generation. Runtime matched StarVLA exactly and repeated it exactly.

A 2,000-step run (batch size 4, VLM learning rate 1e-4, still only the last two layers; 14 minutes at
17.7 GB) brought the loss to about 3.8, and the model then generated 9 to 12 action tokens per chunk
in under a second. None of its sequences decoded, though: they held 48 to 53 DCT coefficients instead
of 56, so StarVLA and Runtime again returned the same fallback. StarVLA's published QwenFast
checkpoints use Qwen2.5-VL, which this plugin does not support.

Decoding was therefore checked on its own, with that run's package. Nine recorded LIBERO-goal action
chunks were normalized with the checkpoint's training transforms, encoded with the package's FAST
tokenizer (9 to 14 tokens each), and given to the inference worker as the language model's output.
Its actions equalled FAST's decoding followed by StarVLA's unnormalization exactly, and matched the
recorded chunks to a mean absolute error of 0.0097 (the actions average 0.24 in magnitude; the worst
element was 0.063), which is FAST's quantization. The FAST code was the plugin's reviewed copy, its
data came from the package, the token ids the worker decodes agree with the `<robot_action_N>` text
used in training, and a sequence one token short gave the zero fallback. How well a fully trained
QwenFast policy acts was not measured.

### Reviewed remote code

Some Hugging Face models and tokenizers ship their own Python code, which Transformers runs when
`trust_remote_code=True`; StarVLA passes that flag. The workers replace Transformers' loader for such
code: they answer requests only for files the plugin has reviewed and ships unmodified, in
`_worker/remote_code/`, and refuse everything else. Today that is one file, FAST's
`processing_action_tokenizer.py`. If a model folder holds its own copy, it must match the reviewed
file's SHA-256, so a different revision is refused rather than run. Packages never contain Python.
`_worker/remote_code/README.md` records the source, revision, licence, hash and review notes, and a
unit test checks that the files and the pinned hashes agree.

## Qwen3.5 backbones

StarVLA also builds QwenGR00T, QwenPI and QwenOFT on Qwen3.5, Qwen's natively multimodal models
(0.8B to 9B); its own LIBERO QwenPI recipe uses Qwen3.5-0.8B. Qwen3.5 needs transformers 5, while the
rest of the plugin is locked on transformers 4.57.1, so Qwen3.5 trains and runs in its own worker
environment, `envs/qwen35`, which the engine declares as the enablement `qwen35`:

```bash
uv run physicalai engine prepare starvla --enablement qwen35
```

Its lock is the Qwen3-VL lock with transformers 5.10.4 and huggingface-hub 1.33.0, plus three small
dependencies of transformers; every other package keeps its version. StarVLA suggests transformers
5.3.0; 5.10.4 also clears that release's three known advisories.

The engine picks the environment as StarVLA picks the VLM: a `framework.qwenvl.base_vlm` whose path
contains `Qwen3.5` (and not `Qwen3-VL`, which StarVLA checks first) trains in `qwen35`, and the
training worker checks that the backbone's `config.json` says `qwen3_5`. Packages keep the backbone
in a `Qwen3.5` folder and record the `qwen35` pins, so Runtime runs them in the same environment.

Not supported on Qwen3.5: QwenFast, which needs a Qwen3.5 backbone with StarVLA's action tokens that
StarVLA does not publish; co-training, whose VLM data loader expects Qwen3-VL; and the OpenVINO
export. The optional `flash-linear-attention` and `causal-conv1d` kernels are not installed, so
Qwen3.5's linear-attention layers run transformers' PyTorch implementation; StarVLA notes that the
fast kernels can change the results of older checkpoints.

`configs/qwenpi_libero_qwen35.yaml` is a 20-step LIBERO smoke run of StarVLA's recipe. With this
small backbone the whole model, VLM included, trains on a 24 GB GPU, as upstream trains it. For
QwenGR00T or QwenOFT on Qwen3.5, override `framework.qwenvl.base_vlm` in their configs.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"],
    local_dir="/absolute/path/to/Qwen3.5-0.8B",
)
```

```bash
WANDB_MODE=disabled library/.venv/bin/physicalai train starvla \
  --config_yaml plugins/policies/starvla/configs/qwenpi_libero_qwen35.yaml \
  framework.qwenvl.base_vlm=/absolute/path/to/Qwen3.5-0.8B \
  datasets.vla_data.data_root_dir=/absolute/path/to/datasets \
  run_root_dir=/absolute/path/to/runs run_id=starvla-qwenpi-qwen35
```

Verified on one RTX 3090 with Qwen3.5-0.8B, 20 LIBERO-goal steps each through the engine:

| Framework | What trains                              | Time | GPU memory | Loss        | Runtime vs StarVLA      |
| --------- | ---------------------------------------- | ---- | ---------- | ----------- | ----------------------- |
| QwenGR00T | action head (VLM frozen)                 | 24 s | 5.5 GB     | 1.32 → 1.23 | exact, 0.59 s per chunk |
| QwenPI    | everything (`qwenpi_libero_qwen35.yaml`) | 29 s | 14.7 GB    | 1.49 → 1.47 | exact, 0.57 s per chunk |
| QwenOFT   | action MLP (VLM frozen)                  | 15 s | 3.1 GB     | 0.67 → 0.41 | exact, 0.64 s per chunk |

Each run recorded the `qwen35` pins, each package held a `Qwen3.5` folder, Runtime matched StarVLA's
own `predict_action` exactly (maximum absolute difference `0.0`) on a real frame and repeated it
exactly after a reset, and an OpenVINO export of the QwenGR00T run was refused. QwenPI starts near
1.5 here, against about 5,200 at initialisation on Qwen3-VL (see [QwenPI](#qwenpi)), so that scale
problem does not show up with this backbone; why was not measured. A Qwen3-VL-2B QwenGR00T run in
the same session still trained in `qwengr00t`, matched StarVLA exactly and exported to OpenVINO.

## Cosmos-Predict2 world models

StarVLA's world-model frameworks put its GR00T, OFT and PI action heads on a video world model
instead of a VLM: CosmoPredict2GR00T, CosmoPredict2OFT and CosmoPredict2PI read their visual features
from Cosmos-Predict2-2B, NVIDIA's video-to-world diffusion model (a T5 text encoder, a Wan VAE and a
video DiT whose last block feeds the head). The pinned wheel runs them with QwenGR00T's dependencies,
so they use the same environment.

The world model is `framework.world_model.base_wm`, not `framework.qwenvl.base_vlm`: a local copy of
the diffusers folder of `nvidia/Cosmos-Predict2-2B-Video2World`. Its path must contain
`Cosmos-Predict2` (any case), the substring StarVLA picks the model by, and the training worker
checks that its `model_index.json` names a `Cosmos2VideoToWorldPipeline`. The model is gated: accept
NVIDIA's Open Model License on its Hugging Face page first. Packages contain its weights, so the
licence applies to them too. Download only the parts StarVLA loads (about 14 GB; the repository also
holds about 24 GB of original checkpoints):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "nvidia/Cosmos-Predict2-2B-Video2World",
    revision="f50c09f5d8ab133a90cac3f4886a6471e9ba3f18",
    allow_patterns=[
        "model_index.json",
        "config.json",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*.json",
        "tokenizer/*.model",
        "tokenizer/*.txt",
        "transformer/*",
        "vae/*",
    ],
    local_dir="/absolute/path/to/Cosmos-Predict2-2B-Video2World",
)
```

```bash
WANDB_MODE=disabled library/.venv/bin/physicalai train starvla \
  --config_yaml plugins/policies/starvla/configs/cosmopredict2groot_libero.yaml \
  framework.world_model.base_wm=/absolute/path/to/Cosmos-Predict2-2B-Video2World \
  datasets.vla_data.data_root_dir=/absolute/path/to/datasets \
  run_root_dir=/absolute/path/to/runs run_id=starvla-cosmos
```

`configs/cosmopredict2groot_libero.yaml` is a 20-step LIBERO smoke run with the world model frozen,
so only the action head trains; CosmoPredict2OFT takes it with `framework.name=CosmoPredict2OFT`.
`configs/cosmopredict2pi_libero.yaml` also freezes the PI head's DiT blocks (one per world-model
layer), which do not fit 24 GB next to the world model, and it sets the head's DiT shape: StarVLA's
layer-wise head reads its depth and width only from `action_model.diffusion_model_cfg`, which QwenPI
fills from its backbone but CosmoPredict2PI does not (at the pinned commit, and on `starVLA_dev` as
of 2026-11-18), so the head would otherwise build 36 blocks for Cosmos-Predict2-2B's 28 and fail at
the first step. The config sets 28 blocks of width 2048, as QwenPI would. StarVLA's own recipes
train everything on 8 GPUs with DeepSpeed.

StarVLA's world-model wrapper resizes every camera image to 320×576 and feeds the cameras as frames
of one short video, and it samples the VAE's latent distribution, so a prediction depends on the
seed; Runtime seeds it the same way as StarVLA's own server. Packages keep the world model in a
`Cosmos-Predict2` folder (`model_index.json` and its parts) and, as for every framework, a checkpoint
that holds the whole model again. Co-training and the OpenVINO export are refused. Wan2.2, StarVLA's
other world model, is not wired: its weights alone take about 23 GB in bf16, more than a 24 GB GPU
holds with anything else.

Verified on one RTX 3090 with Cosmos-Predict2-2B (revision `f50c09f5`), 20 LIBERO-goal steps each
through the engine:

| Framework          | What trains                | Time | GPU memory | Loss        | Runtime vs StarVLA      |
| ------------------ | -------------------------- | ---- | ---------- | ----------- | ----------------------- |
| CosmoPredict2GR00T | action head                | 33 s | 17.2 GB    | 1.29 → 1.28 | exact, 0.64 s per chunk |
| CosmoPredict2OFT   | action MLP                 | 30 s | 16.0 GB    | 0.73 → 0.64 | exact, 0.70 s per chunk |
| CosmoPredict2PI    | head encoders and decoders | 48 s | 21.9 GB    | 24.9 → 28.3 | exact, 0.75 s per chunk |

Each package held the world model's diffusers folder (14.2 GB, without the repository's original
checkpoints or download metadata) and the checkpoint (14.3 to 20.8 GB), and no Python. Runtime
matched StarVLA's own `predict_action` exactly on a real frame and repeated it exactly after a reset,
and the OpenVINO export was refused. CosmoPredict2PI's loss starts near 25, far above the other two
heads, much as QwenPI's does on Qwen3-VL (see [QwenPI](#qwenpi)); its cause was not investigated.

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

## Multi-GPU training

`all_devices` trains on every GPU the caller exposes, with one data-parallel process per GPU, like
Studio's "All GPUs" choice for remote trainers. Pass it with `device="cuda"` (not `cuda:N`), and choose
the GPUs with `CUDA_VISIBLE_DEVICES`; it needs at least two. On the command line, add `--all_devices`.

```python
run = Engine.get("starvla").run(
    {
        "config_yaml": "plugins/policies/starvla/configs/qwengroot_libero.yaml",
        "overrides": ["framework.qwenvl.base_vlm=/absolute/path/to/Qwen3-VL-4B-Instruct"],
        "all_devices": True,
    },
    dataset=Path("/absolute/path/to/datasets"),
    output_dir=Path("/absolute/path/to/runs/starvla-2gpu"),
    device="cuda",
)
```

`datasets.vla_data.per_device_batch_size` stays per GPU, so each step uses that many samples on every
GPU: with two GPUs, twice the single-GPU batch at the same learning rate and step count.

The worker the engine starts keeps the engine's protocol and launches `torchrun` with one rank per
GPU. Each rank runs StarVLA's own trainer, which uses `accelerate`'s DistributedDataParallel under
`torchrun`. Progress comes from rank 0 (its loss, not an average over the ranks), and rank 0 writes
the checkpoint, which exports and runs like any other. A cancellation reaches every rank at the same
step; ranks that have not stopped after 3 seconds are terminated. `torchrun` starts each rank in a
session of its own, so each rank also asks Linux to kill it when `torchrun` dies: stopping the
worker, or a crashed host, frees every GPU.

Co-training stays on one GPU: without DeepSpeed, StarVLA's co-training step runs a second backward
pass for the VLM loss, through the VLM outside the parallel wrapper, and DistributedDataParallel
supports one backward pass per step. The worker refuses `all_devices` for a co-training config.

DistributedDataParallel is told to look for trainable parameters that a step leaves unused (an
untrained state encoder, for one); otherwise it stops at the second step. Upstream's multi-GPU
recipes use DeepSpeed ZeRO-2 instead, which also shards the optimizer state across the GPUs.
DeepSpeed is published only as source code and is not in the worker environments, so here every GPU
still holds the whole model and optimizer state: more GPUs train on more samples per step, but do
not let a larger model fit.

Verified on two RTX 3090s, 20 LIBERO-goal steps each through the engine:

| Run                                            | GPUs | Time | GPU memory       | Runtime vs StarVLA |
| ---------------------------------------------- | ---- | ---- | ---------------- | ------------------ |
| QwenGR00T on Qwen3-VL-2B, VLM frozen           | 2    | 28 s | 10.9 GB on each  | exact              |
| the same run                                   | 1    | 18 s | 8.1 GB           | exact              |
| QwenPI on Qwen3.5-0.8B, whole model (`qwen35`) | 2    | 43 s | 17.9 and 17.4 GB | exact              |

Twenty steps are mostly start-up (one model load per GPU and the process group), so these runs check
correctness, not speed; on two GPUs each step trains on twice the samples. Each GPU also needs more
memory than a single-GPU run (DistributedDataParallel keeps gradient buckets): QwenPI's whole-model
run took 14.7 GB on one GPU. Rank 0 reported every step in order, and Runtime matched StarVLA's own
`predict_action` exactly on a real frame and repeated it exactly after a reset. Cancelling after step
5 stopped both ranks together after step 6, returning 3.2 seconds after the request with no rank left
and both GPUs free. Killing the host process at step 3, as a crash would, also left no worker,
`torchrun` or rank behind. A co-training config was refused before any rank started.

## RL post-training with RLinf

[RLinf](https://github.com/RLinf/RLinf) post-trains StarVLA policies with reinforcement learning in
simulation (GRPO or PPO, with LIBERO in the loop) on Ray. It runs from its own installer and
environment, so the plugin does not run it; it hands a run to RLinf and imports the result back as
a normal run that exports to a Runtime package. RLinf's StarVLA wrapper runs QwenGR00T, QwenOFT and
QwenPI on Qwen3-VL, so those are the runs the plugin hands over. RLinf installs StarVLA at its
`starVLA-v1.6` tag, an ancestor of this plugin's pin with the same checkpoint layout.

Install RLinf at a fixed commit. Its installer runs `requirements/sys_deps.sh` with `sudo apt-get`
unless you pass `--no-root`: on the verification machine that installed 164 system packages (a build
toolchain and OpenGL, OSMesa, ffmpeg and ImageMagick development libraries) and upgraded 77, then
restarted services. With `--no-root` those must already be present.

```bash
git clone https://github.com/RLinf/RLinf.git && cd RLinf && git checkout ed0e1b836d11ec50eef4e5133e6b42ebd370cab9
bash requirements/install.sh embodied --model starvla --env libero --no-root
```

Then hand a run over, train, and import the result. The plugin reads and writes safetensors only
and never unpickles a file; the two conversions to and from RLinf's `.pt` files run in RLinf's own
environment, which loads its checkpoints with PyTorch anyway:

```bash
# 1. Plugin: the run in StarVLA's layout (checkpoints/steps_0_model.safetensors, config.yaml, dataset_statistics.json).
library/.venv/bin/python -m physicalai_starvla_policy_plugin.rlinf export /path/to/runs/starvla /path/to/rlinf-start

# 2. RLinf's environment: the .pt RLinf loads, beside it.
python -c "import sys, torch; from safetensors.torch import load_file; torch.save(load_file(sys.argv[1]), sys.argv[2])" \
  /path/to/rlinf-start/checkpoints/steps_0_model.safetensors /path/to/rlinf-start/checkpoints/steps_0_pytorch_model.pt

# 3. RLinf: set actor.model.model_path and rollout.model.model_path to that .pt, and
#    actor.model.starvla.framework_name, action_dim, num_action_chunks (the run's action_horizon) and unnorm_key.
bash examples/embodiment/run_embodiment.sh <your_config> LIBERO

# 4. RLinf's environment: the trained policy as safetensors. RLinf writes it as
#    checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt (or convert its dcp_checkpoint
#    with rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py).
python -c "import sys, torch; from safetensors.torch import save_file; s = torch.load(sys.argv[1], map_location='cpu', weights_only=True); save_file({k: v.clone().contiguous() for k, v in s.items()}, sys.argv[2])" \
  logs/<run>/<experiment>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt rl.safetensors

# 5. Plugin: a run that exports like any other.
library/.venv/bin/python -m physicalai_starvla_policy_plugin.rlinf import rl.safetensors /path/to/runs/starvla /path/to/runs/starvla-rl
```

The import takes the StarVLA weights from under RLinf's `starvla_model.` prefix and leaves out
RLinf's own parameters (`actor_logstd`, a value head). Every weight must match the starting run in
name and shape, and keeps the starting run's dtype; the run record notes the source file and its
SHA-256 under `post_training`. RLinf holds the policy in bf16 (`actor.model.precision`), so fp32
weights, such as QwenOFT's action head, come back rounded to bf16 even where RL did not change them.

Verified on two RTX 3090s with a 20-step QwenOFT run on Qwen3-VL-2B and RLinf's test-sized GRPO
config on LIBERO-goal (8 simulators, 8-step episodes, one epoch). RLinf loaded the handed-over run,
trained the epoch and saved a checkpoint in about 1.5 minutes at 13.4 GB per GPU. Eight steps are
too short to finish a LIBERO task, so every reward and the gradient were zero; the imported weights
equalled RLinf's, differing from the starting run only by that bf16 rounding of the 16 fp32
tensors. The `.pt` from step 2 was identical to the one RLinf trained from, RLinf's
`full_weights.pt` matched its distributed checkpoint converted with its own tool, and the imported
run exported, with Runtime matching StarVLA's own `predict_action` exactly and repeating it
exactly after a reset. An RL run that improves a policy (longer episodes; RLinf's recipe
uses 8 GPUs) was not tried.

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

## StarVLA's benchmark clients

StarVLA evaluates its models with benchmark clients (`examples/simBenchmarks/*/eval_files`) that talk
to a websocket policy server. The plugin serves any of its Runtime packages, native or OpenVINO, over
the same protocol, so StarVLA's LIBERO, SimplerEnv, RoboTwin and other clients run unchanged:

```bash
library/.venv/bin/python -m physicalai_starvla_policy_plugin.serve exports/starvla --device cuda --port 10093
```

The server needs the plugin's `serve` extra (`websockets` and `msgpack`), which the library's
`starvla` extra includes. It prints `READY <port>` once the package is loaded. Each request's
`image` list maps onto the package's cameras in order, `lang` is the task, and `state` is used only
by a package that maps state. Actions come back unnormalized, in the package's action layout. The
handshake metadata names the cameras, action keys and state keys, and reports the chunk length the
clients use to schedule requests. A request for another normalization key, the wrong number of
images, or missing state is answered with an error, as StarVLA's own server answers failures.

The server speaks plain websockets without authentication, because StarVLA's clients expect that.
It listens on `127.0.0.1` by default; use `--host 0.0.0.0` only on a trusted network.

To run StarVLA's LIBERO evaluation, set up its client environment and LIBERO assets as StarVLA's
LIBERO guide describes, put a checkout of StarVLA at the pinned commit on `PYTHONPATH`, and point the
client at the server:

```bash
cd /path/to/starVLA && PYTHONPATH=$PWD python examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.host 127.0.0.1 --args.port 10093 --args.task-suite-name libero_goal --args.num-trials-per-task 50
```

StarVLA's LIBERO client renders 256-pixel frames and resizes them to 224 pixels before sending them.
Native packages accept that, as StarVLA's own server does. Qwen3-VL's processor enlarges images below
256×256 before encoding them; an OpenVINO package reproduces that enlargement with OpenCV, which does
not match it closely enough at 224 pixels, so a 224-pixel OpenVINO export fails its own validation.
OpenVINO packages therefore need a client that sends the resolution they were exported for.

Verified on one RTX 3090 with StarVLA's published `StarVLA/Qwen3-VL-OFT-LIBERO-4in1` checkpoint
(QwenOFT on Qwen3-VL-4B, revision `39177146`), whose model card reports 98.6% on `libero_goal`.
StarVLA publishes it as a PyTorch pickle, which the plugin refuses; for this check it was converted
to safetensors with PyTorch's weights-only loader and then exported with `export_native`. StarVLA's
unmodified LIBERO client ran 100 `libero_goal` episodes (10 per task, seed 7) twice: against StarVLA's
own server and against this server with the native package, both without `--use_bf16`. Both
succeeded in 98 of 100 episodes, with the same outcome in every episode, in about 8.5 minutes each.

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
