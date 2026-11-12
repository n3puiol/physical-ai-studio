# Repository plugin for a VLA that is not in LeRobot

Model the plugin on `plugins/policies/gr00t/` (Isaac-GR00T N1.7). Replace `<name>` with a short lowercase engine id (`[a-zA-Z0-9_-]{1,64}`), e.g. `openvla`.

## Contents

1. [Layout](#1-layout)
2. [Package metadata](#2-package-metadata)
3. [Worker environment](#3-worker-environment)
4. [Engine](#4-engine)
5. [Training worker](#5-training-worker)
6. [Runtime package](#6-runtime-package)
7. [Runtime adapter and inference worker](#7-runtime-adapter-and-inference-worker)
8. [Register with the library](#8-register-with-the-library)
9. [Tests](#9-tests)
10. [Verify end to end](#10-verify-end-to-end)

## 1. Layout

```
plugins/policies/<name>/
├── pyproject.toml
├── README.md
├── src/physicalai_<name>_policy_plugin/
│   ├── __init__.py
│   ├── environments.py      # pinned sources and profiles; Runtime-only importable
│   ├── engine.py            # TrainingEngine subclass (needs physicalai-train)
│   ├── config.py            # optional typed facade over the repo's training flags
│   ├── package.py           # Runtime package writer/reader, I/O mapping
│   ├── inference.py         # lazy Runtime adapter registration
│   ├── runtime.py           # RuntimeAdapter (+ client for an inference worker)
│   └── _worker/             # stdlib-only scripts run inside the worker environment
│       ├── <name>_worker_common.py
│       ├── <name>_train_worker.py
│       └── <name>_infer_worker.py   # only for out-of-process inference
└── tests/
```

Done when: the package imports and contains no framework imports outside `_worker/`.

## 2. Package metadata

Copy `plugins/policies/gr00t/pyproject.toml` and rename. Keep the host dependencies minimal — this package is installed into Studio's and Runtime's environments:

- `dependencies`: `physicalai` (Runtime), `numpy`, `uv`; never the model's framework.
- `[project.optional-dependencies] train = ["physicalai-train>=0.3.0"]`.
- Entry points:

  ```toml
  [project.entry-points."physicalai.engines"]
  <name> = "physicalai_<name>_policy_plugin.engine:<Name>Engine"

  [project.entry-points."physicalai.inference.adapters"]
  <name> = "physicalai_<name>_policy_plugin.inference:register"
  ```

- `[tool.uv.sources]`: the same `physicalai` source as the library, and `physicalai-train = { path = "../../../library", editable = true }`.

Done when: `python -c "import physicalai_<name>_policy_plugin.inference"` works in an environment without `physicalai-train` and without the framework.

## 3. Worker environment

Declare where the model's code and locked dependencies come from (`physicalai.isolate` sources, as in `gr00t/environments.py`):

- **Repo ships `pyproject.toml` and `uv.lock` at its root** → `GitProject(repository="https://...", revision="<full commit sha>")`. Isolate refuses a checkout without both files.
- **Repo has no uv lock** (most research repos) → `BundledProject(<plugin>/env)`: a plugin-owned `pyproject.toml` that depends on the repo at a pinned commit (`[tool.uv.sources] <dist> = { git = "...", rev = "<sha>" }`) plus the extras/indexes for `cpu`/`cu128`/`xpu`, and its `uv.lock` (lock with `library/.venv/bin/uv lock --project <dir>`; copy the torch index layout from `plugins/policies/lerobot/src/physicalai_lerobot_policy_plugin/env/pyproject.toml`).
- **Needs downstream patches** → `GitProject(..., patches=PublishedPatches(...))` or `BundledPatches`, with SHA-256 per file; patch application needs SDL review before shipping.
- Pin the Python version the repo supports with `SyncProfile(python="3.x")`.

Keep this module importable on a Runtime-only host (no `physicalai.engine` import) if the inference worker needs it; `engine.py` adapts the declaration to `WorkerEnv` (see `_worker_env()` in `gr00t/engine.py`). A plugin without an inference worker can declare `WorkerEnv(project=<source>, ...)` directly in `engine.py`.

Done when: `physicalai engine prepare <name> --accelerator <acc>` builds the environment and the repo imports with its Python.

## 4. Engine

`engine.py` subclasses `physicalai.engine.TrainingEngine` (`library/src/physicalai/engine/__init__.py`):

- Class attributes: `framework = "<name>"`, `env`, `training_accelerators`, `stock_export_formats`, optional `enablements` (each with its own environment) and `policy_envs` (per-model environments).
- Required classmethods (abstract in `TrainingEngine`):
  - `cli(argv, interpreter=None)` → run the training worker or the repo's CLI with the environment's Python, inherited streams, and return its exit code (backs `physicalai train <name> ...`).
  - `run(settings, dataset=, output_dir=, device=, hooks=)` → headless training with Studio-owned dataset/output/device.
  - `export(source, output_path, backend)` → write the Runtime package (section 6).
  - `policies()` and `config_schema(policy)`.
- Optional typed API: `train(config, hooks)` on an instance, implemented with `run_worker(env, arguments, device=..., hooks=hooks, worker=<train worker>, extra_env=...)` from `physicalai.engine`; it returns `Run(framework, output_dir, checkpoint)`. `run()` usually calls the same helper.

Done when: `Engine.get("<name>")` returns the class and `physicalai train <name> --help` reaches the repo's training help.

## 5. Training worker

`_worker/<name>_train_worker.py` runs inside the worker environment, so it may import only the standard library, the repo, and its own `<name>_worker_common.py`. Copy `gr00t_worker_common.py` (`EventChannel`, `start_cancel_listener`, `disable_remote_code`) and follow `gr00t_train_worker.py`:

1. Send `{"type": "hello", "protocol": 1, ...identity}`.
2. Run the repo's own training entry point (`runpy` / its `main`) with the native arguments; report `{"type": "progress", "step", "total", "metrics"}` from a callback or log hook.
3. Finish with `{"type": "result", "output_dir", "checkpoint"}`, or `{"type": "error", ...}` / `{"type": "cancelled", "step"}`; `sys.exit()` with no code is success.
4. Refuse `trust_remote_code` downloads and write a small run-info JSON beside the checkpoint (dataset, embodiment, pins) for export.

Done when: a few training steps produce a checkpoint and the host receives the progress events.

## 6. Runtime package

`package.py` writes a directory with a Runtime `manifest.json` (backend `<name>`, via `physicalai.inference.manifest`), the model files, and a descriptor with `format_version`, the environment key/provenance, and an explicit I/O mapping: which Runtime observation keys (camera names, state slices, task text) feed which model inputs, and how the action chunk is laid out. Validate on load: reject unknown versions and model paths that escape the package (see `load_descriptor()` in `gr00t/package.py`).

Done when: `<Name>Engine.export(run, "exports/<name>")` writes a package that `load_descriptor()` accepts.

## 7. Runtime adapter and inference worker

`inference.py` registers lazily so Runtime does not import the model at startup:

```python
def register(registry):
    registry.register_lazy_module("<name>", "physicalai_<name>_policy_plugin.runtime", extensions=(".<name>",))
```

`runtime.py` subclasses `physicalai.inference.adapters.base.RuntimeAdapter` (required: `load`, `predict`, `input_names`, `output_names`; usually also `reset` and `default_device`), registered with `@adapter_registry.register("<name>", extensions=(".<name>",))`. Choose:

- **In-process** only if the framework's dependencies fit Runtime's environment.
- **Out of process** (the GR00T pattern) otherwise: the adapter prepares the environment through `physicalai.isolate`, starts `_worker/<name>_infer_worker.py`, and exchanges length-prefixed JSON + raw numeric arrays over two pipes (`gr00t_worker_protocol.py`: no pickle, allow-listed dtypes, size limits). When copying `_Worker` from `gr00t/runtime.py`, also fix two known issues: mark the worker unusable if `request()` raises between write and read (otherwise the next reply is stale), and kill the process before closing its pipes in `_stop()` (otherwise `close()` can hang on a stuck worker).

Done when: `physicalai.inference.InferenceModel("exports/<name>")` predicts an action chunk of the expected shape, and `reset()` makes predictions repeat for the same seed.

## 8. Register with the library

In `library/pyproject.toml`:

```toml
[project.optional-dependencies]
<name> = ["physicalai-<name>-policy-plugin[train]"]

[tool.uv.sources]
physicalai-<name>-policy-plugin = { path = "../plugins/policies/<name>", editable = true }
```

Then `uv lock` and `uv sync --extra <name>` from `library/`.

Done when: `physicalai engine list` and `Engine.names()` include `<name>` after `physicalai engine prepare <name>`.

## 9. Tests

Mirror `plugins/policies/gr00t/tests/`: environments (pins, keys), engine (train/cli/run with `run_worker` patched), package (round trip, path escapes), protocol (framing, dtype limits), runtime (adapter with a fake worker, plus the real worker transport with a tiny fake model), workers (protocol events, cancellation). Gate anything that downloads weights or needs a GPU behind an opt-in variable, like `library/tests/integration/test_engine_gr00t_e2e.py` (`PHYSICALAI_RUN_GR00T_E2E=1`).

Done when: `.venv/bin/python -m pytest ../plugins/policies/<name>/tests -q` passes from `library/` without network or GPU.

## 10. Verify end to end

From `library/`:

```bash
.venv/bin/physicalai engine prepare <name> --accelerator <cpu|cuda|xpu>
.venv/bin/physicalai train <name> <repo training args with a tiny step budget>
.venv/bin/python -c "from physicalai.engine import Engine; print(Engine.get('<name>').export('<checkpoint>', 'exports/<name>'))"
.venv/bin/python -c "from physicalai.inference import InferenceModel; m = InferenceModel('exports/<name>'); print(type(m.adapter).__name__, m.adapter.output_names)"
```

Then repeat the last command in an environment with only `physicalai` and the plugin (no `physicalai-train`, no framework) to prove the Runtime-only path.
