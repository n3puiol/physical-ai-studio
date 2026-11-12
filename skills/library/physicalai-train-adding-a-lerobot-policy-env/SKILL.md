---
name: physicalai-train-adding-a-lerobot-policy-env
description: Gives a LeRobot policy family its own locked worker environment in the LeRobot engine plugin. Use when a LeRobot VLA (xvla, eo1, evo1, wall_x, vla_jepa, lingbot_va, fastwam, multi_task_dit, ...) fails under `physicalai train lerobot` with a missing import, when adding an entry to `LeRobotEngine.policy_envs`, when creating a project under plugins/policies/lerobot/src/physicalai_lerobot_policy_plugin/envs/, or when re-locking an existing family environment.
license: Apache-2.0
---

# Adding a LeRobot policy environment

The LeRobot engine (`plugins/policies/lerobot/src/physicalai_lerobot_policy_plugin/engine.py`) trains each policy family in its own locked worker environment, so one family's dependencies never meet another's. Engine core support lives in `library/src/physicalai/engine/__init__.py` (`Engine.policy_envs`, `environment_for_policy()`).

- `env/` — default environment (`lerobot[training,pi]`), used by every unmapped policy (ACT, Diffusion, ...).
- `envs/<family>/` — one bundled uv project per family, installing only `lerobot[training,<extra>]`.
- `LeRobotEngine.policy_envs` — maps native policy types to a family environment; several types may share one (`pi0`, `pi05`, `pi0_fast` → `pi`).

`cli()`, `train()` and `run()` resolve the policy type from `--policy.type`, a local or Hugging Face Hub `--policy.path` checkpoint, or a `--config_path` train config, then launch that environment. Run commands from the repo root unless noted.

## Workflow

1. **Find the policy type and its LeRobot extra.** Extras use hyphens where types use underscores (`vla_jepa` → `vla-jepa`, `lingbot_va` → `lingbot-va`, `multi_task_dit` → `multi-task-dit`, `wall_x` → `wallx`).

   ```bash
   library/.venv/bin/python -c "import importlib.metadata as m; print([e for e in m.distribution('lerobot').metadata.get_all('Provides-Extra') if not e.endswith('-dep')])"
   ```

   - Done when: you know the type string LeRobot registers and the extra that installs its dependencies. `groot` is intentionally unsupported (`policy_support.py`, remote code); use the GR00T plugin instead.

2. **Create the project from an existing family.** Copy `envs/pi/` so the torch indexes and accelerator extras stay identical; change only the project name and the extra.

   ```bash
   P=plugins/policies/lerobot/src/physicalai_lerobot_policy_plugin
   mkdir -p $P/envs/<family>
   sed -e 's/worker-pi"/worker-<family>"/' -e 's/lerobot\[training,pi\]/lerobot[training,<extra>]/' $P/envs/pi/pyproject.toml > $P/envs/<family>/pyproject.toml
   cp $P/env/uv.lock $P/envs/<family>/uv.lock
   ```

   Seeding from `env/uv.lock` keeps LeRobot, torch and transformers on the same pins, so uv hard-links them instead of storing new copies.

   - Done when: `envs/<family>/pyproject.toml` depends only on `lerobot[training,<extra>]>=0.6.1,<0.7` and keeps the `cpu`/`cu128`/`xpu` extras.

3. **Lock with the pinned uv** (the plugin requires `uv>=0.10,<0.11`; `library/.venv/bin/uv` provides it).

   ```bash
   library/.venv/bin/uv lock --project $P/envs/<family>
   ```

   - Done when: uv only reports packages added for the extra. If it bumps `torch`, `lerobot` or `transformers`, the extra needs different versions — keep the lock (that is the point of a separate environment) but say so in the PR.

4. **Map the policy type** in `engine.py` next to the other families:

   ```python
   _<FAMILY>_ENV = _worker_env("<family>")
   ```

   and add `"<policy_type>": _<FAMILY>_ENV` to `policy_envs`. The environment name must equal the `envs/` directory name.

   - Done when: `LeRobotEngine.environment_for_policy("<policy_type>").name == "<family>"`.

5. **Update tests** in `library/tests/unit/engine/`:

   - `test_lerobot_plugin_packaging.py`: add `("<family>", "<a package only this extra brings>")` to `test_each_policy_family_has_its_own_locked_worker_project`, plus the type in its `{family: policy_type}` lookup.
   - `test_lerobot_engine.py`: add the type to the `test_lerobot_cli_trains_each_policy_family_in_its_own_environment` parameters and the family to `test_lerobot_engine_lists_each_family_environment_once`.
   - Done when: the tests below pass.

6. **Build the environment and check it in isolation** (from `library/`):

   ```bash
   .venv/bin/pai engine prepare lerobot --policy <policy_type> --accelerator <cpu|cuda|xpu>
   ```

   Then import the policy with that environment's Python (path printed by `prepare`):

   ```bash
   <env>/bin/python -c "import importlib; from lerobot.policies.factory import get_policy_class; c = get_policy_class('<policy_type>'); importlib.import_module(c.__module__); print('ok', c.__name__)"
   ```

   Importing is necessary but not sufficient — some dependencies load only when training starts (SmolVLA's `num2words` loads with its processor). Step 7 is the real check.

   - Done when: `prepare` prints a new environment path and the import prints `ok`.

7. **Train a few steps through the CLI** to prove routing and lazy dependencies. Follow `physicalai-train-training-a-lerobot-vla` (camera renames, memory, gated weights). A 20-step run that writes a checkpoint is enough.
   - Done when: `checkpoints/000020/pretrained_model/` exists. If the model cannot fit the machine, state that only steps 1–6 were verified.

## Verify

```bash
# from library/
.venv/bin/python -m pytest tests/unit/engine ../plugins/policies/lerobot/tests -q
.venv/bin/uv lock --check --project ../plugins/policies/lerobot/src/physicalai_lerobot_policy_plugin/envs/<family>
```

Run the repo hooks on changed library files (`prek run --files ...`); the plugin is not covered by hooks, so also run `.venv/bin/ruff check` and `.venv/bin/mypy` on `engine.py` from `plugins/policies/lerobot/`.

## Required checks

- Only the family's extra changed versus `env/uv.lock`, or the PR explains why core pins moved.
- The wheel ships the new files: `library/.venv/bin/uv build --wheel -o /tmp/w plugins/policies/lerobot` and list `physicalai_lerobot_policy_plugin/envs/<family>/` in it. Do not add the folder to `force-include` — `packages` already includes it and hatch fails on duplicates.
- `pai engine list` shows the new environment; `pai engine prune` keeps it.
- Disk: sizes from `pai engine list` count hard-linked files per environment; compare `df` before and after `prepare` for the real cost.

## Related skills

- `physicalai-train-training-a-lerobot-vla` — train and evaluate the policy once its environment exists.
- `physicalai-train-adding-a-policy` — first-party Studio policies instead of LeRobot's.
