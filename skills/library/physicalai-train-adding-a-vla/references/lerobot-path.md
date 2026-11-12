# LeRobot path: a VLA that LeRobot implements

The LeRobot engine (`plugins/policies/lerobot/src/physicalai_lerobot_policy_plugin/engine.py`) runs `lerobot-train` in a locked worker environment. Each policy family can have its own environment, so one VLA's dependencies never meet another's:

- `env/` — default environment (`lerobot[training,pi]`), used by every unmapped policy.
- `envs/<family>/` — one uv project per family installing only `lerobot[training,<extra>]` (today: `smolvla`, `pi` for pi0/pi05/pi0_fast, `molmoact2`).
- `LeRobotEngine.policy_envs` — maps policy types to family environments; engine core support is `Engine.policy_envs` / `environment_for_policy()` in `library/src/physicalai/engine/__init__.py`.

`physicalai train lerobot` picks the environment from `--policy.type`, a local or Hugging Face Hub `--policy.path` checkpoint, or a `--config_path` train config. Arguments otherwise pass to `lerobot-train` unchanged. Set `P=plugins/policies/lerobot/src/physicalai_lerobot_policy_plugin` and `SKILL=skills/library/physicalai-train-adding-a-vla`.

## Contents

- [A. Give the family its own environment](#a-give-the-family-its-own-environment)
- [B. Train and evaluate](#b-train-and-evaluate)

## A. Give the family its own environment

Skip if `inspect_vla.py` already reports a non-`default` worker environment.

1. **Find the extra.** Extras use hyphens where types use underscores (`vla_jepa` → `vla-jepa`, `lingbot_va` → `lingbot-va`, `multi_task_dit` → `multi-task-dit`, `wall_x` → `wallx`). From the repo root:

   ```bash
   library/.venv/bin/python -c "import importlib.metadata as m; print([e for e in m.distribution('lerobot').metadata.get_all('Provides-Extra') if not e.endswith('-dep')])"
   ```

   `groot` is intentionally unsupported in the LeRobot engine (remote code); use the GR00T plugin.

   - Done when: you know the policy type and its extra.

2. **Create and lock the project** from an existing family, seeding the lock from `env/uv.lock` so LeRobot, torch and transformers keep their pins and uv hard-links them:

   ```bash
   mkdir -p $P/envs/<family>
   sed -e 's/worker-pi"/worker-<family>"/' -e 's/lerobot\[training,pi\]/lerobot[training,<extra>]/' $P/envs/pi/pyproject.toml > $P/envs/<family>/pyproject.toml
   cp $P/env/uv.lock $P/envs/<family>/uv.lock
   library/.venv/bin/uv lock --project $P/envs/<family>
   ```

   Use `library/.venv/bin/uv` (the plugin needs `uv>=0.10,<0.11`).

   - Done when: uv reports only the extra's packages as added. If core pins move, keep them (that is what separate environments are for) and say so in the PR.

3. **Map the type** in `engine.py`: `_<FAMILY>_ENV = _worker_env("<family>")` and `"<policy_type>": _<FAMILY>_ENV` in `policy_envs`. The name must equal the `envs/` directory.

   - Done when: `LeRobotEngine.environment_for_policy("<policy_type>").name == "<family>"`.

4. **Update tests** in `library/tests/unit/engine/`: add the family to `test_each_policy_family_has_its_own_locked_worker_project` (with a package only its extra brings) in `test_lerobot_plugin_packaging.py`, and to `test_lerobot_cli_trains_each_policy_family_in_its_own_environment` and `test_lerobot_engine_lists_each_family_environment_once` in `test_lerobot_engine.py`. Then from `library/`:

   ```bash
   .venv/bin/python -m pytest tests/unit/engine ../plugins/policies/lerobot/tests -q
   ```

   - Done when: the tests pass. Do not add `envs/` to hatch `force-include`; `packages` already ships it.

5. **Build and import-check** (from `library/`):
   ```bash
   .venv/bin/physicalai engine prepare lerobot --policy <policy_type> --accelerator <cpu|cuda|xpu>
   <env>/bin/python -c "import importlib; from lerobot.policies.factory import get_policy_class; c = get_policy_class('<policy_type>'); importlib.import_module(c.__module__); print('ok', c.__name__)"
   ```
   Imports can pass while lazily loaded dependencies are still missing; part B's smoke run is the real check.
   - Done when: `prepare` prints the new environment and the import prints `ok`.

## B. Train and evaluate

Run from `library/`. On Intel XPU, `export UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` first.

1. **Inspect the checkpoint against the dataset** (metadata only):

   ```bash
   .venv/bin/python ../$SKILL/scripts/inspect_vla.py <hub-id-or-local-dir> <dataset-repo-id>
   ```

   - Done when: you have the policy type, a non-`default` worker environment, a memory estimate that fits (or a freezing/dtype plan), and a `--rename_map` or "not needed". Gated backbones need `hf auth login`.

2. **Smoke-test 20 steps**:

   ```bash
   .venv/bin/physicalai train lerobot --policy.path=<hub-id-or-local-dir> --policy.device=<cpu|cuda|xpu> \
     --policy.push_to_hub=false --dataset.repo_id=<dataset-repo-id> '--rename_map=<json from step 1>' \
     --batch_size=8 --steps=20 --save_freq=20 --output_dir=<run-dir>
   ```

   `--policy.type=<type>` instead of `--policy.path` starts from random weights — plumbing only.

   - Done when: `<run-dir>/checkpoints/000020/pretrained_model/` exists. On failure see [troubleshooting.md](troubleshooting.md).

3. **Train for real** with a new `--output_dir`; resume a stopped run with:

   ```bash
   .venv/bin/physicalai train lerobot --config_path=<run-dir>/checkpoints/last/pretrained_model/train_config.json --resume=true
   ```

   - Done when: `checkpoints/last/` points at the final step.

4. **Evaluate through Runtime** (PushT, fixed seeds 100000+):
   ```bash
   .venv/bin/python ../$SKILL/scripts/eval_pusht.py native <run-dir>/checkpoints/last/pretrained_model --episodes 50 --device <cpu|cuda|xpu>
   ```
   It packages the checkpoint with `LeRobotEngine.export`, loads it with `physicalai.inference.InferenceModel`, and reports `pc_success` and `avg_max_reward` (holding still scores ~0.14). For other tasks, export with `LeRobotEngine.export(checkpoint, output_path=...)` and feed observations with the camera keys the checkpoint was trained on.
   - Done when: the JSON report is written next to the checkpoint and compared with a baseline or earlier run.
