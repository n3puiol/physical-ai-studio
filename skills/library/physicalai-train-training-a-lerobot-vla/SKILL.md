---
name: physicalai-train-training-a-lerobot-vla
description: Trains and evaluates a LeRobot VLA (SmolVLA, Pi0/Pi0.5, MolmoAct2, X-VLA, ...) through Physical AI Studio's LeRobot engine in an isolated worker environment. Use when running `physicalai train lerobot` / `pai train lerobot`, fine-tuning a Hugging Face LeRobot checkpoint with --policy.path, mapping dataset cameras with --rename_map, preparing worker environments with `physicalai engine prepare lerobot --policy`, resuming an engine run, or evaluating a LeRobot checkpoint through Runtime on PushT.
license: Apache-2.0
---

# Training a LeRobot VLA through the engine

`physicalai train lerobot <native lerobot-train args>` runs LeRobot's own trainer inside a locked worker environment (`plugins/policies/lerobot/src/physicalai_lerobot_policy_plugin/engine.py`). Arguments pass through unchanged; the engine only reads `--policy.type`, `--policy.path`, `--config_path` and `--policy.device` to pick the environment and accelerator. SmolVLA, the Pi family (pi0, pi05, pi0_fast) and MolmoAct2 have their own environments; other types use the default one. `pai` is an alias of `physicalai`.

Run commands from `library/`. Scripts live in this skill's `scripts/`; set `SKILL=../skills/library/physicalai-train-training-a-lerobot-vla`.

## Workflow

1. **Inspect the checkpoint against the dataset** — policy type, the environment it will use, size, and camera keys:

   ```bash
   .venv/bin/python $SKILL/scripts/inspect_vla.py <hub-id-or-local-dir> <dataset-repo-id>
   ```

   - Done when: you have the policy type, `worker environment` is not `default` for a VLA (otherwise add one with `physicalai-train-adding-a-lerobot-policy-env`), and you have a `--rename_map` or "not needed".

2. **Check it fits.** The script estimates a full float32 fine-tune at ~16 bytes per parameter before activations. Above the GPU's memory, lower `--batch_size` or use the policy's own freezing/dtype options (see its `--help`); gated backbones need `hf auth login` first.

   - Done when: the run is expected to fit, or you have chosen to verify only plumbing (step 4) on this machine.

3. **Build the worker environment** once per accelerator (`cpu`, `cuda`, `xpu`):

   ```bash
   .venv/bin/physicalai engine prepare lerobot --policy <policy_type> --accelerator <cpu|cuda|xpu>
   ```

   - Done when: it prints `Prepared lerobot <family> worker environment: <path>`.

4. **Smoke-test 20 steps.** On Intel XPU, export `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` first.

   ```bash
   .venv/bin/physicalai train lerobot --policy.path=<hub-id-or-local-dir> --policy.device=<cpu|cuda|xpu> \
     --policy.push_to_hub=false --dataset.repo_id=<dataset-repo-id> '--rename_map=<json from step 1>' \
     --batch_size=8 --steps=20 --save_freq=20 --output_dir=<run-dir>
   ```

   Use `--policy.type=<type>` instead of `--policy.path` only to test plumbing: it starts from random weights.

   - Done when: `<run-dir>/checkpoints/000020/pretrained_model/` exists. On failure, check [references/troubleshooting.md](references/troubleshooting.md).

5. **Train for real** with a new `--output_dir` and the step budget you need. Keep `--save_freq` well below `--steps`. Resume a stopped run with:

   ```bash
   .venv/bin/physicalai train lerobot --config_path=<run-dir>/checkpoints/last/pretrained_model/train_config.json --resume=true
   ```

   - Done when: `checkpoints/last/` points at the final step. The resume command picks the same environment from the saved config.

6. **Evaluate through Runtime.** For PushT, fixed seeds 100000+:
   ```bash
   .venv/bin/python $SKILL/scripts/eval_pusht.py native <run-dir>/checkpoints/last/pretrained_model --episodes 50 --device <cpu|cuda|xpu>
   ```
   It packages the checkpoint with `LeRobotEngine.export`, loads it with `physicalai.inference.InferenceModel` and reports `pc_success` and `avg_max_reward` (holding still scores ~0.14). For other tasks, package with `LeRobotEngine.export(checkpoint, output_path=...)` and feed observations whose camera keys match what the checkpoint was trained on.
   - Done when: the JSON report is written next to the checkpoint and its numbers are compared to a baseline or a previous run.

## Required checks

- Step 1 shows a family environment for the policy, or one was added.
- The 20-step smoke run passes before any long run.
- Camera mapping is explicit (`--rename_map` for training, `--camera` for `eval_pusht.py` when keys differ).
- Evaluation and training used the same policy checkpoint and the same device type you will deploy on.

## Verify

```bash
# from library/
.venv/bin/physicalai engine list
.venv/bin/python -m pytest tests/unit/engine ../plugins/policies/lerobot/tests -q
```

## Related skills

- `physicalai-train-adding-a-lerobot-policy-env` — give a LeRobot policy family its own worker environment.
- `physicalai-train-training-a-policy` — first-party Studio policies with `physicalai fit`.
- `physicalai-train-benchmarking-a-policy` — Studio's benchmark suites.
