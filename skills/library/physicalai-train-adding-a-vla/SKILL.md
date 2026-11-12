---
name: physicalai-train-adding-a-vla
description: Adds support for a new vision-language-action model (VLA) to Physical AI Studio, either one LeRobot already implements or one that lives in its own repository. Use when asked to "add", "support", "try" or "integrate" a VLA such as X-VLA, EO-1, Wall-X, OpenVLA, RDT, Octo or a newly released model, when deciding between the LeRobot engine and a new engine plugin, or when creating a plugin under plugins/policies/<name>/ like the GR00T plugin.
license: Apache-2.0
---

# Adding a VLA

Studio runs external VLAs through **engines**: the model keeps its own training code and dependencies in a locked worker environment, and Studio provides discovery, `physicalai train <engine>`, worker environments (`physicalai engine prepare/list/prune`) and Runtime packaging. Background: `library/docs/explanation/policy/engine-and-frameworks.md` (enablement stages: framework plugin → repository plugin → enablement → first-party policy).

Pick the cheapest stage that works:

| The VLA is…                               | Path                                                                                                                              | Effort                                                                         |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| A LeRobot policy type                     | LeRobot engine (below)                                                                                                            | A worker-environment project and one mapping line                              |
| In its own repository                     | New repository plugin, modelled on `plugins/policies/gr00t/` ([references/repository-plugin.md](references/repository-plugin.md)) | A new package: environment, engine, workers, packaging, Runtime adapter, tests |
| Needed with Studio-native training/export | First-party policy (`physicalai-train-adding-a-policy`)                                                                           | Port the model                                                                 |

## Workflow

1. **Check whether LeRobot implements it** (from `library/`):

   ```bash
   .venv/bin/python -c "import lerobot.policies; from lerobot.configs.policies import PreTrainedConfig as C; print(sorted(C.get_known_choices()))"
   ```

   - Done when: you know the LeRobot policy type, or that there is none. A newer LeRobot release may add it; the plugin pins `lerobot>=0.6.1,<0.7`, so a newer version means re-locking the environments.

2. **Check the model before writing code**: checkpoint id, parameter count, camera keys, gated weights or tokenizers, and the hardware it needs. For a LeRobot checkpoint run `scripts/inspect_vla.py <checkpoint> <dataset>` (from `library/`: `.venv/bin/python ../skills/library/physicalai-train-adding-a-vla/scripts/inspect_vla.py ...`); for other repositories read their model card and config.

   - Done when: you know whether a full fine-tune fits the target GPU (~16 bytes per parameter in float32) or which freezing/LoRA option to use.

3. **LeRobot path** (type found in step 1): follow [references/lerobot-path.md](references/lerobot-path.md).

   1. Part A — give the family its own environment (`envs/<family>/`, lock, `LeRobotEngine.policy_envs`, tests).
   2. Part B — train and evaluate (`physicalai engine prepare lerobot --policy`, 20-step smoke run, `--rename_map`, resume, `scripts/eval_pusht.py` through Runtime).

   Failures and fixes: [references/troubleshooting.md](references/troubleshooting.md).

   - Done when: a 20-step run through `physicalai train lerobot` writes a checkpoint and `eval_pusht.py` (or your task's evaluation) loads it through Runtime.

4. **Repository path** (not in LeRobot): follow [references/repository-plugin.md](references/repository-plugin.md) section by section. Copy from `plugins/policies/gr00t/` rather than writing from scratch.

   - Done when: `physicalai engine prepare <name>`, a short `physicalai train <name> ...`, `<Engine>.export(...)` and `physicalai.inference.InferenceModel(<package>)` all work, and the plugin's tests pass.

5. **Record what was verified.** Say which steps ran on which hardware, and which were only built or imported (large models often cannot be trained on the test machine).
   - Done when: the PR or summary lists verified vs. unverified steps.

## Required checks

- The model's dependencies live only in its worker environment; Studio's `library/` environment gains at most the plugin's small host package.
- Everything that crosses from host to worker is arguments, files or JSON/array messages — never imported framework objects.
- A Runtime-only install can load and run the exported package without `physicalai-train`.
- Training and evaluation use explicit camera/state key mappings.

## Related skills

- `physicalai-train-adding-a-policy` — first-party port.
- `physicalai-train-exporting-and-validating` — the Runtime export/load contract.
