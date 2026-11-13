# Plan: train one remote job on all GPUs

**Problem.** On a remote SSH trainer, the container sees every GPU (`--gpus all` in
`services/ssh/docker_ops.py`), but each job trains on exactly one GPU because
`training/device.py::resolve_devices` hands Lightning `[device_index]` or `1`.
Fine-tuning takes 2+ days on one GPU.

**Goal.** Add an opt-in "All GPUs" device choice for remote CUDA trainers. A job with
it takes the whole trainer and trains with DDP across every visible GPU. Single-GPU
jobs keep working exactly as today.

**Out of scope (first pass).** Multi-XPU (needs DDP + `ccl`), local Studio training,
AWS Batch, multi-node.

Each step below is a separately testable change. Do them in order; don't start a step
until the previous one's "Test" section passes.

---

## Step 0 — Baseline on the multi-GPU host

No code change. Gives you numbers to compare against at the end.

1. Submit a representative fine-tuning job to the remote trainer as you do today.
2. On the host, while it runs: `watch -n1 nvidia-smi`.
3. Note from the job's `metrics.csv` / progress: time per epoch (or steps/s) and peak memory.

**Test / pass criteria:** exactly one GPU shows utilization; you have a recorded epoch time.

---

## Step 1 — Prove the policy trains under DDP (no Studio code)

This tests the riskiest assumption first: that the policy you care about works with
DDP and actually gets faster. The library already switches to
`DDPStrategy(find_unused_parameters=True)` when `devices > 1`
(`library/src/physicalai/train/trainer.py:206`).

On the GPU host, from a checkout with `uv sync --extra cuda`:

```bash
uv run --no-sync physicalai fit --config <your-policy-config>.yaml --trainer.devices 2
```

(Use `--trainer.devices -1` for all GPUs.) Running from a shell is fine here: Lightning's
default DDP launcher re-executes `sys.argv`, which is the CLI itself.

**Test / pass criteria**
- `nvidia-smi` shows all selected GPUs busy.
- No DDP errors (unused parameters, NCCL timeouts, OOM).
- Time per epoch is clearly lower than in Step 0. Note that `batch_size` is per GPU, so
  each optimizer step sees N× the samples; compare wall-clock per epoch, not per step.
- Exactly one checkpoint is written.

If this fails or gives no speedup, stop and fix it in the library before going further.

**Result (2026-11-13, 2× RTX 3090, ACT, 100-episode sim SO-101 dataset, batch 32/GPU, 1 epoch,
via `scripts/try_multi_gpu.py`, i.e. the Step 2–3 code path):**

| | 1 GPU | 2 GPUs (DDP) |
|---|---|---|
| Optimizer steps | 4410 | 2206 |
| Throughput | 4.03 it/s | 3.48 it/s per GPU (86 % efficiency) |
| Training loop | 18:37 | 10:47 (**1.73×**) |
| Whole job incl. val + export | 1204 s | 708 s (**1.70×**) |
| Final train / val loss | 0.267 / 0.318 | 0.361 / 0.352 |

Same val split on both ranks, only rank 0 reported and exported, no hang at the end.
Loss is higher after one epoch because DDP takes half as many optimizer steps at the same LR
(effective batch 64); see the open questions on batch size and LR.

---

## Step 2 — Spec field and device resolution (pure unit change)

Files: `src/training/job.py`, `src/training/device.py`.

- Add `all_devices: bool = False` to `TrainingJobSpec`.
- Model validator: when `all_devices` is true, require `device_type == "cuda"` and
  `device_index is None`. Also reject `auto_scale_batch_size=True`: Lightning's batch
  size finder doesn't support distributed strategies.
- `resolve_devices(device_index, all_devices=False)` returns `-1` when `all_devices`.
- `run_training` passes `spec.all_devices` through.

**Test**
```bash
uv run pytest tests/training -q
```
New tests:
- `resolve_devices(all_devices=True) == -1`; existing `resolve_devices(2) == [2]` and `resolve_devices() == 1` unchanged.
- Spec validation: `all_devices` + `device_index=1` → error; `all_devices` + `device_type="xpu"` → error; `all_devices` + `auto_scale_batch_size` → error.
- An old JSON spec without the field still parses (default `False`).

Nothing uses the field yet, so nothing changes at runtime.

---

## Step 3 — Rank-0 guards in `run_training`

File: `src/training/job.py` (`run_training`).

Under DDP every rank runs `run_training`. Only global rank 0 may:
- call `report(...)` (wrap it: `report if trainer.is_global_zero else no-op`; `ProgressReportingCallback` already only fires on rank 0 if it uses `trainer.is_global_zero`, so check that and fix it if not),
- run `_finalize_checkpoints`, `_publish`, `_export_policy`, `_export`.

Every rank must still run `_finalize_checkpoints`: its fallback `trainer.save_checkpoint` is a
collective call (barrier inside), so skipping it on any rank hangs rank 0. After it, all ranks meet
at `trainer.strategy.barrier(...)`, then non-zero ranks return and rank 0 publishes and exports.
(Done: `report` is wrapped with Lightning's `rank_zero_only`, because the library's
`ProgressReportingCallback` reports on every rank.)

**Test**
```bash
uv run pytest tests/training tests/trainer/test_runner.py -q
```
New test: patch `Trainer` so `is_global_zero` is `False`; assert that `_publish` and `_export` are
not called and `report` is called at most once (the "Training model" call should also be gated).
Existing single-GPU tests must still pass, because `is_global_zero` is `True` there.

---

## Step 4 — Standalone DDP job entrypoint (`torchrun`)

New file: `src/trainer/ddp_entry.py`, runnable as `python -m trainer.ddp_entry`.

Why it's needed: the trainer service runs jobs in a thread or in an `mp.spawn` child
(`trainer/queue_worker.py::_train_in_process`). Lightning's default DDP launcher would
re-execute the trainer service, and `ddp_spawn` can't pickle the `report`/`should_stop`
closures. Under `torchrun`, Lightning sees the elastic env vars and doesn't launch processes itself.

Interface:
```
python -m torch.distributed.run --standalone --nproc_per_node=<N> \
    -m trainer.ddp_entry --job-id <id> --request <request.json> --events <events.jsonl>
```
- Every rank loads the `SubmitJobRequest` JSON and trains. Only rank 0 may then call
  `archive_model` and clean up the uploaded dataset: `TrainerRunner.run` does both after training,
  and on a non-zero rank `models_dir/<job>` doesn't exist (rank 0 owns `output_dir`).
- Rank 0 (`RANK == 0`) appends progress to `events.jsonl` as
  `{"type": "progress"|"completed"|"failed", ...}`; other ranks get a no-op `report`.
- `should_stop` always returns `False`; cancellation is a kill of the process group (Step 5).
  That avoids ranks stopping on different steps and hanging in all-reduce.
- The HF token comes in through the environment (`HF_TOKEN`), never through the request file.

**Test (manual, on the GPU host, no service running)**
1. Dump a real request: submit a job once, or write `request.json` by hand with
   `"spec": {..., "device_type": "cuda", "all_devices": true}` and a small `max_epochs`.
2. Run the command above with `--nproc_per_node=$(nvidia-smi -L | wc -l)`.
3. Check:
   - `nvidia-smi` shows all GPUs busy;
   - `events.jsonl` has monotonically increasing progress and ends with `completed`;
   - the model dir contains a single checkpoint and the exports;
   - exit code is 0, and no stray python processes are left (`pgrep -f ddp_entry`).
4. Run again and press Ctrl-C mid-epoch: all ranks exit, GPUs free up.

Unit test (CPU, no GPUs): run `ddp_entry.main()` with `RANK=1` and assert it writes no events
and doesn't publish.

---

## Step 5 — Queue worker: dispatch and reservation

File: `src/trainer/queue_worker.py`.

- `_next_runnable`: treat `spec.all_devices` like the existing "legacy auto-selection" path
  (`device is None`). It waits until `_active` is empty and no GPU is busy, and while it
  runs it blocks every other job. Message: `"Waiting for all GPUs to become available"`.
- `_run_job`: if `spec.all_devices`, call a new `_run_ddp(...)` instead of
  `self._runner.run` / `_run_isolated`. `_run_ddp`:
  - writes the request JSON (without the token) to the job's cache dir;
  - starts `torchrun` with `asyncio.create_subprocess_exec(..., start_new_session=True)`,
    `env={..., "HF_TOKEN": token}`, `nproc_per_node = torch.cuda.device_count()`;
  - tails `events.jsonl` every 0.5 s and forwards progress to `report`;
  - on cancel, sends `os.killpg(pgid, SIGTERM)`, then `SIGKILL` after a grace period, and raises `JobCanceledError`;
  - a non-zero exit with no `completed` event → `RuntimeError` with the tail of stderr.
- Prefix logs per rank or only forward rank 0's stdout, so the job log isn't garbled.

**Test**
```bash
uv run pytest tests/trainer/test_queue_worker.py -q
```
New tests (stub `_run_ddp` / the subprocess):
- an all-GPU job stays queued while a single-GPU job runs, with the right message;
- a single-GPU job stays queued while an all-GPU job runs;
- an all-GPU job dispatches to `_run_ddp`, while a normal job still goes to `_run_isolated`;
- cancelling an all-GPU job kills the process group and the job ends `CANCELED`.

Manual: start the service from the branch (`uv run --no-sync physicalai-trainer`) and submit with
curl to `POST /jobs` with `all_devices: true`. Upload the dataset as Studio does, then check the
job reaches `COMPLETED` and `GET /jobs/<id>` shows progress.

**Result (2026-11-13, 2× RTX 3090, ACT, batch 32/GPU, 1 epoch):** Step 4 run by hand under torchrun
and Step 5 through the trainer API (`POST /jobs`, streamed `PUT /dataset` with `curl -T`) both
completed; the archive held `model.ckpt` and `exports/torch`, and `ddp/`, `datasets/`, `cache/` were
empty afterwards. A cancel sent right after upload ended the job `canceled` with no process, GPU
memory, or files left behind. Not yet exercised: a cancel mid-epoch.

---

## Step 6 — Studio backend: request plumbing

Files: `src/schemas/job.py` (`TrainingDevice`), `src/services/training_backends/remote.py`,
`local.py`, `aws_batch.py`.

- Add `all_devices: bool = False` to `TrainingDevice`. Validator: only with `type == cuda` and
  `index is None` (relax the "indexed types need an index" check for this case).
- `remote.py::submit_job`: map to `{"device_type": "cuda", "device_index": None, "all_devices": True}`.
  Replace the "selected device no longer available" check with "the trainer reports ≥ 2 CUDA devices".
- `local.py` and `aws_batch.py`: reject `all_devices` with a clear error (not supported yet).
- Bump the trainer API protocol version if the trainer's spec schema is strict about unknown fields
  (`TRAINER_API_PROTOCOL_VERSION`), so an old trainer image isn't silently sent a field it ignores.

**Test**
```bash
uv run pytest tests/services tests/schemas -q
```
- `TrainingDevice(type="cuda", all_devices=True)` is valid; with `index=0` or `type="xpu"` it's invalid.
- The remote backend's submitted payload contains `all_devices: true`.
- The local backend rejects it.

Then regenerate the UI types (backend running):
```bash
cd application/ui && npm run build:api:download && npm run build:api
```

---

## Step 7 — UI: "All GPUs" option

File: `application/ui/src/features/models/train-model-dialog/setup-step.tsx`
(plus `training-device-info.tsx` for the hint).

- When the remote trainer reports ≥ 2 CUDA devices, add an item `cuda:all` →
  "All CUDA GPUs (N)". Mark it busy if any GPU is busy. Pre-select it when it's available
  (the issue asks for this as the default).
- Selecting it sends `{ type: 'cuda', index: null, all_devices: true }`.
- Hint text: "Batch size is per GPU: effective batch = batch size × N." Disable the
  auto batch size toggle while it's selected.

**Test**
```bash
cd application/ui && npx vitest run src/features/models/train-model-dialog
```
- The option appears with 2+ CUDA devices and is absent with 1 device or with XPU.
- The submitted payload matches.

Manual: open the training dialog against the remote trainer and check the option, the hint and the busy badge.

---

## Step 8 — End-to-end on the real remote trainer

The SSH trainer runs a published image (`protocol-<N>` tag), so either run the trainer service
from your branch checkout on the host (`uv run --no-sync physicalai-trainer`), or point Studio at a
`-dev-<sha>` image built from the branch (`docs/remote-trainer.md`, "Images").

Checklist:
- [ ] Same job as Step 0 with "All GPUs": every GPU busy, epoch time ≈ Step 0 / N (expect 70–90 % scaling).
- [ ] Progress bar and metrics in Studio update smoothly (no duplicate or jumping progress).
- [ ] The finished model downloads, loads and runs inference / export like a single-GPU model.
- [ ] Cancel mid-epoch: job goes `CANCELED`, `nvidia-smi` shows no leftover processes within ~30 s.
- [ ] Kill one rank (`kill -9 <pid>`): job goes `FAILED` with a readable message, other ranks exit.
- [ ] While an all-GPU job runs, a second job (single GPU) queues with "Waiting…" and starts afterwards.
- [ ] Regression: a normal single-GPU job still runs, and two single-GPU jobs still run in parallel.

Then run the hooks: `prek run --all-files application/backend/` and `prek run --all-files application/ui/`.

---

## Open questions

1. **Batch size semantics.** Keep `batch_size` per GPU (simplest, matches Lightning) or divide it by N
   so results match single-GPU runs? The plan keeps it per GPU and documents that.
2. **Learning rate.** With an N× larger effective batch, some policies need LR scaling. Leave it to the user for now?
3. **Default.** Should "All GPUs" be pre-selected (as the issue asks), or opt-in?
4. **`TRAINER_MAX_CONCURRENT_JOBS`.** Keep the current default; the all-GPU job reserves the whole trainer anyway.
