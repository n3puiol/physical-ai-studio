"""Evaluate a policy on PushT with fixed seeds, so different runs and policies compare fairly.

"native":      a LeRobot `pretrained_model` dir (from `physicalai train lerobot`), packaged with
               LeRobotEngine.export and run through Runtime's InferenceModel + LeRobot adapter.
               The gym's `top` camera is passed as `observation.image` (the lerobot/pusht key);
               a `--rename_map` saved with the checkpoint then maps it to the policy's slot.
               Override with `--camera top=<key>`.
"first-party": a Studio Lightning checkpoint (.ckpt) of physicalai.policies.SmolVLA.

Reference points (SmolVLA from lerobot/smolvla_base, 10k steps, batch 8): native avg_max_reward
0.41; holding still scores 0.14 on the same seeds.

Protocol: PushTGym(obs_type="pixels_agent_pos"), seeds start..start+N-1, 300 max steps.
Reports pc_success and avg_max_reward (max coverage, the Diffusion Policy PushT metric).

Usage (from library/; SKILL is skills/library/physicalai-train-training-a-lerobot-vla):
  .venv/bin/python ../$SKILL/scripts/eval_pusht.py native <run>/checkpoints/last/pretrained_model
  .venv/bin/python ../$SKILL/scripts/eval_pusht.py first-party <run>/.../last.ckpt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from physicalai.eval import evaluate_policy
from physicalai.gyms import PushTGym
from physicalai.policies.base import Policy


class _NativeLeRobotPolicy(Policy):
    """Rollout-facing wrapper: Studio Observation -> LeRobot adapter inputs via InferenceModel.

    The PushT gym names its camera ``top``; LeRobot's ``lerobot/pusht`` policies expect
    ``observation.image``. The rename is explicit so a wrong mapping fails loudly.
    """

    def __init__(self, package: Path, device: str, camera_map: dict[str, str]) -> None:
        super().__init__()
        from physicalai.inference import InferenceModel

        self._model = InferenceModel(package, device=device)
        self._camera_map = camera_map
        self.name = "native-lerobot-smolvla"

    def forward(self, batch: Any) -> torch.Tensor:  # noqa: ANN401
        return self.select_action(batch)

    def predict_action_chunk(self, batch: Any) -> torch.Tensor:  # noqa: ANN401
        return self.select_action(batch)

    def select_action(self, observation: Any) -> torch.Tensor:  # noqa: ANN401
        obs = observation.to_numpy().to_dict(flatten=False)
        inputs: dict[str, Any] = {"state": obs["state"], "task": obs["task"]}
        for camera, frame in obs["images"].items():
            if camera not in self._camera_map:
                msg = f"No LeRobot key for gym camera {camera!r}; pass --camera {camera}=observation.<key>"
                raise KeyError(msg)
            inputs[self._camera_map[camera]] = frame
        action = self._model.select_action(inputs)
        return torch.from_numpy(np.asarray(action, dtype=np.float32)).reshape(1, -1)

    def reset(self) -> None:
        self._model.reset()


def _native_policy(path: Path, device: str, camera_map: dict[str, str], work: Path) -> Policy:
    from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

    if (path / "manifest.json").is_file():
        package = path
    else:
        package = work / "runtime-package"
        if not (package / "manifest.json").is_file():
            package = Path(LeRobotEngine.export(path, output_path=package))
    return _NativeLeRobotPolicy(package, device, camera_map)


def _first_party_policy(path: Path, device: str) -> Policy:
    from physicalai.policies import SmolVLA

    policy = SmolVLA.load_from_checkpoint(str(path), map_location="cpu")
    policy.name = "first-party-smolvla"
    return policy.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arm", choices=["native", "first-party"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--camera", action="append", default=["top=observation.image"], metavar="GYM=LEROBOT")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or args.path.parent / f"eval-{args.arm}-{args.episodes}ep.json"
    if args.arm == "native":
        camera_map = dict(item.split("=", 1) for item in args.camera)
        policy = _native_policy(args.path, args.device, camera_map, out.parent)
    else:
        policy = _first_party_policy(args.path, args.device)

    env = PushTGym(obs_type="pixels_agent_pos")
    started = time.perf_counter()
    try:
        result = evaluate_policy(env, policy, args.episodes, start_seed=args.start_seed, max_steps=args.max_steps)
    finally:
        env.close()
    aggregated = result["aggregated"]
    report = {
        "arm": args.arm,
        "path": str(args.path),
        "device": args.device,
        "protocol": {"episodes": args.episodes, "start_seed": args.start_seed, "max_steps": args.max_steps},
        "wall_s": round(time.perf_counter() - started, 1),
        "aggregated": aggregated,
        "per_episode": result["per_episode"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{args.arm}: success {aggregated.get('pc_success', 0.0):.1f}%  "
        f"avg_max_reward {aggregated['avg_max_reward']:.3f}  "
        f"len {aggregated['avg_episode_length']:.0f}  fps {aggregated['avg_fps']:.1f}  -> {out}"
    )


if __name__ == "__main__":
    main()
