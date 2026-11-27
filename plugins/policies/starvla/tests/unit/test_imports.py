# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prove Runtime-facing plugin imports never require Studio or model frameworks."""

import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Fixed interpreter and test script.
import sys


def test_runtime_imports_do_not_load_training_dependencies() -> None:
    code = """
import importlib.abc
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "starVLA"} or fullname.startswith("physicalai.engine"):
            raise AssertionError("Runtime imported training dependency: " + fullname)
sys.meta_path.insert(0, BlockTraining())
from physicalai.inference import InferenceModel
from physicalai_starvla_policy_plugin import environments, inference, openvino_runtime, package, runtime
assert environments.QWENGR00T.provenance()["revision"]
assert runtime.StarVLARuntimeAdapter().output_names == ["action"]
assert openvino_runtime.StarVLAOpenVINOAdapter().default_device() == "CPU"
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # No untrusted arguments.
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert result.returncode == 0, result.stderr
