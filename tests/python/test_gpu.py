# tests/python/test_gpu.py
"""
Unit tests for tensorlang/gpu.py.

These test pure Python logic (device selection, env-var construction) and
deliberately mock out `detect_gpus()` / `nvidia-smi` rather than requiring
real hardware, so they run the same on a CI box with no GPU as on a dev
machine with two 1080 Tis.

Run directly:
    python3 -m unittest discover -s tests/python -v

This is a separate convention from the .tl fixture tests under tests/ and
apps/*/tests/ (which TestRunner discovers and runs as subprocesses) —
there's nothing to compile/execute here, just Python functions to call.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tensorlang.gpu import choose_device, pin_env, gpu_count, detect_gpus


TWO_1080TIS = [
    {"index": 0, "name": "NVIDIA GeForce GTX 1080 Ti",
     "memory_total_mb": 11264, "memory_free_mb": 10400, "utilization_pct": 40},
    {"index": 1, "name": "NVIDIA GeForce GTX 1080 Ti",
     "memory_total_mb": 11264, "memory_free_mb": 11200, "utilization_pct": 0},
]


class TestChooseDevice(unittest.TestCase):

    def test_no_gpus_detected_returns_empty_with_warning(self):
        chosen, warning = choose_device(requested_gpus=1, gpus=[])
        self.assertEqual(chosen, [])
        self.assertIsNotNone(warning)
        self.assertIn("No NVIDIA GPUs detected", warning)

    def test_single_gpu_available_no_warning(self):
        chosen, warning = choose_device(requested_gpus=1, gpus=[TWO_1080TIS[0]])
        self.assertEqual(chosen, [0])
        self.assertIsNone(warning)

    def test_prefers_least_utilized_device(self):
        # GPU0 is busy (40% util, less free mem — e.g. driving the desktop),
        # GPU1 is idle: choose_device should pick GPU1.
        chosen, warning = choose_device(requested_gpus=1, gpus=TWO_1080TIS)
        self.assertEqual(chosen, [1])
        self.assertIsNone(warning)

    def test_requesting_more_than_one_gpu_falls_back_with_warning(self):
        # Tier-A only ever hands back one physical device per process;
        # requesting gpus > 1 in app.toml should degrade gracefully, not
        # crash, until Tier-B (multi-context) support exists.
        chosen, warning = choose_device(requested_gpus=2, gpus=TWO_1080TIS)
        self.assertEqual(len(chosen), 1)
        self.assertIn("only supports pinning a single physical GPU", warning)

    def test_equal_utilization_prefers_more_free_memory(self):
        gpus = [
            {"index": 0, "name": "A", "memory_total_mb": 11264,
             "memory_free_mb": 2000, "utilization_pct": 0},
            {"index": 1, "name": "B", "memory_total_mb": 11264,
             "memory_free_mb": 9000, "utilization_pct": 0},
        ]
        chosen, warning = choose_device(requested_gpus=1, gpus=gpus)
        self.assertEqual(chosen, [1])


class TestPinEnv(unittest.TestCase):

    def test_sets_cuda_visible_devices(self):
        env = pin_env(1, base_env={"PATH": "/usr/bin"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_does_not_mutate_base_env(self):
        base = {"PATH": "/usr/bin"}
        pin_env(0, base_env=base)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", base)

    def test_defaults_to_os_environ_copy(self):
        with patch.dict(os.environ, {"SOME_VAR": "x"}, clear=False):
            env = pin_env(0)
            self.assertEqual(env["SOME_VAR"], "x")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")


class TestDetectGpus(unittest.TestCase):

    def test_missing_nvidia_smi_returns_empty_list_not_exception(self):
        detect_gpus.cache_clear()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(detect_gpus(), [])
            self.assertEqual(gpu_count(), 0)
        detect_gpus.cache_clear()

    def test_parses_nvidia_smi_csv_output(self):
        detect_gpus.cache_clear()
        fake_stdout = (
            "0, NVIDIA GeForce GTX 1080 Ti, 11264, 10530, 2\n"
            "1, NVIDIA GeForce GTX 1080 Ti, 11264, 11256, 0\n"
        )
        fake_result = unittest.mock.Mock(stdout=fake_stdout)
        with patch("subprocess.run", return_value=fake_result):
            gpus = detect_gpus()
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["index"], 0)
        self.assertEqual(gpus[0]["memory_total_mb"], 11264)
        self.assertEqual(gpus[1]["utilization_pct"], 0)
        detect_gpus.cache_clear()


if __name__ == "__main__":
    unittest.main()
