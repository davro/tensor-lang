# tensorlang/gpu.py
"""
GPU detection and process-level pinning for TensorLang.

This module is deliberately kept separate from anything that touches
pycuda: it shells out to `nvidia-smi` to enumerate devices, and never
imports `pycuda.driver` or `pycuda.autoinit`. That matters because
`compiler.py` creates its CUDA context lazily, the first time
`pycuda.autoinit` is imported inside the execution path — and that
context is a singleton for the lifetime of the process. Whatever
`CUDA_VISIBLE_DEVICES` says at that moment is what the process is stuck
with. So the pattern this module supports is:

    1. Detect available GPUs (this module — no CUDA context created).
    2. Decide, per subprocess/app-run, which single device it should see.
    3. Set CUDA_VISIBLE_DEVICES in that process's environment *before*
       tensorlang.py (and therefore pycuda.autoinit) ever runs.

This gives every independent tensorlang process its own GPU, without any
cross-device copies, peer access, or NVLink requirement. It does NOT let
a single running program span multiple GPUs at once — that's a much
bigger change to the execution engine in compiler.py, and isn't what
this module is for.
"""

import os
import subprocess
import functools


@functools.lru_cache(maxsize=1)
def detect_gpus():
    """
    Enumerate NVIDIA GPUs visible to this machine via `nvidia-smi`.

    Returns a list of dicts, one per physical device, e.g.:
        [{"index": 0, "name": "NVIDIA GeForce GTX 1080 Ti",
          "memory_total_mb": 11264, "memory_free_mb": 10530,
          "utilization_pct": 2}, ...]

    Returns [] (never raises) if nvidia-smi is missing, times out, or
    reports no devices — callers should treat that as "no pinning
    available" and fall back to running unpinned, not as an error.
    Cached for the life of the process since the device count/names
    don't change mid-run (utilization/free-memory snapshot is only as
    fresh as the first call — call `detect_gpus.cache_clear()` first
    if you need a fresh read).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, mem_total, mem_free, util = parts
        try:
            gpus.append({
                "index": int(index),
                "name": name,
                "memory_total_mb": int(mem_total),
                "memory_free_mb": int(mem_free),
                "utilization_pct": int(util),
            })
        except ValueError:
            continue
    return gpus


def gpu_count():
    """Number of NVIDIA GPUs detected (0 if none / nvidia-smi unavailable)."""
    return len(detect_gpus())


def pin_env(device_index, base_env=None):
    """
    Build an environment dict with CUDA_VISIBLE_DEVICES pinned to a single
    physical device index, for use with subprocess.run(..., env=...).

    Pass the result as the `env=` kwarg of subprocess.run — the child
    process will see only `device_index` as "device 0", so its own
    (unmodified) `pycuda.autoinit` call binds to that physical GPU.
    """
    env = dict(base_env if base_env is not None else os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    return env


def pin_current_process(device_index):
    """
    Pin *this* process to a single physical device by setting
    CUDA_VISIBLE_DEVICES in os.environ directly.

    Only useful if called before anything in this process has imported
    pycuda.autoinit (e.g. early in AppRunner.run_app, before
    compiler.compile_and_execute is reached). Calling this after a CUDA
    context already exists in this process has no effect on that
    context.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)


def choose_device(requested_gpus=1, gpus=None):
    """
    Pick a device (or devices) to satisfy an app.toml `requirements.gpus`
    count, given a detected device list (defaults to detect_gpus()).

    Returns (chosen_indices, warning) where:
      - chosen_indices is a list of physical device indices, length
        min(requested_gpus, available) — currently always length 1,
        since Tier-A pinning only ever hands one physical device to one
        process; a requested_gpus > 1 app is a Tier-B (multi-context)
        case this function does not attempt to satisfy.
      - warning is a human-readable string if the request couldn't be
        fully satisfied (no GPUs detected, or requested_gpus > 1), else
        None.
    """
    if gpus is None:
        gpus = detect_gpus()

    if not gpus:
        return [], "No NVIDIA GPUs detected (nvidia-smi unavailable or returned none); running unpinned."

    if requested_gpus > 1:
        return [gpus[0]["index"]], (
            f"App requests {requested_gpus} GPUs, but TensorLang currently only "
            f"supports pinning a single physical GPU per process (Tier-A). "
            f"Falling back to device {gpus[0]['index']}."
        )

    # Least-utilized device first, so a busy display GPU (e.g. running
    # Xorg/gnome-shell) isn't preferred over an idle compute-only card.
    best = min(gpus, key=lambda g: (g["utilization_pct"], -g["memory_free_mb"]))
    return [best["index"]], None
