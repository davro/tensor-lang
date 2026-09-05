"""
Chunked-invocation runner for TensorLang apps.

TensorLang compiles ahead-of-time from a single .tl file per app, and
(a) save() targets are fixed string literals, and (b) rebind
(`x = x_updated`) requires a fixed-shape GPU buffer each iteration. So a
single main.tl can't accumulate a growing history of snapshots inside
one `for` loop. The pattern that does work: have main.tl load()/save()
its own state (weights, sim state, whatever) to/from a fixed path on
every run, and invoke it repeatedly as separate OS processes — each
invocation is a "chunk" that resumes from wherever the last one left
off. This module drives that repeated invocation and collects one
frame per chunk into a single stacked numpy array.
"""
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk upward looking for tensorlang.py, which marks the repo root."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here] + list(here.parents):
        if (candidate / "tensorlang.py").exists():
            return candidate
    raise RuntimeError(f"Could not locate tensor-lang repo root above {here}")


def cache_dir_for(app_category_path: str, repo_root: Optional[Path] = None) -> Path:
    """The directory TensorLang saves this app's save()/load() outputs into.

    Mirrors compiler.py's cache_base / file_path logic exactly:
        cache/apps/<app_category_path>/main.tl/
    e.g. cache_dir_for("examples/decision_boundary")
      -> <repo_root>/cache/apps/examples/decision_boundary/main.tl
    """
    repo_root = repo_root or find_repo_root()
    return repo_root / "cache" / "apps" / app_category_path / "main.tl"


def run_one(app_category_path: str, repo_root: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Invoke `python3 tensorlang.py --app <app_category_path>` once, raising
    with the captured output if the process itself crashed (nonzero exit).

    Note: a 0 exit code does NOT guarantee the compile/run actually
    succeeded — app_runner.py calls compiler.compile_and_execute()
    without checking its return value, so some compiler failures (e.g.
    a load() shape mismatch) print an error but still exit 0. Callers
    that depend on an artifact existing afterward should still handle
    that artifact being missing; run_chunks does this by showing the
    captured output when collect_fn fails.
    """
    repo_root = repo_root or find_repo_root()
    result = subprocess.run(
        [sys.executable, "tensorlang.py", "--app", app_category_path],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"tensorlang.py --app {app_category_path} failed — see output above")
    return result


def run_chunks(
    app_category_path: str,
    n_chunks: int,
    collect_fn: Callable[[Path], np.ndarray],
    repo_root: Optional[Path] = None,
    progress_fn: Optional[Callable[[int, int], Optional[str]]] = None,
) -> np.ndarray:
    """Run `app_category_path` n_chunks times. After each run, calls
    collect_fn(repo_root) to pull that chunk's frame — typically a
    np.load() from cache_dir_for(app_category_path). Returns the
    stacked frames as one array, shape (n_chunks, *frame.shape).

    progress_fn(chunk_index, n_chunks) -> optional string to print after
    each chunk; if omitted, a plain "chunk i/n" line is printed instead.
    """
    repo_root = repo_root or find_repo_root()
    frames = []
    for chunk in range(n_chunks):
        result = run_one(app_category_path, repo_root)
        try:
            frames.append(collect_fn(repo_root))
        except Exception as e:
            print(f"\n--- chunk {chunk + 1}/{n_chunks}: collect_fn failed ({e}) ---")
            print("--- captured stdout from this chunk's tensorlang.py run ---")
            print(result.stdout)
            print("--- captured stderr ---")
            print(result.stderr)
            raise
        if progress_fn:
            msg = progress_fn(chunk, n_chunks)
            if msg:

                print(msg)
        else:
            print(f"chunk {chunk + 1:3d}/{n_chunks}")
    return np.stack(frames, axis=0)


def save_frames(frames: np.ndarray, out_path: Path) -> Path:
    """Convenience: mkdir -p the parent, save, return the resolved path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, frames)
    return out_path