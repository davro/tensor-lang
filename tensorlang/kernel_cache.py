"""Cache directories and the shared kernel build for data-parallel (--ddp) runs. Standard library only.

An ordinary run keeps everything in cache/<program>/ exactly as before. A run with WORLD_SIZE > 1 splits it:

    cache/<program>/ws{N}/            kernel.cu, kernel.so, .lock: one kernel, shared by every rank
    cache/<program>/ws{N}/rank{r}/    this rank's tensor outputs, save() targets and layer caches

The kernel is built once. Every rank takes the same exclusive file lock; whichever gets it first builds, the others
find the kernel finished when their turn comes and reuse it. (The lock is a file lock because no communicator exists
yet at compile time, and the operating system drops it if the holder dies, so a crashed build never blocks the run.)

A build is written to temporary names and moved into place, the library first and the source last, so "kernel.cu matches
the generated code and kernel.so exists" can only be true once the library is complete. The ordinary single-process path
writes kernel.cu first; a failed nvcc there can leave a new source next to an old library.

Generated kernels may depend on world_size but must never depend on rank: one artifact has to be valid for every rank.
The launcher gives all ranks of a run the same TENSORLANG_DDP_RUN_ID, and the build records which code it was built from.
A rank that generates different code during the same run is stopped with an error instead of silently rebuilding.
"""
import hashlib
import os
from pathlib import Path

try:
    import fcntl
except ImportError:                                        # not POSIX
    fcntl = None

RUN_ID_ENV = "TENSORLANG_DDP_RUN_ID"


class KernelCacheError(RuntimeError):
    """The kernel cache cannot be used safely (ranks generated different code, or no file locking here)."""


def cache_dirs(base, file_path, rank=0, world_size=1):
    """Return (kernel_dir, output_dir) for a program.

    Ordinary runs (world_size 1): both are base/file_path, as always. Data-parallel runs: the kernel directory is
    base/file_path/ws{world_size} and the output directory is that directory's rank{rank} subdirectory."""
    root = Path(base) / file_path
    if world_size <= 1:
        return root, root
    kernel_dir = root / f"ws{world_size}"
    return kernel_dir, kernel_dir / f"rank{rank}"


def _digest(cuda_code):
    return hashlib.sha256(cuda_code.encode()).hexdigest()[:16]


def build_kernel_shared(kernel_dir, cuda_code, compile_fn, run_id=None):
    """Make kernel_dir/kernel.so match `cuda_code`, building it at most once across all processes.

    compile_fn(cu_path, so_path) must compile cu_path to so_path and raise if it fails; it is called with temporary
    paths (the .cu name must end in .cu for nvcc). Returns True if this call built the kernel, False if it was already
    there. `run_id` defaults to $TENSORLANG_DDP_RUN_ID; without one, the different-code check is skipped."""
    if fcntl is None:
        raise KernelCacheError("--ddp needs POSIX file locking (fcntl), which is not available on this platform")
    kernel_dir = Path(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get(RUN_ID_ENV) if run_id is None else run_id
    digest = _digest(cuda_code)
    cu, so = kernel_dir / "kernel.cu", kernel_dir / "kernel.so"
    marker = kernel_dir / ".built_by"

    with open(kernel_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if run_id and marker.exists():
            try:
                built_run, built_digest = marker.read_text().split()
            except ValueError:
                built_run = built_digest = None
            if built_run == run_id and built_digest != digest:
                raise KernelCacheError(
                    "ranks generated different kernel code during the same run (this rank's differs from the code "
                    f"already built in {kernel_dir}); generated code must not depend on the rank")
        if so.exists() and cu.exists() and cu.read_text() == cuda_code:
            return False

        pid = os.getpid()
        tmp_cu, tmp_so = kernel_dir / f"kernel.tmp{pid}.cu", kernel_dir / f"kernel.tmp{pid}.so"
        try:
            tmp_cu.write_text(cuda_code)
            compile_fn(str(tmp_cu), str(tmp_so))
            os.replace(tmp_so, so)                          # library first,
            os.replace(tmp_cu, cu)                          # then the source that vouches for it
            if run_id:
                marker.write_text(f"{run_id} {digest}")
        finally:
            for tmp in (tmp_cu, tmp_so):
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
        return True
