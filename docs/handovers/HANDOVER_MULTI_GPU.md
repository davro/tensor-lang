# TensorLang Multi-GPU Design Handover

**Date:** 2026-10-23
**Status:** Design (not yet implemented). Supersedes the 2026-10-22 draft.
**Goal:** Add opt-in data-parallel training (DDP) across multiple GPUs with **zero change** to single-GPU behaviour. All 109 existing tests must keep passing untouched. Leave a clear path to explicit collectives and, much later, FSDP.

This document is self-contained. It records what the codebase actually does (verified by reading the sources), the resulting design, and the order to build it.

---

## 1. Summary of the Design

1. **SPMD, one process per GPU.** Every rank runs the same `.tl` program on its own GPU. Tensors are rank-local.
2. **Launcher, not syntax.** `python tensorlang.py --ddp --nproc N file.tl` spawns N workers with torchrun-style env vars and a pinned `CUDA_VISIBLE_DEVICES` each.
3. **v1 adds no new `.tl` syntax.** Under `--ddp` (i.e. `WORLD_SIZE > 1`) the compiler averages gradients inside `backward`. Programs run without `--ddp` take today's code path exactly.
4. **Host-side collectives first.** Backward currently runs on the CPU in NumPy, so gradients are host arrays before they are uploaded. A small TCP/host backend is enough for v1. NCCL is deferred until backward moves to the GPU.
5. **`world_size` is a launch-time constant**, read from the environment before type-checking, so shapes such as `all_gather` outputs can be checked exactly.
6. **Explicit collectives** (`all_reduce`, `broadcast`, `all_gather`) are a second step, added as host-op nodes with a small grammar extension.

**Naming, relative to HANDOVER.md section 18 and the README.** *Tier-A* = independent processes, one pinned GPU each (implemented). *Tier-B* = one program spanning several GPUs (not started; **this design does not do it**). This design is **Tier-C**: N cooperating processes, one GPU each, exchanging gradients through collectives. It builds directly on Tier-A pinning and leaves the single-context execution engine alone. Use these three names consistently.

**Expected payoff (confirm before investing heavily).** Backward runs on the host in NumPy, and section 18.3 of HANDOVER.md measured the machine as already close to CPU-core-bound (about 4.6-5x observed parallelism out of 8 jobs). Two ranks therefore run two host backward passes, so scaling is limited by CPU cores and per-step host/device copies, not by GPU count. Before building steps 5 onward, measure one real training workload: what fraction of an iteration is host backward plus the device-to-host and host-to-device copies? If it dominates, moving backward onto the GPU is worth more than DDP, and it is also the precondition for NCCL being useful (section 6.6). Steps 1 to 4 are cheap and low-risk either way.

---

## 2. What the Codebase Does Today (verified facts)

These drive every decision below.

| Area | Fact | Consequence |
|---|---|---|
| Entry (`tensorlang.py`) | `compiler.compile_and_execute(file)` parses, type-checks, generates CUDA, runs `nvcc`, loads the `.so` and executes, all in one process. | "Specialise at launch" is nearly free: read env, then compile. |
| GPU binding | `pycuda.autoinit` is imported lazily in the execution phase, after `nvcc`. The context binds to whatever `CUDA_VISIBLE_DEVICES` was at that moment, for the life of the process. | Pinning must be in the worker's environment at exec time. Inside a worker the device is always index 0. |
| `gpu.py` | Never imports pycuda; shells out to `nvidia-smi`. `pin_env()` / `pin_current_process()` set `CUDA_VISIBLE_DEVICES`. | Safe for a launcher parent process to import. |
| Backward | `ComputationGraph.backward` is pure NumPy. Per iteration the compiler copies all tensors device-to-host, runs backward on the host, then uploads `NAME_grad` tensors with `memcpy_htod` (for-loop path around `compiler.py` lines 237-285; top-level `backward` around lines 1686-1715). | Gradient averaging fits between `comp_graph.backward(...)` and the upload. No NCCL needed. |
| Kernel cache | Every run writes `cache/<file>/kernel.cu`, `kernel.so` and `*.npy` outputs into one shared directory. `nvcc` is skipped only if the generated source is byte-identical to the cached copy. | N ranks would race on `kernel.so` and on output files. See section 6. |
| Kernel generation | `kernel_generator.py` emits one CUDA kernel per op. Collectives are not CUDA kernels. | No kernel-generator changes. |
| Non-kernel nodes | `backward`, `save`, `rebind` are skipped by the kernel loops; for-loop `_flat_items` carries `None` as the kernel index for them. | Explicit collectives slot in as host-op nodes of the same kind. |
| Type checker | `type_checker(ast, env, DEBUG_INFO, DEBUG_MODE)`; `env` maps `name -> {dtype, shape}`. The `for` branch recurses with positional args. `load` reads the file at compile time to infer shape. | `world_size` must be a new keyword argument, not an env key. Placeholders in `load` paths must be resolved before type-checking. |
| Grammar (`tensorlang.lark`) | LALR. `shape` accepts only `NUMBER` literals. Reductions take `NAME` args, not `inline_expr`. No device-placement syntax exists. `for NAME in range(NUMBER)` only. | No `world_size()` in shapes. No placement syntax needed. |
| Language limits | No lists, lambdas, `map`, module objects, or `next_batch`. `fn` bodies are inlined with tensor params. For bodies support `let`, rebind, `backward`, `save`. | The earlier draft's `ddp_wrap`, `ddp_average_grads(grads: [Tensor])`, `model.weights` cannot be expressed. Dropped. |
| Test runner | Spawns `python3 tensorlang.py <test>` per test with `pin_env()`, round-robined across GPUs, in parallel. | DDP tests need their own serial category. `pin_env()` should drop `RANK`, `LOCAL_RANK`, `WORLD_SIZE`. |

### Known issues found along the way (not DDP-specific)

- `if_statement` is in the grammar (`statement` and `for_body_statement`) but neither `ast_builder.py` nor `compiler.py` appears to handle it, so `if` blocks look silently dropped. **Unconfirmed; verify.** It rules out rank-gated logic such as `if rank == 0`, which is one reason this design uses path placeholders and launcher-level output handling instead.
- `for` is limited to a literal iteration count and four statement kinds. Both `if` and `for` need substantial work independent of this feature.
- The type checker does not register tensors whose declared shape contains string (generic) dims.

---

## 3. Execution Model: SPMD

- Every process runs the same program, owns one GPU, and holds its own copy of every tensor. A tensor `w` on rank 0 and `w` on rank 1 are different buffers.
- Data crosses ranks only through collectives, never implicitly.
- Single-process multi-device (pipeline or manual model parallelism) is **out of scope**.
- No placement syntax in v1. The earlier `@ gpu(N)`, `@ gpu(local)` and `device default` are dropped. If single-process multi-device is ever wanted, `@ gpu(N)` is reserved for it.

---

## 4. Launcher

`python tensorlang.py --ddp --nproc N file.tl`

**Parent process:** spawns N workers and waits. It must never import `pycuda.autoinit` (that would create a context on GPU 0).

**Worker environment** (torchrun convention):

| Var | Value |
|---|---|
| `RANK` | 0..N-1 |
| `LOCAL_RANK` | same as `RANK` (single node) |
| `WORLD_SIZE` | N |
| `MASTER_ADDR` / `MASTER_PORT` | `127.0.0.1` / free port (default 29500) |
| `CUDA_VISIBLE_DEVICES` | physical GPU for this rank, set via `pin_env()` |

**GPU selection:** if the parent's `CUDA_VISIBLE_DEVICES` is set, rank r takes its r-th entry; otherwise the r-th index from `detect_gpus()`. If N exceeds available GPUs, fail before spawning, unless `--ddp-same-gpu` is given (see section 9).

**Argv:** workers receive the parent's argv unchanged. The guard is `RANK in os.environ`; workers skip the launcher. Warn if `--ddp` is given while `RANK` is already exported.

**Failure:** `compiler.py` calls `sys.exit(1)` on nvcc, import and execution errors. If any worker exits non-zero, the launcher terminates the rest (survivors would hang waiting on a peer). Exit code is the max of the children's. `Ctrl-C`/`SIGTERM` in the parent terminates all children.

**Output:** rank 0 stdout is forwarded live; other ranks' stdout goes to `cache/ddp_logs/rank{r}.log`; all stderr is forwarded. Done in the launcher because `compiler.py` has many bare `print` calls.

```python
# tensorlang/launcher.py  (no pycuda imports)
import os, sys, signal, socket, subprocess, time
from tensorlang.gpu import detect_gpus, pin_env

def _physical_gpus():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [int(x) for x in cvd.split(",") if x.strip()]
    return [g["index"] for g in detect_gpus()]

def _free_port(default=29500):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", default)); return default
        except OSError:
            s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

def launch_ddp(nproc, argv, same_gpu=False):
    gpus = _physical_gpus()
    if same_gpu:
        gpus = [gpus[0]] * nproc if gpus else []
    if len(gpus) < nproc:
        print(f"--nproc {nproc} but only {len(gpus)} GPU(s) available"); sys.exit(2)

    port = _free_port()
    os.makedirs("cache/ddp_logs", exist_ok=True)
    procs = []
    for r in range(nproc):
        env = pin_env(gpus[r])
        env.update(RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE=str(nproc),
                   MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        out = None if r == 0 else open(f"cache/ddp_logs/rank{r}.log", "w")
        procs.append(subprocess.Popen([sys.executable, *argv], env=env, stdout=out))

    def _kill_all(*_):
        for p in procs:
            if p.poll() is None: p.terminate()
    signal.signal(signal.SIGINT, _kill_all)
    signal.signal(signal.SIGTERM, _kill_all)

    while True:
        codes = [p.poll() for p in procs]
        if any(c not in (None, 0) for c in codes):
            _kill_all(); break
        if all(c == 0 for c in codes):
            break
        time.sleep(0.2)
    for p in procs: p.wait()
    sys.exit(max(p.returncode for p in procs))
```

`tensorlang.py` `main()`, before any GPU work:

```python
if getattr(args, "ddp", False) and "RANK" not in os.environ:
    from tensorlang.launcher import launch_ddp
    launch_ddp(args.nproc, sys.argv, same_gpu=getattr(args, "ddp_same_gpu", False))
# Workers: RANK is set and CUDA_VISIBLE_DEVICES is already pinned. No pin_current_process call.
```

`tensor_lang.py` `parse_arguments` needs `--ddp` (store_true), `--nproc` (int, default 2, requires `--ddp`) and `--ddp-same-gpu` (store_true, dev/CI only). `--ddp` combined with `--test`, `--app` or `--gpu` is a parser error in v1 (the launcher chooses each rank's GPU itself).

---

## 5. Language Surface

### 5.1 v1: no new syntax

A DDP program is ordinary TensorLang:

```tl
let w  = load("w0.npy") with grad      // same file on every rank, so identical init
let x  = load("x_{rank}.npy")          // per-rank data shard
let y  = load("y_{rank}.npy")
let lr: Tensor[f32, (1, 1)] = [[0.01]]

for step in range(100) {
    let pred = matmul(x, w)
    let loss = mse_loss(pred, y)
    backward(loss)                     // under --ddp: all w_grad averaged across ranks here
    let w_updated = minus(w, mult(lr, w_grad))
    w = w_updated
}
```

- **Opt-in** is the launcher flag. Without `--ddp`, `WORLD_SIZE` is unset (treated as 1) and no new code runs.
- **Path placeholders:** `{rank}` and `{world_size}` inside the `STRING` of `load(...)` and `save(...)` are substituted with the run's values. Resolve them in `ast_builder.py` when building those nodes: the `load_call` branch returns `{'type': 'load', 'file_path': ...}`, and both the type checker (shape inference from the file) and `compiler.py` (`_load_data_file`) read `expr['file_path']`, so one substitution there covers both. No grammar change.
- **No `rank()`, `local_rank()`, `world_size()` builtins in v1.** Shapes cannot contain expressions, and `if` is not usable for rank-gating (section 2).
- **No `ddp_init()` / `ddp wrap` / `fsdp wrap` constructs.** The environment plus launcher replaces `ddp_init`; backward replaces `wrap`.
- **Naming constraint still applies.** Tensor names used inside a `for` body cannot be reused at the top level (README, "Naming Constraints"); DDP programs follow the same rule.
- **Printed loss is per-rank.** Rank 0 reports the loss of its own shard only. A global loss needs the explicit `all_reduce(loss, mean)` from step 7; gradients are unaffected.
- **Equal shards and mean losses:** gradient averaging equals the single-process full-batch gradient only when every rank has the same shard size and the loss is mean-reduced (e.g. `mse_loss`). Document this.

### 5.2 Step 2: explicit collectives

Tested against the real grammar under LALR (`start='program'`), no conflicts:

```lark
// add to expr:  | all_reduce_call | broadcast_call | all_gather_call   (before user_function_call)
all_reduce_call:  "all_reduce" "(" NAME ["," reduce_op] ")"
reduce_op:        "sum" | "mean" | "max" | "min"
broadcast_call:   "broadcast" "(" NAME ["," "src" "=" NUMBER] ")"
all_gather_call:  "all_gather" "(" NAME ["," "axis" "=" NUMBER] ")"
```

- Args are `NAME` (like `sum_call`), reusing the existing `sum`/`mean`/`max`/`min` terminals.
- `all_reduce`, `broadcast`, `all_gather` become reserved words. **Grep `tests/` and `apps/` for tensors with these names first.**
- `src=` would reserve `src`. Positional `broadcast(w, 0)` avoids that and should parse equally, but only the `src=` form was tested.

Semantics:

| Op | Result shape | Notes |
|---|---|---|
| `all_reduce(t, op)` | same | Returns a fresh tensor; input unchanged. Default `sum`. `mean` = sum / `world_size`. |
| `broadcast(t, src)` | same | Rank `src`'s value on every rank. |
| `all_gather(t, axis)` | `S[axis] * world_size` | Concatenated in rank order. Equal shapes required in v1. |
| `reduce_scatter` | `S[axis] / world_size` | Reserved for FSDP. Requires divisibility (compile error otherwise). |

At `world_size == 1` every collective is the identity, so the same program runs unchanged on one GPU.

**Collective contract:** every rank must call the same collectives in the same order with the same dtype and shape. The host backend checks a hash of `(op, shape, dtype)` per call and fails loudly on mismatch, and every wait has a timeout with a clear error.

**Differentiability:** collectives are **not** differentiable in v1 (not recorded in the autograd graph; gradient sync happens on gradient tensors after the local backward). Rules needed later for FSDP or tensor parallelism: `all_reduce(sum)` backward is `all_reduce(sum)`; `all_gather` backward is `reduce_scatter(sum)`; `reduce_scatter(sum)` backward is `all_gather`; `broadcast(src)` backward is sum-reduce to `src` and zeros elsewhere.

---

## 6. Runtime Behaviour

### 6.1 Specialisation

`TensorCompiler.__init__` reads `WORLD_SIZE`, `RANK`, `LOCAL_RANK` once (defaults 1, 0, 0). `world_size` is passed to the type checker as a **keyword argument** (`world_size=1`), threaded through the recursive `for` call. Do not add a reserved key to `env`.

A literal in source (if ever added) is an assertion that must match the environment, never an override. The environment is the single source of truth.

### 6.2 Gradient averaging at `backward`

At both backward sites in `compiler.py` (for-loop and top-level), after `comp_graph.backward(loss_name)` and before the `memcpy_htod`:

1. Collect `NAME` for `NAME in comp_graph.requires_grad` that has an entry in `comp_graph.gradients`, **sorted by name** so every rank uses the same order.
2. Flatten to one float32 buffer, send one message per backward, receive the averaged buffer, unpack.
3. Continue with the existing upload.

Guarded by `world_size > 1`; otherwise the code path is unchanged.

### 6.3 Host backend (`tensorlang/dist.py`)

Star topology over `MASTER_ADDR`/`MASTER_PORT`: rank 0 listens, others connect; rank 0 sums (or reduces), then returns the result. Rank-0 gather and sum is O(world_size) traffic, which is fine for 2 to 4 GPUs on one node. API: `init()`, `all_reduce(np_array, op)`, `broadcast(np_array, src)`, `all_gather(np_array, axis)`, `barrier()`. It uses the Python standard library plus NumPy only.

Initialise after `pycuda.autoinit` has run, at the start of execution (not at the top of `compile_and_execute`, where no context exists yet).

### 6.4 Initial parameters

Identical init across ranks is the user's job in v1 (same file). Recommended addition: at the start of execution, broadcast rank 0's values of all `with grad` tensors, so ranks can never start out of sync.

### 6.5 Cache and artifacts (DDP runs only)

When `WORLD_SIZE > 1`, and only then:

- Kernel directory: `cache/<file>/ws{N}/`, so different world sizes never overwrite each other. Non-DDP runs keep `cache/<file>/` exactly as today.
- Rank 0 runs `nvcc` to a temp name and `os.replace`s it into place; other ranks wait on a file lock (`fcntl.flock` on `ws{N}/.lock`), then use the existing "kernel unchanged" check. A file lock is used because no collective backend is up yet.
- Generated code may depend on `world_size` (via `all_gather` shapes) but **never on `rank`**; rank stays a runtime value. This is what makes one artifact valid for every rank. Test it by diffing each rank's generated `kernel.cu`.
- Tensor outputs (`{name}.npy`, `.grad.npy`, `save` targets) go to `ws{N}/rank{r}/`, so ranks never overwrite each other's files.

### 6.6 NCCL (deferred)

NCCL is only worthwhile once backward runs on the GPU. At that point:
- pycuda has no NCCL binding; use `ctypes` against `libnccl` or `cupy.cuda.nccl`.
- NCCL ops run on their own streams; add event synchronisation so later kernels see results and the allocator does not recycle in-flight buffers.
- The 2x GTX 1080 Ti machine has no NVLink, so peer-to-peer goes over PCIe and may be disabled by the platform; NCCL then falls back to host staging. CUDA 13 may have dropped Pascal, so check the toolkit version before planning on NCCL here. The host backend avoids all of this for v1.

---

## 7. Implementation Order

| # | Step | Files | Risk to existing tests |
|---|---|---|---|
| 1 | Launcher + `--ddp`, `--nproc`, `--ddp-same-gpu` args | new `launcher.py`, `tensorlang.py`, `tensor_lang.py` | None (new flags only) |
| 2 | Read env in `TensorCompiler.__init__`; `world_size=1` kwarg in `type_checker` | `compiler.py`, `type_checker.py` | None (default 1) |
| 3 | `{rank}` / `{world_size}` path placeholders | `ast_builder.py` | None unless a path contains literal braces |
| 4 | Host backend | new `dist.py` | None |
| 5 | DDP cache layout (section 6.5) | `compiler.py` | None (gated on `WORLD_SIZE > 1`) |
| 6 | Gradient averaging at both backward sites; optional initial broadcast | `compiler.py` | None (gated) |
| 7 | Explicit collectives: grammar, `build_expression`, type-checker branches, host-op nodes in both execution paths | `.lark`, `ast_builder.py`, `type_checker.py`, `compiler.py` | Reserved words; grep for name clashes |
| 8 | NCCL backend (when backward is on GPU) | `dist.py` | None |
| 9 | FSDP | see section 8 | Large |

---

## 8. Path to FSDP

Collectives are the easy part. FSDP additionally needs per-layer parameter lifecycle (gather, use, free), flat parameter buffers and prefetching, which does touch the memory manager. It also needs `reduce_scatter` and collective gradient rules (section 5.2). Treat it as a separate design, not an extension of this one.

---

## 9. Testing

- **Dev/CI without two GPUs:** `--ddp --nproc 2 --ddp-same-gpu` runs both ranks on one GPU with the host backend. Collective logic and the DDP code paths are exercised on a single card.
- **Concrete equivalence fixture:** the README's `y = 2x` example (`x = [1,2,3,4]`, `lr = 0.1`, `w` starting at 0.5) has a published trajectory (`w` after each update: 2.75, 1.625, 2.1875, 1.90625, 2.046875). Split `x`/`y` into two shards of two rows (`x_0.npy`, `x_1.npy`, and so on) and run `--ddp --nproc 2`. The averaged gradient equals the full-batch gradient (first step: shard gradients -7.5 and -37.5 average to the full-batch -22.5), so `w` must follow the same table. Per-rank printed losses will differ (5.625 and 28.125 in step 1), which is expected.
- **Equivalence test (the key one):** train N ranks on equal shards, and compare final weights against a single-process run on the concatenated full batch. They should match to float tolerance, given equal shards and a mean loss.
- **Ordering and mismatch:** a test where ranks disagree on a collective's shape must fail fast with the hash-mismatch message, not hang.
- **Failure propagation:** a test where one rank exits non-zero must make the launcher terminate the others.
- **Separate category:** DDP tests run serially and are excluded from the parallel pool, since they occupy all GPUs. `pin_env()` should remove `RANK`, `LOCAL_RANK` and `WORLD_SIZE` so stray DDP variables never leak into ordinary tests.
- **Plain-Python unit tests:** put `dist.py` and `launcher.py` tests in `tests/python/` (the plain-`unittest` convention from HANDOVER.md section 18.1, run with `python3 -m unittest discover -s tests/python -v`). The host backend can be tested with subprocesses and no GPU; the launcher with a mocked `Popen`.
- **GPU-free shape checks:** `check.py` (parse + type-check) and `verify.py` (NumPy-interpret a `.tl` AST) already exist at the repo root. They were not reviewed for this design, but are the natural way to test `world_size`-dependent shape inference (`WORLD_SIZE=2 python check.py file.tl`) and to simulate N-rank averaging in NumPy for the equivalence test.
- **Regression:** the existing 109 tests, run without `--ddp`, must pass unchanged. Steps 1 to 4 should be verifiable by that alone.

---

## 10. Open Questions

1. Initial broadcast of `with grad` tensors (6.4): include in v1? Recommended yes.
2. `all_gather` with unequal shapes: require equal in v1 (recommended), or add a variable-size variant later?
3. Per-rank `save`: with placeholders available, is `save("out_{rank}.npy")` enough, or should unplaceholdered saves default to rank 0 only?
4. Timeout default for host-backend waits (suggest 120 s) and whether it is configurable.
5. Do the `if`/`for` upgrades planned separately change anything here? Rank-gated logic would become possible if `if` is implemented and a `rank` value is exposed.

---

**End of Multi-GPU Design Handover**
