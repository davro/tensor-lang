# Data-parallel training

`--ddp` trains one model on several GPUs by giving each GPU its own shard of the data. Every worker runs the same program; after each `backward()` the workers average their gradients, so they all take the same optimizer step and stay identical.

```bash
python3 tensorlang.py --ddp --nproc 2 train.tl                    # one worker per GPU (default --nproc is 2)
python3 tensorlang.py --ddp --nproc 4 --ddp-same-gpu train.tl     # several workers on one GPU: development and tests
```

`--ddp` needs a `.tl` file and cannot be combined with `--test`, `--app`, `--list-apps`, `--list-gpus` or `--gpu` (the launcher chooses each worker's GPU).

## A program for it

The only change to a training program is where its data comes from. `{rank}` and `{world_size}` in a `load()` or `save()` path are replaced by the worker's values (and by `0` and `1` in an ordinary run, so the same program also runs on one process).

```text
let x: Tensor[f32, (32, 8)] = load("data/x_{rank}.npy")        # this worker's shard
let y_true: Tensor[f32, (32, 1)] = load("data/y_{rank}.npy")
let w1: Tensor[f32, (8, 16)] = load("w1.npy") with grad
let w2: Tensor[f32, (16, 1)] = load("w2.npy") with grad

for step in range(5) {
    let h = gelu(matmul(x, w1))
    let pred = matmul(h, w2)
    let loss = mse_loss(pred, y_true)
    backward(loss)
    sgd(w1, w1_grad, 0.1)
    sgd(w2, w2_grad, 0.1)
}

save(w1, "w1_final.npy")        # lands in cache/<program>/ws2/rank0/ and .../rank1/
```

`tools/ddp_check.py` generates exactly this kind of program, with its data, if you want a working example.

## What is synchronised

* **At start-up**, every `with grad` tensor that was loaded takes rank 0's value, before anything is uploaded to the GPU. Workers cannot begin apart; a shape that differs between ranks is an error.
* **After every `backward()`**, the gradients of the `with grad` tensors are averaged across workers, and the overflow flag becomes "any worker overflowed". The executor's existing logic (skip the optimizer step, halve the loss scale) then runs identically everywhere, so dynamic loss scaling stays in step too.

## What the program has to guarantee

* **Equal shards and a mean loss.** The average of the shard gradients is the gradient of the whole batch only if every shard has the same number of rows and the loss is a mean over them (`mse_loss`, `softmax_cross_entropy`, `mean`). Otherwise training still runs, but not on the gradient you meant.
* **The same program everywhere.** All workers run the same command. If data-dependent control flow makes one worker call `backward()` a different number of times, the run stops with a "ranks disagree" error instead of hanging.
* **Generated code must not depend on the rank.** Only file paths may differ; shapes and literals may not.

## What the launcher does

The parent process starts the workers and waits. If any worker fails, it stops the others and exits with that worker's code (128 plus the signal number for a crash by signal); Ctrl-C or `SIGTERM` stops them all. Rank 0 keeps the terminal; the other workers write their output to `cache/ddp_logs/rank{r}.log`.

| Variable | Value for each worker |
|---|---|
| `RANK`, `LOCAL_RANK` | `0 .. N-1` |
| `WORLD_SIZE` | `N` |
| `MASTER_ADDR` | `127.0.0.1` (a value in the parent's environment wins) |
| `MASTER_PORT` | 29500 if it is free, otherwise any free port (a value in the parent's environment wins) |
| `CUDA_VISIBLE_DEVICES` | this worker's GPU: the parent's `CUDA_VISIBLE_DEVICES` entries in order if set, otherwise the detected GPUs |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | CPU cores divided by `N`, at least 1, unless already set |
| `TENSORLANG_DDP_WORKER`, `TENSORLANG_DDP_RUN_ID` | set by the launcher: a marker, and one random id per launch |
| `TENSORLANG_DIST_TIMEOUT` | read, not set: seconds to wait to connect and for each peer (default 120) |

Asking for more workers than GPUs is an error before anything starts, unless `--ddp-same-gpu` is given.

**Why the thread cap.** With the NumPy backward, each process starts one BLAS thread per core. On 8 cores, two workers that each used all of them took 15.8 s per worker for a run that takes 4.7 s on one process; with the cap, 4.5 s per worker.

## Where the files go

An ordinary run keeps everything in `cache/<program>/`. A `--ddp` run splits it:

```text
cache/<program>/ws{N}/             kernel.cu, kernel.so, .lock, .built_by: one kernel for every worker
cache/<program>/ws{N}/rank{r}/     this worker's tensors, save() targets, .grad.npy files and layer caches
```

The kernel is built once: every worker takes the same file lock, the first one in compiles, and the others reuse the result. The library is moved into place before its source, so a half-finished build is never mistaken for a good one, and a worker that generates different code than the one that built it stops with an error.

## Speed

Two GTX 1080 Ti, the 4-layer micro-GPT of `docs/training.md`, 20 Adam steps:

| Run | Per worker |
|---|---|
| one process, GPU backward | about 1.8 s |
| two workers, GPU backward (measured before gradient exchange was added) | 1.84 s and 2.03 s |
| two independent processes, one per card | 1.99 s and 2.05 s |

So two cards do about 1.7 to 1.8 times the work of one in the same time, before the cost of exchanging gradients. That cost has not been measured at scale; for the 867K-parameter micro-GPT I would estimate 5 to 10% per step. The exchange goes through host memory over TCP on one machine and rank 0 does the summing.

## Checking that it works

```bash
python3 tools/ddp_check.py                       # 2 ranks, NumPy backward then GPU backward
python3 tools/ddp_check.py --ranks 4 --same-gpu  # four workers on one card
```

It trains a small network on the whole batch and on shards and compares the final weights. On two GTX 1080 Ti the workers ended with bit-identical weights and differed from one process by 5.4e-08 (NumPy backward) and 5.6e-08 (GPU backward); four workers on one card gave 4.4e-08 and 5.7e-08. This has been checked for SGD on a two-layer GELU network. It has not yet been run with Adam, `layer_norm`, `softmax` or `embedding`. The unit tests (`python3 -m unittest discover -s tests/python`) check the same equivalence on the CPU with the real autograd.

## Messages

| Message | Meaning |
|---|---|
| `only N of M ranks connected within T s; missing rank(s) [...]` | A worker died or took longer than the timeout to get ready (a cold compile counts). Look at `cache/ddp_logs/rank{r}.log`, or raise `TENSORLANG_DIST_TIMEOUT`. |
| `ranks disagree on collective #K: rank 0 called all_reduce(float32 [..]) but rank 1 called ...` | The workers do not have the same parameters or shapes, or ran a different number of backward passes. |
| `lost the connection to rank r during ...` / `rank r did not reach ... within T s` | That worker exited, crashed or is stuck. Its log says why. |
| `ranks generated different kernel code during the same run` | Something other than a file path differs between workers. |
| `rank 0 cannot listen on host:port ...` | Set `MASTER_PORT` to a free port. |
| `--nproc N but only M GPU(s) available` | Use fewer workers, set `CUDA_VISIBLE_DEVICES`, or add `--ddp-same-gpu`. |
| `CUDA_VISIBLE_DEVICES contains entries that are not GPU indices` | GPU UUIDs are not supported here; use numeric indices. |
| `invalid distributed environment: RANK=3, WORLD_SIZE=1` | A stale `RANK` in your shell. Run `unset RANK LOCAL_RANK WORLD_SIZE`. |

A data-parallel failure is reported as `[COMPILER] Data-parallel error on rank N: ...`.

## Limits

* One machine: the launcher starts local processes and the workers meet on `MASTER_ADDR` (127.0.0.1 by default).
* Gradients travel through host memory over TCP, not NCCL. NCCL becomes sensible once gradients stay on the device (the Stage 2 item in Future Work).
* POSIX only (the shared kernel build uses file locking), and tested on Linux.
* No explicit collectives in the language yet: averaging happens automatically after `backward()`.
* Verified on a small network with SGD; transformer-scale checks and a speed measurement that includes the exchange are still to do.

The design and the implementation status are in `docs/handovers/HANDOVER_MULTI_GPU.md`.
