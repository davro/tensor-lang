"""Host-side collectives for data-parallel runs: a star over TCP, standard library plus NumPy only.

Rank 0 listens on MASTER_ADDR:MASTER_PORT and the other ranks connect to it. A collective is one request from
every rank to rank 0, which checks that all ranks asked for the same thing, computes the result in rank order (so
it is deterministic and identical on every rank) and sends it back. A mismatch, a dead peer or a timeout raises
DistError on every rank that is still alive, instead of leaving them waiting.

    comm = dist.init()                      # reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT; world size 1 needs no sockets
    total = comm.all_reduce(array, "sum")   # "sum", "mean", "max" or "min"
    value = comm.broadcast(array, src=0)    # every rank passes an array of the same shape and dtype
    parts = comm.all_gather(array, axis=0)  # concatenated in rank order; shapes may differ along `axis` only
    comm.barrier()

This module imports no CUDA code, so a launcher process can use it without creating a GPU context.

Environment: RANK, WORLD_SIZE, LOCAL_RANK (defaults 0, 1, 0), MASTER_ADDR (127.0.0.1), MASTER_PORT (29500) and
TENSORLANG_DIST_TIMEOUT (seconds, default 120) for connecting and for each wait on a peer.
"""
import json
import os
import socket
import struct
import time

import numpy as np

DEFAULT_PORT = 29500
DEFAULT_TIMEOUT = 120.0

_MAGIC = b"TLD1"
_HEAD = struct.Struct("!4sIQ")           # magic, length of the JSON metadata, length of the payload
_REDUCE_OPS = ("sum", "mean", "max", "min")


class DistError(RuntimeError):
    """A collective failed: ranks disagree, a peer died or went silent, or a wait timed out."""


# ----------------------------------------------------------------------------------------------------------------
# Environment and path placeholders
# ----------------------------------------------------------------------------------------------------------------

def env_info(environ=None):
    """Return (rank, world_size, local_rank) from the environment, with the single-process defaults."""
    e = os.environ if environ is None else environ
    try:
        rank = int(e.get("RANK", 0))
        world = int(e.get("WORLD_SIZE", 1))
        local = int(e.get("LOCAL_RANK", 0))
    except ValueError as exc:
        raise ValueError(f"RANK, WORLD_SIZE and LOCAL_RANK must be integers: {exc}") from None
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"invalid distributed environment: RANK={rank}, WORLD_SIZE={world}")
    return rank, world, local


def expand_placeholders(path, environ=None):
    """Replace {rank} and {world_size} in a file path from the environment (0 and 1 when not distributed).

    Only those two exact tokens are replaced; any other braces are left alone."""
    if "{" not in path:
        return path
    rank, world, _ = env_info(environ)
    return path.replace("{rank}", str(rank)).replace("{world_size}", str(world))


# ----------------------------------------------------------------------------------------------------------------
# Wire format: [magic, metadata length, payload length] + JSON metadata + raw payload bytes
# ----------------------------------------------------------------------------------------------------------------

def _send(sock, meta, payload=None):
    body = json.dumps(meta, separators=(",", ":")).encode()
    size = 0 if payload is None else len(payload)
    sock.sendall(_HEAD.pack(_MAGIC, len(body), size) + body)
    if size:
        sock.sendall(payload)


def _recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("connection closed")
        got += k
    return buf


def _recv(sock):
    magic, meta_len, size = _HEAD.unpack(_recv_exact(sock, _HEAD.size))
    if magic != _MAGIC:
        raise DistError("received data that is not a tensorlang collective message (is something else using this port?)")
    meta = json.loads(bytes(_recv_exact(sock, meta_len)))
    payload = _recv_exact(sock, size) if size else bytearray()
    return meta, payload


def _contiguous(arr):
    """A C-contiguous array with the same shape (np.ascontiguousarray would turn a 0-d array into a 1-d one)."""
    arr = np.asarray(arr)
    return arr if arr.flags.c_contiguous else arr.copy(order="C")


def _payload(arr):
    return memoryview(arr.reshape(-1)).cast("B") if arr.size else None


def _to_array(payload, dtype, shape):
    return np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape)


def _describe(meta):
    op = meta.get("op", "?")
    parts = []
    if "dtype" in meta:
        parts.append(f"{np.dtype(meta['dtype']).name} {meta['shape']}")
    for key in ("red", "src", "axis"):
        if key in meta:
            parts.append(f"{key}={meta[key]}")
    return f"{op}({', '.join(parts)})"


def _agree(op, a, b):
    """Do two requests describe the same collective? all_gather may differ along its own axis only."""
    if a["op"] != b["op"] or a["seq"] != b["seq"]:
        return False
    if a.get("dtype") != b.get("dtype") or a.get("red") != b.get("red") or a.get("src") != b.get("src"):
        return False
    if op == "all_gather":
        if a["axis"] != b["axis"] or len(a["shape"]) != len(b["shape"]):
            return False
        ax = a["axis"]
        return all(x == y for i, (x, y) in enumerate(zip(a["shape"], b["shape"])) if i != ax)
    return a.get("shape") == b.get("shape")


# ----------------------------------------------------------------------------------------------------------------
# Communicators
# ----------------------------------------------------------------------------------------------------------------

def _check_args(op, arr, red=None, src=None, axis=None, world_size=1):
    arr = _contiguous(arr)
    if arr.dtype.hasobject:
        raise ValueError("collectives need a numeric array, not an object array")
    if red is not None:
        if red not in _REDUCE_OPS:
            raise ValueError(f"unknown reduction {red!r}; use one of {_REDUCE_OPS}")
        if red == "mean" and arr.dtype.kind != "f":
            raise ValueError("a mean needs a floating-point array")
    if src is not None and not 0 <= src < world_size:
        raise ValueError(f"src {src} is not a rank in 0..{world_size - 1}")
    if axis is not None and not -max(arr.ndim, 1) <= axis < max(arr.ndim, 1):
        raise ValueError(f"axis {axis} is out of range for an array with {arr.ndim} dimension(s)")
    return arr


def _reduce(arrays, red):
    acc = arrays[0].copy()
    for a in arrays[1:]:
        if red in ("sum", "mean"):
            acc = np.add(acc, a)
        elif red == "max":
            acc = np.maximum(acc, a)
        else:
            acc = np.minimum(acc, a)
    if red == "mean":
        acc = (acc / len(arrays)).astype(acc.dtype)
    return acc


class LocalComm:
    """The single-process communicator: every collective is the identity, so one code path serves both cases."""

    rank = 0
    world_size = 1
    local_rank = 0

    def all_reduce(self, arr, op="sum"):
        return _check_args("all_reduce", arr, red=op).copy()

    def broadcast(self, arr, src=0):
        return _check_args("broadcast", arr, src=src, world_size=1).copy()

    def all_gather(self, arr, axis=0):
        return _check_args("all_gather", arr, axis=axis).copy()

    def barrier(self):
        pass

    def close(self):
        pass


class HostComm:
    """Communicator for world_size > 1. Rank 0 holds one socket per peer; every other rank holds one to rank 0."""

    def __init__(self, rank, world_size, local_rank=0, peers=None, sock=None, timeout=DEFAULT_TIMEOUT):
        self.rank, self.world_size, self.local_rank = rank, world_size, local_rank
        self.timeout = timeout
        self._peers = peers or {}
        self._sock = sock
        self._seq = 0
        self._open = True

    # -- public API ------------------------------------------------------------------------------------------

    def all_reduce(self, arr, op="sum"):
        arr = _check_args("all_reduce", arr, red=op)
        return self._collective("all_reduce", arr, red=op)

    def broadcast(self, arr, src=0):
        arr = _check_args("broadcast", arr, src=src, world_size=self.world_size)
        return self._collective("broadcast", arr, src=src)

    def all_gather(self, arr, axis=0):
        arr = _check_args("all_gather", arr, axis=axis)
        axis = axis % max(arr.ndim, 1)
        return self._collective("all_gather", arr, axis=axis)

    def barrier(self):
        self._collective("barrier", None)

    def close(self):
        self._open = False
        for s in [self._sock, *self._peers.values()]:
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

    # -- machinery -------------------------------------------------------------------------------------------

    def _collective(self, op, arr, **params):
        if not self._open:
            raise DistError("this communicator was closed after an earlier failure; the run cannot continue")
        self._seq += 1
        seq = self._seq
        meta = {"op": op, "seq": seq, "rank": self.rank, **params}
        if arr is not None:
            meta.update(dtype=arr.dtype.str, shape=list(arr.shape))
        try:
            if self.rank == 0:
                return self._coordinate(op, seq, meta, arr)
            return self._participate(op, seq, meta, arr)
        except DistError as exc:
            self._abort(str(exc) if self.rank == 0 else None)
            raise
        except (socket.timeout, TimeoutError):
            self._abort(None)
            raise DistError(f"timed out after {self.timeout:.0f} s in {_describe(meta)} #{seq} "
                            f"(rank {self.rank} is waiting on rank 0)") from None
        except (OSError, ConnectionError):
            self._abort(None)
            raise DistError(f"lost the connection to rank 0 during {_describe(meta)} #{seq} (did it exit?)") from None

    def _abort(self, notify):
        """Close everything; rank 0 first tells the live peers why, so they fail with a message instead of a reset."""
        if notify is not None:
            for s in self._peers.values():
                try:
                    s.settimeout(2.0)
                    _send(s, {"error": notify})
                    s.shutdown(socket.SHUT_WR)
                    s.settimeout(0.5)
                    while s.recv(65536):          # read what the peer already sent: closing with unread data resets the
                        pass                      # connection and the peer would lose the message above
                except OSError:
                    pass
        self.close()

    def _participate(self, op, seq, meta, arr):
        _send(self._sock, meta, None if arr is None else _payload(arr))
        reply, payload = _recv(self._sock)
        if "error" in reply:
            raise DistError(reply["error"])
        if reply.get("seq") != seq:
            raise DistError(f"out of step with rank 0: expected reply #{seq}, got #{reply.get('seq')}")
        if "dtype" not in reply:
            return None
        return _to_array(payload, reply["dtype"], reply["shape"]).copy()

    def _coordinate(self, op, seq, meta, arr):
        requests = {0: (meta, arr)}
        for r in range(1, self.world_size):
            try:
                m, payload = _recv(self._peers[r])
            except (socket.timeout, TimeoutError):
                raise DistError(f"rank {r} did not reach {_describe(meta)} #{seq} within {self.timeout:.0f} s "
                                f"(is it stuck, or running a different program?)") from None
            except (OSError, ConnectionError):
                raise DistError(f"lost the connection to rank {r} during {_describe(meta)} #{seq} "
                                f"(did it exit?)") from None
            peer_arr = _to_array(payload, m["dtype"], m["shape"]) if "dtype" in m else None
            requests[r] = (m, peer_arr)

        for r in range(1, self.world_size):
            if not _agree(op, meta, requests[r][0]):
                raise DistError(f"ranks disagree on collective #{seq}: rank 0 called {_describe(meta)} "
                                f"but rank {r} called {_describe(requests[r][0])}")

        arrays = [requests[r][1] for r in range(self.world_size)]
        if op == "all_reduce":
            result = _reduce(arrays, meta["red"])
        elif op == "broadcast":
            result = arrays[meta["src"]].copy()
        elif op == "all_gather":
            result = np.concatenate(arrays, axis=meta["axis"]) if arrays[0].ndim else np.stack(arrays)
        else:
            result = None

        reply = {"op": op, "seq": seq}
        data = None
        if result is not None:
            result = _contiguous(result)
            reply.update(dtype=result.dtype.str, shape=list(result.shape))
            data = _payload(result)
        for r in range(1, self.world_size):
            try:
                _send(self._peers[r], reply, data)
            except OSError:
                raise DistError(f"lost the connection to rank {r} while sending the result of "
                                f"{_describe(meta)} #{seq}") from None
        return result


# ----------------------------------------------------------------------------------------------------------------
# Start-up
# ----------------------------------------------------------------------------------------------------------------

def _timeout(environ):
    return float(environ.get("TENSORLANG_DIST_TIMEOUT", DEFAULT_TIMEOUT))


def init(environ=None, timeout=None):
    """Create the communicator for this process from the environment (see the module docstring)."""
    e = os.environ if environ is None else environ
    rank, world, local = env_info(e)
    if world == 1:
        return LocalComm()
    timeout = _timeout(e) if timeout is None else timeout
    addr = e.get("MASTER_ADDR", "127.0.0.1")
    port = int(e.get("MASTER_PORT", DEFAULT_PORT))
    if rank == 0:
        return _init_root(world, local, addr, port, timeout)
    return _init_peer(rank, world, local, addr, port, timeout)


def _init_root(world, local, addr, port, timeout):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((addr, port))
    except OSError as exc:
        srv.close()
        raise DistError(f"rank 0 cannot listen on {addr}:{port} ({exc}); set MASTER_PORT to a free port") from None
    srv.listen(world)
    peers = {}
    deadline = time.monotonic() + timeout
    try:
        while len(peers) < world - 1:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(1, world)) - set(peers))
                raise DistError(f"only {len(peers) + 1} of {world} ranks connected within {timeout:.0f} s; "
                                f"missing rank(s) {missing}")
            srv.settimeout(remaining)
            conn, _ = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(timeout)
            hello, _ = _recv(conn)
            r = hello.get("hello")
            if hello.get("world") != world or not isinstance(r, int) or not 1 <= r < world or r in peers:
                conn.close()
                raise DistError(f"rejected a connection claiming rank {r} of {hello.get('world')} "
                                f"(this run has {world} ranks)")
            peers[r] = conn
        for conn in peers.values():
            _send(conn, {"ready": True})
    except (socket.timeout, TimeoutError):
        missing = sorted(set(range(1, world)) - set(peers))
        for c in peers.values():
            c.close()
        raise DistError(f"only {len(peers) + 1} of {world} ranks connected within {timeout:.0f} s; "
                        f"missing rank(s) {missing}") from None
    except Exception:
        for c in peers.values():
            c.close()
        raise
    finally:
        srv.close()
    return HostComm(0, world, local, peers=peers, timeout=timeout)


def _init_peer(rank, world, local, addr, port, timeout):
    deadline = time.monotonic() + timeout
    last = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DistError(f"rank {rank} could not connect to rank 0 at {addr}:{port} within {timeout:.0f} s"
                            f"{f' ({last})' if last else ''}")
        try:
            sock = socket.create_connection((addr, port), timeout=min(5.0, remaining))
            break
        except OSError as exc:
            last = exc
            time.sleep(0.1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(max(deadline - time.monotonic(), 1.0))
    try:
        _send(sock, {"hello": rank, "world": world})
        reply, _ = _recv(sock)
    except (OSError, ConnectionError):
        sock.close()
        raise DistError(f"rank {rank}: rank 0 closed the connection while ranks were joining") from None
    if "error" in reply:
        sock.close()
        raise DistError(reply["error"])
    sock.settimeout(timeout)
    return HostComm(rank, world, local, sock=sock, timeout=timeout)


# ----------------------------------------------------------------------------------------------------------------
# Process-wide communicator
# ----------------------------------------------------------------------------------------------------------------

_comm = None


def get():
    """The process-wide communicator, created from the environment on first use."""
    global _comm
    if _comm is None:
        _comm = init()
    return _comm


def shutdown():
    global _comm
    if _comm is not None:
        _comm.close()
        _comm = None


# ----------------------------------------------------------------------------------------------------------------
# Gradient averaging (data-parallel step)
# ----------------------------------------------------------------------------------------------------------------

def average_gradients(comm, grads, overflow=False):
    """Average gradients over all ranks and agree on the overflow flag.

    `grads` maps parameter name to array; every rank must pass the same names (they are sorted so the order agrees).
    The gradients are packed into one buffer with this rank's overflow flag as its last element, summed in one
    all_reduce, divided by the world size and unpacked with their original shapes and dtypes.

    Returns (averaged, any_overflow): a new dict, and True if any rank reported overflow, so that every rank takes
    the same decision about skipping the optimizer step.
    """
    names = sorted(grads)
    arrays = [np.asarray(grads[n]) for n in names]
    wide = np.float64 if any(a.dtype == np.float64 for a in arrays) else np.float32
    parts = [a.astype(wide, copy=False).ravel() for a in arrays]
    parts.append(np.array([1.0 if overflow else 0.0], dtype=wide))
    summed = comm.all_reduce(np.concatenate(parts), "sum")
    any_overflow = bool(summed[-1] > 0)
    mean = summed[:-1] / comm.world_size
    out, pos = {}, 0
    for n, a in zip(names, arrays):
        out[n] = mean[pos:pos + a.size].reshape(a.shape).astype(a.dtype)
        pos += a.size
    return out, any_overflow


def sync_gradients(comm, graph, reached):
    """The data-parallel step after a backward pass: average the gradients the executor is about to use across all
    ranks, and make the graph's overflow flag the same on every rank.

    `graph` is the autograd graph (it needs `requires_grad`, `gradients`, `keeps_previous_gradient(name, reached)` and
    `overflow_detected`). The gradients exchanged are the ones the executor will write: those of `requires_grad`
    tensors that have a gradient and are not kept from an earlier pass. Every rank must reach this call the same number
    of times with the same set, which holds when all ranks run the same program; a rank that does not fails with a
    DistError naming the mismatch. Averaging happens before the loss scale is divided out, which is fine because the
    scale evolves only from the shared flag and so is identical on every rank.

    Returns the sorted names that were exchanged."""
    names = sorted(n for n in graph.requires_grad
                   if n in graph.gradients and not graph.keeps_previous_gradient(n, reached))
    averaged, any_overflow = average_gradients(
        comm, {n: graph.gradients[n] for n in names}, overflow=graph.overflow_detected)
    graph.gradients.update(averaged)
    graph.overflow_detected = any_overflow
    return names


def broadcast_parameters(comm, tensors, names):
    """Replace this rank's value of each named tensor by rank 0's, so ranks cannot start out of step.

    Names missing from `tensors` are skipped; the order is sorted so every rank sends the same sequence of messages."""
    for name in sorted(names):
        if name in tensors:
            tensors[name] = comm.broadcast(np.asarray(tensors[name]), src=0)
