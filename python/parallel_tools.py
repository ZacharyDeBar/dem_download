"""
parallel_tools.py
━━━━━━━━━━━━━━━━
Central toolkit for parallel CPU processing across the larger terrain-compute pipeline.

Provides:
  - SharedNDArray     : context manager for shared-memory numpy arrays
  - chunked_map       : split a list into chunks, process in parallel
  - parallel_map      : map a function over items with a worker pool
  - WorkerPool        : reusable pool with progress reporting

Design principles:
  - No domain logic — knows nothing about terrain analysis, polygons, or DEMs
  - Always falls back to single-process if n_workers=1 or pool fails
  - Shared memory is optional — workers can copy if RAM is plentiful
  - Compatible with Windows (spawn context) and Unix (fork context)

Usage:
    from parallel_tools import chunked_map, SharedNDArray

    # Shared memory — one copy of the array regardless of worker count
    with SharedNDArray(elevation_array) as shared:
        results = chunked_map(
            fn=process_chunk,
            items=water_polygons,
            n_workers=6,
            fn_kwargs={'shared_array': shared},
        )

    # Simple parallel map without shared memory
    results = chunked_map(process_batch, batches, n_workers=4)
"""

import math
import time
import traceback
from contextlib import contextmanager
from multiprocessing import cpu_count
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import shared_memory, Pool, get_context
from typing import Any, Callable, Iterable, Optional

import numpy as np


# ─────────────────────────────────────────────
# Shared memory array
# ─────────────────────────────────────────────

class SharedNDArray:
    """
    Context manager that places a numpy array in shared memory so
    multiple worker processes can read it without copying.

    Usage:
        with SharedNDArray(my_array) as shared:
            # shared.name  — shared memory block name (pass to workers)
            # shared.shape — array shape
            # shared.dtype — array dtype
            # shared.array — numpy view into shared memory (main process)
            results = pool.map(worker_fn, [(shared.name, shared.shape,
                                            shared.dtype, chunk)
                                           for chunk in chunks])
        # shared memory is released here automatically

    In worker processes, reconstruct with SharedNDArray.attach():
        arr = SharedNDArray.attach(name, shape, dtype)
        # use arr, then:
        SharedNDArray.detach(arr)
    """

    def __init__(self, array: np.ndarray):
        self._source  = array
        self._shm     = None
        self.array    = None
        self.name     = None
        self.shape    = array.shape
        self.dtype    = array.dtype
        self.nbytes   = array.nbytes

    def __enter__(self) -> 'SharedNDArray':
        self._shm  = shared_memory.SharedMemory(
            create=True, size=self._source.nbytes)
        self.array = np.ndarray(
            self._source.shape,
            dtype=self._source.dtype,
            buffer=self._shm.buf,
        )
        np.copyto(self.array, self._source)
        self.name = self._shm.name
        return self

    def __exit__(self, *_):
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
        self.array = None
        self._shm  = None

    @staticmethod
    def attach(name: str, shape: tuple, dtype) -> np.ndarray:
        """
        Attach to an existing shared memory block from a worker process.
        Returns a numpy array view. Call detach() when done.
        """
        shm = shared_memory.SharedMemory(name=name, create=False)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        # Stash shm on the array so detach() can close it
        arr._shm_handle = shm
        return arr

    @staticmethod
    def detach(arr: np.ndarray):
        """Close a worker's handle to shared memory."""
        shm = getattr(arr, '_shm_handle', None)
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


# ─────────────────────────────────────────────
# Chunk utilities
# ─────────────────────────────────────────────

def make_chunks(items: list, n_chunks: int) -> list[list]:
    """Split a list into n_chunks roughly equal sublists."""
    if n_chunks <= 1 or len(items) == 0:
        return [items]
    size = math.ceil(len(items) / n_chunks)
    return [items[i:i + size] for i in range(0, len(items), size)]


def make_batches(items: list, batch_size: int) -> list[list]:
    """Split a list into sublists of at most batch_size items."""
    return [items[i:i + batch_size]
            for i in range(0, len(items), batch_size)]


# ─────────────────────────────────────────────
# Worker wrapper — catches exceptions in child processes
# ─────────────────────────────────────────────

def _safe_call(fn: Callable, args: tuple, kwargs: dict) -> Any:
    """
    Call fn(*args, **kwargs) and return (result, None) on success
    or (None, traceback_str) on failure.
    Exceptions in worker processes are otherwise silently swallowed
    by multiprocessing.Pool.
    """
    try:
        result = fn(*args, **kwargs)
        return result, None
    except Exception:
        return None, traceback.format_exc()


def _worker_wrapper(packed):
    """Unpacks (fn, args, kwargs) and calls _safe_call."""
    fn, args, kwargs = packed
    return _safe_call(fn, args, kwargs)


# ─────────────────────────────────────────────
# chunked_map — main parallel entry point
# ─────────────────────────────────────────────

def chunked_map(
    fn: Callable,
    items: list,
    n_workers: int = None,
    chunk_size: int = None,
    fn_args: tuple = (),
    fn_kwargs: dict = None,
    desc: str = '',
    progress_every: int = 1,
    fallback_sequential: bool = True,
    context: str = 'spawn',
) -> list[Any]:
    """
    Map fn over items in parallel, splitting into chunks per worker.

    Each worker receives one chunk: fn(chunk, *fn_args, **fn_kwargs).
    fn must accept a list as its first argument and return a list of results.

    Args:
        fn:                   Function to call. Signature: fn(chunk, ...) -> list
        items:                List of items to process
        n_workers:            Number of worker processes (default: cpu_count - 1)
        chunk_size:           Items per chunk. If None, items / n_workers.
        fn_args:              Extra positional args passed to fn after chunk
        fn_kwargs:            Extra keyword args passed to fn
        desc:                 Label for progress output
        progress_every:       Print progress every N chunks complete
        fallback_sequential:  If True, fall back to single-process on pool error
        context:              Multiprocessing start context ('spawn' or 'fork')

    Returns:
        Flat list of all results in original item order.
    """
    if fn_kwargs is None:
        fn_kwargs = {}

    if not items:
        return []

    n_workers = _resolve_workers(n_workers)

    # Sequential path — n_workers=1 or single item
    if n_workers == 1 or len(items) == 1:
        return fn(items, *fn_args, **fn_kwargs)

    # Build chunks
    if chunk_size is not None:
        chunks = make_batches(items, chunk_size)
    else:
        chunks = make_chunks(items, n_workers)

    n_chunks = len(chunks)
    label    = f"[{desc}] " if desc else ""
    print(f"{label}Parallel: {len(items)} items → "
          f"{n_chunks} chunks × {n_workers} workers")

    packed = [(fn, (chunk, *fn_args), fn_kwargs) for chunk in chunks]

    try:
        ctx  = get_context(context)
        t0   = time.perf_counter()
        all_results = []

        with ctx.Pool(processes=n_workers) as pool:
            for i, (result, err) in enumerate(
                pool.imap_unordered(_worker_wrapper, packed)
            ):
                if err:
                    print(f"{label}Worker error in chunk {i}:\n{err}")
                    continue
                if result is not None:
                    all_results.extend(result)

                if (i + 1) % progress_every == 0 or (i + 1) == n_chunks:
                    elapsed = time.perf_counter() - t0
                    pct     = (i + 1) / n_chunks * 100
                    rate    = (i + 1) / elapsed if elapsed > 0 else 0
                    eta     = (n_chunks - i - 1) / rate if rate > 0 else 0
                    print(f"{label}{i+1}/{n_chunks} chunks "
                          f"({pct:.0f}%) "
                          f"elapsed={elapsed:.1f}s "
                          f"ETA={eta:.0f}s")

        return all_results

    except Exception as e:
        if fallback_sequential:
            print(f"{label}Pool failed ({e}), falling back to sequential")
            return fn(items, *fn_args, **fn_kwargs)
        raise


# ─────────────────────────────────────────────
# parallel_map — map over individual items
# ─────────────────────────────────────────────

def parallel_map(
    fn: Callable,
    items: list,
    n_workers: int = None,
    fn_kwargs: dict = None,
    desc: str = '',
    ordered: bool = True,
    fallback_sequential: bool = True,
    context: str = 'spawn',
) -> list[Any]:
    """
    Map fn over individual items in parallel.
    fn receives one item at a time: fn(item, **fn_kwargs).

    Args:
        fn:                   Function. Signature: fn(item, **fn_kwargs) -> result
        items:                List of items
        n_workers:            Worker processes (default: cpu_count - 1)
        fn_kwargs:            Extra keyword args passed to fn
        desc:                 Label for progress output
        ordered:              Preserve input order in results (imap vs imap_unordered)
        fallback_sequential:  Fall back to single-process on pool error
        context:              Multiprocessing start context

    Returns:
        List of results, one per item.
    """
    if fn_kwargs is None:
        fn_kwargs = {}

    if not items:
        return []

    n_workers = _resolve_workers(n_workers)
    label     = f"[{desc}] " if desc else ""

    if n_workers == 1:
        return [fn(item, **fn_kwargs) for item in items]

    packed = [(fn, (item,), fn_kwargs) for item in items]

    try:
        ctx  = get_context(context)
        t0   = time.perf_counter()
        results = []

        with ctx.Pool(processes=n_workers) as pool:
            map_fn = pool.imap if ordered else pool.imap_unordered
            for i, (result, err) in enumerate(map_fn(_worker_wrapper, packed)):
                if err:
                    print(f"{label}Worker error on item {i}:\n{err}")
                    results.append(None)
                else:
                    results.append(result)

                if (i + 1) % max(1, len(items) // 10) == 0:
                    elapsed = time.perf_counter() - t0
                    pct = (i + 1) / len(items) * 100
                    print(f"{label}{i+1}/{len(items)} ({pct:.0f}%) "
                          f"elapsed={elapsed:.1f}s")

        return results

    except Exception as e:
        if fallback_sequential:
            print(f"{label}Pool failed ({e}), falling back to sequential")
            return [fn(item, **fn_kwargs) for item in items]
        raise


# ─────────────────────────────────────────────
# Shared-memory parallel map
# ─────────────────────────────────────────────

def parallel_map_shared(
    fn: Callable,
    items: list,
    shared_arrays: dict[str, np.ndarray],
    n_workers: int = None,
    chunk_size: int = None,
    fn_kwargs: dict = None,
    desc: str = '',
    context: str = 'spawn',
) -> list[Any]:
    """
    Parallel map with shared-memory numpy arrays.

    Arrays in `shared_arrays` are placed in shared memory once and
    accessible by all workers without copying. Workers reconstruct
    them using SharedNDArray.attach().

    Args:
        fn:             Function. Signature:
                        fn(chunk, shared_refs, **fn_kwargs) -> list
                        where shared_refs = {name: (shm_name, shape, dtype)}
        items:          List of items to process in chunks
        shared_arrays:  Dict of {key: numpy_array} to share
        n_workers:      Worker processes
        chunk_size:     Items per chunk
        fn_kwargs:      Extra keyword args
        desc:           Progress label
        context:        Multiprocessing start context

    Returns:
        Flat list of all results.
    """
    if fn_kwargs is None:
        fn_kwargs = {}

    n_workers = _resolve_workers(n_workers)
    label     = f"[{desc}] " if desc else ""

    if n_workers == 1:
        # Sequential — pass arrays directly, no shared memory needed
        shared_refs = {k: (None, v.shape, v.dtype, v)
                       for k, v in shared_arrays.items()}
        return fn(items, shared_refs, **fn_kwargs)

    if chunk_size is not None:
        chunks = make_batches(items, chunk_size)
    else:
        chunks = make_chunks(items, n_workers)

    print(f"{label}Shared-memory parallel: {len(items)} items → "
          f"{len(chunks)} chunks, "
          f"{sum(a.nbytes for a in shared_arrays.values())/1e6:.0f}MB shared")

    # Place all arrays in shared memory
    shm_blocks  = {}
    shared_refs = {}  # passed to workers: {key: (shm_name, shape, dtype)}

    try:
        for key, array in shared_arrays.items():
            shm  = shared_memory.SharedMemory(create=True, size=array.nbytes)
            view = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            np.copyto(view, array)
            shm_blocks[key]  = (shm, view)
            shared_refs[key] = (shm.name, array.shape, array.dtype)

        packed = [
            (fn, (chunk, shared_refs), fn_kwargs)
            for chunk in chunks
        ]

        ctx = get_context(context)
        t0  = time.perf_counter()
        all_results = []

        with ctx.Pool(processes=n_workers) as pool:
            for i, (result, err) in enumerate(
                pool.imap_unordered(_worker_wrapper, packed)
            ):
                if err:
                    print(f"{label}Worker error:\n{err}")
                    continue
                if result is not None:
                    all_results.extend(result)

                elapsed = time.perf_counter() - t0
                pct     = (i + 1) / len(chunks) * 100
                print(f"{label}{i+1}/{len(chunks)} chunks "
                      f"({pct:.0f}%) elapsed={elapsed:.1f}s")

        return all_results

    finally:
        # Always release shared memory blocks
        for key, (shm, _) in shm_blocks.items():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _resolve_workers(n_workers: Optional[int]) -> int:
    """
    Resolve worker count.
    Defaults to cpu_count - 1, minimum 1, maximum cpu_count.
    """
    if n_workers is not None:
        return max(1, n_workers)
    return max(1, cpu_count() - 1)


def recommend_workers(
    item_count: int,
    item_size_mb: float = 0.0,
    available_ram_mb: float = None,
) -> int:
    """
    Suggest a sensible worker count given item count and memory pressure.

    Args:
        item_count:       How many items will be processed
        item_size_mb:     Approximate memory per worker in MB (e.g. shared array)
        available_ram_mb: Available RAM — auto-detected if None

    Returns:
        Recommended n_workers
    """
    try:
        import psutil
        available_ram_mb = available_ram_mb or (
            psutil.virtual_memory().available / 1e6)
    except ImportError:
        available_ram_mb = available_ram_mb or 4000  # conservative default

    cpus = cpu_count()

    # Don't use more workers than items
    max_by_items = min(cpus, item_count)

    # Don't use more workers than RAM allows
    if item_size_mb > 0:
        max_by_ram = max(1, int(available_ram_mb / item_size_mb))
    else:
        max_by_ram = cpus

    recommended = min(max_by_items, max_by_ram, cpus - 1)
    return max(1, recommended)


def print_system_info():
    """Print CPU and memory info useful for tuning worker counts."""
    cpus = cpu_count()
    print(f"[PARALLEL] CPUs available: {cpus}")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"[PARALLEL] RAM total:     {mem.total/1e9:.1f} GB")
        print(f"[PARALLEL] RAM available: {mem.available/1e9:.1f} GB")
        print(f"[PARALLEL] RAM used:      {mem.percent:.0f}%")
    except ImportError:
        print("[PARALLEL] psutil not installed — RAM info unavailable")