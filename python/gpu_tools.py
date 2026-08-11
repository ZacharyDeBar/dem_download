"""
gpu_tools.py
━━━━━━━━━━━━
Central GPU acceleration toolkit for the larger terrain-compute pipeline.

Uses CuPy as the GPU backend — a drop-in NumPy replacement that
runs on CUDA-capable GPUs. Falls back to NumPy/SciPy transparently
when CuPy is unavailable or the GPU has insufficient memory.

Provides:
  - GPU_AVAILABLE       : bool — True if CuPy + CUDA are usable
  - gpu_info()          : print GPU device info
  - to_device()         : move numpy array to GPU (or return as-is)
  - from_device()       : move array back to CPU numpy
  - gpu_fallback()      : decorator — try GPU fn, fall back to CPU fn
  - GpuArray            : context manager for GPU array lifetime
  - binary_dilation_gpu : GPU-accelerated binary dilation
  - percentile_gpu      : GPU-accelerated percentile over masked array
  - process_label_array_gpu : GPU version of the DEM water correction
                              label processing loop

Design principles:
  - Zero-cost when GPU unavailable — imports are lazy, no errors on CPU-only
  - Caller never needs to know if GPU ran — same return types always
  - No domain logic — pure array operations only
  - Safe for use inside multiprocessing workers (init GPU per-process)

Usage:
    from gpu_tools import GPU_AVAILABLE, process_label_array_gpu

    corrected, changes, n_ok, n_skip = process_label_array_gpu(
        label_array, corrected, batch,
        buffer_pixels=3, shore_percentile=10.0, min_water_pixels=5,
    )
    # Works identically whether GPU is present or not
"""

import functools
import time
from typing import Callable, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────
# GPU detection — done once at import time
# ─────────────────────────────────────────────

def _detect_gpu() -> tuple[bool, str]:
    """
    Try to import CuPy and verify a CUDA device is available.
    Returns (available: bool, status_message: str).
    """
    try:
        import cupy as cp
        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices == 0:
            return False, "CuPy installed but no CUDA devices found"

        # Disable pinned memory — prevents OOM on large host arrays
        # Pinned memory requires contiguous physical RAM pages which
        # may not be available when the DEM array is already in memory
        cp.cuda.set_pinned_memory_allocator(None)

        # Quick smoke test — allocate a tiny array
        _ = cp.zeros(4, dtype=cp.float32)
        device = cp.cuda.Device(0)
        props  = cp.cuda.runtime.getDeviceProperties(device.id)
        name   = props['name'].decode() if isinstance(
            props['name'], bytes) else props['name']
        mem_gb = props['totalGlobalMem'] / 1e9
        return True, f"{name} ({mem_gb:.1f}GB VRAM)"
    except ImportError:
        return False, "CuPy not installed (pip install cupy-cuda12x)"
    except Exception as e:
        return False, f"CUDA error: {e}"


GPU_AVAILABLE, _GPU_STATUS = _detect_gpu()

if GPU_AVAILABLE:
    try:
        import cupy as cp
        # Disable pinned memory pool — prevents OOM on large arrays
        # Slightly slower transfers but works with limited host RAM
        cp.cuda.set_pinned_memory_allocator(None)
    except Exception:
        pass


# ─────────────────────────────────────────────
# Info
# ─────────────────────────────────────────────

def gpu_info() -> dict:
    """
    Return GPU availability info. Always safe to call.
    """
    info = {
        'gpu_available': GPU_AVAILABLE,
        'status':        _GPU_STATUS,
        'devices':       [],
    }

    if not GPU_AVAILABLE:
        print(f"[GPU] Not available: {_GPU_STATUS}")
        return info

    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name  = props['name']
            if isinstance(name, bytes):
                name = name.decode()
            free, total = cp.cuda.runtime.memGetInfo()
            info['devices'].append({
                'id':        i,
                'name':      name,
                'vram_gb':   round(props['totalGlobalMem'] / 1e9, 1),
                'free_gb':   round(free / 1e9, 1),
                'compute':   f"{props['major']}.{props['minor']}",
            })
            print(f"[GPU] Device {i}: {name} "
                  f"({props['totalGlobalMem']/1e9:.1f}GB total, "
                  f"{free/1e9:.1f}GB free)")
    except Exception as e:
        print(f"[GPU] Info error: {e}")

    return info


def gpu_memory_available_gb() -> float:
    """Return free GPU memory in GB, or 0.0 if GPU unavailable."""
    if not GPU_AVAILABLE:
        return 0.0
    try:
        import cupy as cp
        free, _ = cp.cuda.runtime.memGetInfo()
        return free / 1e9
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# Array transfer
# ─────────────────────────────────────────────

def to_device(array: np.ndarray, dtype=None):
    """
    Move a numpy array to GPU. Returns CuPy array if GPU available,
    otherwise returns the numpy array unchanged.

    Args:
        array: Input numpy array
        dtype: Optional dtype to cast to on transfer

    Returns:
        cupy.ndarray if GPU available, else numpy.ndarray
    """
    if not GPU_AVAILABLE:
        return array if dtype is None else array.astype(dtype)
    try:
        import cupy as cp
        arr = cp.asarray(array)
        return arr.astype(dtype) if dtype is not None else arr
    except Exception:
        return array if dtype is None else array.astype(dtype)


def from_device(array) -> np.ndarray:
    """
    Move an array from GPU to CPU numpy. Safe to call on numpy arrays too.

    Returns:
        numpy.ndarray always
    """
    if not GPU_AVAILABLE:
        return np.asarray(array)
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    except Exception:
        return np.asarray(array)


def is_gpu_array(array) -> bool:
    """Return True if array is a CuPy GPU array."""
    if not GPU_AVAILABLE:
        return False
    try:
        import cupy as cp
        return isinstance(array, cp.ndarray)
    except Exception:
        return False


# ─────────────────────────────────────────────
# Context manager for GPU array lifetime
# ─────────────────────────────────────────────

class GpuArray:
    """
    Context manager that transfers an array to GPU on enter
    and frees GPU memory on exit.

    Usage:
        with GpuArray(elevation_np) as elev_gpu:
            result_gpu = process(elev_gpu)
            result_np  = from_device(result_gpu)
        # GPU memory freed here
    """

    def __init__(self, array: np.ndarray, dtype=None):
        self._source = array
        self._dtype  = dtype
        self.array   = None

    def __enter__(self):
        self.array = to_device(self._source, self._dtype)
        return self.array

    def __exit__(self, *_):
        if GPU_AVAILABLE and is_gpu_array(self.array):
            try:
                del self.array
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        self.array = None


# ─────────────────────────────────────────────
# gpu_fallback decorator
# ─────────────────────────────────────────────

def gpu_fallback(cpu_fn: Callable):
    """
    Decorator factory. Wraps a GPU function so that if it fails
    (OOM, CuPy missing, any CUDA error), the CPU fallback is called
    with the same arguments.

    Usage:
        @gpu_fallback(cpu_binary_dilation)
        def binary_dilation_gpu(mask, iterations=1):
            import cupy as cp
            from cupyx.scipy.ndimage import binary_dilation as cp_dil
            return cp_dil(mask, iterations=iterations)

    The decorated function always returns a numpy array.
    """
    def decorator(gpu_fn: Callable) -> Callable:
        @functools.wraps(gpu_fn)
        def wrapper(*args, **kwargs):
            if not GPU_AVAILABLE:
                return cpu_fn(*args, **kwargs)
            try:
                result = gpu_fn(*args, **kwargs)
                return from_device(result)
            except Exception as e:
                # OOM or other CUDA error — fall back silently
                _maybe_free_gpu()
                return cpu_fn(*args, **kwargs)
        return wrapper
    return decorator


def _maybe_free_gpu():
    """Try to free GPU memory pool after an error."""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


# ─────────────────────────────────────────────
# GPU-accelerated array operations
# ─────────────────────────────────────────────

def _cpu_binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=iterations)


@gpu_fallback(_cpu_binary_dilation)
def binary_dilation_gpu(mask, iterations: int = 1):
    """
    GPU-accelerated binary dilation. Falls back to scipy on CPU.

    Args:
        mask:       Boolean array (numpy or cupy)
        iterations: Dilation radius in pixels

    Returns:
        numpy.ndarray (always, regardless of input type)
    """
    import cupy as cp
    from cupyx.scipy.ndimage import binary_dilation as cp_dilation

    mask_gpu = cp.asarray(mask, dtype=cp.bool_)
    result   = cp_dilation(mask_gpu, iterations=iterations)
    return result  # from_device called by decorator


def _cpu_uniform_filter(array: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    return uniform_filter(array, size=size)


@gpu_fallback(_cpu_uniform_filter)
def uniform_filter_gpu(array, size: int):
    """
    GPU-accelerated sliding-window mean (scipy.ndimage.uniform_filter
    equivalent). Falls back to scipy on CPU.

    Note: GPU and CPU floating-point summation order differ, so results
    are numerically close (~1e-11 max abs diff observed) but not always
    bit-identical at the float64 level -- verified on real DEM data that
    this never changes the flat-zone threshold classification downstream
    (0 mismatched pixels across 16M in a real-strip test), since real
    terrain variance essentially never lands within 1e-11 of the
    threshold. If this is ever reused somewhere threshold-sensitive at
    finer tolerance, re-verify for that use case.

    Args:
        array: 2D array (numpy or cupy), any float dtype
        size:  Uniform filter window size

    Returns:
        numpy.ndarray (always, regardless of input type)
    """
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter as cp_uniform_filter

    arr_gpu = cp.asarray(array)
    return cp_uniform_filter(arr_gpu, size=size)


def _cpu_binary_erosion(mask: np.ndarray, structure=None,
                         border_value: int = 0) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, structure=structure, border_value=border_value)


@gpu_fallback(_cpu_binary_erosion)
def binary_erosion_gpu(mask, structure=None, border_value: int = 0):
    """
    GPU-accelerated binary erosion. Falls back to scipy on CPU. Boolean
    morphology (unlike uniform_filter's float sums) -- verified
    bit-identical to scipy on CPU.

    Args:
        mask:         Boolean array (numpy or cupy)
        structure:    Structuring element (numpy array), or None for the
                      default cross/square scipy would use
        border_value: Value assumed outside the array bounds

    Returns:
        numpy.ndarray (always, regardless of input type)
    """
    import cupy as cp
    from cupyx.scipy.ndimage import binary_erosion as cp_binary_erosion

    mask_gpu = cp.asarray(mask, dtype=cp.bool_)
    struct_gpu = cp.asarray(structure, dtype=cp.bool_) if structure is not None else None
    return cp_binary_erosion(mask_gpu, structure=struct_gpu, border_value=border_value)


def _cpu_percentile_masked(array: np.ndarray, mask: np.ndarray,
                            q: float) -> float:
    vals  = array[mask]
    valid = vals[(vals > -1000) & (vals < 9000)]
    if len(valid) == 0:
        return 0.0
    return float(np.percentile(valid, q))


def percentile_masked_gpu(
    array,
    mask,
    q: float,
) -> float:
    """
    Compute percentile of array values where mask is True,
    excluding nodata values (<-1000 or >9000).

    Works on both numpy and cupy arrays.
    Falls back to numpy if GPU unavailable.

    Returns:
        float scalar always
    """
    if not GPU_AVAILABLE:
        return _cpu_percentile_masked(
            np.asarray(array), np.asarray(mask), q)
    try:
        import cupy as cp
        arr_gpu  = cp.asarray(array)
        mask_gpu = cp.asarray(mask, dtype=cp.bool_)
        vals     = arr_gpu[mask_gpu]
        valid    = vals[(vals > -1000) & (vals < 9000)]
        if len(valid) == 0:
            return 0.0
        return float(cp.percentile(valid, q))
    except Exception:
        _maybe_free_gpu()
        return _cpu_percentile_masked(
            np.asarray(array), np.asarray(mask), q)


# ─────────────────────────────────────────────
# GPU label array processing
# ─────────────────────────────────────────────

def _mask_centroid_latlon(mask, bbox, transform):
    """Rough midpoint for locating a water body: the nearest actual water
    pixel to the mean pixel position of its rasterized mask — not its
    source polygon's vertices, and not the raw mean either (which can
    still fall in a gap for a non-convex/zigzag shape). See the
    identically-named/behaved helper in dem_water_correction.py for the
    full rationale. Kept as a separate copy here rather than imported,
    since gpu_tools.py is meant to have no domain-specific coupling (see
    module docstring) and dem_water_correction.py already imports from
    this module — importing back would be circular.

    `mask` may be a cupy or numpy boolean array; `bbox` is a (row_slice,
    col_slice) tuple for a bbox-local mask, or None if `mask` is already
    full-raster-sized. Returns (lat, lon) or (None, None).
    """
    from rasterio.transform import xy as _transform_xy
    rows, cols = np.where(mask.get() if hasattr(mask, 'get') else mask)
    if len(rows) == 0 or transform is None:
        return None, None
    mean_row, mean_col = rows.mean(), cols.mean()
    nearest = np.argmin((rows - mean_row) ** 2 + (cols - mean_col) ** 2)
    row_off = bbox[0].start if bbox is not None else 0
    col_off = bbox[1].start if bbox is not None else 0
    lon, lat = _transform_xy(transform, row_off + rows[nearest], col_off + cols[nearest])
    return lat, lon


def process_label_array_gpu(
    label_array: np.ndarray,
    corrected: np.ndarray,
    batch: list,
    buffer_pixels: int = 3,
    shore_percentile: float = 10.0,
    min_water_pixels: int = 5,
    transform=None,
) -> tuple:
    """
    GPU-accelerated version of _process_label_array.

    Transfers label_array and corrected to GPU once per batch.
    Runs grey_dilation and percentile ops on GPU.
    Falls back to CPU implementation if GPU unavailable or fails.

    The GPU speedup is largest for:
      - Large rasters (>5000×5000 pixels) where dilation is expensive
      - Large batches with many valid polygons
      - High buffer_pixels values (wider dilation)

    Args:
        label_array:      int32 array [H, W] — each polygon has unique label
        corrected:        float32 elevation array [H, W] — modified in-place
        batch:            list of (orig_idx, polygon_dict) tuples
        buffer_pixels:    dilation radius for shoreline band
        shore_percentile: which percentile of shoreline to use as surface elev
        min_water_pixels: skip polygons with fewer pixels than this
        transform:        affine transform of `label_array`'s raster, for
                           each corrected body's rough lat/lon midpoint
                           (see _mask_centroid_latlon). Optional — None
                           just leaves lat/lon out of elevation_changes.

    Returns:
        (corrected, elevation_changes, corrected_count, skipped_small)
        Same signature as _process_label_array for drop-in compatibility.
    """
    if not GPU_AVAILABLE:
        return _process_label_array_cpu(
            label_array, corrected, batch,
            buffer_pixels, shore_percentile, min_water_pixels,
            transform=transform,
        )

    try:
        return _process_label_array_gpu_impl(
            label_array, corrected, batch,
            buffer_pixels, shore_percentile, min_water_pixels,
            transform=transform,
        )
    except Exception as e:
        # Free any partially allocated GPU memory before falling back
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            cp.cuda.Device(0).synchronize()
        except Exception:
            pass
        print(f"[GPU] Falling back to CPU: {e}")
        return _process_label_array_cpu(
            label_array, corrected, batch,
            buffer_pixels, shore_percentile, min_water_pixels,
            transform=transform,
        )


def _process_label_array_gpu_impl(
    label_array: np.ndarray,
    corrected: np.ndarray,
    batch: list,
    buffer_pixels: int,
    shore_percentile: float,
    min_water_pixels: int,
    transform=None,
) -> tuple:
    """
    GPU implementation. As of the Phase 4 perf patch, this uses
    grey_dilation instead of distance_transform_edt — same semantics
    as the CPU patch in dem_water_correction.py.

    Why grey_dilation, not distance_transform_edt:
      The previous code called distance_transform_edt(... return_indices=True)
      to get BOTH a distance map (for the shoreline threshold) AND nearest-
      label propagation (via the indices output). EDT with return_indices
      computes ~2× the work of a plain EDT, and we never used the actual
      distances — just the `dist <= buffer_pixels` threshold.

      grey_dilation on the label_array with a (2b+1)×(2b+1) square footprint
      propagates each label outward by exactly b pixels (the structuring
      element radius). The dilated_labels > 0 region IS the union of water +
      shoreline-of-width-b, and the value at each shoreline pixel IS the
      propagated label. Single ndimage call, no distance arithmetic, no
      indices array.

    Correctness tradeoff vs the prior EDT approach:
      grey_dilation propagates the MAX label value in the neighborhood;
      EDT-with-indices propagates the NEAREST label. These differ only at
      boundaries between two distinct polygons within buffer_pixels of each
      other (i.e. touching/adjacent water bodies). In those cases the
      shoreline elevations are very similar anyway, so the per-polygon
      surface_elev percentile is barely affected. This is the same
      approximation the CPU patch makes; treating GPU and CPU paths
      consistently here avoids subtle behavior differences between them.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import grey_dilation as cp_grey_dilation

    elevation_changes = []
    corrected_count   = 0
    skipped_small     = 0

    # Upload inputs — already cropped by caller
    label_gpu     = cp.asarray(label_array, dtype=cp.int32)
    corrected_gpu = cp.asarray(corrected,   dtype=cp.float32)

    water_any_gpu = label_gpu > 0

    # ── Single grey_dilation, replaces EDT + indices arithmetic ───
    # Square footprint of radius = buffer_pixels. grey_dilation outputs
    # an int32 array where each pixel = max label in its (2b+1)² window.
    struct_gpu = cp.ones(
        (2 * buffer_pixels + 1, 2 * buffer_pixels + 1), dtype=cp.bool_,
    )
    dilated_labels_gpu = cp_grey_dilation(label_gpu, footprint=struct_gpu)

    # Shoreline mask: dilated region minus the original water = the band
    # of width buffer_pixels just outside each water body.
    in_shore_gpu = (dilated_labels_gpu > 0) & ~water_any_gpu

    shore_labels_gpu = dilated_labels_gpu[in_shore_gpu]
    shore_elevs_gpu  = corrected_gpu[in_shore_gpu]

    for batch_pos, (orig_idx, poly) in enumerate(batch):
        label      = batch_pos + 1
        water_mask = label_gpu == label

        pixel_count = int(water_mask.sum())
        if pixel_count < min_water_pixels:
            skipped_small += 1
            continue

        label_shore = shore_labels_gpu == label
        valid = shore_elevs_gpu[label_shore]
        valid = valid[(valid > -1000) & (valid < 9000)]

        if len(valid) == 0:
            inside = corrected_gpu[water_mask]
            valid  = inside[inside > -1000]
            if len(valid) == 0:
                continue

        surface_elev = float(cp.percentile(valid, shore_percentile))

        orig_vals  = corrected_gpu[water_mask]
        valid_orig = orig_vals[orig_vals > -1000]
        orig_mean  = float(cp.mean(valid_orig)) if len(valid_orig) > 0 else 0.0

        corrected_gpu[water_mask] = surface_elev
        corrected_count += 1

        name = poly.get('properties', {}).get('name', '')
        lat, lon = _mask_centroid_latlon(water_mask, None, transform)
        elevation_changes.append({
            'polygon_index':   orig_idx,
            'name':            name,
            'pixel_count':     pixel_count,
            'surface_elev_m':  round(surface_elev, 1),
            'original_mean_m': round(orig_mean, 1),
            'change_m':        round(surface_elev - orig_mean, 1),
            'lat':             round(lat, 5) if lat is not None else None,
            'lon':             round(lon, 5) if lon is not None else None,
        })

    # Download result and write back to caller's array
    corrected[:] = cp.asnumpy(corrected_gpu)

    del label_gpu, corrected_gpu, water_any_gpu, dilated_labels_gpu
    del struct_gpu, in_shore_gpu, shore_labels_gpu, shore_elevs_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return corrected, elevation_changes, corrected_count, skipped_small



def _process_label_array_cpu(
    label_array: np.ndarray,
    corrected: np.ndarray,
    batch: list,
    buffer_pixels: int,
    shore_percentile: float,
    min_water_pixels: int,
    transform=None,
) -> tuple:
    """
    Pure CPU fallback — identical algorithm to
    _process_label_array_gpu_impl (and to the patched _process_label_array
    in dem_water_correction.py).

    Pre-Phase-4-patch, this function used per-polygon binary_dilation in
    a loop, which made it ~2-5× slower than the patched main-module CPU
    code. Now synchronized to use the same grey_dilation + find_objects
    pattern so behavior is consistent if the GPU path falls back.
    """
    from scipy.ndimage import grey_dilation, find_objects

    elevation_changes = []
    corrected_count   = 0
    skipped_small     = 0

    water_any = label_array > 0

    # Build dilated label array once
    struct = np.ones(
        (2 * buffer_pixels + 1, 2 * buffer_pixels + 1), dtype=bool,
    )
    dilated_labels = grey_dilation(label_array, footprint=struct)

    # Shoreline derived directly from dilated_labels — no EDT needed
    in_shoreline = (dilated_labels > 0) & ~water_any
    shore_elevs  = corrected[in_shoreline]
    shore_labels = dilated_labels[in_shoreline]

    # Per-label bounding boxes for fast slicing
    label_objects = find_objects(label_array)

    for batch_pos, (orig_idx, poly) in enumerate(batch):
        label = batch_pos + 1

        if label - 1 >= len(label_objects):
            continue
        bbox = label_objects[label - 1]
        if bbox is None:
            continue

        label_sub = label_array[bbox]
        water_mask_local = label_sub == label

        pixel_count = int(water_mask_local.sum())
        if pixel_count < min_water_pixels:
            skipped_small += 1
            continue

        label_shore_mask = shore_labels == label
        valid = shore_elevs[label_shore_mask]
        valid = valid[(valid > -1000) & (valid < 9000)]

        if len(valid) == 0:
            inside = corrected[bbox][water_mask_local]
            valid  = inside[inside > -1000]
            if len(valid) == 0:
                continue

        surface_elev = float(np.percentile(valid, shore_percentile))

        orig_vals_in_bbox = corrected[bbox][water_mask_local]
        valid_orig = orig_vals_in_bbox[orig_vals_in_bbox > -1000]
        orig_mean  = float(np.mean(valid_orig)) if len(valid_orig) > 0 else 0.0

        corrected[bbox][water_mask_local] = surface_elev
        corrected_count += 1

        name = poly.get('properties', {}).get('name', '')
        lat, lon = _mask_centroid_latlon(water_mask_local, bbox, transform)
        elevation_changes.append({
            'polygon_index':   orig_idx,
            'name':            name,
            'pixel_count':     pixel_count,
            'surface_elev_m':  round(surface_elev, 1),
            'original_mean_m': round(orig_mean, 1),
            'change_m':        round(surface_elev - orig_mean, 1),
            'lat':             round(lat, 5) if lat is not None else None,
            'lon':             round(lon, 5) if lon is not None else None,
        })

    return corrected, elevation_changes, corrected_count, skipped_small


# ─────────────────────────────────────────────
# GPU memory sizing helper
# ─────────────────────────────────────────────

def fits_on_gpu(arrays_mb: float, safety_factor: float = 0.8) -> bool:
    """
    Check whether a set of arrays would fit in GPU memory.

    Args:
        arrays_mb:     Total size of arrays to upload in MB
        safety_factor: Fraction of free memory to consider safe (default 0.8)

    Returns:
        True if arrays likely fit, False otherwise
    """
    free_gb  = gpu_memory_available_gb()
    free_mb  = free_gb * 1000
    return arrays_mb < free_mb * safety_factor


def recommend_gpu_batch_size(
    raster_shape: tuple,
    dtype_bytes: int = 4,
    safety_factor: float = 0.7,
    avg_crop_fraction: float = 0.05,  # assume crop is ~5% of full raster
) -> int:
    """
    Recommend batch size based on expected crop size per batch,
    not full raster size. Crops are typically 2-10% of the full
    raster for spatially sorted batches.
    """
    if not GPU_AVAILABLE:
        return 500

    h, w     = raster_shape
    # Estimate crop size — spatially sorted batches cover a fraction
    # of the full raster. Conservative default is 5%.
    crop_h   = int(h * avg_crop_fraction ** 0.5)
    crop_w   = int(w * avg_crop_fraction ** 0.5)

    # Per batch: label_crop (int32) + corrected_crop (float32)
    # + dilated_labels (int32) + shoreline mask (bool) — roughly 4x crop
    # NB: was 5× when distance_transform_edt produced a float64 dist array;
    # grey_dilation drops that allocation, so 4× is the new ballpark.
    crop_mb  = (crop_h * crop_w * dtype_bytes * 4) / 1e6

    free_gb  = gpu_memory_available_gb()
    free_mb  = free_gb * 1000 * safety_factor

    if crop_mb > free_mb:
        print(f"[GPU] Estimated crop ({crop_mb:.0f}MB) exceeds "
              f"VRAM ({free_mb:.0f}MB). GPU disabled.")
        return 0

    # More polygons per batch = larger crop = more VRAM
    # Scale batch_size so crop stays within budget
    headroom = free_mb / crop_mb
    if headroom >= 4:
        return 1000
    elif headroom >= 2:
        return 500
    else:
        return 200