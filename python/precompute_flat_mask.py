"""
precompute_flat_mask.py
━━━━━━━━━━━━━━━━━━━━━━
Precomputes the flat zone mask for a study area DEM mosaic.

The flat zone mask identifies terrain pixels where local variance is low
enough that adjacent compute rays can share visibility state without
resampling. A downstream compute engine recomputes this mask from scratch
for every run today — an 8-15 second cost per observer point.

Running this script once produces a uint8 GeoTIFF that the downstream
engine loads once per worker process and slices per observer, eliminating
that recomputation entirely.

The mask matches the exact parameters used by that downstream engine:
  window=5, variance_threshold=4.0

Usage:
    python precompute_flat_mask.py --dem path/to/mosaic.tif --output path/to/mask.tif
"""

import argparse
import time
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

try:
    from gpu_tools import uniform_filter_gpu, binary_erosion_gpu, GPU_AVAILABLE
    _GPU_TOOLS_AVAILABLE = True
except ImportError:
    _GPU_TOOLS_AVAILABLE = False

# ─────────────────────────────────────────────
# Parameters — must match the downstream compute
# engine's own flat-zone computation exactly
# ─────────────────────────────────────────────

FLAT_ZONE_WINDOW    = 5      # uniform_filter window size
FLAT_ZONE_VARIANCE  = 4.0   # variance threshold in m²

# ─────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────

def compute_flat_mask(
    dem_path: str,
    output_path: str,
    window: int   = FLAT_ZONE_WINDOW,
    threshold: float = FLAT_ZONE_VARIANCE,
    strip_height: int = 2000,
) -> dict:
    """
    Compute and save a flat zone mask for a DEM GeoTIFF.

    Processes the DEM in horizontal strips to avoid loading the full
    array into RAM twice simultaneously. At 10m resolution a 300x300km
    DEM is ~27kx27k pixels — the float64 intermediate arrays would use
    ~12GB if processed at once. Strip processing keeps peak RAM under 2GB.

    Args:
        dem_path:     Path to input DEM GeoTIFF (corrected mosaic)
        output_path:  Path to write flat mask GeoTIFF
        window:       Uniform filter window size (must match the downstream engine)
        threshold:    Variance threshold in m² (must match the downstream engine)
        strip_height: Rows per processing strip (default 2000)

    Returns:
        Stats dict with flat zone coverage and timing
    """
    print(f"[FLAT MASK] Input:  {dem_path}")
    print(f"[FLAT MASK] Output: {output_path}")
    print(f"[FLAT MASK] Parameters: window={window}, threshold={threshold}")

    t0 = time.perf_counter()

    with rasterio.open(dem_path) as src:
        h, w      = src.height, src.width
        profile   = src.profile.copy()
        nodata    = src.nodata if src.nodata is not None else -9999.0

    print(f"[FLAT MASK] DEM size: {w:,} x {h:,} pixels "
          f"({w*h/1e6:.0f}M pixels)")
    print(f"[FLAT MASK] Processing in strips of {strip_height} rows...")

    # Output profile — uint8, deflate compressed
    out_profile = profile.copy()
    out_profile.update({
        'dtype':    'uint8',
        'count':    1,
        'compress': 'deflate',
        'predictor': 2,
        'tiled':    True,
        'blockxsize': 512,
        'blockysize': 512,
        'nodata':   None,
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_flat   = 0
    total_pixels = 0
    n_strips     = (h + strip_height - 1) // strip_height
    # Global elevation range across VALID (non-nodata) pixels only -- tracked
    # alongside the flat-mask pass at near-zero extra cost, so tile_builder's
    # validation gate can tell "genuinely flat terrain" (real elevation range,
    # just low local variance) apart from "degenerate/placeholder DEM" (near-
    # zero range across the whole tile) without a second read of the DEM.
    elev_min = None
    elev_max = None

    # Open output for writing — use windowed writes
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        with rasterio.open(dem_path) as src:
            for strip_i in range(n_strips):
                row_start = strip_i * strip_height
                row_end   = min(row_start + strip_height, h)

                # Overlap by window//2 on each side to avoid edge artifacts
                # from the uniform filter at strip boundaries
                overlap   = window // 2 + 1
                read_start = max(0, row_start - overlap)
                read_end   = min(h, row_end   + overlap)

                # Read strip with overlap
                window_rasterio = rasterio.windows.Window(
                    col_off=0,
                    row_off=read_start,
                    width=w,
                    height=read_end - read_start,
                )
                strip_data = src.read(1, window=window_rasterio).astype(np.float64)

                # Elevation range over valid pixels, BEFORE nodata gets
                # zero-filled below (0.0 can be a real elevation in coastal
                # tiles, so this must use the original validity mask).
                valid_mask = strip_data >= (nodata + 1)
                if valid_mask.any():
                    strip_min = float(strip_data[valid_mask].min())
                    strip_max = float(strip_data[valid_mask].max())
                    elev_min = strip_min if elev_min is None else min(elev_min, strip_min)
                    elev_max = strip_max if elev_max is None else max(elev_max, strip_max)

                # Mask nodata
                strip_data = np.where(
                    strip_data < nodata + 1, 0.0, strip_data
                )

                # Compute variance via sliding window. GPU path (falls
                # back to CPU internally if unavailable/fails) -- verified
                # against real DEM data to produce a bit-identical
                # post-threshold flat mask despite ~1e-11 float64
                # summation-order differences (see uniform_filter_gpu's
                # docstring in gpu_tools.py).
                if _GPU_TOOLS_AVAILABLE:
                    mean    = uniform_filter_gpu(strip_data, size=window)
                    mean_sq = uniform_filter_gpu(strip_data ** 2, size=window)
                else:
                    mean    = uniform_filter(strip_data, size=window)
                    mean_sq = uniform_filter(strip_data ** 2, size=window)
                variance = mean_sq - mean ** 2

                flat_strip = (variance < threshold).astype(np.uint8)

                # Trim overlap back to actual strip rows
                trim_top    = row_start - read_start
                trim_bottom = trim_top + (row_end - row_start)
                flat_trimmed = flat_strip[trim_top:trim_bottom, :]

                # Write strip to output
                write_window = rasterio.windows.Window(
                    col_off=0,
                    row_off=row_start,
                    width=w,
                    height=row_end - row_start,
                )
                dst.write(flat_trimmed, 1, window=write_window)

                strip_flat   = int(flat_trimmed.sum())
                strip_pixels = flat_trimmed.size
                total_flat   += strip_flat
                total_pixels += strip_pixels

                pct_flat  = strip_flat / strip_pixels * 100
                pct_total = (strip_i + 1) / n_strips * 100
                elapsed   = time.perf_counter() - t0
                print(f"  Strip {strip_i+1:>3}/{n_strips} "
                      f"(rows {row_start:>6}-{row_end:>6}, "
                      f"{pct_total:.0f}%) "
                      f"flat={pct_flat:.1f}% "
                      f"elapsed={elapsed:.1f}s")

    elapsed    = time.perf_counter() - t0
    flat_pct   = total_flat / total_pixels * 100
    mask_mb    = Path(output_path).stat().st_size / 1e6

    elev_range_m = (elev_max - elev_min) if (elev_min is not None and elev_max is not None) else None

    print(f"\n[FLAT MASK] Complete in {elapsed:.1f}s")
    print(f"  Flat pixels:  {total_flat:,} / {total_pixels:,} "
          f"({flat_pct:.1f}%)")
    print(f"  Elevation range: {elev_range_m if elev_range_m is not None else 'n/a (no valid pixels)'}")
    print(f"  Output size:  {mask_mb:.1f} MB")
    print(f"  Output:       {output_path}")

    return {
        'dem_path':    dem_path,
        'output_path': output_path,
        'h':           h,
        'w':           w,
        'flat_pct':    round(flat_pct, 2),
        'total_flat':  total_flat,
        'total_pixels':total_pixels,
        'elapsed_s':   round(elapsed, 1),
        'mask_mb':     round(mask_mb, 1),
        'window':      window,
        'threshold':   threshold,
        'elev_min_m':   round(elev_min, 2) if elev_min is not None else None,
        'elev_max_m':   round(elev_max, 2) if elev_max is not None else None,
        'elev_range_m': round(elev_range_m, 2) if elev_range_m is not None else None,
    }

# ─────────────────────────────────────────────
# Flat mask compressor
# ─────────────────────────────────────────────
def compute_flat_regions(
    flat_mask_path: str,
    output_path: str,
    strip_height: int = 2000,
) -> dict:
    """
    Compute connected flat zone regions and their boundary masks.
    
    From the flat mask (already computed), derives:
      - boundary_mask: uint8 GeoTIFF — True only at flat/non-flat edges
      - region metadata (stored in .npz alongside the boundary GeoTIFF)
    
    The boundary mask is a drop-in replacement for the full flat mask
    in _apply_flat_zone_cache — rays only cache boundary cell crossings
    rather than every flat cell, reducing cache size by ~10-50x.
    
    A boundary cell is any flat cell that has at least one non-flat
    neighbor in the 4-connected neighborhood (N/S/E/W). This captures
    the exact entry/exit points where rays transition between flat and
    non-flat terrain.
    
    Output:
        {output_path}_boundary.tif  — uint8 boundary mask GeoTIFF
        {output_path}_stats.json    — region statistics
    """
    from scipy.ndimage import label, binary_erosion
    
    print(f"[FLAT REGIONS] Input mask: {flat_mask_path}")
    t0 = time.perf_counter()
    
    with rasterio.open(flat_mask_path) as src:
        flat_mask = src.read(1).astype(bool)
        transform = src.transform
        profile   = src.profile.copy()
        h, w      = src.height, src.width
    
    print(f"[FLAT REGIONS] Mask: {w:,}x{h:,}, "
          f"{flat_mask.mean()*100:.1f}% flat")
    
    # ── Compute boundary mask ─────────────────────────────────────
    # A flat cell is a boundary cell if any 4-connected neighbor is non-flat.
    # Equivalent to: flat AND NOT (eroded flat)
    # binary_erosion with cross-shaped structuring element gives 4-connectivity
    print(f"[FLAT REGIONS] Computing boundary mask...")
    
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)  # 4-connected
    
    # Process in strips to manage RAM — erosion needs 1-pixel overlap
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    overlap = 1
    
    strip_h = strip_height
    n_strips = (h + strip_h - 1) // strip_h
    
    for strip_i in range(n_strips):
        r_start = strip_i * strip_h
        r_end   = min(r_start + strip_h, h)
        r_read_start = max(0, r_start - overlap)
        r_read_end   = min(h, r_end + overlap)
        
        strip = flat_mask[r_read_start:r_read_end, :]
        if _GPU_TOOLS_AVAILABLE:
            eroded = binary_erosion_gpu(strip, structure=struct, border_value=0)
        else:
            eroded = binary_erosion(strip, structure=struct, border_value=0)
        
        # Boundary = flat but not fully surrounded by flat
        strip_boundary = strip & ~eroded
        
        # Trim overlap
        trim_top = r_start - r_read_start
        trim_bot = trim_top + (r_end - r_start)
        boundary_mask[r_start:r_end, :] = \
            strip_boundary[trim_top:trim_bot, :].astype(np.uint8)
        
        pct = (strip_i + 1) / n_strips * 100
        print(f"  Strip {strip_i+1:>3}/{n_strips} ({pct:.0f}%) "
              f"elapsed={time.perf_counter()-t0:.1f}s")
    
    boundary_pct  = boundary_mask.mean() * 100
    flat_pct      = flat_mask.mean() * 100
    reduction     = (1 - boundary_mask.sum() / max(flat_mask.sum(), 1)) * 100
    
    print(f"[FLAT REGIONS] Boundary cells: {boundary_mask.sum():,} "
          f"({boundary_pct:.2f}% of DEM, "
          f"{reduction:.1f}% reduction vs full flat mask)")
    
    # ── Write boundary mask GeoTIFF ───────────────────────────────
    boundary_path = output_path.replace('.tif', '_boundary.tif')
    out_profile = profile.copy()
    out_profile.update({
        'dtype':     'uint8',
        'count':     1,
        'compress':  'deflate',
        'predictor': 2,
        'tiled':     True,
        'blockxsize': 512,
        'blockysize': 512,
        'nodata':    None,
    })
    
    Path(boundary_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(boundary_path, 'w', **out_profile) as dst:
        dst.write(boundary_mask, 1)
    
    boundary_mb = Path(boundary_path).stat().st_size / 1e6
    elapsed = time.perf_counter() - t0
    
    print(f"[FLAT REGIONS] Written: {boundary_path} "
          f"({boundary_mb:.1f}MB, {elapsed:.1f}s)")
    
    stats = {
        'flat_mask_path':   flat_mask_path,
        'boundary_path':    boundary_path,
        'h': h, 'w': w,
        'flat_pct':         round(flat_pct, 2),
        'boundary_pct':     round(boundary_pct, 2),
        'flat_cells':       int(flat_mask.sum()),
        'boundary_cells':   int(boundary_mask.sum()),
        'reduction_pct':    round(reduction, 1),
        'elapsed_s':        round(elapsed, 1),
        'boundary_mb':      round(boundary_mb, 1),
    }
    
    import json
    stats_path = boundary_path.replace('_boundary.tif', '_boundary_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats
# ─────────────────────────────────────────────
# Flat mask reader — used by a downstream compute engine
# ─────────────────────────────────────────────

def load_flat_mask(mask_path: str, window=None) -> tuple:
    """
    Load a precomputed flat mask into memory.
    Returns (flat_mask_array, transform) ready for slicing per observer.

    The array is kept as uint8 in memory (~700MB for 27kx27k mosaic)
    and sliced to the compute window before use.

    Args:
        window: optional rasterio.windows.Window (or (col_off, row_off,
                width, height) tuple) to read just that sub-region of
                the mosaic instead of the whole thing. Runs with only
                a handful of observers only ever need the small window
                around their radius_m reach, not the full multi-tile
                mosaic extent -- this is the same windowed-read
                principle already used for the DEM elevation read
                elsewhere in the larger pipeline, just not previously
                applied here. The returned transform is adjusted to the
                window's own origin via rasterio.windows.transform, so
                slice_flat_mask's downstream lat/lng math is unaffected
                by windowing. None (default) reads the full mosaic,
                unchanged behavior for any other caller.

    Usage in the downstream compute engine:
        flat_mask_array, flat_transform = load_flat_mask(mask_path)
        # then per observer:
        crop = slice_flat_mask(flat_mask_array, flat_transform,
                               observer_lat, observer_lng, radius_m)
    """
    print(f"[FLAT MASK] Loading {mask_path}...")
    t0 = time.perf_counter()

    with rasterio.open(mask_path) as src:
        flat_mask = src.read(1, window=window)  # uint8 array
        transform = (src.window_transform(window) if window is not None
                     else src.transform)

    size_mb = flat_mask.nbytes / 1e6
    flat_pct = flat_mask.mean() * 100
    print(f"[FLAT MASK] Loaded {flat_mask.shape[1]:,}x{flat_mask.shape[0]:,} "
          f"mask in {time.perf_counter()-t0:.1f}s "
          f"({size_mb:.0f}MB, {flat_pct:.1f}% flat)")

    return flat_mask, transform

def load_flat_boundary(boundary_path: str, window=None) -> tuple:
    """
    Load precomputed flat zone boundary mask.
    Drop-in replacement for load_flat_mask — returns same
    (array, transform) tuple but with ~10-50x fewer True cells.

    Use this in the downstream compute engine instead of load_flat_mask
    when the boundary file exists — it produces identical results with
    lower cache memory usage.

    Args:
        window: see load_flat_mask's `window` docstring — same
                windowed-read support, same default (None = full read).
    """
    print(f"[FLAT MASK] Loading boundary mask: {boundary_path}...")
    t0 = time.perf_counter()

    with rasterio.open(boundary_path) as src:
        boundary_mask = src.read(1, window=window)
        transform     = (src.window_transform(window) if window is not None
                          else src.transform)

    size_mb      = boundary_mask.nbytes / 1e6
    boundary_pct = boundary_mask.mean() * 100
    print(f"[FLAT MASK] Boundary mask: "
          f"{boundary_mask.shape[1]:,}x{boundary_mask.shape[0]:,} "
          f"in {time.perf_counter()-t0:.1f}s "
          f"({size_mb:.0f}MB, {boundary_pct:.2f}% boundary cells)")

    return boundary_mask, transform


# ─────────────────────────────────────────────
# Bit-packed loaders — same on-disk contract, ~8x less resident memory
# ─────────────────────────────────────────────
#
# New sibling functions rather than a flag on load_flat_mask/
# load_flat_boundary: every real caller of those two (the downstream
# compute engine and a handful of benchmark/profiling scripts) does a
# hard 2-tuple unpack of the return value.
# Keeping the existing functions byte-for-byte unchanged means those
# callers are provably unaffected -- no new parameter, no conditional
# return arity, nothing to accidentally break.
#
# A full regional flat/boundary mask is 0/1 uint8 (1 byte/pixel) -- pure
# boolean data. np.packbits stores 8 pixels/byte, so a ~1050MB mask
# becomes ~131MB resident. Packing is done directly on the raw uint8
# array (NOT arr.astype(bool) first) -- the tile-generation code in this
# file (compute_flat_mask/compute_flat_regions) guarantees these rasters
# are always strictly 0/1, and np.packbits treats any nonzero element as
# 1 without needing an explicit bool cast, so skipping that cast avoids
# a redundant full-size copy that would otherwise double the transient
# peak during loading. Verified empirically (a small standalone spike
# script) that freeing the original unpacked array after packing actually
# decommits memory (not just Python-level bookkeeping) even under a
# tight Windows Job Object memory cap -- two full pack-then-free cycles
# succeeded under a 2000MB cap with committed memory cleanly returning
# to baseline each time.
#
# Pair with slice_flat_mask(..., packed_width=...) to crop without ever
# unpacking more than the requested window.

def load_flat_mask_packed(mask_path: str, window=None) -> Tuple[np.ndarray, object, int]:
    """Like load_flat_mask, but bit-packs the result. Returns
    (packed_array, transform, logical_width) -- logical_width is the
    true pre-pack pixel width, required by slice_flat_mask's
    packed_width argument since np.packbits pads each row's last byte
    when width isn't a multiple of 8."""
    print(f"[FLAT MASK] Loading {mask_path} (packed)...")
    t0 = time.perf_counter()

    with rasterio.open(mask_path) as src:
        flat_mask = src.read(1, window=window)
        transform = (src.window_transform(window) if window is not None
                     else src.transform)

    width = flat_mask.shape[1]
    packed = np.packbits(flat_mask, axis=1)
    del flat_mask

    print(f"[FLAT MASK] Loaded+packed {width:,}x{packed.shape[0]:,} mask "
          f"in {time.perf_counter()-t0:.1f}s "
          f"({packed.nbytes/1e6:.0f}MB packed, vs "
          f"{packed.shape[0]*width/1e6:.0f}MB unpacked)")

    return packed, transform, width


def load_flat_boundary_packed(boundary_path: str, window=None) -> Tuple[np.ndarray, object, int]:
    """Like load_flat_boundary, but bit-packs the result. See
    load_flat_mask_packed's docstring for the packing rationale;
    identical here."""
    print(f"[FLAT MASK] Loading boundary mask: {boundary_path} (packed)...")
    t0 = time.perf_counter()

    with rasterio.open(boundary_path) as src:
        boundary_mask = src.read(1, window=window)
        transform = (src.window_transform(window) if window is not None
                     else src.transform)

    width = boundary_mask.shape[1]
    packed = np.packbits(boundary_mask, axis=1)
    del boundary_mask

    print(f"[FLAT MASK] Boundary mask (packed): {width:,}x{packed.shape[0]:,} "
          f"in {time.perf_counter()-t0:.1f}s "
          f"({packed.nbytes/1e6:.0f}MB packed, vs "
          f"{packed.shape[0]*width/1e6:.0f}MB unpacked)")

    return packed, transform, width


def slice_flat_mask(
    flat_mask: np.ndarray,
    mask_transform,
    observer_lat: float,
    observer_lng: float,
    radius_m: float,
    dem_transform,
    dem_shape: tuple,
    packed_width: Optional[int] = None,
) -> np.ndarray:
    """
    Slice the regional flat mask to the compute window for one observer.

    The slice matches the DEM window that load_merged_dem would return,
    so it can be passed directly to _apply_flat_zone_cache as a drop-in
    replacement for the per-observer computed mask.

    Args:
        flat_mask:      Full regional flat mask array (uint8), OR a
                         bit-packed array (np.packbits(..., axis=1)) if
                         packed_width is given.
        mask_transform: Affine transform of the flat mask
        observer_lat:   Observer latitude
        observer_lng:   Observer longitude
        radius_m:       Compute radius in metres
        dem_transform:  Transform of the loaded DEM window
        dem_shape:      Shape (h, w) of the loaded DEM window
        packed_width:   If flat_mask came from load_flat_mask_packed/
                         load_flat_boundary_packed, pass its returned
                         logical width here. None (default) means
                         flat_mask is a plain unpacked array — behavior
                         is then identical to before this parameter
                         existed. When set, only the byte-aligned
                         super-region covering the crop is ever
                         unpacked (np.unpackbits), so a call against a
                         packed regional mask never materializes more
                         than the requested window, even when that
                         window happens to span the whole mosaic.

    Returns:
        Boolean array matching dem_shape — True where terrain is flat
    """
    h, w = dem_shape
    mh = flat_mask.shape[0]
    mw = packed_width if packed_width is not None else flat_mask.shape[1]

    # Convert DEM window bounds to mask pixel coordinates
    # DEM window top-left in geographic coords
    win_north = dem_transform.f
    win_west  = dem_transform.c
    win_south = dem_transform.f + dem_transform.e * h
    win_east  = dem_transform.c + dem_transform.a * w

    # Convert to mask pixel indices
    r_min = round((win_north - mask_transform.f) / mask_transform.e)
    r_max = round((win_south - mask_transform.f) / mask_transform.e)
    c_min = round((win_west  - mask_transform.c) / mask_transform.a)
    c_max = round((win_east  - mask_transform.c) / mask_transform.a)

    # Clamp to mask bounds
    r_min_c = max(0, r_min)
    r_max_c = max(0, min(r_max, mh))
    c_min_c = max(0, c_min)
    c_max_c = max(0, min(c_max, mw))

    if packed_width is not None:
        # Unpack only the byte-aligned super-region covering
        # [c_min_c:c_max_c] -- never the whole packed width -- so a
        # crop against a regional mask that's been packed at rest stays
        # cheap even when the requested window spans the entire mosaic
        # (the common case for this pipeline's current radius/cushion
        # settings). np.packbits/np.unpackbits both default to
        # bitorder='big'; neither call site in this codebase overrides
        # it, which is required for them to be exact inverses of each
        # other -- do not pass a bitorder kwarg to either without
        # updating both.
        byte_c_min = c_min_c // 8
        byte_c_max = -(-c_max_c // 8)  # ceil division
        packed_rows = flat_mask[r_min_c:r_max_c, byte_c_min:byte_c_max]
        unpacked_rows = np.unpackbits(packed_rows, axis=1)
        local_c_min = c_min_c - byte_c_min * 8
        local_c_max = local_c_min + (c_max_c - c_min_c)
        crop = unpacked_rows[:, local_c_min:local_c_max]
    else:
        # Crop what's available
        crop = flat_mask[r_min_c:r_max_c, c_min_c:c_max_c]

    # Expected output shape from the raw (unclamped) indices
    expected_h = dem_shape[0]   # use DEM shape as ground truth
    expected_w = dem_shape[1]   # instead of computing from indices

    if crop.shape == (expected_h, expected_w):
        # Clean crop — no edge clipping
        pass
    else:
        # Pad with zeros where mask doesn't cover DEM window
        out = np.zeros((expected_h, expected_w), dtype=np.uint8)
        # Compute paste offsets
        dst_r = r_min_c - r_min
        dst_c = c_min_c - c_min
        paste_h = crop.shape[0]
        paste_w = crop.shape[1]
        out[dst_r:dst_r + paste_h, dst_c:dst_c + paste_w] = crop
        crop = out

    return crop.astype(bool)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Precompute flat zone mask for a study area DEM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--dem',    default=None,
        help='Path to input DEM GeoTIFF')
    parser.add_argument('--output', default=None,
        help='Path to output mask GeoTIFF')
    parser.add_argument('--window',    type=int,   default=FLAT_ZONE_WINDOW,
        help=f'Filter window size (default {FLAT_ZONE_WINDOW}, '
             f'must match the downstream compute engine)')
    parser.add_argument('--threshold', type=float, default=FLAT_ZONE_VARIANCE,
        help=f'Variance threshold m² (default {FLAT_ZONE_VARIANCE}, '
             f'must match the downstream compute engine)')
    parser.add_argument('--strip-height', type=int, default=2000,
        help='Rows per processing strip (default 2000)')
    parser.add_argument('--boundaries', action='store_true',
        help='Also compute flat zone boundary mask from existing flat mask '
             '(run after initial flat mask computation)')

    args = parser.parse_args()

    if args.dem and args.output:
        dem_path    = args.dem
        output_path = args.output
    else:
        parser.error('Provide both --dem and --output')

    if args.boundaries:
        mask_path = output_path
        if not Path(mask_path).exists():
            print(f"[ERROR] Flat mask not found: {mask_path}")
            print(f"        Run without --boundaries first")
            sys.exit(1)
        stats = compute_flat_regions(
            flat_mask_path=mask_path,
            output_path=mask_path,
            strip_height=args.strip_height,
        )
        print(f"\nReduction: {stats['reduction_pct']:.1f}% fewer cache entries")
        return

    compute_flat_mask(
        dem_path=dem_path,
        output_path=output_path,
        window=args.window,
        threshold=args.threshold,
        strip_height=args.strip_height,
    )


if __name__ == '__main__':
    main()