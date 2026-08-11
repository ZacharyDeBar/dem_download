"""
tile_builder.py
━━━━━━━━━━━━━━━
Builds one tile end-to-end: download → gap-fill → water correction →
flat mask → boundary mask → meta.json → registry update.

Composes the existing pipeline code in build_dem_mosaic.py,
dem_download.py, dem_water_correction.py, repair_dem_gaps.py, and
precompute_flat_mask.py — each step is a thin wrapper that scopes
the existing function to a single 1° tile's bbox.

Public entry point:
    build_tile(tile_id, storage_root, ...) → BuildResult

CLI usage:
    python tile_builder.py build N45W110 [--storage-root PATH] [...]

Failure modes:
    Each step has its own validation gate. If a step fails, the tile
    is marked 'failed' in the registry with a reason string, and a
    BuildResult is returned with success=False. The on-disk artifacts
    that were partially built remain in the tile directory for
    inspection (we don't auto-clean failures — operator decides).

Validation gates:
    - Download: file opens with rasterio, has expected dimensions
    - Gap fill: no remaining nodata (except in synthesized ocean tiles)
    - Extent validation (Step 2b): the written DEM's pixel dimensions and
      geographic bounds must exactly match the tile's canonical 1x1 degree
      footprint (tile_validation.validate_tile_raster). Added after a
      real incident (2026-07-30): a truncated download slipped past the
      old Step 2 check (which only compared Affine transform coefficients,
      not array shape) and sat 'ready' in the registry for 6+ weeks missing
      its southern ~0.23deg, producing a silent flat-elevation-0 gap in any
      route that needed it.
    - Water correction: no negative elevations, polygon counts match
    - Flat mask: dimensions match DEM, flat percentage >= 1% (mask must
      trigger somewhere); a flat percentage > 99% only fails if ALSO
      corroborated by a near-zero whole-tile elevation range (< 2m) --
      genuinely flat terrain (real relief over a full 1° tile) is never
      flagged just for having low local (5x5 window) variance almost
      everywhere. This upper-bound check is skipped for a confirmed
      ocean-fallback tile (all-zero by design, not a broken download).
      Unconditionally, regardless of flat_pct: elevation range reading
      "n/a" (no valid/non-nodata pixels found at all) always fails --
      added after a real incident (2026-08-08) where a transient nodata-
      detection failure read the whole DEM as invalid, made flat_pct read
      100% for the wrong reason, and would have sat 'ready' with a
      garbage flat mask had the upper-bound check's `is not None` guard
      not been silently excluding exactly this case.
    - Boundary mask: dimensions match DEM, boundary <= flat
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from tile_id import (
    parse_tile_id, tile_bounds, TileBounds,
)
from tile_registry import (
    TileRegistry, TileEntry, load_registry, update_registry,
    tile_dir_rel, tile_file_rel, absolute_path,
    STATUS_READY, STATUS_PENDING, STATUS_FAILED,
)
from storage_manager import (
    StorageManager, estimate_tile_size_mb,
    POLICY_HARD, DEFAULT_CAP_GB,
)


logger = logging.getLogger(__name__)


# Thresholds for water-body diagnostics (Q3 from design doc)
WATER_ANOMALY_STD_M   = 5.0
WATER_ANOMALY_RANGE_M = 20.0


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ─────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────

@dataclass
class StepResult:
    name:        str
    success:     bool
    elapsed_s:   float
    detail:      Dict
    error:       Optional[str] = None


@dataclass
class BuildResult:
    tile_id:      str
    success:      bool
    storage_root: str
    elapsed_s:    float
    steps:        List[StepResult]
    final_size_mb: float = 0.0
    meta_path:    Optional[str] = None
    failed_step:  Optional[str] = None
    error_summary: Optional[str] = None


# ─────────────────────────────────────────────
# Helpers — file paths and resolution
# ─────────────────────────────────────────────

def _abs_paths_for_tile(storage_root: Path, tile_id: str) -> Dict[str, Path]:
    """Return absolute paths for all per-tile artifacts.

    'gap_mask', 'dem_precorrection', and 'water_stats' are only written
    when build_tile(..., keep_original=True) — see that flag's docstring.
    Their paths are always returned here so callers (e.g. visualize_tile.py)
    can check existence without duplicating the naming convention.
    """
    base = storage_root / 'tiles' / tile_id
    return {
        'tile_dir': base,
        'dem':      base / f'{tile_id}_dem.tif',
        'flat':     base / f'{tile_id}_flat.tif',
        'boundary': base / f'{tile_id}_boundary.tif',
        'water':    base / f'{tile_id}_water.geojson',
        'meta':     base / f'{tile_id}_meta.json',
        'gap_mask':          base / f'{tile_id}_gap_mask.tif',
        'dem_precorrection': base / f'{tile_id}_dem_precorrection.tif',
        'water_stats':       base / f'{tile_id}_water_stats.json',
    }


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute sha256 hex digest of a file, streaming to bound RAM."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def _dir_total_mb(directory: Path) -> float:
    total = 0
    for p in directory.rglob('*'):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


# ─────────────────────────────────────────────
# Step 1 — Download
# ─────────────────────────────────────────────

def _step_download(
    tile_id:      str,
    bounds:       TileBounds,
    downloads_dir: Path,
    source:       str = 'auto',
) -> Tuple[StepResult, Optional[Path]]:
    """
    Download the raw DEM tile via the existing dem_download cascade.
    Returns (StepResult, path_to_raw_tile or None on failure).

    On success, the path is to a file under downloads_dir whose
    geographic extent covers the 1° tile.
    """
    t0 = time.perf_counter()
    detail: Dict = {'source_requested': source}

    try:
        # Import here so a missing dep doesn't break module import
        from dem_download import download_tile

        lat_floor, lng_floor = parse_tile_id(tile_id)
        # dem_download.download_tile expects (lat, lng, output_dir, resolution)
        # where lat/lng are the SW corner integer coordinates.
        result = download_tile(
            lat=lat_floor,
            lng=lng_floor,
            output_dir=str(downloads_dir),
            resolution=source if source != 'auto' else 'best',
        )
        if result is None:
            return StepResult(
                name='download', success=False,
                elapsed_s=time.perf_counter() - t0,
                detail=detail,
                error=f'download_tile returned None — all sources exhausted',
            ), None

        raw_path, res_label = result
        raw_path = Path(raw_path)
        detail.update({
            'raw_path':   str(raw_path),
            'res_label':  res_label,
            'raw_mb':     _file_mb(raw_path),
        })

        # Validation: file opens cleanly with rasterio
        with rasterio.open(raw_path) as src:
            detail['raw_dimensions'] = [src.height, src.width]
            detail['raw_crs']        = str(src.crs)
            detail['raw_bounds']     = list(src.bounds)
            detail['raw_resolution_deg'] = [abs(src.res[0]), abs(src.res[1])]

        return StepResult(
            name='download', success=True,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
        ), raw_path

    except ImportError as e:
        return StepResult(
            name='download', success=False,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
            error=f'Could not import dem_download: {e}',
        ), None
    except Exception as e:
        return StepResult(
            name='download', success=False,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
            error=f'{type(e).__name__}: {e}',
        ), None


# ─────────────────────────────────────────────
# Step 2 — Crop raw to exact tile bbox + gap fill
# ─────────────────────────────────────────────

def _step_crop_and_fill_gaps(
    tile_id:       str,
    bounds:        TileBounds,
    raw_path:      Path,
    dem_out:       Path,
    keep_original: bool = False,
    gap_mask_out:  Optional[Path] = None,
) -> StepResult:
    """
    Crop the raw download to exactly the 1° tile bbox, fill any gaps
    from GLO-30, then reproject to a canonical EPSG:4326 grid and write
    to dem_out as a deflate-compressed GeoTIFF.

    keep_original: if True and gaps were found, also write a uint8
    gap_mask_out GeoTIFF (1 = was a gap pre-fill, 0 = wasn't) on the same
    canonical grid as dem_out, so a later diff/void-map tool can show
    exactly what the multi-source cascade filled in. Off by default —
    gap-filling is otherwise in-place (no "before" raster survives), and
    most callers don't need the extra artifact on disk.

    The raw download may extend slightly beyond the tile (3DEP tiles
    are 1°+ with overlap). Cropping to exactly the tile bbox keeps
    per-tile artifacts geographically aligned.

    OUTPUT CONTRACT (calibration #27):
      - CRS:        always EPSG:4326 (WGS84). Source 3DEP tiles ship
                    in EPSG:4269 (NAD83); the datum offset between
                    them in CONUS is sub-meter (irrelevant at 10m).
                    We standardize on 4326 because (a) existing user
                    mosaics are 4326, and (b) Copernicus GLO-30 is
                    4326, so VRT mosaicking across mixed sources is
                    consistent.
      - Transform:  exact origin = (tile_west, tile_north), pixel size
                    = (1° / dim) where `dim` is inherited from the
                    source raw tile dimensions (so 10m sources produce
                    ~10800x10800 tiles). This ensures fresh tiles
                    snap to the same global pixel grid as migrated
                    tiles, so VRT composition aligns.
    """
    t0 = time.perf_counter()
    detail: Dict = {}
    try:
        from rasterio.windows import from_bounds as window_from_bounds
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds as transform_from_bounds
        from rasterio.crs import CRS

        TARGET_CRS = CRS.from_epsg(4326)

        s, w, n, e = bounds.south, bounds.west, bounds.north, bounds.east
        with rasterio.open(raw_path) as src:
            src_crs = src.crs
            # Accept any geographic CRS — 3DEP tiles ship as EPSG:4269
            # (NAD83), Copernicus GLO-30 ships as EPSG:4326 (WGS84), and
            # both have lat/lng coordinates in decimal degrees.
            #
            # We DO refuse projected CRSes (UTM, web mercator, etc.)
            # because pixel-math semantics change — bounds are in metres
            # not degrees, transform.a is a different scale, etc.
            if not src_crs.is_geographic:
                return StepResult(
                    name='crop_and_fill', success=False,
                    elapsed_s=time.perf_counter() - t0,
                    detail=detail,
                    error=f'Raw tile CRS is {src_crs} (projected). '
                          f'Tile builder requires a geographic CRS '
                          f'(e.g. EPSG:4326 or EPSG:4269) so pixel '
                          f'coordinates are in degrees.',
                )
            detail['src_crs'] = str(src_crs)
            src_crs_str = src_crs.to_string()

            # Window covering the 1° tile within source coords
            win = window_from_bounds(w, s, e, n, transform=src.transform)
            # Round to integer pixel bounds for clean alignment
            win = win.round_offsets().round_lengths()
            if win.width <= 0 or win.height <= 0:
                return StepResult(
                    name='crop_and_fill', success=False,
                    elapsed_s=time.perf_counter() - t0,
                    detail=detail,
                    error=f'Source window has zero size for tile {tile_id} '
                          f'bbox ({s},{w},{n},{e})',
                )

            data = src.read(1, window=win).astype(np.float32)
            cropped_transform = src.window_transform(win)
            profile = src.profile.copy()
            nodata_value = src.nodata if src.nodata is not None else -9999.0

        detail['cropped_dimensions'] = list(data.shape)
        detail['cropped_mb_uncompressed'] = data.nbytes / (1024 * 1024)

        # Detect gaps (zero or nodata)
        nodata_mask = data <= (nodata_value + 1)
        zero_mask = (data == 0.0) & ~nodata_mask
        gap_mask = nodata_mask | zero_mask
        n_gaps = int(gap_mask.sum())
        detail['n_gaps_pre_fill'] = n_gaps

        # If gaps exist, fill from GLO-30 (still in source CRS — the
        # final reproject to canonical 4326 happens below)
        if n_gaps > 0:
            try:
                fill_info = _fill_gaps_from_glo30(
                    tile_id=tile_id,
                    bounds=bounds,
                    data=data,
                    gap_mask=gap_mask,
                    transform=cropped_transform,
                    dst_crs=src_crs_str,
                )
                detail.update(fill_info)
            except Exception as gap_err:
                detail['gap_fill_error'] = str(gap_err)
                detail['gaps_filled_from'] = None
        else:
            detail['gaps_filled_from'] = None
            detail['gaps_filled_count'] = 0

        # ── Canonical-grid reproject to EPSG:4326 ──────────────────
        # Compute the canonical destination transform:
        #   - origin at (tile_west, tile_north)
        #   - pixel size = 1° / cropped_dim (preserves source resolution)
        #   - dst shape matches source shape
        # This guarantees the output snaps to the global integer pixel
        # grid at the source's native resolution, which is required for
        # VRT compatibility across tile sources (migrated + fresh-built).
        # 10m GRID INVARIANT (calibration #27 amended): write EVERY tile on the
        # same 1/10800-degree lattice, regardless of source resolution, so the
        # route VRT is always grid-homogeneous. A GLO30 30m fallback (3600^2) is
        # upsampled here by the existing bilinear reproject below — smooth, no
        # new detail, which is correct for far-field terrain at this compute radius.
        # (These are 1-degree tiles, so this is exactly 10800x10800.)
        STD_GRID_RES_DEG = 1.0 / 10800.0
        dst_width  = int(round((e - w) / STD_GRID_RES_DEG))
        dst_height = int(round((n - s) / STD_GRID_RES_DEG))
        dst_transform = transform_from_bounds(
            west=w, south=s, east=e, north=n,
            width=dst_width, height=dst_height,
        )
        detail['dst_crs'] = 'EPSG:4326'
        detail['dst_transform_origin'] = [w, n]
        detail['dst_pixel_size_deg'] = abs(dst_transform.a)

        # Reproject the working buffer to the canonical grid.
        # Two scenarios:
        #   1. src_crs != 4326 (e.g. 3DEP 4269): full reproject with
        #      bilinear resampling. Sub-meter datum shift; precision
        #      loss is irrelevant at 10m.
        #   2. src_crs == 4326 (e.g. GLO-30 source): the dst transform
        #      may still differ from cropped_transform by a fraction
        #      of a pixel (because source padding shifts the origin
        #      from the canonical tile NW corner). Still reproject to
        #      snap to canonical grid — but the resampling is near-
        #      identity, so quality loss is negligible.
        if (src_crs_str != 'EPSG:4326' or cropped_transform != dst_transform
                or data.shape != (dst_height, dst_width)):
            dst_data = np.full(
                (dst_height, dst_width), nodata_value, dtype=np.float32,
            )
            reproject(
                source=data,
                destination=dst_data,
                src_transform=cropped_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                src_nodata=nodata_value,
                dst_nodata=nodata_value,
                resampling=Resampling.bilinear,
            )
            data = dst_data
            detail['reprojected'] = True
        else:
            detail['reprojected'] = False

        # Optional: persist where the gaps were (pre-fill), reprojected
        # onto the same canonical grid as dem_out so it overlays exactly.
        # Nearest-neighbor resampling — gap_mask is categorical (0/1),
        # bilinear would smear it into fractional values at the edges.
        if keep_original and gap_mask_out is not None and n_gaps > 0:
            gap_mask_dst = np.zeros((dst_height, dst_width), dtype=np.uint8)
            reproject(
                source=gap_mask.astype(np.uint8),
                destination=gap_mask_dst,
                src_transform=cropped_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.nearest,
            )
            gap_profile = profile.copy()
            gap_profile.update({
                'driver': 'GTiff', 'height': dst_height, 'width': dst_width,
                'transform': dst_transform, 'crs': TARGET_CRS,
                'dtype': 'uint8', 'count': 1, 'nodata': None,
                'compress': 'deflate', 'predictor': 2,
                'tiled': True, 'blockxsize': 512, 'blockysize': 512,
            })
            gap_mask_out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(gap_mask_out, 'w', **gap_profile) as dst:
                dst.write(gap_mask_dst, 1)
            detail['gap_mask_written'] = True

        # Write cropped+filled+reprojected tile to dem_out with the
        # canonical EPSG:4326 transform.
        out_profile = profile.copy()
        out_profile.update({
            'driver':    'GTiff',
            'height':    data.shape[0],
            'width':     data.shape[1],
            'transform': dst_transform,
            'crs':       TARGET_CRS,
            'dtype':     'float32',
            'nodata':    nodata_value,
            'compress':  'deflate',
            'predictor': 2,
            'tiled':     True,
            'blockxsize': 512,
            'blockysize': 512,
        })
        dem_out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dem_out, 'w', **out_profile) as dst:
            dst.write(data, 1)
        detail['dem_out_mb'] = _file_mb(dem_out)

        return StepResult(
            name='crop_and_fill', success=True,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
        )

    except Exception as e:
        return StepResult(
            name='crop_and_fill', success=False,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
            error=f'{type(e).__name__}: {e}\n{traceback.format_exc()}',
        )


def _fill_gaps_from_glo30(
    tile_id:    str,
    bounds:     TileBounds,
    data:       np.ndarray,
    gap_mask:   np.ndarray,
    transform,
    dst_crs:    str = 'EPSG:4326',
) -> Dict:
    """
    Fill gaps in `data` (modified in place) using GLO-30 elevation.
    Returns a detail dict describing the fill.

    Reuses repair_dem_gaps.fetch_glo30_mosaic + fill_gaps as a unit.
    Scoped to just this tile's bbox.

    Args:
        dst_crs: CRS of the destination grid. GLO-30 is in EPSG:4326,
                 but if the host DEM is in EPSG:4269 we project to that
                 to avoid an unnecessary additional reprojection later.
                 The datum offset is sub-meter, so this is a no-op in
                 practice — but using the correct dst_crs keeps the
                 reproject call mathematically consistent.
    """
    from repair_dem_gaps import fetch_glo30_mosaic
    from rasterio.warp import reproject, Resampling

    # Need GLO-30 covering the tile's 1° area.
    # repair_dem_gaps.get_required_glo30_tiles uses bounds.bottom/.left/.top/.right
    # — we'll pass an equivalent shim object.
    class _BoundsShim:
        def __init__(self, b: TileBounds):
            self.bottom = b.south
            self.left   = b.west
            self.top    = b.north
            self.right  = b.east

    from repair_dem_gaps import get_required_glo30_tiles
    tiles = get_required_glo30_tiles(_BoundsShim(bounds), verbose=False)
    glo30_data, glo30_transform, _ = fetch_glo30_mosaic(tiles, verbose=False)

    # Reproject GLO-30 onto our tile grid. GLO-30 source is in
    # EPSG:4326; the destination is whatever CRS our host DEM uses
    # (typically EPSG:4269 for 3DEP tiles or 4326 for GLO-30-only).
    glo30_resampled = np.zeros(data.shape, dtype=np.float32)
    reproject(
        source=glo30_data,
        destination=glo30_resampled,
        src_transform=glo30_transform,
        src_crs='EPSG:4326',
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    n_gaps = int(gap_mask.sum())
    data[gap_mask] = glo30_resampled[gap_mask]
    return {
        'gaps_filled_from':  'GLO30',
        'gaps_filled_count': n_gaps,
        'glo30_tiles_used':  [f'({t[0]},{t[1]})' for t in tiles],
    }


# ─────────────────────────────────────────────
# Step 3 — Water correction + diagnostics
# ─────────────────────────────────────────────

def _step_water_correction(
    tile_id:           str,
    bounds:            TileBounds,
    dem_path:          Path,
    water_out:         Path,
    source:            str = 'auto',
    keep_original:     bool = False,
    precorrection_out: Optional[Path] = None,
    stats_out:         Optional[Path] = None,
) -> StepResult:
    """
    Apply water-body correction to the tile's DEM (in place) and
    capture intra-water-body elevation diagnostics per design Q3.

    `source` is 'nhd' (US), 'osm' (global), or 'auto' (NHD if tile
    is in US, else OSM).

    keep_original: if True, copy dem_path to precorrection_out before
    correcting, and write correct_dem_water_bodies's per-polygon
    elevation_changes (surface_elev_m, original_mean_m, change_m) to
    stats_out. Off by default — correction is otherwise in-place (no
    "before" raster survives, and the returned stats are discarded).
    """
    t0 = time.perf_counter()
    detail: Dict = {}

    try:
        from dem_water_correction import (
            fetch_water_bodies_nhd, fetch_water_bodies_osm,
            fetch_ocean_polygons_osm, correct_dem_water_bodies,
        )

        s, w, n, e = bounds.south, bounds.west, bounds.north, bounds.east
        water_bounds = (s, w, n, e)

        # Pick source — same heuristic the existing code uses
        if source == 'auto':
            # US lower 48 + AK + HI approximations
            is_us = (24 <= s <= 72) and (w >= -180) and (e <= -50)
            use_source = 'nhd' if is_us else 'osm'
        else:
            use_source = source
        detail['water_source'] = use_source

        # Fetch
        if use_source == 'nhd':
            polys = fetch_water_bodies_nhd(water_bounds)
        else:
            polys = fetch_water_bodies_osm(water_bounds, include_ocean=False)
            # Add ocean separately if applicable
            try:
                ocean = fetch_ocean_polygons_osm(water_bounds)
                polys.extend(ocean)
            except Exception as e:
                detail['ocean_fetch_warn'] = str(e)
        detail['water_polygons_count'] = len(polys)

        # Save the fetched polygons for traceability
        with open(water_out, 'w') as f:
            json.dump({
                'type': 'FeatureCollection',
                'features': [
                    {'type': 'Feature', 'geometry': p,
                     'properties': p.get('properties', {})}
                    for p in polys
                ],
            }, f)

        # Capture pre-correction elevation statistics per polygon
        # for the cross-tile diagnostics (design Q3)
        diagnostics = _compute_water_diagnostics(
            dem_path=dem_path,
            polygons=polys,
            tile_bounds=bounds,
        )
        detail['water_correction_diagnostics'] = diagnostics

        # Apply correction (modifies dem_path in place)
        if polys:
            if keep_original and precorrection_out is not None:
                precorrection_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dem_path, precorrection_out)

            stats = correct_dem_water_bodies(
                input_path=str(dem_path),
                output_path=str(dem_path),
                water_polygons=polys,
                buffer_pixels=3,
                shore_percentile=10.0,
                batch_size=500,
                verbose=False,
            )
            if keep_original and stats_out is not None:
                stats_out.parent.mkdir(parents=True, exist_ok=True)
                with open(stats_out, 'w') as f:
                    json.dump({
                        'corrected_bodies':   stats.get('corrected_bodies'),
                        'skipped_small':      stats.get('skipped_small'),
                        'total_pixels_fixed': stats.get('total_pixels_fixed'),
                        'elevation_changes':  stats.get('elevation_changes', []),
                    }, f, indent=2)

        return StepResult(
            name='water_correction', success=True,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
        )

    except Exception as e:
        return StepResult(
            name='water_correction', success=False,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
            error=f'{type(e).__name__}: {e}\n{traceback.format_exc()}',
        )


def _compute_water_diagnostics(
    dem_path:    Path,
    polygons:    list,
    tile_bounds: TileBounds,
) -> Dict:
    """
    For each water polygon, measure pre-correction elevation stats
    within the polygon's footprint. Flag polygons with high variance
    (likely cross-tile water bodies with inconsistent surface heights).

    Returns a dict with summary + list of flagged polygons.
    """
    import rasterio.features as rf

    diagnostics = {
        'anomaly_threshold_std_m':   WATER_ANOMALY_STD_M,
        'anomaly_threshold_range_m': WATER_ANOMALY_RANGE_M,
        'flagged_polygons':          [],
        'n_polygons_inspected':      0,
    }

    if not polygons:
        return diagnostics

    s, w, n, e = tile_bounds.south, tile_bounds.west, tile_bounds.north, tile_bounds.east

    with rasterio.open(dem_path) as src:
        elev    = src.read(1).astype(np.float32)
        transform = src.transform
        h, ww   = src.height, src.width
        nodata = src.nodata if src.nodata is not None else -9999.0

    for i, poly in enumerate(polygons):
        coords = poly.get('coordinates', [])
        if not coords:
            continue
        try:
            mask = rf.rasterize(
                [({'type': 'Polygon', 'coordinates': coords}, 1)],
                out_shape=(h, ww), transform=transform,
                fill=0, dtype=np.uint8,
            ).astype(bool)
        except Exception:
            continue

        pixel_count = int(mask.sum())
        if pixel_count == 0:
            continue
        diagnostics['n_polygons_inspected'] += 1

        inside = elev[mask]
        valid = inside[(inside > nodata + 1) & (inside < 9000)]
        if len(valid) == 0:
            continue

        elev_min = float(valid.min())
        elev_max = float(valid.max())
        elev_std = float(valid.std())
        elev_range = elev_max - elev_min

        # Cross-tile detection: polygon's bounding longitudes/latitudes
        # extend beyond this tile's bounds
        try:
            # Flatten coordinates (handle Polygon or MultiPolygon)
            geom_type = poly.get('type', 'Polygon')
            if geom_type == 'MultiPolygon':
                all_coords = [pt for polygon in coords
                                 for ring in polygon
                                 for pt in ring]
            else:
                ring = coords[0]
                all_coords = ring
            lngs = [c[0] for c in all_coords]
            lats = [c[1] for c in all_coords]
            cross_tile = (min(lngs) < w or max(lngs) > e or
                          min(lats) < s or max(lats) > n)
        except (IndexError, TypeError):
            cross_tile = False

        # Flag if anomalous OR cross-tile
        if elev_std > WATER_ANOMALY_STD_M or elev_range > WATER_ANOMALY_RANGE_M:
            diagnostics['flagged_polygons'].append({
                'polygon_index':            i,
                'name':                     poly.get('properties', {}).get('name', ''),
                'pixel_count':              pixel_count,
                'pre_correction_elev_min_m': round(elev_min, 2),
                'pre_correction_elev_max_m': round(elev_max, 2),
                'pre_correction_elev_std_m': round(elev_std, 2),
                'cross_tile':               cross_tile,
                'reason': (
                    f'std={elev_std:.2f}m > {WATER_ANOMALY_STD_M}m'
                    if elev_std > WATER_ANOMALY_STD_M
                    else f'range={elev_range:.2f}m > {WATER_ANOMALY_RANGE_M}m'
                ),
            })

    diagnostics['n_polygons_flagged'] = len(diagnostics['flagged_polygons'])
    return diagnostics


# ─────────────────────────────────────────────
# Step 4 — Flat mask + boundary mask
# ─────────────────────────────────────────────

def _step_flat_mask(
    tile_id:    str,
    dem_path:   Path,
    flat_out:   Path,
    skip_degenerate_check: bool = False,
) -> StepResult:
    """Compute flat zone mask via existing precompute_flat_mask.compute_flat_mask.

    skip_degenerate_check: set True only for a confirmed ocean-fallback tile
    (dem_download._create_ocean_tile's intentional all-zero raster) --
    otherwise that tile's 100% flat / 0m elevation range legitimately trips
    the degenerate-DEM gate below, which exists to catch a genuinely broken
    download, not an intentional zero-fill.
    """
    t0 = time.perf_counter()
    detail: Dict = {}
    try:
        from precompute_flat_mask import compute_flat_mask
        stats = compute_flat_mask(
            dem_path=str(dem_path),
            output_path=str(flat_out),
            strip_height=2000,
        )
        detail.update({
            'flat_pct':      stats.get('flat_pct'),
            'flat_mask_mb':  stats.get('mask_mb'),
            'window':        stats.get('window'),
            'threshold':     stats.get('threshold'),
            'elev_range_m':  stats.get('elev_range_m'),
        })
        # Validation, three distinct failure modes:
        #   - elev_range_m is None: see below, checked first.
        #   - flat_pct < 1%: the mask never triggers anywhere -- suspicious
        #     in its own right (a broken variance computation, e.g. reading
        #     garbage/noise, would look like this).
        #   - flat_pct near 100% used to be flagged outright (originally
        #     [5%,99%], loosened once already to [1%,99%]), but genuinely
        #     flat terrain (West Texas/Llano Estacado plains tiles hit
        #     99.0-99.9%) kept tripping it -- real geography, not a bug.
        #     The actual signature of a degenerate/placeholder DEM (a
        #     corrupted download, or an all-nodata tile filled with one
        #     constant value) is near-ZERO elevation RANGE across the
        #     *whole* tile, not just low local (5x5 window) variance --
        #     real terrain, even the flattest plains, still has tens of
        #     metres of regional relief over a full 1° tile. So the upper
        #     bound now only fires when high flat_pct is CORROBORATED by a
        #     near-flat whole-tile elevation range.
        flat_pct = detail.get('flat_pct') or 0.0
        elev_range_m = detail.get('elev_range_m')
        # elev_range_m is None means compute_flat_mask found NO valid
        # (non-nodata) pixels in the entire DEM -- distinct from, and
        # strictly worse than, "degenerate_range" below (range < 2m, i.e.
        # valid pixels that just happen to be near-constant). A real DEM
        # always has SOME valid pixels; even the intentional all-zero
        # ocean-fallback tile does (0.0 is valid data there, a real
        # nodata sentinel marks the rest -- see compute_flat_mask). None
        # here means the valid-pixel check itself never fired, which is
        # the exact signature that slipped a corrupted flat mask through
        # as 'ready' in production (2026-08-08): flat_pct read 100.0%
        # because every strip's variance was computed over an array of
        # nodata-masked zeros, not because the terrain was flat. Checked
        # unconditionally -- not behind skip_degenerate_check -- because
        # no legitimate tile, ocean-fallback included, produces this.
        if elev_range_m is None:
            return StepResult(
                name='flat_mask', success=False,
                elapsed_s=time.perf_counter() - t0,
                detail=detail,
                error='Elevation range is n/a (no valid, non-nodata pixels '
                      'found while computing the flat mask). A real DEM '
                      'always has some valid pixels -- this means the '
                      'nodata/valid-pixel check itself is broken for this '
                      'file, not that the tile is genuinely flat.',
            )
        degenerate_range = elev_range_m < 2.0
        if flat_pct < 1.0:
            return StepResult(
                name='flat_mask', success=False,
                elapsed_s=time.perf_counter() - t0,
                detail=detail,
                error=f'Flat percentage {flat_pct}% is outside sane range '
                      f'(below the 1% floor) — possible mask computation '
                      f'issue (mask never triggers).',
            )
        if flat_pct > 99.0 and degenerate_range and not skip_degenerate_check:
            return StepResult(
                name='flat_mask', success=False,
                elapsed_s=time.perf_counter() - t0,
                detail=detail,
                error=f'Flat percentage {flat_pct}% with whole-tile '
                      f'elevation range {elev_range_m}m — DEM looks '
                      f'degenerate/constant-value, not just genuinely flat '
                      f'terrain (real relief over a full 1° tile is never '
                      f'this small).',
            )
        return StepResult(
            name='flat_mask', success=True,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
        )
    except Exception as e:
        return StepResult(
            name='flat_mask', success=False,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
            error=f'{type(e).__name__}: {e}\n{traceback.format_exc()}',
        )


def _step_boundary_mask(
    tile_id:       str,
    flat_path:     Path,
    boundary_out:  Path,
) -> StepResult:
    """Compute boundary mask via existing precompute_flat_mask.compute_flat_regions."""
    t0 = time.perf_counter()
    detail: Dict = {}
    try:
        from precompute_flat_mask import compute_flat_regions
        # compute_flat_regions writes to {output_path}_boundary.tif —
        # so we need to pass a "base" that yields the right name.
        # The function does: output_path.replace('.tif', '_boundary.tif')
        # We want our explicit boundary_out path. The easiest path:
        # pass a path whose .replace gives boundary_out.
        base_path = str(boundary_out).replace('_boundary.tif', '.tif')
        stats = compute_flat_regions(
            flat_mask_path=str(flat_path),
            output_path=base_path,
            strip_height=2000,
        )
        detail.update({
            'boundary_pct':    stats.get('boundary_pct'),
            'boundary_cells':  stats.get('boundary_cells'),
            'flat_cells':      stats.get('flat_cells'),
            'reduction_pct':   stats.get('reduction_pct'),
            'boundary_mb':     stats.get('boundary_mb'),
        })
        # Validation: boundary cells should be ≤ flat cells (subset relationship)
        if (stats.get('boundary_cells', 0) > stats.get('flat_cells', 1)):
            return StepResult(
                name='boundary_mask', success=False,
                elapsed_s=time.perf_counter() - t0,
                detail=detail,
                error='Boundary mask has more cells than flat mask — invariant violated.',
            )
        return StepResult(
            name='boundary_mask', success=True,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
        )
    except Exception as e:
        return StepResult(
            name='boundary_mask', success=False,
            elapsed_s=time.perf_counter() - t0,
            detail=detail,
            error=f'{type(e).__name__}: {e}\n{traceback.format_exc()}',
        )


# ─────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────

def build_tile(
    tile_id:        str,
    storage_root:   Path,
    source:         str   = 'auto',
    water_source:   str   = 'auto',
    cap_gb:         float = DEFAULT_CAP_GB,
    cap_policy:     str   = POLICY_HARD,
    protected_tiles: Optional[set] = None,
    skip_if_ready:   bool = True,
    progress_log:    bool = True,
    keep_original:   bool = False,
) -> BuildResult:
    """
    Build a single tile end-to-end. Top-level entry point.

    Args:
        tile_id:        e.g. 'N45W110'
        storage_root:   absolute Path to storage root, e.g. /data — tiles
                        themselves land under <storage_root>/tiles/
        source:         DEM source preference ('auto', '3DEP_10', 'GLO30')
        water_source:   water polygon source ('auto', 'nhd', 'osm')
        cap_gb:         storage cap
        cap_policy:     'hard' | 'lru' | 'warn'
        protected_tiles: tile IDs that must not be evicted under LRU
        skip_if_ready:  if True and the tile is already 'ready' in the
                        registry, do nothing and return success
        progress_log:   if True, print per-step progress to stdout
        keep_original:  if True, also write the pre-gap-fill gap mask,
                        the pre-water-correction DEM, and the water
                        correction stats (see _step_crop_and_fill_gaps
                        and _step_water_correction) — extra artifacts
                        that visualize_tile.py uses for its diff panels.
                        Off by default: extra disk space and I/O for
                        artifacts most callers don't need.

    Returns:
        BuildResult with per-step diagnostics.
    """
    storage_root = Path(storage_root)
    t_total = time.perf_counter()
    result = BuildResult(
        tile_id=tile_id,
        success=False,
        storage_root=str(storage_root),
        elapsed_s=0.0,
        steps=[],
    )

    def _log(msg: str):
        if progress_log:
            print(msg, flush=True)

    # ── Validate tile_id and set up paths ────────────────────────
    try:
        bounds = tile_bounds(tile_id)
    except ValueError as e:
        result.error_summary = f'Invalid tile_id {tile_id!r}: {e}'
        result.elapsed_s = time.perf_counter() - t_total
        return result

    paths = _abs_paths_for_tile(storage_root, tile_id)
    paths['tile_dir'].mkdir(parents=True, exist_ok=True)
    downloads_dir = storage_root / 'downloads'
    downloads_dir.mkdir(parents=True, exist_ok=True)

    _log(f"━━━ Building tile {tile_id} ━━━")
    _log(f"  bounds:       {bounds.as_tuple()}")
    _log(f"  storage_root: {storage_root}")
    _log(f"  output_dir:   {paths['tile_dir']}")

    # ── Load registry, check for skip, check capacity ────────────
    registry = load_registry(storage_root)
    existing = registry.get(tile_id)
    if skip_if_ready and existing and existing.status == STATUS_READY:
        _log(f"  Tile already ready in registry — skipping.")
        result.success = True
        result.meta_path = existing.meta_file
        result.final_size_mb = existing.size_mb
        result.elapsed_s = time.perf_counter() - t_total
        return result

    sm = StorageManager(
        registry=registry, cap_gb=cap_gb,
        policy=cap_policy, protected_tiles=protected_tiles,
    )
    estimate = estimate_tile_size_mb(source)
    cap_check = sm.check_before_build(tile_id, estimate_mb=estimate)
    if not cap_check.allowed:
        result.error_summary = cap_check.refusal_reason
        result.failed_step = 'capacity_check'
        result.steps.append(StepResult(
            name='capacity_check', success=False, elapsed_s=0.0,
            detail={'used_mb': cap_check.used_mb, 'cap_mb': cap_check.cap_mb,
                    'evicted': cap_check.evicted},
            error=cap_check.refusal_reason,
        ))
        # This branch returns before Step 1, so unlike _finalize_failed
        # (which logs step failures below), nothing gets printed unless
        # we do it here explicitly — this was the actual gap that made
        # capacity refusals invisible in the live batch console output.
        _log(f"\n✗ capacity_check FAILED:")
        _log(f"  {cap_check.refusal_reason}")
        # Note: unlike _finalize_failed, this tile is NOT marked 'failed'
        # in the registry — mark_pending() hasn't run yet at this point,
        # so there's no registry entry to correct. Next attempt reassesses
        # cleanly rather than getting stuck on a stale 'pending'/'failed'.
        result.elapsed_s = time.perf_counter() - t_total
        return result
    if cap_check.warning:
        _log(f"  ⚠ {cap_check.warning}")

    # ── Mark pending in registry & save (so concurrent observers see it) ─
    # update_registry() reloads fresh under a cross-process lock, so a
    # sibling tile build running concurrently (same route or a different
    # one) never has its own pending/ready mark clobbered by this save.
    update_registry(storage_root, lambda r: r.mark_pending(tile_id))

    # ── Step 1: Download ──────────────────────────────────────────
    _log(f"\n[1/5] Downloading raw DEM...")
    sr1, raw_path = _step_download(
        tile_id=tile_id, bounds=bounds,
        downloads_dir=downloads_dir, source=source,
    )
    result.steps.append(sr1)
    if not sr1.success:
        return _finalize_failed(result, sr1, registry, storage_root,
                                 t_total, _log)
    _log(f"  Downloaded {raw_path.name} in {sr1.elapsed_s:.1f}s")
    is_ocean_fallback = sr1.detail.get('res_label') == 'ocean0m'

    # ── Step 2: Crop + gap fill ───────────────────────────────────
    _log(f"\n[2/5] Cropping to tile bbox + filling gaps...")
    sr2 = _step_crop_and_fill_gaps(
        tile_id=tile_id, bounds=bounds,
        raw_path=raw_path, dem_out=paths['dem'],
        keep_original=keep_original, gap_mask_out=paths['gap_mask'],
    )
    result.steps.append(sr2)
    if not sr2.success:
        return _finalize_failed(result, sr2, registry, storage_root,
                                 t_total, _log)
    n_gaps_filled = sr2.detail.get('gaps_filled_count', 0)
    _log(f"  DEM written ({_file_mb(paths['dem']):.1f}MB, "
         f"{n_gaps_filled} gap pixels filled) in {sr2.elapsed_s:.1f}s")

    # ── Step 2b: Extent validation ────────────────────────────────
    # Real incident (2026-07-30): a truncated download slipped past Step 2
    # because its origin/pixel-size matched the canonical grid closely
    # enough that the reproject condition (now fixed above to also check
    # array shape) didn't trigger a full-canvas reproject -- the tile sat
    # 'ready' in the registry for 6+ weeks missing its southern ~0.23deg.
    # This gate catches that defect class at creation time regardless of
    # how it happens to slip past Step 2's own internal checks.
    from tile_validation import validate_tile_raster
    extent_check = validate_tile_raster(paths['dem'], tile_id)
    if not extent_check.ok:
        sr2b = StepResult(
            name='extent_validation', success=False,
            elapsed_s=0.0, detail={'reason': extent_check.reason},
            error=f'DEM extent validation failed: {extent_check.reason}',
        )
        result.steps.append(sr2b)
        return _finalize_failed(result, sr2b, registry, storage_root,
                                 t_total, _log)

    # ── Step 3: Water correction ──────────────────────────────────
    _log(f"\n[3/5] Applying water correction...")
    sr3 = _step_water_correction(
        tile_id=tile_id, bounds=bounds,
        dem_path=paths['dem'], water_out=paths['water'],
        source=water_source,
        keep_original=keep_original,
        precorrection_out=paths['dem_precorrection'],
        stats_out=paths['water_stats'],
    )
    result.steps.append(sr3)
    if not sr3.success:
        return _finalize_failed(result, sr3, registry, storage_root,
                                 t_total, _log)
    diag = sr3.detail.get('water_correction_diagnostics', {})
    _log(f"  Corrected {sr3.detail.get('water_polygons_count', 0)} water "
         f"bodies "
         f"({diag.get('n_polygons_flagged', 0)} flagged) "
         f"in {sr3.elapsed_s:.1f}s")

    # ── Step 4: Flat mask ─────────────────────────────────────────
    _log(f"\n[4/5] Computing flat mask...")
    sr4 = _step_flat_mask(
        tile_id=tile_id, dem_path=paths['dem'], flat_out=paths['flat'],
        skip_degenerate_check=is_ocean_fallback,
    )
    result.steps.append(sr4)
    if not sr4.success:
        return _finalize_failed(result, sr4, registry, storage_root,
                                 t_total, _log)
    _log(f"  Flat: {sr4.detail.get('flat_pct'):.1f}% "
         f"({sr4.detail.get('flat_mask_mb'):.1f}MB) "
         f"in {sr4.elapsed_s:.1f}s")

    # ── Step 5: Boundary mask ─────────────────────────────────────
    _log(f"\n[5/5] Computing boundary mask...")
    sr5 = _step_boundary_mask(
        tile_id=tile_id, flat_path=paths['flat'],
        boundary_out=paths['boundary'],
    )
    result.steps.append(sr5)
    if not sr5.success:
        return _finalize_failed(result, sr5, registry, storage_root,
                                 t_total, _log)
    _log(f"  Boundary: {sr5.detail.get('boundary_pct'):.2f}% "
         f"({sr5.detail.get('boundary_mb'):.1f}MB) "
         f"in {sr5.elapsed_s:.1f}s")

    # ── Write meta.json ───────────────────────────────────────────
    final_size_mb = _dir_total_mb(paths['tile_dir'])
    meta = _build_meta(
        tile_id=tile_id, bounds=bounds,
        paths=paths, steps=result.steps,
        final_size_mb=final_size_mb,
        source_label=sr1.detail.get('res_label', 'unknown'),
        elapsed_s=time.perf_counter() - t_total,
    )
    with open(paths['meta'], 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # ── Update registry ───────────────────────────────────────────
    update_registry(storage_root, lambda r: r.mark_ready(
        tile_id=tile_id,
        dem_file=str(paths['dem'].relative_to(storage_root).as_posix()),
        flat_file=str(paths['flat'].relative_to(storage_root).as_posix()),
        boundary_file=str(paths['boundary'].relative_to(storage_root).as_posix()),
        meta_file=str(paths['meta'].relative_to(storage_root).as_posix()),
        size_mb=final_size_mb,
    ))

    # ── Clean raw downloads (optional, per design) ────────────────
    if raw_path and raw_path.exists():
        try:
            raw_path.unlink()
        except OSError:
            pass

    result.success = True
    result.elapsed_s = time.perf_counter() - t_total
    result.final_size_mb = final_size_mb
    result.meta_path = str(paths['meta'])
    _log(f"\n━━━ Tile {tile_id} READY ({final_size_mb:.1f}MB total) "
         f"in {result.elapsed_s:.1f}s ━━━")
    return result


def _finalize_failed(
    result:       BuildResult,
    failed_step:  StepResult,
    registry:     TileRegistry,
    storage_root: Path,
    t_total:      float,
    log_fn,
) -> BuildResult:
    """Helper: update registry to 'failed' and return the BuildResult.

    `registry` is unused for the write (kept in the signature to avoid
    touching every call site) — the mark is applied via update_registry()
    against a fresh, lock-protected load so a concurrent sibling tile
    build's registry entry is never clobbered by this save.
    """
    result.failed_step = failed_step.name
    result.error_summary = failed_step.error
    result.elapsed_s = time.perf_counter() - t_total
    log_fn(f"\n✗ Step {failed_step.name} FAILED:")
    log_fn(f"  {failed_step.error}")
    log_fn(f"  Tile {result.tile_id} marked FAILED in registry.")
    update_registry(storage_root, lambda r: r.mark_failed(
        result.tile_id, f'{failed_step.name}: {failed_step.error}'))
    return result


def _build_meta(
    tile_id:       str,
    bounds:        TileBounds,
    paths:         Dict[str, Path],
    steps:         List[StepResult],
    final_size_mb: float,
    source_label:  str,
    elapsed_s:     float,
) -> Dict:
    """Construct the per-tile meta.json content."""
    # Find the relevant step results
    step_by_name = {s.name: s for s in steps}
    download = step_by_name.get('download', None)
    crop_fill = step_by_name.get('crop_and_fill', None)
    water = step_by_name.get('water_correction', None)
    flat = step_by_name.get('flat_mask', None)
    boundary = step_by_name.get('boundary_mask', None)

    # DEM dimensions from the cropped file
    dem_dimensions = None
    if paths['dem'].exists():
        with rasterio.open(paths['dem']) as src:
            dem_dimensions = [src.height, src.width]

    return {
        'tile_id': tile_id,
        'bounds': {
            'south': bounds.south, 'west':  bounds.west,
            'north': bounds.north, 'east':  bounds.east,
        },
        'resolution_m':        10 if '10' in source_label else 30,   # native (kept for back-compat)
        'grid_resolution_m':   10,                                   # always — enforced lattice
        'native_resolution_m': 10 if '10' in source_label else 30,   # source truth
        'upsampled_from_native': ('10' not in source_label),         # True for GLO30-on-10m-grid
        'source':              source_label,
        'source_url':   download.detail.get('raw_path', '') if download else '',
        'dem_dimensions': dem_dimensions,
        'dem_size_mb':       round(_file_mb(paths['dem']), 2),
        'flat_mask_size_mb': round(_file_mb(paths['flat']), 2),
        'boundary_mask_size_mb': round(_file_mb(paths['boundary']), 2),
        'water_polygons_count':  water.detail.get('water_polygons_count', 0) if water else 0,
        'water_source':          water.detail.get('water_source', '') if water else '',
        'water_correction_diagnostics':
            water.detail.get('water_correction_diagnostics', {}) if water else {},
        'build_timestamp_utc':   _utcnow_iso(),
        'build_elapsed_s':       round(elapsed_s, 2),
        'checksums': {
            'dem':      _sha256_of_file(paths['dem'])      if paths['dem'].exists() else None,
            'flat':     _sha256_of_file(paths['flat'])     if paths['flat'].exists() else None,
            'boundary': _sha256_of_file(paths['boundary']) if paths['boundary'].exists() else None,
        },
        'status': 'ready',
        'last_accessed_utc': _utcnow_iso(),
        'gaps_filled_from':  crop_fill.detail.get('gaps_filled_from') if crop_fill else None,
        'gaps_filled_count': crop_fill.detail.get('gaps_filled_count', 0) if crop_fill else 0,
        'validation_passed': True,
        'warnings': [],
        'final_size_mb': round(final_size_mb, 2),
        'step_summary': [
            {'name': s.name, 'success': s.success,
             'elapsed_s': round(s.elapsed_s, 2)}
            for s in steps
        ],
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main(argv=None):
    import argparse
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        prog='tile_builder.py',
        description='Build a single DEM tile end-to-end.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p_build = sub.add_parser('build', help='Build a tile')
    p_build.add_argument('tile_id', help='Tile ID like N45W110')
    p_build.add_argument('--storage-root', default='data',
        help='Storage root path — tiles land under <storage-root>/tiles/ '
             '(default: data)')
    p_build.add_argument('--source', default='auto',
        choices=['auto', '3DEP_10', 'GLO30'],
        help='DEM source preference (default auto)')
    p_build.add_argument('--water-source', default='auto',
        choices=['auto', 'nhd', 'osm'],
        help='Water polygon source (default auto)')
    p_build.add_argument('--cap-gb', type=float, default=DEFAULT_CAP_GB,
        help=f'Storage cap in GB (default {DEFAULT_CAP_GB})')
    p_build.add_argument('--cap-policy', default=POLICY_HARD,
        choices=['hard', 'lru', 'warn'],
        help='What to do when cap would be exceeded (default hard)')
    p_build.add_argument('--force', action='store_true',
        help='Rebuild even if tile is already ready in registry')
    p_build.add_argument('--keep-original', action='store_true',
        help='Also save pre-gap-fill gap mask, pre-water-correction DEM, '
             'and water correction stats — inputs for visualize_tile.py\'s '
             'diff panels (default: off, saves disk space)')

    args = parser.parse_args(argv)

    if args.command == 'build':
        result = build_tile(
            tile_id=args.tile_id,
            storage_root=Path(args.storage_root),
            source=args.source,
            water_source=args.water_source,
            cap_gb=args.cap_gb,
            cap_policy=args.cap_policy,
            skip_if_ready=not args.force,
            keep_original=args.keep_original,
        )
        if not result.success:
            print(f"\nBUILD FAILED: {result.error_summary}",
                  file=sys.stderr)
            sys.exit(1)
        sys.exit(0)


if __name__ == '__main__':
    main()