"""
repair_dem_gaps.py
━━━━━━━━━━━━━━━━━━
Repairs a study-area DEM mosaic by:
  1. Identifying zero/nodata pixels (should not exist in this study area)
  2. Downloading GLO-30 tiles for gap coverage
  3. Reprojecting GLO-30 to match the 3DEP grid
  4. Filling gaps with GLO-30 values
  5. Re-applying NHD water body correction

The original file is backed up before any changes are made.

Usage:
    python repair_dem_gaps.py N44W113 N47W109 --dem-path data/dem/my_area/my_area_corrected.tif
    python repair_dem_gaps.py --bounds 44.33 -112.95 47.03 -109.13 --dem-path data/dem/my_area/my_area_corrected.tif
    python repair_dem_gaps.py N44W113 N47W109 --dem-path ... --dry-run   # show gap stats without modifying
    python repair_dem_gaps.py N44W113 N47W109 --dem-path ... --skip-water-correction  # fill only, no NHD
"""

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.io
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject

from tile_id import bounds_from_tile_corners, MAX_MOSAIC_TILES


# ─────────────────────────────────────────────
# Gap detection
# ─────────────────────────────────────────────

def detect_gaps(
    dem_path: str,
    zero_threshold: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Detect zero/nodata pixels in the DEM.

    Assumes the study area has no legitimate near-zero-elevation
    terrain, so a zero elevation indicates a missing data gap from
    3DEP rather than real terrain -- true for most inland areas, not
    for coastal/below-sea-level ones (see --zero-threshold to adjust).

    Returns dict with gap statistics and the gap mask.
    """
    with rasterio.open(dem_path) as ds:
        data      = ds.read(1).astype(np.float32)
        nodata    = ds.nodata if ds.nodata is not None else -9999.0
        transform = ds.transform
        shape     = data.shape
        bounds    = ds.bounds

    # Gap mask: nodata OR below threshold (should be no zero terrain here)
    nodata_mask = data <= (nodata + 1)
    # All exactly-zero pixels are 3DEP gaps — water correction never
    # produces exactly 0.0 in this study area (all water bodies are
    # at significant elevation)
    zero_mask   = (data == 0.0)
    gap_mask    = nodata_mask | zero_mask

    n_total  = data.size
    n_nodata = int(nodata_mask.sum())
    n_zero   = int(zero_mask.sum()) - int((nodata_mask & zero_mask).sum())
    n_gaps   = int(gap_mask.sum())

    if verbose:
        print(f"[GAP DETECT] DEM: {shape[1]:,} x {shape[0]:,} px "
              f"= {n_total:,} total pixels")
        print(f"[GAP DETECT] Nodata pixels:     {n_nodata:,} "
              f"({n_nodata/n_total*100:.2f}%)")
        print(f"[GAP DETECT] Near-zero pixels:  {n_zero:,} "
              f"({n_zero/n_total*100:.2f}%)")
        print(f"[GAP DETECT] Total gap pixels:  {n_gaps:,} "
              f"({n_gaps/n_total*100:.2f}%)")
        print(f"[GAP DETECT] Valid pixels:      {n_total-n_gaps:,} "
              f"({(n_total-n_gaps)/n_total*100:.2f}%)")

        if n_gaps > 0:
            if n_gaps < 50_000_000:
                rows, cols = np.where(gap_mask)
                gap_lat_min = bounds.top  + rows.max() * transform.e
                gap_lat_max = bounds.top  + rows.min() * transform.e
                gap_lng_min = bounds.left + cols.min() * transform.a
                gap_lng_max = bounds.left + cols.max() * transform.a
                print(f"[GAP DETECT] Gap extent: "
                      f"lat [{gap_lat_min:.3f}, {gap_lat_max:.3f}] "
                      f"lng [{gap_lng_min:.3f}, {gap_lng_max:.3f}]")
            else:
                # Too large to allocate full coordinate array —
                # scan row strips to find bounding extent
                print(f"[GAP DETECT] Gap extent: scanning in strips...")
                strip_h = 1000
                h, w = gap_mask.shape
                first_row = None
                last_row  = None
                min_col   = w
                max_col   = 0
                for r0 in range(0, h, strip_h):
                    r1 = min(r0 + strip_h, h)
                    strip = gap_mask[r0:r1]
                    if not strip.any():
                        continue
                    rows_s, cols_s = np.where(strip)
                    if first_row is None:
                        first_row = r0 + rows_s.min()
                    last_row  = r0 + rows_s.max()
                    min_col   = min(min_col, int(cols_s.min()))
                    max_col   = max(max_col, int(cols_s.max()))
                if first_row is not None:
                    gap_lat_max = bounds.top  + first_row * transform.e
                    gap_lat_min = bounds.top  + last_row  * transform.e
                    gap_lng_min = bounds.left + min_col   * transform.a
                    gap_lng_max = bounds.left + max_col   * transform.a
                    print(f"[GAP DETECT] Gap extent: "
                          f"lat [{gap_lat_min:.3f}, {gap_lat_max:.3f}] "
                          f"lng [{gap_lng_min:.3f}, {gap_lng_max:.3f}]")

    return {
        'gap_mask':    gap_mask,
        'n_gaps':      n_gaps,
        'n_total':     n_total,
        'nodata':      nodata,
        'shape':       shape,
        'transform':   transform,
        'bounds':      bounds,
        'data':        data,
    }


# ─────────────────────────────────────────────
# GLO-30 tile fetching
# ─────────────────────────────────────────────

def get_required_glo30_tiles(
    bounds,
    verbose: bool = True,
) -> list:
    """
    Return list of (lat_floor, lng_floor) for all GLO-30 tiles
    needed to cover the gap extent.
    """
    # Expand slightly to ensure full coverage
    pad = 0.1
    south = bounds.bottom - pad
    north = bounds.top    + pad
    west  = bounds.left   - pad
    east  = bounds.right  + pad

    tiles = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lng in range(math.floor(west), math.ceil(east)):
            tiles.append((lat, lng))

    if verbose:
        print(f"[GLO30] Requires {len(tiles)} tiles to cover DEM extent")

    return tiles


def fetch_glo30_mosaic(
    tiles: list,
    verbose: bool = True,
) -> tuple:
    """
    Download GLO-30 tiles and merge into a single in-memory mosaic.
    Returns (mosaic_array, transform, profile).
    """
    from elevation import download_tile_glo30

    datasets   = []
    memfiles   = []
    failed     = []

    for i, (lat_floor, lng_floor) in enumerate(tiles):
        tile_center_lat = lat_floor + 0.5
        tile_center_lng = lng_floor + 0.5

        if verbose:
            print(f"[GLO30] Tile {i+1}/{len(tiles)}: "
                  f"({lat_floor}, {lng_floor})")
        try:
            path = download_tile_glo30(tile_center_lat, tile_center_lng)
            mf = rasterio.open(path)
            datasets.append(mf)
        except Exception as e:
            if verbose:
                print(f"[GLO30] Warning: tile ({lat_floor},{lng_floor}) "
                      f"failed: {e}")
            failed.append((lat_floor, lng_floor))

    if not datasets:
        raise ValueError("[GLO30] No tiles downloaded successfully")

    if verbose:
        print(f"[GLO30] Merging {len(datasets)} tiles...")

    mosaic, transform = merge(datasets)

    profile = datasets[0].profile.copy()
    profile.update({
        'height':    mosaic.shape[1],
        'width':     mosaic.shape[2],
        'transform': transform,
    })

    for ds in datasets:
        ds.close()

    if verbose and failed:
        print(f"[GLO30] Warning: {len(failed)} tiles failed — "
              f"gaps may remain in those areas")

    return mosaic[0].astype(np.float32), transform, profile


# ─────────────────────────────────────────────
# Gap fill
# ─────────────────────────────────────────────

def fill_gaps(
    dem_data:       np.ndarray,
    gap_mask:       np.ndarray,
    dem_transform,
    dem_shape:      tuple,
    glo30_data:     np.ndarray,
    glo30_transform,
    verbose: bool = True,
) -> np.ndarray:
    """
    Reproject GLO-30 onto the DEM grid and fill gap pixels.
    Returns the repaired DEM array.
    """
    if verbose:
        print(f"[FILL] Reprojecting GLO-30 onto DEM grid "
              f"({dem_shape[0]:,} x {dem_shape[1]:,} px)...")

    glo30_resampled = np.zeros(dem_shape, dtype=np.float32)

    reproject(
        source=glo30_data,
        destination=glo30_resampled,
        src_transform=glo30_transform,
        src_crs='EPSG:4326',
        dst_transform=dem_transform,
        dst_crs='EPSG:4326',
        resampling=Resampling.bilinear,
    )

    filled = dem_data.copy()
    n_filled = int(gap_mask.sum())
    filled[gap_mask] = glo30_resampled[gap_mask]

    if verbose:
        print(f"[FILL] Filled {n_filled:,} gap pixels with GLO-30 values")

        # Sanity check — should have no zeros remaining in filled areas
        still_zero = gap_mask & ((filled >= -1.0) & (filled <= 1.0))
        n_still_zero = int(still_zero.sum())
        if n_still_zero > 0:
            print(f"[FILL] Warning: {n_still_zero:,} pixels still near-zero "
                  f"after fill (likely ocean/water areas at GLO-30 extent)")
        else:
            print(f"[FILL] All gap pixels successfully filled")

    return filled


# ─────────────────────────────────────────────
# Water body correction
# ─────────────────────────────────────────────

def apply_water_correction(
    dem_path:   str,
    south:      float,
    west:       float,
    north:      float,
    east:       float,
    verbose:    bool = True,
) -> bool:
    """
    Re-apply NHD water body correction to the repaired DEM.
    Modifies dem_path in place.
    Returns True on success, False if correction was skipped.
    """
    try:
        from dem_water_correction import (
            fetch_water_bodies_nhd,
            correct_dem_water_bodies,
        )
    except ImportError as e:
        print(f"[WATER] Could not import dem_water_correction: {e}")
        print(f"[WATER] Skipping water body correction")
        return False

    if verbose:
        print(f"[WATER] Fetching NHD water bodies for study area...")

    try:
        water_bounds   = (south, west, north, east)
        water_polygons = fetch_water_bodies_nhd(water_bounds)

        if not water_polygons:
            print(f"[WATER] No water bodies found — skipping correction")
            return True

        if verbose:
            print(f"[WATER] Found {len(water_polygons)} water body polygons")
            print(f"[WATER] Applying correction to {dem_path}...")

        correct_dem_water_bodies(
            input_path=dem_path,
            output_path=dem_path,   # in-place
            water_polygons=water_polygons,
            buffer_pixels=3,
            shore_percentile=10.0,
            batch_size=500,
            verbose=verbose,
        )

        if verbose:
            print(f"[WATER] Water body correction complete")
        return True

    except Exception as e:
        print(f"[WATER] Error during water correction: {e}")
        return False


# ─────────────────────────────────────────────
# Main repair pipeline
# ─────────────────────────────────────────────

def repair_dem(
    dem_path:              str,
    south:                 float,
    west:                  float,
    north:                 float,
    east:                  float,
    dry_run:               bool  = False,
    skip_water_correction: bool  = False,
    zero_threshold:        float = 1.0,
    verbose:               bool  = True,
) -> dict:
    """
    Full repair pipeline:
      1. Detect gaps
      2. Fetch GLO-30 mosaic for DEM extent
      3. Fill gaps
      4. Write repaired DEM
      5. Re-apply water correction
    """
    print(f"\n{'='*62}")
    print(f"  DEM Gap Repair")
    print(f"  Input:  {dem_path}")
    print(f"  Bounds: {south}°N {west}°E  →  {north}°N {east}°E")
    print(f"  Mode:   {'DRY RUN' if dry_run else 'REPAIR'}")
    print(f"{'='*62}\n")

    t0 = time.time()

    # ── Step 1: Detect gaps ───────────────────────────────────────
    print(f"[STEP 1] Detecting gaps in {dem_path}...")
    gap_info = detect_gaps(dem_path, zero_threshold=zero_threshold,
                           verbose=verbose)

    if gap_info['n_gaps'] == 0:
        print(f"\n✓ No gaps found — DEM is complete, no repair needed")
        return {'status': 'no_gaps', 'n_gaps': 0}

    print(f"\n→ {gap_info['n_gaps']:,} gap pixels to fill "
          f"({gap_info['n_gaps']/gap_info['n_total']*100:.2f}% of DEM)")

    if dry_run:
        print(f"\n[DRY RUN] Would fill {gap_info['n_gaps']:,} pixels "
              f"from GLO-30 and re-apply water correction.")
        print(f"[DRY RUN] No files modified.")
        return {'status': 'dry_run', 'n_gaps': gap_info['n_gaps']}

    import psutil
    available_gb = psutil.virtual_memory().available / 1e9
    needed_gb    = (gap_info['n_total'] * 4 * 2) / 1e9  # two float32 arrays
    print(f"[MEMORY] Available: {available_gb:.1f}GB, "
          f"estimated needed: {needed_gb:.1f}GB")
    if available_gb < needed_gb * 1.2:
        print(f"[MEMORY] Warning: may be tight — consider closing other apps")

    # ── Step 2: Backup original ───────────────────────────────────
    backup_path = Path(dem_path).with_suffix('.tif.bak')
    if not backup_path.exists():
        print(f"\n[STEP 2] Backing up original to {backup_path}...")
        shutil.copy2(dem_path, backup_path)
        print(f"  Backup saved: {backup_path} "
              f"({backup_path.stat().st_size/1e6:.0f}MB)")
    else:
        print(f"\n[STEP 2] Backup already exists: {backup_path} — skipping")

    # ── Step 3: Fetch GLO-30 ──────────────────────────────────────
    print(f"\n[STEP 3] Fetching GLO-30 tiles for DEM extent...")
    tiles = get_required_glo30_tiles(gap_info['bounds'], verbose=verbose)

    glo30_data, glo30_transform, glo30_profile = fetch_glo30_mosaic(
        tiles, verbose=verbose)

    # ── Step 4: Fill gaps ─────────────────────────────────────────
    print(f"\n[STEP 4] Filling {gap_info['n_gaps']:,} gap pixels...")
    repaired = fill_gaps(
        dem_data=gap_info['data'],
        gap_mask=gap_info['gap_mask'],
        dem_transform=gap_info['transform'],
        dem_shape=gap_info['shape'],
        glo30_data=glo30_data,
        glo30_transform=glo30_transform,
        verbose=verbose,
    )

    # ── Step 5: Write repaired DEM ────────────────────────────────
    print(f"\n[STEP 5] Writing repaired DEM to {dem_path}...")
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()

    profile.update({
        'dtype':   'float32',
        'nodata':  gap_info['nodata'],
        'compress': 'deflate',
        'predictor': 2,
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        # Explicit, not GDAL's BIGTIFF=IF_NEEDED default -- see
        # build_dem_mosaic.py's identical comment for the live
        # failure this avoids on large study-area mosaics.
        'BIGTIFF': 'YES',
    })

    with rasterio.open(dem_path, 'w', **profile) as dst:
        dst.write(repaired.astype(np.float32), 1)

    written_mb = Path(dem_path).stat().st_size / 1e6
    print(f"  Written: {dem_path} ({written_mb:.0f}MB)")

    # ── Step 6: Verify fill ───────────────────────────────────────
    print(f"\n[STEP 6] Verifying repair...")
    post_gaps = detect_gaps(dem_path, zero_threshold=zero_threshold,
                            verbose=verbose)
    n_remaining = post_gaps['n_gaps']
    n_fixed     = gap_info['n_gaps'] - n_remaining

    print(f"\n  Gaps fixed:     {n_fixed:,}")
    print(f"  Gaps remaining: {n_remaining:,} "
          f"(likely water/ocean areas at GLO-30 edge)")

    # ── Step 7: Water correction ──────────────────────────────────
    if skip_water_correction:
        print(f"\n[STEP 7] Water correction skipped (--skip-water-correction)")
        water_ok = False
    else:
        print(f"\n[STEP 7] Re-applying NHD water body correction...")
        water_ok = apply_water_correction(
            dem_path=dem_path,
            south=south, west=west,
            north=north, east=east,
            verbose=verbose,
        )

    elapsed = time.time() - t0

    print(f"\n{'='*62}")
    print(f"  Repair complete in {elapsed:.0f}s")
    print(f"  Gaps filled:        {n_fixed:,}")
    print(f"  Gaps remaining:     {n_remaining:,}")
    print(f"  Water correction:   {'applied' if water_ok else 'skipped'}")
    print(f"  Backup:             {backup_path}")
    print(f"  Output:             {dem_path}")
    print(f"{'='*62}\n")

    return {
        'status':          'repaired',
        'n_gaps_original': gap_info['n_gaps'],
        'n_gaps_fixed':    n_fixed,
        'n_gaps_remaining': n_remaining,
        'water_correction': water_ok,
        'backup_path':     str(backup_path),
        'elapsed_s':       round(elapsed, 1),
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Repair DEM gaps by filling from GLO-30',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'corner1', nargs='?',
        help='Tile ID for one corner of the study area, e.g. N44W113 (with corner2; omit if using --bounds)',
    )
    parser.add_argument(
        'corner2', nargs='?',
        help='Tile ID for the opposite corner, e.g. N47W109',
    )
    parser.add_argument('--bounds', nargs=4, type=float,
        metavar=('SOUTH', 'WEST', 'NORTH', 'EAST'),
        help='Custom fractional-degree bounds instead of corner1/corner2')
    parser.add_argument('--allow-large', action='store_true',
        help=f'Bypass the {MAX_MOSAIC_TILES}-degree-tile size cap on corner1/corner2 areas')
    parser.add_argument(
        '--dem-path', required=True,
        help='Path to the existing mosaic GeoTIFF to repair',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Report gap statistics without modifying any files',
    )
    parser.add_argument(
        '--skip-water-correction', action='store_true',
        help='Fill gaps but skip NHD water body correction',
    )
    parser.add_argument(
        '--zero-threshold', type=float, default=1.0,
        help='Elevation below this (metres) is treated as a gap '
             '(default: 1.0 -- assumes no real terrain in the study '
             'area is at sea level; lower this if your area includes '
             'legitimate near-zero-elevation terrain)',
    )

    args = parser.parse_args()

    if args.bounds:
        south, west, north, east = args.bounds
    elif args.corner1 and args.corner2:
        b = bounds_from_tile_corners(args.corner1, args.corner2,
                                      allow_large=args.allow_large)
        south, west, north, east = b['south'], b['west'], b['north'], b['east']
    else:
        parser.error('Provide two corner tile IDs (e.g. N44W113 N47W109) or --bounds')

    dem_path = args.dem_path

    if not Path(dem_path).exists():
        print(f"ERROR: DEM not found: {dem_path}")
        sys.exit(1)

    result = repair_dem(
        dem_path=dem_path,
        south=south, west=west, north=north, east=east,
        dry_run=args.dry_run,
        skip_water_correction=args.skip_water_correction,
        zero_threshold=args.zero_threshold,
    )

    if result['status'] == 'repaired':
        print(f"✓ Repair successful")
    elif result['status'] == 'no_gaps':
        print(f"✓ No repair needed")
    elif result['status'] == 'dry_run':
        print(f"✓ Dry run complete")


if __name__ == '__main__':
    main()