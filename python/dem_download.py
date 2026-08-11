"""
dem_download.py
━━━━━━━━━━━━━━
Downloads the highest available resolution DEM tiles for a study area
and mosaics them into a single GeoTIFF.

Resolution priority per tile (US coverage):
  1. 3DEP 1m   -- where available (urban/surveyed areas)
  2. 3DEP 10m  -- broad US coverage
  3. GLO-30    -- global fallback (Copernicus 30m)

Study areas are specified as two opposite 1x1-degree tile corners
(see tile_id.py's tile ID format, e.g. "N45W110"), or with raw
--bounds for fractional-degree precision.

Usage:
    # Download and mosaic the area between two tile corners (any
    # pairing/order -- NE+SW, NW+SE, whichever two corners bound it)
    python dem_download.py N44W113 N47W109

    # Fractional-degree bounds instead
    python dem_download.py --bounds "44.5,-112.5,47.5,-109.5"

    # Dry run -- list tiles without downloading
    python dem_download.py N44W113 N47W109 --dry-run

    # Resume interrupted download
    python dem_download.py N44W113 N47W109 --resume

    # Skip mosaic step (download tiles only)
    python dem_download.py N44W113 N47W109 --no-mosaic
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import rasterio
import rasterio.merge
from rasterio.crs import CRS
from rasterio.enums import Resampling

from tile_id import bounds_from_tile_corners, MAX_MOSAIC_TILES


# ─────────────────────────────────────────────
# Study area definitions
# ─────────────────────────────────────────────

@dataclass
class StudyArea:
    name:        str
    description: str
    south:       float
    west:        float
    north:       float
    east:        float
    resolution:  str = 'best'
    output_dir:  str = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"data/dem/{self.name}"

    @property
    def bounds(self):
        return (self.south, self.west, self.north, self.east)

    @property
    def width_km(self):
        mid_lat = (self.south + self.north) / 2
        return (self.east - self.west) * 111.32 * math.cos(math.radians(mid_lat))

    @property
    def height_km(self):
        return (self.north - self.south) * 111.32


def study_area_between(corner1_id, corner2_id, name=None, resolution='best',
                        output_dir=None, max_tiles=MAX_MOSAIC_TILES,
                        allow_large=False):
    """
    Build a StudyArea spanning any two opposite 1x1-degree tile
    corners, e.g. "N44W113" and "N47W109" (any pairing/order -- NE+SW,
    NW+SE, whichever two corners bound the area you want). Longitude
    resolves to the shorter arc between the corners, and the request
    is capped at MAX_MOSAIC_TILES degree-tiles unless allow_large=True
    -- see bounds_from_tile_corners() in tile_id.py for the exact
    rules and why the cap exists.

    `name` defaults to "<corner1>_<corner2>" and only affects the
    output directory/manifest/mosaic filename.
    """
    b = bounds_from_tile_corners(corner1_id, corner2_id,
                                  max_tiles=max_tiles, allow_large=allow_large)
    if name is None:
        name = f"{corner1_id}_{corner2_id}"
    return StudyArea(
        name=name,
        description=f"{corner1_id} to {corner2_id} ({b['n_tiles']} degree-tiles)",
        south=b['south'], west=b['west'], north=b['north'], east=b['east'],
        resolution=resolution, output_dir=output_dir,
    )


# ─────────────────────────────────────────────
# Tile coordinate helpers
# ─────────────────────────────────────────────

def get_1deg_tiles(south, west, north, east):
    """
    Returns list of (lat_floor, lng_floor) for all 1-degree tiles
    covering the bounding box. lat_floor is the SW corner latitude.
    e.g. tile (45, -112) covers N45-N46, W111-W112.
    """
    tiles = []
    for lat in range(int(math.floor(south)), int(math.ceil(north))):
        for lng in range(int(math.floor(west)), int(math.ceil(east))):
            tiles.append((lat, lng))
    return tiles


def tile_label(lat, lng):
    ns = 'N' if lat >= 0 else 'S'
    ew = 'W' if lng < 0  else 'E'
    return f"{ns}{abs(lat):02d}{ew}{abs(lng):03d}"


# ─────────────────────────────────────────────
# Source URL builders
# ─────────────────────────────────────────────


def url_3dep_10m_direct(lat, lng):
    """
    3DEP 10m direct tile URL on USGS S3 — current path format.
    """
    if lng >= 0 or lat < 24 or lat > 72:
        return None
    lat_str = f"{abs(lat + 1):02d}"
    lng_str = f"{abs(lng):03d}"
    # Current USGS hosted path (updated naming)
    return (
        f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/"
        f"13/TIFF/current/n{lat_str}w{lng_str}/"
        f"USGS_13_n{lat_str}w{lng_str}.tif"
    )


def url_glo30(lat, lng):
    """
    Copernicus GLO-30 via Copernicus DEM AWS bucket.
    Global coverage at ~30m resolution.
    Public, no credentials required.
    """
    ns  = 'N' if lat >= 0 else 'S'
    ew  = 'E' if lng >= 0 else 'W'
    lat_str = f"{abs(lat):02d}"
    lng_str = f"{abs(lng):03d}"
    # Copernicus DEM on AWS — public bucket, no auth needed
    return (
        f"https://copernicus-dem-30m.s3.amazonaws.com/"
        f"Copernicus_DSM_COG_10_{ns}{lat_str}_00_{ew}{lng_str}_00_DEM/"
        f"Copernicus_DSM_COG_10_{ns}{lat_str}_00_{ew}{lng_str}_00_DEM.tif"
    )


def url_srtm_30m(lat, lng):
    """
    SRTM GL1 30m via OpenTopography S3 using boto3/requests with anon creds.
    Fallback to SRTM if Copernicus also fails.
    """
    ns  = 'N' if lat >= 0 else 'S'
    ew  = 'E' if lng >= 0 else 'W'
    lat_str = f"{abs(lat):02d}"
    lng_str = f"{abs(lng):03d}"
    return (
        f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/"
        f"{ns}{lat_str}/{ns}{lat_str}{ew}{lng_str}.hgt.gz"
    )


def resolve_10m_url(lat, lng):
    """
    Query the USGS TNM API to find the current direct download URL
    for the 10m tile covering this 1-degree cell.
    Falls back to direct path if API is unavailable.
    """
    if lng >= 0 or lat < 24 or lat > 72:
        return None

    api_url = (
        f"https://tnmaccess.nationalmap.gov/api/v1/products"
        f"?datasets=Digital%20Elevation%20Model%20(DEM)%201%2F3%20arc-second"
        f"&bbox={lng},{lat},{lng+1},{lat+1}"
        f"&outputFormat=JSON&max=5"
    )
    try:
        resp = requests.get(api_url, timeout=20)
        if resp.status_code == 200:
            items = resp.json().get('items', [])
            for item in items:
                dl = item.get('downloadURL', '')
                if dl.lower().endswith('.tif'):
                    return dl
    except Exception:
        pass

    # Direct path fallback — USGS current folder
    lat_str = f"{abs(lat + 1):02d}"
    lng_str = f"{abs(lng):03d}"
    return (
        f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/"
        f"13/TIFF/current/n{lat_str}w{lng_str}/"
        f"USGS_13_n{lat_str}w{lng_str}.tif"
    )


def get_candidates(lat, lng, resolution='best'):
    """
    Returns ordered list of (url, res_label) to attempt for a tile.
    'best' tries finest resolution first and falls back automatically.
    """
    if resolution == '10m':
        u = resolve_10m_url(lat, lng)
        return [(u, '10m')] if u else []
    if resolution == '30m':
        return [(url_glo30(lat, lng), '30m')]

    # 'best' -- cascade: 10m USGS -> Copernicus 30m
    result = []
    u10 = resolve_10m_url(lat, lng)
    if u10:
        result.append((u10, '10m'))
    result.append((url_glo30(lat, lng), '30m'))
    return result

# ─────────────────────────────────────────────
# Ocean tiling fill in
# ─────────────────────────────────────────────

_LAND_MASK_RES_DEG = 1.0 / 120  # global_land_mask's native grid spacing (~1km)


def _is_likely_ocean(lat: int, lng: int) -> bool:
    """
    Whole-tile ocean check backed by the `global_land_mask` package (bundled
    GSHHG-derived land/sea raster at ~1km/0.00833deg resolution, no network
    call). Samples the ENTIRE 1x1 degree tile at the mask's own native
    resolution -- not a coarser sub-sample -- so no coastline sliver or small
    island between sample points can be missed. Returns True only if every
    sampled point in the tile is ocean; a single land sample anywhere aborts
    it, since this feeds an irreversible "safe to zero-fill" decision and a
    false positive would silently erase real terrain.

    Replaces a previous heuristic (`lng <= -121 and lat <= 35`) that only
    covered the SoCal coastal strip and silently produced NO fallback for
    open-ocean tiles just east of that cutoff -- e.g. N31W118/N31W119/
    N31W120 (Pacific water off Baja California, incident 2026-07-30).
    """
    from global_land_mask import globe

    lat_samples = np.arange(lat, lat + 1.0, _LAND_MASK_RES_DEG)
    lng_samples = np.arange(lng, lng + 1.0, _LAND_MASK_RES_DEG)
    lat_grid, lng_grid = np.meshgrid(lat_samples, lng_samples, indexing='ij')
    ocean_grid = globe.is_ocean(lat_grid, lng_grid)
    return bool(ocean_grid.all())


def _create_ocean_tile(lat: int, lng: int,
                        output_dir: Path) -> Optional[tuple]:
    """
    Create a zero-elevation GeoTIFF for ocean-only tiles.
    Uses 30m resolution (1/3 arc-second grid).
    """
    from rasterio.transform import from_bounds

    ns  = 'N' if lat >= 0 else 'S'
    ew  = 'W' if lng < 0  else 'E'
    filename = f"{ns}{abs(lat):02d}_{ew}{abs(lng):03d}_ocean0m.tif"
    # output_dir arrives as a plain str from download_tile's own caller
    # (tile_builder.py's _step_download passes str(downloads_dir)) --
    # download_tile itself wraps it (Path(output_dir) / 'raw') before use,
    # but this function didn't, despite being type-hinted Path. Real bug
    # hit live twice this session: TypeError: unsupported operand type(s)
    # for /: 'str' and 'str' for every western/oceanic tile this fallback
    # was supposed to handle cleanly -- fixed once (2026-07-30), then lost
    # when this file was recreated after a working-tree wipe and not
    # re-applied; refixed here (2026-08-02) after N32W122/N33W122/N34W122
    # hit the exact same TypeError again during the gap-recompute run.
    out_path = Path(output_dir) / 'raw' / filename

    if out_path.exists():
        return out_path, 'ocean0m'

    # 1-degree tile at ~30m resolution = 3600×3600 pixels
    res   = 1 / 3600
    w, s  = float(lng), float(lat)
    e, n  = w + 1.0,    s + 1.0
    width = height = 3600

    transform = from_bounds(w, s, e, n, width, height)
    profile = {
        'driver':    'GTiff',
        'dtype':     'float32',
        'width':     width,
        'height':    height,
        'count':     1,
        'crs':       'EPSG:4326',
        'transform': transform,
        'compress':  'deflate',
        'nodata':    None,
    }

    data = np.zeros((1, height, width), dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data)

    print(f"  [{ns}{abs(lat):02d}{ew}{abs(lng):03d}] "
          f"Created zero-elevation ocean tile")
    return out_path, 'ocean0m'

# ─────────────────────────────────────────────
# Single tile download
# ─────────────────────────────────────────────

# Wall-clock cap on one candidate's transfer, independent of the per-read
# `timeout`. requests' timeout only bounds connect time and the gap between
# individual reads — a slow-but-steady trickle (a byte every ~100s) never
# trips it and can hang indefinitely (observed: an 11.9h stall on a single
# tile). This bounds total transfer time so a stalled source falls through
# to the next resolution candidate (e.g. GLO-30) instead of hanging the batch.
DEFAULT_MAX_DOWNLOAD_S = 600


def _raster_is_readable(path) -> bool:
    """
    True if `path` opens AND its pixel data can actually be decoded --
    not just that the header parses. A truncated GeoTIFF (an interrupted
    download, e.g. a killed process mid-write) commonly still has an
    intact, readable header, so `with rasterio.open(path): pass` calls
    that "valid" even though the pixel data is corrupt. Reads through
    every block via block_windows() (no full-array materialization, so
    this stays cheap even for a large tile) so a truncated tile/strip
    anywhere in the file surfaces here, at cache-validation time,
    instead of silently downstream.

    Real incident (2026-08-08): translating this exact validation logic
    into an R port hit precisely this bug on a download interrupted by
    a 2-minute process timeout -- header intact, one pixel tile
    corrupted, a bare open() didn't catch it. This function replaces
    what used to be a bare open-only check here too.
    """
    try:
        with rasterio.open(path) as src:
            for _, window in src.block_windows(1):
                src.read(1, window=window)
        return True
    except Exception:
        return False


def download_tile(lat, lng, output_dir, resolution='best',
                  timeout=120, chunk_mb=1, max_download_s=DEFAULT_MAX_DOWNLOAD_S):
    """
    Download one 1-degree DEM tile at the best available resolution.
    Returns (Path, res_label) on success, None on failure.
    Skips download if file already exists and is valid.
    """
    candidates = get_candidates(lat, lng, resolution)
    if not candidates:
        print(f"  [{tile_label(lat, lng)}] No source available")
        return None

    raw_dir = Path(output_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    for url, res_label in candidates:
        filename = f"{tile_label(lat, lng)}_{res_label}.tif"
        out_path = raw_dir / filename

        # Already cached and valid
        if out_path.exists() and out_path.stat().st_size > 10_000:
            if _raster_is_readable(out_path):
                print(f"  [{tile_label(lat, lng)}] Cached ({res_label})")
                return out_path, res_label
            out_path.unlink(missing_ok=True)

        print(f"  [{tile_label(lat, lng)}] Trying {res_label}...")

        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            if resp.status_code == 404:
                print(f"    404 -- trying next resolution")
                continue
            resp.raise_for_status()

            total      = int(resp.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = chunk_mb * 1024 * 1024
            t_start    = time.monotonic()

            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r    {downloaded/1e6:.1f}/"
                                  f"{total/1e6:.1f} MB ({pct:.0f}%)",
                                  end='', flush=True)
                    elapsed = time.monotonic() - t_start
                    if elapsed > max_download_s:
                        raise TimeoutError(
                            f"stalled — {downloaded/1e6:.1f} MB in "
                            f"{elapsed:.0f}s (cap {max_download_s}s)")
            print(f"\r    Downloaded {downloaded/1e6:.1f} MB at {res_label}")

            # Validate
            if _raster_is_readable(out_path):
                return out_path, res_label
            else:
                print(f"    Invalid raster (failed to decode pixel data), trying next")
                out_path.unlink(missing_ok=True)

        except requests.exceptions.Timeout:
            print(f"    Timeout, trying next")
            if out_path.exists():
                out_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"    Error: {e}, trying next")
            if out_path.exists():
                out_path.unlink(missing_ok=True)

    print(f"  [{tile_label(lat, lng)}] All sources failed")
    if _is_likely_ocean(lat, lng):
        return _create_ocean_tile(lat, lng, output_dir)
    return None


# ─────────────────────────────────────────────
# Mosaic
# ─────────────────────────────────────────────

def mosaic_tiles(tile_paths, output_path, bounds):
    import rasterio.warp
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.enums import Resampling
    
    """
    Merge tiles into a single cropped, compressed GeoTIFF.
    Tiles at different resolutions are resampled to the finest present.
    """
    if not tile_paths:
        print("[MOSAIC] No tiles to merge")
        return

    print(f"\n[MOSAIC] Merging {len(tile_paths)} tiles...")

    # Sort finest resolution first so merge reference grid is finest
    def res_order(p):
        n = p.stem.lower()
        if '1m'  in n: return 0
        if '10m' in n: return 1
        return 2

    tile_paths = sorted(tile_paths, key=res_order)

    with rasterio.open(tile_paths[0]) as ref:
        target_crs = ref.crs


    
    datasets = []
    tmp_files = []

    for tp in tile_paths:
        ds = rasterio.open(tp)
        if ds.crs != target_crs:
            print(f"  Reprojecting {tp.name} "
                  f"({ds.crs} → {target_crs})...")
            ds.close()
            # Reproject to temp file
            tmp_path = tp.parent / f"_tmp_{tp.name}"
            tmp_files.append(tmp_path)
            with rasterio.open(tp) as src:
                transform, width, height = \
                    calculate_default_transform(
                        src.crs, target_crs,
                        src.width, src.height, *src.bounds)
                profile = src.profile.copy()
                profile.update(crs=target_crs, transform=transform,
                               width=width, height=height)
                with rasterio.open(tmp_path, 'w', **profile) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear,
                        )
            datasets.append(rasterio.open(tmp_path))
        else:
            datasets.append(ds)

    if not datasets:
        print("[MOSAIC] No valid tiles")
        return

    south, west, north, east = bounds

    try:
        # dst_path (rather than no dst_path + a separate dst.write() of
        # the returned array) keeps this windowed/bounded-memory --
        # merge() subdivides the output into mem_limit-sized pixel
        # windows internally instead of building the whole mosaic as
        # one in-memory array first. See build_dem_mosaic.py's
        # build_mosaic() for the live RAM-pressure failure this avoids
        # (same underlying pattern, different call site).
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dst_kwds = dict(
            driver='GTiff', crs=CRS.from_epsg(4326), dtype='float32',
            compress='deflate', predictor=2, tiled=True,
            blockxsize=512, blockysize=512,
            # Explicit, not GDAL's BIGTIFF=IF_NEEDED default -- see
            # build_dem_mosaic.py's identical comment for the live
            # failure (real DEM terrain doesn't compress reliably
            # enough for IF_NEEDED's under-4GB-after-compression bet).
            BIGTIFF='YES',
        )
        rasterio.merge.merge(
            datasets,
            bounds=(west, south, east, north),
            resampling=Resampling.bilinear,
            dst_path=output_path,
            dst_kwds=dst_kwds,
            mem_limit=512,  # MB/chunk; see build_mosaic() for why
        )

        with rasterio.open(output_path) as src:
            size_mb = Path(output_path).stat().st_size / 1e6
            res_m   = abs(src.res[0]) * 111_320
            print(f"[MOSAIC] Output: {output_path}")
            print(f"         Size:   {size_mb:.0f} MB")
            print(f"         Grid:   {src.width:,} x {src.height:,} px")
            print(f"         Res:    ~{res_m:.1f} m/px")

    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass
        for tmp in tmp_files:
            try: tmp.unlink()
            except Exception: pass


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def download_study_area(area, dry_run=False, resume=True,
                        max_workers=4, no_mosaic=False):
    tiles = get_1deg_tiles(area.south, area.west, area.north, area.east)

    print(f"\n{'='*62}")
    print(f"  {area.name}: {area.description}")
    print(f"  Bounds: {area.south}S {area.west}W  {area.north}N {area.east}E")
    print(f"  Size:   {area.width_km:.0f} km x {area.height_km:.0f} km")
    print(f"  Tiles:  {len(tiles)} x 1-degree tiles")
    print(f"  Res:    {area.resolution}")
    print(f"  Output: {area.output_dir}")
    print(f"{'='*62}\n")

    if dry_run:
        print("[DRY RUN] Tiles that would be downloaded:\n")
        for lat, lng in sorted(tiles):
            candidates = get_candidates(lat, lng, area.resolution)
            label = tile_label(lat, lng)
            for url, res in candidates:
                print(f"  {label}: {res}")
                print(f"    {url[:72]}...")
                break
            if not candidates:
                print(f"  {label}: no source available")
        print(f"\nTotal: {len(tiles)} tiles")
        return

    # Download in parallel
    downloaded = []
    failed     = []

    print(f"Downloading with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_tile, lat, lng, area.output_dir, area.resolution
            ): (lat, lng)
            for lat, lng in tiles
        }
        for future in as_completed(futures):
            lat, lng = futures[future]
            try:
                result = future.result()
                if result:
                    downloaded.append(result[0])
                else:
                    failed.append((lat, lng))
            except Exception as e:
                print(f"  [{tile_label(lat, lng)}] Error: {e}")
                failed.append((lat, lng))

    print(f"\nDownload complete: {len(downloaded)}/{len(tiles)} tiles")
    if failed:
        print(f"Failed tiles ({len(failed)}):")
        for lat, lng in failed:
            print(f"  {tile_label(lat, lng)}")

    # Save manifest
    manifest = {
        'study_area':    area.name,
        'bounds':        area.bounds,
        'resolution':    area.resolution,
        'total_tiles':   len(tiles),
        'downloaded':    len(downloaded),
        'failed_tiles':  [tile_label(lat, lng) for lat, lng in failed],
        'tile_files':    [str(p) for p in downloaded],
    }
    manifest_path = Path(area.output_dir) / 'download_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")

    if no_mosaic or not downloaded:
        return

    mosaic_path = Path(area.output_dir) / f"{area.name}_mosaic.tif"
    mosaic_tiles(downloaded, str(mosaic_path), area.bounds)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download and mosaic DEM tiles for terrain-analysis study areas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'corner1', nargs='?',
        help='Tile ID for one corner of the area, e.g. N44W113 (with corner2; omit if using --bounds)',
    )
    parser.add_argument(
        'corner2', nargs='?',
        help='Tile ID for the opposite corner, e.g. N47W109',
    )
    parser.add_argument('--bounds',
        help='south,west,north,east -- alternative to corner1/corner2 for fractional-degree precision')
    parser.add_argument('--allow-large', action='store_true',
        help=f'Bypass the {MAX_MOSAIC_TILES}-degree-tile size cap on corner1/corner2 areas')
    parser.add_argument('--resolution', default='best',
        choices=['best', '1m', '10m', '30m'])
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--workers', type=int, default=4,
        help='Parallel download threads (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
        help='List tiles without downloading')
    parser.add_argument('--no-mosaic', action='store_true',
        help='Download only, skip mosaic')
    parser.add_argument('--no-resume', action='store_true',
        help='Re-download even if tile already exists')

    args = parser.parse_args()

    if args.bounds:
        s, w, n, e = [float(x) for x in args.bounds.split(',')]
        area = StudyArea(
            name='custom',
            description='Custom area',
            south=s, west=w, north=n, east=e,
            resolution=args.resolution,
            output_dir=args.output_dir or 'data/dem/custom',
        )
    elif args.corner1 and args.corner2:
        area = study_area_between(
            args.corner1, args.corner2,
            resolution=args.resolution, output_dir=args.output_dir,
            allow_large=args.allow_large,
        )
    else:
        parser.error('either two corner tile IDs (e.g. N44W113 N47W109) or --bounds is required')

    download_study_area(
        area=area,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        max_workers=args.workers,
        no_mosaic=args.no_mosaic,
    )


if __name__ == '__main__':
    main()