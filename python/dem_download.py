"""
dem_download.py
━━━━━━━━━━━━━━
Downloads the highest available resolution DEM tiles for a study area
and mosaics them into a single GeoTIFF.

Resolution priority per tile (US coverage):
  1. 3DEP ~3m  -- where available (urban/surveyed areas). Opt-in only
                  (--try-3m or --resolution 3m) -- coverage is sparse
                  and the raw fetch is expensive, so this is NOT part
                  of the default 'best' cascade. Fetched via USGS WCS
                  (elevation.py's sub-tile fetcher, reused here) as a
                  36000x36000px grid -- NOT literally 1m/px despite
                  pulling from USGS's 3DEP "1m" product: this is a
                  deliberate tradeoff in how the fetch is shaped (a
                  cheap 900-request grid), not a confirmed hard limit
                  of the WCS server itself. Resampled DOWN onto the
                  canonical 10800x10800 grid with area averaging for
                  the main mosaic (a higher-fidelity INPUT there, not
                  a different output resolution -- same treatment
                  GLO-30 already gets on the other end of the
                  cascade), AND kept at its own ~3m resolution,
                  layered over the final mosaic as a
                  `<name>_mosaic_hires.vrt` -- real fine detail where
                  it was actually fetched, the normal mosaic
                  everywhere else. A cheap probe checks for real
                  coverage before the full 900-request sub-tile fetch.
  1b. 3DEP 1m  -- genuine ~1m/px, not the ~3m/px above. Streams
                  straight to disk window-by-window instead of
                  assembling the whole tile in memory (~50GB as one
                  array at this scale) -- see
                  elevation._download_tile_3dep_native. Probes a
                  coarse coverage grid first (probe_3dep_1m_coverage_grid)
                  and skips sub-tiles over confirmed-uncovered ground
                  rather than fetching them. Still up to ~3100 WCS
                  requests and tens of minutes for a well-covered tile.
                  Explicit opt-in only (--resolution 1m) -- never
                  tried under 'best' or --try-3m. Unlike every other
                  resolution here, a tile with no 1m coverage falls
                  back to the normal 10m/GLO-30 cascade below (never
                  to the 3m tier) rather than being left out of the
                  area mosaic entirely -- the point of requesting 1m
                  for a whole area is real detail where it exists, not
                  an all-or-nothing demand.
  2. 3DEP 10m  -- broad US coverage, direct download (the default
                  first tier under 'best')
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
import shutil
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
import rasterio.warp
from rasterio.crs import CRS
from rasterio.enums import Resampling

from tile_id import bounds_from_tile_corners, MAX_MOSAIC_TILES
from build_hires_vrt import build_hires_vrt


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


# ─────────────────────────────────────────────
# 3DEP ~3m / genuine 1m (WCS, sparse coverage)
# ─────────────────────────────────────────────
# Neither is a direct-URL download like 10m -- USGS only serves this
# tier via WCS, fetched as a grid of sub-tile requests
# (elevation.py's _download_tile_3dep_subtiled for the ~3m tier,
# already built/tested for the on-demand single-point lookup and
# reused here; _download_tile_3dep_native for the genuine 1m tier).
# Coverage is sparse (surveyed/urban areas only) and the raw fetch is
# expensive, so both are opt-in only (try_3m/resolution='3m', or the
# much more expensive resolution='1m') -- NOT part of the default
# 'best' cascade every normal build uses. A cheap single-request probe
# runs first in both cases, so the full sub-tile grid is only
# attempted where the probe finds real data.

_3DEP_WCS_BASE = (
    "https://elevation.nationalmap.gov/arcgis/services/"
    "3DEPElevation/ImageServer/WCSServer"
    "?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
    "&COVERAGE=DEP3Elevation&CRS=EPSG:4326&FORMAT=GeoTIFF"
)

_1M_PROBE_PX = 64          # small request -- cheap, but big enough that
                            # landing near a coverage seam still reads
                            # some real data if any is nearby
_1M_PROBE_HALF_DEG = 0.0005  # ~100m at mid-latitudes
_CANONICAL_GRID_PX = 10800  # matches tile_builder.py's downstream
                                 # canonical-grid convention (1/10800
                                 # deg/px)
_3M_GRID_N = 30            # sub-tile grid for the 1m tier -- matches
_3M_SUBTILE_PX = 1200      # elevation.py's own download_tile_3dep()
                            # resolution=1 branch (grid_n=30,
                            # subtile_px=1200 -> 36000x36000px, ~3m/px).
                            # _download_tile_3dep_subtiled's own
                            # `resolution` kwarg is a print label only --
                            # grid_n/subtile_px are what actually control
                            # the request grid, so both must be passed
                            # explicitly here.
_3M_DEGRADED_WIDTH_THRESHOLD_PX = 25_000  # a real 1m sub-tiled fetch is
                                            # 36000px wide (_3M_GRID_N *
                                            # _3M_SUBTILE_PX); well below
                                            # that means the internal
                                            # GLO-30 fallback fired


def _resample_native_to_canonical_grid(native_path, lat_floor: int,
                                        lng_floor: int, out_path,
                                        grid_px: int = None) -> None:
    """
    Area-average a native-resolution raster covering one 1-degree
    tile DOWN onto this pipeline's canonical grid (default
    _CANONICAL_GRID_PX, i.e. the same 10800x10800 lattice every
    other source uses) and write it to `out_path`. Shared by the ~3m
    tier (download_tile_3dep_3m) and the true ~1m tier
    (download_tile_3dep_1m) -- both need the exact same
    "higher-fidelity input, same output grid" treatment for the
    standard mosaic. Area averaging (not bilinear) is correct here
    because we're downsampling real measured data, not upsampling a
    coarser source (see download_tile_3dep_3m's docstring for why that
    distinction matters).
    """
    grid_px = grid_px or _CANONICAL_GRID_PX
    w, s, e, n = lng_floor, lat_floor, lng_floor + 1, lat_floor + 1
    with rasterio.open(native_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        dst_transform = rasterio.transform.from_bounds(w, s, e, n, grid_px, grid_px)
        dst_data = np.full((grid_px, grid_px), nodata, dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=CRS.from_epsg(4326),
            src_nodata=src.nodata,
            dst_nodata=nodata,
            resampling=Resampling.average,
        )
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
            'width': grid_px, 'height': grid_px,
            'crs': 'EPSG:4326', 'transform': dst_transform,
            'compress': 'deflate', 'BIGTIFF': 'YES',
            'nodata': nodata,
        }
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(dst_data, 1)


def _probe_3dep_coverage_bbox(west: float, south: float, east: float,
                               north: float, probe_px: int = _1M_PROBE_PX,
                               timeout: int = 15) -> bool:
    """
    Cheap single-request check for real (non-nodata) 3DEP 1m coverage
    inside an arbitrary bbox. Returns True only if at least half the
    probe window came back as real elevation -- a handful of stray
    valid pixels at a coverage boundary shouldn't count as "covered".
    Shared by both the whole-tile probe and the coarse per-cell grid
    probe below.
    """
    url = (_3DEP_WCS_BASE +
           f"&BBOX={west},{south},{east},{north}"
           f"&WIDTH={probe_px}&HEIGHT={probe_px}")
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return False
        data = resp.content
        if data[:4] not in (b'II*\x00', b'MM\x00*', b'II+\x00'):
            return False
        with rasterio.io.MemoryFile(data) as mf, mf.open() as ds:
            arr = ds.read(1)
            nodata = ds.nodata if ds.nodata is not None else -9999.0
            valid = arr[(arr > nodata + 1) & (arr != 0)]
            return valid.size >= arr.size * 0.5
    except Exception:
        return False


def _probe_3dep_1m_coverage(lat_floor: int, lng_floor: int,
                             timeout: int = 15) -> bool:
    """
    Cheap single-request check for 3DEP 1m coverage near this tile's
    center, before committing to a full sub-tile grid fetch. See
    _probe_3dep_coverage_bbox for the actual check.
    """
    cx, cy = lng_floor + 0.5, lat_floor + 0.5
    h = _1M_PROBE_HALF_DEG
    return _probe_3dep_coverage_bbox(cx - h, cy - h, cx + h, cy + h,
                                      probe_px=_1M_PROBE_PX, timeout=timeout)


def probe_3dep_1m_coverage_grid(lat_floor: int, lng_floor: int,
                                 grid_n: int, max_workers: int = 8,
                                 probe_px: int = _1M_PROBE_PX,
                                 timeout: int = 15):
    """
    Probe a grid_n x grid_n grid of cells spanning this 1-degree tile
    for real 3DEP 1m coverage, one cheap small request per cell (run
    in parallel). Used to decide, before a true-native-resolution
    fetch, which fine sub-tiles are worth fetching at all -- 3DEP 1m
    coverage is sparse (surveyed urban/project areas only), so most of
    a given tile commonly has none, and there's no reason to spend a
    full-size WCS request per fine sub-tile finding that out.

    A coarse grid rather than probing every fine sub-tile individually
    is a deliberate cost/precision tradeoff: real 1m coverage areas are
    typically contiguous surveyed regions well larger than one coarse
    cell at reasonable grid_n, so this rarely misses real coverage, and
    keeps the probe pass itself cheap (grid_n=10 is 100 small requests,
    a few seconds in parallel, vs. thousands for the real fetch this
    guards). Raise grid_n for finer (but more expensive) precision.

    Returns a grid_n x grid_n boolean numpy array, row 0 = southernmost
    strip (matches the (row, col) convention used to build it -- NOT
    raster row order).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    step = 1.0 / grid_n
    coverage = np.zeros((grid_n, grid_n), dtype=bool)

    def probe_cell(args):
        row, col = args
        cx = lng_floor + (col + 0.5) * step
        cy = lat_floor + (row + 0.5) * step
        h = min(_1M_PROBE_HALF_DEG, step / 4)
        return row, col, _probe_3dep_coverage_bbox(
            cx - h, cy - h, cx + h, cy + h,
            probe_px=probe_px, timeout=timeout)

    cells = [(row, col) for row in range(grid_n) for col in range(grid_n)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(probe_cell, c) for c in cells]
        for future in as_completed(futures):
            row, col, has_coverage = future.result()
            coverage[row, col] = has_coverage

    n_covered = int(coverage.sum())
    print(f"[3DEP-native] Coverage probe: {n_covered}/{grid_n*grid_n} "
          f"coarse cells have real 1m data")
    return coverage


def download_tile_3dep_3m(lat, lng, output_dir):
    """
    Attempt the 3DEP ~3m tier for one 1-degree tile: probe for
    coverage cheaply, and if present, fetch the full tile via
    elevation.py's WCS sub-tile fetcher (already gap-filled from
    GLO-30 internally for any sub-tiles that fail, and grid-snapped
    onto an exact target shape/transform rather than trusting
    merge()'s own inference), persist the raw fetch as a durable
    sibling file (`{tile}_3m_native.tif` -- see build_hires_vrt.py,
    which layers it over the 10m mosaic as a real-detail overlay), and
    also resample it DOWN onto this pipeline's canonical 10800x10800
    grid using area averaging for the standard mosaic -- not bilinear,
    which is right for upsampling GLO-30 but wrong for downsampling
    this data (bilinear would just sample a few nearby points instead
    of averaging the ~100 real fine pixels each 10m cell actually
    covers, throwing away most of the accuracy benefit of fetching it
    in the first place).

    This is NOT genuine 1m/px -- see download_tile_3dep_1m for that.
    It requests the 3DEP 1m *product* via a sub-tile grid shaped for a
    cheap 900-request fetch, which comes out to ~3m/px (see this
    module's docstring for the full explanation).

    Returns (Path, '3m') on success, None if there's no coverage or
    the fetch/resample failed for any reason -- callers fall through
    to the existing 10m/GLO-30 cascade unchanged.
    """
    lat_floor, lng_floor = math.floor(lat), math.floor(lng)

    # Coarse US-only gate, same bounds convention resolve_10m_url uses --
    # skip the probe request entirely outside plausible 3DEP coverage.
    if lng_floor >= 0 or lat_floor < 24 or lat_floor > 72:
        return None

    raw_dir = Path(output_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{tile_label(lat, lng)}_3m.tif"
    native_path = raw_dir / f"{tile_label(lat, lng)}_3m_native.tif"

    already_downsampled = False
    if out_path.exists() and out_path.stat().st_size > 10_000:
        if _raster_is_readable(out_path):
            already_downsampled = True
        else:
            out_path.unlink(missing_ok=True)

    native_ready = native_path.exists() and _raster_is_readable(native_path)

    # Full cache hit -- short-circuit before any network call, and
    # before the coverage probe below, which would otherwise discard
    # an already-valid mosaic tile on a transient probe failure.
    if already_downsampled and native_ready:
        print(f"  [{tile_label(lat, lng)}] Cached (3m)")
        return out_path, '3m'

    # Native raster already fetched, only the resampled mosaic tile is
    # missing (e.g. killed between the two steps on a prior run) --
    # resample straight from the cached native file rather than
    # re-running the full 900-request fetch to reproduce data already
    # on disk.
    if native_ready and not already_downsampled:
        try:
            _resample_native_to_canonical_grid(native_path, lat_floor,
                                                lng_floor, out_path)
            print(f"  [{tile_label(lat, lng)}] Resampled cached native "
                  f"(3m) to canonical grid")
            return out_path, '3m'
        except Exception as e:
            print(f"  [{tile_label(lat, lng)}] Resample of cached native "
                  f"failed ({e}) -- refetching")

    # Below this point, a fetch/probe failure only costs the (still
    # missing) native overlay backfill if a valid mosaic tile already
    # exists -- return that cached tile instead of None so a transient
    # failure never discards previously-good output.
    cached_fallback = (out_path, '3m') if already_downsampled else None

    if not _probe_3dep_1m_coverage(lat_floor, lng_floor):
        return cached_fallback

    try:
        from elevation import _download_tile_3dep_subtiled
    except Exception as e:
        print(f"  [{tile_label(lat, lng)}] 3m unavailable "
              f"(elevation.py import failed: {e})")
        return cached_fallback

    try:
        # grid_n/subtile_px must be passed explicitly -- the
        # `resolution` kwarg below is only a print label inside
        # _download_tile_3dep_subtiled; matches elevation.py's own
        # download_tile_3dep(..., resolution=1) branch, which is the
        # tested/correct source of these numbers.
        raw_3m_path = _download_tile_3dep_subtiled(
            lat_floor, lng_floor, resolution=1,
            grid_n=_3M_GRID_N, subtile_px=_3M_SUBTILE_PX)
    except Exception as e:
        print(f"  [{tile_label(lat, lng)}] 3m fetch failed: {e}")
        return cached_fallback

    try:
        with rasterio.open(raw_3m_path) as src:
            # _download_tile_3dep_subtiled falls back to a direct
            # GLO-30 (30m) tile internally if every one of its own
            # 900 sub-tile requests failed -- guard against silently
            # "resampling" that under this label. A real fetch at this
            # pipeline's sub-tile grid is 36000px wide (_3M_GRID_N *
            # _3M_SUBTILE_PX); far below that means the fallback fired.
            if src.width < _3M_DEGRADED_WIDTH_THRESHOLD_PX:
                print(f"  [{tile_label(lat, lng)}] 3m fetch degraded "
                      f"to a fallback resolution, skipping")
                return cached_fallback

            if not native_ready:
                shutil.copyfile(raw_3m_path, native_path)
                if already_downsampled:
                    print(f"  [{tile_label(lat, lng)}] Native kept at "
                          f"{native_path.name} (mosaic tile already cached)")

            if already_downsampled:
                return out_path, '3m'

        _resample_native_to_canonical_grid(raw_3m_path, lat_floor, lng_floor,
                                            out_path)
    except Exception as e:
        print(f"  [{tile_label(lat, lng)}] 3m resample failed: {e}")
        if already_downsampled:
            # out_path predates this backfill attempt -- a failure
            # here only means the native overlay didn't get backfilled,
            # not that the cached mosaic tile is bad.
            return out_path, '3m'
        out_path.unlink(missing_ok=True)
        return None

    print(f"  [{tile_label(lat, lng)}] 3m source resampled to canonical "
          f"10m grid ({out_path.stat().st_size / 1e6:.1f} MB), "
          f"native kept at {native_path.name}")
    return out_path, '3m'


def download_tile_3dep_1m(lat, lng, output_dir):
    """
    Fetch a tile at TRUE native resolution (~1m/px, ~112000x112000px
    for a 1-degree tile) -- not the ~3m/px download_tile_3dep_3m
    settles for. Much more expensive: probes coverage on a coarse grid
    first (probe_3dep_1m_coverage_grid) so sub-tiles in clearly
    uncovered regions are skipped rather than fetched, then streams
    the rest straight into the output file window-by-window
    (elevation._download_tile_3dep_native) instead of assembling the
    whole tile in memory -- required at this scale (~50GB as a single
    array vs. one sub-tile, ~16MB, at a time).

    Explicit opt-in only (resolution='1m') -- NOT part of --try-3m or
    the 'best' cascade. Even with coverage-skipping, a tile with
    substantial real coverage can still mean thousands of WCS requests
    and tens of minutes.

    Doesn't gap-fill sub-tiles that fail after retries -- unlike
    download_tile_3dep_3m's underlying fetcher, which re-reads the
    whole assembled array to patch gaps from GLO-30 (incompatible with
    staying memory-bounded at this scale). Doesn't need to: this
    raster is only ever consumed as a VRT overlay
    (build_hires_vrt.py) on top of the always-complete standard
    mosaic, and the VRT already treats overlay nodata as transparent
    -- a failed sub-tile just shows the normal mosaic value through
    the VRT instead of a hole, no separate gap-fill required. Only
    someone opening this raw native file directly, bypassing the VRT,
    would ever see a real nodata gap from a failed request.

    Returns (Path, '1m') on success, None if there's no coverage at
    all or the fetch failed.
    """
    lat_floor, lng_floor = math.floor(lat), math.floor(lng)

    if lng_floor >= 0 or lat_floor < 24 or lat_floor > 72:
        return None

    raw_dir = Path(output_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{tile_label(lat, lng)}_1m.tif"
    native_path = raw_dir / f"{tile_label(lat, lng)}_1m_native.tif"

    already_downsampled = (out_path.exists() and out_path.stat().st_size > 10_000
                            and _raster_is_readable(out_path))
    native_ready = native_path.exists() and _raster_is_readable(native_path)

    # Full cache hit -- short-circuit before any network call, and
    # before the coverage probe below, which would otherwise discard
    # an already-valid mosaic tile on a transient probe failure.
    if already_downsampled and native_ready:
        print(f"  [{tile_label(lat, lng)}] Cached (1m)")
        return out_path, '1m'

    # Native raster already fetched, only the resampled mosaic tile is
    # missing (e.g. killed between the two steps on a prior run) --
    # resample straight from the cached native file. No network calls
    # needed at all, unlike the 3m tier's equivalent case: native_path
    # here already *is* the full fetch result, not a separate raw file.
    if native_ready and not already_downsampled:
        try:
            _resample_native_to_canonical_grid(native_path, lat_floor,
                                                lng_floor, out_path)
            print(f"  [{tile_label(lat, lng)}] Resampled cached native "
                  f"(1m) to canonical grid")
            return out_path, '1m'
        except Exception as e:
            print(f"  [{tile_label(lat, lng)}] Resample of cached native "
                  f"failed ({e}) -- refetching")

    # Below this point, a probe/fetch failure only costs the (still
    # missing) native overlay backfill if a valid mosaic tile already
    # exists -- return that cached tile instead of None so a transient
    # failure never discards previously-good output.
    cached_fallback = (out_path, '1m') if already_downsampled else None

    if not _probe_3dep_1m_coverage(lat_floor, lng_floor):
        return cached_fallback

    try:
        from elevation import (_download_tile_3dep_native,
                                NATIVE_1M_GRID_N, NATIVE_1M_SUBTILE_PX,
                                NATIVE_1M_PROBE_GRID_N)
    except Exception as e:
        print(f"  [{tile_label(lat, lng)}] 1m unavailable "
              f"(elevation.py import failed: {e})")
        return cached_fallback

    try:
        coverage = probe_3dep_1m_coverage_grid(
            lat_floor, lng_floor, grid_n=NATIVE_1M_PROBE_GRID_N)
        if not coverage.any():
            print(f"  [{tile_label(lat, lng)}] No coverage found on the "
                  f"coarse probe grid, skipping")
            return cached_fallback

        _, n_fetched, n_failed, _ = _download_tile_3dep_native(
            lat_floor, lng_floor, native_path,
            coverage_grid=(NATIVE_1M_PROBE_GRID_N, coverage),
            grid_n=NATIVE_1M_GRID_N, subtile_px=NATIVE_1M_SUBTILE_PX)
    except Exception as e:
        print(f"  [{tile_label(lat, lng)}] 1m fetch failed: {e}")
        return cached_fallback

    if n_fetched == 0 and n_failed > 0:
        # The coarse probe found coverage, but every covered sub-tile
        # then failed to fetch (WCS outage, rate-limiting, etc.) --
        # native_path is all nodata. Fall through to the normal
        # cascade (or the cached tile, if there already is one)
        # instead of reporting this as a successful '1m' tile;
        # otherwise the caller mosaics an all-nodata raster into the
        # study area instead of retrying at a lower resolution.
        native_path.unlink(missing_ok=True)
        print(f"  [{tile_label(lat, lng)}] 1m fetch found no usable data "
              f"({n_failed} sub-tile(s) failed) -- falling back")
        return cached_fallback

    if already_downsampled:
        print(f"  [{tile_label(lat, lng)}] Native kept at "
              f"{native_path.name} (mosaic tile already cached)")
        return out_path, '1m'

    try:
        _resample_native_to_canonical_grid(native_path, lat_floor, lng_floor,
                                            out_path)
    except Exception as e:
        print(f"  [{tile_label(lat, lng)}] 1m resample failed: {e}")
        out_path.unlink(missing_ok=True)
        return None

    print(f"  [{tile_label(lat, lng)}] 1m resampled to canonical "
          f"10m grid ({out_path.stat().st_size / 1e6:.1f} MB), "
          f"native kept at {native_path.name} "
          f"({native_path.stat().st_size / 1e6:.1f} MB)")
    return out_path, '1m'


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


def download_tile(lat, lng, output_dir, resolution='best', try_3m=False,
                  timeout=120, chunk_mb=1, max_download_s=DEFAULT_MAX_DOWNLOAD_S):
    """
    Download one 1-degree DEM tile at the best available resolution.
    Returns (Path, res_label) on success, None on failure.
    Skips download if file already exists and is valid.

    try_3m: 'best' (the default cascade) does NOT attempt the ~3m
    tier unless this is True. The raw fetch is expensive to store
    even though the mosaic tile itself lands at the same 10m-equivalent
    size as everything else, and every tile build would otherwise pay
    for a 900-request WCS fetch (even a coverage-miss probe still costs
    a request) in areas that mostly don't have 1m coverage at all. Opt
    in explicitly (this flag, or resolution='3m') for a run where the
    accuracy is worth the cost, rather than making it the default.
    Whenever this succeeds, the native ~3m raster is also kept (see
    download_tile_3dep_3m) so build_hires_vrt.py can layer it over the
    final mosaic -- download_study_area() does this automatically.
    """
    # 3m/1m tiers: not simple direct-URL candidates like the others
    # (WCS sub-tile fetch, not one GET), so they're tried as a
    # pre-step rather than folded into get_candidates()'s list.
    #
    # resolution='3m' is explicit-only with no fallback -- matches how
    # explicit '10m'/'30m' already behave (no silent substitution
    # under a mismatched label). It's also never tried under 'best'
    # just from try_3m being set for a run -- try_3m only applies
    # per-tile, same as this whole block.
    if resolution == '3m' or (resolution == 'best' and try_3m):
        result = download_tile_3dep_3m(lat, lng, output_dir)
        if result is not None:
            return result
        if resolution == '3m':
            return None

    # resolution='1m' DOES fall through to the normal 10m/GLO-30
    # cascade below when a given tile has no genuine 1m coverage --
    # unlike '3m'/'10m'/'30m', the point of requesting '1m' for a
    # whole study area is real detail where it exists, not an
    # all-or-nothing demand that leaves gaps in the rest of the area.
    # It never falls through to the 3m tier either way -- that tier is
    # a fetch-shape workaround this project's own engineering limits
    # forced on the ~3m tier, not something '1m' requests should ever
    # pay for on the way to its own, unrelated fallback.
    if resolution == '1m':
        result = download_tile_3dep_1m(lat, lng, output_dir)
        if result is not None:
            return result

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
# Direct narrow-AOI download (bypasses the whole-degree-tile pipeline)
# ─────────────────────────────────────────────

def download_direct_aoi(south, west, north, east, resolution, output_dir):
    """
    Fetch exactly [south, west, north, east] from 3DEP at 'resolution'
    ('1m' or '3m'), writing a single clipped GeoTIFF to output_dir --
    see elevation.fetch_dem_direct() for why this skips the coverage
    probe / whole-degree sub-tile grid that download_tile_3dep_1m/_3m
    always run, no matter how small the caller's actual area is.
    """
    from elevation import fetch_dem_direct

    resolution_m = 1.0 if resolution == '1m' else 3.0
    width_km  = (east - west) * 111.32 * math.cos(math.radians((south + north) / 2))
    height_km = (north - south) * 111.32

    print(f"\n{'='*62}")
    print(f"  Direct AOI fetch")
    print(f"  Bounds: {south}N {west}E  ->  {north}N {east}E")
    print(f"  Size:   {width_km*1000:.0f}m x {height_km*1000:.0f}m")
    print(f"  Res:    ~{resolution_m}m/px")
    print(f"{'='*62}\n")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"direct_{resolution}.tif"

    result = fetch_dem_direct(south, west, north, east, resolution_m, out_path)
    if result is None:
        print(f"\n[DIRECT] No 3DEP {resolution} coverage found for this area")
        sys.exit(1)

    print(f"\n[DIRECT] Wrote {result[0]}")
    return result[0]


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def download_study_area(area, dry_run=False, resume=True,
                        max_workers=4, no_mosaic=False, try_3m=False):
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
                download_tile, lat, lng, area.output_dir,
                resolution=area.resolution, try_3m=try_3m,
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

    if try_3m or area.resolution in ('3m', '1m'):
        # The two tiers write to different native filenames -- match
        # whichever one was actually requested for this area, not both
        # (a run using --try-3m never touches the 1m tier and vice
        # versa, so checking the other suffix would find nothing).
        native_suffix = '1m_native.tif' if area.resolution == '1m' else '3m_native.tif'
        raw_dir = Path(area.output_dir) / 'raw'
        native_paths = [
            raw_dir / f"{tile_label(lat, lng)}_{native_suffix}"
            for lat, lng in tiles
        ]
        native_paths = [p for p in native_paths if _raster_is_readable(p)]
        if not native_paths:
            print("[HIRES-VRT] No tiles got real high-resolution coverage -- skipping")
        elif not mosaic_path.exists():
            print("[HIRES-VRT] Mosaic was not written -- skipping")
        else:
            vrt_path = Path(area.output_dir) / f"{area.name}_mosaic_hires.vrt"
            try:
                build_hires_vrt(mosaic_path, native_paths, vrt_path)
                print(f"[HIRES-VRT] {len(native_paths)} native tile(s) layered "
                      f"onto the mosaic -> {vrt_path}")
            except Exception as e:
                # The mosaic itself is already written and reported by
                # this point -- a bad native overlay file (e.g. left
                # partially written by an interrupted fetch) shouldn't
                # turn an otherwise-successful run into a crash.
                print(f"[HIRES-VRT] Failed to build overlay VRT: {e} "
                      f"-- mosaic is still valid without it")


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
        choices=['best', '3m', '1m', '10m', '30m'],
        help="'3m' is ~3m/px (900 WCS requests/tile), explicit-only, "
             "no fallback if a tile has no coverage (like '10m'/'30m'). "
             "'1m' is genuine ~1m/px (up to ~3100 requests/tile, tens "
             "of minutes even with coverage-skipping); a tile with no "
             "1m coverage falls back to the normal 10m/GLO-30 cascade "
             "so the area mosaic stays complete -- it never falls back "
             "to '3m' though. Neither tier is ever tried under 'best' "
             "or --try-3m without an explicit --resolution request.")
    parser.add_argument('--try-3m', action='store_true',
        help="Also attempt the ~3m tier under --resolution best "
             "(off by default -- coverage is sparse and the raw "
             "fetch is expensive: 900 WCS requests per tile). For any "
             "tile that gets real coverage, the native raster is "
             "kept and layered over the final mosaic as a "
             "<name>_mosaic_hires.vrt -- real fine detail where it was "
             "fetched, the normal 10m mosaic everywhere else. Use "
             "--resolution 1m instead for genuine ~1m/px.")
    parser.add_argument('--direct', action='store_true',
        help="Skip the whole-degree-tile pipeline entirely and fetch "
             "exactly the --bounds area from 3DEP, in as few WCS "
             "requests as the server's own per-request pixel limit "
             "allows -- for a small AOI (a few acres up to a couple "
             "km across) that's one request instead of probing and "
             "fetching a whole 1-degree tile just to crop it down "
             "afterward. Requires --bounds (corner1/corner2 only "
             "resolve to whole-degree tiles) and --resolution 1m or "
             "3m. Writes a single clipped GeoTIFF -- no mosaic, no "
             "manifest, no tile cache.")
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

    if args.direct:
        if not args.bounds:
            parser.error('--direct requires --bounds (fractional-degree '
                          'precision -- corner1/corner2 only resolve to '
                          'whole-degree tiles)')
        if args.resolution not in ('1m', '3m'):
            parser.error("--direct requires --resolution 1m or 3m -- the "
                          "10m/30m/best sources aren't fetchable by "
                          "arbitrary bbox")
        s, w, n, e = [float(x) for x in args.bounds.split(',')]
        download_direct_aoi(
            south=s, west=w, north=n, east=e,
            resolution=args.resolution,
            output_dir=args.output_dir or 'data/dem/direct',
        )
        return

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
        try_3m=args.try_3m,
    )


if __name__ == '__main__':
    main()