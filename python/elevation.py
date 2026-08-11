import os
import math
import requests
import numpy as np
import rasterio
import rasterio.io
from rasterio.merge import merge
from rasterio.crs import CRS
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()

# Matches the batch pipeline's --storage-root convention (sibling
# scripts outside this repo) so this cache lands on the same configured
# data drive, not this repo's own directory. The path below is a
# generic placeholder -- point STORAGE_ROOT (or DEM_CACHE_DIR directly)
# at wherever your own data drive/volume actually is.
DEFAULT_STORAGE_ROOT = "D:\\dem_data"

# This cache never expires anything on its own (see
# _enforce_cache_cap below) and previously defaulted to a
# repo-relative path, meaning it grew unbounded directly on the OS/
# boot drive on Windows dev machines -- 17.8GB accumulated there
# before this was caught, contributing to a C-drive-fill crash on
# 2026-08-05 (a sibling script's --routes-dir had the
# identical repo-relative-default bug, fixed the same day). Redirects
# under STORAGE_ROOT (or DEFAULT_STORAGE_ROOT) on Windows when that
# drive is actually present; otherwise falls back to the original
# repo-relative location -- covers Railway's Linux container (no O:
# drive, ephemeral filesystem, never a real risk there) and any
# Windows machine without the data drive mounted. DEM_CACHE_DIR
# always wins outright.
def _resolve_cache_dir() -> Path:
    explicit = os.environ.get("DEM_CACHE_DIR")
    if explicit:
        return Path(explicit)
    if os.name == "nt":
        storage_root = Path(os.environ.get("STORAGE_ROOT", DEFAULT_STORAGE_ROOT))
        if storage_root.drive and Path(storage_root.drive + "\\").exists():
            return storage_root / "dem_cache"
    return Path(__file__).parent / "dem_cache"

CACHE_DIR = _resolve_cache_dir()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Total on-disk cap for CACHE_DIR, enforced by _enforce_cache_cap()
# after each new tile write. 0 disables the cap.
DEM_CACHE_MAX_GB = float(os.environ.get("DEM_CACHE_MAX_GB", "20"))


def _touch(path: Path) -> None:
    """Bumps a cache file's mtime on reuse, so LRU eviction in
    _enforce_cache_cap() ranks by last-used rather than download
    date."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def _enforce_cache_cap(protect: Path = None) -> None:
    """Deletes least-recently-used tiles (by mtime) until CACHE_DIR is
    back under DEM_CACHE_MAX_GB. `protect` (the tile just written) is
    never evicted, even if it alone exceeds the cap -- otherwise a
    single large tile would be deleted and immediately re-downloaded
    on the next request. Best-effort: any OSError (permission, race
    with another process) just leaves that file in place rather than
    raising -- this is disk hygiene, not allowed to break a real
    request over it."""
    if DEM_CACHE_MAX_GB <= 0:
        return
    cap_bytes = DEM_CACHE_MAX_GB * 1e9
    try:
        all_files = [f for f in CACHE_DIR.iterdir() if f.is_file()]
    except OSError:
        return
    total = sum(f.stat().st_size for f in all_files)
    if total <= cap_bytes:
        return
    evictable = sorted(
        (f for f in all_files if f != protect),
        key=lambda f: f.stat().st_mtime,
    )
    for f in evictable:
        if total <= cap_bytes:
            break
        try:
            size = f.stat().st_size
            f.unlink()
            total -= size
            print(f"[DEM cache] evicted {f.name} ({size / 1e6:.0f}MB, "
                  f"LRU) -- cache over {DEM_CACHE_MAX_GB:.0f}GB cap")
        except OSError:
            pass

SOURCES = {
    'GLO30':   30,  # Copernicus, global
    '3DEP_10': 10,  # USGS, US only
    '3DEP_1':   1,  # USGS, US only, large files
}



# Default source — can be overridden per-request
DEFAULT_SOURCE = os.environ.get("DEM_SOURCE", "auto")

# USGS 3DEP bounding box — continental US + Alaska + Hawaii
_3DEP_BOUNDS = {
    "min_lat": 17.0, "max_lat": 72.0,
    "min_lng": -180.0, "max_lng": -60.0,
}

def _is_in_3dep_coverage(lat: float, lng: float) -> bool:
    """Check if coordinates fall within USGS 3DEP coverage area."""
    b = _3DEP_BOUNDS
    return (b["min_lat"] <= lat <= b["max_lat"] and
            b["min_lng"] <= lng <= b["max_lng"])


def _raster_is_readable(path) -> bool:
    """
    True if `path` opens AND its pixel data can actually be decoded --
    not just that the header parses. A truncated GeoTIFF (an
    interrupted download, or a killed process mid-write) commonly
    still has an intact, readable header, so a bare `rasterio.open()`
    would call that "valid" even though the pixel data is corrupt.
    Local copy of dem_download.py's `_raster_is_readable()` --
    duplicated rather than imported, so this module doesn't pull in
    dem_download.py's whole top-level import surface just for this.

    Real incident (2026-08-10): a stale, truncated GLO-30 cache file
    (from an earlier interrupted run during the R port's testing) sat
    unnoticed because neither of this module's two cache checks (here
    and _download_tile_3dep_subtiled()'s) verified readability before
    reusing a cached path -- it surfaced downstream as a hard
    reproject failure in one run and, worse, as silently-still-missing
    output pixels (despite the log claiming a successful GLO-30 fill)
    in another, on the R side. Applied at both cache checks here too,
    since this module had the identical gap.
    """
    try:
        with rasterio.open(path) as src:
            for _, window in src.block_windows(1):
                src.read(1, window=window)
        return True
    except Exception:
        return False

# ─────────────────────────────────────────────
# GLO-30 (Copernicus)
# ─────────────────────────────────────────────

def get_tile_name(lat: float, lng: float) -> str:
    lat_floor = math.floor(lat)
    lng_floor = math.floor(lng)
    lat_str = f"N{abs(lat_floor):02d}" if lat_floor >= 0 else f"S{abs(lat_floor):02d}"
    lng_str = f"E{abs(lng_floor):03d}" if lng_floor >= 0 else f"W{abs(lng_floor):03d}"
    return f"{lat_str}_00_{lng_str}"

def get_tile_url(lat: float, lng: float) -> str:
    tile = get_tile_name(lat, lng)
    return (
        f"https://copernicus-dem-30m.s3.amazonaws.com/"
        f"Copernicus_DSM_COG_10_{tile}_00_DEM/"
        f"Copernicus_DSM_COG_10_{tile}_00_DEM.tif"
    )

def download_tile_glo30(lat: float, lng: float) -> Path:
    tile_name = get_tile_name(lat, lng)
    local_path = CACHE_DIR / f"{tile_name}.tif"
    if local_path.exists():
        if _raster_is_readable(local_path):
            print(f"Using cached tile: {tile_name}")
            _touch(local_path)
            return local_path
        print(f"Cached tile {tile_name} is corrupt/truncated -- re-downloading")
        local_path.unlink()
    url = get_tile_url(lat, lng)
    print(f"Downloading tile: {tile_name} from {url}")
    response = requests.get(url, stream=True, timeout=60)
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {tile_name} "
              f"({local_path.stat().st_size / 1024 / 1024:.1f} MB)")
        _enforce_cache_cap(protect=local_path)
        return local_path
    else:
        raise ValueError(
            f"Could not download GLO-30 tile for lat={lat}, lng={lng}. "
            f"Status: {response.status_code}"
        )

# ─────────────────────────────────────────────
# 3DEP (USGS) — 1m and 10m
# ─────────────────────────────────────────────

def _get_3dep_url(lat: float, lng: float, resolution: int) -> str:
    """
    Build USGS 3DEP WCS request URL for a 1°×1° tile.
    Coverage name: DEP3Elevation (not the old 3DEPElevation:Xmeter format)
    Resolution: 10m → WIDTH/HEIGHT=360, 1m → 3600
    """
    lat_floor = math.floor(lat)
    lng_floor = math.floor(lng)

    # WCS 1.0.0 — coverage name confirmed working April 2026
    base = (
        "https://elevation.nationalmap.gov/arcgis/services/"
        "3DEPElevation/ImageServer/WCSServer"
    )
    if resolution == 1:
        width = height = 108000  # 1m: 1/108000° ≈ 1m at equator
    else:
        width = height = 10800   # 10m: 1/3 arc-second = 10800 per degree

    params = (
        f"?SERVICE=WCS"
        f"&VERSION=1.0.0"
        f"&REQUEST=GetCoverage"
        f"&COVERAGE=DEP3Elevation"
        f"&CRS=EPSG:4326"
        f"&BBOX={lng_floor},{lat_floor},{lng_floor+1},{lat_floor+1}"
        f"&WIDTH={width}"
        f"&HEIGHT={height}"
        f"&FORMAT=GeoTIFF"
    )
    return base + params


def _download_tile_3dep_subtiled(
    lat_floor: int,
    lng_floor: int,
    resolution: int = 10,
    grid_n: int = 9,
    subtile_px: int = 1200,
    max_workers: int = 4,
    timeout: int = 120,
) -> Path:
    """
    Download a full 1°×1° 3DEP tile by requesting a grid_n×grid_n
    grid of sub-tiles, each subtile_px×subtile_px pixels, then
    merging into a single GeoTIFF.

    Default: 9×9 grid of 1200×1200 px sub-tiles = 10800×10800 total
    = 1/3 arc-second (~10m) resolution.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    

    res_tag   = f"3dep_{resolution}m"
    lat_str   = f"N{abs(lat_floor):02d}" if lat_floor >= 0 \
                else f"S{abs(lat_floor):02d}"
    lng_str   = f"W{abs(lng_floor):03d}" if lng_floor < 0 \
                else f"E{abs(lng_floor):03d}"
    tile_name = f"{lat_str}_{lng_str}_{res_tag}"
    local_path = CACHE_DIR / f"{tile_name}.tif"

    if local_path.exists():
        if _raster_is_readable(local_path):
            print(f"Using cached tile: {tile_name}")
            _touch(local_path)
            return local_path
        print(f"Cached tile {tile_name} is corrupt/truncated -- re-downloading")
        local_path.unlink()

    print(f"[3DEP] Downloading {tile_name} as "
          f"{grid_n}×{grid_n} sub-tile grid "
          f"({grid_n*grid_n} requests, ~{subtile_px}×{subtile_px}px each)...")

    step = 1.0 / grid_n  # degrees per sub-tile

    # Build list of (row, col, bbox) for all sub-tiles
    subtiles = []
    for row in range(grid_n):
        for col in range(grid_n):
            xmin = lng_floor + col * step
            xmax = lng_floor + (col + 1) * step
            ymin = lat_floor + row * step
            ymax = lat_floor + (row + 1) * step
            subtiles.append((row, col, xmin, ymin, xmax, ymax))

    base_url = (
        "https://elevation.nationalmap.gov/arcgis/services/"
        "3DEPElevation/ImageServer/WCSServer"
        "?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
        "&COVERAGE=DEP3Elevation&CRS=EPSG:4326&FORMAT=GeoTIFF"
    )

    def fetch_subtile(args):
        row, col, xmin, ymin, xmax, ymax = args
        url = (base_url +
               f"&BBOX={xmin},{ymin},{xmax},{ymax}"
               f"&WIDTH={subtile_px}&HEIGHT={subtile_px}")
        for attempt in range(5):
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 200:
                    data = r.content
                    if data[:4] in (b'II*\x00', b'MM\x00*', b'II+\x00'):
                        return (row, col, data)
                print(f"[3DEP] Sub-tile ({row},{col}) "
                      f"attempt {attempt+1} status={r.status_code}")
            except Exception as e:
                print(f"[3DEP] Sub-tile ({row},{col}) "
                      f"attempt {attempt+1} error: {e}")
            time.sleep(1 + attempt * 2)  # exponential backoff
        return (row, col, None)

    # Download all sub-tiles in parallel
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_subtile, st): st
                   for st in subtiles}
        completed = 0
        for future in as_completed(futures):
            row, col, data = future.result()
            completed += 1
            if data is not None:
                results[(row, col)] = data
            if completed % 9 == 0 or completed == len(subtiles):
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta  = (len(subtiles) - completed) / max(rate, 0.001)
                print(f"[3DEP] {completed}/{len(subtiles)} sub-tiles "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    if len(results) < len(subtiles):
        failed = len(subtiles) - len(results)
        print(f"[3DEP] Warning: {failed} sub-tiles failed — "
              f"tile may have gaps")
        if len(results) == 0:
            print(f"[3DEP] All sub-tiles failed, falling back to GLO-30")
            return download_tile_glo30(lat_floor + 0.5, lng_floor + 0.5)

    

    # Merge sub-tiles into single GeoTIFF
    print(f"[3DEP] Merging {len(results)} sub-tiles...")
    open_memfiles = []
    open_datasets = []

    for (row, col), data in sorted(results.items()):
        mf = rasterio.io.MemoryFile(data)
        ds = mf.open()
        open_memfiles.append(mf)
        open_datasets.append(ds)

    mosaic, transform = merge(open_datasets)

    # Snap onto the canonical tile grid (exact bounds, exact
    # grid_n*subtile_px pixels) rather than trusting merge()'s own
    # extent/resolution inference from the 81 sub-tiles' individually
    # returned bounds. A live WCS server doesn't guarantee a sub-tile's
    # *returned* georeferencing lands exactly on the *requested* bbox
    # to the bit. This happened to round back to exactly 10800x10800
    # in testing (rasterio.merge()'s output size is computed by
    # rounding bounds-span/resolution, which absorbed the sub-pixel
    # noise correctly that time) -- but nothing about that call
    # guarantees it, and the R port's identical, unsnapped merge came
    # out 10801x10811 with non-square pixels on the exact same live
    # tile, confirming this isn't safe to leave implicit.
    # Resampling.nearest because source and target resolution are for
    # all practical purposes identical -- this is a grid snap, not a
    # real resampling.
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
    from rasterio.enums import Resampling
    target_shape = (grid_n * subtile_px, grid_n * subtile_px)
    target_transform = from_bounds(
        lng_floor, lat_floor, lng_floor + 1, lat_floor + 1,
        target_shape[1], target_shape[0])
    snapped = np.zeros((1,) + target_shape, dtype=mosaic.dtype)
    reproject(
        source=mosaic, destination=snapped,
        src_transform=transform, src_crs='EPSG:4326',
        dst_transform=target_transform, dst_crs='EPSG:4326',
        resampling=Resampling.nearest,
    )
    mosaic, transform = snapped, target_transform

    profile = open_datasets[0].profile.copy()
    profile.update({
        'height':    mosaic.shape[1],
        'width':     mosaic.shape[2],
        'transform': transform,
        'compress':  'deflate',
        'predictor': 2,
        'tiled':     True,
        'blockxsize': 512,
        'blockysize': 512,
        # Explicit, not GDAL's BIGTIFF=IF_NEEDED default -- see
        # build_dem_mosaic.py's identical comment. Mainly matters here
        # for the 3DEP_1 (1m) path: a 30x30 sub-tile grid merges to
        # ~5GB uncompressed for a single 1-degree tile, over the
        # classic-TIFF limit the same way the mosaic write was.
        'BIGTIFF':   'YES',
    })

    with rasterio.open(local_path, 'w', **profile) as dst:
        dst.write(mosaic)

    for ds in open_datasets:
        ds.close()
    for mf in open_memfiles:
        mf.close()
    del mosaic

    # ── Gap fill from GLO30 ───────────────────────────────────────
    if len(results) < len(subtiles):
        print(f"[3DEP] Filling {len(subtiles)-len(results)} gap "
              f"sub-tiles from GLO30...")
        try:
            glo_path = download_tile_glo30(
                lat_floor + 0.5, lng_floor + 0.5)

            with rasterio.open(local_path) as dep_ds:
                dep_data  = dep_ds.read(1)
                dep_nodata = dep_ds.nodata if dep_ds.nodata is not None \
                             else -9999.0
                dep_transform = dep_ds.transform
                dep_profile   = dep_ds.profile.copy()

            with rasterio.open(glo_path) as glo_ds:
                from rasterio.enums import Resampling
                from rasterio.warp import reproject
                glo_resampled = np.zeros_like(dep_data, dtype=np.float32)
                reproject(
                    source=rasterio.band(glo_ds, 1),
                    destination=glo_resampled,
                    src_transform=glo_ds.transform,
                    src_crs=glo_ds.crs,
                    dst_transform=dep_transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.bilinear,
                )

            # Fill nodata cells in 3DEP with GLO30 values
            gap_mask = (dep_data == dep_nodata) | (dep_data == 0)
            dep_data[gap_mask] = glo_resampled[gap_mask]
            filled_count = int(gap_mask.sum())

            dep_profile.update({'nodata': dep_nodata, 'BIGTIFF': 'YES'})
            with rasterio.open(local_path, 'w', **dep_profile) as dst:
                dst.write(dep_data, 1)

            print(f"[3DEP] Gap filled {filled_count:,} pixels from GLO30")
        except Exception as e:
            print(f"[3DEP] Gap fill warning: {e}")


    # ── Always check for nodata pixels and fill from GLO30 ────────
    print(f"[3DEP] Checking for nodata pixels in merged tile...")
    try:
        with rasterio.open(local_path) as dep_ds:
            dep_data      = dep_ds.read(1)
            dep_nodata    = dep_ds.nodata if dep_ds.nodata is not None \
                            else -9999.0
            dep_transform = dep_ds.transform
            dep_profile   = dep_ds.profile.copy()

        gap_mask = (dep_data <= dep_nodata + 1) | (dep_data == 0)
        n_gaps   = int(gap_mask.sum())

        if n_gaps > 0:
            print(f"[3DEP] Found {n_gaps:,} nodata pixels "
                  f"({n_gaps / dep_data.size * 100:.1f}%) — "
                  f"filling from GLO30...")
            glo_path = download_tile_glo30(
                lat_floor + 0.5, lng_floor + 0.5)

            with rasterio.open(glo_path) as glo_ds:
                from rasterio.warp import reproject
                from rasterio.enums import Resampling
                glo_resampled = np.zeros_like(dep_data, dtype=np.float32)
                reproject(
                    source=rasterio.band(glo_ds, 1),
                    destination=glo_resampled,
                    src_transform=glo_ds.transform,
                    src_crs=glo_ds.crs,
                    dst_transform=dep_transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.bilinear,
                )

            dep_data[gap_mask] = glo_resampled[gap_mask]
            dep_profile.update({'nodata': dep_nodata, 'BIGTIFF': 'YES'})

            with rasterio.open(local_path, 'w', **dep_profile) as dst:
                dst.write(dep_data, 1)

            print(f"[3DEP] Filled {n_gaps:,} pixels from GLO30")
        else:
            print(f"[3DEP] No nodata gaps found")

    except Exception as e:
        print(f"[3DEP] Gap fill warning: {e}")

    size_mb = local_path.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"[3DEP] Saved {tile_name} "
          f"({size_mb:.1f}MB, {elapsed:.0f}s total)")
    _enforce_cache_cap(protect=local_path)
    return local_path

def download_tile_3dep(lat: float, lng: float, resolution: int = 10) -> Path:
    if not _is_in_3dep_coverage(lat, lng):
        print(f"[3DEP] Outside coverage, falling back to GLO-30")
        return download_tile_glo30(lat, lng)

    lat_floor = math.floor(lat)
    lng_floor = math.floor(lng)

    if resolution == 10:
        return _download_tile_3dep_subtiled(lat_floor, lng_floor,
                                            resolution=10)
    else:
        # 1m — smaller area per request, larger grid needed
        return _download_tile_3dep_subtiled(lat_floor, lng_floor,
                                            resolution=1,
                                            grid_n=30,
                                            subtile_px=1200)
    
# ─────────────────────────────────────────────
# Auto select source
# ─────────────────────────────────────────────
def _auto_select_source(lat: float, lng: float) -> str:
    """
    Auto-select DEM source based on coordinates.
    Uses 3DEP_10 for US coverage, GLO30 elsewhere.
    """
    if _is_in_3dep_coverage(lat, lng):
        return '3DEP_10'
    return 'GLO30'
# ─────────────────────────────────────────────
# Unified download interface
# ─────────────────────────────────────────────

def download_tile(
    lat: float,
    lng: float,
    source: str = None,
) -> Path:
    """
    Download a DEM tile from the specified source.
    source: 'GLO30', '3DEP_10', or '3DEP_1'
    Defaults to DEFAULT_SOURCE environment variable or 'GLO30'.
    """
    src = source or DEFAULT_SOURCE

    if src == 'auto' or src is None:
        src = _auto_select_source(lat, lng)

    if src == '3DEP_1':
        return download_tile_3dep(lat, lng, resolution=1)
    elif src == '3DEP_10':
        return download_tile_3dep(lat, lng, resolution=10)
    else:
        return download_tile_glo30(lat, lng)


# ─────────────────────────────────────────────
# Ocean / water mask
# ─────────────────────────────────────────────

# Rough continental bounding boxes — points clearly inside these
# are candidates for land; points outside are checked more carefully.
# This is a fast pre-filter, not a precise coastline.
_LAND_BBOX = [
    # (min_lat, max_lat, min_lng, max_lng)
    (-56,  72,  -168,  -52),   # Americas
    ( 34,  72,   -25,   60),   # Europe / North Africa
    (-35,  38,    -20,  52),   # Africa
    ( 12,  82,    25,  180),   # Asia
    (-48,  -9,   110,  180),   # Australia / Pacific
    (-90, -60,  -180,  180),   # Antarctica
]

def _is_probably_ocean(lat: float, lng: float) -> bool:
    """
    Fast heuristic check — returns True if the point is very likely
    open ocean with no meaningful terrain.
    Not a precise coastline test; used only to catch clearly bad
    elevation readings over deep water.
    """
    for min_lat, max_lat, min_lng, max_lng in _LAND_BBOX:
        if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
            return False  # inside a land bounding box
    return True  # outside all land bboxes — probably ocean

def _elevation_is_suspect(
    elevation: float,
    lat: float,
    lng: float,
    nodata,
) -> bool:
    """
    Returns True if the elevation value should be treated as 0
    (sea level) rather than trusted as terrain.

    Catches:
      - nodata values
      - physically implausible values
      - small positive noise over open ocean (GLO-30 artifact)
      - small negative values over coastal water
    """
    # Explicit nodata
    if nodata is not None and elevation == nodata:
        return True

    # Physically implausible
    if elevation < -500 or elevation > 9000:
        return True

    # Small noise over open ocean — GLO-30 commonly returns
    # values in the range -5 to +15m over water due to
    # wave height, tidal variation, and interpolation artifacts
    if _is_probably_ocean(lat, lng):
        if -10 <= elevation <= 20:
            return True

    return False




# ─────────────────────────────────────────────
# Elevation sampling
# ─────────────────────────────────────────────

def get_elevation(
    lat: float,
    lng: float,
    source: str = None,
) -> float:
    """
    Get terrain elevation at a point.
    source: 'GLO30', '3DEP_10', '3DEP_1', 'auto', or None
    None uses DEFAULT_SOURCE env var, defaulting to 'auto'.
    """
    tile_path = download_tile(lat, lng, source=source)
    with rasterio.open(tile_path) as dataset:
        coords = [(lng, lat)]
        values = list(dataset.sample(coords))
        elevation = float(values[0][0])
        print(f"Raw elevation value: {elevation}, nodata: {dataset.nodata}")

        if _elevation_is_suspect(elevation, lat, lng, dataset.nodata):
            print(f"Elevation suspect at ({lat}, {lng}): "
                  f"{elevation}m → returning 0.0 (sea level)")
            return 0.0

        return round(elevation, 1)

# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_points = [
        ("Billings, MT",    45.7833,  -108.5007),
        ("Denver, CO",      39.7392,  -104.9903),
        ("Death Valley, CA", 36.5323, -116.9325),
        ("Mount Everest",   27.9881,   86.9250),
    ]
    for source in ["GLO30", "3DEP_10"]:
        print(f"\n=== Source: {source} ===")
        for name, lat, lng in test_points:
            try:
                elev = get_elevation(lat, lng, source=source)
                print(f"  {name}: {elev}m")
            except Exception as e:
                print(f"  {name}: Error — {e}")