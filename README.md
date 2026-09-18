# DEM Download & Tile Pipeline

Downloads, repairs, and mosaics Digital Elevation Model (DEM) data from public
US and global sources, and turns it into a validated, cache-managed tile
corpus.


**New here?** See [QUICKSTART.md](QUICKSTART.md) to get a first download
running. Everything below is the full reference.

## What it does

Two approaches:

**1. Designated mosaic scripts** — download a DEM for a latitude-longitude grid spanning input diagonal corners, then run standalone repair/correction passes
over the result:

| Script | Purpose |
|---|---|
| [`dem_download.py`](python/dem_download.py) | Primary driver script. Downloads and mosaics the best available DEM tiles for an input study area. Default cascade: 3DEP 10m → GLO-30. Two opt-in tiers fetch USGS 3DEP's finest data on top of that — `--try-3m` (really ~3m/px, 900 WCS requests/tile) and `--resolution 1m` (genuine ~1m/px, far more expensive) — see [High-resolution output](#high-resolution-output). `--direct` fetches a small AOI directly instead of a whole tile — see [Direct AOI fetch](#direct-aoi-fetch). |
| [`build_hires_vrt.py`](python/build_hires_vrt.py) | Builds the `.vrt` described above: layers one or more higher-resolution rasters over a lower-resolution base so a single file samples fine detail where available and the base resolution elsewhere. |
| [`visualize_hires_vrt.py`](python/visualize_hires_vrt.py) | Renders a diagnostic figure for a `dem_download.py --try-3m`/`--resolution 1m` study area: a hillshade of the composited VRT plus a per-*tile* coverage map/table — see [Visualizing hi-res coverage](#visualizing-hi-res-coverage). |
| [`visualize_cascade_aoi.py`](python/visualize_cascade_aoi.py) | Same idea for a `--direct` AOI, but per-*pixel* rather than per-tile (the AOI is small enough to check every pixel directly) — see [Direct AOI fetch](#direct-aoi-fetch). |
| [`build_dem_mosaic.py`](python/build_dem_mosaic.py) | Higher-level driver: downloads a source (GLO-30 or 3DEP) and produces one water-corrected mosaic GeoTIFF. |
| [`dem_water_correction.py`](python/dem_water_correction.py) | Flattens elevation noise inside still-water bodies (lakes, ponds, reservoirs) by sampling shoreline elevation and flood-filling, using NHD (US) or OpenStreetMap (global) water polygons. |
| [`repair_dem_gaps.py`](python/repair_dem_gaps.py) | Finds nodata/zero gaps in an already-built mosaic and backfills them from GLO-30, then re-applies water correction. |
| [`precompute_flat_mask.py`](python/precompute_flat_mask.py) | Precomputes a per-pixel "is this terrain locally flat" mask. |
| [`elevation.py`](python/elevation.py) | On-demand single-tile/single-point DEM lookup with its own disk cache — a separate download path from `dem_download.py`'s study-area pipeline, used internally by `build_dem_mosaic.py` and `repair_dem_gaps.py`. Adds sub-tiled 3DEP WCS downloads and a single-point `get_elevation()` that neither of those has. |
| [`gpu_tools.py`](python/gpu_tools.py) | CuPy-backed GPU acceleration helpers (dilation, percentile, water-correction label processing) with automatic fallback to NumPy/SciPy when no GPU is available. |
| [`parallel_tools.py`](python/parallel_tools.py) | Domain-agnostic multiprocessing helpers (shared-memory arrays, chunked/worker-pool map) used by the scripts above. |

**2. Tile-based system** — built on top of the flat scripts to serve terrain
on demand, one 1°×1° tile at a time, with a persistent on-disk cache:

| Module | Purpose |
|---|---|
| [`tile_id.py`](python/tile_id.py) | Pure coordinate math: lat/lng ↔ tile ID (`N45W110`), bounding-box/polyline coverage, neighbor lookup. No I/O. |
| [`tile_registry.py`](python/tile_registry.py) | JSON manifest of every known tile and its build state (`ready` / `pending` / `failed` / `absent`), with atomic save. |
| [`storage_manager.py`](python/storage_manager.py) | Enforces a disk-space cap on the tile cache before each build, with `hard` (refuse), `lru` (evict oldest), or `warn` policies. |
| [`tile_validation.py`](python/tile_validation.py) | Verifies a written DEM raster actually covers its tile's full 1°×1° footprint at the expected resolution — added after a real incident where a silently truncated download sat marked `ready` for 6+ weeks (see the module docstring for the full postmortem). |
| [`tile_builder.py`](python/tile_builder.py) | Orchestrates one tile end-to-end: download → crop/gap-fill → extent validation → water correction → flat mask → boundary mask → registry update. Composes the flat scripts above rather than reimplementing them. |
| [`visualize_tile.py`](python/visualize_tile.py) | Renders a diagnostic figure for a built tile — hillshade/elevation, water polygons, the water-correction diff, flat/boundary coverage, the gap-fill void map, and a correction-magnitude summary. See [Visualizing a tile](#visualizing-a-tile) below. |

**Tile ID convention:** the coordinate in a tile ID is its **southwest
corner** — the tile extends north and east from there. `N45W110` covers
45–46°N by 110–109°W (i.e. north to 46°N, *east* to 109°W, not 111°W —
easy to misread at a glance). `tile_bounds('N45W110')` in
[`tile_id.py`](python/tile_id.py) is the source of truth if you want to
double-check a specific tile.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt
```

(`.venv/bin/pip`/`.venv/bin/python` work as-is in any shell. If you'd
rather activate the venv so plain `python`/`pip` resolve to it, use
`source .venv/bin/activate` in bash/zsh or `source .venv/bin/activate.fish`
in fish.)

GPU acceleration (`gpu_tools.py`) is optional — it falls back to NumPy/SciPy
automatically. To enable it, additionally install a CUDA-matched CuPy build,
e.g. `pip install cupy-cuda12x`.

## Usage

```bash
# Download + mosaic a whole area at the best available resolution.
# The area is the rectangle spanning any two opposite 1°x1° tile
# corners (see "Tile ID convention" above) -- order doesn't matter.
python python/dem_download.py N44W113 N47W109
python python/dem_download.py --bounds 44,-113,47,-109  # fractional-degree alternative

# Or, via the higher-level mosaic driver
python python/build_dem_mosaic.py N44W113 N47W109
python python/build_dem_mosaic.py --bounds 44 -113 47 -109 --output my_area.tif

# Same, but also fetch real fine-detail coverage where it exists --
# see "High-resolution output" below.
python python/dem_download.py N44W113 N47W109 --try-3m

# Build one 1°x1° tile end-to-end into a local tile cache. --storage-root
# is the cache root, not the tiles dir itself — the tile lands at
# data/tiles/N45W110/ (registry + downloads/ also live under data/).
# N45W110 names the tile's SW corner (45N, 110W) — see "Tile ID
# convention" above.
python python/tile_builder.py build N45W110 --storage-root data

# Same, but also keep the pre-correction/pre-fill artifacts that
# visualize_tile.py's diff panels need (see below) — costs extra disk
python python/tile_builder.py build N45W110 --storage-root data --keep-original

# Repair nodata gaps in an existing mosaic
python python/repair_dem_gaps.py N44W113 N47W109 --dem-path my_area.tif --dry-run

# Correct water-body elevation noise in a single tile
python python/dem_water_correction.py correct \
    --input data/dem/raw/N45_00_W111.tif \
    --output data/dem/corrected/N45_00_W111.tif \
    --source osm
```

Areas built from two tile corners are capped at 25 degree-tiles by
default (`--allow-large` to override) — `dem_download.py`, `build_dem_mosaic.py`,
and `repair_dem_gaps.py` all warn/refuse before a mosaic gets large
enough to be a real time/disk commitment; each 3DEP 10m tile runs
roughly 200-300MB. See `bounds_from_tile_corners()` in `tile_id.py`.

Run any script with `--help` for its full option list — all of them are
self-documenting argparse CLIs.

## High-resolution output

Two opt-in tiers fetch USGS 3DEP's finest data wherever it's actually
surveyed (coverage is sparse — most areas have none):

| | `--try-3m` | `--resolution 1m` |
|---|---|---|
| Actual resolution | ~3m/px | genuine ~1m/px |
| Requests per tile | 900 | up to ~3,100 (coverage-probed first, so mostly-uncovered tiles cost far less) |
| Time per tile | a few minutes | tens of minutes for a well-covered tile |

`--try-3m` requests a 30×30 grid of sub-tiles (900 requests) rather than the
true ~111000×111000px native grid, which isn't practical to request this way
— so what comes back is ~3m/px, not literally 1m/px, despite pulling from
USGS's 3DEP "1m" product. Deliberate tradeoff, not a WCS server limit.

`--resolution 1m` goes further: genuine ~1m/px, streamed straight to disk one
piece at a time instead of assembled in memory (the naive approach needs
~50GB of RAM for one tile). A coarse coverage probe runs first so a
mostly-uncovered tile doesn't pay for thousands of pointless requests. It's a
separate, more expensive tier — never triggered by `--try-3m` or the default
`best` cascade, and never uses `--try-3m` as an intermediate step. A tile
with no 1m coverage falls back to the normal 10m/GLO-30 cascade rather than
leaving a hole in the area mosaic.

Whichever tier finds real coverage, `dem_download.py` writes one extra file
next to the usual mosaic: `<name>_mosaic_hires.vrt`. Open it exactly like a
GeoTIFF (QGIS, `gdalinfo`, `rasterio.open(...)`) — it reads real high-resolution
detail wherever it was fetched and the normal mosaic everywhere else, with no
resampling or extra storage cost. It's a small text file that references the
mosaic `.tif` and the native tile(s) by relative path, so keep them together
in the same folder.

Not wired up yet: `tile_builder.py`'s `--source` doesn't expose either
high-resolution tier, and `build_dem_mosaic.py` doesn't have `--try-3m` either.

## Direct AOI fetch

For an area much smaller than a full 1° tile — a few acres up to a couple km
across — `--direct` skips the whole-tile pipeline entirely and fetches
exactly the requested `--bounds`, in as few requests as each source's own
per-request limit allows (often just one). By default it also crops the 10m
3DEP and 30m GLO-30 base tiers for the same bounds (windowed range-reads
against the live S3 sources — no full-tile download) and composites all
three into one VRT: real detail where 3DEP has it, 10m/30m filling any gap
in it — the same fallback priority the whole-tile pipeline uses, scoped to
just the requested area.

```bash
python python/dem_download.py --bounds "44.50000,-110.20000,44.50128,-110.19873" \
    --resolution 1m --direct
```

Requires `--bounds` and `--resolution 1m`/`3m` (the tier to fetch). Writes
`cascade_manifest.json` plus whichever of
`native_<res>.tif`/`base_10m.tif`/`base_30m.tif` actually returned data —
render it with `visualize_cascade_aoi.py` for a per-pixel resolution map.

Add `--top-only` to fetch just the `--resolution` tier — no 10m/30m crops, no
VRT, one clipped GeoTIFF, 2 fewer requests. The tradeoff: any gap in that
tier's own coverage is left as real nodata instead of being filled, since
filling it needs exactly the extra fetches this flag skips.

Known limitation: the 10m/30m crops pick a single source tile from the AOI's
center point, so an AOI straddling a whole-degree line (rare, but it can
coincide with an international border) may miss data just past that seam.

## Visualizing hi-res coverage

```bash
python python/visualize_hires_vrt.py data/dem/N45W111_N45W111
```

Reads a `dem_download.py` study area's `download_manifest.json` (no network,
no re-reading the source rasters at full resolution) and renders a three-panel
PNG: a hillshade of the composited `_mosaic_hires.vrt` (or the plain mosaic,
if no hi-res tier was ever requested for this area), a per-tile map of which
source each tile actually used (GLO-30/3DEP 10m/`--try-3m`/`--resolution 1m`),
and a table of per-tile stats.

A tile's resolution tier is a whole-tile choice, so the map can't show
sub-tile variation — `coverage_pct` in the table is how much of that tile's
own area is real dense native data rather than filled/absent, which is the
number that actually tells you how much fine detail a mostly-uncovered tile
has. For an AOI small enough to check every pixel directly instead, see
`visualize_cascade_aoi.py` above.

## Visualizing a tile

```bash
python python/visualize_tile.py N45W110 --storage-root data
```

Reads a tile on disk under the directory and renders a
one-page PNG report (default: `<tile_dir>/<tile_id>_report.png`). Needs `rasterio` and `matplotlib`, no network access, no scipy/
shapely/GPU deps.

Every panel is optional except the base hillshade — a panel is skipped
(with a one-line explanation printed to the console) rather than faked
when its input artifact isn't there:

| Panel | Needs |
|---|---|
| Hillshade + elevation | `{tile}_dem.tif` (the only required file) |
| Water polygons, colored by source | `{tile}_water.geojson` |
| Flat mask + boundary overlay | `{tile}_flat.tif`, `{tile}_boundary.tif` |
| Elevation diff (corrected − original) | `{tile}_dem_precorrection.tif` † |
| Void map (gap-fill before/after) | `{tile}_gap_mask.tif` † |
| Correction magnitude histogram + top-N table | `{tile}_water_stats.json` † |

† `tile_builder.py`'s water-correction and gap-fill steps normally run
in place — there's no "before" raster left over once a tile is built.
Pass `--keep-original` at build time (see above) to have `tile_builder.py`
save these three extra artifacts alongside the normal ones.

**Example**, from a real tile (`N45W111`, Yellowstone/Hebgen Lake area,
NHD source, `--keep-original`):

![Example visualize_tile.py report](docs/example_report.png)

The top-10 table exists to give visibility into how large water corrections
get. An earlier build of this same tile is what surfaced a real bug: an
8-pixel sliver of misclassified "water" on a steep hillside was getting a
**-21.1m** correction, because its shoreline ring sampled straight down the
slope instead of an actual shoreline. Fixed in both `dem_water_correction.py`
and `.R` by gating on the resulting correction magnitude rather than trusting
every water polygon's own geometry — corrections over `max_correction_m`
(default 10m) are now skipped instead of applied, and corrections built from
a water body whose *own* raw elevation was already noisy are applied but
marked `low_confidence` rather than silently treated the same as a clean
one. The `midpoint (lat, lon)` column still exists for spot-checking any
correction directly.

## Tests

Each module has a matching `test_*.py` that runs standalone (no pytest
required):

```bash
cd python
python test_tile_id.py
python test_tile_registry.py
python test_storage_manager.py
python test_tile_validation.py
python test_tile_builder.py
python test_dem_download_3m.py
python test_dem_download_1m.py
python test_build_hires_vrt.py
python test_visualize_hires_vrt.py
```

`test_tile_builder.py` stubs out the flat pipeline scripts (`dem_download`,
`precompute_flat_mask`, etc.) so the tile-orchestration logic can be tested
without a network connection or real DEM data.

## R port

[`r/`](r/) has an R translation of the core DEM science — download,
mosaic, still-water correction, flat-mask computation, and the
on-demand single-tile lookup in `elevation.py` — built on `terra` and
`httr2`. It's a deliberately narrower port than the Python original
(no tile-cache infrastructure, no GPU/parallel paths — see
[`r/README.md`](r/README.md) for the exact scope and why), verified
against real NHD/OSM API calls, real downloads, and a real R-vs-Python
mosaic diff for the same area. Translating it surfaced several real bugs, all documented in
[`r/README.md`](r/README.md#real-bugs-found-while-writing-this-port)
with the live incident that caught each one — most notably a
truncated-download detection gap that **also existed in this repo's
own Python original** and a 3DEP
sub-tile grid-alignment issue that, before the fix, made the two
ports' output for the same real area disagree on a majority of pixels.

## Data sources

All public, no API keys or credentials required:

- [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) (1m / 10m, US
  coverage) via the National Map API and public S3 bucket
- [Copernicus GLO-30](https://registry.opendata.aws/copernicus-dem/) (30m,
  global) via public S3
- [USGS National Hydrography Dataset](https://www.usgs.gov/national-hydrography)
  (water polygons, US)
- [OpenStreetMap](https://www.openstreetmap.org/) via the Overpass API
  (water polygons, global)
