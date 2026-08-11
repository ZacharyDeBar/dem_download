# R port

An R translation of the core DEM processing pipeline from
[`python/`](../python/) — the download, mosaic, and correction logic,
not the tile-cache infrastructure or GPU/parallel plumbing. See
[Scope](#scope) below for exactly what that means and why.

## Setup

```bash
Rscript -e 'install.packages(c("terra", "httr2", "jsonlite", "maps"))'
```

`terra` handles all raster I/O and
processing; `httr2` handles the NHD/OSM/USGS/Copernicus HTTP calls;
`maps` backs the ocean-tile detection. `sf` is optional — see
[Dependency notes](#dependency-notes).

## Usage

```bash
cd r
# Area = the rectangle spanning any two opposite 1x1-degree tile
# corners (order doesn't matter) -- see bounds_from_tile_corners() in
# tile_id.R. Requests over 25 degree-tiles are refused by default
# (allow_large=TRUE to override); each 3DEP 10m tile runs ~200-300MB.
Rscript -e 'source("dem_download.R"); download_study_area(study_area_between("N44W113", "N47W109"), dry_run=TRUE)'
Rscript -e 'source("dem_water_correction.R"); correct_dem_water_bodies("in.tif", "out.tif", fetch_water_bodies_nhd(c(45,-111,46,-110)))'
Rscript -e 'source("precompute_flat_mask.R"); compute_flat_mask("dem.tif", "flat.tif")'
Rscript test_tile_id.R   # the only file with a ported test suite so far
```

These files are meant to be `source()`d and called from an R session
or script, not run as standalone CLIs the way the Python originals
are — no `optparse`/argparse-equivalent CLI wrapper was added, to keep
the dependency list to exactly the four packages above.

## Files

| File | Translates | Notes |
|---|---|---|
| [`tile_id.R`](tile_id.R) | `tile_id.py` | Full port, full test suite ported ([`test_tile_id.R`](test_tile_id.R)), all 14 cases pass. |
| [`dem_download.R`](dem_download.R) | `dem_download.py` | Sequential downloads (see below); ocean check via `maps` instead of `global_land_mask`. |
| [`precompute_flat_mask.R`](precompute_flat_mask.R) | `precompute_flat_mask.py` | Only `compute_flat_mask`/`compute_flat_regions` — the `load_*`/`slice_flat_mask` readers exist solely to serve a downstream compute engine. |
| [`dem_water_correction.R`](dem_water_correction.R) | `dem_water_correction.py` | Serial/CPU path only (see below); `correct` is the only ported CLI subcommand. |
| [`elevation.R`](elevation.R) | `elevation.py` | On-demand single-tile/single-point elevation lookup with its own disk cache (`DEM_CACHE_DIR`) — distinct from `dem_download.R`'s study-area pipeline. 3DEP downloads via a parallel sub-tiled WCS grid (`httr2::req_perform_parallel()` in place of the original's `ThreadPoolExecutor`); `_get_3dep_url()` not ported (dead code in the Python original — defined, never called). |
| [`build_dem_mosaic.R`](build_dem_mosaic.R) | `build_dem_mosaic.py` | Uses `elevation.R`'s `download_tile()`/`is_in_3dep_coverage()`, matching the Python original's own import. |
| [`repair_dem_gaps.R`](repair_dem_gaps.R) | `repair_dem_gaps.py` | Uses `elevation.R`'s `download_tile_glo30()`, matching the Python original's own import. |

Not ported — this is the scope decision from the start of this
translation, not something discovered partway through:
`tile_registry.py`, `storage_manager.py`, `tile_validation.py`,
`tile_builder.py` (the tile-cache system — infrastructure, not DEM
science), `gpu_tools.py`, `parallel_tools.py` (Python-process/CuPy
specific, no R equivalent worth building), `visualize_tile.py`, and
all `test_*.py` except `test_tile_id.py`.

## Scope

**GPU processing dropped.** No CuPy equivalent for this type of raster work. Every `_GPU_TOOLS_AVAILABLE`/`process_label_array_gpu`
branch in the Python originals is simply absent here — the R port
always takes the CPU path the Python version falls back to anyway.

**Parallelism dropped.** `dem_download.R` downloads tiles sequentially
rather than via a thread pool. R's HTTP/concurrency story doesn't map
cleanly onto Python's `ThreadPoolExecutor`; `parallel::mclapply`
(Unix) or the `future`/`furrr` packages are the natural place to add
it back if a real multi-tile study area needs it.

## Dependency notes

**`sf` is optional, not required**, despite being the obvious choice
for polygon construction in R. It pulls in the `units` package, which
needs the system library `libudunits2` — not always preinstalled, and
not something to require of anyone who just wants to run this (install
via your OS's package manager if you want it: `libudunits2-dev` on
Debian/Ubuntu, `udunits` via Homebrew on macOS, an AUR package on
Arch). So:

- All polygon construction on the hot path (batch rasterization in
  `dem_water_correction.R`) uses `terra`'s own `vect()` via WKT
  strings instead — no `sf` involved at all in the actual correction
  algorithm.
- The one place `sf` is genuinely the right tool — `fetch_ocean_polygons_osm`'s
  polygonize/union/difference chain, building ocean-fill polygons from
  coastline linework — checks `requireNamespace("sf")` at runtime and
  falls back to "treat the whole bbox as ocean" if it's not installed,
  rather than hard-failing the whole file over one best-effort feature.
  Install `sf` (and `libudunits2`) if you want the real coastline-aware
  version of that one function.

**No CLI wrapper.** The Python originals are argparse CLIs; these are
meant to be sourced into an R session. Adding a fifth dependency
(`optparse` or similar) for CLI parsing didn't seem worth it against
the goal of a short, obvious dependency list.

## `elevation.py` / `elevation.R`

`build_dem_mosaic.py` and `repair_dem_gaps.py` both import from a
module called `elevation` (`download_tile`/`_is_in_3dep_coverage` and
`download_tile_glo30` respectively). `python/elevation.py` is a real,
present file in this repo — it's a separate on-demand download path
from `dem_download.R`'s study-area pipeline, with its own disk cache
and a sub-tiled 3DEP WCS downloader `dem_download.R` doesn't have.
`build_dem_mosaic.R` and `repair_dem_gaps.R` `source("elevation.R")`
and call its `download_tile()` / `download_tile_glo30()` /
`is_in_3dep_coverage()` directly, matching the Python originals'
own imports rather than substituting `dem_download.R`'s functions.

## Real bugs found while writing this port

Translating forces you to actually run every code path, including ones
a Python-side review can miss. Several of these are worth calling out
because they're not R-specific — they'd bite the Python original too,
or in a few cases, already do:

- **`focal()`'s masking convention.** terra's `focal(w=matrix, fun=...)`
  excludes a neighborhood cell with `NA` in the weight matrix, not
  `0` — a `0` gets *multiplied into* the window like any other weight.
  The first version of the boundary-erosion cross mask
  (`compute_flat_regions`) used 0s for the excluded corners, which
  collapsed the erosion's `min()` to 0 almost everywhere (a synthetic
  test caught it immediately: 73% "boundary" instead of the expected
  single-digit percentage). Fixed by using `NA`.
- **Truncated-download detection, and a false sense of "fixed."**
  `dem_download.R`'s tile cache validates a cached/downloaded file by
  opening it — but a truncated GeoTIFF (confirmed live: a download
  interrupted by a 2-minute process timeout during testing) commonly
  still has an intact, readable *header*, so `terra::rast(path)`
  succeeds even though the pixel data is corrupt. **The Python
  original had the exact same gap** — `dem_download.py`'s
  `download_tile` only did `with rasterio.open(out_path): pass`, never
  reading pixel data either — fixed there too (`_raster_is_readable()`
  in `python/dem_download.py`), not just noted.
  `raster_is_readable()` was added to force a real decode before
  trusting a cached file — but its first version (`global(r,
  "notNA")`) turned out to have the same class of gap one level down:
  `elevation.R` had two more cache checks that skipped this validation
  entirely (a stale, truncated GLO-30 file sat unnoticed and got
  reused, surfacing as a hard, uncaught crash in one run and
  silently-still-missing output pixels in another — `elevation.py` had
  the identical gap, fixed with a local copy of the same check), and
  while adding those, `global(r, "notNA")` itself failed to catch that
  specific corruption: GDAL reported the failure as a warning, not an
  R error, and `global()`'s internal read path returned a result
  anyway. `values(r)` — a full, direct decode of every pixel, the same
  operation the rest of this codebase actually performs when using a
  tile for real — failed loudly on the identical file as expected.
  Fixed in both `dem_download.R`'s and `elevation.R`'s copies (plus a
  `tryCatch` in `elevation.R`'s gap-fill that only wrapped the
  download call, not the reproject/write after it — widened to cover
  the whole chain, matching the Python original's actual scope, so a
  corrupt file degrades gracefully instead of crashing). Lesson: an
  aggregate statistics function isn't a reliable proxy for "can this
  actually be read" — only doing the real read is, and a fix applied
  in one place needs to actually be applied everywhere the same
  assumption is made.
- **`req_timeout()` bounds the whole request, not just idle gaps.**
  `elevation.R`'s GLO-30 downloader ported `elevation.py`'s
  `requests.get(url, timeout=60)` as `req_timeout(60)` — reasonable on
  the surface, since it's the same number. It isn't the same
  semantics: Python `requests`' `timeout` only bounds connection setup
  and the gap between reads, not total transfer time, while httr2/curl's
  `req_timeout()` is a hard cap on the entire operation. Confirmed live
  — a real ~38MB tile download was aborted by curl at exactly 60s with
  26.7 of 40.2MB received, on a connection `requests` would have let
  finish. Fixed by raising it to 180s in `download_tile_glo30()`, with
  a comment on why the two libraries' `timeout=` isn't a 1:1 port.
- **Unbounded merge memory (Python-only, not an R issue).** Not a bug
  in this port, but the reverse: `terra`'s `merge()`/`writeRaster()`
  already process large rasters in disk-backed chunks internally, which
  is *why* this never showed up here. `build_dem_mosaic.py`'s
  `build_mosaic()` (and `dem_download.py`'s `mosaic_tiles()`) called
  `rasterio.merge.merge()` with no output path, forcing the entire
  mosaic into one in-memory array, then a second full-size copy via a
  `MemoryFile`, then a third for the crop step. Confirmed live on a
  15GB-RAM machine: a real 4x4-degree mosaic's merge step took 10.5
  hours before a second run of the identical operation completed in 33
  seconds — consistent with the first hitting heavy swap thrashing.
  Fixed on the Python side by using `merge()`'s own `dst_path`/
  `dst_kwds`/`mem_limit` parameters (already present in the installed
  rasterio version, just unused) to merge, crop, and write in bounded
  windows instead — peak RSS dropped to ~2.5GB regardless of study-area
  size, confirmed bit-identical output via a windowed diff. No R-side
  change needed.
- **Sub-tile merge doesn't reliably reconstruct the exact 3DEP grid.**
  `download_tile_3dep_subtiled()` in both `elevation.R` and
  `elevation.py` builds a 1x1 degree tile from a 9x9 grid of WCS
  sub-tile requests, then merges the 81 responses with no explicit
  target grid — trusting `terra::merge()`/`rasterio.merge.merge()` to
  infer the right extent/resolution from the sub-tiles' own returned
  georeferencing. A live WCS server doesn't guarantee a sub-tile's
  *returned* bounds land exactly on the *requested* bbox to the bit;
  confirmed live, comparing the two ports' output for the identical
  tile (`N44_W113`, 81/81 sub-tiles OK on both sides): R's unsnapped
  merge came out 10801x10811 with non-square pixels instead of the
  canonical 10800x10800, which meant the two ports' mosaics for the
  same real study area only matched on ~5% of pixels, with outliers up
  to 12.6km. **This is a real bug in both ports, not an R-specific
  one** — Python's `rasterio.merge()` happened to round back to
  exactly 10800x10800 in this test (its output-size math rounds
  bounds-span/resolution, which absorbed the sub-pixel noise
  correctly that time), but nothing about that call guarantees it.
  Fixed in both by explicitly resampling the merged result onto a
  template raster built from the tile's own known bounds and exact
  `grid_n*subtile_px` pixel shape (`method="near"`/`Resampling.nearest`
  — a grid snap, not a real resampling, since source and target
  resolution are for all practical purposes identical) instead of
  trusting either merge library's auto-inference.
