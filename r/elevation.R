#' elevation.R
#' -----------
#' Translation of python/elevation.py: on-demand DEM tile download (GLO-30
#' direct, and 3DEP via a sub-tiled WCS grid + merge) with a persistent,
#' size-capped local cache, plus single-point elevation lookup. This is
#' what build_dem_mosaic.R and repair_dem_gaps.R actually depend on
#' (download_tile/is_in_3dep_coverage and download_tile_glo30
#' respectively) -- see those files' module docstrings for why that
#' matters (the Python originals import this from a module that, until
#' recently, didn't exist anywhere in this repo).
#'
#' This file is functionally DISTINCT from dem_download.R, not a
#' duplicate of it: dem_download.R does cascading-fallback bulk
#' downloads for a whole study area (single direct-file GET per tile,
#' multi-candidate resolution fallback). This file does single-tile,
#' on-demand downloads, including a genuinely different 3DEP strategy
#' (a grid of small WCS sub-tile requests merged into one GeoTIFF,
#' rather than one direct S3 file) and a single-point get_elevation()
#' lookup that dem_download.R has no equivalent of at all.
#'
#' Not translated: `_get_3dep_url()` -- present in the Python original
#' but never called anywhere in it (superseded by
#' download_tile_3dep_subtiled's own inline URL construction); dead
#' code, not translated, same treatment as
#' dem_water_correction.py's sample_shoreline_elevation.
#'
#' Concurrency: the Python original fetches the WCS sub-tile grid via
#' ThreadPoolExecutor with manual per-request retry/backoff. This port
#' uses httr2::req_perform_parallel() (bounded by max_active) with
#' req_retry() attached per-request instead of hand-rolling the retry
#' loop -- httr2's own built-in mechanism for exactly this, not a
#' simplification.
#'
#' Env vars (same names as the Python original, loaded from a plain OS
#' environment -- R's own .Renviron, loaded automatically at session
#' start, serves the same role python-dotenv's .env loading does; no
#' extra package needed for that part):
#'   DEM_CACHE_DIR    - explicit cache directory, wins outright
#'   STORAGE_ROOT      - Windows data-drive root (see resolve_cache_dir)
#'   DEM_CACHE_MAX_GB  - cache size cap, 0 disables (default 20)
#'   DEM_SOURCE        - default source for download_tile() (default "auto")

suppressPackageStartupMessages({
  library(terra)
  library(httr2)
})

# ------------------------------------------------
# Cache directory + LRU eviction
# ------------------------------------------------

# Placeholder -- point STORAGE_ROOT (or DEM_CACHE_DIR directly) at
# wherever your own data drive/volume actually is. See module docstring.
DEFAULT_STORAGE_ROOT <- "D:\\dem_data"

#' Directory this file itself lives in, when loaded via source() --
#' the R equivalent of Python's `Path(__file__).parent`. Walks the call
#' stack for the first frame with an `ofile` (source() records this;
#' an interactive paste-in has no such frame). Falls back to the
#' working directory so this doesn't error in that case, just resolves
#' the cache dir relative to somewhere reasonable instead of "next to
#' this file" specifically.
this_file_dir <- function() {
  for (f in sys.frames()) {
    if (!is.null(f$ofile)) return(dirname(normalizePath(f$ofile)))
  }
  getwd()
}

# terra's default tempdir() is R's session temp dir -- normally under
# /tmp, which on this machine turned out to be a small (7.7GB),
# RAM-backed tmpfs, completely separate from and much smaller than the
# actual project disk. The 81-subtile 3DEP merge writes large
# intermediate rasters there and can fill it even when the real data
# disk has 90+GB free -- confirmed live: two separate
# "No space left on device" failures during this port's own testing
# were actually the tmpfs filling up, not the project disk, and looked
# identical to a real disk-full error until traced back. Point terra's
# tempdir at a folder next to the tile cache, on the real disk,
# instead of trusting the platform default.
.raster_tmp_dir <- file.path(this_file_dir(), "raster_tmp")
dir.create(.raster_tmp_dir, recursive = TRUE, showWarnings = FALSE)
terraOptions(tempdir = .raster_tmp_dir)

#' Resolve the DEM tile cache directory. DEM_CACHE_DIR always wins
#' outright; otherwise, on Windows, redirects under STORAGE_ROOT (or
#' DEFAULT_STORAGE_ROOT) when that drive is actually present (covers a
#' configured data drive without hardcoding it); otherwise falls back
#' to a directory next to this file (covers Linux/macOS, and any
#' Windows machine without the data drive mounted).
resolve_cache_dir <- function() {
  explicit <- Sys.getenv("DEM_CACHE_DIR", unset = "")
  if (nzchar(explicit)) return(explicit)
  if (.Platform$OS.type == "windows") {
    storage_root <- Sys.getenv("STORAGE_ROOT", unset = DEFAULT_STORAGE_ROOT)
    drive <- sub("^([A-Za-z]:).*$", "\\1", storage_root)
    if (nzchar(drive) && dir.exists(paste0(drive, "/"))) {
      return(file.path(storage_root, "dem_cache"))
    }
  }
  file.path(this_file_dir(), "dem_cache")
}

CACHE_DIR <- resolve_cache_dir()
dir.create(CACHE_DIR, recursive = TRUE, showWarnings = FALSE)

# Total on-disk cap for CACHE_DIR, enforced after each new tile write.
# 0 disables the cap.
DEM_CACHE_MAX_GB <- as.numeric(Sys.getenv("DEM_CACHE_MAX_GB", unset = "20"))

#' Bumps a cache file's mtime on reuse, so LRU eviction ranks by
#' last-used rather than download date.
touch_file <- function(path) {
  tryCatch(Sys.setFileTime(path, Sys.time()), error = function(e) invisible(NULL))
}

#' Deletes least-recently-used tiles (by mtime) until CACHE_DIR is back
#' under DEM_CACHE_MAX_GB. `protect` (the tile just written) is never
#' evicted, even if it alone exceeds the cap. Best-effort: any error
#' just leaves that file in place rather than raising -- this is disk
#' hygiene, not allowed to break a request over it.
enforce_cache_cap <- function(protect = NULL) {
  if (DEM_CACHE_MAX_GB <= 0) return(invisible(NULL))
  cap_bytes <- DEM_CACHE_MAX_GB * 1e9
  all_files <- tryCatch(list.files(CACHE_DIR, full.names = TRUE), error = function(e) character(0))
  info <- file.info(all_files)
  total <- sum(info$size, na.rm = TRUE)
  if (total <= cap_bytes) return(invisible(NULL))

  evictable <- all_files[all_files != protect]
  evictable <- evictable[order(file.info(evictable)$mtime)]
  for (f in evictable) {
    if (total <= cap_bytes) break
    size <- file.info(f)$size
    ok <- tryCatch({ unlink(f); TRUE }, error = function(e) FALSE)
    if (ok) {
      total <- total - size
      cat(sprintf("[DEM cache] evicted %s (%.0fMB, LRU) -- cache over %.0fGB cap\n",
                   basename(f), size / 1e6, DEM_CACHE_MAX_GB))
    }
  }
}

SOURCES <- c(GLO30 = 30, "3DEP_10" = 10, "3DEP_1" = 1)
DEFAULT_SOURCE <- Sys.getenv("DEM_SOURCE", unset = "auto")

# USGS 3DEP bounding box -- continental US + Alaska + Hawaii. Wider
# than dem_download.R's own per-URL coverage checks (24-72N) -- this
# is the real Python original's definition, kept as-is rather than
# reconciled with the other file's narrower one.
DEP3_BOUNDS <- list(min_lat = 17.0, max_lat = 72.0, min_lng = -180.0, max_lng = -60.0)

is_in_3dep_coverage <- function(lat, lng) {
  b <- DEP3_BOUNDS
  lat >= b$min_lat && lat <= b$max_lat && lng >= b$min_lng && lng <= b$max_lng
}

#' True if `path` opens AND its pixel data can actually be decoded --
#' not just that the header parses. A truncated GeoTIFF (an interrupted
#' download, or a killed process mid-write) commonly still has an
#' intact, readable header, so a bare rast(path) would call that
#' "valid" even though the pixel data is corrupt. Local copy of
#' dem_download.R's raster_is_readable() -- duplicated rather than
#' shared via source(), since elevation.R is meant to be sourceable on
#' its own (see module docstring).
#'
#' Real incident (2026-08-10): a stale, truncated GLO-30 cache file
#' (from an earlier interrupted run during this port's own testing)
#' sat unnoticed because neither of this file's two cache checks
#' (here and download_tile_3dep_subtiled()'s) verified readability
#' before reusing a cached path -- it surfaced downstream as a hard
#' `project()` failure in one run and, worse, as silently-still-NA
#' output pixels (despite the log claiming a successful GLO-30 fill)
#' in another. This function, applied at both cache checks, forces
#' re-download instead of silently trusting a corrupt cached file.
#'
#' Uses values(r), NOT global(r, "notNA") -- the first version of this
#' check used global(), matching dem_download.R's original, and it
#' missed this exact corrupt file: GDAL reports a block-read failure
#' as a warning, not an R error, and global()'s internal read path can
#' apparently swallow that and return a result anyway. values() -- a
#' full, direct decode of every pixel, the actual operation this file
#' performs on a tile when using it for real -- fails loudly on the
#' same file as expected. Fixed here and in dem_download.R's copy.
raster_is_readable <- function(path) {
  tryCatch({
    r <- rast(path)
    values(r)
    TRUE
  }, error = function(e) FALSE)
}

# ------------------------------------------------
# GLO-30 (Copernicus)
# ------------------------------------------------

get_tile_name <- function(lat, lng) {
  lat_floor <- floor(lat); lng_floor <- floor(lng)
  lat_str <- if (lat_floor >= 0) sprintf("N%02d", abs(lat_floor)) else sprintf("S%02d", abs(lat_floor))
  lng_str <- if (lng_floor >= 0) sprintf("E%03d", abs(lng_floor)) else sprintf("W%03d", abs(lng_floor))
  sprintf("%s_00_%s", lat_str, lng_str)
}

get_tile_url <- function(lat, lng) {
  tile <- get_tile_name(lat, lng)
  sprintf("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_%s_00_DEM/Copernicus_DSM_COG_10_%s_00_DEM.tif",
          tile, tile)
}

download_tile_glo30 <- function(lat, lng) {
  tile_name <- get_tile_name(lat, lng)
  local_path <- file.path(CACHE_DIR, sprintf("%s.tif", tile_name))
  if (file.exists(local_path)) {
    if (raster_is_readable(local_path)) {
      cat(sprintf("Using cached tile: %s\n", tile_name))
      touch_file(local_path)
      return(local_path)
    }
    cat(sprintf("Cached tile %s is corrupt/truncated -- re-downloading\n", tile_name))
    unlink(local_path)
  }
  url <- get_tile_url(lat, lng)
  cat(sprintf("Downloading tile: %s from %s\n", tile_name, url))
  # 180s, not the Python original's 60: httr2/curl's req_timeout() caps
  # the WHOLE request (confirmed the hard way -- it aborted a real
  # GLO-30 download mid-transfer at exactly 60s with 2/3 of the file
  # received), unlike Python requests' timeout, which only bounds
  # connect time and the gap between reads, not total transfer time.
  # 60s matched the Python original fine there; ported as-is here it
  # would spuriously abort any real ~30-40MB tile on an ordinary
  # connection.
  resp <- request(url) |> req_timeout(180) |>
    req_error(is_error = function(resp) FALSE) |> req_perform(path = local_path)
  if (resp_status(resp) == 200) {
    cat(sprintf("Downloaded: %s (%.1f MB)\n", tile_name, file.info(local_path)$size / 1024 / 1024))
    enforce_cache_cap(protect = local_path)
    local_path
  } else {
    unlink(local_path)
    stop(sprintf("Could not download GLO-30 tile for lat=%s, lng=%s. Status: %d",
                  lat, lng, resp_status(resp)))
  }
}

# ------------------------------------------------
# 3DEP (USGS) -- 1m and 10m, via sub-tiled WCS grid
# ------------------------------------------------

#' Download a full 1x1 degree 3DEP tile by requesting a grid_n x grid_n
#' grid of sub-tiles, each subtile_px x subtile_px pixels, then merging
#' into a single GeoTIFF. Default: 9x9 grid of 1200x1200px sub-tiles =
#' 10800x10800 total = 1/3 arc-second (~10m) resolution.
download_tile_3dep_subtiled <- function(lat_floor, lng_floor, resolution = 10,
                                         grid_n = 9, subtile_px = 1200,
                                         max_workers = 4, timeout = 120) {
  res_tag <- sprintf("3dep_%dm", resolution)
  lat_str <- if (lat_floor >= 0) sprintf("N%02d", abs(lat_floor)) else sprintf("S%02d", abs(lat_floor))
  lng_str <- if (lng_floor < 0) sprintf("W%03d", abs(lng_floor)) else sprintf("E%03d", abs(lng_floor))
  tile_name <- sprintf("%s_%s_%s", lat_str, lng_str, res_tag)
  local_path <- file.path(CACHE_DIR, sprintf("%s.tif", tile_name))

  if (file.exists(local_path)) {
    if (raster_is_readable(local_path)) {
      cat(sprintf("Using cached tile: %s\n", tile_name))
      touch_file(local_path)
      return(local_path)
    }
    cat(sprintf("Cached tile %s is corrupt/truncated -- re-downloading\n", tile_name))
    unlink(local_path)
  }

  cat(sprintf("[3DEP] Downloading %s as %dx%d sub-tile grid (%d requests, ~%dx%dpx each)...\n",
              tile_name, grid_n, grid_n, grid_n * grid_n, subtile_px, subtile_px))

  step <- 1.0 / grid_n
  base_url <- paste0("https://elevation.nationalmap.gov/arcgis/services/",
                      "3DEPElevation/ImageServer/WCSServer",
                      "?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage",
                      "&COVERAGE=DEP3Elevation&CRS=EPSG:4326&FORMAT=GeoTIFF")

  subtiles <- expand.grid(row = 0:(grid_n - 1), col = 0:(grid_n - 1))
  tmp_dir <- tempfile("subtiles_")
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  reqs <- vector("list", nrow(subtiles))
  tmp_paths <- character(nrow(subtiles))
  for (i in seq_len(nrow(subtiles))) {
    row <- subtiles$row[i]; col <- subtiles$col[i]
    xmin <- lng_floor + col * step; xmax <- lng_floor + (col + 1) * step
    ymin <- lat_floor + row * step; ymax <- lat_floor + (row + 1) * step
    url <- sprintf("%s&BBOX=%g,%g,%g,%g&WIDTH=%d&HEIGHT=%d",
                    base_url, xmin, ymin, xmax, ymax, subtile_px, subtile_px)
    tmp_paths[i] <- file.path(tmp_dir, sprintf("%d_%d.tif", row, col))
    reqs[[i]] <- request(url) |> req_timeout(timeout) |>
      req_retry(max_tries = 5, retry_on_failure = TRUE)
  }

  cat(sprintf("[3DEP] Fetching %d sub-tiles (up to %d concurrent)...\n", length(reqs), max_workers))
  t0 <- Sys.time()
  resps <- req_perform_parallel(reqs, paths = tmp_paths, on_error = "continue",
                                 max_active = max_workers, progress = FALSE)

  ok <- vapply(seq_along(resps), function(i) {
    r <- resps[[i]]
    !inherits(r, "error") && !is.null(r) && resp_status(r) == 200 && file.exists(tmp_paths[i])
  }, logical(1))
  n_failed <- sum(!ok)
  cat(sprintf("[3DEP] %d/%d sub-tiles OK (%.0fs)\n", sum(ok), length(reqs),
              as.numeric(Sys.time() - t0, units = "secs")))

  if (n_failed > 0) {
    cat(sprintf("[3DEP] Warning: %d sub-tiles failed -- tile may have gaps\n", n_failed))
    if (sum(ok) == 0) {
      cat("[3DEP] All sub-tiles failed, falling back to GLO-30\n")
      return(download_tile_glo30(lat_floor + 0.5, lng_floor + 0.5))
    }
  }

  cat(sprintf("[3DEP] Merging %d sub-tiles...\n", sum(ok)))
  rasters <- lapply(tmp_paths[ok], rast)
  merged <- do.call(terra::merge, rasters)

  # Snap onto the canonical tile grid (exact bounds, exact
  # grid_n*subtile_px pixels) rather than trusting terra::merge()'s own
  # extent/resolution inference from the 81 sub-tiles' individually
  # returned bounds. A live WCS server doesn't guarantee a sub-tile's
  # *returned* georeferencing lands exactly on the *requested* bbox to
  # the bit -- confirmed live: an unsnapped merge of 81/81 successful
  # sub-tiles came out 10801x10811 with non-square pixels instead of
  # the canonical 10800x10800, which also meant this port's output
  # didn't line up with the Python original's pixel-for-pixel. Method
  # "near" because source and target resolution are for all practical
  # purposes identical -- this is a grid snap, not a real resampling.
  template <- rast(xmin = lng_floor, xmax = lng_floor + 1,
                    ymin = lat_floor, ymax = lat_floor + 1,
                    nrows = grid_n * subtile_px, ncols = grid_n * subtile_px,
                    crs = "EPSG:4326")
  merged <- resample(merged, template, method = "near")

  writeRaster(merged, local_path, filetype = "GTiff", datatype = "FLT4S",
              gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES"), overwrite = TRUE)

  # ---- Gap fill from GLO-30 (covers both "some sub-tiles failed" and
  # "always check for nodata pixels", collapsed into one pass here since
  # both do the identical fill -- the Python original runs this same
  # logic twice in a row for those two triggers separately) ----
  dep_r <- rast(local_path)
  dep_m <- matrix(as.numeric(values(dep_r)), nrow(dep_r), ncol(dep_r), byrow = TRUE)
  dep_nodata <- terra::NAflag(dep_r); if (is.na(dep_nodata)) dep_nodata <- -9999.0
  gap_mask <- is.na(dep_m) | (dep_m <= dep_nodata + 1) | (dep_m == 0)
  n_gaps <- sum(gap_mask)

  if (n_gaps > 0) {
    cat(sprintf("[3DEP] Found %s nodata pixels (%.1f%%) -- filling from GLO30...\n",
                format(n_gaps, big.mark = ","), n_gaps / length(dep_m) * 100))
    # tryCatch wraps the WHOLE download+reproject+write chain, not just
    # the download call -- matches the Python original's actual scope
    # (elevation.py wraps its equivalent reproject() in the same
    # try/except as the download). A narrower tryCatch here left
    # project() able to throw an uncaught error straight out of this
    # function -- confirmed live: a corrupt GLO-30 cache file (since
    # fixed separately, see raster_is_readable() above) crashed here
    # with an unhandled "[project] warp failure" instead of degrading
    # to "ships with the gap still unfilled", same as any other
    # gap-fill failure already handles.
    filled <- tryCatch({
      glo_path <- download_tile_glo30(lat_floor + 0.5, lng_floor + 0.5)
      glo_resampled_r <- project(rast(glo_path), dep_r, method = "bilinear")
      glo_resampled <- matrix(as.numeric(values(glo_resampled_r)), nrow(dep_r), ncol(dep_r), byrow = TRUE)
      glo_resampled[is.na(glo_resampled)] <- dep_nodata
      dep_m[gap_mask] <- glo_resampled[gap_mask]
      out_r <- rast(dep_m, extent = ext(dep_r), crs = "EPSG:4326")
      writeRaster(out_r, local_path, filetype = "GTiff", datatype = "FLT4S",
                  gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES"), overwrite = TRUE)
      TRUE
    }, error = function(e) {
      cat(sprintf("[3DEP] Gap fill warning: %s\n", conditionMessage(e)))
      FALSE
    })
    if (filled) {
      cat(sprintf("[3DEP] Filled %s pixels from GLO30\n", format(n_gaps, big.mark = ",")))
    }
  } else {
    cat("[3DEP] No nodata gaps found\n")
  }

  size_mb <- file.info(local_path)$size / 1e6
  cat(sprintf("[3DEP] Saved %s (%.1fMB, %.0fs total)\n", tile_name, size_mb,
              as.numeric(Sys.time() - t0, units = "secs")))
  enforce_cache_cap(protect = local_path)
  local_path
}

download_tile_3dep <- function(lat, lng, resolution = 10) {
  if (!is_in_3dep_coverage(lat, lng)) {
    cat("[3DEP] Outside coverage, falling back to GLO-30\n")
    return(download_tile_glo30(lat, lng))
  }
  lat_floor <- floor(lat); lng_floor <- floor(lng)
  if (resolution == 10) {
    download_tile_3dep_subtiled(lat_floor, lng_floor, resolution = 10)
  } else {
    # 1m -- smaller area per request, larger grid needed
    download_tile_3dep_subtiled(lat_floor, lng_floor, resolution = 1,
                                 grid_n = 30, subtile_px = 1200)
  }
}

# ------------------------------------------------
# Auto select source / unified download interface
# ------------------------------------------------

auto_select_source <- function(lat, lng) {
  if (is_in_3dep_coverage(lat, lng)) "3DEP_10" else "GLO30"
}

#' Download a DEM tile from the specified source ('GLO30', '3DEP_10',
#' '3DEP_1', 'auto', or NULL -- NULL/'auto' use DEFAULT_SOURCE).
download_tile <- function(lat, lng, source = NULL) {
  src <- if (is.null(source)) DEFAULT_SOURCE else source
  if (identical(src, "auto")) src <- auto_select_source(lat, lng)

  if (src == "3DEP_1") {
    download_tile_3dep(lat, lng, resolution = 1)
  } else if (src == "3DEP_10") {
    download_tile_3dep(lat, lng, resolution = 10)
  } else {
    download_tile_glo30(lat, lng)
  }
}

# ------------------------------------------------
# Ocean / water mask
# ------------------------------------------------

# Rough continental bounding boxes -- a fast pre-filter, not a precise
# coastline (dem_download.R's is_likely_ocean, used by the bulk
# pipeline, uses maps::map.where for a real coastline check instead;
# this heuristic is what the Python original's on-demand path uses).
LAND_BBOX <- list(
  c(-56,  72, -168,  -52),  # Americas
  c( 34,  72,  -25,   60),  # Europe / North Africa
  c(-35,  38,  -20,   52),  # Africa
  c( 12,  82,   25,  180),  # Asia
  c(-48,  -9,  110,  180),  # Australia / Pacific
  c(-90, -60, -180,  180)   # Antarctica
)

is_probably_ocean <- function(lat, lng) {
  for (bbox in LAND_BBOX) {
    if (lat >= bbox[1] && lat <= bbox[2] && lng >= bbox[3] && lng <= bbox[4]) return(FALSE)
  }
  TRUE
}

#' TRUE if the elevation value should be treated as 0 (sea level)
#' rather than trusted as terrain: explicit nodata, physically
#' implausible, or small noise over open ocean (a known GLO-30 artifact).
elevation_is_suspect <- function(elevation, lat, lng, nodata) {
  if (!is.null(nodata) && !is.na(nodata) && isTRUE(elevation == nodata)) return(TRUE)
  if (elevation < -500 || elevation > 9000) return(TRUE)
  if (is_probably_ocean(lat, lng) && elevation >= -10 && elevation <= 20) return(TRUE)
  FALSE
}

# ------------------------------------------------
# Elevation sampling
# ------------------------------------------------

#' Get terrain elevation at a single point.
get_elevation <- function(lat, lng, source = NULL) {
  tile_path <- download_tile(lat, lng, source = source)
  r <- rast(tile_path)
  elevation <- as.numeric(terra::extract(r, cbind(lng, lat))[1, 1])
  nodata <- terra::NAflag(r)
  cat(sprintf("Raw elevation value: %s, nodata: %s\n", elevation, nodata))

  if (elevation_is_suspect(elevation, lat, lng, nodata)) {
    cat(sprintf("Elevation suspect at (%s, %s): %sm -> returning 0.0 (sea level)\n", lat, lng, elevation))
    return(0.0)
  }
  round(elevation, 1)
}
