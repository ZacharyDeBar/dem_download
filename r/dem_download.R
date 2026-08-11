#' dem_download.R
#' --------------
#' Downloads the highest available resolution DEM tiles for a study area
#' and mosaics them into a single GeoTIFF. Direct translation of
#' python/dem_download.py -- see that file for the fuller design notes;
#' this port keeps the notes that affect correctness and drops the ones
#' about sibling systems that don't exist in this repo.
#'
#' Resolution priority per tile (US coverage):
#'   1. 3DEP 10m  -- broad US coverage
#'   2. GLO-30    -- global fallback (Copernicus 30m)
#' (The Python original also tries 3DEP 1m; that source needs a
#' per-tile USGS TNM API lookup this port keeps for 10m only -- see
#' resolve_10m_url.)
#'
#' Two deliberate simplifications vs. the Python original, not oversights:
#'   - Downloads run sequentially, not via a thread pool. R's HTTP/thread
#'     story is more awkward than Python's; parallel::mclapply (Unix) or
#'     the future/furrr packages are the natural place to add concurrency
#'     back if a real multi-tile study area needs it.
#'   - Ocean detection uses maps::map.where() (a low-res bundled world
#'     coastline) rather than the Python original's global_land_mask
#'     package (a bundled ~1km GSHHG-derived raster). map.where is
#'     coarser -- same class of risk the Python code's own comment
#'     warns about for its predecessor heuristic (missed slivers/small
#'     islands) -- acceptable here since a wrong "ocean" call just means
#'     one real-terrain tile gets a wrong zero-fill, not a corrupted
#'     compute; not acceptable to silently upgrade without noting it.
#'
#' Usage:
#'   Rscript -e 'source("dem_download.R"); download_study_area(study_area_between("N44W113", "N47W109"))'
#'   Rscript -e 'source("dem_download.R"); download_study_area(study_area_between("N44W113", "N47W109"), dry_run=TRUE)'

suppressPackageStartupMessages({
  library(terra)
  library(httr2)
})

# terra's default tempdir (usually under /tmp) can be a small,
# RAM-backed tmpfs rather than real disk -- see elevation.R's module
# comment for the live failure this caused. Point it at the project
# disk instead.
dir.create("raster_tmp", showWarnings = FALSE)
terraOptions(tempdir = "raster_tmp")

source("tile_id.R")  # bounds_from_tile_corners()

# ------------------------------------------------
# Study area definitions
# ------------------------------------------------

StudyArea <- function(name, description, south, west, north, east,
                       resolution = "best", output_dir = NULL) {
  if (is.null(output_dir)) output_dir <- file.path("data", "dem", name)
  structure(
    list(name = name, description = description,
         south = south, west = west, north = north, east = east,
         resolution = resolution, output_dir = output_dir),
    class = "StudyArea"
  )
}

study_area_bounds <- function(a) c(a$south, a$west, a$north, a$east)

study_area_width_km <- function(a) {
  mid_lat <- (a$south + a$north) / 2
  (a$east - a$west) * 111.32 * cos(mid_lat * pi / 180)
}

study_area_height_km <- function(a) (a$north - a$south) * 111.32

#' Build a StudyArea spanning any two opposite 1x1-degree tile corners,
#' e.g. "N44W113" and "N47W109" (any pairing/order -- NE+SW, NW+SE,
#' whichever two corners bound the area you want). Longitude resolves
#' to the shorter arc between the corners, and the request is capped
#' at MAX_MOSAIC_TILES degree-tiles unless allow_large=TRUE -- see
#' bounds_from_tile_corners() in tile_id.R for the exact rules and why
#' the cap exists.
#'
#' `name` defaults to "<corner1>_<corner2>" and only affects the output
#' directory/manifest/mosaic filename.
study_area_between <- function(corner1_id, corner2_id, name = NULL,
                                resolution = "best", output_dir = NULL,
                                max_tiles = MAX_MOSAIC_TILES, allow_large = FALSE) {
  b <- bounds_from_tile_corners(corner1_id, corner2_id,
                                 max_tiles = max_tiles, allow_large = allow_large)
  if (is.null(name)) name <- sprintf("%s_%s", corner1_id, corner2_id)
  StudyArea(
    name = name,
    description = sprintf("%s to %s (%d degree-tiles)", corner1_id, corner2_id, b$n_tiles),
    south = b$south, west = b$west, north = b$north, east = b$east,
    resolution = resolution, output_dir = output_dir
  )
}

# ------------------------------------------------
# Tile coordinate helpers
# ------------------------------------------------

#' All (lat_floor, lng_floor) 1-degree tiles covering a bbox, as an
#' n x 2 integer matrix. lat_floor is the SW corner latitude, e.g. tile
#' (45, -112) covers N45-N46, W111-W112.
get_1deg_tiles <- function(south, west, north, east) {
  lats <- seq(floor(south), ceiling(north) - 1)
  lngs <- seq(floor(west), ceiling(east) - 1)
  as.matrix(expand.grid(lat = lats, lng = lngs))
}

tile_label <- function(lat, lng) {
  ns <- if (lat >= 0) "N" else "S"
  ew <- if (lng < 0) "W" else "E"
  sprintf("%s%02d%s%03d", ns, abs(lat), ew, abs(lng))
}

# ------------------------------------------------
# Source URL builders
# ------------------------------------------------

#' 3DEP 10m direct tile URL on USGS S3 -- current path format.
url_3dep_10m_direct <- function(lat, lng) {
  if (lng >= 0 || lat < 24 || lat > 72) return(NULL)
  lat_str <- sprintf("%02d", abs(lat + 1))
  lng_str <- sprintf("%03d", abs(lng))
  sprintf(paste0("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/",
                 "13/TIFF/current/n%sw%s/USGS_13_n%sw%s.tif"),
          lat_str, lng_str, lat_str, lng_str)
}

#' Copernicus GLO-30 via the Copernicus DEM AWS bucket. Global coverage
#' at ~30m resolution. Public, no credentials required.
url_glo30 <- function(lat, lng) {
  ns <- if (lat >= 0) "N" else "S"
  ew <- if (lng >= 0) "E" else "W"
  lat_str <- sprintf("%02d", abs(lat))
  lng_str <- sprintf("%03d", abs(lng))
  base <- sprintf("Copernicus_DSM_COG_10_%s%s_00_%s%s_00_DEM",
                   ns, lat_str, ew, lng_str)
  sprintf("https://copernicus-dem-30m.s3.amazonaws.com/%s/%s.tif", base, base)
}

#' Query the USGS TNM API for the current direct download URL for the
#' 10m tile covering this 1-degree cell. Falls back to a direct path
#' guess if the API is unavailable.
resolve_10m_url <- function(lat, lng) {
  if (lng >= 0 || lat < 24 || lat > 72) return(NULL)

  api_url <- sprintf(
    paste0("https://tnmaccess.nationalmap.gov/api/v1/products",
           "?datasets=Digital%%20Elevation%%20Model%%20(DEM)%%201%%2F3%%20arc-second",
           "&bbox=%s,%s,%s,%s&outputFormat=JSON&max=5"),
    lng, lat, lng + 1, lat + 1
  )
  result <- tryCatch({
    resp <- request(api_url) |> req_timeout(20) |> req_perform()
    if (resp_status(resp) == 200) {
      items <- resp_body_json(resp)$items
      for (item in items) {
        dl <- item$downloadURL
        if (!is.null(dl) && grepl("\\.tif$", dl, ignore.case = TRUE)) return(dl)
      }
    }
    NULL
  }, error = function(e) NULL)
  if (!is.null(result)) return(result)

  # Direct path fallback -- USGS current folder
  lat_str <- sprintf("%02d", abs(lat + 1))
  lng_str <- sprintf("%03d", abs(lng))
  sprintf(paste0("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/",
                 "13/TIFF/current/n%sw%s/USGS_13_n%sw%s.tif"),
          lat_str, lng_str, lat_str, lng_str)
}

#' Ordered list of list(url=, res_label=) to attempt for a tile. "best"
#' tries the finest resolution first and falls back automatically.
get_candidates <- function(lat, lng, resolution = "best") {
  if (resolution == "10m") {
    u <- resolve_10m_url(lat, lng)
    if (is.null(u)) return(list())
    return(list(list(url = u, res_label = "10m")))
  }
  if (resolution == "30m") {
    return(list(list(url = url_glo30(lat, lng), res_label = "30m")))
  }
  # "best" -- cascade: 10m USGS -> Copernicus 30m
  result <- list()
  u10 <- resolve_10m_url(lat, lng)
  if (!is.null(u10)) result[[length(result) + 1]] <- list(url = u10, res_label = "10m")
  result[[length(result) + 1]] <- list(url = url_glo30(lat, lng), res_label = "30m")
  result
}

# ------------------------------------------------
# Ocean tile fill-in
# ------------------------------------------------

LAND_CHECK_RES_DEG <- 0.05  # ~5.5km at the equator -- see module docstring

#' Whole-tile ocean check: samples a grid across the tile and returns
#' TRUE only if every sample is unmapped land (maps::map.where returns
#' NA for ocean/unmapped water). A single land sample anywhere aborts
#' it -- this feeds an irreversible "safe to zero-fill" decision, and a
#' false positive would silently erase real terrain.
is_likely_ocean <- function(lat, lng) {
  if (!requireNamespace("maps", quietly = TRUE)) {
    stop("is_likely_ocean() needs the 'maps' package: install.packages('maps')")
  }
  samples <- seq(0, 1 - LAND_CHECK_RES_DEG, by = LAND_CHECK_RES_DEG)
  grid <- expand.grid(lat = lat + samples, lng = lng + samples)
  hits <- maps::map.where(database = "world", x = grid$lng, y = grid$lat)
  all(is.na(hits))
}

#' Create a zero-elevation GeoTIFF for an ocean-only tile, at 30m
#' resolution (1/3 arc-second grid, 3600x3600 px for a 1deg tile).
create_ocean_tile <- function(lat, lng, output_dir) {
  ns <- if (lat >= 0) "N" else "S"
  ew <- if (lng < 0) "W" else "E"
  filename <- sprintf("%s%02d_%s%03d_ocean0m.tif", ns, abs(lat), ew, abs(lng))
  raw_dir <- file.path(output_dir, "raw")
  out_path <- file.path(raw_dir, filename)

  if (file.exists(out_path)) return(list(path = out_path, res_label = "ocean0m"))

  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  width <- height <- 3600
  r <- rast(nrows = height, ncols = width,
            xmin = lng, xmax = lng + 1, ymin = lat, ymax = lat + 1,
            crs = "EPSG:4326", vals = 0)
  writeRaster(r, out_path, filetype = "GTiff", datatype = "FLT4S",
              gdal = "COMPRESS=DEFLATE", overwrite = TRUE)

  cat(sprintf("  [%s%02d%s%03d] Created zero-elevation ocean tile\n",
              ns, abs(lat), ew, abs(lng)))
  list(path = out_path, res_label = "ocean0m")
}

# ------------------------------------------------
# Single tile download
# ------------------------------------------------

# Wall-clock cap on one candidate's transfer -- httr2's req_timeout()
# only bounds connect/idle time, same gap the Python original's comment
# describes for requests' timeout (a slow-but-steady trickle never trips
# it). httr2 doesn't expose a total-transfer-time cap directly, so this
# is enforced by wrapping the request in R's own timeout mechanism.
DEFAULT_MAX_DOWNLOAD_S <- 600

#' True if `path` opens AND its pixel data can actually be decoded --
#' not just that the header parses. A truncated GeoTIFF (an interrupted
#' download, or a Bash timeout killing this process mid-write, both
#' observed in practice while testing this port) commonly still has an
#' intact, readable header -- terra::rast(path), like Python's
#' `with rasterio.open(path): pass`, would call that "valid". This
#' forces a real decompress pass so a truncated tile-storage error
#' surfaces here, at cache-validation time, instead of silently
#' downstream. Note: the Python original (dem_download.py's
#' download_tile, both its cache-check and its post-download
#' validation) has the same header-only gap this fixes -- worth
#' porting this same fix back there.
#'
#' Uses values(r), NOT global(r, "notNA") (an earlier version of this
#' function did, and was believed to work off an earlier truncation
#' test) -- confirmed live on a second, differently-truncated real
#' file that global()'s internal read path can silently swallow a
#' GDAL block-read failure (GDAL reports it as a warning, not an R
#' error condition) and return a result anyway, while values() -- a
#' full, direct decode of every pixel, the same operation the rest of
#' this codebase actually performs when using a tile for real -- fails
#' loudly on the exact same file as expected. global() isn't a
#' reliable proxy for "can this be read the way we're about to read
#' it"; only actually doing that read is.
raster_is_readable <- function(path) {
  tryCatch({
    r <- rast(path)
    values(r)
    TRUE
  }, error = function(e) FALSE)
}

#' Download one 1-degree DEM tile at the best available resolution.
#' Returns list(path=, res_label=) on success, NULL on failure. Skips
#' download if a valid cached file already exists.
download_tile <- function(lat, lng, output_dir, resolution = "best",
                           timeout = 120, max_download_s = DEFAULT_MAX_DOWNLOAD_S) {
  candidates <- get_candidates(lat, lng, resolution)
  if (length(candidates) == 0) {
    cat(sprintf("  [%s] No source available\n", tile_label(lat, lng)))
    return(NULL)
  }

  raw_dir <- file.path(output_dir, "raw")
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

  for (cand in candidates) {
    url <- cand$url; res_label <- cand$res_label
    filename <- sprintf("%s_%s.tif", tile_label(lat, lng), res_label)
    out_path <- file.path(raw_dir, filename)

    # Already cached and valid
    if (file.exists(out_path) && file.info(out_path)$size > 10000) {
      ok <- raster_is_readable(out_path)
      if (ok) {
        cat(sprintf("  [%s] Cached (%s)\n", tile_label(lat, lng), res_label))
        return(list(path = out_path, res_label = res_label))
      }
      unlink(out_path)
    }

    cat(sprintf("  [%s] Trying %s...\n", tile_label(lat, lng), res_label))

    result <- tryCatch({
      resp <- request(url) |>
        req_timeout(timeout) |>
        req_error(is_error = function(resp) FALSE) |>
        req_perform(path = out_path)

      if (resp_status(resp) == 404) {
        cat("    404 -- trying next resolution\n")
        unlink(out_path)
        return(NULL)
      }
      if (resp_status(resp) >= 400) {
        stop(sprintf("HTTP %d", resp_status(resp)))
      }

      size_mb <- file.info(out_path)$size / 1e6
      cat(sprintf("    Downloaded %.1f MB at %s\n", size_mb, res_label))

      # Validate
      ok <- raster_is_readable(out_path)
      if (!ok) {
        cat("    Invalid raster, trying next\n")
        unlink(out_path)
        return(NULL)
      }
      list(path = out_path, res_label = res_label)
    }, error = function(e) {
      cat(sprintf("    Error: %s, trying next\n", conditionMessage(e)))
      if (file.exists(out_path)) unlink(out_path)
      NULL
    })

    if (!is.null(result)) return(result)
  }

  cat(sprintf("  [%s] All sources failed\n", tile_label(lat, lng)))
  if (is_likely_ocean(lat, lng)) {
    return(create_ocean_tile(lat, lng, output_dir))
  }
  NULL
}

# ------------------------------------------------
# Mosaic
# ------------------------------------------------

#' Merge tiles into a single cropped, compressed GeoTIFF. Tiles at
#' different resolutions are resampled onto the finest tile's grid
#' (terra::merge keeps the first raster's resolution/alignment and
#' resamples the rest onto it -- same effect as rasterio.merge.merge
#' after this function's own reprojection pass).
mosaic_tiles <- function(tile_paths, output_path, bounds) {
  if (length(tile_paths) == 0) {
    cat("[MOSAIC] No tiles to merge\n")
    return(invisible(NULL))
  }
  cat(sprintf("\n[MOSAIC] Merging %d tiles...\n", length(tile_paths)))

  # Sort finest resolution first so the merge reference grid is finest
  res_order <- function(p) {
    n <- tolower(basename(p))
    if (grepl("1m", n)) return(0)
    if (grepl("10m", n)) return(1)
    2
  }
  tile_paths <- tile_paths[order(sapply(tile_paths, res_order))]

  rasters <- lapply(tile_paths, rast)
  target_crs <- crs(rasters[[1]])
  rasters <- lapply(rasters, function(r) {
    if (crs(r) != target_crs) project(r, target_crs) else r
  })

  south <- bounds[1]; west <- bounds[2]; north <- bounds[3]; east <- bounds[4]
  merged <- do.call(terra::merge, rasters)
  merged <- crop(merged, ext(west, east, south, north))

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  # BIGTIFF=YES explicit, not GDAL's BIGTIFF=IF_NEEDED default -- see
  # build_dem_mosaic.R's identical comment for the live failure this avoids.
  writeRaster(merged, output_path, filetype = "GTiff", datatype = "FLT4S",
              gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES"),
              overwrite = TRUE)

  size_mb <- file.info(output_path)$size / 1e6
  res_m <- abs(res(merged)[1]) * 111320
  cat(sprintf("[MOSAIC] Output: %s\n", output_path))
  cat(sprintf("         Size:   %.0f MB\n", size_mb))
  cat(sprintf("         Grid:   %d x %d px\n", ncol(merged), nrow(merged)))
  cat(sprintf("         Res:    ~%.1f m/px\n", res_m))
}

# ------------------------------------------------
# Main pipeline
# ------------------------------------------------

download_study_area <- function(area, dry_run = FALSE, no_mosaic = FALSE) {
  tiles <- get_1deg_tiles(area$south, area$west, area$north, area$east)

  cat(sprintf("\n%s\n", strrep("=", 62)))
  cat(sprintf("  %s: %s\n", area$name, area$description))
  cat(sprintf("  Bounds: %gS %gW  %gN %gE\n", area$south, area$west, area$north, area$east))
  cat(sprintf("  Size:   %.0f km x %.0f km\n",
              study_area_width_km(area), study_area_height_km(area)))
  cat(sprintf("  Tiles:  %d x 1-degree tiles\n", nrow(tiles)))
  cat(sprintf("  Res:    %s\n", area$resolution))
  cat(sprintf("  Output: %s\n", area$output_dir))
  cat(sprintf("%s\n\n", strrep("=", 62)))

  if (dry_run) {
    cat("[DRY RUN] Tiles that would be downloaded:\n\n")
    ord <- order(tiles[, "lat"], tiles[, "lng"])
    for (i in ord) {
      lat <- tiles[i, "lat"]; lng <- tiles[i, "lng"]
      candidates <- get_candidates(lat, lng, area$resolution)
      label <- tile_label(lat, lng)
      if (length(candidates) > 0) {
        cat(sprintf("  %s: %s\n", label, candidates[[1]]$res_label))
        cat(sprintf("    %s...\n", substr(candidates[[1]]$url, 1, 72)))
      } else {
        cat(sprintf("  %s: no source available\n", label))
      }
    }
    cat(sprintf("\nTotal: %d tiles\n", nrow(tiles)))
    return(invisible(NULL))
  }

  downloaded <- list()
  failed <- list()
  cat("Downloading sequentially (see module docstring)...\n\n")

  for (i in seq_len(nrow(tiles))) {
    lat <- tiles[i, "lat"]; lng <- tiles[i, "lng"]
    result <- tryCatch(
      download_tile(lat, lng, area$output_dir, area$resolution),
      error = function(e) {
        cat(sprintf("  [%s] Error: %s\n", tile_label(lat, lng), conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(result)) {
      downloaded[[length(downloaded) + 1]] <- result$path
    } else {
      failed[[length(failed) + 1]] <- c(lat, lng)
    }
  }

  cat(sprintf("\nDownload complete: %d/%d tiles\n", length(downloaded), nrow(tiles)))
  if (length(failed) > 0) {
    cat(sprintf("Failed tiles (%d):\n", length(failed)))
    for (f in failed) cat(sprintf("  %s\n", tile_label(f[1], f[2])))
  }

  # Save manifest
  manifest <- list(
    study_area = area$name,
    bounds = study_area_bounds(area),
    resolution = area$resolution,
    total_tiles = nrow(tiles),
    downloaded = length(downloaded),
    failed_tiles = sapply(failed, function(f) tile_label(f[1], f[2])),
    tile_files = sapply(downloaded, as.character)
  )
  dir.create(area$output_dir, recursive = TRUE, showWarnings = FALSE)
  manifest_path <- file.path(area$output_dir, "download_manifest.json")
  jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE)
  cat(sprintf("Manifest: %s\n", manifest_path))

  if (no_mosaic || length(downloaded) == 0) return(invisible(NULL))

  mosaic_path <- file.path(area$output_dir, sprintf("%s_mosaic.tif", area$name))
  mosaic_tiles(downloaded, mosaic_path, study_area_bounds(area))
}
