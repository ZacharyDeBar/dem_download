#' repair_dem_gaps.R
#' ------------------
#' Repairs a study-area DEM mosaic by:
#'   1. Identifying zero/nodata pixels (should not exist in these study areas)
#'   2. Downloading GLO-30 tiles for gap coverage
#'   3. Reprojecting GLO-30 onto the DEM grid
#'   4. Filling gaps with GLO-30 values
#'   5. Re-applying NHD still-water correction
#'
#' The original file is backed up (.tif.bak) before any changes are made.
#' Translation of python/repair_dem_gaps.py.
#'
#' Adaptation note: the Python original imports `download_tile_glo30`
#' from a module called `elevation` (python/elevation.py) -- this port
#' uses elevation.R's translation of the same module, not
#' dem_download.R's own download_tile().
#'
#' Usage:
#'   Rscript -e 'source("repair_dem_gaps.R"); a <- repair_area_between("N44W113", "N47W109"); repair_dem(a$dem_path, a$south, a$west, a$north, a$east, dry_run=TRUE)'

suppressPackageStartupMessages(library(terra))

source("tile_id.R")              # bounds_from_tile_corners()
source("elevation.R")            # download_tile_glo30()
source("dem_water_correction.R") # fetch_water_bodies_nhd(), correct_dem_water_bodies()

#' Build an area spec (list(south=, west=, north=, east=, dem_path=))
#' from two opposite 1x1-degree tile corners, e.g. "N44W113" and
#' "N47W109" -- any pairing/order. See bounds_from_tile_corners() in
#' tile_id.R for the exact corner rules and the size guard. `dem_path`
#' should point at the mosaic you actually want to repair -- this just
#' derives a default from `name` for convenience.
repair_area_between <- function(corner1_id, corner2_id, name = NULL,
                                 max_tiles = MAX_MOSAIC_TILES, allow_large = FALSE) {
  b <- bounds_from_tile_corners(corner1_id, corner2_id,
                                 max_tiles = max_tiles, allow_large = allow_large)
  if (is.null(name)) name <- sprintf("%s_%s", corner1_id, corner2_id)
  list(south = b$south, west = b$west, north = b$north, east = b$east,
       dem_path = sprintf("data/dem/%s/%s_corrected.tif", name, name))
}

# ------------------------------------------------
# Gap detection
# ------------------------------------------------

#' Detect zero/nodata pixels in a DEM. In these study areas all terrain
#' is above sea level, so a zero elevation indicates a missing 3DEP tile,
#' not real terrain -- water correction never legitimately produces
#' exactly 0.0 here.
detect_gaps <- function(dem_path, verbose = TRUE) {
  r <- rast(dem_path)
  data <- matrix(as.numeric(values(r)), nrow = nrow(r), ncol = ncol(r), byrow = TRUE)
  nodata_val <- terra::NAflag(r)
  if (is.na(nodata_val)) nodata_val <- -9999.0
  # unname(): see dem_water_correction.R's correct_dem_water_bodies for
  # why ext()'s $xmin/etc. accessors need this (name leakage through
  # arithmetic corrupts downstream JSON/print output otherwise).
  e_vals <- unname(as.vector(ext(r)))
  names(e_vals) <- c("xmin", "xmax", "ymin", "ymax")
  e <- ext(r)  # SpatExtent object, kept for rast(..., extent=) below

  nodata_mask <- is.na(data) | data <= (nodata_val + 1)
  zero_mask <- (data == 0.0) & !is.na(data)
  gap_mask <- nodata_mask | zero_mask

  n_total <- length(data)
  n_nodata <- sum(nodata_mask)
  n_zero <- sum(zero_mask) - sum(nodata_mask & zero_mask)
  n_gaps <- sum(gap_mask)

  if (verbose) {
    cat(sprintf("[GAP DETECT] DEM: %s x %s px = %s total pixels\n",
                format(ncol(r), big.mark = ","), format(nrow(r), big.mark = ","), format(n_total, big.mark = ",")))
    cat(sprintf("[GAP DETECT] Nodata pixels:     %s (%.2f%%)\n", format(n_nodata, big.mark = ","), n_nodata / n_total * 100))
    cat(sprintf("[GAP DETECT] Near-zero pixels:  %s (%.2f%%)\n", format(n_zero, big.mark = ","), n_zero / n_total * 100))
    cat(sprintf("[GAP DETECT] Total gap pixels:  %s (%.2f%%)\n", format(n_gaps, big.mark = ","), n_gaps / n_total * 100))
    cat(sprintf("[GAP DETECT] Valid pixels:      %s (%.2f%%)\n",
                format(n_total - n_gaps, big.mark = ","), (n_total - n_gaps) / n_total * 100))

    if (n_gaps > 0) {
      idx <- which(gap_mask, arr.ind = TRUE)
      rows0 <- idx[, "row"] - 1; cols0 <- idx[, "col"] - 1
      gap_lat_min <- e_vals[["ymax"]] + max(rows0) * (-res(r)[2])
      gap_lat_max <- e_vals[["ymax"]] + min(rows0) * (-res(r)[2])
      gap_lng_min <- e_vals[["xmin"]] + min(cols0) * res(r)[1]
      gap_lng_max <- e_vals[["xmin"]] + max(cols0) * res(r)[1]
      cat(sprintf("[GAP DETECT] Gap extent: lat [%.3f, %.3f] lng [%.3f, %.3f]\n",
                  gap_lat_min, gap_lat_max, gap_lng_min, gap_lng_max))
    }
  }

  list(gap_mask = gap_mask, n_gaps = n_gaps, n_total = n_total,
       nodata = nodata_val, shape = c(nrow(r), ncol(r)),
       ext = e, res = res(r), data = data)
}

# ------------------------------------------------
# GLO-30 tile fetching
# ------------------------------------------------

#' (lat_floor, lng_floor) for all GLO-30 tiles needed to cover an
#' extent (with a small padding buffer), as an n x 2 matrix.
get_required_glo30_tiles <- function(e, verbose = TRUE) {
  ev <- unname(as.vector(e))  # see detect_gaps for why (ext() name leakage)
  names(ev) <- c("xmin", "xmax", "ymin", "ymax")
  pad <- 0.1
  south <- ev[["ymin"]] - pad; north <- ev[["ymax"]] + pad
  west <- ev[["xmin"]] - pad; east <- ev[["xmax"]] + pad
  tiles <- as.matrix(expand.grid(lat = seq(floor(south), ceiling(north) - 1),
                                  lng = seq(floor(west), ceiling(east) - 1)))
  if (verbose) cat(sprintf("[GLO30] Requires %d tiles to cover DEM extent\n", nrow(tiles)))
  tiles
}

#' Download GLO-30 tiles and merge into a single in-memory mosaic.
#' Returns list(raster=, ext=) -- a SpatRaster plus its extent.
fetch_glo30_mosaic <- function(tiles, verbose = TRUE) {
  rasters <- list(); failed <- list()
  for (i in seq_len(nrow(tiles))) {
    lat_floor <- tiles[i, "lat"]; lng_floor <- tiles[i, "lng"]
    tile_center_lat <- lat_floor + 0.5
    tile_center_lng <- lng_floor + 0.5
    if (verbose) cat(sprintf("[GLO30] Tile %d/%d: (%d, %d)\n", i, nrow(tiles), lat_floor, lng_floor))
    result <- tryCatch(
      download_tile_glo30(tile_center_lat, tile_center_lng),
      error = function(e) NULL
    )
    if (!is.null(result)) {
      rasters[[length(rasters) + 1]] <- rast(result)
    } else {
      failed[[length(failed) + 1]] <- c(lat_floor, lng_floor)
      if (verbose) cat(sprintf("[GLO30] Warning: tile (%d,%d) failed\n", lat_floor, lng_floor))
    }
  }
  if (length(rasters) == 0) stop("[GLO30] No tiles downloaded successfully")

  if (verbose) cat(sprintf("[GLO30] Merging %d tiles...\n", length(rasters)))
  mosaic <- if (length(rasters) == 1) rasters[[1]] else do.call(terra::merge, rasters)

  if (verbose && length(failed) > 0) {
    cat(sprintf("[GLO30] Warning: %d tiles failed -- gaps may remain in those areas\n", length(failed)))
  }
  mosaic
}

# ------------------------------------------------
# Gap fill
# ------------------------------------------------

#' Reproject a GLO-30 mosaic onto the DEM's own grid and fill gap
#' pixels from it. Returns the repaired DEM matrix.
fill_gaps <- function(dem_data, gap_mask, dem_r, glo30_r, verbose = TRUE) {
  if (verbose) {
    cat(sprintf("[FILL] Reprojecting GLO-30 onto DEM grid (%s x %s px)...\n",
                format(nrow(dem_r), big.mark = ","), format(ncol(dem_r), big.mark = ",")))
  }
  glo30_resampled_r <- project(glo30_r, dem_r, method = "bilinear")
  glo30_resampled <- matrix(as.numeric(values(glo30_resampled_r)),
                             nrow = nrow(dem_r), ncol = ncol(dem_r), byrow = TRUE)
  glo30_resampled[is.na(glo30_resampled)] <- 0

  filled <- dem_data
  n_filled <- sum(gap_mask)
  filled[gap_mask] <- glo30_resampled[gap_mask]

  if (verbose) {
    cat(sprintf("[FILL] Filled %s gap pixels with GLO-30 values\n", format(n_filled, big.mark = ",")))
    still_zero <- gap_mask & (filled >= -1.0 & filled <= 1.0)
    n_still_zero <- sum(still_zero)
    if (n_still_zero > 0) {
      cat(sprintf("[FILL] Warning: %s pixels still near-zero after fill (likely ocean/water areas at GLO-30 extent)\n",
                  format(n_still_zero, big.mark = ",")))
    } else {
      cat("[FILL] All gap pixels successfully filled\n")
    }
  }
  filled
}

# ------------------------------------------------
# Water body correction
# ------------------------------------------------

#' Re-apply NHD still-water correction to the repaired DEM, in place.
apply_water_correction <- function(dem_path, south, west, north, east, verbose = TRUE) {
  if (verbose) cat("[WATER] Fetching NHD water bodies for study area...\n")
  result <- tryCatch({
    water_bounds <- c(south, west, north, east)
    water_polygons <- fetch_water_bodies_nhd(water_bounds)

    if (length(water_polygons) == 0) {
      cat("[WATER] No water bodies found -- skipping correction\n")
      return(TRUE)
    }
    if (verbose) {
      cat(sprintf("[WATER] Found %d water body polygons\n", length(water_polygons)))
      cat(sprintf("[WATER] Applying correction to %s...\n", dem_path))
    }
    correct_dem_water_bodies(dem_path, dem_path, water_polygons,
                              buffer_pixels = 3, shore_percentile = 10.0,
                              batch_size = 500, verbose = verbose)
    if (verbose) cat("[WATER] Water body correction complete\n")
    TRUE
  }, error = function(e) {
    cat(sprintf("[WATER] Error during water correction: %s\n", conditionMessage(e)))
    FALSE
  })
  isTRUE(result)
}

# ------------------------------------------------
# Main repair pipeline
# ------------------------------------------------

#' Full repair pipeline: detect gaps -> fetch GLO-30 -> fill -> write ->
#' verify -> re-apply water correction. Returns a status list; see the
#' Python original's repair_dem docstring for the field meanings.
repair_dem <- function(dem_path, south, west, north, east,
                        dry_run = FALSE, skip_water_correction = FALSE) {
  cat(sprintf("\n%s\n", strrep("=", 62)))
  cat("  DEM Gap Repair\n")
  cat(sprintf("  Input:  %s\n", dem_path))
  cat(sprintf("  Bounds: %gN %gE  ->  %gN %gE\n", south, west, north, east))
  cat(sprintf("  Mode:   %s\n", if (dry_run) "DRY RUN" else "REPAIR"))
  cat(sprintf("%s\n\n", strrep("=", 62)))

  t0 <- Sys.time()

  cat(sprintf("[STEP 1] Detecting gaps in %s...\n", dem_path))
  gap_info <- detect_gaps(dem_path, verbose = TRUE)

  if (gap_info$n_gaps == 0) {
    cat("\nNo gaps found -- DEM is complete, no repair needed\n")
    return(list(status = "no_gaps", n_gaps = 0))
  }
  cat(sprintf("\n-> %s gap pixels to fill (%.2f%% of DEM)\n",
              format(gap_info$n_gaps, big.mark = ","), gap_info$n_gaps / gap_info$n_total * 100))

  if (dry_run) {
    cat(sprintf("\n[DRY RUN] Would fill %s pixels from GLO-30 and re-apply water correction.\n",
                format(gap_info$n_gaps, big.mark = ",")))
    cat("[DRY RUN] No files modified.\n")
    return(list(status = "dry_run", n_gaps = gap_info$n_gaps))
  }

  # ---- Backup ----
  backup_path <- sub("\\.tif$", ".tif.bak", dem_path)
  if (!file.exists(backup_path)) {
    cat(sprintf("\n[STEP 2] Backing up original to %s...\n", backup_path))
    file.copy(dem_path, backup_path)
    cat(sprintf("  Backup saved: %s (%.0fMB)\n", backup_path, file.info(backup_path)$size / 1e6))
  } else {
    cat(sprintf("\n[STEP 2] Backup already exists: %s -- skipping\n", backup_path))
  }

  # ---- Fetch GLO-30 ----
  cat("\n[STEP 3] Fetching GLO-30 tiles for DEM extent...\n")
  tiles <- get_required_glo30_tiles(gap_info$ext, verbose = TRUE)
  glo30_r <- fetch_glo30_mosaic(tiles, verbose = TRUE)

  # ---- Fill gaps ----
  cat(sprintf("\n[STEP 4] Filling %s gap pixels...\n", format(gap_info$n_gaps, big.mark = ",")))
  dem_r <- rast(dem_path)
  repaired <- fill_gaps(gap_info$data, gap_info$gap_mask, dem_r, glo30_r, verbose = TRUE)

  # ---- Write repaired DEM ----
  cat(sprintf("\n[STEP 5] Writing repaired DEM to %s...\n", dem_path))
  out_r <- rast(repaired, extent = gap_info$ext, crs = "EPSG:4326")
  # BIGTIFF=YES explicit, not GDAL's BIGTIFF=IF_NEEDED default -- see
  # build_dem_mosaic.R's identical comment for the live failure this avoids.
  writeRaster(out_r, dem_path, filetype = "GTiff", datatype = "FLT4S",
              gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES"),
              overwrite = TRUE)
  cat(sprintf("  Written: %s (%.0fMB)\n", dem_path, file.info(dem_path)$size / 1e6))

  # ---- Verify ----
  cat("\n[STEP 6] Verifying repair...\n")
  post_gaps <- detect_gaps(dem_path, verbose = TRUE)
  n_remaining <- post_gaps$n_gaps
  n_fixed <- gap_info$n_gaps - n_remaining
  cat(sprintf("\n  Gaps fixed:     %s\n", format(n_fixed, big.mark = ",")))
  cat(sprintf("  Gaps remaining: %s (likely water/ocean areas at GLO-30 edge)\n", format(n_remaining, big.mark = ",")))

  # ---- Water correction ----
  if (skip_water_correction) {
    cat("\n[STEP 7] Water correction skipped (--skip-water-correction)\n")
    water_ok <- FALSE
  } else {
    cat("\n[STEP 7] Re-applying NHD water body correction...\n")
    water_ok <- apply_water_correction(dem_path, south, west, north, east, verbose = TRUE)
  }

  elapsed <- as.numeric(Sys.time() - t0, units = "secs")
  cat(sprintf("\n%s\n", strrep("=", 62)))
  cat(sprintf("  Repair complete in %.0fs\n", elapsed))
  cat(sprintf("  Gaps filled:        %s\n", format(n_fixed, big.mark = ",")))
  cat(sprintf("  Gaps remaining:     %s\n", format(n_remaining, big.mark = ",")))
  cat(sprintf("  Water correction:   %s\n", if (water_ok) "applied" else "skipped"))
  cat(sprintf("  Backup:             %s\n", backup_path))
  cat(sprintf("  Output:             %s\n", dem_path))
  cat(sprintf("%s\n\n", strrep("=", 62)))

  list(status = "repaired", n_gaps_original = gap_info$n_gaps, n_gaps_fixed = n_fixed,
       n_gaps_remaining = n_remaining, water_correction = water_ok,
       backup_path = backup_path, elapsed_s = round(elapsed, 1))
}
