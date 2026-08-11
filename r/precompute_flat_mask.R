#' precompute_flat_mask.R
#' ----------------------
#' Precomputes the flat-zone mask for a DEM: a per-pixel "is local
#' elevation variance below a threshold" mask, and a 4-connected
#' boundary mask derived from it. Translation of the compute_flat_mask
#' and compute_flat_regions functions in python/precompute_flat_mask.py
#' (see that file for the fuller design notes).
#'
#' Scope note: the Python original also has load_flat_mask/
#' load_flat_boundary/load_flat_mask_packed/load_flat_boundary_packed/
#' slice_flat_mask -- readers and a bit-packing scheme that exist
#' purely to serve a downstream compute engine's per-observer cache
#' slicing. That engine isn't part of this repo (see README's Scope
#' section), and nothing in the R core pipeline calls them either, so
#' they're dropped here rather than translated to serve a caller that
#' doesn't exist in this port.
#'
#' Implementation note vs. the Python original: python/precompute_flat_mask.py
#' manually loops over row strips to keep peak RAM bounded, because
#' scipy.ndimage operates on a fully in-memory numpy array. terra's
#' focal()/global() work directly against the on-disk raster and chunk
#' internally (see terra::terraOptions() for the memory budget), so
#' this port doesn't need a manual strip loop to get the same bounded-
#' RAM behavior -- letting terra do it is the idiomatic choice here,
#' not a simplification that changes what gets computed.

suppressPackageStartupMessages(library(terra))

# terra's default tempdir (usually under /tmp) can be a small,
# RAM-backed tmpfs rather than real disk -- see elevation.R's module
# comment for the live failure this caused. Point it at the project
# disk instead.
dir.create("raster_tmp", showWarnings = FALSE)
terraOptions(tempdir = "raster_tmp")

# ------------------------------------------------
# Parameters -- must match the downstream compute engine's own
# flat-zone computation exactly (outside this repo; see README)
# ------------------------------------------------

FLAT_ZONE_WINDOW   <- 5     # focal window size
FLAT_ZONE_VARIANCE <- 4.0   # variance threshold in m^2

# ------------------------------------------------
# Core computation
# ------------------------------------------------

#' Compute and save a flat-zone mask for a DEM GeoTIFF.
#'
#' @return list of stats: flat_pct, total_flat, total_pixels, window,
#'   threshold, elev_min_m, elev_max_m, elev_range_m (NULL if the DEM
#'   has no valid/non-nodata pixels at all -- see tile_builder.py's
#'   elev_range_m==NULL validation gate for why that's checked so
#'   strictly on the Python side of this pipeline).
compute_flat_mask <- function(dem_path, output_path,
                               window = FLAT_ZONE_WINDOW,
                               threshold = FLAT_ZONE_VARIANCE) {
  cat(sprintf("[FLAT MASK] Input:  %s\n", dem_path))
  cat(sprintf("[FLAT MASK] Output: %s\n", output_path))
  cat(sprintf("[FLAT MASK] Parameters: window=%d, threshold=%g\n", window, threshold))
  t0 <- Sys.time()

  r <- rast(dem_path)
  cat(sprintf("[FLAT MASK] DEM size: %s x %s pixels (%.0fM pixels)\n",
              format(ncol(r), big.mark = ","), format(nrow(r), big.mark = ","),
              ncell(r) / 1e6))

  # Elevation range over valid (non-NA) pixels only, tracked exactly
  # like the Python original's elev_min/elev_max -- NULL (not 0) when
  # there are no valid pixels at all, which is the signature a broken
  # nodata read leaves behind (see tile_builder.py's gate for why that
  # must never be silently treated as "range 0").
  rng <- global(r, "range", na.rm = TRUE)
  elev_min <- rng[1, "min"]
  elev_max <- rng[1, "max"]
  has_valid <- is.finite(elev_min) && is.finite(elev_max)
  if (!has_valid) { elev_min <- NULL; elev_max <- NULL }

  # Box filter (uniform_filter's box mean, "reflect" edges) for E[x]
  # and E[x^2]; variance = E[x^2] - E[x]^2. na.policy="omit" +
  # na.rm=TRUE excludes NA neighbors from the local mean rather than
  # propagating NA outward, matching the Python original's zero-fill
  # of nodata before filtering closely enough for this mask's purpose
  # (a boundary-adjacent nodata pixel making its neighborhood read as
  # "not flat" either way).
  w <- matrix(1, window, window)
  mean_r    <- focal(r,     w = w, fun = "mean", na.policy = "omit", na.rm = TRUE)
  mean_sq_r <- focal(r ^ 2, w = w, fun = "mean", na.policy = "omit", na.rm = TRUE)
  variance_r <- mean_sq_r - mean_r ^ 2

  flat_r <- variance_r < threshold
  names(flat_r) <- "flat"

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  writeRaster(flat_r, output_path, filetype = "GTiff", datatype = "INT1U",
              gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"),
              overwrite = TRUE, NAflag = NA)

  flat_summary <- global(flat_r, "sum", na.rm = TRUE)
  total_flat   <- as.integer(flat_summary[1, "sum"])
  total_pixels <- as.integer(ncell(flat_r))
  flat_pct     <- total_flat / total_pixels * 100
  mask_mb      <- file.info(output_path)$size / 1e6
  elapsed      <- as.numeric(Sys.time() - t0, units = "secs")

  elev_range_m <- if (has_valid) elev_max - elev_min else NULL

  cat(sprintf("\n[FLAT MASK] Complete in %.1fs\n", elapsed))
  cat(sprintf("  Flat pixels:  %s / %s (%.1f%%)\n",
              format(total_flat, big.mark = ","), format(total_pixels, big.mark = ","), flat_pct))
  cat(sprintf("  Elevation range: %s\n",
              if (has_valid) sprintf("%.2f", elev_range_m) else "n/a (no valid pixels)"))
  cat(sprintf("  Output size:  %.1f MB\n", mask_mb))
  cat(sprintf("  Output:       %s\n", output_path))

  list(
    dem_path = dem_path, output_path = output_path,
    h = nrow(r), w = ncol(r),
    flat_pct = round(flat_pct, 2), total_flat = total_flat, total_pixels = total_pixels,
    elapsed_s = round(elapsed, 1), mask_mb = round(mask_mb, 1),
    window = window, threshold = threshold,
    elev_min_m = if (has_valid) round(elev_min, 2) else NULL,
    elev_max_m = if (has_valid) round(elev_max, 2) else NULL,
    elev_range_m = if (has_valid) round(elev_range_m, 2) else NULL
  )
}

# ------------------------------------------------
# Flat mask compressor
# ------------------------------------------------

#' Compute the 4-connected boundary mask from an already-computed flat
#' mask: a flat cell is a boundary cell if any N/S/E/W neighbor is
#' non-flat. This is a drop-in cache-key reduction for the downstream
#' compute engine's flat-zone caching -- only need to cache boundary
#' crossings, not every flat cell (~10-50x fewer entries).
#'
#' Erosion via focal(fun="min") over a cross-shaped mask: a pixel
#' survives erosion (stays 1) only if every 4-connected neighbor is
#' also 1 -- the standard definition of binary erosion with that
#' structuring element. pad=TRUE, padValue=0 treats out-of-raster
#' neighbors as non-flat, matching scipy.ndimage.binary_erosion's
#' border_value=0.
#'
#' @return list of stats; also writes {output_path}_boundary.tif and
#'   {output_path}_boundary_stats.json (paths mirror the Python
#'   original's `.replace('.tif', '_boundary.tif')` convention).
compute_flat_regions <- function(flat_mask_path, output_path) {
  cat(sprintf("[FLAT REGIONS] Input mask: %s\n", flat_mask_path))
  t0 <- Sys.time()

  flat_r <- rast(flat_mask_path)
  flat_pct <- global(flat_r, "mean", na.rm = TRUE)[1, "mean"] * 100
  cat(sprintf("[FLAT REGIONS] Mask: %sx%s, %.1f%% flat\n",
              format(ncol(flat_r), big.mark = ","), format(nrow(flat_r), big.mark = ","), flat_pct))

  cat("[FLAT REGIONS] Computing boundary mask...\n")
  # NA (not 0) excludes a cell from the neighborhood in terra's focal() --
  # a 0 there would multiply into the window like any other weight,
  # zeroing out fun="min"'s result everywhere (confirmed the hard way:
  # an earlier 0-cornered version of this matrix made almost the entire
  # flat mask read as "boundary", since every window's min got dragged
  # to 0 by its masked-out corners rather than actually being excluded).
  cross <- matrix(c(NA, 1, NA, 1, 1, 1, NA, 1, NA), nrow = 3)  # 4-connected
  eroded <- focal(flat_r, w = cross, fun = "min", na.rm = TRUE, fillvalue = 0)
  boundary_r <- (flat_r == 1) & (eroded == 0)
  names(boundary_r) <- "boundary"

  boundary_sum <- as.integer(global(boundary_r, "sum", na.rm = TRUE)[1, "sum"])
  flat_sum     <- as.integer(global(flat_r, "sum", na.rm = TRUE)[1, "sum"])
  boundary_pct <- boundary_sum / ncell(boundary_r) * 100
  reduction    <- (1 - boundary_sum / max(flat_sum, 1)) * 100

  cat(sprintf("[FLAT REGIONS] Boundary cells: %s (%.2f%% of DEM, %.1f%% reduction vs full flat mask)\n",
              format(boundary_sum, big.mark = ","), boundary_pct, reduction))

  boundary_path <- sub("\\.tif$", "_boundary.tif", output_path)
  dir.create(dirname(boundary_path), recursive = TRUE, showWarnings = FALSE)
  writeRaster(boundary_r, boundary_path, filetype = "GTiff", datatype = "INT1U",
              gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"),
              overwrite = TRUE, NAflag = NA)

  boundary_mb <- file.info(boundary_path)$size / 1e6
  elapsed <- as.numeric(Sys.time() - t0, units = "secs")
  cat(sprintf("[FLAT REGIONS] Written: %s (%.1fMB, %.1fs)\n", boundary_path, boundary_mb, elapsed))

  stats <- list(
    flat_mask_path = flat_mask_path, boundary_path = boundary_path,
    h = nrow(flat_r), w = ncol(flat_r),
    flat_pct = round(flat_pct, 2), boundary_pct = round(boundary_pct, 2),
    flat_cells = flat_sum, boundary_cells = boundary_sum,
    reduction_pct = round(reduction, 1),
    elapsed_s = round(elapsed, 1), boundary_mb = round(boundary_mb, 1)
  )

  stats_path <- sub("_boundary\\.tif$", "_boundary_stats.json", boundary_path)
  jsonlite::write_json(stats, stats_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

  stats
}
