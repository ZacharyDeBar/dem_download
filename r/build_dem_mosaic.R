#' build_dem_mosaic.R
#' -------------------
#' Downloads and assembles a DEM mosaic for a study area: all 1x1 degree
#' tiles covering the (buffered) bounds, merged into one GeoTIFF, with
#' still-water correction applied. Translation of python/build_dem_mosaic.py.
#'
#' Adaptation note: the Python original imports `download_tile` and
#' `_is_in_3dep_coverage` from a module called `elevation`
#' (python/elevation.py) -- this port uses elevation.R's translation of
#' the same module, not dem_download.R's own download_tile(). The two
#' are functionally distinct: elevation.R's download_tile() caches to
#' its own on-disk cache (CACHE_DIR) keyed by tile name and supports
#' sub-tiled 3DEP WCS downloads, neither of which dem_download.R does.
#'
#' Usage:
#'   Rscript -e 'source("build_dem_mosaic.R"); build_mosaic_between("N44W113", "N47W109")'
#'   Rscript -e 'source("build_dem_mosaic.R"); build_mosaic_between("N44W113", "N47W109", source="3DEP_10")'
#'
#' (build_mosaic() itself is still available directly for raw
#' fractional-degree bounds -- see its own docstring below.)

suppressPackageStartupMessages(library(terra))

source("tile_id.R")              # bounds_from_tile_corners()
source("elevation.R")            # download_tile(), is_in_3dep_coverage()
source("dem_water_correction.R") # fetch_water_bodies_nhd(), fetch_ocean_polygons_osm(),
                                  # correct_dem_water_bodies()

#' Build an area spec (list(south=, west=, north=, east=, dem_path=))
#' from two opposite 1x1-degree tile corners, e.g. "N44W113" and
#' "N47W109" -- any pairing/order. See bounds_from_tile_corners() in
#' tile_id.R for the exact corner rules and the size guard (large
#' requests raise unless allow_large=TRUE).
#'
#' `name` defaults to "<corner1>_<corner2>" and only affects the
#' output path this returns (dem_path); build_mosaic() itself doesn't
#' care what output_path you pass it.
mosaic_area_between <- function(corner1_id, corner2_id, name = NULL,
                                 max_tiles = MAX_MOSAIC_TILES, allow_large = FALSE) {
  b <- bounds_from_tile_corners(corner1_id, corner2_id,
                                 max_tiles = max_tiles, allow_large = allow_large)
  if (is.null(name)) name <- sprintf("%s_%s", corner1_id, corner2_id)
  list(south = b$south, west = b$west, north = b$north, east = b$east,
       dem_path = sprintf("data/dem/%s/%s_corrected.tif", name, name))
}

#' Build a mosaic spanning two opposite 1x1-degree tile corners in one
#' call. Composes mosaic_area_between() + build_mosaic() with
#' buffer_deg=0 -- NOT build_mosaic()'s own default (0.1). That default
#' exists for build_mosaic()'s other caller, raw fractional-degree
#' bounds (e.g. from --bounds), where the requested edges can fall
#' mid-tile and a small margin avoids a gap right at the boundary.
#' Corner-tile bounds are already exactly tile-aligned, so the same
#' 0.1deg margin only pushes the request outward across an *extra*
#' whole tile on every side -- confirmed live: it silently turned a
#' 16-tile (4x4deg) request into 36 tiles (6x6deg), well past the
#' 25-tile cap the corners alone had just cleared, and the disk/time
#' cost more than doubled. Use this function (or pass buffer_deg=0
#' explicitly to build_mosaic()) for any corner-based area.
build_mosaic_between <- function(corner1_id, corner2_id, output_path = NULL,
                                  name = NULL, source = "auto",
                                  apply_water_correction = TRUE,
                                  max_tiles = MAX_MOSAIC_TILES, allow_large = FALSE) {
  a <- mosaic_area_between(corner1_id, corner2_id, name = name,
                            max_tiles = max_tiles, allow_large = allow_large)
  if (is.null(output_path)) output_path <- a$dem_path
  build_mosaic(a$south, a$west, a$north, a$east, output_path,
               source = source, apply_water_correction = apply_water_correction,
               buffer_deg = 0)
}

#' All (lat_floor, lng_floor) 1-degree tiles overlapping a bbox, as an
#' n x 2 integer matrix.
get_required_degree_tiles <- function(south, west, north, east) {
  lats <- seq(floor(south), ceiling(north) - 1)
  lngs <- seq(floor(west), ceiling(east) - 1)
  as.matrix(expand.grid(lat = lats, lng = lngs))
}

#' Download all required tiles and merge into a single mosaic GeoTIFF,
#' with still-water correction applied.
#'
#' @param source 'auto', 'GLO30', or '3DEP_10'
#' @param apply_water_correction  correct still-water bodies after merging
#' @param buffer_deg  extra buffer beyond the study area bounds, degrees
build_mosaic <- function(south, west, north, east, output_path,
                          source = "auto", apply_water_correction = TRUE,
                          buffer_deg = 0.1) {
  s <- south - buffer_deg; w <- west - buffer_deg
  n <- north + buffer_deg; e <- east + buffer_deg

  tiles <- get_required_degree_tiles(s, w, n, e)
  cat(sprintf("[MOSAIC] Study area: %.2fN %.2fE to %.2fN %.2fE\n", south, west, north, east))
  cat(sprintf("[MOSAIC] Source: %s\n", source))
  cat(sprintf("[MOSAIC] Tiles required: %d\n", nrow(tiles)))
  cat(sprintf("[MOSAIC] Output: %s\n", output_path))
  t0 <- Sys.time()

  # ---- Download tiles ----
  tile_paths <- list()
  for (i in seq_len(nrow(tiles))) {
    lat_floor <- tiles[i, "lat"]; lng_floor <- tiles[i, "lng"]
    tile_center_lat <- lat_floor + 0.5
    tile_center_lng <- lng_floor + 0.5

    tile_source <- source
    if (source == "auto") {
      tile_source <- if (is_in_3dep_coverage(tile_center_lat, tile_center_lng)) "3DEP_10" else "GLO30"
    }

    cat(sprintf("[MOSAIC] Tile %d/%d: (%d,%d) source=%s\n",
                i, nrow(tiles), lat_floor, lng_floor, tile_source))
    result <- tryCatch(
      download_tile(tile_center_lat, tile_center_lng, source = tile_source),
      error = function(e) {
        cat(sprintf("[MOSAIC] Warning: tile (%d,%d) failed: %s\n", lat_floor, lng_floor, conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(result)) tile_paths[[length(tile_paths) + 1]] <- result
  }

  if (length(tile_paths) == 0) stop("[MOSAIC] No tiles downloaded successfully")
  cat(sprintf("\n[MOSAIC] Downloaded %d/%d tiles in %.0fs\n",
              length(tile_paths), nrow(tiles), as.numeric(Sys.time() - t0, units = "secs")))

  # ---- Merge tiles ----
  cat(sprintf("[MOSAIC] Merging %d tiles...\n", length(tile_paths)))
  t_merge <- Sys.time()
  rasters <- lapply(tile_paths, rast)
  mosaic <- do.call(terra::merge, rasters)
  cat(sprintf("[MOSAIC] Merged shape: %s x %s (%.0fM pixels) in %.0fs\n",
              format(ncol(mosaic), big.mark = ","), format(nrow(mosaic), big.mark = ","),
              ncell(mosaic) / 1e6, as.numeric(Sys.time() - t_merge, units = "secs")))

  # ---- Crop to study area + buffer ----
  cat("[MOSAIC] Cropping to study area...\n")
  cropped <- crop(mosaic, ext(w, e, s, n))

  # ---- Write, then optionally correct still-water in place ----
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  # BIGTIFF=YES explicit, not GDAL's BIGTIFF=IF_NEEDED default: IF_NEEDED
  # predicts final size assuming compression keeps it under 4GB and only
  # upgrades if that prediction says otherwise -- real DEM terrain
  # doesn't compress reliably enough for that bet. The equivalent
  # Python write hit exactly this ("TIFFAppendToStrip: Maximum TIFF
  # file size exceeded") on a real 4x4-degree mosaic (~4.3GB actual
  # output); this R write happened to succeed on the same area, but
  # only because whatever BIGTIFF heuristic this terra/GDAL build uses
  # by default got lucky here -- not something worth relying on.
  writeRaster(cropped, output_path, filetype = "GTiff", datatype = "FLT4S",
              gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES"),
              overwrite = TRUE)

  if (apply_water_correction) {
    cat("[MOSAIC] Applying NHD water body correction...\n")
    result <- tryCatch({
      water_bounds <- c(s, w, n, e)
      water_polygons <- fetch_water_bodies_nhd(water_bounds)

      ocean_polys <- tryCatch(fetch_ocean_polygons_osm(water_bounds), error = function(e) {
        cat(sprintf("[MOSAIC] Ocean fetch warning: %s\n", conditionMessage(e)))
        list()
      })
      if (length(ocean_polys) > 0) {
        cat(sprintf("[MOSAIC] Adding %d ocean polygons\n", length(ocean_polys)))
        water_polygons <- c(water_polygons, ocean_polys)
      }

      if (length(water_polygons) > 0) {
        correct_dem_water_bodies(output_path, output_path, water_polygons,
                                  buffer_pixels = 3, shore_percentile = 10.0,
                                  batch_size = 500, verbose = TRUE)
      } else {
        cat("[MOSAIC] No water bodies found -- skipping correction\n")
      }
      TRUE
    }, error = function(e) {
      cat(sprintf("[MOSAIC] Water correction warning: %s\n", conditionMessage(e)))
      cat("[MOSAIC] Mosaic written uncorrected\n")
      FALSE
    })
  }

  final_r <- rast(output_path)
  output_mb <- file.info(output_path)$size / 1e6
  elapsed <- as.numeric(Sys.time() - t0, units = "secs")

  stats <- list(
    output_path = output_path, source = source,
    shape = c(nrow(final_r), ncol(final_r)),
    resolution_m = res(final_r)[1] * 111320,
    n_tiles = length(tile_paths),
    output_mb = round(output_mb, 1), elapsed_s = round(elapsed, 1)
  )

  cat(sprintf("\n[MOSAIC] Complete in %.0fs\n", elapsed))
  cat(sprintf("  Shape:      %s x %s\n", format(stats$shape[2], big.mark = ","), format(stats$shape[1], big.mark = ",")))
  cat(sprintf("  Resolution: %.1fm/px\n", stats$resolution_m))
  cat(sprintf("  Output:     %s (%.1fMB)\n", output_path, output_mb))

  stats
}
