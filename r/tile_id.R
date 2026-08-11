#' tile_id.R
#' ---------
#' Pure math: converts between (lat, lng) coordinates and tile IDs.
#'
#' A tile is a 1x1 degree square identified by its southwest corner.
#'   N45W110 = the tile covering latitudes 45.0N to 46.0N and
#'             longitudes 110.0W to 109.0W (SW corner at 45N,110W;
#'             the tile extends north and east from there).
#'
#' This file is pure functions over numbers and strings. No I/O, no
#' file access. Direct translation of python/tile_id.py -- see that
#' file for the original design notes; this port keeps only the notes
#' that affect correctness (boundary semantics, the R-specific gotchas
#' called out below) and drops the ones about sibling systems that
#' don't exist in this repo.
#'
#' Coordinate convention:
#'   - Latitudes increase northward, range -90 to +90
#'   - Longitudes increase eastward, range -180 to +180
#'   - Tile lat_floor is integer latitude (SW corner), e.g. 45 for N45
#'   - Tile lng_floor is integer longitude (SW corner), e.g. -111 for W111
#'
#' Boundary semantics:
#'   - A tile owns its southern and western edges (inclusive)
#'   - A tile does NOT own its northern or eastern edges (exclusive)
#'   - This means tile (45, -111) covers lat in [45, 46), lng in [-111, -110)
#'   - Point exactly at (46, -110) belongs to the next tiles north/east

# ------------------------------------------------
# Standard tile grid resolution
# ------------------------------------------------
#
# Single source of truth: every on-disk tile (DEM + flat + boundary
# masks) shares this pixel lattice, so tiles built from different
# native sources (3DEP 10m, GLO-30 30m) sit in the same VRT without a
# resolution seam. Coarser-native tiles are upsampled onto this grid
# at build time (never downsampled).
STANDARD_GRID_RES_DEG <- 1.0 / 10800.0  # ~10m at the equator
STANDARD_GRID_RES_RTOL <- 1e-4

#' True iff (res_x, res_y) match STANDARD_GRID_RES_DEG within tolerance.
is_standard_grid_resolution <- function(res_x_deg, res_y_deg) {
  abs(res_x_deg - STANDARD_GRID_RES_DEG) <= STANDARD_GRID_RES_RTOL * STANDARD_GRID_RES_DEG &&
    abs(res_y_deg - STANDARD_GRID_RES_DEG) <= STANDARD_GRID_RES_RTOL * STANDARD_GRID_RES_DEG
}

# ------------------------------------------------
# Tile identification
# ------------------------------------------------

#' Return the tile ID containing the given point.
#'
#' @examples
#' tile_id_from_latlng(45.5, -110.5)  # "N45W111"
#' tile_id_from_latlng(0.5, 0.5)      # "N00E000"
#' tile_id_from_latlng(-12.7, 178.3)  # "S13E178"
#'
#' Boundary behavior (consistent with [s,n) x [w,e) convention):
#' tile_id_from_latlng(46.0, -110.0)  # "N46W110" -- belongs to the next
#'                                    # tile north/east, not N45W111
tile_id_from_latlng <- function(lat, lng) {
  tile_id_from_floors(floor(lat), floor(lng))
}

#' Build a tile ID from integer SW-corner floors.
#'
#' For lng_floor=-111, the W component is 111 (the absolute value).
tile_id_from_floors <- function(lat_floor, lng_floor) {
  if (!(lat_floor >= -90 && lat_floor < 90)) {
    stop(sprintf("lat_floor %d outside [-90, 90)", lat_floor))
  }
  if (!(lng_floor >= -180 && lng_floor < 180)) {
    stop(sprintf("lng_floor %d outside [-180, 180)", lng_floor))
  }
  ns <- if (lat_floor >= 0) "N" else "S"
  ew <- if (lng_floor < 0) "W" else "E"
  sprintf("%s%02d%s%03d", ns, abs(lat_floor), ew, abs(lng_floor))
}

# Regex for parsing tile IDs of the form N45W110, S05E172, etc.
.TILE_ID_RE <- "^([NS])([0-9]{2})([EW])([0-9]{3})$"

#' Parse a tile ID into a list(lat_floor=, lng_floor=).
#'
#' @examples
#' parse_tile_id("N45W110")  # list(lat_floor = 45,  lng_floor = -110)
#' parse_tile_id("S13E178")  # list(lat_floor = -13, lng_floor = 178)
#' parse_tile_id("N00E000")  # list(lat_floor = 0,   lng_floor = 0)
parse_tile_id <- function(tile_id) {
  m <- regmatches(tile_id, regexec(.TILE_ID_RE, tile_id))[[1]]
  if (length(m) == 0) {
    stop(sprintf("Invalid tile ID format: %s (expected like 'N45W110')", tile_id))
  }
  ns <- m[2]; lat_str <- m[3]; ew <- m[4]; lng_str <- m[5]
  lat_floor <- as.integer(lat_str)
  lng_floor <- as.integer(lng_str)
  if (ns == "S") lat_floor <- -lat_floor
  if (ew == "W") lng_floor <- -lng_floor
  if (!(lat_floor >= -90 && lat_floor < 90)) {
    stop(sprintf("Parsed lat_floor %d from %s outside [-90, 90)", lat_floor, tile_id))
  }
  if (!(lng_floor >= -180 && lng_floor < 180)) {
    stop(sprintf("Parsed lng_floor %d from %s outside [-180, 180)", lng_floor, tile_id))
  }
  list(lat_floor = lat_floor, lng_floor = lng_floor)
}

# ------------------------------------------------
# Tile geometry
# ------------------------------------------------

#' A tile's geographic bounds. All values in decimal degrees.
#'
#' Following the [s, n) x [w, e) convention:
#'   - south, west: SW corner (owned by this tile)
#'   - north, east: NE corner (owned by next tiles N and E)
TileBounds <- function(south, west, north, east) {
  structure(
    list(south = south, west = west, north = north, east = east),
    class = "TileBounds"
  )
}

print.TileBounds <- function(x, ...) {
  cat(sprintf("TileBounds(south=%g, west=%g, north=%g, east=%g)\n",
              x$south, x$west, x$north, x$east))
  invisible(x)
}

#' (lat, lng) of a TileBounds' center.
tile_bounds_center <- function(tb) {
  c(lat = (tb$south + tb$north) / 2, lng = (tb$west + tb$east) / 2)
}

#' True if (lat, lng) falls inside tb's bounds.
tile_bounds_contains <- function(tb, lat, lng) {
  tb$south <= lat && lat < tb$north && tb$west <= lng && lng < tb$east
}

#' True if tb's bounds intersect the given bbox (standard rectangle-overlap test).
tile_bounds_intersects_bbox <- function(tb, south, west, north, east) {
  !(east <= tb$west || west >= tb$east || north <= tb$south || south >= tb$north)
}

#' (south, west, north, east) -- matches the existing bounds order.
tile_bounds_as_tuple <- function(tb) {
  c(tb$south, tb$west, tb$north, tb$east)
}

#' Return the geographic bounds of a tile by its ID.
#'
#' Note that the tile's west longitude is its lng_floor (the SW corner),
#' NOT its NW corner. So N45W110 covers lng in [-110, -109).
#'
#' @examples
#' tile_bounds("N45W110")
#' # TileBounds(south=45, west=-110, north=46, east=-109)
tile_bounds <- function(tile_id) {
  p <- parse_tile_id(tile_id)
  TileBounds(
    south = as.double(p$lat_floor),
    west  = as.double(p$lng_floor),
    north = as.double(p$lat_floor + 1),
    east  = as.double(p$lng_floor + 1)
  )
}

# ------------------------------------------------
# Bbox -> tile set
# ------------------------------------------------

#' Return all tile IDs whose bounds intersect the given bbox.
#'
#' The returned vector is ordered by (lat_floor, lng_floor) -- south to
#' north, then west to east.
#'
#' Boundary behavior:
#'   A bbox whose north edge lies exactly on a tile boundary does NOT
#'   include the next tile north -- consistent with [s,n)x[w,e). But
#'   the eastern bbox edge being slightly past a tile boundary DOES
#'   include the next tile east.
#'
#' @examples
#' tile_ids_covering_bbox(45.2, -110.7, 45.9, -109.3)
#' # c("N45W110", "N45W109")
tile_ids_covering_bbox <- function(south, west, north, east) {
  if (south >= north) stop(sprintf("south %s >= north %s", south, north))
  if (west >= east) stop(sprintf("west %s >= east %s", west, east))

  lat_min <- floor(south)
  lat_max <- ceiling(north)
  lng_min <- floor(west)
  lng_max <- ceiling(east)

  # south < north and west < east (checked above) guarantee lat_max >
  # lat_min and lng_max > lng_min, so these ranges are never empty --
  # but guard explicitly anyway: unlike Python's range(), R's seq()/`:`
  # count DOWNWARD when the upper bound is smaller than the lower one,
  # rather than producing an empty sequence. Silent wrong-direction
  # iteration is a much worse failure mode than an explicit empty guard.
  lat_floors <- if (lat_max > lat_min) seq(lat_min, lat_max - 1) else integer(0)
  lng_floors <- if (lng_max > lng_min) seq(lng_min, lng_max - 1) else integer(0)

  tile_ids <- character(0)
  for (lat_floor in lat_floors) {
    for (lng_floor in lng_floors) {
      tile_ids <- c(tile_ids, tile_id_from_floors(lat_floor, lng_floor))
    }
  }
  tile_ids
}

#' Return all tile IDs whose bounds intersect the buffered polyline.
#'
#' @param latlngs  a two-column matrix/data.frame of (lat, lng) points,
#'   or a list of c(lat, lng) pairs. Empty input returns character(0).
#' @param buffer_m buffer width in metres (typically the compute radius
#'   plus a small safety margin).
#'
#' Implementation: compute the polyline's bbox, expand by the buffer in
#' degrees (equirectangular approximation), then call
#' tile_ids_covering_bbox. This may include slightly more tiles than
#' strictly necessary at polyline curves; the over-coverage is bounded
#' and harmless.
#'
#' @examples
#' tile_ids_covering_polyline(
#'   rbind(c(45.0, -109.5), c(45.0, -109.0)),
#'   buffer_m = 10000
#' )
#' # c("N44W110", "N44W109", "N45W110", "N45W109")
tile_ids_covering_polyline <- function(latlngs, buffer_m) {
  if (is.list(latlngs) && !is.data.frame(latlngs)) {
    if (length(latlngs) == 0) return(character(0))
    latlngs <- do.call(rbind, latlngs)
  }
  if (is.null(latlngs) || nrow(latlngs) == 0) return(character(0))

  lats <- latlngs[, 1]
  lngs <- latlngs[, 2]
  s_lat <- min(lats)
  n_lat <- max(lats)
  w_lng <- min(lngs)
  e_lng <- max(lngs)

  # Equirectangular buffer (lat is uniform, lng depends on cos(lat))
  DEG_PER_M_LAT <- 1.0 / 111320.0
  mean_lat <- (s_lat + n_lat) / 2.0
  cos_lat <- max(cos(mean_lat * pi / 180), 1e-6)
  deg_per_m_lng <- DEG_PER_M_LAT / cos_lat

  buf_lat <- buffer_m * DEG_PER_M_LAT
  buf_lng <- buffer_m * deg_per_m_lng

  tile_ids_covering_bbox(
    south = s_lat - buf_lat,
    west  = w_lng - buf_lng,
    north = n_lat + buf_lat,
    east  = e_lng + buf_lng
  )
}

# ------------------------------------------------
# Neighbor helpers
# ------------------------------------------------

#' Return the 8 neighbors of a tile keyed by compass direction.
#'
#' Returns a named list with keys N, S, E, W, NE, NW, SE, SW. Values
#' are tile ID strings, or NULL for any neighbor that would fall
#' outside the valid lat/lng range (e.g. polar neighbors).
#'
#' R gotcha: `out[[name]] <- NULL` would REMOVE that key from the list
#' rather than set its value to NULL (unlike Python's `d[key] = None`,
#' which keeps the key). Using `out[name] <- list(NULL)` instead, so
#' the returned list always has exactly these 8 keys, matching the
#' Python original.
#'
#' @examples
#' n <- neighbor_tile_ids("N45W110")
#' n[["N"]]   # "N46W110"
#' n[["SE"]]  # "N44W109"
neighbor_tile_ids <- function(tile_id) {
  p <- parse_tile_id(tile_id)
  lat_floor <- p$lat_floor
  lng_floor <- p$lng_floor

  deltas <- list(
    N  = c(1, 0),  S  = c(-1, 0),
    E  = c(0, 1),  W  = c(0, -1),
    NE = c(1, 1),  NW = c(1, -1),
    SE = c(-1, 1), SW = c(-1, -1)
  )

  out <- vector("list", length(deltas))
  names(out) <- names(deltas)
  for (name in names(deltas)) {
    d <- deltas[[name]]
    nlat <- lat_floor + d[1]
    nlng <- lng_floor + d[2]
    if (!(nlat >= -90 && nlat < 90)) {
      out[name] <- list(NULL)
      next
    }
    if (!(nlng >= -180 && nlng < 180)) {
      # Could wrap longitude here, but for our purposes the
      # antimeridian doesn't matter -- flagging as NULL is safer.
      out[name] <- list(NULL)
      next
    }
    out[[name]] <- tile_id_from_floors(nlat, nlng)
  }
  out
}

# ------------------------------------------------
# Corners -> bounds, with a size guard
# ------------------------------------------------

# Above this many 1x1-degree tiles, bounds_from_tile_corners() refuses
# unless allow_large=TRUE. Not an arbitrary number: a real 20-tile,
# source="auto" (mostly 3DEP_10) mosaic build during this port's own
# testing filled the disk partway through tile 9 -- each 3DEP 10m tile
# merges from an 81-subtile WCS grid to roughly 200-300MB before
# GLO-30 gap-fill and water correction add more. 25 tiles (a 5x5deg
# block) is comfortably past typical single-region use without being
# an accident waiting to happen.
MAX_MOSAIC_TILES <- 25

#' Given two 1x1-degree tile IDs marking any two opposite corners of a
#' desired rectangular study area (e.g. "N44W113" and "N47W109" --
#' order doesn't matter, NE+SW or NW+SE or any other pairing all work),
#' return the bounding list(south=, west=, north=, east=, n_tiles=)
#' spanning both tiles' full footprints.
#'
#' Longitude is resolved to the SHORTER arc between the two corners
#' (e.g. corners at 10E and 20E span 10 degrees eastward, not 350
#' degrees the other way around) -- almost always what's intended for
#' a regional mosaic, and avoids requiring a third "which way around"
#' argument. If that shorter arc still crosses the +/-180 antimeridian
#' -- i.e. the corners themselves straddle the date line -- this
#' raises rather than silently producing a reversed or wrapped bbox:
#' nothing downstream (crop/merge/tile enumeration) supports a
#' dateline-crossing extent. Build two separate mosaics instead.
#'
#' @param max_tiles    size cap in degree-tiles (see MAX_MOSAIC_TILES)
#' @param allow_large  set TRUE to bypass the cap for a real large run
#'
#' @examples
#' bounds_from_tile_corners("N44W113", "N47W109")
#' # list(south=44, west=-113, north=48, east=-108, n_tiles=20)
bounds_from_tile_corners <- function(corner1_id, corner2_id,
                                      max_tiles = MAX_MOSAIC_TILES,
                                      allow_large = FALSE) {
  b1 <- tile_bounds(corner1_id)
  b2 <- tile_bounds(corner2_id)

  south <- min(b1$south, b2$south)
  north <- max(b1$north, b2$north)

  w1 <- b1$west; w2 <- b2$west
  delta <- ((w2 - w1 + 180) %% 360) - 180  # signed shortest arc, (-180, 180]
  if (delta >= 0) {
    west <- w1; east <- w2 + 1
  } else {
    west <- w2; east <- w1 + 1
  }
  if (east <= west) {
    stop(sprintf(
      "bounds_from_tile_corners(%s, %s): the shorter arc between these tiles crosses the +/-180 antimeridian (west=%g, east=%g), which isn't supported downstream. Pick corners that don't straddle the date line, or build two separate mosaics on either side of it.",
      corner1_id, corner2_id, west, east))
  }

  n_tiles <- length(tile_ids_covering_bbox(south, west, north, east))
  if (n_tiles > max_tiles && !allow_large) {
    stop(sprintf(
      "bounds_from_tile_corners(%s, %s): area spans %d degree-tiles, above the %d-tile default cap. 3DEP 10m tiles run roughly 200-300MB each -- a mosaic this size can be several GB and take a long time to download. Pick closer corners, or pass allow_large=TRUE (optionally with a higher max_tiles) if you really want this.",
      corner1_id, corner2_id, n_tiles, max_tiles))
  }
  if (n_tiles > 9) {
    cat(sprintf("[BOUNDS] Note: %d degree-tiles required for %s to %s -- large mosaics take a while and real disk space (each 3DEP 10m tile is roughly 200-300MB).\n",
                n_tiles, corner1_id, corner2_id))
  }

  list(south = south, west = west, north = north, east = east, n_tiles = n_tiles)
}
