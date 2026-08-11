#' dem_water_correction.R
#' -----------------------
#' Corrects elevation artifacts in still-water bodies (lakes, ponds,
#' reservoirs) by sampling shoreline elevation and flood-filling water
#' body pixels to a consistent surface elevation. Deliberately
#' still-water only -- rivers, marshes, and glaciers aren't
#' single-elevation surfaces, so flattening them the same way would be
#' wrong; see NHD_STILL_WATER_FTYPES below.
#'
#' Translation of python/dem_water_correction.py's serial/CPU path.
#' Dropped from this port (see r/README.md for the full scope note):
#'   - the GPU path (process_label_array_gpu et al.) and the
#'     multiprocessing parallel path (_parallel_worker/correct_dem_parallel)
#'     -- gpu_tools.py/parallel_tools.py are out of scope for this port
#'   - sample_shoreline_elevation() -- defined in the Python original but
#'     never called anywhere in it; dead code, not translated
#'   - the batch/preview/index CLI subcommands -- only `correct` (the
#'     one path this repo's pipeline and README examples actually use)
#'     is ported
#'
#' Dependency note: polygon construction throughout uses terra's own
#' SpatVector (via WKT strings), not sf -- sf pulls in the `units`
#' package, which needs the system libudunits2 (not available via
#' pacman on this machine without an AUR build). The one place sf
#' would have been genuinely convenient -- fetch_ocean_polygons_osm's
#' polygonize/union/difference chain, building ocean-fill polygons from
#' coastline linework -- checks for sf at runtime and falls back to
#' "treat the whole bbox as ocean" (the same fallback the Python
#' original already uses when no coastline is found at all) if sf
#' isn't installed, rather than hard-failing the whole file on a
#' dependency needed by one best-effort feature.
#'
#' Usage:
#'   Rscript dem_water_correction.R correct \
#'     --input data/dem/raw/N45W111.tif \
#'     --output data/dem/corrected/N45W111.tif \
#'     --source nhd

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

# ------------------------------------------------
# Water body fetching
# ------------------------------------------------

# NHD FType codes for genuinely flat, still-water surfaces -- the only
# kind the shoreline-percentile flattening below is valid for. See
# dem_water_correction.py's module docstring for the full incident
# writeup (N45W111, 2026-08-08: SwampMarsh/IceMass mixed in alongside
# real lakes produced 80m+ spurious corrections).
NHD_STILL_WATER_FTYPES <- c(
  390,  # LakePond
  436,  # Reservoir
  493   # Estuary
)

#' Fetch water body polygons from OpenStreetMap via the Overpass API.
#' Returns a list of GeoJSON-like polygon dicts: list(type=, coordinates=,
#' properties=). Deliberately does NOT query waterway=river/canal/stream
#' or natural=wetland -- same still-water-only reasoning as
#' NHD_STILL_WATER_FTYPES; a river/marsh forced into a closed polygon
#' isn't a flat surface, and flattening it produces large spurious
#' corrections.
fetch_water_bodies_osm <- function(bounds, water_types = c("water", "bay")) {
  south <- bounds[1]; west <- bounds[2]; north <- bounds[3]; east <- bounds[4]
  bbox <- sprintf("%s,%s,%s,%s", south, west, north, east)
  filters <- paste(water_types, collapse = "|")

  query <- sprintf('
    [out:json][timeout:60];
    (
      way["natural"~"%s"](%s);
      relation["natural"~"%s"](%s);
      way["landuse"="reservoir"](%s);
      relation["landuse"="reservoir"](%s);
    );
    out geom;
  ', filters, bbox, filters, bbox, bbox, bbox)

  cat(sprintf("[OSM] Querying water bodies in bbox %s...\n", bbox))
  resp <- request("https://overpass-api.de/api/interpreter") |>
    req_body_form(data = query) |>
    req_timeout(90) |>
    req_perform()
  data <- resp_body_json(resp)

  polygons <- list()
  for (el in data$elements) {
    geom <- el$geometry
    if (is.null(geom) || length(geom) == 0) next
    coords <- lapply(geom, function(node) c(node$lon, node$lat))
    if (length(coords) < 3) next
    if (!identical(coords[[1]], coords[[length(coords)]])) {
      coords[[length(coords) + 1]] <- coords[[1]]
    }
    tags <- el$tags
    if (is.null(tags)) tags <- list()
    polygons[[length(polygons) + 1]] <- list(
      type = "Polygon", coordinates = list(coords),
      properties = list(source = "osm",
                         natural = if (is.null(tags$natural)) "" else tags$natural,
                         landuse = if (is.null(tags$landuse)) "" else tags$landuse,
                         name = if (is.null(tags$name)) "" else tags$name)
    )
  }
  cat(sprintf("[OSM] Found %d water body polygons\n", length(polygons)))
  polygons
}

#' Fetch water body polygons from USGS NHD via the ArcGIS REST API.
#' Layer 12 = Waterbody Large Scale -- excludes rivers by construction
#' (they live in NHD's separate Flowline layer, as lines, which this
#' function never queries). still_water_only additionally filters out
#' non-flat waterbody FTypes (marsh, ice, playa) -- see
#' NHD_STILL_WATER_FTYPES.
fetch_water_bodies_nhd <- function(bounds, still_water_only = TRUE) {
  south <- bounds[1]; west <- bounds[2]; north <- bounds[3]; east <- bounds[4]
  url <- "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/12/query"

  cat(sprintf("[NHD] Querying waterbodies in bbox %.2f,%.2f,%.2f,%.2f...\n",
              west, south, east, north))

  all_polygons <- list()
  n_filtered_ftype <- 0
  offset <- 0
  repeat {
    resp <- request(url) |>
      req_url_query(
        geometry = sprintf("%s,%s,%s,%s", west, south, east, north),
        geometryType = "esriGeometryEnvelope", inSR = "4326",
        spatialRel = "esriSpatialRelIntersects",
        outFields = "GNIS_Name,FType,FCode",
        returnGeometry = "true", outSR = "4326", f = "geojson",
        resultRecordCount = 2000, resultOffset = offset
      ) |>
      req_timeout(60) |>
      req_perform()
    data <- resp_body_json(resp)
    features <- data$features
    if (is.null(features) || length(features) == 0) break

    for (feat in features) {
      geom <- feat$geometry
      if (is.null(geom)) next
      # The ArcGIS service returns field names in its own casing
      # (observed: GNIS_NAME/FTYPE/FCODE, all caps) regardless of how
      # outFields was spelled in the request. Look up case-insensitively.
      props <- feat$properties
      if (is.null(props)) props <- list()
      names(props) <- tolower(names(props))
      ftype <- props$ftype

      if (still_water_only && !(!is.null(ftype) && ftype %in% NHD_STILL_WATER_FTYPES)) {
        n_filtered_ftype <- n_filtered_ftype + 1
        next
      }

      prop_dict <- list(source = "nhd",
                         name = if (is.null(props$gnis_name)) "" else props$gnis_name,
                         ftype = ftype)

      if (identical(geom$type, "MultiPolygon")) {
        for (polygon_coords in geom$coordinates) {
          all_polygons[[length(all_polygons) + 1]] <- list(
            type = "Polygon", coordinates = polygon_coords, properties = prop_dict
          )
        }
      } else if (identical(geom$type, "Polygon")) {
        all_polygons[[length(all_polygons) + 1]] <- list(
          type = "Polygon", coordinates = geom$coordinates, properties = prop_dict
        )
      }
    }
    cat(sprintf("[NHD]   ... %d polygons so far (offset %d)\n", length(all_polygons), offset))
    if (length(features) < 2000) break
    offset <- offset + 2000
  }

  cat(sprintf("[NHD] Found %d water body polygons%s\n", length(all_polygons),
              if (still_water_only) sprintf(" (%d non-still-water filtered out)", n_filtered_ftype) else ""))
  all_polygons
}

#' Fetch ocean/sea fill polygons from OSM coastline data (natural=coastline
#' ways) within bounds, via sf's polygonize/union/difference -- direct
#' equivalents of the Python original's shapely calls of the same names.
#' Falls back to the whole bbox as one ocean polygon when sf isn't
#' installed, no coastline is found (open ocean, no nearby land), or
#' construction otherwise fails -- the first case is an R-port-specific
#' addition (see module docstring's dependency note), the other two
#' match the Python original's own fallbacks.
fetch_ocean_polygons_osm <- function(bounds, simplify_tolerance_deg = 0.001) {
  south <- bounds[1]; west <- bounds[2]; north <- bounds[3]; east <- bounds[4]
  bbox_str <- sprintf("%s,%s,%s,%s", south, west, north, east)

  whole_bbox_fallback <- function() {
    cat("[OCEAN] Treating entire bbox as ocean\n")
    ring <- list(c(west, south), c(east, south), c(east, north), c(west, north), c(west, south))
    list(list(type = "Polygon", coordinates = list(ring),
              properties = list(source = "osm_ocean", name = "ocean")))
  }

  if (!requireNamespace("sf", quietly = TRUE)) {
    cat("[OCEAN] sf not installed -- skipping coastline polygonize, using whole-bbox fallback\n")
    return(whole_bbox_fallback())
  }

  cat(sprintf("[OCEAN] Fetching OSM coastline for bbox %s...\n", bbox_str))
  query <- sprintf('[out:json][timeout:120];(way["natural"="coastline"](%s););out geom;', bbox_str)
  data <- tryCatch({
    resp <- request("https://overpass-api.de/api/interpreter") |>
      req_body_form(data = query) |>
      req_headers(`User-Agent` = "dem-water-correction-r/1.0") |>
      req_timeout(120) |>
      req_perform()
    resp_body_json(resp)
  }, error = function(e) {
    cat(sprintf("[OCEAN] Coastline fetch failed: %s\n", conditionMessage(e)))
    NULL
  })
  if (is.null(data)) return(list())

  elements <- data$elements
  cat(sprintf("[OCEAN] Got %d coastline segments\n", length(elements)))
  if (length(elements) == 0) return(whole_bbox_fallback())

  bbox_poly <- sf::st_as_sfc(sf::st_bbox(c(xmin = west, ymin = south, xmax = east, ymax = north),
                                          crs = sf::st_crs(4326)))
  as_ocean_poly <- function(sfg) {
    coords <- sf::st_coordinates(sfg)[, c("X", "Y")]
    list(type = "Polygon",
         coordinates = list(lapply(seq_len(nrow(coords)), function(i) c(coords[i, 1], coords[i, 2]))),
         properties = list(source = "osm_ocean", name = "ocean"))
  }

  lines <- Filter(Negate(is.null), lapply(elements, function(el) {
    geom <- el$geometry
    if (is.null(geom) || length(geom) < 2) return(NULL)
    coords <- do.call(rbind, lapply(geom, function(n) c(n$lon, n$lat)))
    sf::st_linestring(coords)
  }))
  if (length(lines) == 0) {
    cat("[OCEAN] No usable coastline geometry\n")
    return(list())
  }

  clipped_lines <- Filter(function(g) !is.null(g) && !sf::st_is_empty(g),
                           lapply(lines, function(l) suppressWarnings(sf::st_intersection(sf::st_sfc(l, crs = 4326), bbox_poly))))
  if (length(clipped_lines) == 0) {
    cat("[OCEAN] No coastline intersects bbox\n")
    return(list())
  }

  result <- tryCatch({
    clipped_sfc <- sf::st_sfc(unlist(lapply(clipped_lines, function(x) sf::st_geometry(x)), recursive = FALSE), crs = 4326)
    merged_coastline <- sf::st_line_merge(sf::st_union(clipped_sfc))

    bbox_coords <- rbind(c(west, south), c(east, south), c(east, north), c(west, north), c(west, south))
    bbox_lines <- sf::st_sfc(lapply(seq_len(nrow(bbox_coords) - 1), function(i) {
      sf::st_linestring(bbox_coords[i:(i + 1), ])
    }), crs = 4326)

    all_lines <- c(clipped_sfc, bbox_lines)
    all_merged <- sf::st_union(all_lines)
    all_polys <- sf::st_collection_extract(sf::st_polygonize(all_merged), "POLYGON")
    cat(sprintf("[OCEAN] Polygonized into %d polygons\n", length(all_polys)))
    if (length(all_polys) == 0) return(whole_bbox_fallback())

    coast_centroid_lng <- sf::st_coordinates(sf::st_centroid(sf::st_union(clipped_sfc)))[1, "X"]
    centroids <- sf::st_coordinates(sf::st_centroid(all_polys))
    bboxes <- lapply(seq_along(all_polys), function(i) sf::st_bbox(all_polys[i]))

    is_ocean <- vapply(seq_along(all_polys), function(i) {
      touches_west <- abs(bboxes[[i]]["xmin"] - west) < 0.01
      touches_south <- abs(bboxes[[i]]["ymin"] - south) < 0.01
      west_of_coast <- centroids[i, "X"] < coast_centroid_lng
      touches_west || touches_south || west_of_coast
    }, logical(1))

    ocean_polys <- all_polys[is_ocean]
    if (length(ocean_polys) == 0) {
      land_union <- if (sum(!is_ocean) > 0) sf::st_union(all_polys[!is_ocean]) else bbox_poly
      ocean_polys <- sf::st_sfc(sf::st_difference(bbox_poly, land_union), crs = 4326)
    }

    ocean_union <- sf::st_union(ocean_polys)
    if (simplify_tolerance_deg > 0) {
      ocean_union <- sf::st_simplify(ocean_union, dTolerance = simplify_tolerance_deg,
                                      preserveTopology = TRUE)
    }
    parts <- sf::st_cast(sf::st_sfc(ocean_union, crs = 4326), "POLYGON", warn = FALSE)
    parts <- Filter(function(g) !sf::st_is_empty(g) && as.numeric(sf::st_area(g)) > 1e-8, parts)

    lapply(parts, as_ocean_poly)
  }, error = function(e) {
    cat(sprintf("[OCEAN] Ocean polygon construction failed: %s\n", conditionMessage(e)))
    list()
  })

  cat(sprintf("[OCEAN] Constructed %d ocean polygon(s)\n", length(result)))
  result
}

#' Load water body polygons from a local GeoJSON FeatureCollection.
load_water_bodies_from_file <- function(path) {
  data <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (identical(data$type, "FeatureCollection")) {
    Filter(Negate(is.null), lapply(data$features, function(f) f$geometry))
  } else if (identical(data$type, "Polygon")) {
    list(data)
  } else {
    stop(sprintf("Unsupported GeoJSON type: %s", data$type))
  }
}

# ------------------------------------------------
# Batch geometry helpers
# ------------------------------------------------

#' Filter water_polygons to those overlapping bounds (a terra ext()-like
#' c(xmin, xmax, ymin, ymax), or an sf/terra bbox). Returns a list of
#' list(orig_idx=, poly=) pairs, orig_idx 0-based to match the Python
#' original (matters for cross-referencing against water.geojson, whose
#' features are written in the caller's original 0-based order).
filter_overlapping <- function(water_polygons, bounds_left, bounds_bottom,
                                bounds_right, bounds_top) {
  result <- list()
  for (i in seq_along(water_polygons)) {
    poly <- water_polygons[[i]]
    coords <- poly$coordinates
    if (is.null(coords) || length(coords) == 0) next
    geom_type <- if (is.null(poly$type)) "Polygon" else poly$type
    all_coords <- if (identical(geom_type, "MultiPolygon")) {
      unlist(unlist(coords, recursive = FALSE), recursive = FALSE)
    } else {
      coords[[1]]
    }
    lngs <- vapply(all_coords, function(c) c[[1]], numeric(1))
    lats <- vapply(all_coords, function(c) c[[2]], numeric(1))
    if (max(lngs) < bounds_left || min(lngs) > bounds_right ||
        max(lats) < bounds_bottom || min(lats) > bounds_top) next
    result[[length(result) + 1]] <- list(orig_idx = i - 1, poly = poly)
  }
  result
}

#' Sort (orig_idx, poly) pairs along a Morton (Z-order) curve, so any
#' consecutive slice (a batch) is geographically compact in both lat
#' and lng -- see dem_water_correction.py's _spatial_sort_morton
#' docstring for why a plain (lat, lng) sort doesn't achieve this.
spatial_sort_morton <- function(overlapping) {
  if (length(overlapping) == 0) return(overlapping)

  morton_code <- function(x, y) {
    interleave <- function(v) {
      v <- bitwAnd(v, 0xFFFF)
      v <- bitwAnd(bitwOr(v, bitwShiftL(v, 8)), 0x00FF00FF)
      v <- bitwAnd(bitwOr(v, bitwShiftL(v, 4)), 0x0F0F0F0F)
      v <- bitwAnd(bitwOr(v, bitwShiftL(v, 2)), 0x33333333)
      v <- bitwAnd(bitwOr(v, bitwShiftL(v, 1)), 0x55555555)
      v
    }
    bitwOr(interleave(x), bitwShiftL(interleave(y), 1))
  }

  entries <- lapply(overlapping, function(item) {
    ring <- item$poly$coordinates[[1]]
    if (length(ring) == 0) return(list(item = item, lat = 0, lng = 0))
    lats <- vapply(ring, function(c) c[[2]], numeric(1))
    lngs <- vapply(ring, function(c) c[[1]], numeric(1))
    list(item = item, lat = mean(lats), lng = mean(lngs))
  })

  lat_vals <- vapply(entries, function(e) e$lat, numeric(1))
  lng_vals <- vapply(entries, function(e) e$lng, numeric(1))
  lat_min <- min(lat_vals); lat_span <- max(max(lat_vals) - lat_min, 1e-9)
  lng_min <- min(lng_vals); lng_span <- max(max(lng_vals) - lng_min, 1e-9)

  xq <- pmin(65535, pmax(0, as.integer((lat_vals - lat_min) / lat_span * 65535)))
  yq <- pmin(65535, pmax(0, as.integer((lng_vals - lng_min) / lng_span * 65535)))
  codes <- mapply(morton_code, xq, yq)

  ord <- order(codes, lat_vals, lng_vals)
  lapply(entries[ord], function(e) e$item)
}

#' A GeoJSON-style coordinate ring (list of c(lon, lat) pairs) as a WKT
#' POLYGON string, for building a terra SpatVector directly -- no sf
#' needed (see module docstring's dependency note).
ring_to_wkt_polygon <- function(ring) {
  pts <- vapply(ring, function(c) sprintf("%.10g %.10g", c[[1]], c[[2]]), character(1))
  sprintf("POLYGON((%s))", paste(pts, collapse = ","))
}

#' Rasterize a batch of polygons into a single int label raster (a plain
#' matrix, 1 = no water, label = batch position + 1). transform is a
#' list(a=,b=,c=,d=,e=,f=) GDAL-style affine (see mask_centroid_latlon).
rasterize_batch <- function(batch, crop_h, crop_w, transform) {
  if (length(batch) == 0) return(matrix(0L, crop_h, crop_w))

  template <- rast(nrows = crop_h, ncols = crop_w,
                    xmin = transform$c, xmax = transform$c + crop_w * transform$a,
                    ymax = transform$f, ymin = transform$f + crop_h * transform$e,
                    crs = "EPSG:4326")

  wkts <- character(0); labels <- integer(0)
  for (pos in seq_along(batch)) {
    poly <- batch[[pos]]$poly
    coords <- poly$coordinates
    if (is.null(coords) || length(coords) == 0) next
    wkts <- c(wkts, ring_to_wkt_polygon(coords[[1]]))
    labels <- c(labels, pos)
  }
  if (length(wkts) == 0) return(matrix(0L, crop_h, crop_w))

  v <- vect(wkts, crs = "EPSG:4326")
  v$label <- labels
  label_rast <- terra::rasterize(v, template, field = "label", background = 0)
  matrix(as.integer(values(label_rast)), nrow = crop_h, ncol = crop_w, byrow = TRUE)
}

# ------------------------------------------------
# Per-polygon correction (CPU path)
# ------------------------------------------------

#' Rough midpoint for locating a water body: the nearest actual water
#' pixel to the mean pixel position of its rasterized mask -- not the
#' source polygon's vertices (a river/marsh polygon's vertex mean can
#' land nowhere near the actual shape), and not the raw pixel mean
#' either (which can still fall in a gap for a non-convex/zigzag
#' shape). See dem_water_correction.py's identically-named helper for
#' the full incident writeup this fix came from.
#'
#' @param mask boolean matrix, water pixels TRUE (bbox-local or full-raster).
#' @param row_off,col_off 0-based pixel offset of mask[1,1] within the
#'   full raster transform is defined on (0 if mask is already full-size).
#' @param transform list(a=,b=,c=,d=,e=,f=) GDAL-style affine, pixel
#'   CENTER convention (matching rasterio.transform.xy()'s default
#'   offset='center', which the Python original relies on implicitly):
#'   lon = c + (col+0.5)*a + (row+0.5)*b; lat = f + (col+0.5)*d + (row+0.5)*e
#' @return list(lat=, lon=), or list(lat=NA, lon=NA) if mask is empty.
mask_centroid_latlon <- function(mask, row_off, col_off, transform) {
  idx <- which(mask, arr.ind = TRUE)
  if (nrow(idx) == 0) return(list(lat = NA, lon = NA))
  rows0 <- idx[, "row"] - 1
  cols0 <- idx[, "col"] - 1
  mean_row <- mean(rows0); mean_col <- mean(cols0)
  nearest <- which.min((rows0 - mean_row) ^ 2 + (cols0 - mean_col) ^ 2)
  # +0.5: pixel CENTER, not corner -- confirmed the hard way (a zigzag
  # polygon test's midpoint landed just outside the water, off by
  # half a pixel, matching this exact corner-vs-center gap) that
  # skipping this offset can push the reported point outside a thin
  # water body even though the nearest-pixel selection above is correct.
  abs_row <- row_off + rows0[nearest] + 0.5
  abs_col <- col_off + cols0[nearest] + 0.5
  list(
    lon = transform$c + abs_col * transform$a + abs_row * transform$b,
    lat = transform$f + abs_col * transform$d + abs_row * transform$e
  )
}

#' Per-label bounding boxes for a label matrix -- R equivalent of
#' scipy.ndimage.find_objects, via one vectorized which() + split()
#' rather than an O(labels x h x w) per-label scan.
#' @return named list, keyed by label (as character), each
#'   list(r_min=, r_max=, c_min=, c_max=) 1-indexed and inclusive.
find_label_bboxes <- function(label_array) {
  idx <- which(label_array > 0, arr.ind = TRUE)
  if (nrow(idx) == 0) return(list())
  lbl <- label_array[idx]
  row_by <- split(idx[, "row"], lbl)
  col_by <- split(idx[, "col"], lbl)
  Map(function(rows, cols) list(r_min = min(rows), r_max = max(rows),
                                 c_min = min(cols), c_max = max(cols)),
      row_by, col_by)
}

#' Core per-batch correction: for each labeled water body, sample
#' shoreline elevation (a `buffer_pixels`-wide band just outside the
#' water, via a max-filter/"grey dilation" of the label array) at the
#' given percentile, and flood-fill the water pixels to that elevation.
#'
#' @param label_array,corrected  h x w matrices (int labels; float
#'   elevation, modified and returned).
#' @param batch list of list(orig_idx=, poly=), label i's polygon is
#'   batch[[i]] (1-based label = batch position).
#' @param transform GDAL-style affine list (see mask_centroid_latlon);
#'   NULL means midpoint lat/lon are left NA.
#' @return list(corrected=, elevation_changes=, corrected_count=, skipped_small=)
process_label_array <- function(label_array, corrected, batch,
                                 buffer_pixels, shore_percentile, min_water_pixels,
                                 transform = NULL) {
  elevation_changes <- list()
  corrected_count <- 0L
  skipped_small <- 0L

  h <- nrow(label_array); w <- ncol(label_array)
  water_any <- label_array > 0

  # Shoreline band: dilate the label raster by buffer_pixels (a
  # square max-filter propagates each label outward by exactly its
  # radius -- the "grey dilation" trick dem_water_correction.py's own
  # comment documents), then the shoreline is (dilated > 0) & !water_any.
  win <- 2 * buffer_pixels + 1
  label_rast <- rast(label_array, extent = ext(0, w, 0, h))
  dilated_rast <- focal(label_rast, w = matrix(1, win, win), fun = "max",
                         na.policy = "omit", na.rm = TRUE, fillvalue = 0)
  dilated <- matrix(as.integer(values(dilated_rast)), nrow = h, ncol = w, byrow = TRUE)

  in_shoreline <- (dilated > 0) & !water_any
  shore_elevs <- corrected[in_shoreline]
  shore_labels <- dilated[in_shoreline]

  bboxes <- find_label_bboxes(label_array)

  for (batch_pos in seq_along(batch)) {
    label <- batch_pos
    bbox <- bboxes[[as.character(label)]]
    if (is.null(bbox)) next

    r_range <- bbox$r_min:bbox$r_max
    c_range <- bbox$c_min:bbox$c_max
    label_sub <- label_array[r_range, c_range, drop = FALSE]
    water_mask_local <- label_sub == label

    pixel_count <- sum(water_mask_local)
    if (pixel_count < min_water_pixels) {
      skipped_small <- skipped_small + 1L
      next
    }

    label_shore_mask <- shore_labels == label
    valid <- shore_elevs[label_shore_mask]
    valid <- valid[valid > -1000 & valid < 9000]

    if (length(valid) == 0) {
      inside <- corrected[r_range, c_range, drop = FALSE][water_mask_local]
      valid <- inside[inside > -1000]
      if (length(valid) == 0) next
    }

    surface_elev <- as.numeric(quantile(valid, probs = shore_percentile / 100,
                                         names = FALSE, type = 7))

    orig_vals <- corrected[r_range, c_range, drop = FALSE][water_mask_local]
    valid_orig <- orig_vals[orig_vals > -1000]
    orig_mean <- if (length(valid_orig) > 0) mean(valid_orig) else 0.0

    corrected_block <- corrected[r_range, c_range, drop = FALSE]
    corrected_block[water_mask_local] <- surface_elev
    corrected[r_range, c_range] <- corrected_block
    corrected_count <- corrected_count + 1L

    poly <- batch[[batch_pos]]$poly
    name <- if (is.null(poly$properties$name)) "" else poly$properties$name
    mid <- mask_centroid_latlon(water_mask_local, bbox$r_min - 1, bbox$c_min - 1, transform)

    elevation_changes[[length(elevation_changes) + 1]] <- list(
      polygon_index = batch[[batch_pos]]$orig_idx, name = name,
      pixel_count = pixel_count,
      surface_elev_m = round(surface_elev, 1),
      original_mean_m = round(orig_mean, 1),
      change_m = round(surface_elev - orig_mean, 1),
      lat = if (is.na(mid$lat)) NULL else round(mid$lat, 5),
      lon = if (is.na(mid$lon)) NULL else round(mid$lon, 5)
    )
  }

  list(corrected = corrected, elevation_changes = elevation_changes,
       corrected_count = corrected_count, skipped_small = skipped_small)
}

# ------------------------------------------------
# Core correction function -- batch rasterization
# ------------------------------------------------

#' Apply water body elevation correction to a DEM GeoTIFF.
#'
#' Uses batch rasterization -- rasterizes up to `batch_size` polygons at
#' once into a single label array, then processes all labels in one
#' pass, rather than one rasterize call per polygon.
#'
#' @return stats list: total_polygons, corrected_bodies, skipped_small,
#'   skipped_no_overlap, total_pixels_fixed, elevation_changes, n_batches.
correct_dem_water_bodies <- function(input_path, output_path, water_polygons,
                                      buffer_pixels = 3, shore_percentile = 10.0,
                                      min_water_pixels = 5, batch_size = 500,
                                      verbose = TRUE) {
  stats <- list(total_polygons = length(water_polygons), corrected_bodies = 0L,
                skipped_small = 0L, skipped_no_overlap = 0L,
                total_pixels_fixed = 0L, elevation_changes = list(),
                batch_size = batch_size, n_batches = 0L)

  r <- rast(input_path)
  elevation <- matrix(as.numeric(values(r)), nrow = nrow(r), ncol = ncol(r), byrow = TRUE)
  # unname(): ext()'s $xmin/$xmax/... accessors return NAMED numerics
  # (name = "xmin" etc.), and R arithmetic propagates that name onto
  # every value derived from it -- left unfixed, lat/lon in
  # elevation_changes below would carry a stray "ymax"/"xmin" name that
  # corrupts jsonlite's serialization of water_stats.json.
  e <- unname(as.vector(ext(r)))
  names(e) <- c("xmin", "xmax", "ymin", "ymax")
  h <- nrow(r); w <- ncol(r)
  transform <- list(a = res(r)[1], b = 0, c = e[["xmin"]], d = 0, e = -res(r)[2], f = e[["ymax"]])
  bounds <- list(left = e[["xmin"]], bottom = e[["ymin"]], right = e[["xmax"]], top = e[["ymax"]])

  corrected <- elevation

  cat("[CORRECT] Using CPU (GPU path out of scope for this port)\n")

  overlapping <- filter_overlapping(water_polygons, bounds$left, bounds$bottom,
                                     bounds$right, bounds$top)
  stats$skipped_no_overlap <- length(water_polygons) - length(overlapping)
  cat(sprintf("[CORRECT] %d polygons overlap raster (%d outside bounds skipped)\n",
              length(overlapping), stats$skipped_no_overlap))

  ocean_polys <- Filter(function(p) identical(p$properties$source, "osm_ocean"), water_polygons)
  regular_polys <- Filter(function(p) !identical(p$properties$source, "osm_ocean"), water_polygons)

  if (length(ocean_polys) > 0) {
    cat(sprintf("[CORRECT] Applying fast zero-fill to %d ocean polygon(s)...\n", length(ocean_polys)))
    template <- rast(nrows = h, ncols = w, extent = e, crs = "EPSG:4326")
    for (ocean_poly in ocean_polys) {
      v <- vect(ring_to_wkt_polygon(ocean_poly$coordinates[[1]]), crs = "EPSG:4326")
      mask_r <- terra::rasterize(v, template, field = 1, background = 0)
      ocean_mask <- matrix(as.logical(values(mask_r)), nrow = h, ncol = w, byrow = TRUE)
      n_ocean <- sum(ocean_mask)
      cat(sprintf("[CORRECT] Zeroing %s ocean pixels\n", format(n_ocean, big.mark = ",")))
      corrected[ocean_mask] <- 0.0
      stats$corrected_bodies <- stats$corrected_bodies + 1L
      stats$total_pixels_fixed <- stats$total_pixels_fixed + n_ocean
    }
  }

  water_polygons <- regular_polys
  overlapping <- filter_overlapping(water_polygons, bounds$left, bounds$bottom,
                                     bounds$right, bounds$top)
  overlapping <- spatial_sort_morton(overlapping)
  cat(sprintf("[CORRECT] %d polygons sorted spatially (Morton/Z-order) for compact batching\n",
              length(overlapping)))

  # Pigeonhole guard: too few batches for a dense, uniformly-spread
  # polygon set can't achieve compact crops even after sorting -- see
  # dem_water_correction.py's comment for the empirical numbers this
  # threshold is based on.
  MIN_BATCHES_FOR_LOCALITY <- 8
  if (length(overlapping) > 0 && length(overlapping) / batch_size < MIN_BATCHES_FOR_LOCALITY) {
    batch_size <- max(50, ceiling(length(overlapping) / MIN_BATCHES_FOR_LOCALITY))
  }

  n_batches <- if (length(overlapping) == 0) 0 else ceiling(length(overlapping) / batch_size)
  cat(sprintf("[CORRECT] Processing in batches of %d (%d batches)...\n", batch_size, n_batches))

  for (batch_num in seq_len(n_batches)) {
    batch_start <- (batch_num - 1) * batch_size + 1
    batch_end <- min(batch_start + batch_size - 1, length(overlapping))
    batch <- overlapping[batch_start:batch_end]

    t_batch <- Sys.time()
    if (verbose) {
      cat(sprintf("[CORRECT] Batch %d/%d (polygons %d-%d, %.0f%%) ... ",
                   batch_num, n_batches, batch_start - 1, batch_end - 1,
                   batch_num / n_batches * 100))
    }

    pad <- buffer_pixels + 2
    all_lngs <- c(); all_lats <- c()
    for (item in batch) {
      coords <- item$poly$coordinates
      if (is.null(coords) || length(coords) == 0) next
      ring <- coords[[1]]
      all_lngs <- c(all_lngs, vapply(ring, function(c) c[[1]], numeric(1)))
      all_lats <- c(all_lats, vapply(ring, function(c) c[[2]], numeric(1)))
    }
    if (length(all_lngs) == 0) {
      stats$skipped_no_overlap <- stats$skipped_no_overlap + length(batch)
      if (verbose) cat(sprintf("0 corrected, %d no coords\n", length(batch)))
      next
    }

    c_min <- max(0, as.integer((min(all_lngs) - transform$c) / transform$a) - pad)
    c_max <- min(w, as.integer((max(all_lngs) - transform$c) / transform$a) + pad + 1)
    r_min <- max(0, as.integer((max(all_lats) - transform$f) / transform$e) - pad)
    r_max <- min(h, as.integer((min(all_lats) - transform$f) / transform$e) + pad + 1)

    crop_h <- r_max - r_min
    crop_w <- c_max - c_min
    if (crop_h <= 0 || crop_w <= 0) {
      stats$skipped_no_overlap <- stats$skipped_no_overlap + length(batch)
      if (verbose) cat(sprintf("0 corrected, %d zero-size crop\n", length(batch)))
      next
    }

    crop_transform <- list(
      a = transform$a, b = 0, c = transform$c + c_min * transform$a,
      d = 0, e = transform$e, f = transform$f + r_min * transform$e
    )

    label_crop <- tryCatch(rasterize_batch(batch, crop_h, crop_w, crop_transform),
                            error = function(e) { cat(sprintf("\n[CORRECT] Batch %d rasterize failed: %s\n",
                                                               batch_num, conditionMessage(e))); NULL })
    if (is.null(label_crop)) next
    if (max(label_crop) == 0) {
      stats$skipped_no_overlap <- stats$skipped_no_overlap + length(batch)
      if (verbose) cat(sprintf("0 corrected, %d no pixels\n", length(batch)))
      next
    }

    corrected_crop <- corrected[(r_min + 1):r_max, (c_min + 1):c_max, drop = FALSE]
    step_result <- process_label_array(label_crop, corrected_crop, batch,
                                        buffer_pixels, shore_percentile, min_water_pixels,
                                        transform = crop_transform)

    corrected[(r_min + 1):r_max, (c_min + 1):c_max] <- step_result$corrected

    elapsed_batch <- as.numeric(Sys.time() - t_batch, units = "secs")
    stats$corrected_bodies <- stats$corrected_bodies + step_result$corrected_count
    stats$skipped_small <- stats$skipped_small + step_result$skipped_small
    stats$total_pixels_fixed <- stats$total_pixels_fixed +
      sum(vapply(step_result$elevation_changes, function(c) c$pixel_count, numeric(1)))
    stats$elevation_changes <- c(stats$elevation_changes, step_result$elevation_changes)
    stats$n_batches <- stats$n_batches + 1L

    if (verbose) {
      cat(sprintf("%d corrected, %d too small (%.1fs)\n",
                   step_result$corrected_count, step_result$skipped_small, elapsed_batch))
    }
  }

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  out_r <- rast(corrected, extent = e, crs = "EPSG:4326")
  # BIGTIFF=YES explicit, not GDAL's BIGTIFF=IF_NEEDED default -- see
  # build_dem_mosaic.R's identical comment for the live failure this avoids.
  writeRaster(out_r, output_path, filetype = "GTiff", datatype = "FLT4S",
              gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=YES"),
              overwrite = TRUE)

  cat("\n[CORRECT] Summary (CPU):\n")
  cat(sprintf("  Batches processed:    %d (batch size %d)\n", stats$n_batches, batch_size))
  cat(sprintf("  Corrected bodies:     %d\n", stats$corrected_bodies))
  cat(sprintf("  Skipped (too small):  %d\n", stats$skipped_small))
  cat(sprintf("  Skipped (no overlap): %d\n", stats$skipped_no_overlap))
  cat(sprintf("  Total pixels fixed:   %s\n", format(stats$total_pixels_fixed, big.mark = ",")))
  cat(sprintf("  Output: %s\n", output_path))

  stats
}
