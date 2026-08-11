#' Unit tests for tile_id.R. Direct translation of test_tile_id.py --
#' same cases, same expected values, run with `Rscript test_tile_id.R`.

source("tile_id.R")

assert_that <- function(cond, msg = "assertion failed") {
  if (!isTRUE(cond)) stop(msg, call. = FALSE)
}

expect_error <- function(expr, msg) {
  ok <- tryCatch({ force(expr); FALSE }, error = function(e) TRUE)
  assert_that(ok, msg)
}

test_tile_id_from_latlng <- function() {
  assert_that(tile_id_from_latlng(45.5, -110.5) == "N45W111")
  assert_that(tile_id_from_latlng(0.5, 0.5) == "N00E000")
  assert_that(tile_id_from_latlng(-12.7, 178.3) == "S13E178")
  # Edge of a tile (point exactly at SW corner -- belongs to that tile)
  assert_that(tile_id_from_latlng(45.0, -110.0) == "N45W110")
  assert_that(tile_id_from_latlng(45.5, -109.001) == "N45W110")
  assert_that(tile_id_from_latlng(45.999, -109.5) == "N45W110")
  # Just past northern edge -> belongs to N46
  assert_that(tile_id_from_latlng(46.001, -109.5) == "N46W110")
  cat("  test_tile_id_from_latlng OK\n")
}

test_parse_tile_id <- function() {
  cases <- list(
    list("N45W110", 45, -110), list("N00E000", 0, 0),
    list("S13E178", -13, 178), list("N89E179", 89, 179),
    list("S89W180", -89, -180)
  )
  for (c in cases) {
    got <- parse_tile_id(c[[1]])
    assert_that(got$lat_floor == c[[2]] && got$lng_floor == c[[3]],
                sprintf("%s: got (%d,%d), expected (%d,%d)",
                        c[[1]], got$lat_floor, got$lng_floor, c[[2]], c[[3]]))
  }
  cat("  test_parse_tile_id OK\n")
}

test_round_trip <- function() {
  cases <- list(c(45, -110), c(0, 0), c(-13, 178), c(89, 179),
                c(-89, -180), c(45, -111), c(-1, -1), c(1, 1))
  for (c in cases) {
    lat <- c[1]; lng <- c[2]
    tid <- tile_id_from_floors(lat, lng)
    parsed <- parse_tile_id(tid)
    assert_that(parsed$lat_floor == lat && parsed$lng_floor == lng,
                sprintf("Round trip failed: (%d,%d) -> %s -> (%d,%d)",
                        lat, lng, tid, parsed$lat_floor, parsed$lng_floor))
    tid2 <- tile_id_from_latlng(lat + 0.5, lng + 0.5)
    assert_that(tid2 == tid, sprintf("From latlng round trip: %s != %s", tid, tid2))
  }
  cat("  test_round_trip OK\n")
}

test_parse_invalid <- function() {
  bad <- c("", "N45W110X", "X45W110", "N99W500", "N5W110", "n45w110", "N45-W110")
  for (tid in bad) {
    expect_error(parse_tile_id(tid), sprintf("Should have raised for '%s'", tid))
  }
  cat("  test_parse_invalid OK\n")
}

test_tile_bounds <- function() {
  b <- tile_bounds("N45W110")
  assert_that(b$south == 45.0)
  assert_that(b$west == -110.0)
  assert_that(b$north == 46.0)
  assert_that(b$east == -109.0)
  center <- tile_bounds_center(b)
  assert_that(center[["lat"]] == 45.5 && center[["lng"]] == -109.5)
  assert_that(tile_bounds_contains(b, 45.5, -109.5))
  assert_that(tile_bounds_contains(b, 45.0, -110.0))       # SW corner inclusive
  assert_that(!tile_bounds_contains(b, 46.0, -109.5))      # N edge exclusive
  assert_that(!tile_bounds_contains(b, 45.5, -109.0))      # E edge exclusive
  assert_that(all(tile_bounds_as_tuple(b) == c(45.0, -110.0, 46.0, -109.0)))

  # Southern hemisphere
  b <- tile_bounds("S13E178")
  assert_that(b$south == -13.0)
  assert_that(b$west == 178.0)
  assert_that(b$north == -12.0)
  assert_that(b$east == 179.0)
  assert_that(tile_bounds_contains(b, -12.5, 178.5))
  cat("  test_tile_bounds OK\n")
}

test_tile_bounds_intersects_bbox <- function() {
  b <- tile_bounds("N45W110")  # covers (45.0, -110.0, 46.0, -109.0)
  assert_that(tile_bounds_intersects_bbox(b, 45.1, -109.9, 45.9, -109.1))
  assert_that(!tile_bounds_intersects_bbox(b, 50.0, -100.0, 51.0, -99.0))
  # Bbox touching east edge -- exclusive, so should NOT intersect
  assert_that(!tile_bounds_intersects_bbox(b, 45.5, -109.0, 45.6, -108.9))
  assert_that(tile_bounds_intersects_bbox(b, 45.5, -109.5, 45.6, -108.5))
  cat("  test_tile_bounds_intersects_bbox OK\n")
}

test_tile_ids_covering_bbox_small <- function() {
  result <- tile_ids_covering_bbox(45.2, -110.7, 45.9, -109.3)
  assert_that(identical(result, c("N45W111", "N45W110")),
              paste(result, collapse = ","))
  cat("  test_tile_ids_covering_bbox_small OK\n")
}

test_tile_ids_covering_bbox_3x3 <- function() {
  result <- tile_ids_covering_bbox(44.5, -111.5, 46.5, -108.5)
  expected <- c(
    "N44W112", "N44W111", "N44W110", "N44W109",
    "N45W112", "N45W111", "N45W110", "N45W109",
    "N46W112", "N46W111", "N46W110", "N46W109"
  )
  assert_that(identical(result, expected),
              sprintf("Got: %s\nExpected: %s",
                      paste(result, collapse = ","), paste(expected, collapse = ",")))
  cat("  test_tile_ids_covering_bbox_3x3 OK\n")
}

test_tile_ids_covering_bbox_boundary <- function() {
  # north edge exactly at 46.0 -- should NOT include tile starting at 46
  result <- tile_ids_covering_bbox(45.0, -110.0, 46.0, -109.0)
  assert_that(identical(result, c("N45W110")), paste(result, collapse = ","))

  # Expanding slightly past should include the next
  result <- tile_ids_covering_bbox(45.0, -110.0, 46.001, -109.0)
  assert_that(identical(result, c("N45W110", "N46W110")), paste(result, collapse = ","))
  cat("  test_tile_ids_covering_bbox_boundary OK\n")
}

test_tile_ids_covering_bbox_invalid <- function() {
  expect_error(tile_ids_covering_bbox(46.0, -110.0, 45.0, -109.0), "south > north should raise")
  expect_error(tile_ids_covering_bbox(45.0, -109.0, 46.0, -110.0), "west > east should raise")
  cat("  test_tile_ids_covering_bbox_invalid OK\n")
}

test_tile_ids_covering_polyline <- function() {
  # Beartooth-ish horizontal segment at 45N
  result <- tile_ids_covering_polyline(
    rbind(c(45.0, -109.5), c(45.0, -109.0)),
    buffer_m = 10000
  )
  expected <- c("N44W110", "N44W109", "N45W110", "N45W109")
  assert_that(identical(result, expected),
              sprintf("Got: %s\nExpected: %s",
                      paste(result, collapse = ","), paste(expected, collapse = ",")))

  # Single point polyline still works
  result <- tile_ids_covering_polyline(rbind(c(45.5, -109.5)), buffer_m = 5000)
  assert_that(identical(result, c("N45W110")), paste(result, collapse = ","))

  # Long N-S polyline with 300km buffer
  result <- tile_ids_covering_polyline(
    rbind(c(45.0, -110.0), c(46.0, -110.0)),
    buffer_m = 300000
  )
  assert_that(length(result) >= 9,
              sprintf("Expected at least 9 tiles, got %d: %s",
                      length(result), paste(result, collapse = ",")))
  cat("  test_tile_ids_covering_polyline OK\n")
}

test_tile_ids_covering_polyline_empty <- function() {
  assert_that(identical(tile_ids_covering_polyline(list(), buffer_m = 10000), character(0)))
  cat("  test_tile_ids_covering_polyline_empty OK\n")
}

test_neighbor_tile_ids <- function() {
  n <- neighbor_tile_ids("N45W110")
  assert_that(n[["N"]]  == "N46W110")
  assert_that(n[["S"]]  == "N44W110")
  assert_that(n[["E"]]  == "N45W109")
  assert_that(n[["W"]]  == "N45W111")
  assert_that(n[["NE"]] == "N46W109")
  assert_that(n[["NW"]] == "N46W111")
  assert_that(n[["SE"]] == "N44W109")
  assert_that(n[["SW"]] == "N44W111")

  # Polar -- N89 has no northern neighbor
  n <- neighbor_tile_ids("N89W110")
  assert_that(is.null(n[["N"]]))    # would be at lat 90
  assert_that(is.null(n[["NE"]]))
  assert_that(is.null(n[["NW"]]))
  assert_that(n[["S"]] == "N88W110")  # southern still works
  cat("  test_neighbor_tile_ids OK\n")
}

test_invalid_floors <- function() {
  bad <- list(c(90, 0), c(-91, 0), c(0, 180), c(0, -181))
  for (b in bad) {
    expect_error(tile_id_from_floors(b[1], b[2]),
                 sprintf("Should have raised for (%d,%d)", b[1], b[2]))
  }
  cat("  test_invalid_floors OK\n")
}

test_tile_id_from_latlng()
test_parse_tile_id()
test_round_trip()
test_parse_invalid()
test_tile_bounds()
test_tile_bounds_intersects_bbox()
test_tile_ids_covering_bbox_small()
test_tile_ids_covering_bbox_3x3()
test_tile_ids_covering_bbox_boundary()
test_tile_ids_covering_bbox_invalid()
test_tile_ids_covering_polyline()
test_tile_ids_covering_polyline_empty()
test_neighbor_tile_ids()
test_invalid_floors()
cat("\nAll tile_id tests PASS\n")
