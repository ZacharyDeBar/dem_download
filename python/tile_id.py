"""
tile_id.py
━━━━━━━━━━
Pure math: converts between (lat, lng) coordinates and tile IDs.

A tile is a 1°×1° square identified by its southwest corner.
Tile naming follows the existing dem_download.tile_label() convention:
  N45W110 = the tile covering latitudes 45.0°N to 46.0°N and
            longitudes 110.0°W to 109.0°W (SW corner at 45N,110W;
            the tile extends north and east from there).

This module is pure functions over numbers and strings. No I/O, no
filesystem, no rasterio. Easy to unit-test, safe to import anywhere.

Coordinate convention:
  - Latitudes increase northward, range -90 to +90
  - Longitudes increase eastward, range -180 to +180
  - Tile lat_floor is integer latitude (SW corner), e.g. 45 for N45
  - Tile lng_floor is integer longitude (SW corner), e.g. -111 for W111

Boundary semantics:
  - A tile owns its southern and western edges (inclusive)
  - A tile does NOT own its northern or eastern edges (exclusive)
  - This means tile (45, -111) covers lat in [45, 46), lng in [-111, -110)
  - Point exactly at (46, -110) belongs to the next tiles north/east
  - This convention matches existing get_1deg_tiles() in dem_download.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


# Regex for parsing tile IDs of the form N45W110, S05E172, etc.
# Captures: NS, lat digits, EW, lng digits
_TILE_ID_RE = re.compile(r'^([NS])(\d{2})([EW])(\d{3})$')


# ─────────────────────────────────────────────
# Standard tile grid resolution
# ─────────────────────────────────────────────
#
# Single source of truth: every on-disk tile (DEM + flat + boundary
# masks) shares this pixel lattice, so tiles built from different
# native sources (3DEP 10m, GLO-30 30m) sit in the same VRT without a
# resolution seam. Coarser-native tiles are upsampled onto this grid
# at build/migrate time (never downsampled — that would destroy real
# detail that's actually there). Previously duplicated as separate
# hardcoded constants in several sibling scripts outside this repo —
# consolidated here so there's one place to change if the standard
# grid resolution itself ever changes.
#
# NOTE on a future finer native source (e.g. 1m lidar): this constant
# is NOT the seam for that. Mixing 1m tiles into today's 10m lattice by
# just lowering STANDARD_GRID_RES_DEG would mean upsampling every
# surrounding 10m tile to 1m too — ~100x the pixels per tile, per axis,
# for an entire VRT footprint that could span dozens of tiles. That's a
# distance-from-observer LOD problem (fine detail matters most near the
# observer, not at the edge of the compute radius), not a "make the
# whole grid finer" problem.
STANDARD_GRID_RES_DEG = 1.0 / 10800.0  # ~10m at the equator
STANDARD_GRID_RES_RTOL = 1e-4


def is_standard_grid_resolution(res_x_deg: float, res_y_deg: float) -> bool:
    """True iff (res_x, res_y) match STANDARD_GRID_RES_DEG within tolerance."""
    return (abs(res_x_deg - STANDARD_GRID_RES_DEG)
            <= STANDARD_GRID_RES_RTOL * STANDARD_GRID_RES_DEG and
            abs(res_y_deg - STANDARD_GRID_RES_DEG)
            <= STANDARD_GRID_RES_RTOL * STANDARD_GRID_RES_DEG)


# ─────────────────────────────────────────────
# Tile identification
# ─────────────────────────────────────────────

def tile_id_from_latlng(lat: float, lng: float) -> str:
    """
    Return the tile ID containing the given point.

    Examples:
        >>> tile_id_from_latlng(45.5, -110.5)
        'N45W111'
        >>> tile_id_from_latlng(0.5, 0.5)
        'N00E000'
        >>> tile_id_from_latlng(-12.7, 178.3)
        'S13E178'

    Boundary behavior (consistent with [s,n) × [w,e) convention):
        >>> tile_id_from_latlng(46.0, -110.0)  # NE corner of N45W111
        'N46W110'  # belongs to next tile north and east
    """
    lat_floor = math.floor(lat)
    lng_floor = math.floor(lng)
    return tile_id_from_floors(lat_floor, lng_floor)


def tile_id_from_floors(lat_floor: int, lng_floor: int) -> str:
    """
    Build a tile ID from integer SW-corner floors.

    Args:
        lat_floor: integer latitude in [-90, 90)
        lng_floor: integer longitude in [-180, 180)

    Returns:
        Tile ID string like 'N45W110' (2-digit lat, 3-digit lng).

    For lng_floor=-111, the W component is 111 (the absolute value).
    """
    if not -90 <= lat_floor < 90:
        raise ValueError(f"lat_floor {lat_floor} outside [-90, 90)")
    if not -180 <= lng_floor < 180:
        raise ValueError(f"lng_floor {lng_floor} outside [-180, 180)")

    ns = 'N' if lat_floor >= 0 else 'S'
    ew = 'W' if lng_floor < 0 else 'E'
    # For S tiles, the number is |lat_floor|; for N tiles it's lat_floor.
    # For W tiles, |lng_floor|; for E tiles, lng_floor.
    lat_num = abs(lat_floor)
    lng_num = abs(lng_floor)
    return f"{ns}{lat_num:02d}{ew}{lng_num:03d}"


def parse_tile_id(tile_id: str) -> Tuple[int, int]:
    """
    Parse a tile ID into (lat_floor, lng_floor) integers.

    Examples:
        >>> parse_tile_id('N45W110')
        (45, -110)
        >>> parse_tile_id('S13E178')
        (-13, 178)
        >>> parse_tile_id('N00E000')
        (0, 0)
    """
    m = _TILE_ID_RE.match(tile_id)
    if not m:
        raise ValueError(f"Invalid tile ID format: {tile_id!r} "
                         f"(expected like 'N45W110')")
    ns, lat_str, ew, lng_str = m.groups()
    lat_floor = int(lat_str)
    lng_floor = int(lng_str)
    if ns == 'S':
        lat_floor = -lat_floor
    if ew == 'W':
        lng_floor = -lng_floor
    # Re-validate ranges
    if not -90 <= lat_floor < 90:
        raise ValueError(f"Parsed lat_floor {lat_floor} from {tile_id!r} "
                         f"outside [-90, 90)")
    if not -180 <= lng_floor < 180:
        raise ValueError(f"Parsed lng_floor {lng_floor} from {tile_id!r} "
                         f"outside [-180, 180)")
    return lat_floor, lng_floor


# ─────────────────────────────────────────────
# Tile geometry
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class TileBounds:
    """
    A tile's geographic bounds. All values in decimal degrees.

    Following the [s, n) × [w, e) convention:
      - south, west: SW corner (owned by this tile)
      - north, east: NE corner (owned by next tiles N and E)
    """
    south: float
    west:  float
    north: float
    east:  float

    @property
    def center(self) -> Tuple[float, float]:
        """Return (lat, lng) of tile center."""
        return ((self.south + self.north) / 2,
                (self.west + self.east) / 2)

    def contains(self, lat: float, lng: float) -> bool:
        """True if (lat, lng) falls inside this tile's bounds."""
        return (self.south <= lat < self.north and
                self.west  <= lng < self.east)

    def intersects_bbox(self, south: float, west: float,
                              north: float, east: float) -> bool:
        """
        True if this tile's bounds intersect the given bbox.
        Uses standard rectangle-overlap test.
        """
        return not (east  <= self.west  or west  >= self.east  or
                    north <= self.south or south >= self.north)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """(south, west, north, east) — matches the existing bounds order."""
        return (self.south, self.west, self.north, self.east)


def tile_bounds(tile_id: str) -> TileBounds:
    """
    Return the geographic bounds of a tile by its ID.

    Examples:
        >>> tile_bounds('N45W110')
        TileBounds(south=45.0, west=-110.0, north=46.0, east=-109.0)
        >>> tile_bounds('S05E172').as_tuple()
        (-5.0, 172.0, -4.0, 173.0)

    Note that the tile's west longitude is its lng_floor (the SW corner),
    NOT its NW corner. So N45W110 covers lng in [-110, -109).
    """
    lat_floor, lng_floor = parse_tile_id(tile_id)
    return TileBounds(
        south=float(lat_floor),
        west =float(lng_floor),
        north=float(lat_floor + 1),
        east =float(lng_floor + 1),
    )


# ─────────────────────────────────────────────
# Bbox → tile set
# ─────────────────────────────────────────────

def tile_ids_covering_bbox(
    south: float, west: float,
    north: float, east: float,
) -> List[str]:
    """
    Return all tile IDs whose bounds intersect the given bbox.

    The returned list is ordered by (lat_floor, lng_floor) — south to
    north, then west to east. This is a deterministic order useful for
    logging and registry operations.

    Args:
        south, west, north, east: bbox in decimal degrees.

    Examples:
        >>> tile_ids_covering_bbox(45.2, -110.7, 45.9, -109.3)
        ['N45W110', 'N45W109']
        >>> tile_ids_covering_bbox(44.5, -111.5, 46.5, -109.5)
        ['N44W111', 'N44W110', 'N44W109',
         'N45W111', 'N45W110', 'N45W109',
         'N46W111', 'N46W110', 'N46W109']

    Boundary behavior:
      A bbox whose north edge lies exactly on a tile boundary does
      NOT include the next tile north — consistent with [s,n)×[w,e).
      But the eastern bbox edge being slightly past a tile boundary
      DOES include the next tile east.
    """
    if south >= north:
        raise ValueError(f"south {south} >= north {north}")
    if west >= east:
        raise ValueError(f"west {west} >= east {east}")

    # Tiles whose [lat_floor, lat_floor+1) intersects [south, north)
    # are those where lat_floor < north AND lat_floor+1 > south,
    # i.e. lat_floor in [floor(south), ceil(north))
    # Special-case: when north is exactly an integer, we don't want
    # to include that tile (since its lat_floor = north has bounds
    # [north, north+1), which doesn't intersect [south, north)).
    # math.ceil(north) gives us this naturally:
    #   ceil(45.9) = 46, range(floor(45.2), 46) = [45]
    #   ceil(46.0) = 46, range(44, 46) = [44, 45] — correct, excludes 46
    lat_min = math.floor(south)
    lat_max = math.ceil(north)
    if lat_max == north:
        # north exactly on a boundary; the tile starting at north
        # doesn't intersect [south, north)
        pass  # ceil already gives us the right exclusive upper
    lng_min = math.floor(west)
    lng_max = math.ceil(east)

    tile_ids = []
    for lat_floor in range(lat_min, lat_max):
        for lng_floor in range(lng_min, lng_max):
            tile_ids.append(tile_id_from_floors(lat_floor, lng_floor))
    return tile_ids


def tile_ids_covering_polyline(
    latlngs: Iterable[Tuple[float, float]],
    buffer_m: float,
) -> List[str]:
    """
    Return all tile IDs whose bounds intersect the buffered polyline.

    Args:
        latlngs:  iterable of (lat, lng) points along the polyline
        buffer_m: buffer width in metres (typically the compute radius
                  plus a small safety margin)

    Returns:
        Sorted list of tile IDs. Empty if the input is empty.

    This is the basic predictive-tile-coverage primitive: given a
    route's observer positions and the compute radius, return the
    set of 1° tiles that any observer along the route could possibly
    need. Used by expansion_predictor.

    Implementation: compute the polyline's bbox, expand by the buffer
    in degrees (using equirectangular approximation), then call
    tile_ids_covering_bbox. This may include slightly more tiles than
    strictly necessary at polyline curves, but the over-coverage is
    bounded and harmless. Tighter coverage (per-segment buffering)
    can come later if proven necessary.

    Examples:
        >>> # Beartooth-ish polyline
        >>> tile_ids_covering_polyline(
        ...     [(45.0, -109.5), (45.0, -109.0)],
        ...     buffer_m=10_000,
        ... )
        ['N44W110', 'N44W109', 'N45W110', 'N45W109']
    """
    pts = list(latlngs)
    if not pts:
        return []

    lats = [p[0] for p in pts]
    lngs = [p[1] for p in pts]
    s_lat = min(lats)
    n_lat = max(lats)
    w_lng = min(lngs)
    e_lng = max(lngs)

    # Equirectangular buffer (lat is uniform, lng depends on cos(lat))
    DEG_PER_M_LAT = 1.0 / 111_320.0
    mean_lat = (s_lat + n_lat) / 2.0
    cos_lat = max(math.cos(math.radians(mean_lat)), 1e-6)
    deg_per_m_lng = DEG_PER_M_LAT / cos_lat

    buf_lat = buffer_m * DEG_PER_M_LAT
    buf_lng = buffer_m * deg_per_m_lng

    return tile_ids_covering_bbox(
        south=s_lat - buf_lat,
        west =w_lng - buf_lng,
        north=n_lat + buf_lat,
        east =e_lng + buf_lng,
    )


# ─────────────────────────────────────────────
# Neighbor helpers
# ─────────────────────────────────────────────

def neighbor_tile_ids(tile_id: str) -> dict:
    """
    Return the 8 neighbors of a tile keyed by compass direction.

    Returns a dict with keys: 'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'
    Values are tile IDs. Returns None for any neighbor that would
    fall outside the valid lat/lng range (e.g. polar neighbors).

    Examples:
        >>> n = neighbor_tile_ids('N45W110')
        >>> n['N']
        'N46W110'
        >>> n['SE']
        'N44W109'
    """
    lat_floor, lng_floor = parse_tile_id(tile_id)
    out = {}
    for name, (dlat, dlng) in [
        ('N',  ( 1,  0)),
        ('S',  (-1,  0)),
        ('E',  ( 0,  1)),
        ('W',  ( 0, -1)),
        ('NE', ( 1,  1)),
        ('NW', ( 1, -1)),
        ('SE', (-1,  1)),
        ('SW', (-1, -1)),
    ]:
        nlat = lat_floor + dlat
        nlng = lng_floor + dlng
        # Skip neighbors outside valid range
        if not -90 <= nlat < 90:
            out[name] = None
            continue
        if not -180 <= nlng < 180:
            # Could wrap longitude here, but for our purposes the
            # antimeridian doesn't matter — flagging as None is safer
            out[name] = None
            continue
        out[name] = tile_id_from_floors(nlat, nlng)
    return out


# ─────────────────────────────────────────────
# Corners → bounds, with a size guard
# ─────────────────────────────────────────────

# Above this many 1°×1° tiles, bounds_from_tile_corners() refuses
# unless allow_large=True. Not an arbitrary number: a real 20-tile,
# source="auto" (mostly 3DEP_10) mosaic build filled a test disk
# partway through tile 9 — each 3DEP 10m tile merges from an 81-subtile
# WCS grid to roughly 200-300MB before GLO-30 gap-fill and water
# correction add more. 25 tiles (a 5x5deg block) is comfortably past
# typical single-region use without being an accident waiting to happen.
MAX_MOSAIC_TILES = 25


def bounds_from_tile_corners(
    corner1_id: str,
    corner2_id: str,
    max_tiles: int = MAX_MOSAIC_TILES,
    allow_large: bool = False,
) -> dict:
    """
    Given two 1°×1° tile IDs marking any two opposite corners of a
    desired rectangular study area (e.g. "N44W113" and "N47W109" —
    order doesn't matter, NE+SW or NW+SE or any other pairing all
    work), return the bounding dict(south=, west=, north=, east=,
    n_tiles=) spanning both tiles' full footprints.

    Longitude is resolved to the SHORTER arc between the two corners
    (e.g. corners at 10E and 20E span 10 degrees eastward, not 350
    degrees the other way around) — almost always what's intended for
    a regional mosaic, and avoids requiring a third "which way around"
    argument. If that shorter arc still crosses the ±180 antimeridian
    — i.e. the corners themselves straddle the date line — this raises
    rather than silently producing a reversed or wrapped bbox: nothing
    downstream (crop/merge/tile enumeration) supports a
    dateline-crossing extent. Build two separate mosaics instead.

    Args:
        max_tiles:   size cap in degree-tiles (see MAX_MOSAIC_TILES)
        allow_large: set True to bypass the cap for a real large run

    Examples:
        >>> bounds_from_tile_corners('N44W113', 'N47W109')
        {'south': 44, 'west': -113, 'north': 48, 'east': -108, 'n_tiles': 20}
    """
    b1 = tile_bounds(corner1_id)
    b2 = tile_bounds(corner2_id)

    south = min(b1.south, b2.south)
    north = max(b1.north, b2.north)

    w1, w2 = b1.west, b2.west
    delta = ((w2 - w1 + 180) % 360) - 180  # signed shortest arc, (-180, 180]
    if delta >= 0:
        west, east = w1, w2 + 1
    else:
        west, east = w2, w1 + 1

    if east <= west:
        raise ValueError(
            f"bounds_from_tile_corners({corner1_id}, {corner2_id}): the "
            f"shorter arc between these tiles crosses the ±180 "
            f"antimeridian (west={west}, east={east}), which isn't "
            f"supported downstream. Pick corners that don't straddle the "
            f"date line, or build two separate mosaics on either side of it."
        )

    n_tiles = len(tile_ids_covering_bbox(south, west, north, east))
    if n_tiles > max_tiles and not allow_large:
        raise ValueError(
            f"bounds_from_tile_corners({corner1_id}, {corner2_id}): area "
            f"spans {n_tiles} degree-tiles, above the {max_tiles}-tile "
            f"default cap. 3DEP 10m tiles run roughly 200-300MB each — a "
            f"mosaic this size can be several GB and take a long time to "
            f"download. Pick closer corners, or pass allow_large=True "
            f"(optionally with a higher max_tiles) if you really want this."
        )
    if n_tiles > 9:
        print(f"[BOUNDS] Note: {n_tiles} degree-tiles required for "
              f"{corner1_id} to {corner2_id} -- large mosaics take a "
              f"while and real disk space (each 3DEP 10m tile is "
              f"roughly 200-300MB).")

    return {'south': south, 'west': west, 'north': north, 'east': east,
            'n_tiles': n_tiles}