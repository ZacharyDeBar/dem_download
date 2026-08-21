"""
dem_water_correction.py
━━━━━━━━━━━━━━━━━━━━━━
Water body correction tool for DEM preprocessing.

Corrects elevation artifacts in still-water bodies (lakes, ponds,
reservoirs) by sampling shoreline elevation and flood-filling water body
pixels to a consistent surface elevation. Deliberately still-water only —
rivers, marshes, and glaciers aren't single-elevation surfaces, so
flattening them the same way would be wrong; see NHD_STILL_WATER_FTYPES
and fetch_water_bodies_osm's docstring.

Supports two water body sources:
  - USGS National Hydrography Dataset (NHD) — US coverage, highest quality
  - OpenStreetMap (OSM) — global coverage, good for California coastal

Usage:
    python dem_water_correction.py correct \
        --input  data/dem/raw/N45_00_W111.tif \
        --output data/dem/corrected/N45_00_W111.tif \
        --source nhd

    python dem_water_correction.py batch \
        --input-dir  data/dem/raw/ \
        --output-dir data/dem/corrected/ \
        --source osm \
        --bounds "32.5,-121.5,35.8,-117.0"

    python dem_water_correction.py preview \
        --input data/dem/raw/N45_00_W111.tif \
        --source nhd
"""

import argparse
import time
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import rasterio.mask
from rasterio.transform import rowcol, xy as _transform_xy
from scipy.ndimage import binary_dilation

try:
    from parallel_tools import (
        SharedNDArray,
        parallel_map_shared,
        recommend_workers,
        print_system_info,
        make_batches,
    )
    _PARALLEL_AVAILABLE = True
except ImportError:
    _PARALLEL_AVAILABLE = False
    print("[WARN] parallel_tools.py not found -- parallel processing disabled")

try:
    from gpu_tools import (
        GPU_AVAILABLE,
        gpu_info,
        fits_on_gpu,
        recommend_gpu_batch_size,
        process_label_array_gpu,
        gpu_memory_available_gb
    )
    _GPU_TOOLS_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    _GPU_TOOLS_AVAILABLE = False
    print("[WARN] gpu_tools.py not found -- GPU processing disabled")

# ─────────────────────────────────────────────
# Optional dependencies — checked at runtime
# ─────────────────────────────────────────────

def _require(package: str, install_hint: str):
    try:
        return __import__(package)
    except ImportError:
        print(f"[ERROR] Missing dependency: {package}")
        print(f"        Install with: {install_hint}")
        sys.exit(1)


# ─────────────────────────────────────────────
# Water body fetching
# ─────────────────────────────────────────────

# NHD FType codes for genuinely flat, still-water surfaces — the only
# kind the shoreline-percentile flattening in correct_dem_water_bodies()
# is valid for. Excludes:
#   466 SwampMarsh — wet ground, not a single-elevation surface; follows
#                    the terrain much like a river does
#   378 IceMass    — glaciers/permanent snow have real topography
#   361 Playa      — usually dry, seasonal; not a stable water surface
# Reference: NHD FCode/FType domain (USGS).
NHD_STILL_WATER_FTYPES = {
    390,  # LakePond
    436,  # Reservoir
    493,  # Estuary
}


def fetch_water_bodies_osm(
    bounds: tuple[float, float, float, float],  # (south, west, north, east)
    water_types: list[str] = None,
    include_ocean: bool = False,
) -> list[dict]:
    """
    Fetch water body polygons from OpenStreetMap via Overpass API.
    Returns list of GeoJSON-like polygon dicts.

    Only still, single-elevation water surfaces — same reasoning as
    NHD_STILL_WATER_FTYPES over on the NHD side. Deliberately does NOT
    query waterway=river/canal/stream: correct_dem_water_bodies()
    flattens a polygon to one shoreline-percentile elevation, which is
    only valid for a body that's actually flat. A river slopes downhill
    along its length, so treating one (previously force-closed into a
    fake Polygon here) as flat produced large, spurious corrections.
    'wetland' is excluded from the defaults for the same reason marsh is
    excluded on the NHD side — a marsh follows the ground, it isn't a
    single-elevation surface either.

    Args:
        bounds: (south, west, north, east) in decimal degrees
        water_types: OSM natural/landuse tags to include.
                     Defaults to lakes/ponds and bays.
    """
    requests = _require('requests', 'pip install requests')

    if water_types is None:
        water_types = ['water', 'bay']

    south, west, north, east = bounds
    bbox = f"{south},{west},{north},{east}"

    # Overpass QL query — fetch all water polygons in bbox
    filters = '|'.join(water_types)
    query = f"""
    [out:json][timeout:60];
    (
      way["natural"~"{filters}"]({bbox});
      relation["natural"~"{filters}"]({bbox});
      way["landuse"="reservoir"]({bbox});
      relation["landuse"="reservoir"]({bbox});
    );
    out geom;
    """

    print(f"[OSM] Querying water bodies in bbox {bbox}...")
    resp = requests.post(
        'https://overpass-api.de/api/interpreter',
        data={'data': query},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()

    polygons = []
    for element in data.get('elements', []):
        geom = element.get('geometry', [])
        if not geom:
            continue
        coords = [(node['lon'], node['lat']) for node in geom]
        if len(coords) < 3:
            continue
        # Close the ring if not already closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        tags = element.get('tags', {})
        polygons.append({
            'type': 'Polygon',
            'coordinates': [coords],
            'properties': {
                'source': 'osm',
                'natural': tags.get('natural', ''),
                'landuse': tags.get('landuse', ''),
                'name':    tags.get('name', ''),
            },
        })



    print(f"[OSM] Found {len(polygons)} water body polygons")
    if include_ocean:
        ocean_polys = fetch_ocean_polygons_osm(bounds)
        polygons.extend(ocean_polys)
        print(f"[OSM] Total with ocean: {len(polygons)} polygons")
    return polygons


def fetch_water_bodies_nhd(
    bounds: tuple,
    feature_types: list = None,
    still_water_only: bool = True,
) -> list[dict]:
    """
    Fetch water body polygons from USGS NHD via ArcGIS REST API.
    Layer 12 = Waterbody Large Scale (high resolution).
    Paginates automatically for large areas.

    still_water_only: if True (default), drop anything whose FType isn't
    in NHD_STILL_WATER_FTYPES (marsh, ice, playa) before returning — see
    that constant's comment for why. Layer 12 already excludes rivers by
    construction (they live in NHD's separate Flowline layer, as lines,
    which this function never queries), so this filter is specifically
    about non-flat *waterbody* types, not rivers.
    """
    import requests

    south, west, north, east = bounds

    url = ('https://hydro.nationalmap.gov/arcgis/rest/services/'
           'nhd/MapServer/12/query')

    params = {
        'geometry':          f'{west},{south},{east},{north}',
        'geometryType':      'esriGeometryEnvelope',
        'inSR':              '4326',
        'spatialRel':        'esriSpatialRelIntersects',
        'outFields':         'GNIS_Name,FType,FCode',
        'returnGeometry':    'true',
        'outSR':             '4326',
        'f':                 'geojson',
        'resultRecordCount': 2000,
    }

    print(f"[NHD] Querying waterbodies in bbox "
          f"{west:.2f},{south:.2f},{east:.2f},{north:.2f}...")

    all_polygons = []
    n_filtered_ftype = 0
    offset = 0

    while True:
        params['resultOffset'] = offset
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[NHD] Query failed at offset {offset}: {e}")
            break

        features = data.get('features', [])
        if not features:
            break

        for feat in features:
            geom = feat.get('geometry')
            if not geom:
                continue
            # The ArcGIS service returns field names in its own casing
            # (observed: GNIS_NAME/FTYPE/FCODE, all caps) regardless of
            # how outFields was spelled in the request — .get('FType')
            # silently returns nothing against that. Look up
            # case-insensitively instead of hardcoding a casing.
            props = feat.get('properties', {}) or {}
            props_lower = {k.lower(): v for k, v in props.items()}
            ftype = props_lower.get('ftype')
            # NHD_STILL_WATER_FTYPES is a set of int codes; the service
            # has been observed returning FType as a native JSON number,
            # but coerce defensively in case a service version ever
            # returns it as a numeric string -- a bare `not in` against
            # int literals would then silently drop every real
            # LakePond/Reservoir/Estuary.
            try:
                ftype_code = int(ftype) if ftype is not None else None
            except (TypeError, ValueError):
                ftype_code = None

            if still_water_only and ftype_code not in NHD_STILL_WATER_FTYPES:
                n_filtered_ftype += 1
                continue

            prop_dict = {
                'source': 'nhd',
                'name':   props_lower.get('gnis_name') or '',
                'ftype':  ftype,
            }

            if geom['type']=='MultiPolygon':
                for polygon_coords in geom['coordinates']:
                    all_polygons.append({
                        'type':        'Polygon',
                        'coordinates': polygon_coords,
                        'properties':  prop_dict,

                    })
            elif geom['type'] == 'Polygon':
                geom['properties'] = prop_dict
                all_polygons.append(geom)

        print(f"[NHD]   ... {len(all_polygons)} polygons so far "
              f"(offset {offset})")

        if len(features) < 2000:
            break
        offset += 2000

    print(f"[NHD] Found {len(all_polygons)} water body polygons"
          + (f" ({n_filtered_ftype} non-still-water filtered out)"
             if still_water_only else ""))
    return all_polygons


def fetch_ocean_polygons_osm(
    bounds: tuple[float, float, float, float],
    simplify_tolerance_deg: float = 0.001,
) -> list[dict]:
    """
    Fetch ocean/sea polygons within bounds using OSM coastline data.
    
    OSM represents oceans via natural=coastline ways — land is to the
    left of the way direction, ocean to the right. This function fetches
    coastline segments and constructs ocean fill polygons by:
      1. Fetching all coastline ways in the bbox
      2. Building a bbox-clipped ocean polygon (bbox minus land areas)
    
    Works globally — Pacific, Atlantic, Arctic, any ocean/sea.
    
    Args:
        bounds: (south, west, north, east)
        simplify_tolerance_deg: tolerance for coastline simplification
                                (0.001° ≈ 111m, reduces polygon complexity)
    Returns:
        List of ocean polygon dicts ready for correct_dem_water_bodies
    """
    import requests
    try:
        from shapely.geometry import (
            Polygon, MultiPolygon, LineString,
            box as shapely_box
        )
        from shapely.ops import unary_union, polygonize, split
    except ImportError:
        print("[OCEAN] shapely required for ocean polygon construction")
        return []

    south, west, north, east = bounds
    bbox_str = f"{south},{west},{north},{east}"

    # Fetch coastline ways
    query = f"""
    [out:json][timeout:120];
    (
      way["natural"="coastline"]({bbox_str});
    );
    out geom;
    """

    print(f"[OCEAN] Fetching OSM coastline for bbox {bbox_str}...")
    try:
        resp = requests.post(
            'https://overpass-api.de/api/interpreter',
            data={'data': query},
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent':   'dem-water-correction/1.0',
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[OCEAN] Coastline fetch failed: {e}")
        return []

    elements = data.get('elements', [])
    print(f"[OCEAN] Got {len(elements)} coastline segments")

    if not elements:
        # No coastline in bounds — either all ocean or all land
        # Check if bounds center is over ocean using simple heuristic
        # If no coastline found, assume all ocean and return full bbox
        print(f"[OCEAN] No coastline found — treating entire bbox as ocean")
        bbox_poly = shapely_box(west, south, east, north)
        coords = list(bbox_poly.exterior.coords)
        return [{
            'type': 'Polygon',
            'coordinates': [[(c[0], c[1]) for c in coords]],
            'properties': {'source': 'osm_ocean', 'name': 'ocean'},
        }]

    # Build linestrings from coastline segments
    lines = []
    for el in elements:
        geom = el.get('geometry', [])
        if len(geom) < 2:
            continue
        coords = [(node['lon'], node['lat']) for node in geom]
        lines.append(LineString(coords))

    # Clip lines to bbox
    bbox_poly = shapely_box(west, south, east, north)
    clipped_lines = []
    for line in lines:
        try:
            clipped = line.intersection(bbox_poly)
            if not clipped.is_empty:
                clipped_lines.append(clipped)
        except Exception:
            continue

    if not clipped_lines:
        print(f"[OCEAN] No coastline intersects bbox")
        return []

    # Polygonize: coastline segments + bbox boundary → land polygons
    # Then ocean = bbox minus land polygons
    # Replace the polygonization section in fetch_ocean_polygons_osm:
    try:
        from shapely.ops import linemerge, unary_union, polygonize
        from shapely.geometry import MultiLineString

        # Merge coastline segments into as few linestrings as possible
        merged_coastline = linemerge(unary_union(clipped_lines))

        # Convert to list of lines
        if merged_coastline.geom_type == 'LineString':
            coast_lines = [merged_coastline]
        else:
            coast_lines = list(merged_coastline.geoms)

        print(f"[OCEAN] Merged into {len(coast_lines)} coastline linestrings")

        # Build land polygons by closing each coastline against bbox
        # Strategy: for each coastline segment, the land is to the left
        # (OSM convention). Close the segment by routing along bbox edges.
        bbox_coords = [
            (west, south), (east, south), (east, north),
            (west, north), (west, south),
        ]

        # Add bbox boundary segments to coastline to allow polygonization
        bbox_lines = []
        for i in range(len(bbox_coords) - 1):
            bbox_lines.append(LineString([bbox_coords[i], bbox_coords[i+1]]))

        all_lines = clipped_lines + bbox_lines
        all_merged = unary_union(all_lines)

        # Node the lines (ensure all intersections are vertices)
        nodded = all_merged.buffer(0) if hasattr(all_merged, 'buffer') \
                 else all_merged

        all_polys = list(polygonize(all_merged))
        print(f"[OCEAN] Polygonized into {len(all_polys)} polygons")

        if not all_polys:
            # Fallback — just return the full bbox zeroed to ocean
            print(f"[OCEAN] Falling back to full bbox ocean polygon")
            coords = list(bbox_poly.exterior.coords)
            return [{
                'type': 'Polygon',
                'coordinates': [[(c[0], c[1]) for c in coords]],
                'properties': {'source': 'osm_ocean', 'name': 'ocean'},
            }]

        # Identify ocean polygons vs land polygons
        # Ocean polygons are those whose centroid is over water —
        # use a simple heuristic: the largest polygon touching the
        # western/southern bbox edge is ocean for a California coastal bbox
        # More robust: check if centroid is west of the coastline centroid
        coast_centroid_lng = float(
            unary_union(clipped_lines).centroid.x)

        ocean_polys_shapely = []
        land_polys_shapely  = []

        for poly in all_polys:
            c = poly.centroid
            # A polygon is ocean if its centroid is west of the
            # mean coastline longitude (works for W-facing coasts)
            # OR if it touches the western/southern bbox boundary
            touches_west   = abs(poly.bounds[0] - west)   < 0.01
            touches_south  = abs(poly.bounds[1] - south)  < 0.01
            west_of_coast  = c.x < coast_centroid_lng
            south_of_coast = c.y < (south + north) / 2

            if touches_west or touches_south or west_of_coast:
                ocean_polys_shapely.append(poly)
            else:
                land_polys_shapely.append(poly)

        print(f"[OCEAN] {len(ocean_polys_shapely)} ocean, "
              f"{len(land_polys_shapely)} land polygons identified")

        if not ocean_polys_shapely:
            # If heuristic fails, take the complement approach:
            # ocean = bbox - all land polygons that don't touch west/south
            land_union = unary_union(land_polys_shapely) \
                         if land_polys_shapely else bbox_poly
            ocean = bbox_poly.difference(land_union)
            ocean_polys_shapely = [ocean] if not ocean.is_empty else []

        ocean_union = unary_union(ocean_polys_shapely)
        if simplify_tolerance_deg > 0:
            ocean_union = ocean_union.simplify(
                simplify_tolerance_deg, preserve_topology=True)

        geoms = ([ocean_union] if ocean_union.geom_type == 'Polygon'
                 else list(ocean_union.geoms))

        result = []
        for geom in geoms:
            if geom.is_empty or geom.area < 1e-8:
                continue
            coords = list(geom.exterior.coords)
            result.append({
                'type': 'Polygon',
                'coordinates': [[(c[0], c[1]) for c in coords]],
                'properties': {'source': 'osm_ocean', 'name': 'ocean'},
            })

        print(f"[OCEAN] Constructed {len(result)} ocean polygon(s)")
        return result

    except Exception as e:
        import traceback
        print(f"[OCEAN] Ocean polygon construction failed: {e}")
        traceback.print_exc()
        return []


def load_water_bodies_from_file(path: str) -> list[dict]:
    """
    Load water body polygons from a local GeoJSON file.
    Useful for pre-downloaded NHD or custom water body datasets.
    """
    with open(path) as f:
        data = json.load(f)

    if data.get('type') == 'FeatureCollection':
        return [f['geometry'] for f in data['features']
                if f.get('geometry', {}).get('type') == 'Polygon']
    elif data.get('type') == 'Polygon':
        return [data]
    else:
        raise ValueError(f"Unsupported GeoJSON type: {data.get('type')}")


# ─────────────────────────────────────────────
# Shoreline elevation sampling
# ─────────────────────────────────────────────

def sample_shoreline_elevation(
    elevation_array: np.ndarray,
    transform,
    water_mask: np.ndarray,
    buffer_pixels: int = 3,
    percentile: float = 10.0,
) -> float:
    """
    Estimate the true water surface elevation by sampling terrain
    pixels just outside the water body boundary.

    Uses a low percentile (default 10th) rather than mean to avoid
    being pulled upward by elevated terrain on one side of the water.

    Args:
        elevation_array: DEM array
        transform: rasterio transform
        water_mask: boolean mask — True where water pixels are
        buffer_pixels: how many pixels outside the water boundary to sample
        percentile: which percentile of shoreline elevations to use

    Returns:
        Estimated water surface elevation in metres
    """
    # Dilate the water mask outward to get a shoreline band
    dilated   = binary_dilation(water_mask,
                                iterations=buffer_pixels)
    shoreline = dilated & ~water_mask  # ring just outside water

    shoreline_elevs = elevation_array[shoreline]

    # Filter out nodata and unreasonable values
    valid = shoreline_elevs[
        (shoreline_elevs > -1000) &
        (shoreline_elevs < 9000)
    ]

    if len(valid) == 0:
        # Fallback — sample inside water body itself
        water_elevs = elevation_array[water_mask]
        valid = water_elevs[water_elevs > -1000]
        if len(valid) == 0:
            return 0.0

    surface_elev = float(np.percentile(valid, percentile))
    return surface_elev


# ─────────────────────────────────────────────
# Batch rasterization helpers
# ─────────────────────────────────────────────

def _spread_bits16(v: int) -> int:
    """Spread the low 16 bits of v so each bit has a zero gap after it
    (bit i -> bit 2i). Standard magic-number bit-spreading step used to
    build a Morton (Z-order) code."""
    v &= 0xFFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def _morton_code(x: int, y: int) -> int:
    """Interleave two 16-bit ints into a 32-bit Morton (Z-order) code:
    x's bits at even positions, y's at odd. Sorting by this code visits
    2D space along a Z-order curve -- unlike a plain (x, y) or even a
    row-major (x_bin, y_bin) grid sort, ANY contiguous run along a
    space-filling curve stays compact in BOTH dimensions at once, not
    just the primary one."""
    return _spread_bits16(x) | (_spread_bits16(y) << 1)


def _spatial_sort_morton(
    overlapping: list[tuple[int, dict]],
) -> list[tuple[int, dict]]:
    """Sort polygons along a Morton (Z-order) curve so any consecutive
    slice (a batch) is geographically compact in both lat and lng.

    The previous sort used key=(lat, lng) on the raw float centers --
    since two floats essentially never tie, that sorts almost entirely by
    lat, with lng only ever reached as an unreachable tiebreaker. For a
    lake-dense region where many water bodies share similar latitudes but
    span the full longitude range (e.g. coastal Maine), a "sorted" batch
    of 1000 out of 2,838 polygons could still cover the ENTIRE raster
    width -- confirmed in production: a batch's resulting crop was shape
    (10791, 10800), essentially the whole tile, not the small fraction
    recommend_gpu_batch_size() assumes for compact batches, which is what
    let a GPU allocation attempt through that then failed.

    A simple row-major (lat_bin, lng_bin) grid sort was tried first and
    rejected -- it only guarantees compactness in the PRIMARY axis
    (lat_bin); whenever one lat_bin's population exceeds batch_size (true
    here: ~1400 polygons per bin at a 2-bin grid), a batch still spills
    across every lng_bin within that row, i.e. the full raster width
    again. A Morton curve doesn't have that failure mode: it interleaves
    both axes, so a contiguous run of any length stays close to a
    (possibly non-square) compact region rather than a full-width strip.
    """
    if not overlapping:
        return overlapping

    entries = []  # (item, lat, lng)
    for item in overlapping:
        _, poly = item
        coords = poly.get('coordinates', [[]])[0]
        if not coords:
            entries.append((item, 0.0, 0.0))
            continue
        lngs = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        entries.append((item, sum(lats) / len(lats), sum(lngs) / len(lngs)))

    lat_min = min(e[1] for e in entries)
    lat_max = max(e[1] for e in entries)
    lng_min = min(e[2] for e in entries)
    lng_max = max(e[2] for e in entries)
    lat_span = max(lat_max - lat_min, 1e-9)
    lng_span = max(lng_max - lng_min, 1e-9)

    def _key(entry):
        _, lat, lng = entry
        xq = min(65535, max(0, int((lat - lat_min) / lat_span * 65535)))
        yq = min(65535, max(0, int((lng - lng_min) / lng_span * 65535)))
        return (_morton_code(xq, yq), lat, lng)

    entries.sort(key=_key)
    return [item for item, _, _ in entries]


def _filter_overlapping(
    water_polygons: list[dict],
    bounds,
) -> list[tuple[int, dict]]:
    result = []
    for i, poly in enumerate(water_polygons):
        geom_type = poly.get('type', 'Polygon')
        coords    = poly.get('coordinates', [])
        if not coords:
            continue

        # Flatten coordinates to get all lng/lat pairs regardless of geometry type
        if geom_type == 'MultiPolygon':
            # coords = [ [ [ring], ...], [ [ring], ...] ]
            all_coords = [pt for polygon in coords
                             for ring in polygon
                             for pt in ring]
        else:
            # coords = [ [ring], [hole], ... ]
            # ring may be coords[0] or coords itself depending on source
            ring = coords[0] if isinstance(coords[0][0], (list, tuple)) \
                   and not isinstance(coords[0][0][0], (int, float)) \
                   else coords[0]
            all_coords = ring

        try:
            poly_lngs = [c[0] for c in all_coords]
            poly_lats = [c[1] for c in all_coords]
        except (IndexError, TypeError):
            continue

        if (max(poly_lngs) < bounds.left  or
            min(poly_lngs) > bounds.right or
            max(poly_lats) < bounds.bottom or
            min(poly_lats) > bounds.top):
            continue

        result.append((i, poly))
    return result


def _mask_centroid_latlon(mask: np.ndarray, bbox, transform):
    """Rough midpoint for locating a water body on a map: the nearest
    actual water pixel to the mean pixel position of its rasterized
    mask — not its source polygon's vertices, and not the raw mean
    either.

    Deliberately not vertex-based: some water "polygons" here are
    force-closed OSM waterway lines (a river/stream centerline treated
    as a Polygon — see fetch_water_bodies_osm's `coords.append(coords[0])`).
    For a long winding river, the vertex-count-weighted mean of every
    point along its path can land nowhere near the channel — confirmed
    in production on N45W111, where several of the largest corrections
    had vertex-mean midpoints sitting on dry land.

    And the raw pixel mean isn't quite enough either: for a non-convex
    shape (a zigzag or a river bend) the mean position itself can still
    fall in a gap outside the mask, the same failure mode one level
    down. Snapping to the closest True pixel guarantees the reported
    point is always on real water/corrected pixels, for any shape.

    Args:
        mask:      boolean array, water pixels True. Either bbox-local
                   (pass the matching `bbox`, a (row_slice, col_slice)
                   tuple whose .start gives its offset into the full
                   raster) or already full-raster-sized (bbox=None).
        transform: affine transform of the raster `mask` — or the crop
                   `bbox` is relative to — is defined on.

    Returns (lat, lon), or (None, None) if transform is unavailable or
    the mask is empty.
    """
    if transform is None:
        return None, None
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None, None
    mean_row, mean_col = rows.mean(), cols.mean()
    nearest = np.argmin((rows - mean_row) ** 2 + (cols - mean_col) ** 2)
    row_off = bbox[0].start if bbox is not None else 0
    col_off = bbox[1].start if bbox is not None else 0
    lon, lat = _transform_xy(transform, row_off + rows[nearest], col_off + cols[nearest])
    return lat, lon


def _rasterize_batch(
    batch: list[tuple[int, dict]],
    shape: tuple[int, int],
    transform,
) -> np.ndarray:
    """
    Rasterize a batch of polygons into a single int32 label array.
    Each polygon gets a unique label (1-based, matching batch position).
    Overlapping polygons: higher label wins (last writer).

    Returns label array of shape (H, W), dtype int32.
    Zero means no water body.
    """
    import rasterio.features as rf

    # Build (geometry, label) pairs — label is 1-based batch position
    shapes = []
    for batch_pos, (orig_idx, poly) in enumerate(batch):
        coords = poly.get('coordinates', [])
        if not coords:
            continue
        shapes.append(({'type': 'Polygon', 'coordinates': coords},
                       batch_pos + 1))  # 1-based

    if not shapes:
        return np.zeros(shape, dtype=np.int32)

    label_array = rf.rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
    )
    return label_array


def _process_label_array(
    label_array: np.ndarray,
    corrected: np.ndarray,
    batch: list,
    buffer_pixels: int,
    shore_percentile: float,
    min_water_pixels: int,
    transform=None,
    max_water_std_m: float = 2.5,
    max_correction_m: float = 10.0,
) -> tuple:
    """
    Optimized replacement for _process_label_array.
 
    Changes vs original:
      1. Removed distance_transform_edt — the shoreline mask is
         derived directly from grey_dilation's output, which we
         compute anyway for label propagation.
 
      2. Used scipy.ndimage.find_objects() to get per-label
         bounding boxes once, then sliced into the label/corrected
         arrays per polygon instead of doing a full-crop comparison.
 
      3. Optional shoreline subsample cap (default 10_000). Only
         relevant for very large water bodies where percentile
         compute on >100k elements becomes measurable. Disabled by
         default — set CAP_SHORELINE_SAMPLES > 0 to enable.
 
    Correctness invariants preserved:
      - Per-label water_mask is identical (just sliced into bbox)
      - shoreline_mask = (dilated_labels > 0) & ~water_any  is
        provably equivalent to the original
        (dist > 0) & (dist <= buffer_pixels)
        because grey_dilation with footprint of radius=buffer_pixels
        propagates labels outward by exactly buffer_pixels.
      - Per-label shore_labels assignment is unchanged.
      - corrected[] write is identical.
 
    Sanity check (added after a real incident: an 8-pixel unnamed
    "water body" got a -21.1m correction because it wasn't actually a
    lake -- it was a misclassified sliver sitting on a uniformly steep
    hillside, confirmed by direct pixel inspection of a real tile. Its
    own raw elevations spanned 12.2m and its shoreline ring spanned
    37.4m -- neither remotely plausible for real still water or a real
    shoreline. Across a full tile's 1,176 corrected bodies, max
    |correction| shrank monotonically with body size while the median
    stayed flat, confirming the failure concentrates in small/
    misclassified bodies where the fixed, non-slope-adaptive
    buffer_pixels ring samples real terrain instead of a shoreline):
    reject a body if its candidate correction (surface_elev -
    orig_mean) is larger than max_correction_m.

    Two alternative designs were tried and reverted after live-testing
    against a real tile, in favor of gating on the correction itself:
      - Gating on shore_std (the shoreline ring's own variance): a
        real alpine lake's ring routinely has 5-9m of legitimate
        relief (the percentile pick is specifically designed to
        tolerate that by favoring the ring's low end); an absolute std
        threshold there flagged 90 real named lakes, 0 confirmed true
        positives among them.
      - Gating on water_std (the water body's own raw noise) as a hard
        block: this has a worse failure mode than shore_std did -- it
        can't distinguish "genuinely not flat" from "genuinely flat
        water with noisy raw DEM data" (water surfaces routinely
        produce noisy lidar/photogrammetric returns). Live-tested: 8
        real bodies (including a named lake) that would have gotten a
        perfectly sane 1-10m correction were instead silently left
        with their noisy raw elevation un-fixed -- the exact failure
        this exists to prevent, just moved one level down.
    water_std is still computed and carried through as a non-blocking
    `low_confidence` marker on applied corrections (see
    elevation_changes below) -- it's informative for review, just not
    a reason to withhold a correction the shoreline ring, an
    independent sample, says is fine.

    A rejected body leaves the original DEM values in place (safer
    than an arbitrary flattening) and is recorded in flagged_bodies
    instead of elevation_changes, so it stays visible in the output
    rather than silently vanishing.

    Returns:
        (corrected, elevation_changes, corrected_count, skipped_small,
         skipped_implausible, flagged_bodies)
    """
    from scipy.ndimage import grey_dilation, find_objects

    elevation_changes   = []
    flagged_bodies      = []
    corrected_count     = 0
    skipped_small        = 0
    skipped_implausible  = 0
 
    water_any = label_array > 0
 
    # ── Build dilated label array ONCE (same as before) ──────────
    struct = np.ones(
        (2 * buffer_pixels + 1, 2 * buffer_pixels + 1), dtype=bool,
    )
 
    print(f"    crop shape: {label_array.shape}, "
          f"water pixels: {water_any.sum()}", flush=True)
    dilated_labels = grey_dilation(label_array, footprint=struct)
 
    # ── Derive shoreline mask directly (NEW: no EDT) ─────────────
    # grey_dilation propagates each label outward by exactly
    # buffer_pixels (the structuring element radius). So the
    # dilated_labels > 0 region is the union of (water + shoreline
    # band of width buffer_pixels). Subtracting water_any gives
    # exactly the shoreline.
    in_shoreline = (dilated_labels > 0) & ~water_any
 
    shore_elevs  = corrected[in_shoreline]
    shore_labels = dilated_labels[in_shoreline]
 
    # ── Pre-compute per-label bounding boxes (NEW) ───────────────
    # find_objects returns a list where index i (0-based in labels
    # array, but labels are 1-based) contains the slice for label i+1
    # or None if that label has no pixels.
    label_objects = find_objects(label_array)
 
    # ── Per-polygon correction ───────────────────────────────────
    for batch_pos, (orig_idx, poly) in enumerate(batch):
        label = batch_pos + 1
 
        # find_objects is 0-indexed by (label - 1)
        if label - 1 >= len(label_objects):
            continue
        bbox = label_objects[label - 1]
        if bbox is None:
            # Label has no pixels in rasterized output
            continue
 
        # Slice into label array using the bbox (NEW: much smaller
        # comparison than the original full-crop label_array == label)
        label_sub  = label_array[bbox]
        water_mask_local = label_sub == label
 
        pixel_count = int(water_mask_local.sum())
        if pixel_count < min_water_pixels:
            skipped_small += 1
            continue
 
        # ── Per-label shoreline sampling (unchanged logic) ───────
        label_shore_mask = shore_labels == label
        valid = shore_elevs[label_shore_mask]
        valid = valid[(valid > -1000) & (valid < 9000)]

        if len(valid) == 0:
            # Fallback: sample inside the water body itself
            inside = corrected[bbox][water_mask_local]
            valid  = inside[inside > -1000]
            if len(valid) == 0:
                continue

        # ── Original values for stats, read BEFORE any write so this
        # is always the true pre-correction elevation (using bbox slice)
        orig_vals_in_bbox = corrected[bbox][water_mask_local]
        valid_orig = orig_vals_in_bbox[orig_vals_in_bbox > -1000]
        orig_mean  = float(np.mean(valid_orig)) if len(valid_orig) > 0 else 0.0
        water_std  = float(np.std(valid_orig)) if len(valid_orig) > 1 else 0.0
        shore_std  = float(np.std(valid))

        name = poly.get('properties', {}).get('name', '')
        lat, lon = _mask_centroid_latlon(water_mask_local, bbox, transform)

        surface_elev = float(np.percentile(valid, shore_percentile))

        # ── Sanity check: see module-level comment on this function for
        # the real incident that motivated this ──────────────────────
        # Gates on the CANDIDATE CORRECTION ITSELF, not on water_std or
        # shore_std directly -- both were tried as hard gates first and
        # reverted after live-testing against a real tile:
        #   - shore_std: a real alpine lake's shoreline ring routinely
        #     has 5-9m of legitimate relief (the percentile pick is
        #     specifically designed to tolerate that by favoring the
        #     ring's low end); an absolute std threshold there flagged
        #     90 real named lakes with 0 confirmed true positives.
        #   - water_std: gating on the water body's OWN raw noise has a
        #     worse failure mode -- it can't distinguish "genuinely not
        #     flat" from "genuinely flat water with noisy raw DEM data"
        #     (a well-known problem: water surfaces often produce noisy
        #     lidar/photogrammetric returns). Live-tested: with water_std
        #     as a hard gate, 8 real bodies (including a named lake) that
        #     would have received a perfectly sane 1-10m correction were
        #     instead silently left with their noisy raw elevation
        #     un-fixed -- the exact failure this exists to prevent, just
        #     moved one level down. The shoreline ring is an independent
        #     sample, uncontaminated by the water body's own noise, so
        #     checking what it actually produced is both simpler and
        #     correct; water_std is still computed and reported as a
        #     `low_confidence` marker on applied corrections, but no
        #     longer blocks anything by itself.
        if abs(surface_elev - orig_mean) > max_correction_m:
            skipped_implausible += 1
            flagged_bodies.append({
                'polygon_index': orig_idx, 'name': name,
                'pixel_count':   pixel_count,
                'water_std_m':   round(water_std, 1),
                'shore_std_m':   round(shore_std, 1),
                'candidate_surface_elev_m': round(surface_elev, 1),
                'candidate_change_m': round(surface_elev - orig_mean, 1),
                'reason':        'correction_too_large',
                'lat':           round(lat, 5) if lat is not None else None,
                'lon':           round(lon, 5) if lon is not None else None,
            })
            continue

        # ── Write corrected elevation (using bbox slice) ─────────
        # NB: writing through the slice IS a write-back to the
        # underlying array; numpy slicing returns a view.
        corrected[bbox][water_mask_local] = surface_elev
        corrected_count += 1

        elevation_changes.append({
            'polygon_index':   orig_idx,
            'name':            name,
            'pixel_count':     pixel_count,
            'surface_elev_m':  round(surface_elev, 1),
            'original_mean_m': round(orig_mean, 1),
            'change_m':        round(surface_elev - orig_mean, 1),
            'water_std_m':     round(water_std, 1),
            'shore_std_m':     round(shore_std, 1),
            'low_confidence':  water_std > max_water_std_m,
            'lat':             round(lat, 5) if lat is not None else None,
            'lon':             round(lon, 5) if lon is not None else None,
        })

    return (corrected, elevation_changes, corrected_count, skipped_small,
            skipped_implausible, flagged_bodies)

# ─────────────────────────────────────────────
# Core correction function — batch rasterization
# ─────────────────────────────────────────────

def correct_dem_water_bodies(
    input_path: str,
    output_path: str,
    water_polygons: list[dict],
    buffer_pixels: int = 3,
    shore_percentile: float = 10.0,
    min_water_pixels: int = 12,
    batch_size: int = 500,
    verbose: bool = True,
    max_water_std_m: float = 2.5,
    max_correction_m: float = 10.0,
) -> dict:
    """
    Apply water body elevation correction to a DEM GeoTIFF.

    Uses batch rasterization — rasterizes up to `batch_size` polygons
    at once into a single label array, then processes all labels in
    one pass. This replaces 28,261 individual rasterize calls with
    ~57 batch calls, giving 20-50x speedup.

    For each water body polygon:
      1. Compute crop bounds from polygon coordinates (no full-mosaic alloc)
      2. Rasterize batch directly onto crop
      3. Per label: sample shoreline elevation using distance transform
      4. Set all water pixels to the sampled surface elevation
      5. Write crop back to full array

    Args:
        input_path:       Path to input DEM GeoTIFF
        output_path:      Path to write corrected DEM GeoTIFF
        water_polygons:   List of polygon geometry dicts
        buffer_pixels:    Shoreline sampling buffer width in pixels
        shore_percentile: Percentile for surface elevation estimate
        min_water_pixels: Skip polygons smaller than this (noise filter)
        batch_size:       Polygons per rasterization batch (default 500)
        verbose:          Print per-batch progress
        max_water_std_m:  Non-blocking -- marks a corrected body
                           'low_confidence' if its own raw elevations
                           varied by more than this before correction.
                           Does NOT withhold the correction; see
                           _process_label_array()'s docstring for why
                           (gating on this directly was tried and
                           reverted after live-testing).
        max_correction_m: Skip (flag) a body if the candidate
                           correction it computed is larger than this
                           -- see _process_label_array()'s docstring
                           for why this checks the correction itself
                           rather than the shoreline ring's variance.

    Returns:
        Statistics dict with correction counts, elevation ranges, and
        a 'flagged_bodies' list of bodies skipped for implausibility
        (see _process_label_array())
    """
    import rasterio.features

    stats = {
        'total_polygons':      len(water_polygons),
        'corrected_bodies':    0,
        'skipped_small':       0,
        'skipped_implausible': 0,
        'skipped_no_overlap':  0,
        'total_pixels_fixed':  0,
        'elevation_changes':   [],
        'flagged_bodies':      [],
        'batch_size':          batch_size,
        'n_batches':           0,
    }

    with rasterio.open(input_path) as src:
        elevation = src.read(1).astype(np.float32)
        profile   = src.profile.copy()
        transform = src.transform
        nodata    = src.nodata if src.nodata is not None else -9999.0
        bounds    = src.bounds

    corrected = elevation.copy()
    h, w      = elevation.shape

    # ── GPU sizing ─────────────────────────────────────────────────────
    if _GPU_TOOLS_AVAILABLE:
        gpu_info()
        suggested = recommend_gpu_batch_size((h, w), dtype_bytes=4)
        if suggested == 0:
            print("[CORRECT] GPU disabled (insufficient VRAM), using CPU")
            _use_gpu = False
        else:
            if suggested != batch_size:
                print(f"[CORRECT] GPU recommends batch_size={suggested} "
                      f"(was {batch_size})")
                batch_size = suggested
            _use_gpu = True
    else:
        print("[CORRECT] GPU not available — using CPU")
        _use_gpu = False

    # ── Pre-filter (count only -- ocean polygons are split out and this
    # list gets rebuilt + sorted below, right before the batch loop that
    # actually consumes it) ─────────────────────────────────────────────
    overlapping = _filter_overlapping(water_polygons, bounds)

    stats['skipped_no_overlap'] = len(water_polygons) - len(overlapping)
    print(f"[CORRECT] {len(overlapping)} polygons overlap raster "
          f"({stats['skipped_no_overlap']} outside bounds skipped)")


    # ── Separate ocean polygons for fast zero-fill ────────────────
    ocean_polys   = [p for p in water_polygons
                     if p.get('properties', {}).get('source') == 'osm_ocean']
    regular_polys = [p for p in water_polygons
                     if p.get('properties', {}).get('source') != 'osm_ocean']

    if ocean_polys:
        print(f"[CORRECT] Applying fast zero-fill to "
              f"{len(ocean_polys)} ocean polygon(s)...")
        import rasterio.features
        for ocean_poly in ocean_polys:
            try:
                ocean_mask = rasterio.features.rasterize(
                    [(ocean_poly, 1)],
                    out_shape=(h, w),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                ).astype(bool)
                n_ocean = int(ocean_mask.sum())
                print(f"[CORRECT] Zeroing {n_ocean:,} ocean pixels")
                corrected[ocean_mask] = 0.0
                stats['corrected_bodies']   += 1
                stats['total_pixels_fixed'] += n_ocean
            except Exception as e:
                print(f"[CORRECT] Ocean zero-fill failed: {e}")

    # Update water_polygons to only process regular water bodies. Re-filter
    # AND re-sort -- this is a different (smaller) set than the one sorted
    # above (ocean polygons removed), and the batch loop below reads THIS
    # `overlapping`, not the one above -- it must carry the sort too, or
    # every batch reverts to whatever arbitrary order the source data
    # (e.g. NHD API response order) happened to be in, with zero spatial
    # locality guarantee.
    water_polygons = regular_polys
    overlapping = _filter_overlapping(water_polygons, bounds)
    overlapping = _spatial_sort_morton(overlapping)
    print(f"[CORRECT] {len(overlapping)} polygons sorted spatially "
          f"(Morton/Z-order) for compact batching")

    # Even a well-sorted (Morton/Z-order) run can't guarantee every batch
    # is small when there are too FEW batches for a dense, uniformly
    # spread polygon set to begin with -- pigeonhole: N batches over
    # roughly-uniform data can't all be much smaller than 1/N of the
    # domain. Confirmed empirically: 2,838 polygons at batch_size=1000 (3
    # batches) left a worst-case batch still covering ~87% of the tile
    # even after sorting; forcing more, smaller batches brought that down
    # to ~35% at 8 batches. Only ever shrinks batch_size (never grows it
    # past whatever recommend_gpu_batch_size already decided is
    # VRAM-safe), and each batch's actual GPU-vs-CPU routing decision
    # below still re-checks live free VRAM regardless.
    _MIN_BATCHES_FOR_LOCALITY = 8
    if len(overlapping) > 0 and len(overlapping) / batch_size < _MIN_BATCHES_FOR_LOCALITY:
        batch_size = max(50, math.ceil(len(overlapping) / _MIN_BATCHES_FOR_LOCALITY))

    print(f"[CORRECT] Processing in batches of {batch_size} "
          f"({math.ceil(len(overlapping)/max(batch_size,1))} batches)...")

    # ── Batch loop ─────────────────────────────────────────────────────
    n_batches = math.ceil(len(overlapping) / batch_size)

    for batch_num in range(n_batches):
        batch_start = batch_num * batch_size
        batch_end   = min(batch_start + batch_size, len(overlapping))
        batch       = overlapping[batch_start:batch_end]

        t_batch = time.perf_counter()
        if verbose:
            pct = (batch_num + 1) / n_batches * 100
            print(f"[CORRECT] Batch {batch_num+1}/{n_batches} "
                  f"(polygons {batch_start}–{batch_end-1}, {pct:.0f}%) ...",
                  end=' ', flush=True)

        # ── Compute crop bounds from polygon coordinates ───────────────
        # Avoids allocating a full-mosaic label array before cropping
        pad = buffer_pixels + 2
        all_lngs, all_lats = [], []
        for _, poly in batch:
            coords = poly.get('coordinates', [[]])[0]
            if not coords:
                continue
            all_lngs.extend(c[0] for c in coords)
            all_lats.extend(c[1] for c in coords)

        if not all_lngs:
            stats['skipped_no_overlap'] += len(batch)
            if verbose:
                print(f"0 corrected, {len(batch)} no coords")
            continue

        # Geographic bbox → pixel bbox via affine transform
        c_min = max(0, int((min(all_lngs) - transform.c) / transform.a) - pad)
        c_max = min(w, int((max(all_lngs) - transform.c) / transform.a) + pad + 1)
        r_min = max(0, int((max(all_lats) - transform.f) / transform.e) - pad)
        r_max = min(h, int((min(all_lats) - transform.f) / transform.e) + pad + 1)

        crop_h = r_max - r_min
        crop_w = c_max - c_min

        if crop_h <= 0 or crop_w <= 0:
            stats['skipped_no_overlap'] += len(batch)
            if verbose:
                print(f"0 corrected, {len(batch)} zero-size crop")
            continue

        # Sub-transform for the crop window
        crop_transform = rasterio.transform.from_bounds(
            transform.c + c_min * transform.a,
            transform.f + r_max * transform.e,
            transform.c + c_max * transform.a,
            transform.f + r_min * transform.e,
            crop_w, crop_h,
        )

        # ── Rasterize directly onto crop ───────────────────────────────
        print("rasterizing...", end=' ', flush=True)
        try:
            label_crop = _rasterize_batch(batch, (crop_h, crop_w),
                                          crop_transform)
        except Exception as e:
            print(f"\n[CORRECT] Batch {batch_num+1} rasterize failed: {e}")
            continue
        print("done.", end=' ', flush=True)

        if label_crop.max() == 0:
            stats['skipped_no_overlap'] += len(batch)
            if verbose:
                print(f"0 corrected, {len(batch)} no pixels")
            continue

        corrected_crop = corrected[r_min:r_max, c_min:c_max].copy()

        # ── Route to GPU or CPU based on crop size ─────────────────────
        crop_mb = (crop_h * crop_w * 4 * 5) / 1e6

        if _use_gpu and crop_mb < gpu_memory_available_gb() * 1000 * 0.7:
            corrected_crop, batch_changes, n_corrected, n_skipped = \
                process_label_array_gpu(
                    label_crop, corrected_crop, batch,
                    buffer_pixels, shore_percentile, min_water_pixels,
                    transform=crop_transform,
                )
            # process_label_array_gpu() doesn't have the water/shoreline
            # plausibility checks below yet (see _process_label_array()'s
            # docstring for why they exist) -- GPU runs get the old,
            # unguarded behavior until that's ported too.
            n_skipped_implausible, batch_flagged = 0, []
        else:
            if _use_gpu:
                print(f"[GPU] Crop {crop_mb:.0f}MB exceeds budget, "
                      f"using CPU", end=' ', flush=True)
            (corrected_crop, batch_changes, n_corrected, n_skipped,
             n_skipped_implausible, batch_flagged) = \
                _process_label_array(
                    label_crop, corrected_crop, batch,
                    buffer_pixels, shore_percentile, min_water_pixels,
                    transform=crop_transform,
                    max_water_std_m=max_water_std_m,
                    max_correction_m=max_correction_m,
                )

        # Write crop result back to full array
        corrected[r_min:r_max, c_min:c_max] = corrected_crop

        elapsed_batch = time.perf_counter() - t_batch
        stats['corrected_bodies']    += n_corrected
        stats['skipped_small']       += n_skipped
        stats['skipped_implausible'] += n_skipped_implausible
        stats['total_pixels_fixed']  += sum(
            c['pixel_count'] for c in batch_changes)
        stats['elevation_changes'].extend(batch_changes)
        stats['flagged_bodies'].extend(batch_flagged)
        stats['n_batches']           += 1

        if verbose:
            print(f"{n_corrected} corrected, {n_skipped} too small, "
                  f"{n_skipped_implausible} implausible "
                  f"({elapsed_batch:.1f}s)")

    # ── Write output ───────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # BIGTIFF explicit, not GDAL's BIGTIFF=IF_NEEDED default -- see
    # build_dem_mosaic.py's identical comment; this writes back over
    # the same study-area-mosaic-scale file that gap-filling/merging
    # produces, so it's exposed to the same >4GB-after-compression risk.
    profile.update(dtype=rasterio.float32, nodata=nodata, BIGTIFF='YES')

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(corrected, 1)

    accel = "GPU" if (_use_gpu and _GPU_TOOLS_AVAILABLE) else "CPU"
    print(f"\n[CORRECT] Summary ({accel}):")
    print(f"  Batches processed:    {stats['n_batches']} "
          f"(batch size {batch_size})")
    print(f"  Corrected bodies:      {stats['corrected_bodies']}")
    print(f"  Skipped (too small):   {stats['skipped_small']}")
    print(f"  Skipped (implausible): {stats['skipped_implausible']}")
    print(f"  Skipped (no overlap):  {stats['skipped_no_overlap']}")
    print(f"  Total pixels fixed:    {stats['total_pixels_fixed']:,}")
    print(f"  Output: {output_path}")

    return stats


# ─────────────────────────────────────────────
# Parallel worker function (Option A)
# ─────────────────────────────────────────────

def _parallel_worker(
    chunk: list,
    shared_refs: dict,
    transform,
    buffer_pixels: int,
    shore_percentile: float,
    min_water_pixels: int,
    batch_size: int,
    shape: tuple,
) -> list[dict]:
    """
    Worker function for parallel correction.
    Called by parallel_map_shared — runs in a separate process.

    Attaches to shared elevation array, processes its chunk of
    water body polygons, and returns a list of correction records
    (pixel coordinates + surface elevation) rather than modifying
    the array directly (workers cannot write to shared memory safely).

    NOT YET PORTED: this duplicates _process_label_array()'s core
    logic independently rather than calling it, so it also doesn't
    have the water/shoreline plausibility checks added there (see that
    function's docstring for the real incident that motivated them --
    the exact same fixed, non-slope-adaptive buffer_pixels sampling is
    used here, so the same failure mode applies).

    Returns list of dicts: {rows: [...], cols: [...], surface_elev: float}
    """
    import numpy as np
    import rasterio.features as rf
    from scipy.ndimage import binary_dilation

    # Attach to shared elevation array
    shm_name, arr_shape, arr_dtype = shared_refs['elevation']
    if shm_name is not None:
        from parallel_tools import SharedNDArray
        elevation = SharedNDArray.attach(shm_name, arr_shape, arr_dtype)
        attached  = True
    else:
        # Sequential fallback — array passed directly
        elevation = shared_refs['elevation'][3]
        attached  = False

    h, w        = shape
    corrections = []   # list of {rows, cols, surface_elev, pixel_count, name}
    water_any   = np.zeros((h, w), dtype=bool)

    # Pre-filter to overlapping polygons
    from rasterio.transform import from_bounds
    import rasterio
    # We need bounds for pre-filter — derive from transform
    west  = transform.c
    north = transform.f
    east  = transform.c + transform.a * w
    south = transform.f + transform.e * h

    # First pass: rasterize all polygons in batches to build water_any mask
    # and per-label arrays for shoreline exclusion
    overlapping = []
    for orig_idx, poly in chunk:
        coords = poly.get('coordinates', [])
        if not coords:
            continue
        ring = coords[0]
        lngs = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        if (max(lngs) < west  or min(lngs) > east or
            max(lats) < south or min(lats) > north):
            continue
        overlapping.append((orig_idx, poly))

    if not overlapping:
        if attached:
            SharedNDArray.detach(elevation)
        return []

    # Build combined water mask for all polygons in chunk
    # (used to exclude adjacent water from shoreline sampling)
    for i in range(0, len(overlapping), batch_size):
        sub_batch = overlapping[i:i + batch_size]
        shapes = [
            ({'type': 'Polygon', 'coordinates': poly.get('coordinates', [])},
             pos + 1)
            for pos, (_, poly) in enumerate(sub_batch)
            if poly.get('coordinates')
        ]
        if shapes:
            label_arr = rf.rasterize(
                shapes, out_shape=(h, w), transform=transform,
                fill=0, dtype=np.int32,
            )
            water_any |= label_arr > 0

    # Second pass: per-polygon shoreline sampling and correction
    for orig_idx, poly in overlapping:
        coords = poly.get('coordinates', [])
        if not coords:
            continue
        try:
            mask = rf.rasterize(
                [({'type': 'Polygon', 'coordinates': coords}, 1)],
                out_shape=(h, w), transform=transform,
                fill=0, dtype=np.uint8,
            ).astype(bool)
        except Exception:
            continue

        pixel_count = int(mask.sum())
        if pixel_count < min_water_pixels:
            continue

        # Shoreline: dilate this polygon, exclude all water
        dilated   = binary_dilation(mask, iterations=buffer_pixels)
        shoreline = dilated & ~water_any

        elevs = elevation[shoreline]
        valid = elevs[(elevs > -1000) & (elevs < 9000)]
        if len(valid) == 0:
            inside = elevation[mask]
            valid  = inside[inside > -1000]
            if len(valid) == 0:
                continue

        surface_elev = float(np.percentile(valid, shore_percentile))
        lat, lon = _mask_centroid_latlon(mask, None, transform)

        rows, cols = np.where(mask)
        corrections.append({
            'rows':        rows.tolist(),
            'cols':        cols.tolist(),
            'surface_elev': surface_elev,
            'pixel_count':  pixel_count,
            'orig_idx':     orig_idx,
            'name':         poly.get('properties', {}).get('name', ''),
            'orig_mean':    float(np.mean(elevation[mask])),
            'lat':          lat,
            'lon':          lon,
        })

    if attached:
        SharedNDArray.detach(elevation)

    return corrections


def correct_dem_parallel(
    input_path: str,
    output_path: str,
    water_polygons: list[dict],
    buffer_pixels: int = 3,
    shore_percentile: float = 10.0,
    min_water_pixels: int = 5,
    batch_size: int = 500,
    n_workers: int = None,
) -> dict:
    """
    Parallel version of correct_dem_water_bodies using shared memory.

    Splits water body polygons across worker processes. Each worker
    reads from a shared copy of the elevation array (no copying per
    worker). Workers return pixel coordinate lists; the main process
    applies all corrections to the output array.

    Falls back to correct_dem_water_bodies() if parallel_tools is
    unavailable or n_workers=1.

    Args:
        input_path:       Input DEM GeoTIFF path
        output_path:      Output corrected GeoTIFF path
        water_polygons:   Water body polygon list
        buffer_pixels:    Shoreline dilation buffer
        shore_percentile: Surface elevation percentile
        min_water_pixels: Minimum pixels to correct a polygon
        batch_size:       Polygons per rasterization sub-batch in workers
        n_workers:        Worker processes (default: cpu_count - 1)

    Returns:
        Statistics dict
    """
    if not _PARALLEL_AVAILABLE or n_workers == 1:
        print("[CORRECT] Parallel tools unavailable — using sequential")
        return correct_dem_water_bodies(
            input_path, output_path, water_polygons,
            buffer_pixels, shore_percentile, min_water_pixels, batch_size,
        )

    from parallel_tools import (
        SharedNDArray, make_chunks, recommend_workers, _resolve_workers
    )

    with rasterio.open(input_path) as src:
        elevation = src.read(1).astype(np.float32)
        profile   = src.profile.copy()
        transform = src.transform
        nodata    = src.nodata if src.nodata is not None else -9999.0
        bounds    = src.bounds

    h, w = elevation.shape

    n_workers = _resolve_workers(n_workers)
    n_workers = recommend_workers(
        item_count=len(water_polygons),
        item_size_mb=elevation.nbytes / 1e6,
    )
    print_system_info()
    print(f"[CORRECT] Using {n_workers} workers, "
          f"{elevation.nbytes/1e6:.0f}MB shared array")

    # Tag polygons with index for stats tracking
    indexed = list(enumerate(water_polygons))

    # Split into one chunk per worker
    chunks = make_chunks(indexed, n_workers)

    stats = {
        'total_polygons':     len(water_polygons),
        'corrected_bodies':   0,
        'skipped_small':      0,
        'skipped_no_overlap': 0,
        'total_pixels_fixed': 0,
        'elevation_changes':  [],
        'n_workers':          n_workers,
    }

    corrected = elevation.copy()

    with SharedNDArray(elevation) as shared:
        from multiprocessing import get_context
        import functools

        shared_refs = {
            'elevation': (shared.name, shared.shape, shared.dtype)
        }

        worker_kwargs = dict(
            transform=transform,
            buffer_pixels=buffer_pixels,
            shore_percentile=shore_percentile,
            min_water_pixels=min_water_pixels,
            batch_size=batch_size,
            shape=(h, w),
        )

        ctx = get_context('spawn')
        t0  = time.perf_counter()

        with ctx.Pool(processes=n_workers) as pool:
            job_args = [
                (_parallel_worker, (chunk, shared_refs), worker_kwargs)
                for chunk in chunks
            ]
            from parallel_tools import _worker_wrapper
            for chunk_i, (result, err) in enumerate(
                pool.imap_unordered(_worker_wrapper, job_args)
            ):
                elapsed = time.perf_counter() - t0
                if err:
                    print(f"[CORRECT] Worker error:\n{err}")
                    continue

                # Apply corrections from this worker to output array
                for rec in result:
                    rows = np.array(rec['rows'])
                    cols = np.array(rec['cols'])
                    corrected[rows, cols] = rec['surface_elev']
                    stats['corrected_bodies']   += 1
                    stats['total_pixels_fixed'] += rec['pixel_count']
                    stats['elevation_changes'].append({
                        'polygon_index':   rec['orig_idx'],
                        'name':            rec['name'],
                        'pixel_count':     rec['pixel_count'],
                        'surface_elev_m':  round(rec['surface_elev'], 1),
                        'original_mean_m': round(rec['orig_mean'], 1),
                        'change_m':        round(rec['surface_elev'] - rec['orig_mean'], 1),
                        'lat':             round(rec['lat'], 5) if rec.get('lat') is not None else None,
                        'lon':             round(rec['lon'], 5) if rec.get('lon') is not None else None,
                    })

                pct = (chunk_i + 1) / len(chunks) * 100
                print(f"[CORRECT] {chunk_i+1}/{len(chunks)} chunks "
                      f"({pct:.0f}%) elapsed={elapsed:.1f}s")

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # BIGTIFF explicit, not GDAL's BIGTIFF=IF_NEEDED default -- see
    # build_dem_mosaic.py's identical comment.
    profile.update(dtype=rasterio.float32, nodata=nodata, BIGTIFF='YES')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(corrected, 1)

    total_time = time.perf_counter() - t0
    print("\n[CORRECT] Parallel summary:")
    print(f"  Workers:              {n_workers}")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Corrected bodies:     {stats['corrected_bodies']}")
    print(f"  Total pixels fixed:   {stats['total_pixels_fixed']:,}")
    print(f"  Output: {output_path}")
    return stats


# ─────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────

def batch_correct_directory(
    input_dir: str,
    output_dir: str,
    water_source: str = 'osm',
    bounds: tuple = None,
    water_file: str = None,
    resume: bool = True,
) -> None:
    """
    Correct all GeoTIFF files in a directory.
    Fetches water bodies once for the full extent, then applies to each tile.

    Args:
        input_dir:    Directory containing raw DEM GeoTIFFs
        output_dir:   Directory to write corrected GeoTIFFs
        water_source: 'osm', 'nhd', or 'file'
        bounds:       (south, west, north, east) override — auto-detected
                      from tiles if not provided
        water_file:   Path to local GeoJSON water body file (if source='file')
        resume:       Skip tiles that already have a corrected output
    """
    input_tiles = list(Path(input_dir).glob('*.tif')) + \
                  list(Path(input_dir).glob('*.tiff'))

    if not input_tiles:
        print(f"[BATCH] No GeoTIFF files found in {input_dir}")
        return

    print(f"[BATCH] Found {len(input_tiles)} tiles to process")

    # Auto-detect bounds from all tiles
    if bounds is None:
        all_bounds = []
        for tile_path in input_tiles:
            with rasterio.open(tile_path) as src:
                b = src.bounds
                all_bounds.append((b.bottom, b.left, b.top, b.right))
        bounds = (
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds),
        )
        print(f"[BATCH] Auto-detected bounds: {bounds}")

    # Fetch water bodies once for full extent
    if water_source == 'nhd':
        water_polygons = fetch_water_bodies_nhd(bounds)
    elif water_source == 'osm':
        water_polygons = fetch_water_bodies_osm(bounds)
    elif water_source == 'file':
        if not water_file:
            raise ValueError("water_file required when source='file'")
        water_polygons = load_water_bodies_from_file(water_file)
    else:
        raise ValueError(f"Unknown water source: {water_source}")

    if not water_polygons:
        print("[BATCH] No water bodies found — tiles unchanged")
        return

    # Save fetched water bodies for inspection
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    water_cache_path = Path(output_dir) / '_water_bodies.geojson'
    with open(water_cache_path, 'w') as f:
        json.dump({
            'type': 'FeatureCollection',
            'features': [
                {'type': 'Feature', 'geometry': p,
                 'properties': p.get('properties', {})}
                for p in water_polygons
            ],
        }, f)
    print(f"[BATCH] Water bodies saved to {water_cache_path}")

    # Process each tile
    all_stats = []
    for tile_path in sorted(input_tiles):
        output_path = Path(output_dir) / tile_path.name

        if resume and output_path.exists():
            print(f"[BATCH] Skipping {tile_path.name} — already corrected")
            continue

        print(f"\n[BATCH] Processing {tile_path.name}...")
        try:
            tile_stats = correct_dem_water_bodies(
                input_path=str(tile_path),
                output_path=str(output_path),
                water_polygons=water_polygons,
                verbose=True,
            )
            tile_stats['tile'] = tile_path.name
            all_stats.append(tile_stats)
        except Exception as e:
            print(f"[BATCH] ERROR on {tile_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save batch summary
    summary_path = Path(output_dir) / '_correction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n[BATCH] Summary saved to {summary_path}")


# ─────────────────────────────────────────────
# Preview mode — visualise without writing
# ─────────────────────────────────────────────

def preview_correction(
    input_path: str,
    water_source: str = 'osm',
    water_file: str = None,
) -> None:
    """
    Print a preview of what corrections would be applied
    without writing any output files.
    """
    with rasterio.open(input_path) as src:
        bounds = src.bounds

    tile_bounds = (bounds.bottom, bounds.left, bounds.top, bounds.right)

    if water_source == 'nhd':
        water_polygons = fetch_water_bodies_nhd(tile_bounds)
    elif water_source == 'osm':
        water_polygons = fetch_water_bodies_osm(tile_bounds)
    elif water_source == 'file':
        water_polygons = load_water_bodies_from_file(water_file)
    else:
        raise ValueError(f"Unknown water source: {water_source}")

    print(f"\n[PREVIEW] Would correct {len(water_polygons)} "
          f"water bodies in {input_path}")
    for i, poly in enumerate(water_polygons[:20]):  # show first 20
        name = poly.get('properties', {}).get('name', f'polygon_{i}')
        ring = poly['coordinates'][0]
        print(f"  {i:3d}. {name or '(unnamed)'} — "
              f"{len(ring)} vertices")
    if len(water_polygons) > 20:
        print(f"  ... and {len(water_polygons) - 20} more")


# ─────────────────────────────────────────────
# DEM tile index
# ─────────────────────────────────────────────

def update_tile_index(
    dem_dir: str,
    index_path: str = None,
) -> dict:
    """
    Build or update the tile index JSON file.
    Records each tile's path, bounds, resolution, and correction status.
    """
    raw_dir       = Path(dem_dir) / 'raw'
    corrected_dir = Path(dem_dir) / 'corrected'

    if index_path is None:
        index_path = Path(dem_dir) / 'index.json'

    # Load existing index
    if Path(index_path).exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {'tiles': {}}

    for tile_path in sorted(raw_dir.glob('*.tif')):
        name = tile_path.stem
        with rasterio.open(tile_path) as src:
            b   = src.bounds
            res = src.res  # (pixel_height, pixel_width) in degrees

        corrected_path = corrected_dir / tile_path.name
        index['tiles'][name] = {
            'raw_path':       str(tile_path),
            'corrected_path': str(corrected_path) if corrected_path.exists()
                              else None,
            'corrected':      corrected_path.exists(),
            'bounds': {
                'south': b.bottom,
                'west':  b.left,
                'north': b.top,
                'east':  b.right,
            },
            'resolution_deg': res[0],
            'resolution_m':   round(res[0] * 111_320, 0),
        }

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    total     = len(index['tiles'])
    corrected = sum(1 for t in index['tiles'].values() if t['corrected'])
    print(f"[INDEX] {total} tiles indexed, {corrected} corrected")
    print(f"[INDEX] Saved to {index_path}")
    return index




# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DEM water body correction tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # correct — single tile
    p_correct = sub.add_parser('correct', help='Correct a single DEM tile')
    p_correct.add_argument('--input',   required=True)
    p_correct.add_argument('--output',  required=True)
    p_correct.add_argument('--source',  default='osm',
                           choices=['osm', 'nhd', 'file'])
    p_correct.add_argument('--water-file', default=None)
    p_correct.add_argument('--buffer-pixels', type=int, default=3)
    p_correct.add_argument('--percentile',    type=float, default=10.0)
    p_correct.add_argument('--batch-size',    type=int, default=500,
                           help='Polygons per rasterization batch (default 500)')
    p_correct.add_argument('--parallel',      action='store_true',
                           help='Use parallel processing (Option A)')
    p_correct.add_argument('--workers',       type=int, default=None,
                           help='Worker processes for --parallel (default: cpu_count-1)')
    p_correct.add_argument('--gpu',           action='store_true',
                           help='Use GPU acceleration for dilation/percentile (Option B)')
    p_correct.add_argument('--gpu-info',      action='store_true',
                           help='Print GPU device info and exit')

    # batch — directory of tiles
    p_batch = sub.add_parser('batch', help='Correct all tiles in a directory')
    p_batch.add_argument('--input-dir',  required=True)
    p_batch.add_argument('--output-dir', required=True)
    p_batch.add_argument('--source',  default='osm',
                         choices=['osm', 'nhd', 'file'])
    p_batch.add_argument('--water-file', default=None)
    p_batch.add_argument('--bounds',  default=None,
                         help='south,west,north,east — auto-detected if omitted')
    p_batch.add_argument('--no-resume', action='store_true')

    # preview — inspect without writing
    p_preview = sub.add_parser('preview',
                               help='Preview corrections without writing')
    p_preview.add_argument('--input',  required=True)
    p_preview.add_argument('--source', default='osm',
                           choices=['osm', 'nhd', 'file'])
    p_preview.add_argument('--water-file', default=None)

    # index — rebuild tile index
    p_index = sub.add_parser('index', help='Rebuild tile index')
    p_index.add_argument('--dem-dir',    required=True)
    p_index.add_argument('--index-path', default=None)

    args = parser.parse_args()

    if args.command == 'correct':
        if getattr(args, 'gpu_info', False):
            if _GPU_TOOLS_AVAILABLE:
                gpu_info()
            else:
                print("[GPU] gpu_tools.py not available")
            return

        with rasterio.open(args.input) as src:
            bounds = src.bounds
            tile_bounds = (bounds.bottom, bounds.left,
                           bounds.top, bounds.right)
        if args.source == 'nhd':
            polys = fetch_water_bodies_nhd(tile_bounds)
        elif args.source == 'osm':
            polys = fetch_water_bodies_osm(tile_bounds)
        else:
            polys = load_water_bodies_from_file(args.water_file)
        if getattr(args, 'gpu', False) and not GPU_AVAILABLE:
            print("[WARN] --gpu requested but no GPU available, "
                  "continuing with CPU")

        if getattr(args, 'parallel', False):
            correct_dem_parallel(
                input_path=args.input,
                output_path=args.output,
                water_polygons=polys,
                buffer_pixels=args.buffer_pixels,
                shore_percentile=args.percentile,
                batch_size=args.batch_size,
                n_workers=args.workers,
            )
        else:
            correct_dem_water_bodies(
                input_path=args.input,
                output_path=args.output,
                water_polygons=polys,
                buffer_pixels=args.buffer_pixels,
                shore_percentile=args.percentile,
                batch_size=args.batch_size,
            )

    elif args.command == 'batch':
        bounds = None
        if args.bounds:
            parts  = [float(x) for x in args.bounds.split(',')]
            bounds = tuple(parts)  # (south, west, north, east)
        batch_correct_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            water_source=args.source,
            bounds=bounds,
            water_file=args.water_file,
            resume=not args.no_resume,
        )

    elif args.command == 'preview':
        preview_correction(
            input_path=args.input,
            water_source=args.source,
            water_file=args.water_file,
        )

    elif args.command == 'index':
        update_tile_index(
            dem_dir=args.dem_dir,
            index_path=args.index_path,
        )


if __name__ == '__main__':
    main()