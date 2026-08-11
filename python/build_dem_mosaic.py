"""
build_dem_mosaic.py
━━━━━━━━━━━━━━━━━━
Downloads and assembles a DEM mosaic for a study area.

Supports both GLO30 (global, 30m) and 3DEP (US only, 10m) sources.
Output is a single merged GeoTIFF with water correction applied,
ready for use as the study area DEM in a downstream compute pipeline.

Usage:
    python build_dem_mosaic.py N44W113 N47W109
    python build_dem_mosaic.py N44W113 N47W109 --source 3DEP_10
    python build_dem_mosaic.py N44W113 N47W109 --source GLO30
    python build_dem_mosaic.py --bounds 44 -113 47 -109 --output my_area.tif
"""

import argparse
import math
import time
from pathlib import Path

import rasterio
from rasterio.merge import merge

from tile_id import bounds_from_tile_corners, MAX_MOSAIC_TILES

# ─────────────────────────────────────────────
# Study area helper
# ─────────────────────────────────────────────

def mosaic_area_between(corner1_id, corner2_id, name=None,
                         max_tiles=MAX_MOSAIC_TILES, allow_large=False):
    """
    Build an area dict (south/west/north/east/dem_path) from two
    opposite 1x1-degree tile corners, e.g. "N44W113" and "N47W109" --
    any pairing/order. See bounds_from_tile_corners() in tile_id.py
    for the exact corner rules and the size guard.

    `name` defaults to "<corner1>_<corner2>" and only affects the
    returned dem_path; build_mosaic() itself doesn't care what
    output_path you pass it.
    """
    b = bounds_from_tile_corners(corner1_id, corner2_id,
                                  max_tiles=max_tiles, allow_large=allow_large)
    if name is None:
        name = f"{corner1_id}_{corner2_id}"
    return {
        'south': b['south'], 'west': b['west'],
        'north': b['north'], 'east': b['east'],
        'dem_path': f'data/dem/{name}/{name}_corrected.tif',
    }


def build_mosaic_between(corner1_id, corner2_id, output_path=None, name=None,
                          source='auto', apply_water_correction=True,
                          max_tiles=MAX_MOSAIC_TILES, allow_large=False):
    """
    Build a mosaic spanning two opposite 1x1-degree tile corners in
    one call. Composes mosaic_area_between() + build_mosaic() with
    buffer_deg=0 -- NOT build_mosaic()'s own default (0.1). See the
    CLI's main() for why: that default exists for raw fractional-degree
    bounds, where the edges can fall mid-tile; corner-tile bounds are
    already exactly tile-aligned, so the same margin only pushes the
    request across an extra whole tile on every side.
    """
    area = mosaic_area_between(corner1_id, corner2_id, name=name,
                                max_tiles=max_tiles, allow_large=allow_large)
    if output_path is None:
        output_path = area['dem_path']
    return build_mosaic(
        south=area['south'], west=area['west'],
        north=area['north'], east=area['east'],
        output_path=output_path, source=source,
        apply_water_correction=apply_water_correction,
        buffer_deg=0,
    )


def get_required_degree_tiles(
    south: float, west: float,
    north: float, east: float,
) -> list:
    """
    Return list of (lat_floor, lng_floor) for all 1°×1° tiles
    that overlap the bounding box.
    """
    tiles = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lng in range(math.floor(west), math.ceil(east)):
            tiles.append((lat, lng))
    return tiles


def build_mosaic(
    south: float, west: float,
    north: float, east: float,
    output_path: str,
    source: str = 'auto',
    apply_water_correction: bool = True,
    buffer_deg: float = 0.1,
) -> dict:
    """
    Download all required tiles and merge into a single mosaic GeoTIFF.

    Args:
        south, west, north, east: Study area bounds in decimal degrees
        output_path:  Output GeoTIFF path
        source:       'auto', 'GLO30', '3DEP_10'
        apply_water_correction: Zero out suspect ocean/water pixels
        buffer_deg:   Extra buffer beyond study area bounds (degrees)
    """
    from elevation import download_tile, _is_in_3dep_coverage

    # Expand bounds by buffer
    s = south - buffer_deg
    w = west  - buffer_deg
    n = north + buffer_deg
    e = east  + buffer_deg

    tiles = get_required_degree_tiles(s, w, n, e)
    print(f"[MOSAIC] Study area: {south:.2f}°N {west:.2f}°E to "
          f"{north:.2f}°N {east:.2f}°E")
    print(f"[MOSAIC] Source: {source}")
    print(f"[MOSAIC] Tiles required: {len(tiles)}")
    print(f"[MOSAIC] Output: {output_path}")

    t0 = time.time()

    # ── Download tiles ────────────────────────────────────────────
    tile_paths = []
    for i, (lat_floor, lng_floor) in enumerate(tiles):
        tile_center_lat = lat_floor + 0.5
        tile_center_lng = lng_floor + 0.5

        # Auto-select source per tile — some tiles may be outside 3DEP
        tile_source = source
        if source == 'auto':
            tile_source = ('3DEP_10'
                           if _is_in_3dep_coverage(
                               tile_center_lat, tile_center_lng)
                           else 'GLO30')

        print(f"[MOSAIC] Tile {i+1}/{len(tiles)}: "
              f"({lat_floor},{lng_floor}) source={tile_source}")
        try:
            path = download_tile(
                tile_center_lat, tile_center_lng,
                source=tile_source,
            )
            tile_paths.append(path)
        except Exception as e:
            print(f"[MOSAIC] Warning: tile ({lat_floor},{lng_floor}) "
                  f"failed: {e}")

    if not tile_paths:
        raise ValueError("[MOSAIC] No tiles downloaded successfully")

    print(f"\n[MOSAIC] Downloaded {len(tile_paths)}/{len(tiles)} tiles "
          f"in {time.time()-t0:.0f}s")

    # ── Merge + crop + write, windowed ──────────────────────────────
    # A single rasterio.merge.merge() call with dst_path set, rather
    # than the old three-array-copies approach (merge() into one
    # in-memory mosaic array -> write that into a MemoryFile -> read
    # it back for a rio_mask() crop). That old path peaked at ~3x the
    # final output size in RAM (whole-mosaic array + MemoryFile buffer
    # + cropped array) with nothing bounding it to the machine's actual
    # memory -- confirmed live: a 4x4-degree run's merge step alone
    # took 10.5 hours on this 15GB-RAM machine (vs. 22 minutes for the
    # equivalent step in the R port, which processes rasters in
    # disk-backed chunks internally) before a second run of the exact
    # same operation completed in 33 seconds, consistent with the first
    # hitting heavy swap thrashing.
    #
    # merge(..., dst_path=...) instead subdivides the output into
    # mem_limit-sized pixel windows internally, reads only the source
    # data each window needs, writes that window straight to disk, and
    # discards it before the next one -- never holding more than one
    # window's worth of data. bounds=(w, s, e, n) makes the same call
    # do the crop too (the buffered study-area bounds, matching what
    # the old separate rio_mask() step cropped to), so there's no
    # separate crop pass or intermediate array at all.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[MOSAIC] Merging {len(tile_paths)} tiles into {output_path} "
          f"(windowed, bounded memory)...")
    t_merge = time.time()

    open_datasets = []
    for path in tile_paths:
        try:
            open_datasets.append(rasterio.open(path))
        except Exception as e:
            print(f"[MOSAIC] Warning: could not open {path}: {e}")

    dst_kwds = dict(
        compress='deflate', predictor=2, tiled=True,
        blockxsize=512, blockysize=512,
        # Explicit, not GDAL's BIGTIFF=IF_NEEDED default: IF_NEEDED
        # predicts final size assuming compression will keep it under
        # 4GB and only upgrades if that prediction says otherwise --
        # real DEM terrain doesn't compress reliably enough for that
        # bet. Confirmed live: a real 4x4-degree, water-correction-off
        # mosaic (~4.3GB actual output) failed here with
        # "TIFFAppendToStrip: Maximum TIFF file size exceeded" the
        # first time this was tried without it.
        BIGTIFF='YES',
    )
    merge(
        open_datasets,
        bounds=(w, s, e, n),
        nodata=-9999.0,
        dst_path=output_path,
        dst_kwds=dst_kwds,
        mem_limit=512,  # MB/chunk; default (64) works too, just more
                         # (smaller) chunks and more loop overhead
    )

    for ds in open_datasets:
        ds.close()

    print(f"[MOSAIC] Merged in {time.time()-t_merge:.0f}s")

    # ── Water correction ──────────────────────────────────────────
    if apply_water_correction:
        print(f"[MOSAIC] Applying NHD water body correction...")
        try:
            from dem_water_correction import (
                fetch_water_bodies_nhd, correct_dem_water_bodies
            )
            water_bounds = (s, w, n, e)
            water_polygons = fetch_water_bodies_nhd(water_bounds)

            try:
                from dem_water_correction import fetch_ocean_polygons_osm
                ocean_polys = fetch_ocean_polygons_osm(water_bounds)
                if ocean_polys:
                    print(f"[MOSAIC] Adding {len(ocean_polys)} ocean polygons")
                    water_polygons.extend(ocean_polys)
            except Exception as e:
                print(f"[MOSAIC] Ocean fetch warning: {e}")

            if water_polygons:
                # Correct in place — overwrites output_path
                correct_dem_water_bodies(
                    input_path=output_path,
                    output_path=output_path,
                    water_polygons=water_polygons,
                    buffer_pixels=3,
                    shore_percentile=10.0,
                    batch_size=500,
                    verbose=True,
                )
            else:
                print(f"[MOSAIC] No water bodies found — skipping correction")

        except Exception as e:
            print(f"[MOSAIC] Water correction warning: {e}")
            print(f"[MOSAIC] Mosaic written uncorrected")

    output_mb = Path(output_path).stat().st_size / 1e6
    elapsed   = time.time() - t0

    with rasterio.open(output_path) as final_ds:
        shape        = (final_ds.height, final_ds.width)
        resolution_m = final_ds.transform.a * 111_320

    stats = {
        'output_path':  output_path,
        'source':       source,
        'shape':        shape,
        'resolution_m': resolution_m,
        'n_tiles':      len(tile_paths),
        'output_mb':    round(output_mb, 1),
        'elapsed_s':    round(elapsed, 1),
    }

    print(f"\n[MOSAIC] Complete in {elapsed:.0f}s")
    print(f"  Shape:      {stats['shape'][1]:,} x {stats['shape'][0]:,}")
    print(f"  Resolution: {stats['resolution_m']:.1f}m/px")
    print(f"  Output:     {output_path} ({output_mb:.1f}MB)")

    return stats


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build DEM mosaic for a study area',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'corner1', nargs='?',
        help='Tile ID for one corner of the area, e.g. N44W113 (with corner2; omit if using --bounds)',
    )
    parser.add_argument(
        'corner2', nargs='?',
        help='Tile ID for the opposite corner, e.g. N47W109',
    )
    parser.add_argument('--source', default='auto',
        choices=['auto', 'GLO30', '3DEP_10', '3DEP_1'],
        help='DEM source (default: auto — 3DEP for US, GLO30 elsewhere)')
    parser.add_argument('--output', default=None,
        help='Output path (overrides the corner-derived default)')
    parser.add_argument('--bounds', nargs=4, type=float,
        metavar=('SOUTH', 'WEST', 'NORTH', 'EAST'),
        help='Custom fractional-degree bounds instead of corner1/corner2')
    parser.add_argument('--allow-large', action='store_true',
        help=f'Bypass the {MAX_MOSAIC_TILES}-degree-tile size cap on corner1/corner2 areas')
    parser.add_argument('--buffer', type=float, default=None,
        help='Buffer beyond bounds in degrees (default: 0.1 for --bounds, '
             '0 for corner1/corner2 -- see main() for why those differ)')
    parser.add_argument('--no-water-correction', action='store_true',
        help='Skip water correction step')

    args = parser.parse_args()

    if args.bounds:
        south, west, north, east = args.bounds
        if not args.output:
            parser.error('--output required when using --bounds')
        output_path = args.output
        # 0.1 default: raw bounds can fall mid-tile, so a small margin
        # avoids a gap right at the edge.
        buffer_deg = args.buffer if args.buffer is not None else 0.1
    elif args.corner1 and args.corner2:
        area = mosaic_area_between(args.corner1, args.corner2,
                                    allow_large=args.allow_large)
        south, west = area['south'], area['west']
        north, east = area['north'], area['east']
        output_path = args.output or area['dem_path']
        # 0 default: corner1/corner2 bounds are already exactly
        # tile-aligned (see bounds_from_tile_corners()), so the same
        # 0.1deg margin would silently push the request across an
        # extra whole tile on every side instead of just avoiding a
        # mid-tile edge gap -- confirmed live in the R port testing:
        # it turned a 16-tile (4x4deg) request into 36 tiles (6x6deg),
        # well past the size cap the corners alone had just cleared.
        buffer_deg = args.buffer if args.buffer is not None else 0.0
    else:
        parser.error('Provide two corner tile IDs (e.g. N44W113 N47W109) or --bounds')

    build_mosaic(
        south=south, west=west,
        north=north, east=east,
        output_path=output_path,
        source=args.source,
        apply_water_correction=not args.no_water_correction,
        buffer_deg=buffer_deg,
    )


if __name__ == '__main__':
    main()