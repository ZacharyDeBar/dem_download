"""Unit tests for tile_id.py."""

import math

from tile_id import (
    tile_id_from_latlng, tile_id_from_floors, parse_tile_id,
    tile_bounds, tile_ids_covering_bbox, tile_ids_covering_polyline,
    neighbor_tile_ids, TileBounds,
)


def test_tile_id_from_latlng():
    """Basic mapping of points to their containing tile."""
    # Bozeman area
    assert tile_id_from_latlng(45.5, -110.5) == 'N45W111'
    # Equator and prime meridian
    assert tile_id_from_latlng(0.5, 0.5) == 'N00E000'
    # Southern hemisphere
    assert tile_id_from_latlng(-12.7, 178.3) == 'S13E178'
    # Edge of a tile (point exactly at SW corner — belongs to that tile)
    assert tile_id_from_latlng(45.0, -110.0) == 'N45W110'
    # Negative longitude small fraction
    assert tile_id_from_latlng(45.5, -109.001) == 'N45W110'
    # Just inside N45W110 northern edge
    assert tile_id_from_latlng(45.999, -109.5) == 'N45W110'
    # Just past northern edge → belongs to N46
    assert tile_id_from_latlng(46.001, -109.5) == 'N46W110'
    print('  test_tile_id_from_latlng OK')


def test_parse_tile_id():
    """Round-trip parsing."""
    cases = [
        ('N45W110', (45, -110)),
        ('N00E000', (0,  0)),
        ('S13E178', (-13, 178)),
        ('N89E179', (89, 179)),
        ('S89W180', (-89, -180)),
    ]
    for tid, expected in cases:
        got = parse_tile_id(tid)
        assert got == expected, f"{tid}: got {got}, expected {expected}"
    print('  test_parse_tile_id OK')


def test_round_trip():
    """Encode → decode → encode should be stable."""
    cases = [
        (45, -110), (0, 0), (-13, 178), (89, 179), (-89, -180),
        (45, -111), (-1, -1), (1, 1),
    ]
    for lat, lng in cases:
        tid = tile_id_from_floors(lat, lng)
        parsed = parse_tile_id(tid)
        assert parsed == (lat, lng), \
            f"Round trip failed: ({lat},{lng}) → {tid} → {parsed}"
        # And from a point in the tile, we should land back at the same ID
        # (use center of tile to avoid boundary issues)
        tid2 = tile_id_from_latlng(lat + 0.5, lng + 0.5)
        assert tid2 == tid, f"From latlng round trip: {tid} != {tid2}"
    print('  test_round_trip OK')


def test_parse_invalid():
    """Bad inputs should raise."""
    bad = ['', 'N45W110X', 'X45W110', 'N99W500', 'N5W110',
           'n45w110', 'N45-W110']
    for tid in bad:
        try:
            parse_tile_id(tid)
            assert False, f"Should have raised for {tid!r}"
        except ValueError:
            pass
    print('  test_parse_invalid OK')


def test_tile_bounds():
    """Bounds are correct for various tiles."""
    b = tile_bounds('N45W110')
    assert b.south == 45.0
    assert b.west  == -110.0
    assert b.north == 46.0
    assert b.east  == -109.0
    assert b.center == (45.5, -109.5)
    assert b.contains(45.5, -109.5)
    assert b.contains(45.0, -110.0)         # SW corner inclusive
    assert not b.contains(46.0, -109.5)     # N edge exclusive
    assert not b.contains(45.5, -109.0)     # E edge exclusive
    assert b.as_tuple() == (45.0, -110.0, 46.0, -109.0)

    # Southern hemisphere
    b = tile_bounds('S13E178')
    assert b.south == -13.0
    assert b.west  == 178.0
    assert b.north == -12.0
    assert b.east  == 179.0
    assert b.contains(-12.5, 178.5)
    print('  test_tile_bounds OK')


def test_tile_bounds_intersects_bbox():
    b = tile_bounds('N45W110')  # covers (45.0, -110.0, 46.0, -109.0)
    # Bbox fully inside
    assert b.intersects_bbox(45.1, -109.9, 45.9, -109.1)
    # Bbox fully outside (far)
    assert not b.intersects_bbox(50.0, -100.0, 51.0, -99.0)
    # Bbox touching east edge — exclusive, so should NOT intersect
    assert not b.intersects_bbox(45.5, -109.0, 45.6, -108.9)
    # Bbox overlapping east edge
    assert b.intersects_bbox(45.5, -109.5, 45.6, -108.5)
    print('  test_tile_bounds_intersects_bbox OK')


def test_tile_ids_covering_bbox_small():
    # Bbox crossing one tile boundary east-west
    result = tile_ids_covering_bbox(45.2, -110.7, 45.9, -109.3)
    # Should hit N45W110 (covers -110..-109) and N45W111 (covers -111..-110)
    # Sorted by (lat_floor, lng_floor): N45W111 first, N45W110 second
    assert result == ['N45W111', 'N45W110'], result
    print('  test_tile_ids_covering_bbox_small OK')


def test_tile_ids_covering_bbox_3x3():
    # Bbox spanning 3 tiles each direction
    result = tile_ids_covering_bbox(44.5, -111.5, 46.5, -108.5)
    expected = [
        'N44W112', 'N44W111', 'N44W110', 'N44W109',
        'N45W112', 'N45W111', 'N45W110', 'N45W109',
        'N46W112', 'N46W111', 'N46W110', 'N46W109',
    ]
    assert result == expected, f"\nGot:      {result}\nExpected: {expected}"
    print('  test_tile_ids_covering_bbox_3x3 OK')


def test_tile_ids_covering_bbox_boundary():
    """Bbox edges exactly on tile boundaries."""
    # north edge exactly at 46.0 — should NOT include tile starting at 46
    result = tile_ids_covering_bbox(45.0, -110.0, 46.0, -109.0)
    # bbox lat in [45, 46), bbox lng in [-110, -109) — exactly one tile
    assert result == ['N45W110'], result

    # And expanding slightly past should include the next
    result = tile_ids_covering_bbox(45.0, -110.0, 46.001, -109.0)
    assert result == ['N45W110', 'N46W110'], result
    print('  test_tile_ids_covering_bbox_boundary OK')


def test_tile_ids_covering_bbox_invalid():
    try:
        tile_ids_covering_bbox(46.0, -110.0, 45.0, -109.0)  # south > north
        assert False
    except ValueError:
        pass
    try:
        tile_ids_covering_bbox(45.0, -109.0, 46.0, -110.0)  # west > east
        assert False
    except ValueError:
        pass
    print('  test_tile_ids_covering_bbox_invalid OK')


def test_tile_ids_covering_polyline():
    # Beartooth-ish horizontal segment at 45N
    result = tile_ids_covering_polyline(
        [(45.0, -109.5), (45.0, -109.0)],
        buffer_m=10_000,  # 10km buffer
    )
    # Should cover 1° around 45N -109.5..-109 with 10km buffer.
    # 10km ≈ 0.09°. So bbox is roughly (44.91, -109.6, 45.09, -108.9).
    # That spans tiles N44W110, N44W109, N45W110, N45W109
    expected = ['N44W110', 'N44W109', 'N45W110', 'N45W109']
    assert result == expected, f"\nGot:      {result}\nExpected: {expected}"

    # Single point polyline still works
    result = tile_ids_covering_polyline(
        [(45.5, -109.5)],
        buffer_m=5_000,
    )
    # Single point in N45W110 with small buffer should just give that tile
    assert result == ['N45W110'], result

    # Long N-S polyline with 300km buffer
    result = tile_ids_covering_polyline(
        [(45.0, -110.0), (46.0, -110.0)],
        buffer_m=300_000,
    )
    # 300km buffer ≈ 2.7° lat, 3.8° lng at 45N
    # Should give a big set
    assert len(result) >= 9, f"Expected at least 9 tiles, got {len(result)}: {result}"
    print('  test_tile_ids_covering_polyline OK')


def test_tile_ids_covering_polyline_empty():
    assert tile_ids_covering_polyline([], buffer_m=10_000) == []
    print('  test_tile_ids_covering_polyline_empty OK')


def test_neighbor_tile_ids():
    n = neighbor_tile_ids('N45W110')
    assert n['N']  == 'N46W110'
    assert n['S']  == 'N44W110'
    assert n['E']  == 'N45W109'
    assert n['W']  == 'N45W111'
    assert n['NE'] == 'N46W109'
    assert n['NW'] == 'N46W111'
    assert n['SE'] == 'N44W109'
    assert n['SW'] == 'N44W111'

    # Polar — N89 has no northern neighbor
    n = neighbor_tile_ids('N89W110')
    assert n['N']  is None    # would be at lat 90
    assert n['NE'] is None
    assert n['NW'] is None
    assert n['S']  == 'N88W110'  # southern still works
    print('  test_neighbor_tile_ids OK')


def test_invalid_floors():
    for bad in [(90, 0), (-91, 0), (0, 180), (0, -181)]:
        try:
            tile_id_from_floors(*bad)
            assert False, f"Should have raised for {bad}"
        except ValueError:
            pass
    print('  test_invalid_floors OK')


# Run all tests
if __name__ == '__main__':
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
    print('\nAll tile_id tests PASS')