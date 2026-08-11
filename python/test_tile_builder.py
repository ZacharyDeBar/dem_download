"""
Unit tests for tile_builder.py — tests the parts we can verify
without network access (gap detect, flat mask, boundary mask, water
diagnostics, registry transitions). Network-dependent paths (download,
GLO-30 fetch, water polygon fetch) are tested via mocks.
"""

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from tile_id import tile_bounds
from tile_registry import (
    load_registry, save_registry, STATUS_READY, STATUS_PENDING,
    STATUS_FAILED,
)
from storage_manager import POLICY_HARD


def _tmp_root():
    return Path(tempfile.mkdtemp(prefix='dem_builder_test_'))


def _make_synthetic_dem(path: Path, tile_id: str,
                        height: int = 1080, width: int = 1080):
    """Create a small synthetic GeoTIFF at the tile's bbox."""
    import rasterio
    from rasterio.transform import from_bounds
    from tile_id import tile_bounds
    b = tile_bounds(tile_id)
    # Make terrain with both flat and varied regions
    data = np.zeros((height, width), dtype=np.float32)
    # Add a hill in one corner (lots of variance → not flat)
    for r in range(height):
        for c in range(width):
            if r < height // 2 and c < width // 2:
                # Tall hill with steep slope
                data[r, c] = 1000 + 50 * np.sin(r * 0.1) + 50 * np.cos(c * 0.1)
            else:
                # Flat plain
                data[r, c] = 500.0
    transform = from_bounds(b.west, b.south, b.east, b.north, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, 'w', driver='GTiff', height=height, width=width, count=1,
        dtype='float32', crs='EPSG:4326', transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def test_step_crop_accepts_epsg_4269():
    """Cropping should accept EPSG:4269 (NAD83) — 3DEP's native CRS."""
    root = _tmp_root()
    try:
        from tile_builder import _step_crop_and_fill_gaps
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        tile_id = 'N45W110'
        b = tile_bounds(tile_id)
        # Make a raw tile in EPSG:4269 (NAD83) — geographic, not 4326
        h = w = 720
        # Note: 3DEP raw tiles typically have ~6 arc-second padding
        # beyond the nominal 1° — simulate by making the raw cover
        # a slightly larger area
        pad = 0.01
        data = np.full((h, w), 1500.0, dtype=np.float32)
        raw_path = root / 'downloads' / f'{tile_id}_raw.tif'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            raw_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs=CRS.from_epsg(4269),  # NAD83
            transform=from_bounds(b.west - pad, b.south - pad,
                                   b.east + pad, b.north + pad, w, h),
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        dem_out = root / 'tiles' / tile_id / f'{tile_id}_dem.tif'
        sr = _step_crop_and_fill_gaps(
            tile_id=tile_id, bounds=b, raw_path=raw_path, dem_out=dem_out,
        )
        assert sr.success, f'Failed: {sr.error}'
        assert sr.detail['src_crs'] == 'EPSG:4269'
        # OUTPUT IS ALWAYS EPSG:4326 (calibration #27 — tile_builder
        # standardizes on 4326 so VRT composition with migrated tiles
        # works). The reproject was applied.
        assert sr.detail['dst_crs'] == 'EPSG:4326'
        assert sr.detail['reprojected'] is True
        assert dem_out.exists()
        with rasterio.open(dem_out) as src:
            assert src.crs.to_epsg() == 4326
            # Output should be tile-bbox-aligned: NW corner at
            # (tile_west, tile_north).
            assert abs(src.bounds.left   - b.west)  < 1e-9
            assert abs(src.bounds.top    - b.north) < 1e-9
            assert abs(src.bounds.right  - b.east)  < 1e-9
            assert abs(src.bounds.bottom - b.south) < 1e-9
        print('  test_step_crop_accepts_epsg_4269 OK')
    finally:
        shutil.rmtree(root)


def test_step_crop_accepts_epsg_4326():
    """Cropping should still accept EPSG:4326 — GLO-30's native CRS."""
    root = _tmp_root()
    try:
        from tile_builder import _step_crop_and_fill_gaps
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        tile_id = 'N45W110'
        b = tile_bounds(tile_id)
        h = w = 720
        pad = 0.01
        data = np.full((h, w), 1500.0, dtype=np.float32)
        raw_path = root / 'downloads' / f'{tile_id}_raw.tif'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            raw_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs=CRS.from_epsg(4326),
            transform=from_bounds(b.west - pad, b.south - pad,
                                   b.east + pad, b.north + pad, w, h),
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        dem_out = root / 'tiles' / tile_id / f'{tile_id}_dem.tif'
        sr = _step_crop_and_fill_gaps(
            tile_id=tile_id, bounds=b, raw_path=raw_path, dem_out=dem_out,
        )
        assert sr.success, f'Failed: {sr.error}'
        # Output is EPSG:4326 and tile-bbox-aligned.
        assert sr.detail['dst_crs'] == 'EPSG:4326'
        with rasterio.open(dem_out) as src:
            assert src.crs.to_epsg() == 4326
            assert abs(src.bounds.left  - b.west)  < 1e-9
            assert abs(src.bounds.top   - b.north) < 1e-9
        print('  test_step_crop_accepts_epsg_4326 OK')
    finally:
        shutil.rmtree(root)


def test_step_crop_rejects_projected_crs():
    """Cropping should refuse projected CRSes (UTM, web mercator)."""
    root = _tmp_root()
    try:
        from tile_builder import _step_crop_and_fill_gaps
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.crs import CRS

        tile_id = 'N45W110'
        b = tile_bounds(tile_id)
        h = w = 720
        # UTM 12N — projected, metres as units
        data = np.full((h, w), 1500.0, dtype=np.float32)
        raw_path = root / 'downloads' / f'{tile_id}_raw.tif'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            raw_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs=CRS.from_epsg(32612),   # UTM 12N
            transform=from_origin(500000.0, 5000000.0, 30, 30),  # ~30m pixels
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        dem_out = root / 'tiles' / tile_id / f'{tile_id}_dem.tif'
        sr = _step_crop_and_fill_gaps(
            tile_id=tile_id, bounds=b, raw_path=raw_path, dem_out=dem_out,
        )
        assert sr.success is False
        assert 'projected' in sr.error.lower()
        print('  test_step_crop_rejects_projected_crs OK')
    finally:
        shutil.rmtree(root)


def test_step_crop_canonical_grid_alignment():
    """Fresh-built tile output must snap to canonical grid for VRT compat.

    Specifically: even though source raw 3DEP tiles are padded (extend
    beyond the nominal 1° bbox with ~6 arc-second overlap), the output
    must have:
      - Exact bounds matching the tile's nominal 1° bbox
      - Pixel-perfect alignment to a global integer-pixel grid
        (so that VRT composition with other tile sources works)
    """
    root = _tmp_root()
    try:
        from tile_builder import _step_crop_and_fill_gaps
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        tile_id = 'N45W110'
        b = tile_bounds(tile_id)
        # Simulate 3DEP-style raw with significant padding
        # If the raw is 720x720 covering 1.02° in lat and lng (i.e.
        # pad = 0.01° on each side), the cropped tile-bbox extent will
        # be 705x705 — but our output should EXACTLY 1° × output dim.
        h = w = 720
        pad = 0.01
        data = np.full((h, w), 1500.0, dtype=np.float32)
        raw_path = root / 'downloads' / f'{tile_id}_raw.tif'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            raw_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs=CRS.from_epsg(4269),
            transform=from_bounds(b.west - pad, b.south - pad,
                                   b.east + pad, b.north + pad, w, h),
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        dem_out = root / 'tiles' / tile_id / f'{tile_id}_dem.tif'
        sr = _step_crop_and_fill_gaps(
            tile_id=tile_id, bounds=b, raw_path=raw_path, dem_out=dem_out,
        )
        assert sr.success, f'Failed: {sr.error}'

        with rasterio.open(dem_out) as src:
            t = src.transform
            # Origin is at NW corner of the tile bbox (canonical)
            assert abs(t.c - b.west) < 1e-9, \
                f'Origin lon: expected {b.west}, got {t.c}'
            assert abs(t.f - b.north) < 1e-9, \
                f'Origin lat: expected {b.north}, got {t.f}'
            # Pixel size = 1° / dim — exact (no fractional adjustment)
            expected_px = 1.0 / src.width
            assert abs(abs(t.a) - expected_px) < 1e-15, \
                f'Pixel size x: expected {expected_px}, got {abs(t.a)}'
            assert abs(abs(t.e) - expected_px) < 1e-15, \
                f'Pixel size y: expected {expected_px}, got {abs(t.e)}'
            # Bounds match tile bbox exactly
            assert abs(src.bounds.left   - b.west)  < 1e-9
            assert abs(src.bounds.bottom - b.south) < 1e-9
            assert abs(src.bounds.right  - b.east)  < 1e-9
            assert abs(src.bounds.top    - b.north) < 1e-9
        print('  test_step_crop_canonical_grid_alignment OK')
    finally:
        shutil.rmtree(root)


def test_step_crop_keep_original_writes_gap_mask():
    """keep_original=True should persist a gap mask matching the
    pre-fill nodata footprint, reprojected onto dem_out's own grid."""
    root = _tmp_root()
    try:
        from tile_builder import _step_crop_and_fill_gaps
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        tile_id = 'N45W110'
        b = tile_bounds(tile_id)
        h = w = 720
        data = np.full((h, w), 1500.0, dtype=np.float32)
        data[100:150, 100:150] = -9999.0  # a nodata gap block
        raw_path = root / 'downloads' / f'{tile_id}_raw.tif'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            raw_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs=CRS.from_epsg(4326),
            transform=from_bounds(b.west, b.south, b.east, b.north, w, h),
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        dem_out = root / 'tiles' / tile_id / f'{tile_id}_dem.tif'
        gap_mask_out = root / 'tiles' / tile_id / f'{tile_id}_gap_mask.tif'

        def fake_fill(tile_id, bounds, data, gap_mask, transform, dst_crs='EPSG:4326'):
            data[gap_mask] = 1500.0  # pretend GLO-30 filled it flat
            return {'gaps_filled_from': 'GLO30',
                     'gaps_filled_count': int(gap_mask.sum()),
                     'glo30_tiles_used': []}

        with patch('tile_builder._fill_gaps_from_glo30', side_effect=fake_fill):
            sr = _step_crop_and_fill_gaps(
                tile_id=tile_id, bounds=b, raw_path=raw_path, dem_out=dem_out,
                keep_original=True, gap_mask_out=gap_mask_out,
            )
        assert sr.success, f'Failed: {sr.error}'
        assert sr.detail['n_gaps_pre_fill'] == 2500  # 50x50 block, in the
                                                      # source (720x720) grid

        assert gap_mask_out.exists()
        with rasterio.open(dem_out) as dem_src, rasterio.open(gap_mask_out) as mask_src:
            # gap_mask_out is reprojected onto the SAME canonical grid as
            # dem_out (always 10800x10800 — the 10m grid invariant — not
            # the source's 720x720), so they must align exactly.
            assert mask_src.shape == dem_src.shape
            mask = mask_src.read(1)
            assert mask.dtype == np.uint8
            # Nearest-neighbor resampling of the 50x50/720x720 gap block
            # preserves area fraction up to boundary-pixel rounding.
            expected_fraction = 2500 / (720 * 720)
            actual_fraction = mask.sum() / mask.size
            assert abs(actual_fraction - expected_fraction) < 0.005, \
                f'expected ~{expected_fraction:.5f}, got {actual_fraction:.5f}'
        # dem_out itself has no remaining gap (fake_fill filled it)
        with rasterio.open(dem_out) as src:
            assert (src.read(1) > -9998.0).all()
        print('  test_step_crop_keep_original_writes_gap_mask OK')
    finally:
        shutil.rmtree(root)


def test_step_crop_without_keep_original_skips_gap_mask():
    """Default (keep_original=False) should not write a gap mask, even
    when gaps exist — extra I/O most callers don't want."""
    root = _tmp_root()
    try:
        from tile_builder import _step_crop_and_fill_gaps
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        tile_id = 'N45W110'
        b = tile_bounds(tile_id)
        h = w = 720
        data = np.full((h, w), 1500.0, dtype=np.float32)
        data[100:150, 100:150] = -9999.0
        raw_path = root / 'downloads' / f'{tile_id}_raw.tif'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            raw_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs=CRS.from_epsg(4326),
            transform=from_bounds(b.west, b.south, b.east, b.north, w, h),
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        dem_out = root / 'tiles' / tile_id / f'{tile_id}_dem.tif'
        gap_mask_out = root / 'tiles' / tile_id / f'{tile_id}_gap_mask.tif'

        def fake_fill(tile_id, bounds, data, gap_mask, transform, dst_crs='EPSG:4326'):
            data[gap_mask] = 1500.0
            return {'gaps_filled_from': 'GLO30',
                     'gaps_filled_count': int(gap_mask.sum()),
                     'glo30_tiles_used': []}

        with patch('tile_builder._fill_gaps_from_glo30', side_effect=fake_fill):
            sr = _step_crop_and_fill_gaps(
                tile_id=tile_id, bounds=b, raw_path=raw_path, dem_out=dem_out,
                gap_mask_out=gap_mask_out,   # keep_original defaults False
            )
        assert sr.success, f'Failed: {sr.error}'
        assert not gap_mask_out.exists()
        print('  test_step_crop_without_keep_original_skips_gap_mask OK')
    finally:
        shutil.rmtree(root)


def test_step_flat_mask_synthetic():
    """Compute flat mask on synthetic DEM and verify outputs."""
    root = _tmp_root()
    try:
        from tile_builder import _step_flat_mask, _abs_paths_for_tile

        tile_id = 'N45W110'
        # Need precompute_flat_mask module — copy it into testing location
        # OR add to PYTHONPATH. For sibling-import, add user project path.
        # In this test env we don't have it. Use a mock instead.
        with patch('precompute_flat_mask.compute_flat_mask') as mock_compute:
            mock_compute.return_value = {
                'flat_pct': 50.0, 'mask_mb': 1.0,
                'window': 5, 'threshold': 4.0,
                'elev_min_m': 500.0, 'elev_max_m': 600.0, 'elev_range_m': 100.0,
            }
            paths = _abs_paths_for_tile(root, tile_id)
            paths['tile_dir'].mkdir(parents=True, exist_ok=True)
            # Create empty dem file (mock doesn't need to read it)
            paths['dem'].touch()
            sr = _step_flat_mask(
                tile_id=tile_id, dem_path=paths['dem'],
                flat_out=paths['flat'],
            )
            assert sr.success
            assert sr.name == 'flat_mask'
            assert sr.detail['flat_pct'] == 50.0
            mock_compute.assert_called_once()
        print('  test_step_flat_mask_synthetic OK')
    finally:
        shutil.rmtree(root)


def test_step_flat_mask_validation_gate():
    """If compute returns insane flat_pct, step should fail."""
    root = _tmp_root()
    try:
        from tile_builder import _step_flat_mask, _abs_paths_for_tile

        with patch('precompute_flat_mask.compute_flat_mask') as mock_compute:
            mock_compute.return_value = {
                'flat_pct': 0.0,  # nonsense
                'mask_mb': 1.0, 'window': 5, 'threshold': 4.0,
                # A valid (non-None) range, so this exercises the flat_pct
                # gate specifically and not the elev_range_m-is-None gate.
                'elev_min_m': 500.0, 'elev_max_m': 550.0, 'elev_range_m': 50.0,
            }
            paths = _abs_paths_for_tile(root, 'N45W110')
            paths['tile_dir'].mkdir(parents=True, exist_ok=True)
            paths['dem'].touch()
            sr = _step_flat_mask(
                tile_id='N45W110', dem_path=paths['dem'],
                flat_out=paths['flat'],
            )
            assert sr.success is False
            assert 'outside sane range' in sr.error.lower() or \
                   'sane range' in sr.error.lower()
        print('  test_step_flat_mask_validation_gate OK')
    finally:
        shutil.rmtree(root)


def test_step_flat_mask_none_elev_range_fails():
    """elev_range_m is None (compute_flat_mask found zero valid/non-nodata
    pixels) must fail unconditionally -- even when flat_pct reads high,
    even for a tile_id that would otherwise pass the degenerate-range
    check. Real incident (2026-08-08): a transient nodata-detection
    failure read the whole DEM as invalid, made flat_pct read 100% for
    the wrong reason, and the old gate's `elev_range_m is not None`
    guard let it through as 'ready'."""
    root = _tmp_root()
    try:
        from tile_builder import _step_flat_mask, _abs_paths_for_tile

        with patch('precompute_flat_mask.compute_flat_mask') as mock_compute:
            mock_compute.return_value = {
                'flat_pct': 100.0, 'mask_mb': 1.0, 'window': 5, 'threshold': 4.0,
                'elev_min_m': None, 'elev_max_m': None, 'elev_range_m': None,
            }
            paths = _abs_paths_for_tile(root, 'N45W111')
            paths['tile_dir'].mkdir(parents=True, exist_ok=True)
            paths['dem'].touch()
            sr = _step_flat_mask(
                tile_id='N45W111', dem_path=paths['dem'],
                flat_out=paths['flat'],
            )
            assert sr.success is False
            assert 'n/a' in sr.error.lower()
            assert 'no valid' in sr.error.lower()

            # Even with skip_degenerate_check=True (ocean-fallback tiles) --
            # that flag only exempts the flat_pct>99%+degenerate-range gate,
            # not this one. No legitimate tile, ocean fallback included,
            # produces elev_range_m=None.
            sr2 = _step_flat_mask(
                tile_id='N45W111', dem_path=paths['dem'],
                flat_out=paths['flat'], skip_degenerate_check=True,
            )
            assert sr2.success is False
        print('  test_step_flat_mask_none_elev_range_fails OK')
    finally:
        shutil.rmtree(root)


def test_step_flat_mask_high_pct_with_real_relief_passes():
    """Genuinely flat terrain (e.g. West Texas plains) can legitimately hit
    99%+ local-window flatness -- this must NOT fail as long as the whole
    tile still has real elevation relief (a few tens of metres), which is
    what real terrain always has even over the flattest 1° tile."""
    root = _tmp_root()
    try:
        from tile_builder import _step_flat_mask, _abs_paths_for_tile

        with patch('precompute_flat_mask.compute_flat_mask') as mock_compute:
            mock_compute.return_value = {
                'flat_pct': 99.93, 'mask_mb': 1.0, 'window': 5, 'threshold': 4.0,
                'elev_min_m': 850.0, 'elev_max_m': 910.0, 'elev_range_m': 60.0,
            }
            paths = _abs_paths_for_tile(root, 'N32W103')
            paths['tile_dir'].mkdir(parents=True, exist_ok=True)
            paths['dem'].touch()
            sr = _step_flat_mask(
                tile_id='N32W103', dem_path=paths['dem'],
                flat_out=paths['flat'],
            )
            assert sr.success is True, sr.error
        print('  test_step_flat_mask_high_pct_with_real_relief_passes OK')
    finally:
        shutil.rmtree(root)


def test_step_flat_mask_high_pct_with_degenerate_range_fails():
    """A near-100% flat mask AND a near-zero whole-tile elevation range is
    the actual signature of a degenerate/placeholder DEM (corrupted
    download, all-nodata filled with one constant value) -- this SHOULD
    still fail."""
    root = _tmp_root()
    try:
        from tile_builder import _step_flat_mask, _abs_paths_for_tile

        with patch('precompute_flat_mask.compute_flat_mask') as mock_compute:
            mock_compute.return_value = {
                'flat_pct': 99.99, 'mask_mb': 1.0, 'window': 5, 'threshold': 4.0,
                'elev_min_m': 100.0, 'elev_max_m': 100.5, 'elev_range_m': 0.5,
            }
            paths = _abs_paths_for_tile(root, 'N45W110')
            paths['tile_dir'].mkdir(parents=True, exist_ok=True)
            paths['dem'].touch()
            sr = _step_flat_mask(
                tile_id='N45W110', dem_path=paths['dem'],
                flat_out=paths['flat'],
            )
            assert sr.success is False
            assert 'degenerate' in sr.error.lower()
        print('  test_step_flat_mask_high_pct_with_degenerate_range_fails OK')
    finally:
        shutil.rmtree(root)


def test_step_boundary_mask_validation():
    """If boundary cells > flat cells, step should fail."""
    root = _tmp_root()
    try:
        from tile_builder import _step_boundary_mask, _abs_paths_for_tile

        with patch('precompute_flat_mask.compute_flat_regions') as mock_cfr:
            mock_cfr.return_value = {
                'boundary_pct': 50.0, 'boundary_cells': 1000,
                'flat_cells': 500,   # invariant violation
                'reduction_pct': -100.0, 'boundary_mb': 1.0,
            }
            paths = _abs_paths_for_tile(root, 'N45W110')
            paths['tile_dir'].mkdir(parents=True, exist_ok=True)
            sr = _step_boundary_mask(
                tile_id='N45W110', flat_path=paths['flat'],
                boundary_out=paths['boundary'],
            )
            assert sr.success is False
            assert 'invariant' in sr.error.lower() or \
                   'more cells' in sr.error.lower()
        print('  test_step_boundary_mask_validation OK')
    finally:
        shutil.rmtree(root)


def test_step_water_correction_keep_original_writes_precorrection_and_stats():
    """keep_original=True should copy the pre-correction DEM and persist
    correct_dem_water_bodies's elevation_changes to disk."""
    root = _tmp_root()
    try:
        from tile_builder import _step_water_correction, _abs_paths_for_tile
        import rasterio

        tile_id = 'N45W110'
        b = tile_bounds(tile_id)
        paths = _abs_paths_for_tile(root, tile_id)
        paths['tile_dir'].mkdir(parents=True, exist_ok=True)
        _make_synthetic_dem(paths['dem'], tile_id, height=200, width=200)

        with rasterio.open(paths['dem']) as src:
            original_data = src.read(1).copy()

        fake_poly = {
            'type': 'Polygon',
            'coordinates': [[
                (b.west + 0.1, b.south + 0.1), (b.west + 0.2, b.south + 0.1),
                (b.west + 0.2, b.south + 0.2), (b.west + 0.1, b.south + 0.2),
                (b.west + 0.1, b.south + 0.1),
            ]],
            'properties': {'source': 'osm', 'name': 'test lake'},
        }

        def fake_correct(input_path, output_path, water_polygons, **kwargs):
            with rasterio.open(input_path) as src:
                data = src.read(1)
                profile = src.profile.copy()
            data = data.copy()
            data[80:100, 80:100] = 490.0  # pretend correction flattened a patch
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            return {
                'corrected_bodies': 1, 'skipped_small': 0,
                'total_pixels_fixed': 400,
                'elevation_changes': [{
                    'polygon_index': 0, 'pixel_count': 400,
                    'surface_elev_m': 490.0, 'original_mean_m': 500.0,
                    'change_m': -10.0,
                    'lat': b.south + 0.15, 'lon': b.west + 0.15,
                }],
            }

        with patch('dem_water_correction.fetch_water_bodies_osm', return_value=[fake_poly]), \
             patch('dem_water_correction.fetch_ocean_polygons_osm', return_value=[]), \
             patch('dem_water_correction.correct_dem_water_bodies', side_effect=fake_correct):
            sr = _step_water_correction(
                tile_id=tile_id, bounds=b,
                dem_path=paths['dem'], water_out=paths['water'],
                source='osm', keep_original=True,
                precorrection_out=paths['dem_precorrection'],
                stats_out=paths['water_stats'],
            )
        assert sr.success, sr.error

        assert paths['dem_precorrection'].exists()
        with rasterio.open(paths['dem_precorrection']) as src:
            assert np.array_equal(src.read(1), original_data)

        assert paths['water_stats'].exists()
        with open(paths['water_stats']) as f:
            stats = json.load(f)
        assert stats['corrected_bodies'] == 1
        assert stats['elevation_changes'][0]['change_m'] == -10.0
        assert stats['elevation_changes'][0]['lat'] == b.south + 0.15
        assert stats['elevation_changes'][0]['lon'] == b.west + 0.15
        print('  test_step_water_correction_keep_original_writes_precorrection_and_stats OK')
    finally:
        shutil.rmtree(root)


def test_step_water_correction_without_keep_original_skips_extra_artifacts():
    """Default (keep_original=False) should not write precorrection DEM
    or stats — matches the old in-place, no-extra-I/O behavior."""
    root = _tmp_root()
    try:
        from tile_builder import _step_water_correction, _abs_paths_for_tile
        import rasterio

        tile_id = 'N45W110'
        b = tile_bounds(tile_id)
        paths = _abs_paths_for_tile(root, tile_id)
        paths['tile_dir'].mkdir(parents=True, exist_ok=True)
        _make_synthetic_dem(paths['dem'], tile_id, height=200, width=200)

        fake_poly = {
            'type': 'Polygon',
            'coordinates': [[
                (b.west + 0.1, b.south + 0.1), (b.west + 0.2, b.south + 0.1),
                (b.west + 0.2, b.south + 0.2), (b.west + 0.1, b.south + 0.2),
                (b.west + 0.1, b.south + 0.1),
            ]],
            'properties': {'source': 'osm', 'name': 'test lake'},
        }

        def fake_correct(input_path, output_path, water_polygons, **kwargs):
            with rasterio.open(input_path) as src:
                data = src.read(1)
                profile = src.profile.copy()
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            return {'corrected_bodies': 1, 'skipped_small': 0,
                    'total_pixels_fixed': 0, 'elevation_changes': []}

        with patch('dem_water_correction.fetch_water_bodies_osm', return_value=[fake_poly]), \
             patch('dem_water_correction.fetch_ocean_polygons_osm', return_value=[]), \
             patch('dem_water_correction.correct_dem_water_bodies', side_effect=fake_correct):
            sr = _step_water_correction(
                tile_id=tile_id, bounds=b,
                dem_path=paths['dem'], water_out=paths['water'],
                source='osm',
                precorrection_out=paths['dem_precorrection'],
                stats_out=paths['water_stats'],
            )
        assert sr.success, sr.error
        assert not paths['dem_precorrection'].exists()
        assert not paths['water_stats'].exists()
        print('  test_step_water_correction_without_keep_original_skips_extra_artifacts OK')
    finally:
        shutil.rmtree(root)


def test_compute_water_diagnostics_no_polygons():
    """Empty polygon list returns empty diagnostics."""
    from tile_builder import _compute_water_diagnostics
    bounds = tile_bounds('N45W110')
    d = _compute_water_diagnostics(
        dem_path=Path('/nonexistent'),  # won't be read
        polygons=[], tile_bounds=bounds,
    )
    assert d['n_polygons_inspected'] == 0
    assert d['flagged_polygons'] == []
    assert d['anomaly_threshold_std_m'] == 5.0
    print('  test_compute_water_diagnostics_no_polygons OK')


def test_compute_water_diagnostics_flat_lake():
    """A polygon over a uniformly-flat region should NOT be flagged."""
    root = _tmp_root()
    try:
        from tile_builder import _compute_water_diagnostics
        tile_id = 'N45W110'
        dem_path = root / 'tiles' / tile_id / f'{tile_id}_dem.tif'

        # Make synthetic DEM: uniform 500m elevation across whole tile
        import rasterio
        from rasterio.transform import from_bounds
        b = tile_bounds(tile_id)
        h = w = 720
        data = np.full((h, w), 500.0, dtype=np.float32)
        dem_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            dem_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs='EPSG:4326',
            transform=from_bounds(b.west, b.south, b.east, b.north, w, h),
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        # Polygon covering 10% of the tile, all flat
        polygons = [{
            'type': 'Polygon',
            'coordinates': [[
                (b.west + 0.1, b.south + 0.1),
                (b.west + 0.4, b.south + 0.1),
                (b.west + 0.4, b.south + 0.4),
                (b.west + 0.1, b.south + 0.4),
                (b.west + 0.1, b.south + 0.1),
            ]],
            'properties': {'name': 'flat_lake'},
        }]
        d = _compute_water_diagnostics(
            dem_path=dem_path, polygons=polygons, tile_bounds=b,
        )
        assert d['n_polygons_inspected'] == 1
        assert d['flagged_polygons'] == [], \
            f"Flat lake should not be flagged: {d['flagged_polygons']}"
        print('  test_compute_water_diagnostics_flat_lake OK')
    finally:
        shutil.rmtree(root)


def test_compute_water_diagnostics_anomalous_lake():
    """A polygon over varied terrain SHOULD be flagged."""
    root = _tmp_root()
    try:
        from tile_builder import _compute_water_diagnostics
        tile_id = 'N45W110'
        dem_path = root / 'tiles' / tile_id / f'{tile_id}_dem.tif'

        # DEM: linear ramp from 500 to 1000 across the tile
        import rasterio
        from rasterio.transform import from_bounds
        b = tile_bounds(tile_id)
        h = w = 720
        data = np.zeros((h, w), dtype=np.float32)
        for c in range(w):
            data[:, c] = 500.0 + (c / w) * 500.0   # ramps 500→1000 W→E
        dem_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            dem_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs='EPSG:4326',
            transform=from_bounds(b.west, b.south, b.east, b.north, w, h),
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        # Polygon covering wide horizontal stripe — should see large range
        polygons = [{
            'type': 'Polygon',
            'coordinates': [[
                (b.west + 0.1, b.south + 0.4),
                (b.west + 0.9, b.south + 0.4),
                (b.west + 0.9, b.south + 0.6),
                (b.west + 0.1, b.south + 0.6),
                (b.west + 0.1, b.south + 0.4),
            ]],
            'properties': {'name': 'wide_lake'},
        }]
        d = _compute_water_diagnostics(
            dem_path=dem_path, polygons=polygons, tile_bounds=b,
        )
        assert d['n_polygons_inspected'] == 1
        assert len(d['flagged_polygons']) == 1, \
            f"Wide lake on ramp should be flagged: {d}"
        flagged = d['flagged_polygons'][0]
        # Range should be substantial — polygon spans wide longitude range
        assert flagged['pre_correction_elev_max_m'] > \
               flagged['pre_correction_elev_min_m'] + 100
        # Not flagged as cross-tile (polygon is within bounds)
        assert flagged['cross_tile'] is False
        print('  test_compute_water_diagnostics_anomalous_lake OK')
    finally:
        shutil.rmtree(root)


def test_compute_water_diagnostics_cross_tile():
    """A polygon extending outside the tile bbox should have cross_tile=True."""
    root = _tmp_root()
    try:
        from tile_builder import _compute_water_diagnostics
        tile_id = 'N45W110'
        dem_path = root / 'tiles' / tile_id / f'{tile_id}_dem.tif'
        import rasterio
        from rasterio.transform import from_bounds
        b = tile_bounds(tile_id)
        h = w = 720
        data = np.full((h, w), 500.0, dtype=np.float32)
        dem_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            dem_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs='EPSG:4326',
            transform=from_bounds(b.west, b.south, b.east, b.north, w, h),
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        # Polygon extends past the tile's eastern edge — but flat
        # so won't be flagged by anomaly thresholds. We're just
        # checking that cross_tile is detected — but flat polygons
        # don't reach the flagging code. Make it anomalous too.
        import rasterio as _rio  # for ramp version
        # Re-write DEM with ramp so the polygon is also anomalous
        data = np.zeros((h, w), dtype=np.float32)
        for c in range(w):
            data[:, c] = 500.0 + (c / w) * 500.0
        with rasterio.open(
            dem_path, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype='float32', crs='EPSG:4326',
            transform=from_bounds(b.west, b.south, b.east, b.north, w, h),
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        polygons = [{
            'type': 'Polygon',
            'coordinates': [[
                (b.west + 0.5, b.south + 0.3),
                (b.east + 0.3, b.south + 0.3),   # past east edge
                (b.east + 0.3, b.south + 0.7),
                (b.west + 0.5, b.south + 0.7),
                (b.west + 0.5, b.south + 0.3),
            ]],
            'properties': {'name': 'cross_tile_lake'},
        }]
        d = _compute_water_diagnostics(
            dem_path=dem_path, polygons=polygons, tile_bounds=b,
        )
        # rasterize will clip the polygon to the raster extent — it should
        # still register inspected pixels (those within the raster)
        assert d['n_polygons_inspected'] >= 1
        # And the polygon should be flagged AND marked cross_tile
        assert len(d['flagged_polygons']) >= 1
        flagged = d['flagged_polygons'][0]
        assert flagged['cross_tile'] is True
        print('  test_compute_water_diagnostics_cross_tile OK')
    finally:
        shutil.rmtree(root)


def test_skip_if_ready():
    """If tile is already in 'ready' state, build_tile should no-op."""
    root = _tmp_root()
    try:
        import tile_builder
        # Seed registry with a 'ready' entry
        reg = load_registry(root)
        reg.mark_ready(
            tile_id='N45W110',
            dem_file='tiles/N45W110/N45W110_dem.tif',
            flat_file='tiles/N45W110/N45W110_flat.tif',
            boundary_file='tiles/N45W110/N45W110_boundary.tif',
            meta_file='tiles/N45W110/N45W110_meta.json',
            size_mb=200.0,
        )
        save_registry(reg)

        result = tile_builder.build_tile(
            tile_id='N45W110', storage_root=root,
            skip_if_ready=True, progress_log=False,
        )
        assert result.success
        assert result.steps == []   # no steps executed
        assert result.final_size_mb == 200.0
        print('  test_skip_if_ready OK')
    finally:
        shutil.rmtree(root)


def test_invalid_tile_id():
    """An invalid tile_id should fail fast."""
    root = _tmp_root()
    try:
        import tile_builder
        result = tile_builder.build_tile(
            tile_id='garbage', storage_root=root,
            progress_log=False,
        )
        assert result.success is False
        assert 'Invalid tile_id' in result.error_summary
        print('  test_invalid_tile_id OK')
    finally:
        shutil.rmtree(root)


def test_capacity_refusal():
    """If cap policy is HARD and cap is already exceeded, refuse."""
    root = _tmp_root()
    try:
        import tile_builder
        # Seed registry with tiles taking up nearly the whole cap
        reg = load_registry(root)
        reg.mark_ready('N40W100', 'a', 'b', 'c', 'd', size_mb=950)
        save_registry(reg)

        result = tile_builder.build_tile(
            tile_id='N45W110', storage_root=root,
            cap_gb=1.0, cap_policy='hard',
            progress_log=False,
        )
        assert result.success is False
        assert result.failed_step == 'capacity_check'
        assert 'HARD' in result.error_summary
        # Registry should NOT have been touched (no pending entry created)
        reg2 = load_registry(root)
        assert reg2.get('N45W110') is None
        print('  test_capacity_refusal OK')
    finally:
        shutil.rmtree(root)


def test_download_failure_marks_failed():
    """If download fails, registry should mark tile as 'failed'."""
    root = _tmp_root()
    try:
        import tile_builder

        # Mock the download to return None (all sources exhausted)
        with patch('dem_download.download_tile', return_value=None):
            result = tile_builder.build_tile(
                tile_id='N45W110', storage_root=root,
                cap_gb=100,  # plenty of room
                progress_log=False,
            )
        assert result.success is False
        assert result.failed_step == 'download'
        # Registry should have a 'failed' entry
        reg = load_registry(root)
        e = reg.get('N45W110')
        assert e is not None
        assert e.status == STATUS_FAILED
        assert 'download' in e.failed_reason.lower()
        print('  test_download_failure_marks_failed OK')
    finally:
        shutil.rmtree(root)


def test_step2b_extent_validation_fires_on_bad_crop_output():
    """Integration test for the Step 2b extent-validation gate added
    after a real incident (2026-07-30): a truncated download slipped
    past Step 2's OLD reproject-trigger check (which only compared
    Affine transform coefficients, not array shape) and got written
    straight to the tile's DEM file at the wrong size. Step 2 itself now
    also catches this (shape check added to the same condition), so a
    bad crop output is no longer constructible via a real raw download
    input -- this test instead verifies Step 2b's OWN wiring into
    build_tile() by monkeypatching Steps 1 and 2 directly, isolating
    Step 2b's gate logic from how a bad shape could arise."""
    root = _tmp_root()
    try:
        import tile_builder
        from tile_builder import StepResult
        tile_id = 'N45W110'

        def fake_download(tile_id, bounds, downloads_dir, source='auto'):
            raw_path = downloads_dir / f'{tile_id}_raw.tif'
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.touch()
            return StepResult(name='download', success=True, elapsed_s=0.01,
                detail={'raw_path': str(raw_path), 'res_label': '3DEP_10'}), raw_path

        def fake_crop(tile_id, bounds, raw_path, dem_out, **kwargs):
            import rasterio
            from rasterio.transform import from_bounds
            b = tile_bounds(tile_id)
            h, w = 541, 1405  # real-incident truncated shape (N44W114)
            data = np.full((h, w), 500.0, dtype=np.float32)
            dem_out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dem_out, 'w', driver='GTiff', height=h, width=w,
                count=1, dtype='float32', crs='EPSG:4326',
                transform=from_bounds(b.west, b.south, b.east, b.south + 0.05, w, h),
                nodata=-9999.0) as dst:
                dst.write(data, 1)
            return StepResult(name='crop_and_fill', success=True, elapsed_s=0.02,
                detail={'cropped_dimensions': [h, w], 'reprojected': False})

        with patch.object(tile_builder, '_step_download', side_effect=fake_download), \
             patch.object(tile_builder, '_step_crop_and_fill_gaps', side_effect=fake_crop):
            result = tile_builder.build_tile(tile_id=tile_id, storage_root=root,
                cap_gb=100, progress_log=False)

        assert result.success is False
        assert result.failed_step == 'extent_validation'
        assert 'wrong size' in result.error_summary.lower()
        assert [s.name for s in result.steps] == ['download', 'crop_and_fill', 'extent_validation']
        reg = load_registry(root)
        e = reg.get(tile_id)
        assert e is not None
        assert e.status == STATUS_FAILED
        assert 'extent_validation' in e.failed_reason.lower()
        print('  test_step2b_extent_validation_fires_on_bad_crop_output OK')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    # Stub out the existing-code modules at import time so the tests
    # can run without them being present in the environment.
    # The tests use @patch where they need specific mock behavior.
    import sys
    # Create stub modules for the existing-pipeline imports if they're
    # not on the path. The actual modules will be imported when running
    # against real data on the user's machine; here we just need the
    # names to exist for import-time resolution.
    for modname in ['dem_download', 'precompute_flat_mask',
                    'dem_water_correction', 'repair_dem_gaps']:
        if modname not in sys.modules:
            sys.modules[modname] = MagicMock()
    test_step_crop_accepts_epsg_4269()
    test_step_crop_accepts_epsg_4326()
    test_step_crop_rejects_projected_crs()
    test_step_crop_canonical_grid_alignment()
    test_step_crop_keep_original_writes_gap_mask()
    test_step_crop_without_keep_original_skips_gap_mask()
    test_step_flat_mask_synthetic()
    test_step_flat_mask_validation_gate()
    test_step_flat_mask_none_elev_range_fails()
    test_step_flat_mask_high_pct_with_real_relief_passes()
    test_step_flat_mask_high_pct_with_degenerate_range_fails()
    test_step_boundary_mask_validation()
    test_step_water_correction_keep_original_writes_precorrection_and_stats()
    test_step_water_correction_without_keep_original_skips_extra_artifacts()
    test_compute_water_diagnostics_no_polygons()
    test_compute_water_diagnostics_flat_lake()
    test_compute_water_diagnostics_anomalous_lake()
    test_compute_water_diagnostics_cross_tile()
    test_skip_if_ready()
    test_invalid_tile_id()
    test_capacity_refusal()
    test_download_failure_marks_failed()
    test_step2b_extent_validation_fires_on_bad_crop_output()
    print('\nAll tile_builder tests PASS')