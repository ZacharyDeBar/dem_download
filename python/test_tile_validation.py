"""Unit tests for tile_validation.py."""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from tile_validation import validate_tile_raster
from tile_id import tile_bounds


def _tmp_root():
    return Path(tempfile.mkdtemp(prefix='tile_validation_test_'))


def _write_raster(path: Path, tile_id: str, size_px: int,
                   bounds_override=None) -> None:
    """Writes a small synthetic raster for tile_id. size_px is deliberately
    tiny (not the real 10800) -- tests pass a matching expected_size_px
    to validate_tile_raster rather than writing production-scale files.

    bounds_override, if given, is a plain (south, west, north, east)
    tuple -- lets a test write a raster with deliberately wrong/truncated
    bounds. Defaults to tile_id's own real expected bounds."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    if bounds_override is not None:
        south, west, north, east = bounds_override
    else:
        b = tile_bounds(tile_id)
        south, west, north, east = b.south, b.west, b.north, b.east

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=size_px, width=size_px, count=1, dtype='float32',
        crs=CRS.from_string('EPSG:4326'),
        transform=from_bounds(west, south, east, north, size_px, size_px),
        nodata=-9999.0,
    ) as dst:
        dst.write(np.full((size_px, size_px), 1500.0, dtype=np.float32), 1)


def test_correct_size_and_bounds_passes():
    root = _tmp_root()
    try:
        path = root / 'N45W110_dem.tif'
        _write_raster(path, 'N45W110', size_px=10)
        result = validate_tile_raster(path, 'N45W110', expected_size_px=10)
        assert result.ok, result.reason
        assert result.width == 10 and result.height == 10
        print('  test_correct_size_and_bounds_passes OK')
    finally:
        shutil.rmtree(root)


def test_truncated_size_fails():
    """Reproduces the real incident's signature: correct origin/pixel-size
    convention but fewer rows/cols than the full tile footprint."""
    root = _tmp_root()
    try:
        path = root / 'N45W110_dem.tif'
        full = tile_bounds('N45W110')
        # Truncate the northern portion -- same west/south, short north/height,
        # matching the real incident's failure signature exactly.
        truncated_bounds = (full.south, full.west, full.south + 0.7, full.east)
        _write_raster(path, 'N45W110', size_px=7, bounds_override=truncated_bounds)
        result = validate_tile_raster(path, 'N45W110', expected_size_px=10)
        assert not result.ok
        assert 'wrong size' in result.reason or 'bounds' in result.reason
        print('  test_truncated_size_fails OK')
    finally:
        shutil.rmtree(root)


def test_wrong_bounds_but_right_size_fails():
    """Correct pixel dimensions but shifted geographic bounds (e.g. wrong
    tile_id was matched to this file) must also be flagged -- size alone
    isn't sufficient."""
    root = _tmp_root()
    try:
        path = root / 'N45W110_dem.tif'
        wb = tile_bounds('N45W111')  # one tile west
        _write_raster(path, 'N45W110', size_px=10,
                       bounds_override=(wb.south, wb.west, wb.north, wb.east))
        result = validate_tile_raster(path, 'N45W110', expected_size_px=10)
        assert not result.ok
        assert 'bounds' in result.reason
        print('  test_wrong_bounds_but_right_size_fails OK')
    finally:
        shutil.rmtree(root)


def test_missing_file_fails():
    root = _tmp_root()
    try:
        result = validate_tile_raster(root / 'nonexistent.tif', 'N45W110')
        assert not result.ok
        assert 'missing' in result.reason
        print('  test_missing_file_fails OK')
    finally:
        shutil.rmtree(root)


def test_unreadable_file_fails():
    root = _tmp_root()
    try:
        path = root / 'not_a_tif.tif'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'not a real geotiff')
        result = validate_tile_raster(path, 'N45W110')
        assert not result.ok
        assert 'unreadable' in result.reason
        print('  test_unreadable_file_fails OK')
    finally:
        shutil.rmtree(root)


def test_bounds_tolerance_allows_tiny_float_noise():
    """A file whose bounds differ from the exact expected footprint by
    less than bounds_tol_deg (floating-point transform noise, not a real
    truncation) must still pass."""
    root = _tmp_root()
    try:
        path = root / 'N45W110_dem.tif'
        full = tile_bounds('N45W110')
        noisy_bounds = (full.south + 1e-7, full.west - 1e-7,
                        full.north - 1e-7, full.east + 1e-7)
        _write_raster(path, 'N45W110', size_px=10, bounds_override=noisy_bounds)
        result = validate_tile_raster(path, 'N45W110', expected_size_px=10)
        assert result.ok, result.reason
        print('  test_bounds_tolerance_allows_tiny_float_noise OK')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    test_correct_size_and_bounds_passes()
    test_truncated_size_fails()
    test_wrong_bounds_but_right_size_fails()
    test_missing_file_fails()
    test_unreadable_file_fails()
    test_bounds_tolerance_allows_tiny_float_noise()
    print('\nAll tile_validation tests PASS')
