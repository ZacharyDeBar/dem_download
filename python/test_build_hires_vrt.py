"""
test_build_hires_vrt.py
━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for build_hires_vrt.py: the base+overlay compositing (real
overlay values win inside its coverage, base values show through both
outside the overlay's extent and inside it wherever the overlay itself
is nodata), CRS-mismatch rejection, and the empty-overlay-list guard.

All rasters are small synthetic fixtures -- no network, no real DEM
data. Runs standalone (no pytest required), same convention as this
repo's other test_*.py files.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from build_hires_vrt import build_hires_vrt


def _approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def _write_raster(path, west, south, east, north, size, value, nodata=-9999.0,
                   crs='EPSG:4326'):
    arr = np.full((size, size), value, dtype=np.float32)
    transform = from_bounds(west, south, east, north, size, size)
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'width': size, 'height': size, 'crs': crs,
        'transform': transform, 'nodata': nodata,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)
    return path


def test_overlay_values_win_inside_its_coverage():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Base: a 10x10 tile covering [0,1]x[0,1], all 100.0.
        base_path = _write_raster(tmp_path / 'base.tif', 0, 0, 1, 1, 10, 100.0)
        # Overlay: covers the middle third ([0.4,0.6]x[0.4,0.6]) at a
        # much finer grid, all 999.0, no nodata pixels.
        overlay_path = _write_raster(
            tmp_path / 'overlay.tif', 0.4, 0.4, 0.6, 0.6, 20, 999.0, nodata=None)

        vrt_path = build_hires_vrt(base_path, [overlay_path], tmp_path / 'out.vrt')

        with rasterio.open(vrt_path) as ds:
            # VRT grid should be at the overlay's (finer) pixel size,
            # spanning the base's full extent.
            assert _approx(ds.res[0], 0.01)  # 0.2/20 = 0.01 deg/px
            assert ds.width == 100 and ds.height == 100

            data = ds.read(1)
            # Center pixel (inside overlay coverage) should read the
            # overlay's real value.
            cy, cx = ds.height // 2, ds.width // 2
            assert _approx(data[cy, cx], 999.0)
            # A corner pixel (outside overlay coverage entirely) should
            # fall back to the base value.
            assert _approx(data[0, 0], 100.0)
    print('  test_overlay_values_win_inside_its_coverage OK')


def test_overlay_nodata_lets_base_show_through():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        base_path = _write_raster(tmp_path / 'base.tif', 0, 0, 1, 1, 10, 100.0)

        # Overlay covers [0.4,0.6]x[0.4,0.6] at 20x20px, but only the
        # inner 10x10 block has real data -- the rest is the overlay's
        # own nodata, simulating a 1m tile with partial internal
        # coverage (e.g. a surveyed strip inside a mostly-empty tile).
        overlay_size = 20
        arr = np.full((overlay_size, overlay_size), -9999.0, dtype=np.float32)
        arr[5:15, 5:15] = 999.0
        transform = from_bounds(0.4, 0.4, 0.6, 0.6, overlay_size, overlay_size)
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
            'width': overlay_size, 'height': overlay_size, 'crs': 'EPSG:4326',
            'transform': transform, 'nodata': -9999.0,
        }
        overlay_path = tmp_path / 'overlay.tif'
        with rasterio.open(overlay_path, 'w', **profile) as dst:
            dst.write(arr, 1)

        vrt_path = build_hires_vrt(base_path, [overlay_path], tmp_path / 'out.vrt')

        with rasterio.open(vrt_path) as ds:
            data = ds.read(1)
            cy, cx = ds.height // 2, ds.width // 2
            # Center (inside the overlay's real-data block) -> overlay wins.
            assert _approx(data[cy, cx], 999.0)
            # Just inside the overlay's bounding box, but in the region
            # where the overlay itself is nodata -> base shows through,
            # not the overlay's nodata value.
            edge_row = int((1 - 0.42) / (1.0 / ds.height))
            edge_col = int(0.42 / (1.0 / ds.width))
            assert _approx(data[edge_row, edge_col], 100.0)
    print('  test_overlay_nodata_lets_base_show_through OK')


def test_multiple_overlays_all_composited():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        base_path = _write_raster(tmp_path / 'base.tif', 0, 0, 2, 1, 20, 100.0)
        overlay_a = _write_raster(
            tmp_path / 'a.tif', 0.1, 0.1, 0.3, 0.3, 10, 1.0, nodata=None)
        overlay_b = _write_raster(
            tmp_path / 'b.tif', 1.5, 0.5, 1.7, 0.7, 10, 2.0, nodata=None)

        vrt_path = build_hires_vrt(
            base_path, [overlay_a, overlay_b], tmp_path / 'out.vrt')

        with rasterio.open(vrt_path) as ds:
            data = ds.read(1)
            row_a = ds.height - int(0.2 / ds.res[1]) - 1
            col_a = int(0.2 / ds.res[0])
            row_b = ds.height - int(0.6 / ds.res[1]) - 1
            col_b = int(1.6 / ds.res[0])
            assert _approx(data[row_a, col_a], 1.0)
            assert _approx(data[row_b, col_b], 2.0)
    print('  test_multiple_overlays_all_composited OK')


def test_rejects_empty_overlay_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        base_path = _write_raster(tmp_path / 'base.tif', 0, 0, 1, 1, 10, 100.0)
        try:
            build_hires_vrt(base_path, [], tmp_path / 'out.vrt')
            assert False, "expected ValueError"
        except ValueError:
            pass
    print('  test_rejects_empty_overlay_list OK')


def test_rejects_crs_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        base_path = _write_raster(tmp_path / 'base.tif', 0, 0, 1, 1, 10, 100.0)
        overlay_path = _write_raster(
            tmp_path / 'overlay.tif', 0.4, 0.4, 0.6, 0.6, 20, 999.0,
            nodata=None, crs='EPSG:3857')
        try:
            build_hires_vrt(base_path, [overlay_path], tmp_path / 'out.vrt')
            assert False, "expected ValueError"
        except ValueError:
            pass
    print('  test_rejects_crs_mismatch OK')


if __name__ == '__main__':
    test_overlay_values_win_inside_its_coverage()
    test_overlay_nodata_lets_base_show_through()
    test_multiple_overlays_all_composited()
    test_rejects_empty_overlay_list()
    test_rejects_crs_mismatch()
    print('\nAll build_hires_vrt tests PASS')
