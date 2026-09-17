"""
test_dem_download_1m.py
━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for the true native-resolution (~1m/px) 3DEP path (distinct
from the ~3m tier in test_dem_download_3m.py):
- dem_download.probe_3dep_1m_coverage_grid: parallel coarse coverage probing.
- elevation._download_tile_3dep_native: windowed streaming fetch that
  skips sub-tiles outside probed coverage and never assembles the
  whole tile in memory.
- dem_download.download_tile_3dep_1m: the orchestration function
  (probe -> native fetch -> downsample to canonical grid).

All network calls are mocked -- these never hit a real USGS endpoint.
Runs standalone (no pytest required), same convention as this repo's
other test_*.py files.
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import rasterio
from rasterio.transform import from_bounds

import dem_download
import elevation


def _fake_subtile_bytes(xmin, ymin, xmax, ymax, px, value):
    arr = np.full((px, px), value, dtype=np.float32)
    transform = from_bounds(xmin, ymin, xmax, ymax, px, px)
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'width': px, 'height': px, 'crs': 'EPSG:4326',
        'transform': transform, 'nodata': -9999.0,
    }
    with rasterio.io.MemoryFile() as mf:
        with mf.open(**profile) as ds:
            ds.write(arr, 1)
        return mf.read()


# ── probe_3dep_1m_coverage_grid ──────────────────────────────────────

def test_coverage_grid_shape_and_request_count():
    calls = []

    def fake_probe(west, south, east, north, probe_px=64, timeout=15):
        calls.append((west, south, east, north))
        return True

    with patch('dem_download._probe_3dep_coverage_bbox', side_effect=fake_probe):
        coverage = dem_download.probe_3dep_1m_coverage_grid(39, -105, grid_n=5)

    assert coverage.shape == (5, 5)
    assert coverage.all()
    assert len(calls) == 25
    print('  test_coverage_grid_shape_and_request_count OK')


def test_coverage_grid_reflects_mixed_results():
    def fake_probe(west, south, east, north, probe_px=64, timeout=15):
        # "Covered" only in the western half of the tile.
        return west < -104.5

    with patch('dem_download._probe_3dep_coverage_bbox', side_effect=fake_probe):
        coverage = dem_download.probe_3dep_1m_coverage_grid(39, -105, grid_n=4)

    assert coverage[:, :2].all()
    assert not coverage[:, 2:].any()
    print('  test_coverage_grid_reflects_mixed_results OK')


# ── elevation._download_tile_3dep_native ─────────────────────────────

def test_native_fetch_skips_uncovered_subtiles_without_fetching():
    lat_floor, lng_floor = 39, -105
    grid_n, subtile_px = 4, 50
    coverage = np.zeros((2, 2), dtype=bool)
    coverage[:, 0] = True  # west half covered, east half not

    fetch_calls = []

    def fake_fetch(url, timeout=60, label=""):
        fetch_calls.append(label)
        # Extract the requested bbox back out of the URL to return a
        # plausible constant-value tile for it.
        import re
        m = re.search(r'BBOX=([^&]+)', url)
        xmin, ymin, xmax, ymax = [float(x) for x in m.group(1).split(',')]
        return _fake_subtile_bytes(xmin, ymin, xmax, ymax, subtile_px, 500.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / 'native.tif'
        with patch('elevation._fetch_wcs_geotiff_bytes', side_effect=fake_fetch):
            elevation._download_tile_3dep_native(
                lat_floor, lng_floor, out_path,
                coverage_grid=(2, coverage),
                grid_n=grid_n, subtile_px=subtile_px, max_workers=4)

        # Only the covered (west) half's sub-tiles should have been
        # fetched -- 4 rows x 2 west columns = 8 of the 16 total.
        assert len(fetch_calls) == 8

        with rasterio.open(out_path) as ds:
            assert ds.width == grid_n * subtile_px
            data = ds.read(1)
            west_half = data[:, :ds.width // 2]
            east_half = data[:, ds.width // 2:]
            assert (west_half == 500.0).all()
            assert (east_half == ds.nodata).all()
    print('  test_native_fetch_skips_uncovered_subtiles_without_fetching OK')


def test_native_fetch_leaves_nodata_on_failed_subtile():
    lat_floor, lng_floor = 39, -105
    grid_n, subtile_px = 2, 40

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / 'native.tif'
        # Every fetch "fails" (server error / timeout) -- output
        # should still be a valid, fully-nodata raster, not a crash.
        with patch('elevation._fetch_wcs_geotiff_bytes', return_value=None):
            elevation._download_tile_3dep_native(
                lat_floor, lng_floor, out_path,
                coverage_grid=None,  # fetch everything
                grid_n=grid_n, subtile_px=subtile_px, max_workers=2)

        with rasterio.open(out_path) as ds:
            data = ds.read(1)
            assert (data == ds.nodata).all()
    print('  test_native_fetch_leaves_nodata_on_failed_subtile OK')


def test_native_fetch_default_grid_math():
    # 1 degree (~111320m) / 2000px-per-subtile -> 56 sub-tiles/side,
    # ~0.994m/px -- confirms the module constants weren't hand-typed
    # wrong (this is what --resolution 1m actually requests).
    assert elevation.NATIVE_1M_SUBTILE_PX == 2000
    assert elevation.NATIVE_1M_GRID_N == 56
    total_px = elevation.NATIVE_1M_GRID_N * elevation.NATIVE_1M_SUBTILE_PX
    px_size_m = 111_320 / total_px
    assert 0.9 < px_size_m < 1.1
    print('  test_native_fetch_default_grid_math OK')


# ── dem_download.download_tile_3dep_1m ───────────────────────────────

def test_native_1m_skips_probe_outside_us_bounds():
    with patch('dem_download._probe_3dep_1m_coverage') as mock_probe:
        result = dem_download.download_tile_3dep_1m(45, 10, '/tmp/whatever')
        assert result is None
        mock_probe.assert_not_called()
    print('  test_native_1m_skips_probe_outside_us_bounds OK')


def test_native_1m_returns_none_when_no_coverage_at_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('dem_download._probe_3dep_1m_coverage', return_value=True), \
             patch('dem_download.probe_3dep_1m_coverage_grid',
                   return_value=np.zeros((10, 10), dtype=bool)) as mock_grid, \
             patch('elevation._download_tile_3dep_native') as mock_fetch:
            result = dem_download.download_tile_3dep_1m(45, -110, tmpdir)
            assert result is None
            mock_grid.assert_called_once()
            mock_fetch.assert_not_called()
    print('  test_native_1m_returns_none_when_no_coverage_at_all OK')


def test_native_1m_happy_path_downsamples_and_persists_native():
    with tempfile.TemporaryDirectory() as tmpdir:
        src_px = 40
        arr = np.full((src_px, src_px), 1500.0, dtype=np.float32)
        arr[:, src_px // 2:] = 2500.0
        transform = from_bounds(-110, 45, -109, 46, src_px, src_px)
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
            'width': src_px, 'height': src_px, 'crs': 'EPSG:4326',
            'transform': transform, 'nodata': -9999.0,
        }

        def fake_native_fetch(lat_floor, lng_floor, out_path, **kwargs):
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(arr, 1)
            return out_path

        canonical_px = 60
        with patch('dem_download._probe_3dep_1m_coverage', return_value=True), \
             patch('dem_download.probe_3dep_1m_coverage_grid',
                   return_value=np.ones((10, 10), dtype=bool)), \
             patch('elevation._download_tile_3dep_native',
                   side_effect=fake_native_fetch), \
             patch('dem_download._CANONICAL_GRID_PX', canonical_px):
            result = dem_download.download_tile_3dep_1m(45, -110, tmpdir)

        assert result is not None
        out_path, label = result
        assert label == '1m'
        assert out_path.exists()

        native_path = (Path(tmpdir) / 'raw' /
                        f"{dem_download.tile_label(45, -110)}_1m_native.tif")
        assert native_path.exists()

        with rasterio.open(out_path) as ds:
            assert ds.width == canonical_px and ds.height == canonical_px
            data = ds.read(1)
            mean = float(data[data != ds.nodata].mean())
            assert 1900.0 < mean < 2100.0  # averages the 1500/2500 split
    print('  test_native_1m_happy_path_downsamples_and_persists_native OK')


def test_native_1m_cached_output_short_circuits_refetch():
    # Deliberately writes the cache fixtures directly (uncompressed,
    # matching test_dem_download_3m.py's own cache-hit test) rather
    # than chaining off a real resample -- deflate compresses a small
    # synthetic constant-value raster so well that even a few hundred
    # pixels square lands under the 10,000-byte cache-hit threshold,
    # which would make this test pass for the wrong reason.
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / 'raw'
        raw_dir.mkdir()
        out_path = raw_dir / f"{dem_download.tile_label(45, -110)}_1m.tif"
        native_path = raw_dir / f"{dem_download.tile_label(45, -110)}_1m_native.tif"
        px = 100
        arr = np.full((px, px), 1234.0, dtype=np.float32)
        transform = from_bounds(-110, 45, -109, 46, px, px)
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
            'width': px, 'height': px, 'crs': 'EPSG:4326',
            'transform': transform, 'nodata': -9999.0,
        }
        for p in (out_path, native_path):
            with rasterio.open(p, 'w', **profile) as dst:
                dst.write(arr, 1)

        # The cheap whole-tile probe still runs even on a cache hit
        # (same established pattern as download_tile_3dep_3m) -- what
        # must be skipped is the expensive coarse-grid probe and the
        # full windowed fetch.
        with patch('dem_download._probe_3dep_1m_coverage', return_value=True), \
             patch('dem_download.probe_3dep_1m_coverage_grid') as mock_grid, \
             patch('elevation._download_tile_3dep_native') as mock_fetch:
            result = dem_download.download_tile_3dep_1m(45, -110, tmpdir)
            assert result == (out_path, '1m')
            mock_grid.assert_not_called()
            mock_fetch.assert_not_called()
    print('  test_native_1m_cached_output_short_circuits_refetch OK')


def test_download_tile_dispatches_native_1m_explicitly():
    fake_result = (Path('/tmp/fake_native.tif'), '1m')
    with patch('dem_download.download_tile_3dep_1m',
               return_value=fake_result) as mock_native, \
         patch('dem_download.get_candidates') as mock_candidates:
        result = dem_download.download_tile(45, -110, '/tmp/whatever',
                                              resolution='1m')
        assert result == fake_result
        mock_native.assert_called_once()
        mock_candidates.assert_not_called()
    print('  test_download_tile_dispatches_native_1m_explicitly OK')


def test_1m_falls_back_to_10m_30m_cascade_when_no_coverage():
    # Unlike '3m'/'10m'/'30m' (explicit-only, no fallback), a
    # resolution='1m' request over a whole study area should still
    # produce a complete mosaic: a tile with no genuine 1m coverage
    # falls back to the normal 10m/GLO-30 cascade instead of being
    # left out of the mosaic entirely.
    fake_10m_result = (Path('/tmp/fake_10m.tif'), '10m')
    with patch('dem_download.download_tile_3dep_1m',
               return_value=None) as mock_native, \
         patch('dem_download.download_tile_3dep_3m') as mock_3m, \
         patch('dem_download.get_candidates',
               return_value=[('http://example.com/x.tif', '10m')]) as mock_candidates, \
         patch('dem_download.requests.get') as mock_get, \
         patch('dem_download._raster_is_readable', return_value=True):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / 'raw'
            raw_dir.mkdir()
            cached_path = raw_dir / f"{dem_download.tile_label(45, -110)}_10m.tif"
            cached_path.write_bytes(b'0' * 20_000)  # pre-cached, skips the real GET

            result = dem_download.download_tile(45, -110, tmpdir, resolution='1m')

            assert result == (cached_path, '10m')
            mock_native.assert_called_once()
            mock_3m.assert_not_called()  # the 1m tier's own fallback never touches 3m
            mock_candidates.assert_called_once()
            mock_get.assert_not_called()  # cache hit, no real network call needed
    print('  test_1m_falls_back_to_10m_30m_cascade_when_no_coverage OK')


def test_try_3m_does_not_trigger_1m():
    # try_3m must only ever reach the ~3m tier -- the true 1m tier is
    # far more expensive and always needs its own explicit request.
    with patch('dem_download.download_tile_3dep_3m',
               return_value=None) as mock_3m, \
         patch('dem_download.download_tile_3dep_1m') as mock_native, \
         patch('dem_download.get_candidates', return_value=[]):
        dem_download.download_tile(45, -110, '/tmp/whatever',
                                    resolution='best', try_3m=True)
        mock_3m.assert_called_once()
        mock_native.assert_not_called()
    print('  test_try_3m_does_not_trigger_1m OK')


if __name__ == '__main__':
    test_coverage_grid_shape_and_request_count()
    test_coverage_grid_reflects_mixed_results()
    test_native_fetch_skips_uncovered_subtiles_without_fetching()
    test_native_fetch_leaves_nodata_on_failed_subtile()
    test_native_fetch_default_grid_math()
    test_native_1m_skips_probe_outside_us_bounds()
    test_native_1m_returns_none_when_no_coverage_at_all()
    test_native_1m_happy_path_downsamples_and_persists_native()
    test_native_1m_cached_output_short_circuits_refetch()
    test_download_tile_dispatches_native_1m_explicitly()
    test_1m_falls_back_to_10m_30m_cascade_when_no_coverage()
    test_try_3m_does_not_trigger_1m()
    print('\nAll dem_download 1m-tier tests PASS')
