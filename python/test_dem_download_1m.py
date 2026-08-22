"""
test_dem_download_1m.py
━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for dem_download.py's 3DEP 1m tier: the coverage probe
(_probe_3dep_1m_coverage), the fetch+resample orchestration
(download_tile_3dep_1m), and download_tile()'s opt-in gating (1m is
NOT attempted under the default 'best' cascade unless try_1m=True --
sparse coverage + an expensive raw fetch made this too costly to run
on every tile by default). All network calls and elevation.py's
sub-tile fetcher are mocked -- these never hit a real USGS endpoint.

Runs standalone (no pytest required), same convention as this repo's
other test_*.py files -- see README.md's Tests section.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import rasterio
from rasterio.transform import from_bounds

import dem_download


def _fake_probe_response(width, height, valid_fraction, nodata=-9999.0):
    """Build a small in-memory GeoTIFF's raw bytes, mimicking a WCS
    GetCoverage response, with a given fraction of real (non-nodata)
    pixels -- used to drive _probe_3dep_1m_coverage's coverage decision."""
    arr = np.full((height, width), nodata, dtype=np.float32)
    n_valid = int(width * height * valid_fraction)
    arr.flat[:n_valid] = 1500.0  # plausible elevation
    transform = from_bounds(-110.0005, 44.9995, -109.9995, 45.0005, width, height)
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'width': width, 'height': height, 'crs': 'EPSG:4326',
        'transform': transform, 'nodata': nodata,
    }
    with rasterio.io.MemoryFile() as mf:
        with mf.open(**profile) as ds:
            ds.write(arr, 1)
        return bytes(mf.read())


def _fake_tif_bytes(width, height, value, nodata=-9999.0):
    """A small real GeoTIFF on disk, used as a stand-in for
    _download_tile_3dep_subtiled's return value."""
    arr = np.full((height, width), value, dtype=np.float32)
    transform = from_bounds(-110, 44, -109, 45, width, height)
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'width': width, 'height': height, 'crs': 'EPSG:4326',
        'transform': transform, 'nodata': nodata,
    }
    return arr, profile


# ── _probe_3dep_1m_coverage ─────────────────────────────────────────

def test_probe_returns_false_on_non_200():
    resp = MagicMock(status_code=404)
    with patch('dem_download.requests.get', return_value=resp):
        assert dem_download._probe_3dep_1m_coverage(45, -110) is False
    print('  test_probe_returns_false_on_non_200 OK')


def test_probe_returns_false_when_mostly_nodata():
    data = _fake_probe_response(64, 64, valid_fraction=0.1)
    resp = MagicMock(status_code=200, content=data)
    with patch('dem_download.requests.get', return_value=resp):
        assert dem_download._probe_3dep_1m_coverage(45, -110) is False
    print('  test_probe_returns_false_when_mostly_nodata OK')


def test_probe_returns_true_when_real_coverage():
    data = _fake_probe_response(64, 64, valid_fraction=0.9)
    resp = MagicMock(status_code=200, content=data)
    with patch('dem_download.requests.get', return_value=resp):
        assert dem_download._probe_3dep_1m_coverage(45, -110) is True
    print('  test_probe_returns_true_when_real_coverage OK')


def test_probe_returns_false_on_request_exception():
    with patch('dem_download.requests.get', side_effect=ConnectionError('boom')):
        assert dem_download._probe_3dep_1m_coverage(45, -110) is False
    print('  test_probe_returns_false_on_request_exception OK')


# ── download_tile_3dep_1m ───────────────────────────────────────────

def test_skips_probe_outside_us_bounds():
    with patch('dem_download._probe_3dep_1m_coverage') as mock_probe:
        result = dem_download.download_tile_3dep_1m(45, 10, '/tmp/whatever')
        assert result is None
        mock_probe.assert_not_called()
    print('  test_skips_probe_outside_us_bounds OK')


def test_returns_none_when_probe_finds_no_coverage():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('dem_download._probe_3dep_1m_coverage', return_value=False):
            with patch('elevation._download_tile_3dep_subtiled') as mock_fetch:
                result = dem_download.download_tile_3dep_1m(45, -110, tmpdir)
                assert result is None
                mock_fetch.assert_not_called()
    print('  test_returns_none_when_probe_finds_no_coverage OK')


def test_returns_none_when_subtiled_fetch_raises():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('dem_download._probe_3dep_1m_coverage', return_value=True), \
             patch('elevation._download_tile_3dep_subtiled',
                   side_effect=RuntimeError('WCS down')):
            result = dem_download.download_tile_3dep_1m(45, -110, tmpdir)
            assert result is None
    print('  test_returns_none_when_subtiled_fetch_raises OK')


def test_returns_none_when_fetch_degraded_to_glo30_fallback():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # A "degraded" result: narrow width, well under the real-1m
        # threshold -- simulates elevation.py's own internal
        # all-sub-tiles-failed -> GLO-30 fallback firing.
        arr, profile = _fake_tif_bytes(3600, 3600, 1500.0)
        degraded_path = Path(tmpdir) / 'degraded.tif'
        with rasterio.open(degraded_path, 'w', **profile) as dst:
            dst.write(arr, 1)

        with patch('dem_download._probe_3dep_1m_coverage', return_value=True), \
             patch('elevation._download_tile_3dep_subtiled',
                   return_value=degraded_path), \
             patch('dem_download._1M_DEGRADED_WIDTH_THRESHOLD_PX', 5000):
            result = dem_download.download_tile_3dep_1m(45, -110, tmpdir)
            assert result is None
    print('  test_returns_none_when_fetch_degraded_to_glo30_fallback OK')


def test_happy_path_resamples_onto_canonical_grid_with_averaging():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Shrink the "real 1m" grid and the canonical output grid so
        # this test runs fast while exercising the real reproject/
        # average code path, not a special test-only branch.
        src_px = 900  # stands in for the real ~108000px sub-tiled fetch
        canonical_px = 90  # stands in for the real 10800px canonical grid

        # Half the source at 1000.0, half at 2000.0 -- area-averaging
        # onto a coarser grid should land close to the 1500.0 mean,
        # NOT snap to one of the two source values the way a
        # nearest/bilinear-at-sparse-points resample plausibly could.
        arr = np.full((src_px, src_px), 1000.0, dtype=np.float32)
        arr[:, src_px // 2:] = 2000.0
        # Must match the (lat=45, lng=-110) tile's actual bounds --
        # tile_id convention names a tile by its SW corner and extends
        # north/east from there, i.e. south=45/north=46, not 44/45.
        transform = from_bounds(-110, 45, -109, 46, src_px, src_px)
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
            'width': src_px, 'height': src_px, 'crs': 'EPSG:4326',
            'transform': transform, 'nodata': -9999.0,
        }
        raw_path = Path(tmpdir) / 'raw_1m.tif'
        with rasterio.open(raw_path, 'w', **profile) as dst:
            dst.write(arr, 1)

        with patch('dem_download._probe_3dep_1m_coverage', return_value=True), \
             patch('elevation._download_tile_3dep_subtiled',
                   return_value=raw_path), \
             patch('dem_download._1M_DEGRADED_WIDTH_THRESHOLD_PX', 100), \
             patch('dem_download._1M_CANONICAL_GRID_PX', canonical_px):
            result = dem_download.download_tile_3dep_1m(45, -110, tmpdir)

        assert result is not None
        out_path, res_label = result
        assert res_label == '1m'
        assert out_path.exists()

        with rasterio.open(out_path) as ds:
            assert ds.width == canonical_px
            assert ds.height == canonical_px
            data = ds.read(1)
            valid = data[data != ds.nodata]
            mean = float(valid.mean())
            # Should land near the true 1500.0 average, not collapse
            # to either source value.
            assert 1400.0 < mean < 1600.0, f"unexpected mean {mean}"
            assert data.min() > 900.0
            assert data.max() < 2100.0
    print('  test_happy_path_resamples_onto_canonical_grid_with_averaging OK')


def test_cached_output_short_circuits_refetch():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / 'raw'
        raw_dir.mkdir()
        cached_path = raw_dir / f"{dem_download.tile_label(45, -110)}_1m.tif"
        # Must be genuinely > 10,000 bytes to pass the real cache-hit
        # size check in download_tile_3dep_1m -- a too-small fixture
        # would fall through to the real fetch path instead of
        # exercising the cache short-circuit this test is for.
        px = 100
        arr = np.full((px, px), 1234.0, dtype=np.float32)
        transform = from_bounds(-110, 45, -109, 46, px, px)
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
            'width': px, 'height': px, 'crs': 'EPSG:4326',
            'transform': transform, 'nodata': -9999.0,
        }
        with rasterio.open(cached_path, 'w', **profile) as dst:
            dst.write(arr, 1)

        with patch('dem_download._probe_3dep_1m_coverage', return_value=True), \
             patch('elevation._download_tile_3dep_subtiled') as mock_fetch:
            result = dem_download.download_tile_3dep_1m(45, -110, tmpdir)
            assert result == (cached_path, '1m')
            mock_fetch.assert_not_called()
    print('  test_cached_output_short_circuits_refetch OK')


# ── download_tile()'s opt-in gating ─────────────────────────────────

def test_best_resolution_does_not_try_1m_by_default():
    with patch('dem_download.download_tile_3dep_1m') as mock_1m, \
         patch('dem_download.get_candidates', return_value=[]):
        result = dem_download.download_tile(45, -110, '/tmp/whatever',
                                              resolution='best')
        assert result is None
        mock_1m.assert_not_called()
    print('  test_best_resolution_does_not_try_1m_by_default OK')


def test_best_resolution_tries_1m_when_opted_in():
    fake_result = (Path('/tmp/fake_1m.tif'), '1m')
    with patch('dem_download.download_tile_3dep_1m',
               return_value=fake_result) as mock_1m:
        result = dem_download.download_tile(45, -110, '/tmp/whatever',
                                              resolution='best', try_1m=True)
        assert result == fake_result
        mock_1m.assert_called_once()
    print('  test_best_resolution_tries_1m_when_opted_in OK')


def test_explicit_1m_resolution_tries_it_regardless_of_try_1m_flag():
    with patch('dem_download.download_tile_3dep_1m',
               return_value=None) as mock_1m, \
         patch('dem_download.get_candidates') as mock_candidates:
        result = dem_download.download_tile(45, -110, '/tmp/whatever',
                                              resolution='1m', try_1m=False)
        # Explicit '1m' still attempts it even with try_1m=False --
        # and does NOT silently fall through to 10m/GLO-30 on failure,
        # matching how explicit '10m'/'30m' already behave.
        assert result is None
        mock_1m.assert_called_once()
        mock_candidates.assert_not_called()
    print('  test_explicit_1m_resolution_tries_it_regardless_of_try_1m_flag OK')


if __name__ == '__main__':
    test_probe_returns_false_on_non_200()
    test_probe_returns_false_when_mostly_nodata()
    test_probe_returns_true_when_real_coverage()
    test_probe_returns_false_on_request_exception()
    test_skips_probe_outside_us_bounds()
    test_returns_none_when_probe_finds_no_coverage()
    test_returns_none_when_subtiled_fetch_raises()
    test_returns_none_when_fetch_degraded_to_glo30_fallback()
    test_happy_path_resamples_onto_canonical_grid_with_averaging()
    test_cached_output_short_circuits_refetch()
    test_best_resolution_does_not_try_1m_by_default()
    test_best_resolution_tries_1m_when_opted_in()
    test_explicit_1m_resolution_tries_it_regardless_of_try_1m_flag()
    print('\nAll dem_download 1m-tier tests PASS')
