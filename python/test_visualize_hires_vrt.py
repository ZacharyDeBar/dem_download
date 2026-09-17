"""
test_visualize_hires_vrt.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for visualize_hires_vrt.py: parsing tile filenames out of
download_manifest.json (including the ocean-tile naming exception),
enumerating a study area's tile grid, locating native hi-res rasters
on disk, computing per-cell coverage fraction from a native raster's
mask band, and the end-to-end resolution-grid/report assembly.

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

import visualize_hires_vrt as vhv


def _write_raster(path, west, south, east, north, size, value, nodata=-9999.0):
    arr = np.full((size, size), value, dtype=np.float32)
    transform = from_bounds(west, south, east, north, size, size)
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'width': size, 'height': size, 'crs': 'EPSG:4326',
        'transform': transform, 'nodata': nodata,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)
    return path


# ── filename / manifest parsing ──────────────────────────────────────

def test_parse_tile_filename_standard():
    assert vhv._parse_tile_filename('raw/N45W110_3m.tif') == ('N45W110', '3m')
    assert vhv._parse_tile_filename('N45W110_10m.tif') == ('N45W110', '10m')
    assert vhv._parse_tile_filename('S13E178_30m.tif') == ('S13E178', '30m')
    print('  test_parse_tile_filename_standard OK')


def test_parse_tile_filename_ocean_exception():
    # _create_ocean_tile() names files differently from every other
    # tier (underscore between lat/lng, 'ocean0m' suffix) -- see
    # dem_download.py's own comment on that historical bug.
    assert vhv._parse_tile_filename('raw/N45_W110_ocean0m.tif') == ('N45W110', 'ocean')
    assert vhv._parse_tile_filename('S05_E172_ocean0m.tif') == ('S05E172', 'ocean')
    print('  test_parse_tile_filename_ocean_exception OK')


def test_parse_tile_filename_unrecognized_returns_none():
    assert vhv._parse_tile_filename('readme.txt') is None
    assert vhv._parse_tile_filename('not_a_tile.tif') is None
    print('  test_parse_tile_filename_unrecognized_returns_none OK')


def test_tile_resolution_map_drops_unrecognized_entries():
    manifest = {'tile_files': [
        'raw/N45W110_3m.tif',
        'raw/N45W111_10m.tif',
        'raw/N46_W110_ocean0m.tif',
        'raw/garbage.tif',
    ]}
    result = vhv._tile_resolution_map(manifest)
    assert result == {'N45W110': '3m', 'N45W111': '10m', 'N46W110': 'ocean'}
    print('  test_tile_resolution_map_drops_unrecognized_entries OK')


# ── tile grid enumeration ────────────────────────────────────────────

def test_all_tile_ids_in_bounds_single_tile():
    assert vhv._all_tile_ids_in_bounds((45, -111, 46, -110)) == ['N45W111']
    print('  test_all_tile_ids_in_bounds_single_tile OK')


def test_all_tile_ids_in_bounds_2x2_grid():
    ids = vhv._all_tile_ids_in_bounds((45, -111, 47, -109))
    assert set(ids) == {'N45W111', 'N45W110', 'N46W111', 'N46W110'}
    assert len(ids) == 4
    print('  test_all_tile_ids_in_bounds_2x2_grid OK')


# ── native raster discovery ──────────────────────────────────────────

def test_native_tile_path_prefers_1m_over_3m():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / 'raw'
        raw_dir.mkdir()
        _write_raster(raw_dir / 'N45W110_1m_native.tif', -110, 45, -109, 46, 5, 1500.0)
        _write_raster(raw_dir / 'N45W110_3m_native.tif', -110, 45, -109, 46, 5, 1500.0)
        result = vhv._native_tile_path(Path(tmpdir), 'N45W110')
        assert result[1] == '1m'
    print('  test_native_tile_path_prefers_1m_over_3m OK')


def test_native_tile_path_none_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert vhv._native_tile_path(Path(tmpdir), 'N45W110') is None
    print('  test_native_tile_path_none_when_missing OK')


# ── coverage fraction ─────────────────────────────────────────────────

def test_coverage_fraction_half_covered():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'native.tif'
        size = 40
        arr = np.full((size, size), -9999.0, dtype=np.float32)
        arr[:, :size // 2] = 1500.0  # west half real, east half nodata
        transform = from_bounds(-110, 45, -109, 46, size, size)
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
            'width': size, 'height': size, 'crs': 'EPSG:4326',
            'transform': transform, 'nodata': -9999.0,
        }
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(arr, 1)

        frac = vhv._coverage_fraction(path, out_shape=(2, 2))
        # West column of the 2x2 output should read ~fully covered,
        # east column ~fully uncovered.
        assert (frac[:, 0] > 0.9).all()
        assert (frac[:, 1] < 0.1).all()
    print('  test_coverage_fraction_half_covered OK')


# ── resolution grid assembly ─────────────────────────────────────────

def _write_manifest(output_dir, bounds, tile_files):
    import json
    manifest = {
        'study_area': 'test_area', 'bounds': list(bounds),
        'resolution': 'best', 'total_tiles': len(tile_files),
        'downloaded': len(tile_files), 'failed_tiles': [],
        'tile_files': tile_files,
    }
    with open(output_dir / 'download_manifest.json', 'w') as f:
        json.dump(manifest, f)
    return manifest


def test_build_resolution_grid_single_tile_no_native():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        manifest = _write_manifest(output_dir, (45, -111, 46, -110),
                                    ['raw/N45W111_10m.tif'])
        rank_grid, extent, stats = vhv.build_resolution_grid(output_dir, manifest, max_dim=20)
        assert (rank_grid == vhv.RESOLUTION_RANK['10m']).all()
        assert extent == (-111, -110, 45, 46)
        assert len(stats) == 1 and stats[0]['native'] is None
    print('  test_build_resolution_grid_single_tile_no_native OK')


def test_build_resolution_grid_reports_partial_native_coverage_as_a_stat():
    # A "3m"-tier tile's own internal gaps (sub-tiles the real 3DEP
    # fetch missed, already gap-filled from GLO-30 *inside* the same
    # raster by _download_tile_3dep_subtiled) are NOT a different
    # recorded resolution -- the whole tile is still gridded and
    # labeled "3m". So the map (rank_grid) stays uniformly "3m" across
    # the tile; the partial-coverage signal belongs in coverage_pct,
    # not as a different color painted over part of the tile.
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        raw_dir = output_dir / 'raw'
        raw_dir.mkdir()
        manifest = _write_manifest(output_dir, (45, -111, 46, -110),
                                    ['raw/N45W111_3m.tif'])

        # Native raster with real data only in the western half.
        size = 40
        arr = np.full((size, size), -9999.0, dtype=np.float32)
        arr[:, :size // 2] = 1500.0
        transform = from_bounds(-111, 45, -110, 46, size, size)
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
            'width': size, 'height': size, 'crs': 'EPSG:4326',
            'transform': transform, 'nodata': -9999.0,
        }
        with rasterio.open(raw_dir / 'N45W111_3m_native.tif', 'w', **profile) as dst:
            dst.write(arr, 1)

        rank_grid, extent, stats = vhv.build_resolution_grid(output_dir, manifest, max_dim=20)
        assert (rank_grid == vhv.RESOLUTION_RANK['3m']).all()
        assert stats[0]['native'] == '3m'
        assert 40 < stats[0]['coverage_pct'] < 60
    print('  test_build_resolution_grid_reports_partial_native_coverage_as_a_stat OK')


def test_build_resolution_grid_marks_failed_tile_as_no_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        # 2x1 area, only the west tile actually downloaded.
        manifest = _write_manifest(output_dir, (45, -111, 46, -109),
                                    ['raw/N45W111_10m.tif'])
        rank_grid, extent, stats = vhv.build_resolution_grid(output_dir, manifest, max_dim=20)
        west_cols = rank_grid[:, :rank_grid.shape[1] // 2]
        east_cols = rank_grid[:, rank_grid.shape[1] // 2:]
        assert (west_cols == vhv.RESOLUTION_RANK['10m']).all()
        assert (east_cols == -1).all()
        assert len(stats) == 1  # the failed tile contributes no stat entry
    print('  test_build_resolution_grid_marks_failed_tile_as_no_data OK')


# ── end-to-end report ─────────────────────────────────────────────────

def test_build_report_runs_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        raw_dir = output_dir / 'raw'
        raw_dir.mkdir()
        manifest = _write_manifest(output_dir, (45, -111, 46, -110),
                                    ['raw/N45W111_10m.tif'])
        _write_raster(output_dir / 'test_area_mosaic.tif', -111, 45, -110, 46, 30, 1500.0)

        output = output_dir / 'report.png'
        vhv.build_report(output_dir, manifest, output, max_dim=20)
        assert output.exists() and output.stat().st_size > 0
    print('  test_build_report_runs_end_to_end OK')


def test_build_report_raises_without_mosaic_or_vrt():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        manifest = _write_manifest(output_dir, (45, -111, 46, -110),
                                    ['raw/N45W111_10m.tif'])
        try:
            vhv.build_report(output_dir, manifest, output_dir / 'report.png', max_dim=20)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass
    print('  test_build_report_raises_without_mosaic_or_vrt OK')


if __name__ == '__main__':
    test_parse_tile_filename_standard()
    test_parse_tile_filename_ocean_exception()
    test_parse_tile_filename_unrecognized_returns_none()
    test_tile_resolution_map_drops_unrecognized_entries()
    test_all_tile_ids_in_bounds_single_tile()
    test_all_tile_ids_in_bounds_2x2_grid()
    test_native_tile_path_prefers_1m_over_3m()
    test_native_tile_path_none_when_missing()
    test_coverage_fraction_half_covered()
    test_build_resolution_grid_single_tile_no_native()
    test_build_resolution_grid_reports_partial_native_coverage_as_a_stat()
    test_build_resolution_grid_marks_failed_tile_as_no_data()
    test_build_report_runs_end_to_end()
    test_build_report_raises_without_mosaic_or_vrt()
    print('\nAll visualize_hires_vrt tests PASS')
