"""Unit tests for tile_registry.py."""

import json
import shutil
import tempfile
from pathlib import Path

from tile_registry import (
    TileRegistry, TileEntry, load_registry, save_registry,
    tile_dir_rel, tile_file_rel, absolute_path,
    STATUS_READY, STATUS_PENDING, STATUS_FAILED, STATUS_ABSENT,
    REGISTRY_FILENAME, REGISTRY_VERSION,
)


def _tmp_root():
    return Path(tempfile.mkdtemp(prefix='dem_registry_test_'))


def test_path_helpers():
    assert tile_dir_rel('N45W110') == 'tiles/N45W110'
    assert tile_file_rel('N45W110', 'dem.tif') == 'tiles/N45W110/N45W110_dem.tif'
    assert tile_file_rel('S05E172', 'meta.json') == 'tiles/S05E172/S05E172_meta.json'
    root = Path('/some/root')
    assert absolute_path(root, 'tiles/N45W110/N45W110_dem.tif') == \
        Path('/some/root/tiles/N45W110/N45W110_dem.tif')
    print('  test_path_helpers OK')


def test_empty_registry():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        assert reg.tile_count == 0
        assert reg.ready_count == 0
        assert reg.list_ready() == []
        assert reg.status_of('N45W110') == STATUS_ABSENT
        assert reg.total_size_mb() == 0.0
        print('  test_empty_registry OK')
    finally:
        shutil.rmtree(root)


def test_upsert_and_save_load():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        # Add a pending entry
        reg.mark_pending('N45W110')
        assert reg.status_of('N45W110') == STATUS_PENDING
        # Mark it ready
        reg.mark_ready(
            tile_id='N45W110',
            dem_file='tiles/N45W110/N45W110_dem.tif',
            flat_file='tiles/N45W110/N45W110_flat.tif',
            boundary_file='tiles/N45W110/N45W110_boundary.tif',
            meta_file='tiles/N45W110/N45W110_meta.json',
            size_mb=287.4,
        )
        assert reg.status_of('N45W110') == STATUS_READY
        assert reg.ready_count == 1
        save_registry(reg)
        assert (root / REGISTRY_FILENAME).exists()

        # Round-trip load
        reg2 = load_registry(root)
        assert reg2.tile_count == 1
        e = reg2.get('N45W110')
        assert e is not None
        assert e.status == STATUS_READY
        assert e.dem_file == 'tiles/N45W110/N45W110_dem.tif'
        assert e.size_mb == 287.4
        assert e.built_utc is not None
        assert e.last_accessed_utc is not None
        print('  test_upsert_and_save_load OK')
    finally:
        shutil.rmtree(root)


def test_status_transitions():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        # pending → failed
        reg.mark_pending('N45W110')
        reg.mark_failed('N45W110', '404 from 3DEP and GLO30')
        assert reg.status_of('N45W110') == STATUS_FAILED
        assert reg.get('N45W110').failed_reason.startswith('404')

        # failed → pending (retry) → ready
        reg.mark_pending('N45W110')
        assert reg.status_of('N45W110') == STATUS_PENDING
        assert reg.get('N45W110').failed_reason is None  # cleared on retry
        reg.mark_ready(
            tile_id='N45W110',
            dem_file='tiles/N45W110/N45W110_dem.tif',
            flat_file='tiles/N45W110/N45W110_flat.tif',
            boundary_file='tiles/N45W110/N45W110_boundary.tif',
            meta_file='tiles/N45W110/N45W110_meta.json',
            size_mb=200.0,
        )
        assert reg.status_of('N45W110') == STATUS_READY
        print('  test_status_transitions OK')
    finally:
        shutil.rmtree(root)


def test_touch():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        # touch on absent returns False
        assert reg.touch('N99W999') is False
        # add and touch
        reg.mark_pending('N45W110')
        ts_before = reg.get('N45W110').last_accessed_utc
        # touch updates timestamp
        import time
        time.sleep(1.01)   # ensure measurable difference
        assert reg.touch('N45W110') is True
        ts_after = reg.get('N45W110').last_accessed_utc
        assert ts_after is not None
        # ts_before may be None for entries that were never built
        if ts_before is not None:
            assert ts_after > ts_before
        print('  test_touch OK')
    finally:
        shutil.rmtree(root)


def test_atomic_save_doesnt_corrupt():
    """If we save twice, the second save replaces the first cleanly."""
    root = _tmp_root()
    try:
        reg = load_registry(root)
        reg.mark_pending('N45W110')
        save_registry(reg)
        size1 = (root / REGISTRY_FILENAME).stat().st_size

        # Add another tile and save again
        reg.mark_pending('N45W109')
        save_registry(reg)
        size2 = (root / REGISTRY_FILENAME).stat().st_size

        # The file should be larger now (more tiles)
        assert size2 > size1

        # And re-loading should give us both tiles
        reg3 = load_registry(root)
        assert reg3.tile_count == 2
        assert reg3.status_of('N45W110') == STATUS_PENDING
        assert reg3.status_of('N45W109') == STATUS_PENDING

        # No leftover temp files
        tmp_files = list(root.glob('.tile_registry.*.tmp'))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"
        print('  test_atomic_save_doesnt_corrupt OK')
    finally:
        shutil.rmtree(root)


def test_total_size_mb():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        reg.mark_ready('N45W110', 'a', 'b', 'c', 'd', size_mb=100.0)
        reg.mark_ready('N45W109', 'a', 'b', 'c', 'd', size_mb=200.5)
        # pending tile with no size
        reg.mark_pending('N44W110')
        assert reg.total_size_mb() == 300.5
        print('  test_total_size_mb OK')
    finally:
        shutil.rmtree(root)


def test_remove():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        reg.mark_pending('N45W110')
        assert reg.remove('N45W110') is True
        assert reg.remove('N45W110') is False  # already gone
        assert reg.tile_count == 0
        print('  test_remove OK')
    finally:
        shutil.rmtree(root)


def test_load_invalid_json():
    root = _tmp_root()
    try:
        # Write garbage to registry file
        with open(root / REGISTRY_FILENAME, 'w') as f:
            f.write('not json at all {{{{')
        try:
            load_registry(root)
            assert False, "Should have raised"
        except ValueError as e:
            assert 'invalid JSON' in str(e)
        print('  test_load_invalid_json OK')
    finally:
        shutil.rmtree(root)


def test_load_unsupported_version():
    root = _tmp_root()
    try:
        with open(root / REGISTRY_FILENAME, 'w') as f:
            json.dump({'version': 999, 'tiles': {}}, f)
        try:
            load_registry(root)
            assert False
        except ValueError as e:
            assert 'version' in str(e)
        print('  test_load_unsupported_version OK')
    finally:
        shutil.rmtree(root)


def test_invalid_status_rejected():
    """Creating a TileEntry with bogus status should raise."""
    try:
        TileEntry(tile_id='N45W110', status='nonsense')
        assert False
    except ValueError:
        pass
    print('  test_invalid_status_rejected OK')


def test_list_ready_ordered():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        # Add in random order
        reg.mark_ready('N45W110', 'a', 'b', 'c', 'd', size_mb=1)
        reg.mark_pending('N44W110')
        reg.mark_ready('N44W109', 'a', 'b', 'c', 'd', size_mb=1)
        reg.mark_ready('N45W109', 'a', 'b', 'c', 'd', size_mb=1)
        ready = reg.list_ready()
        # Sorted alphabetically
        assert ready == ['N44W109', 'N45W109', 'N45W110']
        # And pending list
        pending = reg.list_with_status(STATUS_PENDING)
        assert pending == ['N44W110']
        print('  test_list_ready_ordered OK')
    finally:
        shutil.rmtree(root)


def test_forward_compat_extra_keys_ignored():
    """If a future version adds keys, current version should drop them
    on load rather than crash. (We're version-locked, so this is moot
    unless we relax that; the test documents the current behavior.)"""
    root = _tmp_root()
    try:
        with open(root / REGISTRY_FILENAME, 'w') as f:
            json.dump({
                'version': REGISTRY_VERSION,
                'tiles': {
                    'N45W110': {
                        'status': 'ready',
                        'tile_dir': 'tiles/N45W110',
                        'dem_file': 'tiles/N45W110/dem.tif',
                        'size_mb': 100,
                        'future_field_we_dont_know_about': 'foo',
                    }
                }
            }, f)
        reg = load_registry(root)
        assert reg.get('N45W110').status == 'ready'
        print('  test_forward_compat_extra_keys_ignored OK')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    test_path_helpers()
    test_empty_registry()
    test_upsert_and_save_load()
    test_status_transitions()
    test_touch()
    test_atomic_save_doesnt_corrupt()
    test_total_size_mb()
    test_remove()
    test_load_invalid_json()
    test_load_unsupported_version()
    test_invalid_status_rejected()
    test_list_ready_ordered()
    test_forward_compat_extra_keys_ignored()
    print('\nAll tile_registry tests PASS')