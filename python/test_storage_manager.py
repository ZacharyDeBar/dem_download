"""Unit tests for storage_manager.py."""

import shutil
import tempfile
import time
from pathlib import Path

from tile_registry import (
    load_registry, save_registry, STATUS_READY, STATUS_FAILED,
)
from storage_manager import (
    StorageManager, CapacityCheck, estimate_tile_size_mb,
    POLICY_HARD, POLICY_LRU, POLICY_WARN,
    DEFAULT_ESTIMATE_3DEP_MB, DEFAULT_ESTIMATE_GLO30_MB,
)


def _tmp_root():
    return Path(tempfile.mkdtemp(prefix='dem_storage_test_'))


def _seed_tile_on_disk(root: Path, tile_id: str, size_mb: float):
    """
    Create a tile dir with a fake file of the requested size so
    refresh_size_from_disk has something to measure.
    """
    tile_dir = root / 'tiles' / tile_id
    tile_dir.mkdir(parents=True, exist_ok=True)
    fake_file = tile_dir / f'{tile_id}_dem.tif'
    with open(fake_file, 'wb') as f:
        f.write(b'\0' * int(size_mb * 1024 * 1024))


def test_size_estimator():
    assert estimate_tile_size_mb('3dep_10m') == DEFAULT_ESTIMATE_3DEP_MB
    assert estimate_tile_size_mb('GLO30')    == DEFAULT_ESTIMATE_GLO30_MB
    assert estimate_tile_size_mb('auto') > 0
    assert estimate_tile_size_mb(None)  > 0
    print('  test_size_estimator OK')


def test_hard_policy_allows_when_room():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        # No tiles yet
        sm = StorageManager(reg, cap_gb=1.0, policy=POLICY_HARD)
        check = sm.check_before_build('N45W110', estimate_mb=100)
        assert check.allowed is True
        assert check.refusal_reason is None
        assert check.headroom_mb == 1024.0
        print('  test_hard_policy_allows_when_room OK')
    finally:
        shutil.rmtree(root)


def test_hard_policy_refuses_when_full():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        reg.mark_ready('N45W109', 'a', 'b', 'c', 'd', size_mb=950)
        sm = StorageManager(reg, cap_gb=1.0, policy=POLICY_HARD)
        check = sm.check_before_build('N45W110', estimate_mb=200)
        assert check.allowed is False
        assert check.refusal_reason is not None
        assert 'HARD' in check.refusal_reason
        assert 'shortfall' in check.refusal_reason
        assert check.evicted == []
        print('  test_hard_policy_refuses_when_full OK')
    finally:
        shutil.rmtree(root)


def test_warn_policy_proceeds_with_warning():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        reg.mark_ready('N45W109', 'a', 'b', 'c', 'd', size_mb=950)
        sm = StorageManager(reg, cap_gb=1.0, policy=POLICY_WARN)
        check = sm.check_before_build('N45W110', estimate_mb=200)
        assert check.allowed is True
        assert check.warning is not None
        assert 'warn' in check.warning.lower()
        assert check.evicted == []
        print('  test_warn_policy_proceeds_with_warning OK')
    finally:
        shutil.rmtree(root)


def test_lru_policy_evicts_oldest():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        # Three ready tiles totaling 900MB
        # Different access times to set up LRU ordering
        # Use _seed so refresh_size_from_disk works if needed
        reg.mark_ready('N44W110', 'tiles/N44W110/dem.tif', 'b', 'c', 'd', size_mb=300)
        reg.get('N44W110').last_accessed_utc = '2026-06-01T00:00:00Z'  # oldest
        _seed_tile_on_disk(root, 'N44W110', 0.01)

        reg.mark_ready('N45W110', 'tiles/N45W110/dem.tif', 'b', 'c', 'd', size_mb=300)
        reg.get('N45W110').last_accessed_utc = '2026-06-02T00:00:00Z'
        _seed_tile_on_disk(root, 'N45W110', 0.01)

        reg.mark_ready('N46W110', 'tiles/N46W110/dem.tif', 'b', 'c', 'd', size_mb=300)
        reg.get('N46W110').last_accessed_utc = '2026-06-03T00:00:00Z'  # newest
        _seed_tile_on_disk(root, 'N46W110', 0.01)

        sm = StorageManager(reg, cap_gb=1.0, policy=POLICY_LRU)
        # Need 250MB but only 124MB free → must evict
        check = sm.check_before_build('N45W109', estimate_mb=250)

        assert check.allowed is True
        # Should have evicted oldest first
        assert 'N44W110' in check.evicted, \
            f"Should evict oldest (N44W110), got {check.evicted}"
        # And only enough to make room — evicting one 300MB tile gives
        # 424MB headroom, enough for 250MB
        assert len(check.evicted) == 1, \
            f"Should have evicted only 1, got {check.evicted}"
        # N44W110's tile dir should be removed
        assert not (root / 'tiles' / 'N44W110').exists()
        # Other tiles still present
        assert (root / 'tiles' / 'N45W110').exists()
        assert (root / 'tiles' / 'N46W110').exists()
        # Registry no longer has N44W110
        assert reg.get('N44W110') is None
        # Warning explains the eviction
        assert check.warning is not None
        assert 'LRU' in check.warning
        print('  test_lru_policy_evicts_oldest OK')
    finally:
        shutil.rmtree(root)


def test_lru_respects_protected():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        reg.mark_ready('N44W110', 'tiles/N44W110/dem.tif', 'b', 'c', 'd', size_mb=300)
        reg.get('N44W110').last_accessed_utc = '2026-06-01T00:00:00Z'
        _seed_tile_on_disk(root, 'N44W110', 0.01)
        reg.mark_ready('N45W110', 'tiles/N45W110/dem.tif', 'b', 'c', 'd', size_mb=300)
        reg.get('N45W110').last_accessed_utc = '2026-06-02T00:00:00Z'
        _seed_tile_on_disk(root, 'N45W110', 0.01)

        # Protect the oldest tile
        sm = StorageManager(reg, cap_gb=0.7, policy=POLICY_LRU,
                            protected_tiles={'N44W110'})
        # Want 300MB but only ~117MB free → must evict, but oldest is protected
        check = sm.check_before_build('N45W109', estimate_mb=300)

        # Should evict N45W110 (next-oldest, not protected) instead
        assert 'N44W110' not in check.evicted
        assert 'N45W110' in check.evicted
        # Now we have ~417MB free, enough for 300MB
        assert check.allowed is True
        print('  test_lru_respects_protected OK')
    finally:
        shutil.rmtree(root)


def test_lru_refuses_when_no_evictable():
    """All remaining tiles protected; can't free enough."""
    root = _tmp_root()
    try:
        reg = load_registry(root)
        reg.mark_ready('N44W110', 'tiles/N44W110/dem.tif', 'b', 'c', 'd', size_mb=950)
        reg.get('N44W110').last_accessed_utc = '2026-06-01T00:00:00Z'

        # Protect the only tile
        sm = StorageManager(reg, cap_gb=1.0, policy=POLICY_LRU,
                            protected_tiles={'N44W110'})
        check = sm.check_before_build('N45W109', estimate_mb=200)
        assert check.allowed is False
        assert check.evicted == []
        assert check.refusal_reason is not None
        assert 'protected' in check.refusal_reason.lower() or \
               'insufficient' in check.refusal_reason.lower()
        print('  test_lru_refuses_when_no_evictable OK')
    finally:
        shutil.rmtree(root)


def test_refresh_size_from_disk():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        reg.mark_ready('N45W110', 'tiles/N45W110/dem.tif', 'b', 'c', 'd', size_mb=999.0)  # wrong
        _seed_tile_on_disk(root, 'N45W110', 5.0)  # actual: 5MB

        sm = StorageManager(reg, cap_gb=1.0)
        new_size = sm.refresh_size_from_disk('N45W110')
        assert new_size is not None
        # Allow some slack for filesystem overhead
        assert 4.5 <= new_size <= 5.5, f"Expected ~5MB, got {new_size}"
        assert reg.get('N45W110').size_mb == new_size

        # Non-existent tile returns None
        assert sm.refresh_size_from_disk('N99W999') is None
        print('  test_refresh_size_from_disk OK')
    finally:
        shutil.rmtree(root)


def test_prune_failed():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        reg.mark_failed('N44W110', 'bad URL')
        reg.mark_failed('N45W110', 'timeout')
        reg.mark_ready('N46W110', 'a', 'b', 'c', 'd', size_mb=100)

        sm = StorageManager(reg, cap_gb=1.0)
        pruned = sm.prune_failed()
        assert sorted(pruned) == ['N44W110', 'N45W110']
        assert reg.get('N44W110') is None
        assert reg.get('N45W110') is None
        assert reg.get('N46W110') is not None  # ready ones survive
        print('  test_prune_failed OK')
    finally:
        shutil.rmtree(root)


def test_invalid_policy():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        try:
            StorageManager(reg, cap_gb=1.0, policy='lazy')
            assert False
        except ValueError:
            pass
        print('  test_invalid_policy OK')
    finally:
        shutil.rmtree(root)


def test_invalid_cap():
    root = _tmp_root()
    try:
        reg = load_registry(root)
        try:
            StorageManager(reg, cap_gb=0)
            assert False
        except ValueError:
            pass
        try:
            StorageManager(reg, cap_gb=-1)
            assert False
        except ValueError:
            pass
        print('  test_invalid_cap OK')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    test_size_estimator()
    test_hard_policy_allows_when_room()
    test_hard_policy_refuses_when_full()
    test_warn_policy_proceeds_with_warning()
    test_lru_policy_evicts_oldest()
    test_lru_respects_protected()
    test_lru_refuses_when_no_evictable()
    test_refresh_size_from_disk()
    test_prune_failed()
    test_invalid_policy()
    test_invalid_cap()
    print('\nAll storage_manager tests PASS')