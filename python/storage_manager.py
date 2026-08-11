"""
storage_manager.py
━━━━━━━━━━━━━━━━━━
Storage capacity enforcement for the DEM tile cache.

Provides a check-before-build interface: when the tile builder is
about to download/process a tile, it asks the storage manager whether
the new tile (estimated size) would fit under the cap. Three policies:

  - 'hard'  : refuse to build if cap would be exceeded
  - 'lru'   : evict oldest tiles (by last_accessed_utc) until enough
              room exists, then proceed
  - 'warn'  : log a warning and proceed regardless

Default policy is 'hard' per design Q4.

Eviction (LRU) removes a tile's on-disk artifacts and its registry
entry. It does NOT remove tiles currently in the protected_tiles set —
those are explicitly off-limits (e.g. tiles needed for an in-flight
compute).

Tile size estimates (used before download, when actual size is unknown):
  - 3DEP 10m: ~250MB per 1° tile (DEM + flat + boundary, compressed)
  - GLO-30:   ~80MB per 1° tile
  - Default estimate: 250MB (conservative — pessimistic)

These estimates are calibrated to be a bit high so we don't accidentally
overshoot the cap. After a tile is built, actual size is recorded in
the registry and used for total-used calculations.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from tile_registry import (
    TileRegistry, save_registry, STATUS_READY,
)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

POLICY_HARD = 'hard'
POLICY_LRU  = 'lru'
POLICY_WARN = 'warn'
_VALID_POLICIES = {POLICY_HARD, POLICY_LRU, POLICY_WARN}

DEFAULT_CAP_GB             = 500.0
DEFAULT_ESTIMATE_3DEP_MB   = 250.0
DEFAULT_ESTIMATE_GLO30_MB  = 80.0
DEFAULT_ESTIMATE_UNKNOWN_MB = 250.0


# ─────────────────────────────────────────────
# Outcomes
# ─────────────────────────────────────────────

@dataclass
class CapacityCheck:
    """Result of asking 'can we add a tile of this size?'"""
    allowed:       bool          # may we proceed with the build?
    used_mb:       float         # current used MB (before this build)
    estimate_mb:   float         # estimated MB the new tile would add
    cap_mb:        float         # the cap value (informational)
    headroom_mb:   float         # cap_mb - used_mb (before this build)
    policy:        str
    evicted:       List[str]     # tile IDs evicted to make room (LRU)
    refusal_reason: Optional[str] = None
    warning:       Optional[str] = None


# ─────────────────────────────────────────────
# Storage manager
# ─────────────────────────────────────────────

class StorageManager:
    """
    Owner of the storage cap policy. Uses the registry as its source
    of truth for what's on disk.

    Note: this trusts the registry's size_mb values. If the registry
    falls out of sync with disk reality, callers can rebuild via
    refresh_size_from_disk(). For Phase 1 we keep it simple — registry
    size_mb is authoritative.
    """

    def __init__(
        self,
        registry:           TileRegistry,
        cap_gb:             float = DEFAULT_CAP_GB,
        policy:             str   = POLICY_HARD,
        protected_tiles:    Optional[Set[str]] = None,
    ):
        if policy not in _VALID_POLICIES:
            raise ValueError(
                f"Invalid cap policy {policy!r}, "
                f"must be one of {sorted(_VALID_POLICIES)}")
        if cap_gb <= 0:
            raise ValueError(f"cap_gb must be positive, got {cap_gb}")
        self.registry        = registry
        self.cap_mb          = cap_gb * 1024.0
        self.policy          = policy
        self.protected_tiles = set(protected_tiles or [])

    # ── Inspection ───────────────────────────────────────────────
    def total_used_mb(self) -> float:
        """MB used across all tiles in the registry (any status)."""
        return self.registry.total_size_mb()

    def headroom_mb(self) -> float:
        """MB remaining under the cap. Can be negative if over-committed."""
        return self.cap_mb - self.total_used_mb()

    # ── The main entry point ─────────────────────────────────────
    def check_before_build(
        self,
        tile_id:     str,
        estimate_mb: float = DEFAULT_ESTIMATE_UNKNOWN_MB,
    ) -> CapacityCheck:
        """
        Determine whether we may proceed with building this tile.

        Side effect: if policy is 'lru' and eviction is needed, this
        method evicts tiles BEFORE returning, deleting their on-disk
        files and removing their registry entries. The caller is
        expected to save_registry() after a successful build.

        Returns a CapacityCheck describing what was decided and why.
        """
        used = self.total_used_mb()
        headroom = self.cap_mb - used
        result = CapacityCheck(
            allowed     = True,
            used_mb     = used,
            estimate_mb = estimate_mb,
            cap_mb      = self.cap_mb,
            headroom_mb = headroom,
            policy      = self.policy,
            evicted     = [],
        )

        if estimate_mb <= headroom:
            return result  # plenty of room, proceed

        # We would exceed the cap.
        shortfall = estimate_mb - headroom
        if self.policy == POLICY_HARD:
            result.allowed = False
            result.refusal_reason = (
                f"Build refused under HARD policy. "
                f"Used {used:.0f}MB of {self.cap_mb:.0f}MB cap "
                f"(headroom {headroom:.0f}MB), but tile {tile_id!r} "
                f"is estimated at {estimate_mb:.0f}MB "
                f"(shortfall {shortfall:.0f}MB). "
                f"Options: raise --max-storage-gb, prune old tiles, "
                f"or switch to --cap-policy=lru or warn."
            )
            return result

        if self.policy == POLICY_WARN:
            result.warning = (
                f"Storage cap would be exceeded by {shortfall:.0f}MB "
                f"(used {used:.0f} + new {estimate_mb:.0f} > "
                f"cap {self.cap_mb:.0f}MB) but --cap-policy=warn — proceeding."
            )
            return result

        # POLICY_LRU — try to evict oldest tiles to make room
        # Candidates: ready tiles, not in protected_tiles, sorted by
        # last_accessed_utc ascending (oldest first). Empty timestamps
        # are treated as oldest.
        candidates = [
            (entry.last_accessed_utc or '', entry.tile_id, entry.size_mb)
            for entry in self.registry.tiles.values()
            if entry.status == STATUS_READY
            and entry.tile_id not in self.protected_tiles
            and entry.tile_id != tile_id      # don't evict self
        ]
        candidates.sort()  # oldest first by timestamp string

        freed = 0.0
        evicted = []
        for _ts, tid, sz in candidates:
            if freed >= shortfall:
                break
            self._delete_tile_files(tid)
            self.registry.remove(tid)
            freed += sz
            evicted.append(tid)

        result.evicted = evicted

        # Recompute headroom now that we've freed space
        new_used = self.total_used_mb()
        new_headroom = self.cap_mb - new_used
        result.used_mb = new_used
        result.headroom_mb = new_headroom

        if estimate_mb <= new_headroom:
            result.warning = (
                f"LRU policy evicted {len(evicted)} tile(s) "
                f"({freed:.0f}MB freed) to make room for {tile_id!r}: "
                f"{evicted}"
            )
            return result

        # Even after evicting everything we could, still not enough room.
        # Treat as refusal — the build is too big for the cap, OR all
        # remaining tiles are protected.
        result.allowed = False
        result.refusal_reason = (
            f"Build refused after LRU eviction. "
            f"Evicted {len(evicted)} tile(s) freeing {freed:.0f}MB, "
            f"but still short {shortfall - freed:.0f}MB. "
            f"Remaining tiles are protected or insufficient. "
            f"Tile {tile_id!r} cannot fit under cap {self.cap_mb:.0f}MB."
        )
        return result

    # ── Eviction helper ──────────────────────────────────────────
    def _delete_tile_files(self, tile_id: str) -> None:
        """
        Remove a tile's on-disk artifacts (the tile directory).
        Idempotent — if files are already gone, that's fine.
        """
        entry = self.registry.get(tile_id)
        if entry is None:
            return
        # Delete the tile directory rather than individual files —
        # captures any companions (water.geojson, etc.) we may not
        # have explicitly listed.
        tile_dir = self.registry.storage_root / entry.tile_dir
        if tile_dir.exists():
            try:
                shutil.rmtree(tile_dir)
                logger.info(f"Removed tile directory {tile_dir}")
            except OSError as e:
                logger.warning(
                    f"Could not remove tile dir {tile_dir}: {e}")

    # ── Maintenance ──────────────────────────────────────────────
    def refresh_size_from_disk(self, tile_id: str) -> Optional[float]:
        """
        Re-measure a tile's size by walking its on-disk directory and
        update the registry entry. Returns the new size in MB, or None
        if the tile is absent.
        """
        entry = self.registry.get(tile_id)
        if entry is None:
            return None
        tile_dir = self.registry.storage_root / entry.tile_dir
        if not tile_dir.exists():
            entry.size_mb = 0.0
            return 0.0
        total_bytes = 0
        for p in tile_dir.rglob('*'):
            if p.is_file():
                try:
                    total_bytes += p.stat().st_size
                except OSError:
                    pass
        new_size_mb = total_bytes / (1024 * 1024)
        entry.size_mb = new_size_mb
        return new_size_mb

    def prune_failed(self) -> List[str]:
        """
        Remove all entries with status == 'failed'. Returns the list
        of pruned tile IDs. Used during maintenance to clear out
        retry markers.
        """
        from tile_registry import STATUS_FAILED
        failed_ids = self.registry.list_with_status(STATUS_FAILED)
        for tid in failed_ids:
            self._delete_tile_files(tid)
            self.registry.remove(tid)
        return failed_ids


# ─────────────────────────────────────────────
# Size estimator
# ─────────────────────────────────────────────

def estimate_tile_size_mb(source: str = 'auto') -> float:
    """
    Return a conservative size estimate for a tile from a given source.
    These are pre-build estimates used by the cap check. Actual sizes
    after build are recorded separately in the registry.
    """
    s = (source or '').lower()
    if s == '3dep_10m' or s == '3dep_10':
        return DEFAULT_ESTIMATE_3DEP_MB
    if s == 'glo30':
        return DEFAULT_ESTIMATE_GLO30_MB
    # 'auto' or anything else → conservative default
    return DEFAULT_ESTIMATE_UNKNOWN_MB