"""
tile_registry.py
━━━━━━━━━━━━━━━━
Tile registry: a JSON manifest tracking all known tiles and their state.

The registry is the source of truth for "what tiles exist, what state
they're in, where to find them". It lives at:
  {storage_root}/tile_registry.json

Schema (top-level):
  {
    "version":       1,
    "updated_utc":   "2026-06-11T10:00:00Z",
    "tile_count":    24,
    "ready_count":   22,
    "tiles": {
      "N45W110": {
        "status":           "ready" | "pending" | "failed" | "absent",
        "tile_dir":         "tiles/N45W110",
        "dem_file":         "tiles/N45W110/N45W110_dem.tif",
        "flat_file":        "tiles/N45W110/N45W110_flat.tif",
        "boundary_file":    "tiles/N45W110/N45W110_boundary.tif",
        "meta_file":        "tiles/N45W110/N45W110_meta.json",
        "size_mb":          287.4,
        "built_utc":        "2026-06-11T09:34:21Z",
        "last_accessed_utc": "2026-06-11T10:22:08Z",
        "failed_reason":    null
      },
      ...
    }
  }

Paths are stored as POSIX-style strings relative to the storage root,
so the registry is portable across machines and cloud paths.

Concurrency: atomic write via temp file + rename prevents a torn/partial
file, but does NOT prevent lost updates when two processes each load a
snapshot, mutate different tiles, and save — the second save silently
discards the first process's changes (last-writer-wins on the whole
file). This matters now that tile provisioning runs multiple tiles
concurrently (2026-07-11: parallel tile downloads/builds within one
route). Use `update_registry()` for any mutation that might race with
another process on the same storage_root — it wraps the whole
load→mutate→save cycle in a cross-process advisory file lock (POSIX
`fcntl.flock` / Windows `msvcrt.locking`, so it works the same on the
Linux and Windows compute machines) and always reloads fresh under the
lock, so concurrent writers to *different* tiles never clobber each
other. A plain `load_registry()` for a read-only check (e.g.
skip-if-ready) doesn't need the lock.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional


REGISTRY_VERSION = 1
REGISTRY_FILENAME = 'tile_registry.json'

# Valid status values
STATUS_READY    = 'ready'
STATUS_PENDING  = 'pending'
STATUS_FAILED   = 'failed'
STATUS_ABSENT   = 'absent'   # not tracked but legal as a query result
_VALID_STATUSES = {STATUS_READY, STATUS_PENDING, STATUS_FAILED, STATUS_ABSENT}


def _utcnow_iso() -> str:
    """Current UTC time in ISO 8601 with 'Z' suffix."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ─────────────────────────────────────────────
# Per-tile entry
# ─────────────────────────────────────────────

@dataclass
class TileEntry:
    """
    Registry record for a single tile. All paths are POSIX strings
    relative to the storage root.
    """
    tile_id:           str
    status:            str = STATUS_PENDING
    tile_dir:          str = ''
    dem_file:          Optional[str] = None
    flat_file:         Optional[str] = None
    boundary_file:     Optional[str] = None
    meta_file:         Optional[str] = None
    size_mb:           float = 0.0
    built_utc:         Optional[str] = None
    last_accessed_utc: Optional[str] = None
    failed_reason:     Optional[str] = None

    def __post_init__(self):
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status {self.status!r}, "
                             f"must be one of {sorted(_VALID_STATUSES)}")
        if not self.tile_dir:
            self.tile_dir = f"tiles/{self.tile_id}"


# ─────────────────────────────────────────────
# Registry container
# ─────────────────────────────────────────────

@dataclass
class TileRegistry:
    """
    In-memory representation of the registry. Load/save with the
    module-level functions; mutate via the methods on this class.

    Backing file path is stored on the instance so save() doesn't
    need to be told where to write.
    """
    storage_root:  Path                          # absolute path
    tiles:         Dict[str, TileEntry] = field(default_factory=dict)
    version:       int = REGISTRY_VERSION
    updated_utc:   str = ''

    @property
    def registry_path(self) -> Path:
        return self.storage_root / REGISTRY_FILENAME

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def ready_count(self) -> int:
        return sum(1 for e in self.tiles.values()
                   if e.status == STATUS_READY)

    # ── Queries ──────────────────────────────────────────────────
    def get(self, tile_id: str) -> Optional[TileEntry]:
        """Return the entry for tile_id, or None if absent."""
        return self.tiles.get(tile_id)

    def status_of(self, tile_id: str) -> str:
        """Return the status of a tile, or STATUS_ABSENT if not tracked."""
        e = self.tiles.get(tile_id)
        return e.status if e else STATUS_ABSENT

    def list_ready(self) -> List[str]:
        return sorted(tid for tid, e in self.tiles.items()
                      if e.status == STATUS_READY)

    def list_with_status(self, status: str) -> List[str]:
        return sorted(tid for tid, e in self.tiles.items()
                      if e.status == status)

    def total_size_mb(self) -> float:
        """Sum of size_mb across all entries (any status)."""
        return sum(e.size_mb for e in self.tiles.values())

    # ── Mutations ────────────────────────────────────────────────
    def upsert(self, entry: TileEntry) -> None:
        """Add or replace a tile entry."""
        self.tiles[entry.tile_id] = entry

    def remove(self, tile_id: str) -> bool:
        """Drop a tile from the registry. Returns True if removed."""
        if tile_id in self.tiles:
            del self.tiles[tile_id]
            return True
        return False

    def touch(self, tile_id: str) -> bool:
        """
        Update last_accessed_utc on a tile. Used for LRU tracking.
        Returns True if the tile exists and was touched, False if absent.

        Note: this does NOT save() automatically — caller must save.
        """
        e = self.tiles.get(tile_id)
        if e is None:
            return False
        e.last_accessed_utc = _utcnow_iso()
        return True

    def mark_pending(self, tile_id: str) -> None:
        """Mark a tile as build-in-progress, creating entry if needed."""
        e = self.tiles.get(tile_id)
        if e is None:
            e = TileEntry(tile_id=tile_id, status=STATUS_PENDING)
            self.tiles[tile_id] = e
        else:
            e.status = STATUS_PENDING
            e.failed_reason = None

    def mark_ready(
        self,
        tile_id:    str,
        dem_file:   str,
        flat_file:  str,
        boundary_file: str,
        meta_file:  str,
        size_mb:    float,
    ) -> None:
        """Mark a tile as successfully built. Fills file paths and size."""
        e = self.tiles.get(tile_id)
        if e is None:
            e = TileEntry(tile_id=tile_id)
            self.tiles[tile_id] = e
        e.status            = STATUS_READY
        e.dem_file          = dem_file
        e.flat_file         = flat_file
        e.boundary_file     = boundary_file
        e.meta_file         = meta_file
        e.size_mb           = size_mb
        e.built_utc         = _utcnow_iso()
        e.last_accessed_utc = e.built_utc
        e.failed_reason     = None

    def mark_failed(self, tile_id: str, reason: str) -> None:
        """Record a build failure with a reason string."""
        e = self.tiles.get(tile_id)
        if e is None:
            e = TileEntry(tile_id=tile_id)
            self.tiles[tile_id] = e
        e.status        = STATUS_FAILED
        e.failed_reason = reason


# ─────────────────────────────────────────────
# Load / save (with atomic write)
# ─────────────────────────────────────────────

def load_registry(storage_root: Path) -> TileRegistry:
    """
    Load the registry from storage_root. Returns a fresh empty
    registry if the file doesn't exist.

    Will raise ValueError if the file exists but contains invalid
    or unsupported content. This is intentional — corrupt registries
    should not silently become empty.
    """
    storage_root = Path(storage_root)
    registry_path = storage_root / REGISTRY_FILENAME

    if not registry_path.exists():
        return TileRegistry(storage_root=storage_root)

    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Registry at {registry_path} contains invalid JSON: {e}. "
            f"Inspect manually before proceeding; do not delete.")

    version = data.get('version', 0)
    if version != REGISTRY_VERSION:
        raise ValueError(
            f"Registry at {registry_path} is version {version}, "
            f"expected {REGISTRY_VERSION}. Migration required.")

    reg = TileRegistry(
        storage_root=storage_root,
        version=version,
        updated_utc=data.get('updated_utc', ''),
    )
    for tid, tdata in data.get('tiles', {}).items():
        # Drop extra keys defensively (forward-compat)
        valid_keys = set(TileEntry.__dataclass_fields__.keys())
        tdata_filtered = {k: v for k, v in tdata.items() if k in valid_keys}
        tdata_filtered['tile_id'] = tid   # in case stored data missed it
        try:
            entry = TileEntry(**tdata_filtered)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Registry tile {tid!r} has invalid record: {e}")
        reg.tiles[tid] = entry

    return reg


def save_registry(registry: TileRegistry) -> None:
    """
    Save the registry atomically: write to a temp file in the same
    directory, then rename over the target. Rename is atomic on
    POSIX and Windows for same-volume operations.

    This guarantees a partial-write crash never leaves the registry
    in a half-written state. Either the new version is fully visible
    or the old version remains.
    """
    registry.updated_utc = _utcnow_iso()
    registry_path = registry.registry_path
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'version':     registry.version,
        'updated_utc': registry.updated_utc,
        'tile_count':  registry.tile_count,
        'ready_count': registry.ready_count,
        'tiles': {
            tid: {k: v for k, v in asdict(entry).items() if k != 'tile_id'}
            for tid, entry in sorted(registry.tiles.items())
        },
    }

    # Write to a temp file in the same directory
    fd, tmp_path = tempfile.mkstemp(
        prefix='.tile_registry.', suffix='.tmp',
        dir=str(registry_path.parent),
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        # Atomic rename. os.replace is atomic on both POSIX and Windows.
        os.replace(tmp_path, registry_path)
    except Exception:
        # On any failure, clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────────
# Cross-process lock (Windows msvcrt.locking / POSIX fcntl.flock)
# ─────────────────────────────────────────────

LOCK_FILENAME = 'tile_registry.json.lock'


def _try_lock(f) -> bool:
    """Try to acquire an exclusive, non-blocking lock on f. True on success."""
    if os.name == 'nt':
        import msvcrt
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    else:
        import fcntl
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def _unlock(f) -> None:
    if os.name == 'nt':
        import msvcrt
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


@contextmanager
def registry_lock(storage_root: Path, timeout: float = 30.0,
                   poll_interval: float = 0.05):
    """
    Cross-process advisory lock guarding the tile registry at
    storage_root. Acquire → do work → release, with a timeout so a
    crashed lock-holder can't wedge every future run forever (the OS
    releases the lock automatically when the holder process exits or
    dies, so a timeout is only needed for an unusually slow — not
    dead — holder).

    Prefer `update_registry()` over using this directly; it's the
    lock + load + mutate + save pattern already wired together.
    """
    storage_root = Path(storage_root)
    lock_path = storage_root / LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    f = open(lock_path, 'a+b')
    try:
        if f.tell() == 0:
            # msvcrt.locking needs at least 1 byte to lock on Windows
            f.write(b'\0')
            f.flush()

        deadline = time.monotonic() + timeout
        acquired = _try_lock(f)
        while not acquired:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Could not acquire tile registry lock at {lock_path} "
                    f"within {timeout}s — another process may be stuck "
                    f"holding it (check for a hung/crashed build).")
            time.sleep(poll_interval)
            acquired = _try_lock(f)
        try:
            yield
        finally:
            _unlock(f)
    finally:
        f.close()


def update_registry(
    storage_root: Path,
    mutate_fn: Callable[[TileRegistry], None],
    timeout: float = 30.0,
) -> TileRegistry:
    """
    Safely read-modify-write the registry: acquire the cross-process
    lock, load a FRESH copy from disk, apply mutate_fn(registry), save,
    release. Use this for any registry write that might run concurrently
    with another process touching the same storage_root (parallel tile
    builds, concurrent routes) — it's immune to the lost-update race a
    bare load_registry()/save_registry() pair has under concurrency.

    mutate_fn receives the registry and should call its mutation
    methods (mark_pending/mark_ready/mark_failed/upsert/remove) — it
    does not need to save.

    Returns the updated registry (already saved).
    """
    storage_root = Path(storage_root)
    with registry_lock(storage_root, timeout=timeout):
        registry = load_registry(storage_root)
        mutate_fn(registry)
        save_registry(registry)
        return registry


# ─────────────────────────────────────────────
# Path helpers (relative-to-storage-root, POSIX form)
# ─────────────────────────────────────────────

def tile_dir_rel(tile_id: str) -> str:
    """Return the relative tile-dir path (POSIX form)."""
    return f"tiles/{tile_id}"


def tile_file_rel(tile_id: str, suffix: str) -> str:
    """
    Return the relative path of a tile-companion file.

    Args:
        tile_id: tile ID like 'N45W110'
        suffix:  the artifact suffix, e.g. 'dem.tif', 'flat.tif',
                 'boundary.tif', 'meta.json', 'water.geojson'

    Examples:
        >>> tile_file_rel('N45W110', 'dem.tif')
        'tiles/N45W110/N45W110_dem.tif'
    """
    return f"tiles/{tile_id}/{tile_id}_{suffix}"


def absolute_path(storage_root: Path, rel_path: str) -> Path:
    """
    Resolve a registry-relative POSIX path to an absolute filesystem
    Path on this machine.
    """
    return Path(storage_root) / rel_path