"""
tile_validation.py
━━━━━━━━━━━━━━━━━━

Shared "is this tile's DEM file actually the shape it claims to be"
check, used both at tile-BUILD time (tile_builder.py, so a bad tile
is never marked READY in the first place) and for retroactive corpus
audits (a sibling integrity-auditor script outside this repo) -- one
source of truth for what a valid tile looks like, rather than
duplicating the check in both places.

Real incident this exists because of (2026-07-30): 27 tiles, all built in the same ~16-minute
window on 2026-06-11, had truncated DEM files (as small as 541x1405
against an expected 10800x10800) that sat marked STATUS_READY in the
registry for 6+ weeks, undetected until a route's stored geometry was
inspected closely. Root cause (tile_builder.py's
_step_crop_and_fill_gaps): the reproject-into-canonical-grid step was
gated on `cropped_transform != dst_transform`, which only compares the
affine origin/pixel-size coefficients, not the actual array shape -- a
truncated/partial provider download (same origin and pixel size as
expected, just fewer rows/cols) compared equal and skipped
reprojection, silently writing the truncated array straight to disk.
Fixed at the source, but this module exists as defense-in-depth: verify
what actually landed on disk, not just that the pipeline ran without
raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tile_id import tile_bounds, STANDARD_GRID_RES_DEG

# 1deg / (1/10800 deg/pixel) = 10800 pixels per tile edge.
EXPECTED_TILE_SIZE_PX = round(1.0 / STANDARD_GRID_RES_DEG)

# How far (in degrees) a file's reported bounds may drift from the
# tile's exact expected footprint and still count as valid. Generous
# relative to floating-point transform noise, tight enough that a real
# truncation (which was off by 0.03-0.95 degrees in the real incident)
# is never missed.
DEFAULT_BOUNDS_TOL_DEG = 1e-4


@dataclass
class TileValidationResult:
    ok: bool
    reason: Optional[str] = None   # human-readable, only set when ok=False
    width: Optional[int] = None
    height: Optional[int] = None
    bounds: Optional[tuple] = None   # (left, bottom, right, top) as actually read


def validate_tile_raster(
    path: Path,
    tile_id: str,
    expected_size_px: int = EXPECTED_TILE_SIZE_PX,
    bounds_tol_deg: float = DEFAULT_BOUNDS_TOL_DEG,
) -> TileValidationResult:
    """
    Open `path` (a DEM or sidecar mask/correction raster for `tile_id`)
    and verify it actually covers the tile's full 1x1 degree footprint
    at the standard grid resolution -- not just that it opens cleanly.

    Read-only, header-only (width/height/bounds), never reads pixel
    data -- fast and safe to run against a live storage root.
    """
    import rasterio

    if not path.exists():
        return TileValidationResult(ok=False, reason="file missing on disk")

    expected = tile_bounds(tile_id)
    try:
        with rasterio.open(path) as src:
            w, h = src.width, src.height
            b = src.bounds
    except Exception as e:
        return TileValidationResult(
            ok=False, reason=f"unreadable: {type(e).__name__}: {e}")

    issues = []
    if w != expected_size_px or h != expected_size_px:
        issues.append(f"wrong size {w}x{h} (expected "
                       f"{expected_size_px}x{expected_size_px})")
    if (abs(b.left - expected.west) > bounds_tol_deg or
            abs(b.bottom - expected.south) > bounds_tol_deg or
            abs(b.right - expected.east) > bounds_tol_deg or
            abs(b.top - expected.north) > bounds_tol_deg):
        issues.append(
            f"bounds ({b.left:.4f},{b.bottom:.4f},{b.right:.4f},{b.top:.4f}) "
            f"!= expected ({expected.west:.4f},{expected.south:.4f},"
            f"{expected.east:.4f},{expected.north:.4f})")

    if issues:
        return TileValidationResult(
            ok=False, reason='; '.join(issues), width=w, height=h,
            bounds=(b.left, b.bottom, b.right, b.top))
    return TileValidationResult(
        ok=True, width=w, height=h, bounds=(b.left, b.bottom, b.right, b.top))
