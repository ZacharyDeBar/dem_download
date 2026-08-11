"""
visualize_tile.py
━━━━━━━━━━━━━━━━━━
Renders a one-page diagnostic figure for a single built tile: hillshade +
elevation, water polygons, the water-correction elevation diff, flat/
boundary mask coverage, the gap-fill void map, and a correction magnitude
summary — whichever of these the tile's on-disk artifacts support.

Reads only what's already on disk under a tile's directory. Needs nothing
but rasterio and matplotlib (numpy comes along as rasterio's own
dependency) — no scipy, shapely, requests, or GPU deps, so it runs
anywhere the finished tile artifacts live, not just on a full pipeline
checkout.

Artifacts read (all except the DEM itself are optional — the panel that
needs a missing one is skipped, not faked):
    {tile}_dem.tif               required — hillshade + elevation panel
    {tile}_meta.json              optional — header stats
    {tile}_water.geojson          optional — water polygons panel
    {tile}_flat.tif + _boundary.tif   optional — flat/boundary panel
    {tile}_gap_mask.tif           optional, needs build_tile(keep_original=True)
                                   or `tile_builder.py build --keep-original`
                                   — void (gap-fill) panel
    {tile}_dem_precorrection.tif  optional, needs --keep-original
                                   — elevation-diff panel
    {tile}_water_stats.json       optional, needs --keep-original
                                   — correction histogram + top-N table

Usage:
    python visualize_tile.py N45W110 --storage-root data
    python visualize_tile.py N45W110 --tile-dir /path/to/tiles/N45W110
    python visualize_tile.py N45W110 --storage-root data --output report.png
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm

WATER_SOURCE_COLORS = {
    'nhd':       '#2b7fd6',
    'osm':       '#33b0a4',
    'osm_ocean': '#1a3f8f',
}
WATER_SOURCE_LABELS = {
    'nhd':       'NHD (US)',
    'osm':       'OSM',
    'osm_ocean': 'Ocean (OSM)',
}


# ─────────────────────────────────────────────
# Artifact discovery + reading
# ─────────────────────────────────────────────

def _resolve_tile_dir(tile_id: str, tile_dir: str = None, storage_root: str = None) -> Path:
    if tile_dir:
        return Path(tile_dir)
    root = Path(storage_root or 'data')
    return root / 'tiles' / tile_id


def _read_band(path: Path, max_dim: int, resampling: Resampling):
    """Decimated read: downsamples during I/O rather than after, so a
    10800x10800 tile doesn't need a full-res array in RAM just to make
    a 2000px PNG. Returns (data, transform, nodata, bounds) all matching
    the actual (possibly downsampled) array."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = min(1.0, max_dim / max(h, w))
        out_h, out_w = max(1, round(h * scale)), max(1, round(w * scale))
        data = src.read(1, out_shape=(out_h, out_w), resampling=resampling)
        out_transform = src.transform * src.transform.scale(w / out_w, h / out_h)
        return data, out_transform, src.nodata, src.bounds


def _pixel_size_m(transform, center_lat_deg: float):
    """Approximate meters-per-pixel at the tile's central latitude —
    good enough for a hillshade's light geometry, not for measurement."""
    deg_per_px_x = abs(transform.a)
    deg_per_px_y = abs(transform.e)
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(math.radians(center_lat_deg))
    return deg_per_px_x * m_per_deg_lng, deg_per_px_y * m_per_deg_lat


def _load_water_features(path: Path):
    with open(path) as f:
        fc = json.load(f)
    return fc.get('features', [])


def _polygon_rings(geometry: dict):
    """Yield exterior rings (list of (lon, lat)) for a Polygon or
    MultiPolygon geometry dict. Holes are ignored — fine for a
    visualization overlay."""
    gtype = geometry.get('type', 'Polygon')
    coords = geometry.get('coordinates', [])
    if gtype == 'Polygon':
        if coords:
            yield coords[0]
    elif gtype == 'MultiPolygon':
        for poly in coords:
            if poly:
                yield poly[0]


# ─────────────────────────────────────────────
# Panel renderers — each takes an Axes and draws into it
# ─────────────────────────────────────────────

def panel_hillshade_elevation(ax, dem_path: Path, max_dim: int):
    elev, transform, nodata, bounds = _read_band(dem_path, max_dim, Resampling.average)
    nodata = nodata if nodata is not None else -9999.0
    masked = np.ma.masked_where(elev <= nodata + 1, elev)

    dx, dy = _pixel_size_m(transform, (bounds.top + bounds.bottom) / 2)
    ls = LightSource(azdeg=315, altdeg=45)
    try:
        rgb = ls.shade(masked.filled(masked.mean() if masked.count() else 0.0),
                        cmap=plt.cm.terrain, blend_mode='overlay',
                        vert_exag=1.5, dx=max(dx, 1e-6), dy=max(dy, 1e-6))
        ax.imshow(rgb, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                   origin='upper')
    except Exception:
        im = ax.imshow(masked, cmap='terrain',
                        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                        origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.04, label='elevation (m)')

    valid = masked.compressed()
    stats = (f'min {valid.min():.0f}m  max {valid.max():.0f}m  '
             f'mean {valid.mean():.0f}m' if valid.size else 'no valid pixels')
    ax.set_title(f'Hillshade + elevation\n{stats}', fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')
    return bounds


def _hillshade_basemap(ax, dem_path: Path, max_dim: int):
    """Shared grey hillshade backdrop for overlay panels (water/void)."""
    elev, transform, nodata, bounds = _read_band(dem_path, max_dim, Resampling.average)
    nodata = nodata if nodata is not None else -9999.0
    masked = np.ma.masked_where(elev <= nodata + 1, elev)
    dx, dy = _pixel_size_m(transform, (bounds.top + bounds.bottom) / 2)
    ls = LightSource(azdeg=315, altdeg=45)
    try:
        hs = ls.hillshade(masked.filled(masked.mean() if masked.count() else 0.0),
                           vert_exag=1.5, dx=max(dx, 1e-6), dy=max(dy, 1e-6))
        ax.imshow(hs, cmap='gray',
                   extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                   origin='upper')
    except Exception:
        pass
    return bounds


def panel_water_polygons(ax, dem_path: Path, water_path: Path, max_dim: int):
    bounds = _hillshade_basemap(ax, dem_path, max_dim)
    features = _load_water_features(water_path)

    seen_sources = set()
    for feat in features:
        geom = feat.get('geometry', {})
        source = feat.get('properties', {}).get('source', 'unknown')
        color = WATER_SOURCE_COLORS.get(source, '#999999')
        for ring in _polygon_rings(geom):
            lons = [pt[0] for pt in ring]
            lats = [pt[1] for pt in ring]
            ax.fill(lons, lats, color=color, alpha=0.6,
                     edgecolor=color, linewidth=0.3)
        seen_sources.add(source)

    handles = [plt.Rectangle((0, 0), 1, 1, color=WATER_SOURCE_COLORS.get(s, '#999999'))
               for s in seen_sources]
    labels = [WATER_SOURCE_LABELS.get(s, s) for s in seen_sources]
    if handles:
        ax.legend(handles, labels, loc='lower right', fontsize=8, framealpha=0.8)

    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_title(f'Water polygons ({len(features)} total)', fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')


def panel_elevation_diff(ax, dem_path: Path, precorrection_path: Path, max_dim: int):
    corrected, transform, nodata, bounds = _read_band(dem_path, max_dim, Resampling.average)
    original, _, _, _ = _read_band(precorrection_path, max_dim, Resampling.average)
    nodata = nodata if nodata is not None else -9999.0

    valid = (corrected > nodata + 1) & (original > nodata + 1)
    diff = np.ma.masked_where(~valid, corrected - original)

    max_abs = float(np.abs(diff).max()) if diff.count() else 1.0
    max_abs = max(max_abs, 1e-3)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    im = ax.imshow(diff, cmap='RdBu_r', norm=norm,
                    extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                    origin='upper')
    plt.colorbar(im, ax=ax, fraction=0.04, label='corrected − original (m)')

    changed = diff.compressed()
    changed = changed[changed != 0]
    pct_changed = 100.0 * changed.size / max(diff.count(), 1)
    ax.set_title(f'Water correction: elevation diff\n'
                  f'{pct_changed:.1f}% of pixels changed', fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')


def panel_flat_boundary(ax, dem_path: Path, flat_path: Path, boundary_path: Path, max_dim: int):
    bounds = _hillshade_basemap(ax, dem_path, max_dim)
    flat, ftransform, _, fbounds = _read_band(flat_path, max_dim, Resampling.nearest)
    boundary, _, _, _ = _read_band(boundary_path, max_dim, Resampling.nearest)

    flat_rgba = np.zeros((*flat.shape, 4))
    flat_rgba[flat > 0] = (0.2, 0.6, 1.0, 0.35)
    ax.imshow(flat_rgba, extent=(fbounds.left, fbounds.right, fbounds.bottom, fbounds.top),
               origin='upper')

    boundary_rgba = np.zeros((*boundary.shape, 4))
    boundary_rgba[boundary > 0] = (1.0, 0.25, 0.1, 0.9)
    ax.imshow(boundary_rgba, extent=(fbounds.left, fbounds.right, fbounds.bottom, fbounds.top),
               origin='upper')

    flat_n = int((flat > 0).sum())
    boundary_n = int((boundary > 0).sum())
    reduction_pct = 100.0 * (1 - boundary_n / flat_n) if flat_n else 0.0
    ax.set_title(f'Flat mask (blue) + boundary (red)\n'
                  f'{100*flat_n/flat.size:.1f}% flat, boundary is '
                  f'{reduction_pct:.1f}% smaller than full flat mask', fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')


def panel_void_map(ax, dem_path: Path, gap_mask_path: Path, meta: dict, max_dim: int):
    bounds = _hillshade_basemap(ax, dem_path, max_dim)
    gap, _, _, gbounds = _read_band(gap_mask_path, max_dim, Resampling.nearest)

    gap_rgba = np.zeros((*gap.shape, 4))
    gap_rgba[gap > 0] = (1.0, 0.15, 0.15, 0.85)
    ax.imshow(gap_rgba, extent=(gbounds.left, gbounds.right, gbounds.bottom, gbounds.top),
               origin='upper')

    n_gap = int((gap > 0).sum())
    pct = 100.0 * n_gap / gap.size
    filled_from = (meta or {}).get('gaps_filled_from') or 'GLO-30'
    ax.set_title(f'Void map: gaps filled from {filled_from}\n'
                  f'{pct:.2f}% of tile was void pre-fill', fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')


def panel_correction_histogram(ax, elevation_changes: list):
    changes = [c['change_m'] for c in elevation_changes]
    ax.hist(changes, bins=min(30, max(5, len(changes) // 2 or 1)),
             color='#2b7fd6', edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'Water correction magnitude\n{len(changes)} bodies corrected', fontsize=10)
    ax.set_xlabel('change_m (surface_elev − original_mean)')
    ax.set_ylabel('water body count')


def _format_midpoint(c: dict) -> str:
    lat, lon = c.get('lat'), c.get('lon')
    if lat is None or lon is None:
        return '—'  # older water_stats.json, written before this field existed
    return f'{lat:.4f}, {lon:.4f}'


def panel_correction_table(ax, elevation_changes: list, top_n: int):
    ax.axis('off')
    ranked = sorted(elevation_changes, key=lambda c: abs(c['change_m']), reverse=True)[:top_n]
    rows = [[str(c.get('polygon_index', '')), f"{c['pixel_count']:,}",
             f"{c['original_mean_m']:.1f}", f"{c['surface_elev_m']:.1f}",
             f"{c['change_m']:+.1f}", _format_midpoint(c)] for c in ranked]
    col_labels = ['poly #', 'pixels', 'orig (m)', 'corrected (m)', 'Δ (m)',
                  'midpoint (lat, lon)']
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc='right', bbox=[0, 0, 1, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # Equal-width columns overflow into their neighbors once cell text is
    # wider than 1/ncols of the axes (the lat/lon column especially) —
    # size columns to their actual content instead.
    table.auto_set_column_width(col=list(range(len(col_labels))))
    ax.set_title(f'Top {len(ranked)} corrections by magnitude\n'
                 f'midpoint = mean of ring vertices, rough — not a true centroid',
                 fontsize=10)


# ─────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────

def build_report(tile_id: str, tile_dir: Path, output: Path, max_dim: int, top_n: int):
    dem_path = tile_dir / f'{tile_id}_dem.tif'
    if not dem_path.exists():
        raise FileNotFoundError(
            f'{dem_path} not found — visualize_tile.py needs at least the '
            f'built DEM. Check --tile-dir/--storage-root and tile_id.')

    meta_path = tile_dir / f'{tile_id}_meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

    water_path = tile_dir / f'{tile_id}_water.geojson'
    flat_path = tile_dir / f'{tile_id}_flat.tif'
    boundary_path = tile_dir / f'{tile_id}_boundary.tif'
    gap_mask_path = tile_dir / f'{tile_id}_gap_mask.tif'
    precorrection_path = tile_dir / f'{tile_id}_dem_precorrection.tif'
    water_stats_path = tile_dir / f'{tile_id}_water_stats.json'

    elevation_changes = []
    if water_stats_path.exists():
        elevation_changes = json.loads(water_stats_path.read_text()).get('elevation_changes', [])

    # Each entry: (label, kind, render closure taking an Axes)
    panels = [('Hillshade + elevation', 'single',
               lambda ax: panel_hillshade_elevation(ax, dem_path, max_dim))]

    if water_path.exists():
        panels.append(('Water polygons', 'single',
                        lambda ax: panel_water_polygons(ax, dem_path, water_path, max_dim)))
    else:
        print(f'[SKIP] water polygons panel — {water_path.name} not found')

    if precorrection_path.exists():
        panels.append(('Elevation diff', 'single',
                        lambda ax: panel_elevation_diff(ax, dem_path, precorrection_path, max_dim)))
    else:
        print(f'[SKIP] elevation diff panel — {precorrection_path.name} not found '
              f'(build with tile_builder.py build --keep-original to get it)')

    if flat_path.exists() and boundary_path.exists():
        panels.append(('Flat + boundary', 'single',
                        lambda ax: panel_flat_boundary(ax, dem_path, flat_path, boundary_path, max_dim)))
    else:
        print('[SKIP] flat/boundary panel — flat.tif and/or boundary.tif not found')

    if gap_mask_path.exists():
        panels.append(('Void map', 'single',
                        lambda ax: panel_void_map(ax, dem_path, gap_mask_path, meta, max_dim)))
    else:
        print(f'[SKIP] void map panel — {gap_mask_path.name} not found '
              f'(build with tile_builder.py build --keep-original to get it)')

    if elevation_changes:
        panels.append(('Correction histogram', 'single',
                        lambda ax: panel_correction_histogram(ax, elevation_changes)))
        panels.append(('Correction table', 'single',
                        lambda ax: panel_correction_table(ax, elevation_changes, top_n)))
    else:
        print(f'[SKIP] correction summary panels — {water_stats_path.name} not found '
              f'or empty (build with tile_builder.py build --keep-original to get it)')

    ncols = 2
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for i, (label, _, render) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        render(ax)

    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    suptitle = f'{tile_id}'
    if meta:
        suptitle += (f"  —  {meta.get('source', '?')}, "
                      f"{meta.get('final_size_mb', 0):.0f}MB, "
                      f"built {meta.get('build_timestamp_utc', '?')}")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f'\nWrote {output} ({len(panels)} panels)')


def main():
    parser = argparse.ArgumentParser(
        description='Render a diagnostic figure for a built DEM tile.')
    parser.add_argument('tile_id', help='Tile ID like N45W110')
    parser.add_argument('--storage-root', default='data',
        help='Storage root — tile lives under <storage-root>/tiles/<tile_id>/ '
             '(default: data)')
    parser.add_argument('--tile-dir', default=None,
        help='Direct path to the tile\'s directory, overrides --storage-root')
    parser.add_argument('--output', default=None,
        help='Output image path (default: <tile_dir>/<tile_id>_report.png)')
    parser.add_argument('--max-dim', type=int, default=2000,
        help='Max raster dimension to read/plot at — larger tiles are '
             'decimated during read, not after (default: 2000)')
    parser.add_argument('--top-n', type=int, default=10,
        help='Rows in the top-corrections table (default: 10)')
    args = parser.parse_args()

    tile_dir = _resolve_tile_dir(args.tile_id, args.tile_dir, args.storage_root)
    output = Path(args.output) if args.output else tile_dir / f'{args.tile_id}_report.png'

    build_report(args.tile_id, tile_dir, output, args.max_dim, args.top_n)


if __name__ == '__main__':
    main()
