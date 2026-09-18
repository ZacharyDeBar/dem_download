"""
visualize_cascade_aoi.py
━━━━━━━━━━━━━━━━━━━━━━━━
Renders a diagnostic report for a dem_download.py --cascade AOI
(cascade_manifest.json + whichever of base_30m.tif/base_10m.tif/
native_<res>.tif actually got real data): a hillshade of the
composited result, plus a per-pixel map of which tier actually
supplied each pixel's value.

This is genuinely per-pixel, unlike visualize_hires_vrt.py's coverage
map (which is necessarily per-*tile*, since a whole-degree study area
can only record one resolution label per tile) -- a --cascade AOI is
small enough (a few acres to a couple km) to check every pixel
directly against each of the (at most three) known layer files instead
of decimating or sampling.

Usage:
    python visualize_cascade_aoi.py data/dem/cascade_<name>
    python visualize_cascade_aoi.py data/dem/cascade_<name> --output report.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm

# Same ordinal ramp/tier convention as visualize_hires_vrt.py, minus
# the ocean/3m entries this narrower tool doesn't need.
TIER_ORDER  = ['30m', '10m', '3m', '1m']  # lowest to highest priority
TIER_COLORS = {
    '30m': '#5598e7',
    '10m': '#2a78d6',
    '3m':  '#1c5cab',
    '1m':  '#104281',
}
TIER_LABELS = {
    '30m': 'GLO-30 (30m, global fallback)',
    '10m': '3DEP (10m, US base)',
    '3m':  '3DEP ~3m',
    '1m':  '3DEP ~1m (genuine native)',
}
NO_DATA_COLOR = '#c9c8bd'
NODATA = -9999.0


def _load_manifest(cascade_dir: Path) -> dict:
    with open(cascade_dir / 'cascade_manifest.json') as f:
        return json.load(f)


def _read_values_on_grid(path, ref_transform, ref_crs, out_shape):
    """Reprojects `path`'s band 1 onto the reference grid, nodata-aware."""
    with rasterio.open(path) as src:
        src_nodata = src.nodata if src.nodata is not None else NODATA
        arr = np.full(out_shape, NODATA, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            src_nodata=src_nodata, dst_nodata=NODATA,
            resampling=Resampling.nearest,
        )
    return arr


def build_provenance(manifest: dict):
    """
    Returns (rank_grid, bounds, hillshade_path):
      rank_grid  int8 array, one of TIER_ORDER's indices per pixel (or
                 -1 where no layer had real data), on the composite's
                 own full-resolution grid.
      bounds     (west, east, south, north) for imshow.
      hillshade_path  the composite raster to render in panel 1.
    """
    composite_path = manifest['composite']
    with rasterio.open(composite_path) as ref:
        ref_transform, ref_crs = ref.transform, ref.crs
        out_shape = (ref.height, ref.width)
        b = ref.bounds

    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    rank = np.full(out_shape, -1, dtype=np.int8)

    # manifest['layers'] is already lowest-to-highest priority (see
    # download_cascade_aoi) -- iterating in that order and overwriting
    # with each subsequent layer's valid mask reproduces the exact
    # same priority compositing build_hires_vrt applied to the pixel
    # values themselves.
    for layer in manifest['layers']:
        tier = layer['tier']
        arr = _read_values_on_grid(layer['file'], ref_transform, ref_crs, out_shape)
        # Exact 0.0 is this pipeline's own out-of-coverage sentinel in
        # several places (the WCS native fetch's fill value, this
        # pipeline's "no legitimate near-zero terrain" convention) --
        # see elevation.py's fetch_dem_direct for where this was
        # discovered live.
        valid = (arr > NODATA + 1) & (arr != 0)
        rank[valid] = tier_rank[tier]

    return rank, (b.left, b.right, b.bottom, b.top), composite_path


def panel_hillshade(ax, raster_path, title):
    with rasterio.open(raster_path) as src:
        elev = src.read(1)
        nodata = src.nodata if src.nodata is not None else NODATA
        bounds = src.bounds
    masked = np.ma.masked_where((elev <= nodata + 1) | (elev == 0), elev)

    ls = LightSource(azdeg=315, altdeg=45)
    try:
        rgb = ls.shade(masked.filled(masked.mean() if masked.count() else 0.0),
                        cmap=plt.cm.terrain, blend_mode='overlay', vert_exag=3.0)
        ax.imshow(rgb, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                   origin='upper')
    except Exception:
        im = ax.imshow(masked, cmap='terrain',
                        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                        origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.04, label='elevation (m)')

    valid = masked.compressed()
    stats = (f'min {valid.min():.1f}m  max {valid.max():.1f}m  mean {valid.mean():.1f}m'
              if valid.size else 'no valid pixels')
    ax.set_title(f'{title}\n{stats}', fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')


def panel_provenance(ax, rank, extent, manifest):
    present_tiers = [layer['tier'] for layer in manifest['layers']]
    cmap = ListedColormap([NO_DATA_COLOR] + [TIER_COLORS[t] for t in TIER_ORDER])
    bounds_list = list(range(-1, len(TIER_ORDER) + 1))
    norm = BoundaryNorm(bounds_list, cmap.N)

    ax.imshow(rank, cmap=cmap, norm=norm, extent=extent, origin='upper')

    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[t])
               for t in TIER_ORDER if t in present_tiers]
    labels = [TIER_LABELS[t] for t in TIER_ORDER if t in present_tiers]
    if (rank == -1).any():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=NO_DATA_COLOR))
        labels.append('No data at any fetched tier')
    ax.legend(handles, labels, loc='lower right', fontsize=8, framealpha=0.85)

    total = rank.size
    pct_lines = []
    for t in TIER_ORDER:
        if t in present_tiers:
            pct = (rank == TIER_ORDER.index(t)).sum() / total * 100
            if pct > 0:
                pct_lines.append(f"{t}: {pct:.0f}%")
    title = 'Per-pixel resolution provenance  --  ' + ', '.join(pct_lines)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')


def build_report(cascade_dir: Path, output: Path):
    manifest = _load_manifest(cascade_dir)
    rank, extent, composite_path = build_provenance(manifest)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    panel_hillshade(axes[0], composite_path, 'Composited output (cascade.vrt)')
    panel_provenance(axes[1], rank, extent, manifest)

    tiers_used = ', '.join(l['tier'] for l in manifest['layers'])
    fig.suptitle(f"Cascade AOI: {cascade_dir.name}  --  tiers fetched: {tiers_used}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f'Wrote {output}')
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a dem_download.py --cascade AOI's per-pixel "
                    "resolution provenance.")
    parser.add_argument('cascade_dir',
        help='--cascade output directory (dem_download.py --output-dir)')
    parser.add_argument('--output', default=None,
        help='Output image path (default: <cascade_dir>/cascade_report.png)')
    args = parser.parse_args()

    cascade_dir = Path(args.cascade_dir)
    output = Path(args.output) if args.output else cascade_dir / 'cascade_report.png'
    build_report(cascade_dir, output)


if __name__ == '__main__':
    main()
