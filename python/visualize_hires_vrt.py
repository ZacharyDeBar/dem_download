"""
visualize_hires_vrt.py
━━━━━━━━━━━━━━━━━━━━━━
Renders a diagnostic figure for a study area built by dem_download.py:
a hillshade of the composited output (the `<area>_mosaic_hires.vrt` if
--try-3m/--resolution 1m ever found real coverage for this area, the
plain `<area>_mosaic.tif` otherwise), plus a resolution-coverage panel
showing which parts of the area are actually backed by which source --
GLO-30 (30m), 3DEP (10m), the ~3m tier, or genuine ~1m -- rather than
just "the hi-res VRT exists somewhere in here".

The per-tile base resolution comes from parsing each downloaded tile's
own filename in download_manifest.json (dem_download.py names every
cached tile `<tile_id>_<res_label>.tif`, e.g. `N45W110_3m.tif` --
ocean fill tiles are the one exception, named `N45_W110_ocean0m.tif`).
Within a tile, wherever a `<tile_id>_<tier>_native.tif` file also
exists (the raw, undownsampled ~3m/~1m fetch dem_download.py keeps
for the VRT overlay), that tile's real coverage fraction there is read
back via a decimated read of the raster's own GDAL mask band --
resolution capped during I/O, not after, so this stays cheap even
against a ~50GB native 1m raster (same technique dem_download.py's own
canonical-grid resampling relies on).

Reads only files already on disk under a study area's output
directory (as `dem_download.py <corners> [--try-3m | --resolution 1m]`
produces) -- no network access, no dem_download.py import. Needs
rasterio and matplotlib, same as visualize_tile.py.

Usage:
    python visualize_hires_vrt.py data/dem/N45W111_N45W111
    python visualize_hires_vrt.py data/dem/N45W111_N45W111 --output report.png
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm

from tile_id import parse_tile_id, tile_bounds

# Ordinal ramp (dataviz skill's single-hue "blue" sequential steps 250-650,
# the ordinal-safe band -- lightest step still clears 2:1 on a light
# surface) mapped onto DEM source quality, lightest/least-detailed first.
RESOLUTION_TIERS = ['ocean', '30m', '10m', '3m', '1m']
RESOLUTION_RANK = {tier: i for i, tier in enumerate(RESOLUTION_TIERS)}
RESOLUTION_COLORS = {
    'ocean': '#86b6ef',
    '30m':   '#5598e7',
    '10m':   '#2a78d6',
    '3m':    '#1c5cab',
    '1m':    '#104281',
}
RESOLUTION_LABELS = {
    'ocean': 'Ocean (flat 0m)',
    '30m':   'GLO-30 (30m, global)',
    '10m':   '3DEP (10m, US base)',
    '3m':    '3DEP ~3m (--try-3m)',
    '1m':    '3DEP ~1m (--resolution 1m)',
}
NO_DATA_COLOR = '#c9c8bd'  # failed/missing tile -- not part of the ordinal ramp

_OCEAN_FILENAME_RE = re.compile(
    r'([NS])(\d{2})_([EW])(\d{3})_(ocean\w*)\.tif$')
_STANDARD_FILENAME_RE = re.compile(r'([NS]\d{2}[EW]\d{3})_(\w+)\.tif$')


# ─────────────────────────────────────────────
# Manifest / artifact discovery
# ─────────────────────────────────────────────

def _load_manifest(output_dir: Path) -> dict:
    with open(output_dir / 'download_manifest.json') as f:
        return json.load(f)


def _parse_tile_filename(filename: str):
    """Returns (tile_id, res_label) parsed from a tile_files entry, or
    None if the filename doesn't match either naming convention this
    pipeline uses (see module docstring re: the ocean-tile exception)."""
    name = Path(filename).name
    m = _OCEAN_FILENAME_RE.match(name)
    if m:
        ns, lat, ew, lng, res = m.groups()
        return f"{ns}{lat}{ew}{lng}", 'ocean'
    m = _STANDARD_FILENAME_RE.match(name)
    if m:
        return m.group(1), m.group(2)
    return None


def _tile_resolution_map(manifest: dict) -> dict:
    """tile_id -> resolution tier ('ocean'/'30m'/'10m'/'3m'/'1m') --
    every res_label dem_download.py's own download_tile()/
    _create_ocean_tile() can produce already matches a RESOLUTION_TIERS
    entry verbatim once _parse_tile_filename normalizes the ocean
    special case, so entries that don't match anything known are
    dropped rather than guessed at."""
    out = {}
    for f in manifest.get('tile_files', []):
        parsed = _parse_tile_filename(f)
        if parsed is None:
            continue
        tile_id, res = parsed
        if res in RESOLUTION_RANK:
            out[tile_id] = res
    return out


def _all_tile_ids_in_bounds(bounds) -> list:
    """Every 1-degree tile id the study area's bounds touch -- south/
    west inclusive, north/east exclusive, matching tile_id.py's own
    convention (and dem_download.get_1deg_tiles)."""
    import math
    south, west, north, east = bounds
    lat_floors = range(math.floor(south), math.ceil(north))
    lng_floors = range(math.floor(west), math.ceil(east))
    return [f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'W' if lo < 0 else 'E'}{abs(lo):03d}"
            for la in lat_floors for lo in lng_floors]


def _native_tile_path(output_dir: Path, tile_id: str):
    """(path, tier) for tile_id's genuine ~1m or ~3m native raster, or
    None if neither exists. Prefers 1m -- a run can only ever produce
    one or the other per tile (--try-3m and --resolution 1m write to
    different native filenames and are never both requested for the
    same area), so this is just "whichever is actually there"."""
    raw_dir = output_dir / 'raw'
    for tier in ('1m', '3m'):
        p = raw_dir / f"{tile_id}_{tier}_native.tif"
        if p.exists():
            return p, tier
    return None


# ─────────────────────────────────────────────
# Raster reading
# ─────────────────────────────────────────────

def _decimated_band(path, max_dim: int, resampling=Resampling.average):
    """Decimated read, same convention as visualize_tile.py: downsample
    during I/O so a large mosaic/VRT never needs a full-res array in RAM
    just to make a report-sized panel."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = min(1.0, max_dim / max(h, w))
        out_h, out_w = max(1, round(h * scale)), max(1, round(w * scale))
        data = src.read(1, out_shape=(out_h, out_w), resampling=resampling)
        transform = src.transform * src.transform.scale(w / out_w, h / out_h)
        return data, transform, src.nodata, src.bounds


def _coverage_fraction(native_path: Path, out_shape) -> np.ndarray:
    """Fraction (0..1) of real (non-nodata) native pixels landing in
    each output cell, read via a decimated pass over the raster's own
    GDAL mask band -- never materializes the full native array (which
    can be ~50GB for a genuine 1m tile)."""
    with rasterio.open(native_path) as src:
        mask = src.read_masks(1, out_shape=out_shape, resampling=Resampling.average)
    return mask.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# Panels
# ─────────────────────────────────────────────

def panel_hillshade(ax, raster_path: Path, max_dim: int, title: str):
    elev, transform, nodata, bounds = _decimated_band(raster_path, max_dim)
    nodata = nodata if nodata is not None else -9999.0
    masked = np.ma.masked_where(elev <= nodata + 1, elev)

    ls = LightSource(azdeg=315, altdeg=45)
    try:
        rgb = ls.shade(masked.filled(masked.mean() if masked.count() else 0.0),
                        cmap=plt.cm.terrain, blend_mode='overlay', vert_exag=1.5)
        ax.imshow(rgb, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                   origin='upper')
    except Exception:
        im = ax.imshow(masked, cmap='terrain',
                        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                        origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.04, label='elevation (m)')

    valid = masked.compressed()
    stats = (f'min {valid.min():.0f}m  max {valid.max():.0f}m  mean {valid.mean():.0f}m'
              if valid.size else 'no valid pixels')
    ax.set_title(f'{title}\n{stats}', fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')


_COVERAGE_SAMPLE_GRID = 32  # just for the scalar mean in per_tile_stats


def build_resolution_grid(output_dir: Path, manifest: dict, max_dim: int):
    """
    Returns (rank_grid, extent, per_tile_stats):
      rank_grid   int array, one of RESOLUTION_RANK's values per cell
                  (the tile's *base* tier -- see note below), or -1
                  where no tile data exists at all (failed/missing).
      extent      (west, east, south, north) for imshow.
      per_tile_stats  list of dicts, one per tile: tile_id, base tier,
                  native tier (or None), coverage_pct (what fraction
                  of that tile's own area the native raster actually
                  has real, non-nodata data for).

    Note on what "native" coverage means here: a tile's resolution
    tier (10m/30m/3m/1m) is a whole-tile choice made once by
    download_tile() -- there's no *different*, lower tier recorded for
    the parts of a "3m"/"1m" tile that its own native fetch didn't
    reach. For the ~3m tier those gaps are already gap-filled from
    GLO-30 *inside* the same raster (still labeled and gridded as
    "3m"); for genuine ~1m they can be real nodata within an otherwise
    "1m" tile. Either way, painting those cells a different color on
    this map would claim a resolution the pipeline never actually
    recorded for them. `coverage_pct` instead reports, per tile, how
    much of it is real dense native measurement vs. filled/absent --
    the map answers "which resolution tier did this area use", the
    per-tile stats answer "how much of that tier's own area is real".
    """
    south, west, north, east = manifest['bounds']
    deg_h, deg_w = north - south, east - west
    out_h = max(1, round(max_dim * deg_h / max(deg_h, deg_w)))
    out_w = max(1, round(max_dim * deg_w / max(deg_h, deg_w)))
    deg_per_px = deg_h / out_h

    rank_grid = np.full((out_h, out_w), -1, dtype=np.int32)
    tile_res = _tile_resolution_map(manifest)
    per_tile_stats = []

    for tile_id in _all_tile_ids_in_bounds(manifest['bounds']):
        tb = tile_bounds(tile_id)
        col0 = int(round((tb.west - west) / deg_per_px))
        col1 = int(round((tb.east - west) / deg_per_px))
        row0 = int(round((north - tb.north) / deg_per_px))
        row1 = int(round((north - tb.south) / deg_per_px))
        col0, col1 = max(col0, 0), min(col1, out_w)
        row0, row1 = max(row0, 0), min(row1, out_h)
        if col1 <= col0 or row1 <= row0:
            continue  # tile doesn't actually overlap the display grid

        base_res = tile_res.get(tile_id)
        if base_res is None:
            continue  # failed/never-downloaded tile -- left as -1 (no data)

        base_rank = RESOLUTION_RANK.get(base_res, RESOLUTION_RANK['10m'])
        rank_grid[row0:row1, col0:col1] = base_rank

        stat = {'tile_id': tile_id, 'base': base_res, 'native': None,
                'coverage_pct': 0.0}
        native = _native_tile_path(output_dir, tile_id)
        if native is not None:
            native_path, tier = native
            frac = _coverage_fraction(
                native_path, out_shape=(_COVERAGE_SAMPLE_GRID, _COVERAGE_SAMPLE_GRID))
            stat['native'] = tier
            stat['coverage_pct'] = float(frac.mean() * 100.0)
        per_tile_stats.append(stat)

    return rank_grid, (west, east, south, north), per_tile_stats


def panel_resolution_coverage(ax, rank_grid, extent, per_tile_stats):
    cmap = ListedColormap([NO_DATA_COLOR] + [RESOLUTION_COLORS[t] for t in RESOLUTION_TIERS])
    bounds_list = list(range(-1, len(RESOLUTION_TIERS) + 1))
    norm = BoundaryNorm(bounds_list, cmap.N)

    ax.imshow(rank_grid, cmap=cmap, norm=norm, extent=extent, origin='upper')

    present_tiers = ({s['base'] for s in per_tile_stats} |
                      {s['native'] for s in per_tile_stats if s['native']})
    handles = [plt.Rectangle((0, 0), 1, 1, color=RESOLUTION_COLORS[t])
               for t in RESOLUTION_TIERS if t in present_tiers]
    labels = [RESOLUTION_LABELS[t] for t in RESOLUTION_TIERS if t in present_tiers]
    if -1 in rank_grid:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=NO_DATA_COLOR))
        labels.append('No data (failed/missing tile)')
    ax.legend(handles, labels, loc='lower right', fontsize=8, framealpha=0.85)

    hires_tiles = [s for s in per_tile_stats if s['native']]
    if hires_tiles:
        mean_cov = np.mean([s['coverage_pct'] for s in hires_tiles])
        title = (f'Resolution coverage  --  {len(hires_tiles)}/{len(per_tile_stats)} '
                 f'tile(s) have real hi-res data (avg {mean_cov:.0f}% of tile area)')
    else:
        title = 'Resolution coverage  --  no hi-res tier requested/found for this area'
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('lon'); ax.set_ylabel('lat')


def panel_tile_table(ax, per_tile_stats):
    ax.axis('off')
    if not per_tile_stats:
        ax.text(0.5, 0.5, 'No tiles to report', ha='center', va='center')
        return

    rows = [[s['tile_id'], RESOLUTION_LABELS[s['base']],
             RESOLUTION_LABELS[s['native']] if s['native'] else '-- ',
             f"{s['coverage_pct']:.0f}%" if s['native'] else '-- ']
            for s in sorted(per_tile_stats, key=lambda s: s['tile_id'])]
    col_labels = ['Tile', 'Base source', 'Native hi-res tier', 'Hi-res coverage of tile']
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1, 1.4)
    ax.set_title('Per-tile breakdown', fontsize=10)


# ─────────────────────────────────────────────
# Report assembly
# ─────────────────────────────────────────────

def build_report(output_dir: Path, manifest: dict, output: Path, max_dim: int = 1500):
    name = manifest['study_area']

    vrt_path = output_dir / f'{name}_mosaic_hires.vrt'
    mosaic_path = output_dir / f'{name}_mosaic.tif'
    hillshade_path = vrt_path if vrt_path.exists() else mosaic_path
    if not hillshade_path.exists():
        raise FileNotFoundError(
            f"Neither {vrt_path.name} nor {mosaic_path.name} exists under "
            f"{output_dir} -- run dem_download.py for this area first.")

    rank_grid, extent, per_tile_stats = build_resolution_grid(
        output_dir, manifest, max_dim)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5),
                              gridspec_kw={'width_ratios': [1, 1, 1.1]})
    panel_hillshade(axes[0], hillshade_path, max_dim,
                     'Composited output'
                     f' ({"hi-res VRT" if hillshade_path == vrt_path else "base mosaic"})')
    panel_resolution_coverage(axes[1], rank_grid, extent, per_tile_stats)
    panel_tile_table(axes[2], per_tile_stats)

    fig.suptitle(f"{name}  --  {manifest.get('downloaded', '?')}/"
                 f"{manifest.get('total_tiles', '?')} tiles downloaded",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f'Wrote {output}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize a study area\'s hi-res coverage (dem_download.py '
                    '--try-3m / --resolution 1m output).')
    parser.add_argument('output_dir',
        help='Study area output directory (dem_download.py\'s --output-dir, '
             'default data/dem/<corner1>_<corner2>)')
    parser.add_argument('--output', default=None,
        help='Output image path (default: <output_dir>/<area>_hires_coverage.png)')
    parser.add_argument('--max-dim', type=int, default=1500,
        help='Max raster/grid dimension to read/render at (default: 1500)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = _load_manifest(output_dir)
    output = Path(args.output) if args.output else (
        output_dir / f"{manifest['study_area']}_hires_coverage.png")

    build_report(output_dir, manifest, output, args.max_dim)


if __name__ == '__main__':
    main()
