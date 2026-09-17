"""
build_hires_vrt.py
━━━━━━━━━━━━━━━━━━
Builds a GDAL VRT (virtual raster) that layers one or more
higher-resolution rasters over a lower-resolution base, so a single
logical raster samples real fine detail where it exists and the base
resolution everywhere else -- without resampling the base up to the
finer grid (no storage blowup) and without touching the base file at
all. Any VRT-aware reader (GDAL, QGIS, rasterio) opens the resulting
`.vrt` exactly like a GeoTIFF.

Written by hand rather than via `gdal.BuildVRT()`: this project's venv
has rasterio (which bundles its own GDAL internally) but not the
separate `osgeo` Python bindings, and the VRT XML format is simple
enough not to need them -- see the GDAL VRT format docs for the
<ComplexSource>/<DstRect>/resampling= mechanics this mirrors.

Everything in this pipeline is EPSG:4326 with square pixels (degrees),
so this deliberately does not support mixed CRS or non-square pixels
-- reproject onto the base CRS first if that ever changes.
"""
import os
from pathlib import Path
from typing import List, Sequence, Union
from xml.sax.saxutils import escape as xml_escape

import rasterio
from rasterio.transform import from_origin

PathLike = Union[str, Path]

_GDAL_DTYPE = {
    'float32': 'Float32', 'float64': 'Float64',
    'uint8': 'Byte', 'int16': 'Int16', 'uint16': 'UInt16',
    'int32': 'Int32', 'uint32': 'UInt32',
}


def _raster_info(path: PathLike) -> dict:
    with rasterio.open(path) as ds:
        if ds.count != 1:
            raise ValueError(
                f"{path}: build_hires_vrt only supports single-band "
                f"rasters, got {ds.count} bands")
        if ds.transform.b != 0 or ds.transform.d != 0:
            raise ValueError(f"{path}: rotated/sheared rasters aren't supported")
        return {
            'path':      Path(path).resolve(),
            'crs':       ds.crs,
            'dtype':     ds.dtypes[0],
            'nodata':    ds.nodata,
            'width':     ds.width,
            'height':    ds.height,
            'transform': ds.transform,
            'bounds':    ds.bounds,
        }


def _source_xml(info: dict, vrt_dir: Path, origin_x: float, origin_y: float,
                 px_size: float, resampling: str = None) -> str:
    left, bottom, right, top = info['bounds']
    dst_x_off = round((left - origin_x) / px_size)
    dst_y_off = round((origin_y - top) / px_size)
    dst_w = max(1, round((right - left) / px_size))
    dst_h = max(1, round((top - bottom) / px_size))

    try:
        rel = os.path.relpath(info['path'], start=vrt_dir)
        filename, rel_flag = rel, '1'
    except ValueError:
        # e.g. different drives on Windows -- fall back to absolute.
        filename, rel_flag = str(info['path']), '0'

    resampling_attr = f' resampling="{resampling}"' if resampling else ''
    nodata_xml = (f"\n      <NODATA>{info['nodata']}</NODATA>"
                  if info['nodata'] is not None else '')

    return f'''    <ComplexSource{resampling_attr}>
      <SourceFilename relativeToVRT="{rel_flag}">{xml_escape(filename)}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{info['width']}" ySize="{info['height']}"/>
      <DstRect xOff="{dst_x_off}" yOff="{dst_y_off}" xSize="{dst_w}" ySize="{dst_h}"/>{nodata_xml}
    </ComplexSource>'''


def build_hires_vrt(base_path: PathLike, overlay_paths: Sequence[PathLike],
                     vrt_path: PathLike) -> Path:
    """
    Compose `base_path` (e.g. the standard 10m mosaic) with one or
    more `overlay_paths` (higher-resolution rasters, e.g. the native
    ~3m/~1m tiles that download_tile_3dep_3m()/download_tile_3dep_1m()
    persist whenever --try-3m/--resolution 1m finds real coverage)
    into a single `.vrt` at `vrt_path`.

    Overlays are drawn on top of the base in the given order. Each
    overlay's own NODATA value lets the base (or an earlier overlay)
    show through wherever that overlay has no real data, so a
    partial-coverage tile blends in at its edges instead of punching a
    nodata hole into the mosaic.

    The VRT's pixel grid is the *finest* resolution among all inputs,
    spanning the base raster's full extent. Reading it costs no more
    storage than the base + overlay files already do on disk -- the
    coarser source(s) are resampled on the fly per read (bilinear for
    the base, matching the resampling already used elsewhere in this
    pipeline for GLO-30-onto-10m), never materialized at the fine grid.

    Raises ValueError if `overlay_paths` is empty, any input isn't
    single-band, or an overlay's CRS doesn't match the base's
    (reprojection is out of scope here -- see module docstring).
    """
    overlay_paths = list(overlay_paths)
    if not overlay_paths:
        raise ValueError("build_hires_vrt needs at least one overlay raster")

    base = _raster_info(base_path)
    overlays = [_raster_info(p) for p in overlay_paths]

    for o in overlays:
        if o['crs'] != base['crs']:
            raise ValueError(
                f"{o['path']}: CRS {o['crs']} does not match base CRS "
                f"{base['crs']} -- build_hires_vrt does not reproject")

    finest = min([base] + overlays, key=lambda r: abs(r['transform'].a))
    px_size = abs(finest['transform'].a)

    left, bottom, right, top = base['bounds']
    vrt_width  = max(1, round((right - left) / px_size))
    vrt_height = max(1, round((top - bottom) / px_size))
    vrt_transform = from_origin(left, top, px_size, px_size)

    gdal_dtype = _GDAL_DTYPE.get(base['dtype'], 'Float32')
    nodata_band_xml = (f"\n    <NoDataValue>{base['nodata']}</NoDataValue>"
                        if base['nodata'] is not None else '')

    vrt_path = Path(vrt_path)
    vrt_dir = vrt_path.parent.resolve()

    sources = [_source_xml(base, vrt_dir, left, top, px_size, resampling='bilinear')]
    sources += [_source_xml(o, vrt_dir, left, top, px_size) for o in overlays]

    vrt_xml = (
        f'<VRTDataset rasterXSize="{vrt_width}" rasterYSize="{vrt_height}">\n'
        f'  <SRS>{xml_escape(base["crs"].to_wkt())}</SRS>\n'
        f'  <GeoTransform>{vrt_transform.c}, {vrt_transform.a}, '
        f'{vrt_transform.b}, {vrt_transform.f}, {vrt_transform.d}, '
        f'{vrt_transform.e}</GeoTransform>\n'
        f'  <VRTRasterBand dataType="{gdal_dtype}" band="1">'
        f'{nodata_band_xml}\n'
        + "\n".join(sources) + "\n"
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )

    vrt_dir.mkdir(parents=True, exist_ok=True)
    vrt_path.write_text(vrt_xml)
    return vrt_path
