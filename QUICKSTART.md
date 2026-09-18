# Quickstart

This tool downloads and processes elevation data for an area you choose, and saves it as a .tif file.

## 1. Get the code

Download or clone this repository to your computer.

## 2. Install Python

You need Python 3.10 or newer. Check what you have installed:

```bash
python3 --version
```

Download from [python.org](https://www.python.org/downloads/).

## 3. Install the required packages

From the project folder, run the commands:

```bash
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt
```

This creates a private folder (`.venv`) with everything the tool needs, isolated from rest of computer.

## 4. Download elevation data

Select a region of whole degree tiles by its two opposite corner coordinates, given as latitude/longitude points in the form `N44W113` (44°N, 113°W). Then run:

```bash
.venv/bin/python python/dem_download.py N44W113 N47W109
```

This downloads elevation data covering that rectangle and saves it as one combined .tif file in the current folder.

To fetch the highest resolution available, ~1m/px via USGS (United States Geological Survey) 3DEP (3D Elevation Program), add `--resolution 1m`. Coverage is sparse, so this is saved as a separate .tif file and connected to the base 10m/30m DEM (Digital Elevation Model) via GDAL (Geospatial Data Abstraction Library) VRT (Virtual Raster) to prevent coverage gaps. It's also far more expensive in both time to download and storage cost (over 50GB per tile), so it's recommended to narrow the request to a sub-tile region with the --bounds flag and diagonal corner input as shown here:

```bash
python python/dem_download.py --bounds "43.6150,-116.2050,43.6186,-116.2005" --resolution 1m

```



## Prefer R?

```bash
Rscript -e 'install.packages(c("terra", "httr2", "jsonlite", "maps"))'
```

See [r/README.md](r/README.md) for how to run the R version.

## For more details:

[README](README.md) covers all scripts and command-line options.
