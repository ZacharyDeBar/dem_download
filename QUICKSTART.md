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

Select an area by its two opposite corner coordinates, given as latitude/longitude points in the form `N44W113` (44°N, 113°W). Then run:

```bash
.venv/bin/python python/dem_download.py N44W113 N47W109
```

This downloads elevation data covering that rectangle and saves it as one combined .tif file in the current folder.

To fetch the highest resolution available (genuine ~1m/px where USGS 3DEP has it surveyed), add `--resolution 1m`. Coverage is sparse, so this is saved as a separate .tif file and connected to the base 10m/30m DEM via GDAL VRT. It's also far more expensive than the default — potentially thousands of requests and tens of minutes per tile — since it streams real full-resolution data instead of a cheaper approximation.

```bash
.venv/bin/python python/dem_download.py N44W113 N47W109 --resolution 1m
```
There's also a cheaper `--try-3m` tier (~3m/px, ~900 requests/tile, a few minutes) if 1m's cost isn't worth it for your area — see [High-resolution output](README.md#high-resolution-output) in the full README for the tradeoffs between the two.


## Prefer R?

```bash
Rscript -e 'install.packages(c("terra", "httr2", "jsonlite", "maps"))'
```

See [r/README.md](r/README.md) for how to run the R version.

## Want more control?

The [full README](README.md) covers all scripts and command-line options.
