#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate daily LST (land surface temperature) to annual mean
for each RTS polygon.

For each RTS polygon ID:
1. For each daily LST raster:
   - Mask the raster by the polygon geometry.
   - Compute the mean LST over all intersecting pixels
     (polygon mean value for that day).
2. For each year, average all daily polygon-mean values to obtain
   one annual mean LST per polygon and per year.
3. Save, for each RTS ID, a text file with:
       Year    LST_mean

Notes
-----
- This script assumes that all daily LST rasters share the same grid
  (transform, CRS, pixel size, etc.).
- Polygon IDs are read from the "ID" field of the shapefile.
"""

import os
from typing import Dict, Optional, Tuple, Iterable

import geopandas as gpd
import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.features import geometry_mask
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm


# ================================================================
# USER SETTINGS (EDIT THESE PATHS)
# ================================================================

# Optional: set GDAL/PROJ environment variables if needed.
# Comment these out if your GDAL/PROJ are configured globally.
os.environ["GDAL_DATA"] = r"D:\app\Anoconda\envs\lab\Lib\site-packages\osgeo\data\gdal"
os.environ["PROJ_LIB"] = r"D:\app\Anoconda\envs\lab\Lib\site-packages\osgeo\data\proj"

# RTS_SHP_PATH:
#   Full path to the RTS polygon shapefile.
#   Required format: ESRI Shapefile (.shp) containing polygon geometries
#   and an "ID" attribute for each RTS.
RTS_SHP_PATH = r"E:\zyz\DiplomaProject\data\RTS-QTP\RTS-QTP_BLH_84.shp"

# LST_ROOT:
#   Root folder of daily LST rasters.
#   Required directory structure:
#       LST_ROOT/YYYY/MM/*.tif
#   where each GeoTIFF (*.tif) is a single-band daily LST raster
#   (e.g., in Kelvin or °C; units are not modified by this script).
LST_ROOT = r"F:\LST_ma\Daily_LST_TP"  # <-- EDIT to your actual LST root

# SAMPLE_TIF:
#   Path to a representative daily LST GeoTIFF file, used to read the CRS
#   and ensure the shapefile is reprojected to the same coordinate system.
#   This MUST point to an existing .tif file within LST_ROOT.
SAMPLE_TIF = r"F:\LST_ma\Daily_LST_TP\2016\01\2016_01_01_LST.tif"  # <-- EDIT

# OUTPUT_DIR:
#   Folder where annual mean LST results will be saved.
#   The script will create this folder if it does not exist.
#   For each RTS ID, a file "<ID>_LST_mean.txt" will be written.
OUTPUT_DIR = r"E:\zyz\DiplomaProject\data\Slumps\Copula\LST_mean_GMCP"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# RASTER / GEOMETRY HELPERS
# ================================================================

def polygon_window(
    src: rasterio.io.DatasetReader,
    geom: BaseGeometry,
) -> Optional[Window]:
    """
    Compute a minimal raster window covering a polygon's bounding box.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Open raster dataset.
    geom : shapely geometry
        Polygon or multipolygon geometry.

    Returns
    -------
    window : rasterio.windows.Window or None
        Window covering the polygon's bounding box, or None if the
        polygon does not intersect the raster grid.
    """
    minx, miny, maxx, maxy = geom.bounds
    row_min, col_min = src.index(minx, maxy)
    row_max, col_max = src.index(maxx, miny)

    row_off = max(0, min(row_min, row_max))
    col_off = max(0, min(col_min, col_max))
    row_stop = min(src.height, max(row_min, row_max) + 1)
    col_stop = min(src.width, max(col_min, col_max) + 1)

    height = row_stop - row_off
    width = col_stop - col_off

    if height <= 0 or width <= 0:
        return None

    return Window(col_off, row_off, width, height)


def polygon_mean_from_raster(
    src: rasterio.io.DatasetReader,
    geom: BaseGeometry,
) -> Optional[float]:
    """
    Compute the mean value of all pixels intersecting a polygon.

    Steps
    -----
    1. Restrict the raster to a window around the polygon bounding box.
    2. Build a geometry mask and keep only pixels within the polygon.
    3. Remove nodata and non-finite values.
    4. Return the mean of remaining pixels, or None if empty.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Open raster dataset.
    geom : shapely geometry
        Polygon or multipolygon geometry.

    Returns
    -------
    mean_val : float or None
        Mean raster value within the polygon, or None if no valid pixels.
    """
    window = polygon_window(src, geom)
    if window is None:
        return None

    data = src.read(1, window=window)
    window_transform = src.window_transform(window)
    nodata = src.nodata

    # Mask: True inside polygon, False outside
    mask = geometry_mask(
        [geom],
        out_shape=data.shape,
        transform=window_transform,
        invert=True,      # True where geometry intersects
        all_touched=True, # Count any pixel touched by the polygon
    )

    vals = data[mask]

    if nodata is not None:
        vals = vals[vals != nodata]

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    return float(np.mean(vals))


def iter_year_month_dirs(root: str) -> Iterable[Tuple[int, str]]:
    """
    Iterate over (year, year_path) for all year folders in the root.

    Only directories whose names can be parsed as integer years
    are considered.

    Parameters
    ----------
    root : str
        Root folder containing year subfolders.

    Yields
    ------
    year : int
        Year parsed from folder name.
    year_path : str
        Full path to the year folder.
    """
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            year = int(name)
        except ValueError:
            continue
        yield year, path


# ================================================================
# MAIN PROCESSING
# ================================================================

def main() -> None:
    # ----------------------------------------------------------------
    # Load RTS polygons
    # ----------------------------------------------------------------
    gdf = gpd.read_file(RTS_SHP_PATH)
    if "ID" not in gdf.columns:
        raise ValueError('Shapefile must contain an "ID" field for each RTS.')

    print(f"Loaded {len(gdf)} RTS polygons from shapefile.")

    # ----------------------------------------------------------------
    # Read sample raster to get CRS and align shapefile
    # ----------------------------------------------------------------
    if not os.path.isfile(SAMPLE_TIF):
        raise FileNotFoundError(f"SAMPLE_TIF does not exist: {SAMPLE_TIF}")

    with rasterio.open(SAMPLE_TIF) as src_sample:
        raster_crs = src_sample.crs

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
        print(f"Reprojected RTS shapefile to match raster CRS: {raster_crs}")

    # RTS_ID -> geometry
    rts_geoms: Dict[int, BaseGeometry] = {}
    for _, row in gdf.iterrows():
        rts_id = row["ID"]
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        rts_geoms[rts_id] = geom

    if not rts_geoms:
        raise RuntimeError("No valid RTS geometries found in shapefile.")

    # Initialize annual sums and counts:
    #   LST_sum[RTS_ID][year]   = sum of daily polygon-mean LST
    #   LST_count[RTS_ID][year] = number of days with valid LST
    lst_sum: Dict[int, Dict[int, float]] = {rid: {} for rid in rts_geoms.keys()}
    lst_count: Dict[int, Dict[int, int]] = {rid: {} for rid in rts_geoms.keys()}

    # ----------------------------------------------------------------
    # Loop over all years, months, and daily LST rasters
    # ----------------------------------------------------------------
    print("Starting daily LST processing...")
    for year, year_path in tqdm(
        list(iter_year_month_dirs(LST_ROOT)),
        desc="Years",
    ):
        month_names = [
            m for m in os.listdir(year_path)
            if os.path.isdir(os.path.join(year_path, m))
        ]
        month_names.sort()

        for month_name in tqdm(
            month_names,
            desc=f"Year {year}",
            leave=False,
        ):
            month_path = os.path.join(year_path, month_name)

            # Iterate over daily GeoTIFF files
            for tif_name in os.listdir(month_path):
                if not tif_name.lower().endswith(".tif"):
                    continue

                tif_path = os.path.join(month_path, tif_name)
                with rasterio.open(tif_path) as src:
                    # For each RTS polygon, compute daily mean LST
                    for rts_id, geom in rts_geoms.items():
                        mean_val = polygon_mean_from_raster(src, geom)
                        if mean_val is None:
                            continue

                        # Update sum
                        if year in lst_sum[rts_id]:
                            lst_sum[rts_id][year] += mean_val
                            lst_count[rts_id][year] += 1
                        else:
                            lst_sum[rts_id][year] = mean_val
                            lst_count[rts_id][year] = 1

    # ----------------------------------------------------------------
    # Compute annual mean LST and write results for each RTS
    # ----------------------------------------------------------------
    print("Writing annual mean LST results...")

    for rts_id in tqdm(rts_geoms.keys(), desc="RTS IDs"):
        sum_dict = lst_sum.get(rts_id, {})
        count_dict = lst_count.get(rts_id, {})

        if not sum_dict or not count_dict:
            print(f"RTS ID {rts_id}: no valid LST data, skip writing.")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{rts_id}_LST_mean.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Year\tLST_mean\n")
            for year in sorted(sum_dict.keys()):
                total = sum_dict[year]
                n = count_dict.get(year, 0)
                if n <= 0:
                    continue
                mean_val = total / float(n)
                f.write(f"{year}\t{mean_val}\n")

    print("Processing completed.")


if __name__ == "__main__":
    main()
