#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Elevation distribution of RTS polygons vs entire DEM.

For each RTS polygon:
1. Read a DEM raster.
2. Mask the raster by the RTS polygon and compute the mean elevation
   of all intersecting pixels (polygon mean value).
3. Collect polygon-mean elevation values for all RTS polygons.
4. Extract all valid DEM pixels in the study area.
5. Plot a histogram (RTS polygon means) and probability density curves
   (RTS vs full DEM).
6. Print basic statistics and interval-based proportions for RTS
   polygon-mean elevations.
"""

import os
import math
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window
from rasterio.features import geometry_mask

plt.close("all")

# ================================================================
# USER SETTINGS (EDIT THESE PATHS)
# ================================================================

# DEM_TIF:
#   Full path to the DEM raster.
#   Required format: single-band GeoTIFF (.tif), elevation in meters.
DEM_TIF = r"F:\zyz\DiplomaProject\data\DEM\SRTM\dem_blh_84.tif"

# RTS_SHP_PATH:
#   Full path to the RTS polygons.
#   Required format: ESRI Shapefile (.shp) with polygon geometries.
#   CRS must be compatible with the DEM (or convertible to it).
RTS_SHP_PATH = r"F:\zyz\DiplomaProject\data\RTS-QTP\RTS-QTP_BLH_84.shp"

# OUTPUT_DIR:
#   Folder where figures and text outputs will be saved.
#   The folder will be created if it does not exist.
OUTPUT_DIR = r"F:\zyz\DiplomaProject\scripts\revised1_fig"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# RASTER–POLYGON UTILITIES
# ================================================================

def polygon_window(
    src: rasterio.io.DatasetReader,
    geom,
) -> Optional[Window]:
    """
    Compute a minimal raster window covering a polygon's bounding box.

    Returns
    -------
    window : rasterio.windows.Window or None
        Window covering the polygon's bounds, clipped to the raster.
        Returns None if the polygon does not intersect the raster grid.
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
    geom,
) -> Optional[float]:
    """
    Compute the mean value of all pixels whose area intersects a polygon.

    Steps
    -----
    1. Restrict raster to a window around the polygon's bounding box.
    2. Build a mask where pixels intersect the polygon geometry.
    3. Extract valid pixel values and compute their mean.

    Returns
    -------
    mean_val : float or None
        Polygon-mean value. None if no valid pixel is found.
    """
    window = polygon_window(src, geom)
    if window is None:
        return None

    data = src.read(1, window=window)
    window_transform = src.window_transform(window)
    nodata = src.nodata

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


# ================================================================
# PLOTTING
# ================================================================

def plot_histogram(
    elevation_values: List[float],
    dem_values: List[float],
    filename: str,
) -> None:
    """
    Plot elevation histogram (RTS polygon means) + KDE curves (RTS and full DEM).

    Parameters
    ----------
    elevation_values : list of float
        Polygon-mean elevations for RTS polygons (meters).
    dem_values : list of float
        All valid DEM pixel values (meters).
    filename : str
        Output SVG file name, saved under OUTPUT_DIR.
    """
    plt.figure(figsize=(10, 6))

    bin_width = 10.0
    min_val = min(np.min(elevation_values), np.min(dem_values))
    max_val = max(np.max(elevation_values), np.max(dem_values))

    min_edge = math.floor(min_val / bin_width) * bin_width
    max_edge = math.ceil(max_val / bin_width) * bin_width
    if max_edge == max_val:
        max_edge += bin_width
    bin_edges = np.arange(min_edge, max_edge + bin_width, bin_width)

    ax1 = plt.gca()

    # RTS histogram (polygon means, frequency, left axis)
    counts, bins, patches = ax1.hist(
        elevation_values,
        bins=bin_edges,
        color="mediumpurple",
        edgecolor="black",
        alpha=0.7,
        label="RTS polygon-mean elevation (10 m bins)",
    )
    for patch in patches:
        patch.set_snap(False)

    ax1.set_xlabel("Elevation (m)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12, color="mediumpurple")
    ax1.tick_params(axis="y", labelcolor="mediumpurple")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Optional: fixed x-axis range and ticks (adjust to your DEM)
    ax1.set_xlim(4600, 5000)
    ax1.set_xticks(np.arange(4600, 5001, 100))

    # KDE curves (right axis)
    ax2 = ax1.twinx()
    x_vals = np.linspace(min_val, max_val, 500)

    # RTS KDE (polygon means)
    kde_rts = gaussian_kde(elevation_values)
    y_rts = kde_rts(x_vals)
    ax2.plot(
        x_vals,
        y_rts,
        color="darkorange",
        linewidth=2,
        label="RTS polygon-mean KDE",
    )

    # DEM KDE (full DEM)
    kde_dem = gaussian_kde(dem_values)
    y_dem = kde_dem(x_vals)
    ax2.plot(
        x_vals,
        y_dem,
        color="gray",
        linestyle="--",
        linewidth=2,
        label="Full DEM KDE",
    )
    ax2.set_ylim(bottom=0)
    ax2.set_yticks(np.arange(0, 0.0061, 0.002))
    ax2.set_ylabel("Probability density", fontsize=12, color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Statistics (RTS polygon means)
    stats_text = (
        f"RTS polygons: {len(elevation_values)}\n"
        f"Bin width: 10 m\n"
        f"Max count: {int(np.max(counts))}\n"
        f"Max density (RTS): {np.max(y_rts):.4f}\n"
        f"Elevation range (RTS means): "
        f"{min(elevation_values):.1f}–{max(elevation_values):.1f} m"
    )
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=10,
    )

    plt.title("Elevation distribution: RTS vs entire DEM", fontsize=14)
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"SVG saved to: {out_path}")


# ================================================================
# EXTRACTION AND DRIVER
# ================================================================

def extract_elevations() -> Tuple[List[float], List[float]]:
    """
    Extract polygon-mean elevation for RTS polygons and full DEM values.

    Returns
    -------
    elevation_values : list of float
        Polygon-mean elevations for RTS polygons.
    dem_values : list of float
        All valid DEM pixel values in the raster.
    """
    gdf = gpd.read_file(RTS_SHP_PATH)
    elevation_values: List[float] = []

    with rasterio.open(DEM_TIF) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # Ensure CRS consistency
        if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        # Polygon-mean elevation for each RTS
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                mean_val = polygon_mean_from_raster(src, geom)
                if mean_val is not None:
                    elevation_values.append(mean_val)
            except Exception as e:
                print(f"ID {row.get('ID', idx)} polygon mean extraction failed: {e}")

        # Full DEM values
        band = src.read(1)
        if nodata is not None:
            dem_values = band[band != nodata]
        else:
            dem_values = band
        dem_values = dem_values[np.isfinite(dem_values)].flatten().tolist()

    if not elevation_values:
        raise ValueError("No valid RTS polygon-mean elevation values extracted.")

    return elevation_values, dem_values


def main() -> None:
    elevation_data, dem_data = extract_elevations()

    plot_histogram(
        elevation_values=elevation_data,
        dem_values=dem_data,
        filename="elevation_histogram_SRTM.svg",
    )

    print(f"\nRTS polygon count (with valid elevation): {len(elevation_data)}")
    print(
        f"Elevation range (RTS polygon means): "
        f"{min(elevation_data):.2f}–{max(elevation_data):.2f} m"
    )
    print(f"DEM valid pixel count: {len(dem_data)}")
    print(
        f"Elevation range (DEM): "
        f"{min(dem_data):.2f}–{max(dem_data):.2f} m"
    )

    # Interval-based proportion analysis for RTS polygon-mean elevations
    interval_list = [(4600, 4800), (4700, 4800), (4600, 4700), (4600, 4900)]
    print("\nElevation interval proportions (RTS polygon means):")
    for lower, upper in interval_list:
        count = sum(lower <= v <= upper for v in elevation_data)
        percent = count / len(elevation_data) * 100.0
        print(
            f"Elevation {lower}–{upper} m: count = {count}, "
            f"fraction = {percent:.2f}%"
        )


if __name__ == "__main__":
    main()
