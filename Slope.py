#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Slope distribution of RTS polygons vs full study area.

For each RTS polygon:
1. Read a slope raster.
2. Mask the raster by the RTS polygon and compute the mean slope
   of all intersecting pixels (polygon mean value).
3. Collect polygon-mean slope values for all RTS polygons.
4. Extract all valid slope values in the study area.
5. Plot a histogram (RTS polygon means) and probability density curves
   (RTS vs full area).
6. Print basic statistics and interval-based proportions for RTS
   polygon-mean slopes.
"""

import os
from typing import List, Optional

import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterio.features import geometry_mask

from scipy.stats import gaussian_kde

plt.close("all")

# ================================================================
# USER SETTINGS (EDIT THESE PATHS)
# ================================================================

# SLOPE_TIF:
#   Full path to the slope raster.
#   Required format: single-band GeoTIFF (.tif), slope in degrees.
SLOPE_TIF = r"F:\zyz\DiplomaProject\data\DEM\SRTM\slope.tif"

# RTS_SHP_PATH:
#   Full path to the RTS polygons.
#   Required format: ESRI Shapefile (.shp) with polygon geometries.
#   CRS must be compatible with the slope raster (or convertible to it).
RTS_SHP_PATH = r"F:\zyz\DiplomaProject\data\RTS-QTP\RTS-QTP_BLH_84.shp"

# OUTPUT_DIR:
#   Folder where figures and text outputs will be saved.
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

    Returns the polygon-mean slope in degrees; returns None if no
    valid pixels are found inside the polygon.
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
    slope_values: List[float],
    full_slope_values: List[float],
    filename: str,
) -> None:
    """
    Plot slope histogram (RTS polygon means) + KDE curves (RTS and full area).

    Parameters
    ----------
    slope_values : list of float
        Polygon-mean slopes for RTS polygons (degrees).
    full_slope_values : list of float
        Slopes for all valid pixels in the study area (degrees).
    filename : str
        Output SVG file name, saved under OUTPUT_DIR.
    """
    plt.figure(figsize=(10, 6))

    min_slope = min(np.min(slope_values), np.min(full_slope_values))
    max_slope = max(np.max(slope_values), np.max(full_slope_values))
    bin_edges = np.arange(
        np.floor(min_slope),
        np.ceil(max_slope) + 0.5,
        0.5,
    )

    ax1 = plt.gca()

    # RTS histogram (polygon means, frequency, left axis)
    counts, bins, patches = ax1.hist(
        slope_values,
        bins=bin_edges,
        color="#4e79a7",
        edgecolor="white",
        alpha=0.6,
        label="RTS polygon-mean slope",
    )

    ax1.set_xlabel("Slope (°)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12, color="#4e79a7")
    ax1.tick_params(axis="y", labelcolor="#4e79a7")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_xlim(0, 20)
    ax1.set_xticks(np.arange(0, 21, 5))

    # KDE curves (right axis)
    ax2 = ax1.twinx()
    x_vals = np.linspace(min_slope, max_slope, 500)

    # RTS KDE
    kde_rts = gaussian_kde(slope_values)
    y_rts = kde_rts(x_vals)
    ax2.plot(
        x_vals,
        y_rts,
        color="darkorange",
        linewidth=2,
        label="RTS polygon-mean KDE",
    )

    # Full-area KDE
    kde_full = gaussian_kde(full_slope_values)
    y_full = kde_full(x_vals)
    ax2.plot(
        x_vals,
        y_full,
        color="gray",
        linestyle="--",
        linewidth=2,
        label="Study area KDE",
    )
    ax2.set_ylim(bottom=0)
    ax2.set_yticks(np.arange(0, 0.21, 0.05))
    ax2.set_ylabel("Probability density", fontsize=12, color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Statistics (RTS polygon means)
    stats_text = (
        f"RTS polygons: {len(slope_values)}\n"
        f"Bin width: 0.5°\n"
        f"Max count: {int(np.max(counts))}\n"
        f"Max density (RTS): {np.max(y_rts):.3f}\n"
        f"Slope range (RTS means): "
        f"{min(slope_values):.1f}–{max(slope_values):.1f}°"
    )
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.title("Slope distribution: RTS vs entire DEM", fontsize=14)
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"SVG saved to: {out_path}")


# ================================================================
# EXTRACTION AND DRIVER
# ================================================================

def extract_slopes() -> List[float]:
    """
    Extract polygon-mean slope for RTS polygons and full-area slopes.

    Returns
    -------
    slope_values : list of float
        Polygon-mean slopes for RTS polygons.
    """
    gdf = gpd.read_file(RTS_SHP_PATH)
    slope_values: List[float] = []

    with rasterio.open(SLOPE_TIF) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # Ensure CRS consistency
        if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        # Polygon-mean slope for each RTS
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                mean_val = polygon_mean_from_raster(src, geom)
                if mean_val is not None:
                    slope_values.append(mean_val)
            except Exception as e:
                print(f"ID {row.get('ID', idx)} polygon mean extraction failed: {e}")

        # Full-area slopes
        band = src.read(1)
        if nodata is not None:
            full_slope_values = band[band != nodata]
        else:
            full_slope_values = band
        full_slope_values = full_slope_values[np.isfinite(full_slope_values)]
        full_slope_values = full_slope_values.flatten().tolist()

    if not slope_values:
        raise ValueError("No valid RTS polygon-mean slope values extracted.")

    # Plot distributions
    plot_histogram(
        slope_values=slope_values,
        full_slope_values=full_slope_values,
        filename="slope_histogram_SRTM.svg",
    )

    return slope_values


def main() -> None:
    results = extract_slopes()
    print(
        f"\nProcessing finished. RTS polygon count "
        f"(with valid slope): {len(results)}"
    )
    print(
        f"Slope range (RTS polygon means): "
        f"{min(results):.2f}–{max(results):.2f}°"
    )

    interval_list = [(3, 8), (3, 10), (2, 8), (2, 10)]
    print("\nSlope interval proportions (RTS polygon means):")
    for lower, upper in interval_list:
        count = sum(lower <= v <= upper for v in results)
        percent = count / len(results) * 100.0
        print(
            f"Slope {lower}–{upper}°: count = {count}, "
            f"fraction = {percent:.2f}%"
        )


if __name__ == "__main__":
    main()
