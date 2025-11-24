#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract mean coherence values for RTS polygons from ascending and
descending coherence rasters, then plot their distributions.

For each RTS polygon, the raster is masked by the polygon and the mean
of all intersecting pixels is used as the polygon's coherence value.
"""

# ================================================================
# USER SETTINGS
# ================================================================

import os
from typing import List, Sequence

import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import rowcol
from rasterio.features import geometry_mask
from rasterio.windows import Window
from scipy.stats import gaussian_kde
from shapely.geometry.base import BaseGeometry
from typing import List, Sequence, Optional

plt.close("all")

# ----------------------------------------------------------------
# INPUT FILE PATHS (EDIT THESE)
# ----------------------------------------------------------------
# ASC_COH_TIF:
#   Full path to the ascending-track coherence raster.
#   Required format: single-band GeoTIFF (.tif) where pixel values
#   represent coherence (typically between 0 and 1).
ASC_COH_TIF = r"F:\zyz\DiplomaProject\data\Slumps\T143F111_SpatialCoh.tif"

# DSC_COH_TIF:
#   Full path to the descending-track coherence raster.
#   Required format: single-band GeoTIFF (.tif) with coherence values
#   on the same grid / CRS as ASC_COH_TIF.
DSC_COH_TIF = r"F:\zyz\DiplomaProject\data\Slumps\geo_avgSpatialCoh_D.tif"

# RTS_SHP_PATH:
#   Full path to the RTS polygons.
#   Required format: ESRI Shapefile (.shp) containing polygon or
#   multipolygon geometries in a CRS compatible with the rasters.
RTS_SHP_PATH = r"F:\zyz\DiplomaProject\data\RTS-QTP\RTS-QTP_BLH_84.shp"

# OUTPUT_DIR:
#   Folder where output figures will be saved.
#   The folder will be created if it does not exist.
OUTPUT_DIR = r"F:\zyz\DiplomaProject\scripts\revised1_fig"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Histogram bin width for coherence (typically between 0 and 1)
BIN_WIDTH = 0.02

# Threshold for "high" coherence statistics
COH_THRESHOLD = 0.7


# ================================================================
# FUNCTIONS
# ================================================================

def plot_histogram(
    values: Sequence[float],
    filename: str,
    title: str,
    bin_width: float = BIN_WIDTH,
    threshold: float = COH_THRESHOLD,
) -> None:
    """
    Plot and save a histogram and KDE of coherence values.

    Parameters
    ----------
    values : Sequence[float]
        Coherence values (e.g., per-polygon means).
    filename : str
        Output image file name (without directory), saved in OUTPUT_DIR.
    title : str
        Plot title.
    bin_width : float, optional
        Histogram bin width.
    threshold : float, optional
        Threshold to compute percentage of "high" coherence values.
    """
    data = np.asarray(values, dtype="float64")
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError("No valid coherence data available for plotting.")

    min_val = data.min()
    max_val = data.max()

    start = np.floor(min_val / bin_width) * bin_width
    end = np.ceil(max_val / bin_width) * bin_width
    bin_edges = np.arange(start, end + bin_width, bin_width)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Histogram (frequency, left axis)
    counts, bins, patches = ax1.hist(
        data,
        bins=bin_edges,
        color="#4e79a7",
        edgecolor="white",
        alpha=0.6,
        label="Frequency",
    )

    ax1.set_xlabel("Coherence", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12, color="#4e79a7")
    ax1.tick_params(axis="y", labelcolor="#4e79a7")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # KDE curve (probability density, right axis)
    ax2 = ax1.twinx()
    kde = gaussian_kde(data)
    x_vals = np.linspace(start, end, 500)
    y_vals = kde(x_vals)

    ax2.plot(
        x_vals,
        y_vals,
        color="darkorange",
        linewidth=2,
        label="Probability density",
    )
    ax2.set_ylabel("Probability density", fontsize=12, color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    # Percentage of values above threshold
    pct_high = (data > threshold).mean() * 100.0
    print(f"{title}: proportion of coherence > {threshold:.2f} is {pct_high:.2f}%")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Statistics text box
    stats_text = (
        f"Samples: {data.size}\n"
        f"Bin width: {bin_width:.2f}\n"
        f"Max count: {int(max(counts))}\n"
        f"Max density: {y_vals.max():.3f}\n"
        f"Range: {min_val:.2f} – {max_val:.2f}\n"
        f"> {threshold:.2f}: {pct_high:.2f}%"
    )

    ax1.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    ax1.set_title(title, fontsize=14)

    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {out_path}")


def _polygon_window(
    src: rasterio.io.DatasetReader,
    geom: BaseGeometry,
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

    # Upper-left and lower-right corners in raster indices
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


def extract_coherence_values(
    raster_path: str,
    shp_path: str = RTS_SHP_PATH,
) -> List[float]:
    """
    Compute mean coherence per RTS polygon by masking the raster.

    For each polygon, all pixels whose area intersects the polygon are
    selected (using a raster mask), and their mean coherence value is
    used as that polygon's coherence.

    Parameters
    ----------
    raster_path : str
        Path to the coherence raster (single-band GeoTIFF).
    shp_path : str, optional
        Path to the RTS polygon Shapefile.

    Returns
    -------
    values : list of float
        Mean coherence for each polygon that overlaps the raster and
        has at least one valid pixel.
    """
    gdf = gpd.read_file(shp_path)

    values: List[float] = []

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            try:
                window = _polygon_window(src, geom)
                if window is None:
                    continue

                data = src.read(1, window=window)
                window_transform = src.window_transform(window)

                mask = geometry_mask(
                    [geom],
                    out_shape=data.shape,
                    transform=window_transform,
                    invert=True,      # True where geometry intersects
                    all_touched=True  # count any pixel touched by polygon
                )

                masked_values = data[mask]

                if nodata is not None:
                    masked_values = masked_values[masked_values != nodata]

                masked_values = masked_values[np.isfinite(masked_values)]

                if masked_values.size == 0:
                    continue

                mean_val = float(np.mean(masked_values))
                values.append(mean_val)

            except Exception as e:
                feature_id = row.get("ID", idx)
                print(f"Mean coherence extraction failed for feature ID {feature_id}: {e}")

    return values


def main() -> None:
    """
    Main driver function.

    1. Compute per-polygon mean coherence for ascending raster.
    2. Print basic statistics and plot/save histogram + KDE.
    3. Repeat for descending raster.
    """
    # Ascending
    asc_vals = extract_coherence_values(ASC_COH_TIF, RTS_SHP_PATH)
    if asc_vals:
        print(
            f"\nAscending coherence (polygon means): "
            f"samples = {len(asc_vals)}, "
            f"range = {min(asc_vals):.3f} – {max(asc_vals):.3f}"
        )
        plot_histogram(
            asc_vals,
            filename="coherence_hist_ascending.svg",
            title="Ascending-track coherence distribution (polygon means)",
        )
    else:
        print("No valid ascending polygon mean coherence values. No figure generated.")

    # Descending
    dsc_vals = extract_coherence_values(DSC_COH_TIF, RTS_SHP_PATH)
    if dsc_vals:
        print(
            f"\nDescending coherence (polygon means): "
            f"samples = {len(dsc_vals)}, "
            f"range = {min(dsc_vals):.3f} – {max(dsc_vals):.3f}"
        )
        plot_histogram(
            dsc_vals,
            filename="coherence_hist_descending.svg",
            title="Descending-track coherence distribution (polygon means)",
        )
    else:
        print("No valid descending polygon mean coherence values. No figure generated.")


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()
