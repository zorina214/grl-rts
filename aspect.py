#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aspect statistics for RTS polygons.

Steps
-----
1. Read RTS polygons and an aspect raster.
2. For each polygon:
   - Clip the aspect raster to the polygon.
   - Compute the mean aspect of all intersecting pixels (circular mean
     in degrees, 0–360).
   - Classify the mean aspect into 16 directional bins (N, NNE, ..., NNW).
3. Save per-polygon results to a text file.
4. Count the number of samples per directional label and save
   the summary to another text file.
"""

import os
from typing import List, Tuple

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np
import pandas as pd

# ================================================================
# USER SETTINGS (EDIT THESE PATHS)
# ================================================================

# RTS_SHP_PATH:
#   Full path to the RTS polygons.
#   Required format: ESRI Shapefile (.shp) with polygon geometries.
RTS_SHP_PATH = r"F:\zyz\DiplomaProject\data\RTS-QTP\RTS-QTP_BLH.shp"

# ASPECT_TIF_PATH:
#   Full path to the aspect raster.
#   Required format: single-band GeoTIFF (.tif), aspect in degrees
#   from 0 to 360 (e.g., SRTM_DEM aspect).
ASPECT_TIF_PATH = r"F:\zyz\DiplomaProject\data\DEM\SRTM\SRTM_DEM30_aspect.tif"

# OUTPUT_DIR:
#   Folder where per-polygon aspect results and label counts will be saved.
OUTPUT_DIR = r"F:\zyz\DiplomaProject\scripts\revised1_fig"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file names
ASPECT_SAMPLES_TXT = os.path.join(OUTPUT_DIR, "aspect_SRTM.txt")
ASPECT_COUNTS_TXT = os.path.join(OUTPUT_DIR, "aspect_label_counts_SRTM.txt")


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def circular_mean_deg(values_deg: np.ndarray) -> float:
    """
    Compute the circular mean of aspect values in degrees (0–360).

    Parameters
    ----------
    values_deg : np.ndarray
        Aspect values in degrees.

    Returns
    -------
    mean_deg : float
        Circular mean aspect in degrees, mapped to [0, 360).
    """
    if values_deg.size == 0:
        return np.nan

    rad = np.deg2rad(values_deg)
    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))

    if np.isclose(sin_mean, 0.0) and np.isclose(cos_mean, 0.0):
        # Undefined mean direction (e.g., uniform distribution);
        # return NaN to signal no dominant direction.
        return np.nan

    mean_rad = np.arctan2(sin_mean, cos_mean)
    mean_deg = np.rad2deg(mean_rad)
    if mean_deg < 0.0:
        mean_deg += 360.0
    return mean_deg


def classify_aspect_direction(aspect_deg: float) -> Tuple[int, str]:
    """
    Classify aspect (degrees) into 16 directional bins and labels.

    Parameters
    ----------
    aspect_deg : float
        Aspect in degrees, expected in [0, 360).

    Returns
    -------
    aspect_label : int
        Direction index from 1 to 16.
    direction : str
        Direction code (e.g., "N", "NNE", "NE", ..., "NNW").
    """
    adjusted = (aspect_deg + 11.25) % 360.0
    index = int(adjusted // 22.5)  # 0–15

    direction_labels = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW",
    ]

    aspect_label = index + 1  # 1–16
    return aspect_label, direction_labels[index]


# ================================================================
# CORE COMPUTATION
# ================================================================

def compute_polygon_aspects() -> pd.DataFrame:
    """
    Compute mean aspect and direction class for each RTS polygon.

    For each polygon, the aspect raster is clipped and a circular mean
    is computed from all valid pixels inside the polygon.

    Returns
    -------
    df : pandas.DataFrame
        Per-polygon aspect statistics with columns:
        ['ID', 'Longitude', 'Latitude',
         'Average_Aspect', 'Aspect_Label', 'New_Label'].
    """
    gdf = gpd.read_file(RTS_SHP_PATH)

    with rasterio.open(ASPECT_TIF_PATH) as src:
        aspect_crs = src.crs
        nodata = src.nodata

        # Ensure CRS match
        if gdf.crs is not None and aspect_crs is not None and gdf.crs != aspect_crs:
            gdf = gdf.to_crs(aspect_crs)

        ids: List[int] = []
        centroids_x: List[float] = []
        centroids_y: List[float] = []
        avg_aspects: List[float] = []
        aspect_labels: List[int] = []
        new_labels: List[str] = []

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            poly_id = row.get("ID", idx)
            centroid = geom.centroid
            cx, cy = centroid.x, centroid.y

            try:
                # Clip raster to polygon
                clipped, out_transform = mask(
                    src,
                    [mapping(geom)],
                    crop=True,
                )
                data = clipped[0]  # first band

                if nodata is not None:
                    valid = data[data != nodata]
                else:
                    valid = data

                valid = valid[np.isfinite(valid)]

                if valid.size > 0:
                    avg_aspect = circular_mean_deg(valid)
                else:
                    avg_aspect = np.nan

            except Exception as e:
                print(f"Error processing ID={poly_id}: {e}")
                avg_aspect = np.nan

            if not np.isnan(avg_aspect):
                label_int, label_dir = classify_aspect_direction(avg_aspect)
            else:
                label_int = None
                label_dir = np.nan

            ids.append(poly_id)
            centroids_x.append(cx)
            centroids_y.append(cy)
            avg_aspects.append(avg_aspect)
            aspect_labels.append(label_int)
            new_labels.append(label_dir)

    df = pd.DataFrame(
        {
            "ID": ids,
            "Longitude": centroids_x,
            "Latitude": centroids_y,
            "Average_Aspect": avg_aspects,
            "Aspect_Label": aspect_labels,
            "New_Label": new_labels,
        }
    )

    # Drop polygons with no valid aspect mean
    df = df.dropna(subset=["Average_Aspect"])
    return df


def save_aspect_samples(df: pd.DataFrame, output_path: str) -> None:
    """
    Save per-polygon aspect samples to a tab-separated text file.

    Columns:
        ID, Longitude, Latitude, Average_Aspect, Aspect_Label, New_Label
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "ID\tLongitude\tLatitude\tAverage_Aspect\t"
            "Aspect_Label\tNew_Label\n"
        )
        for _, row in df.iterrows():
            f.write(
                f"{row['ID']}\t{row['Longitude']}\t{row['Latitude']}\t"
                f"{row['Average_Aspect']}\t{row['Aspect_Label']}\t"
                f"{row['New_Label']}\n"
            )
    print(f"Aspect samples saved to: {output_path}")


def count_labels(input_path: str, output_path: str) -> None:
    """
    Count the number of RTS polygons in each aspect direction class.

    Parameters
    ----------
    input_path : str
        Path to the per-polygon aspect txt file (tab-separated).
    output_path : str
        Path to the output txt file for label counts.
    """
    df = pd.read_csv(input_path, sep="\t")
    label_counts = df["New_Label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]

    print("\nAspect label counts:")
    print(label_counts)

    label_counts.to_csv(output_path, sep="\t", index=False)
    print(f"Label counts saved to: {output_path}")


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    # 1) Compute per-polygon mean aspect and save samples
    df_aspect = compute_polygon_aspects()
    save_aspect_samples(df_aspect, ASPECT_SAMPLES_TXT)

    # 2) Count aspect direction labels and save summary
    count_labels(ASPECT_SAMPLES_TXT, ASPECT_COUNTS_TXT)


if __name__ == "__main__":
    main()
