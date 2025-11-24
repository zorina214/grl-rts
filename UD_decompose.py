#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Seasonal and trend decomposition of polygon-based UD time series
using cubic interpolation only.

For each RTS polygon:
1. Read UD displacement rasters (GeoTIFF stack).
2. For each raster, mask by the RTS polygon and compute the mean of
   all intersecting pixels (polygon mean value).
3. Build a time series (date, mean UD) for the polygon.
4. Interpolate the irregular time series to a daily grid using cubic
   interpolation.
5. Fit a seasonal + linear trend model:
       y(t) = A * sin(w t + phase) + trend * t + offset
6. Check residuals:
   - mean(|residual|) < 1 mm
   - residuals behave like white noise (ACF within CI for lags 15–90)
7. If the residual test is passed, extract peak and trough in the
   internal full year and save:
   peak_date, peak_value, trough_date, trough_value, amplitude, trend.
"""

# ================================================================
# USER SETTINGS
# ================================================================

import os
from typing import List, Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from datetime import datetime
from statsmodels.tsa.stattools import acf
from tqdm import tqdm
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask

plt.close("all")

# ----------------------------------------------------------------
# INPUT / OUTPUT PATHS (EDIT THESE)
# ----------------------------------------------------------------
# UD_RASTER_FOLDER:
#   Folder containing UD displacement rasters.
#   Required format: multiple single-band GeoTIFF (.tif) files.
#   Each file represents one epoch and the date must be encoded
#   in the file name as YYYYMMDD (e.g., "20160115_UD.tif").
UD_RASTER_FOLDER = r"F:\zyz\DiplomaProject\data\UD_tifs"

# RTS_SHP_PATH:
#   Full path to the RTS polygons.
#   Required format: ESRI Shapefile (.shp) containing polygon or
#   multipolygon geometries. CRS must be compatible with the rasters
#   (or convertible to it).
RTS_SHP_PATH = r"F:\zyz\DiplomaProject\data\RTS-QTP\RTS-QTP_BLH_84.shp"

# OUTPUT_FOLDER:
#   Folder where result text files and figures will be saved.
#   It will be created if it does not exist.
OUTPUT_FOLDER = r"F:\zyz\DiplomaProject\scripts\revised1_fig\UD_cubic_results"
PLOT_FOLDER = os.path.join(OUTPUT_FOLDER, "fig")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# ================================================================
# MODEL AND TIME-RANGE SETTINGS
# ================================================================


def seasonal_fit(
    t: np.ndarray,
    amplitude: float,
    phase: float,
    offset: float,
    trend: float,
) -> np.ndarray:
    """
    Seasonal + linear trend model:
        y(t) = amplitude * sin(w t + phase) + trend * t + offset
    """
    w = 2 * np.pi / 365.25
    return amplitude * np.sin(w * t + phase) + trend * t + offset


def define_date_ranges():
    """
    Define fixed time windows and internal full years.

    Returns
    -------
    list of tuples
        Each tuple: (start_date, end_date, internal_start, internal_end)
    """
    return [
        (datetime(2015, 9, 1), datetime(2017, 4, 30),
         datetime(2016, 1, 1), datetime(2016, 12, 31)),
        (datetime(2016, 9, 1), datetime(2018, 4, 30),
         datetime(2017, 1, 1), datetime(2017, 12, 31)),
        (datetime(2017, 9, 1), datetime(2019, 4, 30),
         datetime(2018, 1, 1), datetime(2018, 12, 31)),
        (datetime(2018, 9, 1), datetime(2020, 4, 30),
         datetime(2019, 1, 1), datetime(2019, 12, 31)),
        (datetime(2019, 9, 1), datetime(2021, 4, 30),
         datetime(2020, 1, 1), datetime(2020, 12, 31)),
        (datetime(2020, 9, 1), datetime(2022, 4, 30),
         datetime(2021, 1, 1), datetime(2021, 12, 31)),
    ]


def is_white_noise(residuals: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Test whether residuals behave like white noise, using ACF.

    Only lags 21–90 are checked; all ACF values in that range must be
    within the (1 - alpha) confidence interval.
    """
    acf_values, confint = acf(residuals, nlags=90, alpha=alpha, fft=False)
    start_lag = 15
    relevant_acf = acf_values[start_lag:]
    relevant_lower = confint[start_lag:, 0]
    relevant_upper = confint[start_lag:, 1]
    in_conf_int = (relevant_lower < relevant_acf) & (relevant_acf < relevant_upper)
    return bool(np.all(in_conf_int))


# ================================================================
# RASTER AND GEOMETRY UTILITIES
# ================================================================


def parse_date_from_filename(filename: str) -> datetime:
    """
    Extract a YYYYMMDD date from a file name.

    The function scans the name for an 8-digit substring and parses it
    as YYYYMMDD. Example: "20170115_UD.tif" -> 2017-01-15.
    """
    digits = "".join(ch if ch.isdigit() else " " for ch in filename)
    parts = digits.split()
    for part in parts:
        if len(part) == 8:
            return datetime.strptime(part, "%Y%m%d")
    raise ValueError(f"Could not find YYYYMMDD date in file name: {filename}")


def list_ud_rasters(folder: str) -> List[str]:
    """
    List and sort UD raster files in a folder.

    Only .tif files are kept. Sorting is lexicographic, which works
    if file names start with the date or contain a sortable date.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    files = [f for f in os.listdir(folder) if f.lower().endswith(".tif")]
    if not files:
        raise ValueError(f"No .tif files found in folder: {folder}")
    files.sort()
    return files


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

    The raster is restricted to a window around the polygon's bounds,
    then a geometry mask is applied, and the mean over valid pixels is
    returned. If no valid pixels are found, returns None.
    """
    window = polygon_window(src, geom)
    if window is None:
        return None

    data = src.read(1, window=window)
    window_transform = src.window_transform(window)

    mask = geometry_mask(
        [geom],
        out_shape=data.shape,
        transform=window_transform,
        invert=True,      # True where geometry intersects
        all_touched=True, # Count any pixel touched by the polygon
    )

    vals = data[mask]
    nodata = src.nodata

    if nodata is not None:
        vals = vals[vals != nodata]

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    return float(np.mean(vals))


# ================================================================
# PLOTTING UTILITIES
# ================================================================


def plot_series_and_components(
    region_id: str,
    start_date: datetime,
    end_date: datetime,
    dates_ord_subset: np.ndarray,
    values_subset: np.ndarray,
    fitted_values: np.ndarray,
    residuals: np.ndarray,
    trend: np.ndarray,
    peak_date: datetime,
    peak_value: float,
    trough_date: datetime,
    trough_value: float,
    plot_folder: str,
) -> None:
    """Plot original, fitted, trend, and residuals for one time window."""
    dates_dt = [datetime.fromordinal(d) for d in dates_ord_subset]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Original and fitted
    axes[0].plot(dates_dt, values_subset, label="Original Data")
    axes[0].plot(dates_dt, fitted_values, label="Fitted Data")
    axes[0].scatter([peak_date], [peak_value], color="r", label="Peak")
    axes[0].scatter([trough_date], [trough_value], color="g", label="Trough")
    axes[0].set_ylabel("Value")
    axes[0].set_title(
        f"Region ID: {region_id}, "
        f"Time Range: {start_date:%Y-%m-%d} – {end_date:%Y-%m-%d}"
    )
    axes[0].legend()

    # Trend
    axes[1].plot(dates_dt, trend, label="Trend")
    axes[1].set_ylabel("Trend")
    axes[1].legend()

    # Residuals
    axes[2].plot(dates_dt, residuals, label="Residuals")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Residual")
    axes[2].legend()

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()

    filename = f"{region_id}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.svg"
    save_path = os.path.join(plot_folder, filename)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_acf_figure(
    residuals: np.ndarray,
    region_id: str,
    start_date: datetime,
    end_date: datetime,
    plot_folder: str,
) -> None:
    """Save ACF plot of residuals for diagnostic purposes."""
    from statsmodels.graphics.tsaplots import plot_acf

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(
        residuals,
        lags=90,
        ax=ax,
        title=f"ACF – {region_id} {start_date:%Y%m%d}-{end_date:%Y%m%d}",
    )
    acf_save_path = os.path.join(
        plot_folder,
        f"ACF_{region_id}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.svg",
    )
    fig.savefig(acf_save_path, bbox_inches="tight", format="svg")
    plt.close(fig)


# ================================================================
# CORE TIME-SERIES PROCESSING
# ================================================================


def process_time_series(
    dates: Sequence[datetime],
    values: Sequence[float],
    date_ranges,
    region_id: str,
    plot_folder: str,
):
    """
    Process a single polygon time series:
    - cubic interpolation to daily grid
    - seasonal + trend fit for each time window
    - residual tests
    - peak/trough extraction in internal full year
    """
    dates_arr = np.array(dates)
    values_arr = np.array(values, dtype=float)

    # Sort by date
    order = np.argsort(dates_arr)
    dates_arr = dates_arr[order]
    values_arr = values_arr[order]

    dates_ord = np.array([d.toordinal() for d in dates_arr], dtype=int)

    if len(dates_ord) < 4:
        print(f"{region_id}: fewer than 4 observations, skip.")
        return [], [], [], [], [], [], []

    # Daily grid for interpolation
    full_ord = np.arange(dates_ord.min(), dates_ord.max() + 1, dtype=int)

    from scipy.interpolate import interp1d

    # Cubic interpolation only
    f_interp = interp1d(
        dates_ord,
        values_arr,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )
    values_interp = f_interp(full_ord)

    t_full = full_ord - full_ord[0]

    all_peak_dates = []
    all_peak_values = []
    all_trough_dates = []
    all_trough_values = []
    all_amplitudes = []
    all_residuals = []
    all_trends = []

    for start_date, end_date, internal_start, internal_end in tqdm(
        date_ranges, desc=f"Processing {region_id}"
    ):
        start_ord = start_date.toordinal()
        end_ord = end_date.toordinal()
        internal_start_ord = internal_start.toordinal()
        internal_end_ord = internal_end.toordinal()

        mask = (full_ord >= start_ord) & (full_ord <= end_ord)
        if not np.any(mask):
            continue

        dates_subset_ord = full_ord[mask]
        t_subset = t_full[mask]
        values_subset = values_interp[mask]

        # Basic data check
        if len(values_subset) < 10 or np.sum(np.isnan(values_subset)) > 10:
            print(
                f"{region_id}: insufficient data in "
                f"{start_date:%Y-%m-%d} – {end_date:%Y-%m-%d}, skip."
            )
            continue

        try:
            # Seasonal + trend fit
            popt, _ = curve_fit(
                seasonal_fit,
                t_subset,
                values_subset,
                bounds=(
                    [-100.0, -np.pi, -100.0, -100.0],
                    [100.0, np.pi, 100.0, 100.0],
                ),
                maxfev=20000,
            )
            fitted_values = seasonal_fit(t_subset, *popt)
            residuals = values_subset - fitted_values

            # Residual tests
            # Mean close to zero: threshold = 1 mm
            mean_close_to_zero = np.abs(np.mean(residuals)) < 1.0
            white_noise = is_white_noise(residuals)

            if not (mean_close_to_zero and white_noise):
                print(
                    f"{region_id}: residual test failed for "
                    f"{start_date:%Y-%m-%d} – {end_date:%Y-%m-%d}, skip."
                )
                continue

            # Save ACF figure
            plot_acf_figure(residuals, region_id, start_date, end_date, plot_folder)

            amplitude = float(popt[0])
            trend_slope = float(popt[3])

        except RuntimeError as e:
            print(
                f"{region_id}: curve fitting failed for "
                f"{start_date:%Y-%m-%d} – {end_date:%Y-%m-%d}: {e}"
            )
            continue

        # Peak and trough in fitted seasonal+trend
        peak_indices = argrelextrema(fitted_values, np.greater)[0]
        trough_indices = argrelextrema(fitted_values, np.less)[0]

        if len(peak_indices) == 0 or len(trough_indices) == 0:
            print(
                f"{region_id}: no extrema in "
                f"{start_date:%Y-%m-%d} – {end_date:%Y-%m-%d}, skip."
            )
            continue

        peak_dates_ord = dates_subset_ord[peak_indices]
        trough_dates_ord = dates_subset_ord[trough_indices]
        peak_values = fitted_values[peak_indices]
        trough_values = fitted_values[trough_indices]

        # Restrict extrema to the internal full year
        internal_peak_idx = np.where(
            (peak_dates_ord >= internal_start_ord) & (peak_dates_ord <= internal_end_ord)
        )[0]
        internal_trough_idx = np.where(
            (trough_dates_ord >= internal_start_ord)
            & (trough_dates_ord <= internal_end_ord)
        )[0]

        if len(internal_peak_idx) == 0 or len(internal_trough_idx) == 0:
            print(
                f"{region_id}: no extrema in internal year "
                f"{internal_start:%Y-%m-%d} – {internal_end:%Y-%m-%d}, skip."
            )
            continue

        max_peak_idx = internal_peak_idx[np.argmax(peak_values[internal_peak_idx])]
        min_trough_idx = internal_trough_idx[np.argmin(trough_values[internal_trough_idx])]

        peak_date = datetime.fromordinal(int(peak_dates_ord[max_peak_idx]))
        trough_date = datetime.fromordinal(int(trough_dates_ord[min_trough_idx]))
        peak_value = float(peak_values[max_peak_idx])
        trough_value = float(trough_values[min_trough_idx])

        print(
            f"Region {region_id}, fit range {start_date:%Y-%m-%d} – {end_date:%Y-%m-%d}"
        )
        print(
            f"Internal year {internal_start:%Y-%m-%d} – {internal_end:%Y-%m-%d}, "
            f"amplitude={amplitude:.2f}, phase={popt[1]:.2f}, "
            f"offset={popt[2]:.2f}, trend={trend_slope:.2f}"
        )
        print(
            f"Peak:   {peak_date:%Y-%m-%d}, value={peak_value:.2f}\n"
            f"Trough: {trough_date:%Y-%m-%d}, value={trough_value:.2f}"
        )
        print("-" * 50)

        trend_component = popt[3] * t_subset + popt[2]

        all_peak_dates.append(peak_date)
        all_peak_values.append(peak_value)
        all_trough_dates.append(trough_date)
        all_trough_values.append(trough_value)
        all_amplitudes.append(amplitude)
        all_residuals.append(residuals)
        all_trends.append(trend_slope)

        # Plot and save figures
        plot_series_and_components(
            region_id=region_id,
            start_date=start_date,
            end_date=end_date,
            dates_ord_subset=dates_subset_ord,
            values_subset=values_subset,
            fitted_values=fitted_values,
            residuals=residuals,
            trend=trend_component,
            peak_date=peak_date,
            peak_value=peak_value,
            trough_date=trough_date,
            trough_value=trough_value,
            plot_folder=plot_folder,
        )

    return (
        all_peak_dates,
        all_peak_values,
        all_trough_dates,
        all_trough_values,
        all_amplitudes,
        all_residuals,
        all_trends,
    )


# ================================================================
# MAIN DRIVER
# ================================================================


def main() -> None:
    # List UD rasters and parse dates from file names
    ud_files = list_ud_rasters(UD_RASTER_FOLDER)
    ud_dates = [parse_date_from_filename(f) for f in ud_files]

    # Read CRS from first raster and align RTS polygons to it
    first_raster_path = os.path.join(UD_RASTER_FOLDER, ud_files[0])
    with rasterio.open(first_raster_path) as src_template:
        raster_crs = src_template.crs

    gdf = gpd.read_file(RTS_SHP_PATH)
    if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    date_ranges = define_date_ranges()

    # Process each RTS polygon
    for idx, row in tqdm(
        gdf.iterrows(), total=len(gdf), desc="Processing RTS polygons"
    ):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Use "ID" field if present; otherwise use index
        region_id = str(row.get("ID", idx))

        ts_dates: List[datetime] = []
        ts_values: List[float] = []

        # Build polygon time series by masking each raster
        for fname, obs_date in zip(ud_files, ud_dates):
            raster_path = os.path.join(UD_RASTER_FOLDER, fname)
            with rasterio.open(raster_path) as src:
                mean_val = polygon_mean_from_raster(src, geom)

            if mean_val is not None:
                ts_dates.append(obs_date)
                ts_values.append(mean_val)

        if len(ts_dates) < 4:
            print(f"{region_id}: fewer than 4 valid observations across rasters, skip.")
            continue

        (
            peak_dates,
            peak_values,
            trough_dates,
            trough_values,
            amplitudes,
            residuals,
            trends,
        ) = process_time_series(
            ts_dates,
            ts_values,
            date_ranges,
            region_id,
            PLOT_FOLDER,
        )

        # Save results if at least one valid window
        if peak_dates:
            output_file = os.path.join(
                OUTPUT_FOLDER,
                f"{region_id}_UD_pt_with_amplitude_trend.txt",
            )
            with open(output_file, "w") as f:
                f.write(
                    "peak_date\tpeak_value\ttrough_date\t"
                    "trough_value\tamplitude\ttrend\n"
                )
                for (
                    peak_date,
                    peak_value,
                    trough_date,
                    trough_value,
                    amplitude,
                    trend,
                ) in zip(
                    peak_dates,
                    peak_values,
                    trough_dates,
                    trough_values,
                    amplitudes,
                    trends,
                ):
                    f.write(
                        f"{peak_date:%Y%m%d}\t{peak_value}\t"
                        f"{trough_date:%Y%m%d}\t{trough_value}\t"
                        f"{amplitude}\t{trend}\n"
                    )

            print(f"Saved file: {output_file}")
            print("=" * 50)


if __name__ == "__main__":
    main()
