#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute annual lag between temperature peaks and deformation troughs
for ALL RTS time series in a folder.

Workflow
--------
1. Load a single daily temperature time series from TEMP_TS_FILE.
2. Read all RTS IDs from shapefiles in RTS_ID_SHP_FOLDER
   (e.g., 166.shp -> RTS_ID = "166").
3. For each RTS_ID:
   - Find the corresponding deformation time-series txt file in
     UD_TS_FOLDER (file name must start with the RTS_ID).
   - Load the deformation time series.
   - For each fitting window in DATE_RANGES:
       * Detrend with a linear polynomial.
       * Fit a sinusoid + linear trend model:
           y_detrended(t) ~ A * sin(2πt/365.25 + phase) + trend * t + offset
       * Store the fitted seasonal component.
   - For each year in YEARS_TO_ANALYZE:
       * In [1 Jan, 31 Dec] of that year, find:
           - the first local maximum (peak) of the fitted temperature curve,
           - the first local minimum (trough) of the fitted deformation curve.
       * If both exist, compute lag_days = trough_date - peak_date (in days).
       * Save one record per year.
4. Write all lag records for all RTS IDs into a single, tab-separated
   output file:

       RTS_ID  Year  Temp_Peak_Date  Deform_Trough_Date  Lag_Days
"""

import os
import glob
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from datetime import datetime


# ================================================================
# USER SETTINGS (EDIT THESE PATHS / PARAMETERS)
# ================================================================

# TEMP_TS_FILE:
#   Full path to the daily temperature time-series file.
#   Required format: text file (.txt) with:
#       - one header line
#       - tab-separated columns:
#           time    value
#       where:
#           time  = "YYYY-MM-DD"
#           value = float (temperature)
TEMP_TS_FILE = r"F:\zyz\DiplomaProject\data\LST\daily_temperature_time_series.txt"

# UD_TS_FOLDER:
#   Folder containing deformation time-series files for ALL RTS.
#   Required format: multiple text files (.txt) with:
#       - one header line
#       - tab-separated columns:
#           time    value
#       where:
#           time  = "YYYY-MM-DD"
#           value = float (UD displacement)
#   IMPORTANT:
#       File names must start with the RTS ID, e.g.:
#           166_UD_ts.txt
#           426_UD_ts.txt
UD_TS_FOLDER = r"F:\zyz\DiplomaProject\data\RTS-QTP\RTSs_UD_txt"

# RTS_ID_SHP_FOLDER:
#   Folder containing per-RTS shapefiles, one shapefile per RTS ID.
#   Required format: ESRI Shapefile (.shp), file name is the RTS ID.
#   Example:
#       166.shp, 426.shp, ...
RTS_ID_SHP_FOLDER = r"F:\zyz\DiplomaProject\data\RTS-QTP\RTS_QTP_84_ID"

# OUTPUT_LAG_FILE:
#   Full path to the combined lag result file (tab-separated).
#   If None, it will be created in UD_TS_FOLDER with name:
#       "RTS_all_lag_dates.txt"
OUTPUT_LAG_FILE: Optional[str] = None

# YEARS_TO_ANALYZE:
#   List of years for which lag will be computed.
YEARS_TO_ANALYZE = [2017, 2018, 2019, 2020, 2021]

# DATE_RANGES:
#   Fitting windows used to estimate seasonal components.
#   Each tuple is (start_date, end_date) in datetime format.
DATE_RANGES = [
    (datetime(2017, 1, 1), datetime(2018, 3, 1)),
    (datetime(2017, 11, 1), datetime(2019, 3, 1)),
    (datetime(2018, 11, 1), datetime(2020, 3, 1)),
    (datetime(2019, 11, 1), datetime(2021, 3, 1)),
    (datetime(2021, 11, 1), datetime(2022, 3, 1)),
]

# Minimum number of points required in a fitting window
MIN_POINTS = 10


# ================================================================
# MODEL DEFINITIONS
# ================================================================

def seasonal_sin_with_trend(
    t: np.ndarray,
    amplitude: float,
    phase: float,
    trend: float,
    offset: float,
) -> np.ndarray:
    """
    Sinusoid + linear trend model.

    Parameters
    ----------
    t : np.ndarray
        Time in days (relative, e.g., days since first observation).
    amplitude : float
        Sinusoidal amplitude.
    phase : float
        Phase shift in radians.
    trend : float
        Linear term coefficient.
    offset : float
        Constant term.

    Returns
    -------
    y : np.ndarray
        Model values at times t.
    """
    w = 2.0 * np.pi / 365.25
    return amplitude * np.sin(w * t + phase) + trend * t + offset


# ================================================================
# I/O HELPERS
# ================================================================

def load_time_series(file_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load a time series from a tab-separated text file.

    Assumes a single header line and two columns: time, value.

    Parameters
    ----------
    file_path : str
        Path to the time-series file.

    Returns
    -------
    dates_ord : np.ndarray
        Ordinal dates (days since fixed origin).
    values : np.ndarray
        Series values as float.
    t0_ordinal : int
        Ordinal date of the first observation (used as reference).
    """
    data = np.loadtxt(
        file_path,
        delimiter="\t",
        skiprows=1,
        dtype={"names": ("time", "value"), "formats": ("U10", "f8")},
    )

    dates_str = data["time"]
    values = data["value"].astype(float)

    dates_ord = np.array(
        [datetime.strptime(d, "%Y-%m-%d").toordinal() for d in dates_str],
        dtype=int,
    )

    t0_ordinal = int(dates_ord[0])
    return dates_ord, values, t0_ordinal


def list_rts_ids_from_shapefiles(folder: str) -> List[str]:
    """
    List RTS IDs from shapefiles in a folder.

    Each shapefile name is assumed to be "<ID>.shp".

    Parameters
    ----------
    folder : str
        Folder containing *.shp files.

    Returns
    -------
    ids : list of str
        RTS IDs derived from shapefile base names.
    """
    pattern = os.path.join(folder, "*.shp")
    files = glob.glob(pattern)
    ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    ids = sorted(set(ids))
    return ids


def find_ud_file_for_rts(rts_id: str, folder: str) -> Optional[str]:
    """
    Find a deformation time-series file for a given RTS ID.

    The function searches for *.txt files whose file name starts with
    the RTS ID (case-sensitive).

    Parameters
    ----------
    rts_id : str
        RTS ID (e.g., "166").
    folder : str
        Folder containing UD time-series txt files.

    Returns
    -------
    full_path : str or None
        Full path to the first matching file, or None if not found.
    """
    rts_id_str = str(rts_id)
    candidates = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".txt"):
            continue
        if fname.startswith(rts_id_str):
            candidates.append(fname)

    if not candidates:
        return None

    candidates.sort()
    return os.path.join(folder, candidates[0])


def resolve_output_path() -> str:
    """
    Determine the output path for the combined lag results file.
    """
    if OUTPUT_LAG_FILE:
        return OUTPUT_LAG_FILE

    folder = UD_TS_FOLDER or "."
    return os.path.join(folder, "RTS_all_lag_dates.txt")


# ================================================================
# FITTING AND LAG COMPUTATION
# ================================================================

def fit_seasonal_component(
    dates_ord: np.ndarray,
    values: np.ndarray,
    t0_ordinal: int,
) -> List[Dict]:
    """
    Fit seasonal + trend component for each fitting window.

    For each (start_date, end_date) in DATE_RANGES:
    1. Subset the original series.
    2. Detrend with a simple linear polynomial.
    3. Fit seasonal_sin_with_trend to the detrended series.
    4. Store the fitted seasonal component for later peak/trough search.

    Parameters
    ----------
    dates_ord : np.ndarray
        Ordinal dates of the series.
    values : np.ndarray
        Series values (temperature or deformation).
    t0_ordinal : int
        Ordinal of the first observation in the series.

    Returns
    -------
    results : list of dict
        Each dict has:
            - "start_date", "end_date" : datetime
            - "t_subset" : np.ndarray (time offsets used in fitting)
            - "fitted_values" : np.ndarray (fitted seasonal component)
    """
    t_all = dates_ord - t0_ordinal
    results: List[Dict] = []

    for start_date, end_date in DATE_RANGES:
        start_ord = start_date.toordinal()
        end_ord = end_date.toordinal()

        mask = (dates_ord >= start_ord) & (dates_ord <= end_ord)
        if not np.any(mask):
            continue

        t_subset = t_all[mask]
        values_subset = values[mask]

        if len(values_subset) <= MIN_POINTS:
            continue

        # Linear detrend
        p = np.polyfit(t_subset, values_subset, 1)
        detrended = values_subset - np.polyval(p, t_subset)

        try:
            popt, _ = curve_fit(
                seasonal_sin_with_trend,
                t_subset,
                detrended,
                bounds=(
                    [-100.0, -np.pi, -100.0, -100.0],
                    [100.0, np.pi, 100.0, 100.0],
                ),
                maxfev=20000,
            )
            fitted = seasonal_sin_with_trend(t_subset, *popt)

            results.append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "t_subset": t_subset,
                    "fitted_values": fitted,
                }
            )
        except RuntimeError as e:
            print(
                f"Fit failed for window {start_date:%Y-%m-%d} – "
                f"{end_date:%Y-%m-%d}: {e}"
            )
            continue

    return results


def convert_offsets_to_dates(
    t_subset: np.ndarray,
    t0_ordinal: int,
) -> List[datetime]:
    """
    Convert time offsets in days back to datetime objects.

    Parameters
    ----------
    t_subset : np.ndarray
        Time offsets in days (relative to t0_ordinal).
    t0_ordinal : int
        Reference ordinal date.

    Returns
    -------
    dates : list of datetime
        Corresponding datetime objects.
    """
    return [datetime.fromordinal(int(t0_ordinal + t)) for t in t_subset]


def find_annual_lags_for_rts(
    temp_results: List[Dict],
    deform_results: List[Dict],
    t0_temp: int,
    t0_deform: int,
) -> List[Dict]:
    """
    For each year in YEARS_TO_ANALYZE, compute the lag between
    the first temperature peak and the first deformation trough.

    The function zips temp_results and deform_results, assuming they
    correspond to the same DATE_RANGES (some windows may be missing
    if data coverage is insufficient).

    Parameters
    ----------
    temp_results : list of dict
        Seasonal-fit results for temperature.
    deform_results : list of dict
        Seasonal-fit results for deformation (for one RTS).
    t0_temp : int
        Reference ordinal date for temperature series.
    t0_deform : int
        Reference ordinal date for deformation series.

    Returns
    -------
    lag_records : list of dict
        Each dict contains:
            - "year"
            - "peak_date_temp"       (datetime)
            - "trough_date_deform"   (datetime)
            - "lag_days"             (int)
    """
    lag_records: List[Dict] = []

    for year in YEARS_TO_ANALYZE:
        year_start_ord = datetime(year, 1, 1).toordinal()
        year_end_ord = datetime(year, 12, 31).toordinal()

        lag_for_year_found = False

        for res_temp, res_deform in zip(temp_results, deform_results):
            dates_temp = convert_offsets_to_dates(
                res_temp["t_subset"],
                t0_temp,
            )
            dates_deform = convert_offsets_to_dates(
                res_deform["t_subset"],
                t0_deform,
            )

            idx_temp_year = [
                i
                for i, d in enumerate(dates_temp)
                if year_start_ord <= d.toordinal() <= year_end_ord
            ]
            idx_deform_year = [
                i
                for i, d in enumerate(dates_deform)
                if year_start_ord <= d.toordinal() <= year_end_ord
            ]

            if not idx_temp_year or not idx_deform_year:
                continue

            year_dates_temp = [dates_temp[i] for i in idx_temp_year]
            year_vals_temp = np.array(
                [res_temp["fitted_values"][i] for i in idx_temp_year]
            )

            year_dates_deform = [dates_deform[i] for i in idx_deform_year]
            year_vals_deform = np.array(
                [res_deform["fitted_values"][i] for i in idx_deform_year]
            )

            if year_vals_temp.size == 0 or year_vals_deform.size == 0:
                continue

            # Temperature peaks (local maxima)
            peak_idx = argrelextrema(year_vals_temp, np.greater)[0]
            # Deformation troughs (local minima)
            trough_idx = argrelextrema(year_vals_deform, np.less)[0]

            if peak_idx.size == 0 or trough_idx.size == 0:
                continue

            peak_date_temp = year_dates_temp[int(peak_idx[0])]
            trough_date_deform = year_dates_deform[int(trough_idx[0])]
            lag_days = (trough_date_deform - peak_date_temp).days

            lag_records.append(
                {
                    "year": year,
                    "peak_date_temp": peak_date_temp,
                    "trough_date_deform": trough_date_deform,
                    "lag_days": lag_days,
                }
            )
            lag_for_year_found = True
            break

        if not lag_for_year_found:
            print(f"No valid lag found for year {year} for this RTS.")

    return lag_records


# ================================================================
# SAVE RESULTS
# ================================================================

def save_all_lag_results(
    all_records: List[Dict],
    output_path: str,
) -> None:
    """
    Save lag results for all RTS IDs to a tab-separated text file.

    Columns:
        RTS_ID  Year  Temp_Peak_Date  Deform_Trough_Date  Lag_Days
    """
    all_records_sorted = sorted(
        all_records,
        key=lambda r: (str(r["rts_id"]), int(r["year"])),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "RTS_ID\tYear\tTemp_Peak_Date\tDeform_Trough_Date\tLag_Days\n"
        )
        for rec in all_records_sorted:
            f.write(
                f"{rec['rts_id']}\t"
                f"{rec['year']}\t"
                f"{rec['peak_date_temp']:%Y-%m-%d}\t"
                f"{rec['trough_date_deform']:%Y-%m-%d}\t"
                f"{rec['lag_days']}\n"
            )

    print(f"\nAll lag results saved to: {output_path}")


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    # 1) Temperature series and seasonal fit (once)
    print("Loading temperature time series...")
    dates_temp, values_temp, t0_temp = load_time_series(TEMP_TS_FILE)
    print("Fitting temperature seasonal component (all windows)...")
    temp_results = fit_seasonal_component(
        dates_ord=dates_temp,
        values=values_temp,
        t0_ordinal=t0_temp,
    )
    if not temp_results:
        raise RuntimeError("No valid fitting windows for temperature series.")

    # 2) RTS IDs from shapefiles
    print("\nListing RTS IDs from shapefiles...")
    rts_ids = list_rts_ids_from_shapefiles(RTS_ID_SHP_FOLDER)
    if not rts_ids:
        raise RuntimeError("No shapefiles found in RTS_ID_SHP_FOLDER.")

    print(f"Found {len(rts_ids)} RTS IDs.")

    all_records: List[Dict] = []

    # 3) Loop over RTS IDs
    for rts_id in rts_ids:
        print(f"\nProcessing RTS ID: {rts_id}")
        ud_path = find_ud_file_for_rts(rts_id, UD_TS_FOLDER)
        if ud_path is None:
            print(f"  No UD time-series txt found for RTS {rts_id}, skip.")
            continue

        try:
            dates_deform, values_deform, t0_deform = load_time_series(ud_path)
        except Exception as e:
            print(f"  Failed to load deformation series for {rts_id}: {e}")
            continue

        deform_results = fit_seasonal_component(
            dates_ord=dates_deform,
            values=values_deform,
            t0_ordinal=t0_deform,
        )
        if not deform_results:
            print(
                f"  No valid fitting windows for deformation (RTS {rts_id}), skip."
            )
            continue

        lag_records = find_annual_lags_for_rts(
            temp_results=temp_results,
            deform_results=deform_results,
            t0_temp=t0_temp,
            t0_deform=t0_deform,
        )
        if not lag_records:
            print(f"  No lag records computed for RTS {rts_id}.")
            continue

        for rec in lag_records:
            rec_with_id = rec.copy()
            rec_with_id["rts_id"] = rts_id
            all_records.append(rec_with_id)

    if not all_records:
        print("\nNo lag records were produced for any RTS.")
        return

    # 4) Save combined results
    output_path = resolve_output_path()
    save_all_lag_results(all_records, output_path)


if __name__ == "__main__":
    main()
