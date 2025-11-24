#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge annual LST mean, annual precipitation sum, and vertical deformation
trend for each RTS polygon into per-ID copula input files.

For each RTS ID:
1. Read annual mean LST from:
       <ID>_LST_mean.txt
2. Read annual precipitation sum from:
       <ID>_pre_sum.txt
3. Read deformation peak/trough and trend from:
       <ID>_UD_pt_with_amplitude_trend.txt
4. Extract the year from the peak_date field (YYYY) and the absolute
   value of the trend parameter.
5. Join the three tables on Year.
6. Save merged data to:
       <ID>_copula.txt
   with columns:
       Year  x1  x2  y1  ID
   where:
       x1 = LST_mean
       x2 = Pre_sum
       y1 = |trend| (absolute value).
"""

import os
import pandas as pd


# ================================================================
# USER SETTINGS (EDIT THESE PATHS)
# ================================================================

# LST_DIR:
#   Folder containing annual mean LST files produced by the LST script.
#   Required format: one file per RTS ID, named:
#       <ID>_LST_mean.txt
#   File content (tab-separated):
#       Year    LST_mean
LST_DIR = r"E:\zyz\DiplomaProject\data\Slumps\Copula\LST_mean_GMCP"

# PRE_DIR:
#   Folder containing annual precipitation sum files produced by the
#   precipitation script.
#   Required format: one file per RTS ID, named:
#       <ID>_pre_sum.txt
#   File content (tab-separated):
#       Year    Pre_sum
PRE_DIR = r"E:\zyz\DiplomaProject\data\Slumps\Copula\Pre_sum_GMCP"

# UD_DIR:
#   Folder containing deformation peak/trough + trend results.
#   Required format: one file per RTS ID, named:
#       <ID>_UD_pt_with_amplitude_trend.txt
#   File content (tab-separated), at least columns:
#       peak_date   ...   trend
#   where:
#       peak_date = YYYYMMDD or YYYY-MM-DD (year must be the first 4 chars)
#       trend     = vertical trend value (e.g., mm/yr)
UD_DIR = r"E:\zyz\DiplomaProject\data\check\RTSs_UD_peaktrough_txtA_trend_no2f_2016"

# OUTPUT_DIR:
#   Folder where merged copula input files will be written.
#   For each RTS ID, one file will be created:
#       <ID>_copula.txt
OUTPUT_DIR = r"E:\zyz\DiplomaProject\data\check\copula_LSTmean_Psum_GMCP"


# ================================================================
# CORE MERGE FUNCTION
# ================================================================

def load_and_merge_data(
    lst_dir: str,
    pre_dir: str,
    ud_dir: str,
    output_dir: str,
) -> None:
    """
    Traverse three folders, merge data by RTS ID and year, and save one
    copula input file per RTS ID.

    Parameters
    ----------
    lst_dir : str
        Folder containing <ID>_LST_mean.txt files.
    pre_dir : str
        Folder containing <ID>_pre_sum.txt files.
    ud_dir : str
        Folder containing <ID>_UD_pt_with_amplitude_trend.txt files.
    output_dir : str
        Folder where <ID>_copula.txt files will be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all RTS IDs from LST files
    lst_files = [
        f for f in os.listdir(lst_dir)
        if f.endswith("_LST_mean.txt")
    ]
    ids = [f.split("_")[0] for f in lst_files]

    for rts_id in ids:
        # Build input file paths
        lst_path = os.path.join(lst_dir, f"{rts_id}_LST_mean.txt")
        pre_path = os.path.join(pre_dir, f"{rts_id}_pre_sum.txt")
        ud_path = os.path.join(ud_dir, f"{rts_id}_UD_pt_with_amplitude_trend.txt")

        # Ensure all files exist for this RTS ID
        if not (os.path.exists(lst_path) and os.path.exists(pre_path) and os.path.exists(ud_path)):
            print(f"Skipping RTS ID {rts_id}: missing one or more input files.")
            continue

        # Load annual mean LST (x1)
        df_lst = pd.read_csv(lst_path, sep="\t")
        if "Year" not in df_lst.columns or "LST_mean" not in df_lst.columns:
            print(f"RTS ID {rts_id}: invalid LST file format, skip.")
            continue
        df_lst.rename(columns={"LST_mean": "x1"}, inplace=True)

        # Load annual precipitation sum (x2)
        df_pre = pd.read_csv(pre_path, sep="\t")
        if "Year" not in df_pre.columns or "Pre_sum" not in df_pre.columns:
            print(f"RTS ID {rts_id}: invalid Pre_sum file format, skip.")
            continue
        df_pre.rename(columns={"Pre_sum": "x2"}, inplace=True)

        # Load deformation trend (y1)
        df_ud = pd.read_csv(ud_path, sep="\t")
        if "peak_date" not in df_ud.columns or "trend" not in df_ud.columns:
            print(f"RTS ID {rts_id}: invalid UD file format, skip.")
            continue

        # Extract year from peak_date
        df_ud["Year"] = df_ud["peak_date"].astype(str).str[:4].astype(int)
        df_ud.rename(columns={"trend": "y1"}, inplace=True)
        df_ud["y1"] = df_ud["y1"].abs()
        df_ud = df_ud[["Year", "y1"]]

        # Merge on Year: (LST ∩ Precip ∩ UD)
        df_merged = (
            df_lst.merge(df_pre, on="Year", how="inner")
                  .merge(df_ud, on="Year", how="inner")
        )

        if df_merged.empty:
            print(f"RTS ID {rts_id}: no overlapping years between LST, Pre_sum and UD, skip.")
            continue

        # Add RTS ID column
        df_merged["ID"] = rts_id

        # Save merged copula input file
        output_path = os.path.join(output_dir, f"{rts_id}_copula.txt")
        df_merged.to_csv(output_path, sep="\t", index=False)
        print(f"Saved merged copula file: {output_path}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    load_and_merge_data(
        lst_dir=LST_DIR,
        pre_dir=PRE_DIR,
        ud_dir=UD_DIR,
        output_dir=OUTPUT_DIR,
    )
