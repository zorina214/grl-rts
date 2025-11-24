#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Conditional Kendall's Tau analysis for RTS copula data.

Input
-----
For each RTS ID, there is a tab-separated file:
    <ID>_copula.txt

All such files are stored under DATA_FOLDER. Each file must contain at least:
    Year    x1    x2    y1    ID

where:
    x1 : first covariate  (e.g., annual mean LST)
    x2 : second covariate (e.g., annual precipitation sum)
    y1 : response variable (e.g., absolute vertical trend)
    ID : RTS polygon identifier

Workflow
--------
1. Read and concatenate all <ID>_copula.txt files into one DataFrame.
2. Conditional Kendall's Tau vs x2 for the pair (x1, y1):
   - Bin x2 into quantiles.
   - Compute Kendall's Tau and p-value in each bin.
   - Plot Tau vs bin center.
3. Conditional Kendall's Tau vs x1 for the pair (x2, y1):
   - Same binning procedure using x1.
   - Compute and plot Tau vs bin center.
   - Find bins with minimum and maximum Kendall's Tau and:
       * Plot x2 vs y1 scatter in those bins.
       * Fit a linear regression y1 ~ x2 and plot the line.
       * Optionally format y-axis as y * 365.25.
"""

import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import kendalltau


# ================================================================
# USER SETTINGS (EDIT THESE PATHS)
# ================================================================

# DATA_FOLDER:
#   Folder containing all <ID>_copula.txt files produced by the
#   LST_mean / Pre_sum / UD merging script.
#   Each file: tab-separated with columns at least:
#       Year    x1    x2    y1    ID
DATA_FOLDER = r"E:\zyz\DiplomaProject\data\check\copula_LSTmean_Psum_GMCP"

# FIG_DIR:
#   Folder where all figures (SVG) will be saved.
FIG_DIR = r"E:\zyz\DiplomaProject\data\check\fig_Helvetica"
os.makedirs(FIG_DIR, exist_ok=True)

# Number of quantile bins for conditional analysis
N_BINS_X2 = 10
N_BINS_X1 = 10

# Whether to scale y1 axis in scatter plots by 365.25
SCALE_Y1_BY_36525 = True


# ================================================================
# DATA LOADING
# ================================================================

def load_all_copula_data(data_folder: str) -> pd.DataFrame:
    """
    Load and concatenate all <ID>_copula.txt files.

    Parameters
    ----------
    data_folder : str
        Folder containing <ID>_copula.txt (tab-separated).

    Returns
    -------
    df : pandas.DataFrame
        Concatenated DataFrame with columns including:
        Year, x1, x2, y1, ID.
    """
    all_frames: List[pd.DataFrame] = []

    for fname in os.listdir(data_folder):
        if not fname.endswith("_copula.txt"):
            continue

        fpath = os.path.join(data_folder, fname)
        df = pd.read_csv(fpath, sep="\t")
        # Drop completely empty columns if any
        df = df.dropna(axis=1, how="all")
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError(f"No *_copula.txt files found in {data_folder}")

    combined_df = pd.concat(all_frames, ignore_index=True)
    return combined_df


# ================================================================
# CONDITIONAL KENDALL'S TAU
# ================================================================

def conditional_kendall(
    df: pd.DataFrame,
    condition_var: str,
    var1: str,
    var2: str,
    n_bins: int,
    fig_prefix: str,
    with_extreme_scatter: bool = False,
) -> Dict:
    """
    Compute conditional Kendall's Tau in quantile bins of a condition variable.

    Optionally, also plot scatter + linear fit for bins with minimum and
    maximum Kendall's Tau.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with columns [condition_var, var1, var2].
    condition_var : str
        Name of the conditioning variable (e.g., "x2" or "x1").
    var1 : str
        First variable in the Kendall's Tau pair.
    var2 : str
        Second variable in the Kendall's Tau pair.
    n_bins : int
        Number of quantile bins.
    fig_prefix : str
        Prefix used for figure file names.
    with_extreme_scatter : bool, default False
        If True, also plot scatter + linear fit for bins corresponding to
        minimum and maximum Kendall's Tau.

    Returns
    -------
    result : dict
        Dictionary with keys:
            "bin_edges" : np.ndarray of length (n_bins + 1)
            "tau"       : list of Kendall's Tau per bin (may contain NaN)
            "p_value"   : list of p-values per bin (may contain NaN)
    """
    # Use only relevant columns and drop rows with NaNs in them
    df_clean = df[[condition_var, var1, var2]].dropna().copy()
    if df_clean.empty:
        raise ValueError("No valid rows after dropping NaNs.")

    # Create quantile-based bins
    # duplicates='drop' avoids errors when some bins would be identical
    df_clean["condition_bin"], bin_edges = pd.qcut(
        df_clean[condition_var],
        n_bins,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    n_effective_bins = len(bin_edges) - 1
    tau_values: List[float] = []
    p_values: List[float] = []
    bin_indices: List[int] = list(range(n_effective_bins))

    # For optional scatter plots, store per-bin data
    bin_data_dict: Dict[int, pd.DataFrame] = {}

    for bin_idx in bin_indices:
        bin_data = df_clean[df_clean["condition_bin"] == bin_idx]
        bin_data_dict[bin_idx] = bin_data

        if len(bin_data) < 3:
            tau_values.append(np.nan)
            p_values.append(np.nan)
            continue

        tau, p = kendalltau(bin_data[var1], bin_data[var2])
        tau_values.append(tau)
        p_values.append(p)

    # Plot Kendall's Tau vs bin center
    bin_centers = [
        0.5 * (bin_edges[i] + bin_edges[i + 1])
        for i in range(n_effective_bins)
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(
        bin_centers,
        tau_values,
        marker="o",
        label=f"{var1}-{var2}",
    )

    # Annotate p-values at each point
    for x, y, p in zip(bin_centers, tau_values, p_values):
        if np.isfinite(y):
            plt.text(
                x,
                y,
                f"p={p:.3f}" if np.isfinite(p) else "p=nan",
                fontsize=9,
                ha="right",
                va="bottom",
            )

    plt.xlabel(f"{condition_var} bin center")
    plt.ylabel("Kendall's Tau")
    plt.title(
        f"Conditional Kendall's Tau of {var1}–{var2} vs {condition_var}"
    )
    plt.grid(True)
    plt.legend()

    line_fig_path = os.path.join(
        FIG_DIR,
        f"{fig_prefix}_tau_vs_{condition_var}.svg",
    )
    plt.savefig(line_fig_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved conditional Kendall line plot: {line_fig_path}")

    result = {
        "bin_edges": bin_edges,
        "tau": tau_values,
        "p_value": p_values,
    }

    # ----------------------------------------------------------------
    # Optional: scatter plots for bins with min and max Kendall's Tau
    # ----------------------------------------------------------------
    if with_extreme_scatter:
        tau_arr = np.array(tau_values, dtype=float)
        valid_mask = np.isfinite(tau_arr)
        if not np.any(valid_mask):
            print("No valid Kendall's Tau values for extreme scatter plots.")
            return result

        valid_indices = np.where(valid_mask)[0]
        # Global index of min and max within the list of bins
        min_global_idx = valid_indices[np.argmin(tau_arr[valid_mask])]
        max_global_idx = valid_indices[np.argmax(tau_arr[valid_mask])]

        # Minimum Kendall's Tau bin
        min_data = bin_data_dict[min_global_idx]
        if not min_data.empty:
            x = min_data[var1].values
            y = min_data[var2].values

            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, color="red", alpha=0.5, label="Min Kendall bin")

            # Linear fit
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = poly(x_fit)
                plt.plot(
                    x_fit,
                    y_fit,
                    color="blue",
                    label=f"Linear fit: y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}",
                )
                print(
                    f"[{fig_prefix}] Min-Ta u bin linear fit "
                    f"(y vs {var1}): y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}"
                )

            if SCALE_Y1_BY_36525 and var2 == "y1":
                plt.gca().yaxis.set_major_formatter(
                    FuncFormatter(lambda val, _: f"{val * 365.25:.0f}")
                )

            plt.xlabel(var1)
            plt.ylabel(var2 if not SCALE_Y1_BY_36525 else f"{var2} (× 365.25)")
            plt.title(
                f"Scatter in bin with minimum Kendall's Tau "
                f"({condition_var} conditional)"
            )
            plt.grid(True)
            plt.legend()

            min_fig_path = os.path.join(
                FIG_DIR,
                f"{fig_prefix}_min_tau_bin_scatter.svg",
            )
            plt.savefig(min_fig_path, format="svg", bbox_inches="tight")
            plt.close()
            print(f"Saved scatter for minimum Kendall bin: {min_fig_path}")

        # Maximum Kendall's Tau bin
        max_data = bin_data_dict[max_global_idx]
        if not max_data.empty:
            x = max_data[var1].values
            y = max_data[var2].values

            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, color="green", alpha=0.5, label="Max Kendall bin")

            # Linear fit
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = poly(x_fit)
                plt.plot(
                    x_fit,
                    y_fit,
                    color="blue",
                    label=f"Linear fit: y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}",
                )
                print(
                    f"[{fig_prefix}] Max-Tau bin linear fit "
                    f"(y vs {var1}): y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}"
                )

            if SCALE_Y1_BY_36525 and var2 == "y1":
                plt.gca().yaxis.set_major_formatter(
                    FuncFormatter(lambda val, _: f"{val * 365.25:.0f}")
                )

            plt.xlabel(var1)
            plt.ylabel(var2 if not SCALE_Y1_BY_36525 else f"{var2} (× 365.25)")
            plt.title(
                f"Scatter in bin with maximum Kendall's Tau "
                f"({condition_var} conditional)"
            )
            plt.grid(True)
            plt.legend()

            max_fig_path = os.path.join(
                FIG_DIR,
                f"{fig_prefix}_max_tau_bin_scatter.svg",
            )
            plt.savefig(max_fig_path, format="svg", bbox_inches="tight")
            plt.close()
            print(f"Saved scatter for maximum Kendall bin: {max_fig_path}")

    return result


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    # Load combined copula data
    df = load_all_copula_data(DATA_FOLDER)
    print(f"Loaded combined copula data with {len(df)} rows.")

    # 1) Conditional Kendall's Tau vs x2 for (x1, y1) – line plot only
    print("Computing conditional Kendall's Tau vs x2 for (x1, y1)...")
    _ = conditional_kendall(
        df=df,
        condition_var="x2",
        var1="x1",
        var2="y1",
        n_bins=N_BINS_X2,
        fig_prefix="cond_kendall_x2_x1y1",
        with_extreme_scatter=False,
    )

    # 2) Conditional Kendall's Tau vs x1 for (x2, y1) – line + extreme scatter
    print("Computing conditional Kendall's Tau vs x1 for (x2, y1) and extreme bins...")
    _ = conditional_kendall(
        df=df,
        condition_var="x1",
        var1="x2",
        var2="y1",
        n_bins=N_BINS_X1,
        fig_prefix="cond_kendall_x1_x2y1",
        with_extreme_scatter=True,
    )


if __name__ == "__main__":
    main()
