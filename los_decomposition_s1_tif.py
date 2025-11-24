#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decompose Sentinel-1 ascending and descending LOS displacement
time series (stored as GeoTIFF files) into vertical (Up-Down)
and East-West components.

Main features
-------------
1. Reads LOS displacement time series from two folders:
   one for ascending track and one for descending track.
2. Uses GeoTIFF (.tif) files as input, with one band per file.
3. Automatically checks and aligns the time series using file names.
4. Decomposes LOS displacement into East and Up components.
5. Saves per-epoch East and Up components as GeoTIFF files,
   and optionally saves full time series stacks as NumPy arrays.

IMPORTANT
---------
- This script uses the 'rasterio' package to read and write GeoTIFF files.
  Please install it before running:
      pip install rasterio
- The script assumes that:
    * Each LOS file is a single-band GeoTIFF (.tif).
    * All ascending and descending files have the same:
        - number of rows and columns,
        - georeferencing (transform),
        - coordinate reference system (CRS).
    * Each file represents one epoch in the time series.
- Time series alignment is based on file names:
  ascending and descending folders must contain files with identical names
  (order does not matter, they will be sorted).

"""

# ================================================================
# USER SETTINGS  (PLEASE MODIFY THIS SECTION)
# ================================================================

# 1) Input folders for LOS displacement time series (GeoTIFF)
ASCENDING_FOLDER = r"/path/to/your/ascending_los_tif_folder"
DESCENDING_FOLDER = r"/path/to/your/descending_los_tif_folder"

# 2) Output folder for decomposed components (GeoTIFF and optional NumPy)
OUTPUT_FOLDER = r"/path/to/your/output_folder"

# 3) Input file extension
#    This script assumes:
#       - Each LOS file is a single-band GeoTIFF with this extension.
#       - File names are used to align the time series between
#         ascending and descending tracks.
INPUT_FILE_EXTENSION = ".tif"

# 4) Sentinel-1 viewing geometry
#    NOTE:
#    - Incidence angle: measured from the local vertical (degrees).
#      Typical Sentinel-1 IW values are ~34–42 degrees.
#    - LOS azimuth: horizontal direction of LOS projection (degrees),
#      measured clockwise from geographic north, pointing from ground
#      to satellite.
#
#    For example
#       - LOS azimuth ≈ 80° for ascending tracks
#       - LOS azimuth ≈ -80° (or 280°) for descending tracks
#
#    You SHOULD replace these with the exact values from your
INC_ASC_DEG = 36.5    # TODO: replace with your ascending incidence angle (deg)
AZ_ASC_DEG = 78.0     # TODO: replace with your ascending LOS azimuth (deg)
INC_DESC_DEG = 38.5   # TODO: replace with your descending incidence angle (deg)
AZ_DESC_DEG = 280.0   # TODO: replace with your descending LOS azimuth (deg)

# 5) LOS sign convention
#    Set this flag according to how your LOS displacement is defined.
#    - If your LOS data are POSITIVE when motion is TOWARDS the satellite
#      (range shortening), set this to True.
#    - If your LOS data are POSITIVE when motion is AWAY from the satellite,
#      set this to False (the script will flip the sign internally).
LOS_POSITIVE_TOWARDS_SATELLITE = True

# 6) Output settings
#    - Per-epoch GeoTIFF:
#        Each epoch will be saved as two GeoTIFF files:
#            "<name>_east.tif"  and "<name>_up.tif"
#        using the georeferencing of the input LOS files.
#    - Optional NumPy stacks:
#        east_stack.npy and up_stack.npy will store the full time series
#        as 3D arrays: (n_time, n_rows, n_cols).
SAVE_PER_EPOCH_GEOTIFF = True
SAVE_STACK_AS_NPY = True

# ================================================================
# END OF USER SETTINGS
# ================================================================

import os
from typing import List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.crs import CRS


def los_unit_vector(incidence_deg: float, azimuth_deg: float) -> np.ndarray:
    """
    Compute the LOS unit vector in the local ENU coordinate system.

    Parameters
    ----------
    incidence_deg : float
        Incidence angle in degrees, measured from the local vertical.
        Typical Sentinel-1 IW values are ~34–42 degrees.
    azimuth_deg : float
        LOS azimuth in degrees, measured clockwise from geographic north.
        This is the azimuth of the projection of the LOS vector on the
        horizontal plane, pointing from ground to satellite.

    Returns
    -------
    los_vec : np.ndarray
        1D array with shape (3,) representing the LOS unit vector
        components in ENU coordinates:
        [E_component, N_component, U_component].

    Notes
    -----
    The sign convention is:
        - Positive LOS is defined as motion TOWARDS the satellite.
        - The returned vector is constructed so that:

              d_los = E * los_vec[0] +
                      N * los_vec[1] +
                      U * los_vec[2]

          where d_los is the LOS displacement.
    """
    incidence_rad = np.deg2rad(incidence_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)

    # Horizontal projection of LOS on the ground plane.
    # East and North components follow the standard ENU system.
    e_east = -np.sin(incidence_rad) * np.sin(azimuth_rad)
    e_north = -np.sin(incidence_rad) * np.cos(azimuth_rad)
    e_up = np.cos(incidence_rad)

    los_vec = np.array([e_east, e_north, e_up], dtype=float)
    return los_vec


def build_eu_decomposition_matrix(
    inc_asc_deg: float,
    az_asc_deg: float,
    inc_desc_deg: float,
    az_desc_deg: float,
) -> np.ndarray:
    """
    Build the 2x2 coefficient matrix to decompose LOS into East and Up.

    Parameters
    ----------
    inc_asc_deg : float
        Incidence angle of the ascending track in degrees.
    az_asc_deg : float
        LOS azimuth of the ascending track in degrees (clockwise from north).
    inc_desc_deg : float
        Incidence angle of the descending track in degrees.
    az_desc_deg : float
        LOS azimuth of the descending track in degrees (clockwise from north).

    Returns
    -------
    A : np.ndarray
        2x2 coefficient matrix that relates [E, U] to [d_los_asc, d_los_desc]:

            [d_los_asc]   [A11  A12] [E]
            [d_los_desc] = [A21  A22] [U]

        where:
            - A11, A21 are the East components of the LOS unit vectors.
            - A12, A22 are the Up components of the LOS unit vectors.

    Notes
    -----
    The North component is assumed to be zero in the decomposition. This
    is a common approximation when only a pair of ascending and
    descending tracks is available and no additional constraints exist.
    """
    los_asc = los_unit_vector(inc_asc_deg, az_asc_deg)
    los_desc = los_unit_vector(inc_desc_deg, az_desc_deg)

    # Extract East and Up components only.
    A11 = los_asc[0]  # East component for ascending
    A12 = los_asc[2]  # Up component for ascending
    A21 = los_desc[0]  # East component for descending
    A22 = los_desc[2]  # Up component for descending

    A = np.array([[A11, A12],
                  [A21, A22]], dtype=float)
    return A


def decompose_los_to_east_up(
    los_asc: np.ndarray,
    los_desc: np.ndarray,
    inc_asc_deg: float,
    az_asc_deg: float,
    inc_desc_deg: float,
    az_desc_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose ascending and descending LOS displacement into East and Up.

    Parameters
    ----------
    los_asc : np.ndarray
        LOS displacement(s) for the ascending track.
        The array can have any shape, for example:
            - (n_time, n_rows, n_cols)
            - (n_rows, n_cols)   (single epoch)
        Units are arbitrary but must be consistent with the desired
        East/Up units (e.g., mm, cm, m).
    los_desc : np.ndarray
        LOS displacement(s) for the descending track, with the same shape
        as `los_asc`. Each element must correspond to the same spatial
        location and (optionally) the same time epoch as in `los_asc`.
    inc_asc_deg : float
        Incidence angle of the ascending track in degrees.
    az_asc_deg : float
        LOS azimuth of the ascending track in degrees (clockwise from north).
    inc_desc_deg : float
        Incidence angle of the descending track in degrees.
    az_desc_deg : float
        LOS azimuth of the descending track in degrees (clockwise from north).

    Returns
    -------
    east : np.ndarray
        East-West displacement component with the same shape as `los_asc`.
        Positive values indicate motion towards the east.
    up : np.ndarray
        Vertical displacement component with the same shape as `los_asc`.
        Positive values indicate motion upwards (uplift).

    Raises
    ------
    ValueError
        If the ascending and descending LOS arrays do not have the same
        shape, or if the geometry matrix is singular (non-invertible).

    Notes
    -----
    The solution is obtained by solving the linear system:

        [d_los_asc]   [A11  A12] [E]
        [d_los_desc] = [A21  A22] [U]

    for E and U at each element. The same linear operator is applied to
    all elements of the LOS arrays, assuming uniform viewing geometry
    over the entire study area.
    """
    if los_asc.shape != los_desc.shape:
        raise ValueError("Ascending and descending LOS arrays must have the same shape.")

    # Build the 2x2 decomposition matrix from geometry.
    A = build_eu_decomposition_matrix(
        inc_asc_deg=inc_asc_deg,
        az_asc_deg=az_asc_deg,
        inc_desc_deg=inc_desc_deg,
        az_desc_deg=az_desc_deg,
    )

    # Check that the matrix is invertible.
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0.0):
        raise ValueError(
            "Geometry matrix is nearly singular. "
            "Check that ascending and descending tracks have sufficiently "
            "different viewing geometries."
        )

    A_inv = np.linalg.inv(A)

    # Flatten the LOS arrays to apply the same 2x2 linear operator.
    los_stack = np.stack([los_asc, los_desc], axis=0)    # Shape: (2, ...)
    flat_shape = (2, -1)                                 # Collapse all non-first dimensions
    los_flat = los_stack.reshape(flat_shape)             # Shape: (2, n)

    # Apply the inverse matrix to obtain East and Up in flattened form.
    eu_flat = A_inv @ los_flat                           # Shape: (2, n)

    # Reshape back to the original LOS array shape.
    east = eu_flat[0].reshape(los_asc.shape)
    up = eu_flat[1].reshape(los_asc.shape)

    return east, up


def list_los_files(folder: str, extension: str) -> List[str]:
    """
    List and sort LOS files in a folder with a given extension.

    Parameters
    ----------
    folder : str
        Path to the folder containing LOS GeoTIFF files.
    extension : str
        File extension to filter (e.g., ".tif" or ".tiff").

    Returns
    -------
    files : list of str
        Sorted list of file names (not full paths) that match the extension.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    ValueError
        If no files with the given extension are found.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith(extension.lower())]
    if not files:
        raise ValueError(f"No files with extension '{extension}' found in folder: {folder}")

    files.sort()
    return files


def read_raster_profile(folder: str, file_name: str) -> Tuple[Affine, CRS, int, int, Optional[float]]:
    """
    Read basic raster properties from a single GeoTIFF file.

    Parameters
    ----------
    folder : str
        Path to the folder containing the raster.
    file_name : str
        Name of the raster file.

    Returns
    -------
    transform : Affine
        Affine transform of the raster.
    crs : rasterio.crs.CRS
        Coordinate reference system of the raster.
    width : int
        Number of columns (pixels) in the raster.
    height : int
        Number of rows (pixels) in the raster.
    nodata : float or None
        NoData value of the raster (may be None).
    """
    full_path = os.path.join(folder, file_name)
    with rasterio.open(full_path) as ds:
        transform = ds.transform
        crs = ds.crs
        width = ds.width
        height = ds.height
        nodata = ds.nodata
    return transform, crs, width, height, nodata


def load_los_stack_tif(folder: str, file_names: List[str]) -> np.ndarray:
    """
    Load a stack of LOS arrays from GeoTIFF files into a single NumPy array.

    Parameters
    ----------
    folder : str
        Path to the folder containing LOS GeoTIFF files.
    file_names : list of str
        List of file names to load (relative to the folder).

    Returns
    -------
    los_stack : np.ndarray
        LOS time series as a NumPy array.
        Shape: (n_time, n_rows, n_cols), where:
            - n_time = number of files
            - n_rows, n_cols = shape of each 2D LOS array

    Raises
    ------
    ValueError
        If the shapes of the loaded arrays are not consistent.
    """
    los_list = []
    reference_shape = None

    for fname in file_names:
        full_path = os.path.join(folder, fname)
        with rasterio.open(full_path) as ds:
            data = ds.read(1)  # Read first band
            nodata = ds.nodata

        # Convert nodata values (if any) to NaN to propagate them
        # into the decomposed results.
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        if reference_shape is None:
            reference_shape = data.shape
        elif data.shape != reference_shape:
            raise ValueError(
                f"Inconsistent array shape detected in file '{full_path}'. "
                f"Expected shape {reference_shape}, but got {data.shape}."
            )

        los_list.append(data.astype(np.float32))

    los_stack = np.stack(los_list, axis=0)  # Shape: (n_time, n_rows, n_cols)
    return los_stack


def ensure_output_folder(folder: str) -> None:
    """
    Create the output folder if it does not exist.

    Parameters
    ----------
    folder : str
        Path to the output folder.
    """
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)


def save_stack_as_npy(output_folder: str, file_name: str, array: np.ndarray) -> str:
    """
    Save a NumPy array to a ".npy" file inside the given output folder.

    Parameters
    ----------
    output_folder : str
        Path to the output folder.
    file_name : str
        Name of the output file (e.g., "east_stack.npy").
    array : np.ndarray
        Array to be saved.

    Returns
    -------
    full_path : str
        Full path to the saved file.
    """
    full_path = os.path.join(output_folder, file_name)
    np.save(full_path, array)
    return full_path


def save_per_epoch_geotiff(
    output_folder: str,
    base_file_names: List[str],
    east_stack: np.ndarray,
    up_stack: np.ndarray,
    template_folder: str,
) -> None:
    """
    Save East and Up components per epoch as GeoTIFF files.

    Parameters
    ----------
    output_folder : str
        Path to the output folder.
    base_file_names : list of str
        List of original LOS file names (e.g., from ascending folder).
        This defines the time order.
    east_stack : np.ndarray
        East component time series with shape (n_time, n_rows, n_cols).
    up_stack : np.ndarray
        Up component time series with shape (n_time, n_rows, n_cols).
    template_folder : str
        Folder containing the original LOS files used as templates
        for georeferencing (transform, CRS, nodata, etc.).

    Notes
    -----
    This function will create two sets of files per epoch:
        - "<original_name>_east.tif"
        - "<original_name>_up.tif"

    Example:
        Input LOS file:  "20180101.tif"
        Output files:    "20180101_east.tif" and "20180101_up.tif"
    """
    n_time = east_stack.shape[0]
    if n_time != len(base_file_names):
        raise ValueError(
            "Number of time steps in East/Up stack does not match the "
            "number of file names."
        )

    # Use the first LOS file as a template for GeoTIFF metadata
    template_path = os.path.join(template_folder, base_file_names[0])
    with rasterio.open(template_path) as ds_template:
        profile = ds_template.profile

    # Adjust profile for output (single-band float32)
    profile.update(
        dtype=rasterio.float32,
        count=1
    )

    for idx, fname in enumerate(base_file_names):
        stem, _ = os.path.splitext(fname)

        east_out_path = os.path.join(output_folder, f"{stem}_east.tif")
        up_out_path = os.path.join(output_folder, f"{stem}_up.tif")

        east_data = east_stack[idx].astype(np.float32)
        up_data = up_stack[idx].astype(np.float32)

        with rasterio.open(east_out_path, "w", **profile) as dst_e:
            dst_e.write(east_data, 1)

        with rasterio.open(up_out_path, "w", **profile) as dst_u:
            dst_u.write(up_data, 1)


def main() -> None:
    """
    Main driver function.

    Steps
    -----
    1. Read ascending and descending LOS file lists (.tif).
    2. Check that the time series are aligned (same file names).
    3. Check that ascending and descending rasters share the same
       shape, transform, and CRS (using the first file as reference).
    4. Load LOS time series from GeoTIFF into NumPy arrays.
    5. Adjust LOS sign if necessary (according to LOS_POSITIVE_TOWARDS_SATELLITE).
    6. Decompose LOS into East and Up components.
    7. Save output stacks as NumPy arrays (optional).
    8. Save per-epoch East/Up components as GeoTIFF (optional).
    9. Print a short summary.
    """
    print("====================================================")
    print(" InSAR LOS decomposition (Sentinel-1, Asc + Desc) ")
    print("       Input format: GeoTIFF time series          ")
    print("====================================================\n")

    # --------------------------------------------------
    # 1) List and align LOS files for ascending/descending tracks
    # --------------------------------------------------
    print("Reading file lists...")
    asc_files = list_los_files(ASCENDING_FOLDER, INPUT_FILE_EXTENSION)
    desc_files = list_los_files(DESCENDING_FOLDER, INPUT_FILE_EXTENSION)

    # Time series alignment: ascending and descending folders must have
    # the same file names (order does not matter).
    if sorted(asc_files) != sorted(desc_files):
        raise ValueError(
            "Time series alignment error:\n"
            "Ascending and descending folders must contain files with "
            "IDENTICAL names.\n"
            "Please check that both folders have the same dates/epochs."
        )

    # Use the ascending file list as the reference order
    asc_files.sort()
    desc_files = asc_files.copy()

    print(f"Number of epochs: {len(asc_files)}")
    print("Example epoch file name:", asc_files[0])

    # --------------------------------------------------
    # 2) Check basic raster compatibility (shape, transform, CRS)
    # --------------------------------------------------
    print("\nChecking raster compatibility between ascending and descending...")
    asc_transform, asc_crs, asc_width, asc_height, asc_nodata = read_raster_profile(
        ASCENDING_FOLDER, asc_files[0]
    )
    desc_transform, desc_crs, desc_width, desc_height, desc_nodata = read_raster_profile(
        DESCENDING_FOLDER, desc_files[0]
    )

    if (asc_width != desc_width) or (asc_height != desc_height):
        raise ValueError(
            "Raster size mismatch between ascending and descending datasets.\n"
            f"Ascending size : {asc_width} x {asc_height}\n"
            f"Descending size: {desc_width} x {desc_height}"
        )

    if asc_transform != desc_transform:
        raise ValueError("Geotransform mismatch between ascending and descending datasets.")

    if asc_crs != desc_crs:
        raise ValueError("CRS mismatch between ascending and descending datasets.")

    print("Raster compatibility check passed.")

    # --------------------------------------------------
    # 3) Load LOS stacks from GeoTIFF
    # --------------------------------------------------
    print("\nLoading ascending LOS stack from GeoTIFF...")
    los_asc_stack = load_los_stack_tif(ASCENDING_FOLDER, asc_files)
    print("Ascending LOS stack shape:", los_asc_stack.shape)

    print("\nLoading descending LOS stack from GeoTIFF...")
    los_desc_stack = load_los_stack_tif(DESCENDING_FOLDER, desc_files)
    print("Descending LOS stack shape:", los_desc_stack.shape)

    # --------------------------------------------------
    # 4) Adjust LOS sign convention if needed
    # --------------------------------------------------
    if LOS_POSITIVE_TOWARDS_SATELLITE:
        print("\nLOS sign convention: positive = TOWARDS satellite (no change).")
    else:
        print("\nLOS sign convention: positive = AWAY from satellite.")
        print("Flipping LOS signs to match the internal convention...")
        los_asc_stack = -los_asc_stack
        los_desc_stack = -los_desc_stack

    # --------------------------------------------------
    # 5) Decompose LOS to East and Up components
    # --------------------------------------------------
    print("\nDecomposing LOS to East and Up components...")
    east_stack, up_stack = decompose_los_to_east_up(
        los_asc=los_asc_stack,
        los_desc=los_desc_stack,
        inc_asc_deg=INC_ASC_DEG,
        az_asc_deg=AZ_ASC_DEG,
        inc_desc_deg=INC_DESC_DEG,
        az_desc_deg=AZ_DESC_DEG,
    )
    print("Decomposition finished.")
    print("East stack shape:", east_stack.shape)
    print("Up stack shape:  ", up_stack.shape)

    # --------------------------------------------------
    # 6) Save results
    # --------------------------------------------------
    print("\nSaving results...")
    ensure_output_folder(OUTPUT_FOLDER)

    # Optional: save NumPy stacks
    if SAVE_STACK_AS_NPY:
        east_stack_path = save_stack_as_npy(OUTPUT_FOLDER, "east_stack.npy", east_stack)
        up_stack_path = save_stack_as_npy(OUTPUT_FOLDER, "up_stack.npy", up_stack)
        print(f"Saved East stack to: {east_stack_path}")
        print(f"Saved Up stack   to: {up_stack_path}")

    # Save per-epoch GeoTIFFs
    if SAVE_PER_EPOCH_GEOTIFF:
        print("Saving per-epoch East/Up GeoTIFF files...")
        save_per_epoch_geotiff(
            output_folder=OUTPUT_FOLDER,
            base_file_names=asc_files,
            east_stack=east_stack,
            up_stack=up_stack,
            template_folder=ASCENDING_FOLDER,
        )
        print("Per-epoch GeoTIFF files saved.")

    # --------------------------------------------------
    # 7) Summary
    # --------------------------------------------------
    print("\n====================== SUMMARY =====================")
    print(f"Ascending LOS folder : {ASCENDING_FOLDER}")
    print(f"Descending LOS folder: {DESCENDING_FOLDER}")
    print(f"Output folder        : {OUTPUT_FOLDER}")
    print(f"Number of epochs     : {east_stack.shape[0]}")
    print("Spatial size         : "
          f"{east_stack.shape[-2]} rows x {east_stack.shape[-1]} cols")
    print("\nOutputs:")
    if SAVE_STACK_AS_NPY:
        print(" - 'east_stack.npy' : East-West displacement time series")
        print(" - 'up_stack.npy'   : Vertical (Up) displacement time series")
    if SAVE_PER_EPOCH_GEOTIFF:
        print(" - '<date>_east.tif' and '<date>_up.tif' per epoch")
    print("====================================================")


if __name__ == "__main__":
    main()
