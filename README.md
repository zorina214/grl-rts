# grl-rts
Code and supplementary materials for the GRL paper: “Decoupling Hydroclimatic Controls on Displacement of Retrogressive Thaw Slump in Qinghai-Tibet Plateau”

Script Overview

All scripts below are located in the code_user_version/ folder.

1. Coherence_histogram.py

Purpose

Extract coherence values from ascending and descending Sentinel-1 coherence rasters.

Mask rasters by RTS polygons and compute per-polygon mean coherence.

Plot histograms and kernel-density curves (KDEs) of coherence distributions for each track.

Key I/O

Inputs

Coherence GeoTIFFs (ascending and descending).

RTS polygon shapefile (with ID field).

Outputs

SVG histograms and KDE plots for coherence distributions.

Printed statistics (range, sample size, fraction above threshold, etc.).

2. los_decomposition_s1_tif.py

Purpose

Decompose Sentinel-1 ascending and descending line-of-sight (LOS) displacement time series
into East–West and Up–Down components.

Key I/O

Inputs

Folders with single-band LOS displacement GeoTIFFs for ascending and descending tracks.

User-defined incidence and azimuth angles for each track.

Outputs

Per-epoch East and Up GeoTIFFs.

Optional NumPy stacks (east_stack.npy, up_stack.npy).

3. UD_decompose.py

Purpose

Seasonal and trend decomposition of polygon-based vertical displacement (UD) time series.

For each RTS polygon, fit a sinusoid + linear trend model, perform residual checks,
and extract seasonal amplitude and linear trend.

Key I/O

Inputs

Per-polygon UD time series (tab-separated text files; time, value).

Outputs

For each RTS, a text file with:

Peak and trough dates/values within internal full years.

Seasonal amplitude.

Linear trend parameter.

4. LST_decompose.py

Purpose

Seasonal and trend decomposition of polygon-based LST (land surface temperature) time series,
analogous to the UD decomposition.

Uses daily LST rasters, masked by RTS polygons, to form per-polygon time series, then
fits a sinusoid + trend model with residual diagnostics.

Key I/O

Inputs

Daily LST GeoTIFFs (YYYY/MM/*.tif).

RTS polygon shapefile (with ID field).

Outputs

Per-polygon text files containing:

Peak/trough information.

Seasonal amplitude.

Linear trend.

5. LST_mean.py

Purpose

Compute annual mean LST for each RTS polygon.

For each daily LST raster, the script:

Masks by RTS polygons.

Computes the mean LST over all intersecting pixels (polygon mean per day).

Annual means are obtained by averaging valid daily values per year.

Key I/O

Inputs

RTS polygon shapefile.

Root folder of daily LST GeoTIFFs (LST_ROOT/YYYY/MM/*.tif).

Outputs

For each RTS ID, a file:

<ID>_LST_mean.txt

Columns:

Year, LST_mean.

6. Pre_sum.py

Purpose

Compute annual precipitation sums for each RTS polygon.

For each daily precipitation raster, the script:

Masks by RTS polygons.

Computes the mean precipitation over all intersecting pixels (polygon mean per day).

Annual sums are obtained by summing valid daily values per year.

Key I/O

Inputs

RTS polygon shapefile.

Root folder of daily precipitation GeoTIFFs (INPUT_ROOT/YYYY/MM/*.tif).

Outputs

For each RTS ID, a file:

<ID>_pre_sum.txt

Columns:

Year, Pre_sum.

7. Elevation.py

Purpose

Analyse elevation distribution of RTS polygons relative to a DEM.

Key I/O

Inputs

DEM GeoTIFF.

RTS polygon shapefile.

Workflow

Mask DEM by each RTS polygon.

Compute polygon-mean elevation.

Extract DEM-wide elevation distribution.

Outputs

SVG figure: elevation histograms and KDE curves (RTS vs entire DEM).

Printed statistics and interval-based proportions.

8. Slope.py

Purpose

Analyse slope distribution of RTS polygons relative to a slope raster.

Key I/O

Inputs

Slope GeoTIFF.

RTS polygon shapefile.

Workflow

Mask slope raster by each RTS polygon.

Compute polygon-mean slope.

Extract slope distribution for full study area.

Outputs

SVG figure: slope histograms and KDE curves (RTS vs entire domain).

Printed statistics and interval-based proportions.

9. aspect.py

Purpose

Compute mean aspect for each RTS polygon and classify into directional bins.

Key I/O

Inputs

Aspect GeoTIFF.

RTS polygon shapefile.

Workflow

Mask aspect raster by each polygon.

Compute mean aspect inside each polygon.

Map aspect to 16 directional classes (N, NNE, NE, …, NNW).

Outputs

Text file with:

RTS ID, centroid coordinates, mean aspect, numeric class, direction label.

Additional file with counts per aspect class.

10. UD_LST_lag.py

Purpose

Compute the temporal lag between seasonal LST peaks and UD troughs for each RTS.

Key I/O

Inputs

One global daily temperature/LST time series.

Per-RTS UD time-series text files (e.g., <ID>_UD_ts.txt).

Per-RTS shapefiles (<ID>.shp) for ID discovery.

Workflow

Fit sinusoid+trend models to temperature and deformation.

For each year and RTS:

Identify the first temperature peak and first deformation trough.

Compute lag in days.

Outputs

Combined text file with lag information for all RTS IDs:

RTS_ID, Year, Temp_Peak_Date, Deform_Trough_Date, Lag_Days.

11. Kendall_preprocess.py

Purpose

Prepare copula input data by merging annual mean LST, annual precipitation sum,
and deformation trend for each RTS.

Key I/O

Inputs

<ID>_LST_mean.txt files (from LST_mean.py).

<ID>_pre_sum.txt files (from Pre_sum.py).

<ID>_UD_pt_with_amplitude_trend.txt files (from UD decomposition).

Workflow

Extract Year and LST_mean → x1.

Extract Year and Pre_sum → x2.

Extract Year (from peak_date) and absolute trend → y1.

Merge on Year.

Outputs

For each RTS ID:

<ID>_copula.txt

Columns:

Year, x1 (LST_mean), x2 (Pre_sum), y1 (|trend|), ID.

12. Kendall.py

Purpose

Perform conditional Kendall’s tau analysis on the merged copula data and
generate diagnostic plots.

Key I/O

Inputs

All <ID>_copula.txt files (from Kendall_preprocess.py).

Workflow

Concatenate all RTS records into a single DataFrame.

Conditional Kendall’s tau vs x2 for pair (x1, y1) in quantile bins.

Conditional Kendall’s tau vs x1 for pair (x2, y1) in quantile bins.

Identify bins with minimum and maximum tau (for x1 conditioning).

For extreme bins:

Plot x2 vs y1 scatter.

Fit and plot linear relationship (y1 ~ x2).

Outputs

SVG figures:

Kendall’s tau vs conditioning variable.

Scatter + linear fit for bins with min/max tau.

Printed coefficients of linear fits.

Typical Processing Workflow

A typical end-to-end workflow is:

Terrain and coherence

Coherence_histogram.py

Elevation.py

Slope.py

aspect.py

LOS and vertical deformation

los_decomposition_s1_tif.py

UD_decompose.py

LST / precipitation statistics

LST_mean.py

Pre_sum.py

LST_decompose.py

UD_LST_lag.py

Copula and dependence analysis

Kendall_preprocess.py

Kendall.py

Each script has a clearly separated “USER SETTINGS / PATHS” section at the top where
you must point to your own data (GeoTIFFs, shapefiles, and time-series text files).
