import os
import requests
import logging
from pathlib import Path
import time
from typing import Union
import argparse

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_AOI_BBOX = (34.62, 92.18, 35.39, 93.14)
DEFAULT_TARGET_DIR = Path(r"F:\zyz\DiplomaProject\data\DEM\Cop")
DEFAULT_INITIAL_DOWNLOAD = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download Copernicus 30m DEM tiles.')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('LAT_MIN', 'LON_MIN', 'LAT_MAX', 'LON_MAX'),
                        default=DEFAULT_AOI_BBOX,
                        help='Bounding box coordinates (lat_min lon_min lat_max lon_max)')
    parser.add_argument('--target-dir', type=Path, default=DEFAULT_TARGET_DIR,
                        help='Directory to save downloaded files')
    parser.add_argument('--initial-download', type=bool, default=DEFAULT_INITIAL_DOWNLOAD,
                        help='Whether to attempt downloading all files or just retry failed ones')
    return parser.parse_args()


def download_cop_dsm(aoi_bbox: tuple, target_dir: Union[str, Path], initial_download: bool = False):
    """Download Copernicus 30m DEM tiles for the specified area of interest.

    Args:
        aoi_bbox: Tuple of (lat_min, lon_min, lat_max, lon_max) defining the area of interest
        target_dir: Directory to save downloaded files
        initial_download: Whether to attempt downloading all files or just retry failed ones
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    base_url = "https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/"
    target_dir = Path(target_dir)

    def get_paths(aoi_bbox):
        # Validate bbox coordinates
        if len(aoi_bbox) != 4:
            raise ValueError("bbox must contain exactly 4 coordinates")
        if aoi_bbox[0] >= aoi_bbox[2] or aoi_bbox[1] >= aoi_bbox[3]:
            raise ValueError("Invalid bbox: min values must be less than max values")

        paths = []

        # Convert float coordinates to integers by flooring
        lat_min = int(aoi_bbox[0])  # Remove -1 since we want to include the tile containing start point
        lon_min = int(aoi_bbox[1])
        lat_max = int(aoi_bbox[2] + 0.99)  # Add 0.99 to ensure we get the last tile
        lon_max = int(aoi_bbox[3] + 0.99)

        logger.debug(f"Tile coordinates: N{lat_min}-{lat_max}, E{lon_min}-{lon_max}")

        for lat in range(lat_min, lat_max + 1):  # Include last tile
            for lon in range(lon_min, lon_max + 1):
                # Handle negative latitudes
                ns = 'S' if lat < 0 else 'N'
                lat_str = f"{abs(lat):02d}"

                # Handle negative longitudes
                ew = 'W' if lon < 0 else 'E'
                lon_str = f"{abs(lon):03d}"

                paths.append(f"Copernicus_DSM_30_{ns}{lat_str}_00_{ew}{lon_str}_00_DEM.tif")
        return paths

    def download_cop_dsm(paths, target_dir):
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        failed_paths = []
        for path in paths:
            url = base_url + path
            target_file = target_dir / path

            if target_file.exists():
                if target_file.stat().st_size > 0:  # Check if file is not empty
                    logger.info(f"File already exists: {path}")
                    continue
                else:
                    logger.warning(f"Found empty file, will redownload: {path}")
                    target_file.unlink()  # Remove empty file

            try:
                response = requests.get(url, timeout=60)  # Increased timeout
                response.raise_for_status()

                # Stream the response to handle large files
                with open(target_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Verify file size
                if target_file.stat().st_size < 1000:  # Arbitrary small size threshold
                    raise ValueError("Downloaded file is too small")

                logger.info(f"Successfully downloaded {path}")

            except Exception as e:
                failed_paths.append(path)
                logger.error(f"Failed to download {path}: {str(e)}")
                if target_file.exists():
                    target_file.unlink()  # Clean up partial downloads
                continue

        return failed_paths

    def identify_failed_files(paths, target_dir):
        failed_paths = []
        for path in paths:
            target_file = target_dir / path
            if not target_file.exists() or target_file.stat().st_size < 1000:
                failed_paths.append(path)
        return failed_paths

    try:
        paths = get_paths(aoi_bbox)
        logger.info(f"Attempting to download {len(paths)} files")

        try:
            if initial_download:
                logger.info("Starting initial download of all files")
                failed_paths = download_cop_dsm(paths, target_dir)
            else:
                logger.info("Skipping initial download and identifying failed files")
                failed_paths = identify_failed_files(paths, target_dir)

            retry_count = 0
            max_retries = 5  # Increased retries

            while failed_paths and retry_count < max_retries:
                retry_count += 1
                logger.info(f"Retry attempt {retry_count} for {len(failed_paths)} failed files")

                # Exponential backoff
                wait_time = min(300, 5 * (2 ** (retry_count - 1)))  # Cap at 5 minutes
                logger.info(f"Waiting {wait_time} seconds before retry {retry_count}")
                time.sleep(wait_time)

                failed_paths = download_cop_dsm(failed_paths, target_dir)

                if failed_paths:
                    logger.warning(f"Failed to download {len(failed_paths)} files after retry {retry_count}")
                    logger.debug("Failed files: " + ", ".join(failed_paths))
                    failed_paths = identify_failed_files(paths, target_dir)
                else:
                    logger.info("All files downloaded successfully")
                    break

            if failed_paths:
                if retry_count >= max_retries:
                    logger.error(f"Max retries ({max_retries}) reached. Could not download {len(failed_paths)} files")
                logger.warning("Failed files: " + ", ".join(failed_paths))
                with open(target_dir / "failed_downloads.txt", "w") as f:
                    f.write("\n".join(failed_paths))
                logger.error(f"Failed files saved to {target_dir}/failed_downloads.txt")
                logger.error(
                    "Please check the file and run the script with --initial-download=False to retry downloading the failed files.")
            else:
                logger.info("All files downloaded successfully")

        except Exception as e:
            logger.error(f"Error during download process: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    try:
        # Check if default parameters are being used and inform user
        if args.bbox == DEFAULT_AOI_BBOX or args.target_dir == DEFAULT_TARGET_DIR or args.initial_download == DEFAULT_INITIAL_DOWNLOAD:
            # Print example usage
            logger.info("Example usage:")
            logger.info(
                "python download_cop_dsm.py --bbox 30.12 90.67 32.15 92.03 --target-dir data/raw/cop_dsm30 --initial-download True")

        logger.info(
            f"Your current download set-up is:\n bbox: {args.bbox}\n target_dir: {args.target_dir}\n initial_download: {args.initial_download}")
        with open("download_config.txt", "w") as f:
            f.write(f"bbox: {args.bbox}\n target_dir: {args.target_dir}\n initial_download: {args.initial_download}")
        logger.info(f"Current download configuration saved to download_config.txt")

        # Download DEM tiles
        download_cop_dsm(
            aoi_bbox=args.bbox,
            target_dir=args.target_dir,
            initial_download=args.initial_download
        )
        logger.info("Successfully downloaded Copernicus DSM tiles")

    except Exception as e:
        logger.error(f"Error downloading DSM tiles: {str(e)}")
        raise  # Re-raise exception to ensure script fails on error