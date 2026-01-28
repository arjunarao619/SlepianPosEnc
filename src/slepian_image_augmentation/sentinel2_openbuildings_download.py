#!/usr/bin/env python3
"""
Sentinel-2 Imagery Download for Galileo Embeddings
===================================================
Downloads S2 median composites for building density grid cells.
Includes checkpointing, progress tracking, and periodic visualizations.

Usage:
    python sentinel2_openbuildings_download.py \
        --data-dir $DATA_DIR \
        --regions mexicocity dhaka maharashtra southflorida \
        --output-dir $DATA_DIR/s2_patches \
        --ee-project $EE_PROJECT
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CRITICAL: Patch urllib3 connection pool BEFORE importing Earth Engine
# This fixes "connection pool is full, discarding connection" warnings
# =============================================================================
import urllib3

# Increase default pool size from 10 to 100
urllib3.connectionpool.HTTPConnectionPool.DEFAULT_MAXSIZE = 100
urllib3.connectionpool.HTTPSConnectionPool.DEFAULT_MAXSIZE = 100

# Also patch PoolManager defaults
_original_pool_manager_init = urllib3.PoolManager.__init__
def _patched_pool_manager_init(self, num_pools=100, headers=None, **connection_pool_kw):
    connection_pool_kw.setdefault('maxsize', 100)
    _original_pool_manager_init(self, num_pools=num_pools, headers=headers, **connection_pool_kw)
urllib3.PoolManager.__init__ = _patched_pool_manager_init

# Suppress the warning entirely if pool is still exceeded
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# =============================================================================
# Now safe to import Earth Engine and other libraries
# =============================================================================
import ee
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# viz imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not found, using basic plots")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== CONSTANTS ====================

# bands galileo needs (in order)
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
PATCH_SIZE_PIXELS = 64  # 64x64 pixels
PATCH_RESOLUTION_M = 10  # 10m per pixel
PATCH_SIZE_M = PATCH_SIZE_PIXELS * PATCH_RESOLUTION_M  # 640m


# ==================== EARTH ENGINE ====================

def configure_connection_pool(pool_size: int = 100):
    """
    Configure urllib3 connection pool for MAXIMUM throughput.

    Args:
        pool_size: Maximum connections per pool (default 100, can go 500+)

    Note: urllib3 defaults are patched at import time, but this function
          allows runtime configuration via CLI --pool-size argument.
    """
    # Update urllib3 defaults at runtime
    urllib3.connectionpool.HTTPConnectionPool.DEFAULT_MAXSIZE = pool_size
    urllib3.connectionpool.HTTPSConnectionPool.DEFAULT_MAXSIZE = pool_size

    # Re-patch PoolManager with new size
    global _original_pool_manager_init
    def _patched_pool_manager_init_runtime(self, num_pools=pool_size, headers=None, **connection_pool_kw):
        connection_pool_kw.setdefault('maxsize', pool_size)
        _original_pool_manager_init(self, num_pools=num_pools, headers=headers, **connection_pool_kw)
    urllib3.PoolManager.__init__ = _patched_pool_manager_init_runtime

    logger.info(f"Configured urllib3 pool_size={pool_size}")


def init_ee(project: str = None, pool_size: int = 100) -> bool:
    """
    Initialize Earth Engine with increased connection pool.

    Args:
        project: GEE project ID
        pool_size: Connection pool size (default 100, can go 500+)
    """
    try:
        # Configure connection pool BEFORE EE init takes effect
        configure_connection_pool(pool_size=pool_size)

        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()

        _ = ee.Number(1).getInfo()
        logger.info(f"EE initialized (project={project}, pool_size={pool_size})")
        return True
    except Exception as e:
        logger.error(f"EE init failed: {e}")
        raise


def get_s2_median_composite(
    lon: float, 
    lat: float,
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    cloud_thresh: int = 20
) -> Optional[np.ndarray]:
    """
    Get median S2 composite for a point.
    
    Returns:
        numpy array [H, W, 10] or None if failed
    """
    try:
        # define region (square patch centered on point)
        half_size = PATCH_SIZE_M / 2
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(half_size).bounds()
        
        # load and filter S2
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(region)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
              .select(S2_BANDS))
        
        # check if we have images
        count = s2.size().getInfo()
        if count == 0:
            return None

        # median composite
        composite = s2.median()

        # CRITICAL: Reproject to fixed scale before sampling
        # After .median(), the image has no well-defined projection/scale,
        # so sampleRectangle returns only 1 pixel. We must explicitly set
        # the projection and scale to get the correct 64x64 pixel output.
        proj = s2.first().select('B4').projection()
        composite = composite.reproject(crs=proj, scale=PATCH_RESOLUTION_M)

        # sample to array
        pixels = composite.sampleRectangle(
            region=region,
            defaultValue=0
        )
        
        # extract band arrays
        band_arrays = []
        for band in S2_BANDS:
            arr = np.array(pixels.get(band).getInfo(), dtype=np.float32)
            band_arrays.append(arr)

        # Verify all bands have the same shape
        if len(set(a.shape for a in band_arrays)) != 1:
            logger.debug(f"Inconsistent band shapes at ({lon:.4f}, {lat:.4f})")
            return None

        # stack [H, W, C] using dstack for clarity
        patch = np.dstack(band_arrays).astype(np.float32)

        return patch
        
    except Exception as e:
        logger.debug(f"Failed at ({lon:.4f}, {lat:.4f}): {e}")
        return None


def get_s2_batch_getpixels(
    coordinates: List[Tuple[float, float]],
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    cloud_thresh: int = 20,
    n_workers: int = 8
) -> Dict[Tuple[float, float], np.ndarray]:
    """
    Download S2 patches for multiple coordinates at MAXIMUM speed.

    Args:
        coordinates: List of (lon, lat) tuples
        start_date: Composite start date
        end_date: Composite end date
        cloud_thresh: Max cloud percentage
        n_workers: Number of parallel workers (can go to 16-32 with large pool)
    """
    results = {}
    failed = []

    def download_one(coord):
        lon, lat = coord
        # No delay - fire as fast as possible, let retry handle failures
        patch = get_s2_median_composite(lon, lat, start_date, end_date, cloud_thresh)
        return coord, patch

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(download_one, c): c for c in coordinates}

        for future in as_completed(futures):
            try:
                coord, patch = future.result(timeout=180)  # 3 min timeout for slow requests
                if patch is not None:
                    results[coord] = patch
                else:
                    failed.append(coord)
            except Exception as e:
                coord = futures[future]
                logger.debug(f"Error {coord}: {e}")
                failed.append(coord)

    return results, failed


# ==================== CHECKPOINTING ====================

class DownloadTracker:
    """Track download progress with checkpointing."""
    
    def __init__(self, output_dir: Path, region: str):
        self.output_dir = output_dir
        self.region = region
        self.checkpoint_file = output_dir / f"{region}_checkpoint.json"
        self.h5_file = output_dir / f"{region}_s2_patches.h5"
        
        # load or init state
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                self.state = json.load(f)
            logger.info(f"Resumed checkpoint: {len(self.state['completed'])} done")
        else:
            self.state = {
                'completed': [],
                'failed': [],
                'started_at': datetime.now().isoformat(),
                'last_update': None,
                'total': 0
            }
        
        # stats for viz
        self.stats = {
            'download_times': [],
            'patch_shapes': [],
            'timestamps': []
        }
    
    def is_done(self, coord: Tuple[float, float]) -> bool:
        return list(coord) in self.state['completed']
    
    def mark_done(self, coord: Tuple[float, float]):
        self.state['completed'].append(list(coord))
    
    def mark_failed(self, coord: Tuple[float, float]):
        self.state['failed'].append(list(coord))
    
    def save_checkpoint(self):
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f)
    
    def get_progress(self) -> Tuple[int, int, int]:
        """Returns (completed, failed, total)"""
        return len(self.state['completed']), len(self.state['failed']), self.state['total']


def save_patches_h5(h5_path: Path, patches: Dict, mode: str = 'a'):
    """Save patches to HDF5 file."""
    with h5py.File(h5_path, mode) as f:
        for (lon, lat), patch in patches.items():
            key = f"{lon:.4f}_{lat:.4f}"
            if key not in f:
                f.create_dataset(key, data=patch, compression='gzip', compression_opts=4)


# ==================== VISUALIZATION ====================

def plot_acquisition_map(
    df: pd.DataFrame,
    completed_coords: List,
    failed_coords: List,
    region: str,
    output_path: Path,
    show_density: bool = True
):
    """
    Plot map showing acquisition progress overlaid on building density.
    """
    completed_set = set(tuple(c) for c in completed_coords)
    failed_set = set(tuple(c) for c in failed_coords)
    
    # classify points
    df = df.copy()
    df['status'] = 'pending'
    for idx, row in df.iterrows():
        coord = (row['lon'], row['lat'])
        if coord in completed_set:
            df.loc[idx, 'status'] = 'done'
        elif coord in failed_set:
            df.loc[idx, 'status'] = 'failed'
    
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # add map features
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
        
        extent = [df['lon'].min() - 0.5, df['lon'].max() + 0.5,
                  df['lat'].min() - 0.5, df['lat'].max() + 0.5]
        ax.set_extent(extent)
    else:
        fig, ax = plt.subplots(figsize=(14, 10))
    
    # plot building density as background
    if show_density:
        scatter_bg = ax.scatter(
            df['lon'], df['lat'],
            c=np.log1p(df['building_count']),
            cmap='YlOrRd',
            s=5,
            alpha=0.3,
            transform=ccrs.PlateCarree() if HAS_CARTOPY else None
        )
        plt.colorbar(scatter_bg, ax=ax, label='log(building_count)', shrink=0.7, pad=0.02)
    
    # overlay acquisition status
    done_df = df[df['status'] == 'done']
    pending_df = df[df['status'] == 'pending']
    failed_df = df[df['status'] == 'failed']
    
    if len(done_df) > 0:
        ax.scatter(done_df['lon'], done_df['lat'], c='green', s=8, alpha=0.6,
                  label=f'Done ({len(done_df):,})',
                  transform=ccrs.PlateCarree() if HAS_CARTOPY else None)
    
    if len(pending_df) > 0:
        ax.scatter(pending_df['lon'], pending_df['lat'], c='blue', s=3, alpha=0.2,
                  label=f'Pending ({len(pending_df):,})',
                  transform=ccrs.PlateCarree() if HAS_CARTOPY else None)
    
    if len(failed_df) > 0:
        ax.scatter(failed_df['lon'], failed_df['lat'], c='red', s=10, alpha=0.8,
                  marker='x', label=f'Failed ({len(failed_df):,})',
                  transform=ccrs.PlateCarree() if HAS_CARTOPY else None)
    
    # stats
    pct_done = len(done_df) / len(df) * 100 if len(df) > 0 else 0
    ax.set_title(f'{region}: S2 Acquisition Progress\n'
                f'{len(done_df):,}/{len(df):,} ({pct_done:.1f}%) complete',
                fontsize=14)
    ax.legend(loc='upper right')
    
    if HAS_CARTOPY:
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved map: {output_path}")


def normalize_rgb_for_display(rgb: np.ndarray) -> np.ndarray:
    """
    Normalize RGB array for display using percentile stretch.
    This provides better contrast than fixed division.

    Args:
        rgb: Array of shape (H, W, 3) with raw Sentinel-2 values

    Returns:
        Normalized array with values in [0, 1]
    """
    rgb_normalized = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        band = rgb[:, :, i].astype(np.float32)
        # Use percentile stretch for better contrast
        p2, p98 = np.percentile(band[band > 0], [2, 98]) if np.any(band > 0) else (0, 1)
        if p98 > p2:
            band = (band - p2) / (p98 - p2)
        else:
            band = band / 3000.0  # fallback to simple normalization
        rgb_normalized[:, :, i] = np.clip(band, 0, 1)

    return rgb_normalized


def plot_sample_patches(
    patches: Dict[Tuple[float, float], np.ndarray],
    region: str,
    output_path: Path,
    n_samples: int = 16
):
    """Plot grid of sample patches (RGB)."""
    coords = list(patches.keys())[:n_samples]
    n = len(coords)
    if n == 0:
        return

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, coord in enumerate(coords):
        patch = patches[coord]

        # S2_BANDS = ['B2', 'B3', 'B4', ...] means:
        # index 0 = B2 (Blue), index 1 = B3 (Green), index 2 = B4 (Red)
        # For RGB display: need [Red, Green, Blue] = [B4, B3, B2] = indices [2, 1, 0]
        rgb = patch[:, :, [2, 1, 0]].astype(np.float32)

        # normalize for display using percentile stretch
        rgb = normalize_rgb_for_display(rgb)

        axes[i].imshow(rgb)
        axes[i].set_title(f'({coord[0]:.3f}, {coord[1]:.3f})\n{patch.shape}', fontsize=8)
        axes[i].axis('off')

    # hide empty
    for i in range(n, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'{region}: Sample S2 Patches (RGB)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved samples: {output_path}")


def plot_random_samples_to_temp(
    h5_path: Path,
    n_samples: int = 5,
    temp_dir: Path = None
):
    """
    Plot random samples from an HDF5 file to a temporary directory.

    Args:
        h5_path: Path to HDF5 file containing patches
        n_samples: Number of random samples to plot (default 5)
        temp_dir: Directory to save plots (default: ./tmp_samples)
    """
    import random
    import tempfile

    if temp_dir is None:
        temp_dir = Path.cwd() / "tmp_samples"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        logger.error(f"HDF5 file not found: {h5_path}")
        return

    with h5py.File(h5_path, 'r') as f:
        all_keys = list(f.keys())

        if len(all_keys) == 0:
            logger.warning("No patches found in HDF5 file")
            return

        # randomly sample
        sample_keys = random.sample(all_keys, min(n_samples, len(all_keys)))

        for i, key in enumerate(sample_keys):
            patch = np.array(f[key])

            # Extract RGB: indices [2,1,0] for [B4, B3, B2] -> [R, G, B]
            rgb = patch[:, :, [2, 1, 0]].astype(np.float32)

            # Normalize for display
            rgb = normalize_rgb_for_display(rgb)

            # Parse coordinates from key
            parts = key.split('_')
            lon, lat = float(parts[0]), float(parts[1])

            # Plot individual sample
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(rgb)
            ax.set_title(f'Sample {i+1}: ({lon:.4f}, {lat:.4f})\nShape: {patch.shape}')
            ax.axis('off')

            output_path = temp_dir / f"sample_{i+1}_lon{lon:.4f}_lat{lat:.4f}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved: {output_path}")

    logger.info(f"Saved {len(sample_keys)} random samples to {temp_dir}")


def plot_progress_timeline(tracker: DownloadTracker, output_path: Path):
    """Plot download rate over time."""
    if len(tracker.stats['timestamps']) < 2:
        return
    
    times = pd.to_datetime(tracker.stats['timestamps'])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # cumulative progress
    ax.plot(times, range(1, len(times)+1), 'b-', linewidth=1)
    ax.fill_between(times, 0, range(1, len(times)+1), alpha=0.3)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Patches Downloaded')
    ax.set_title(f'{tracker.region}: Download Progress')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


# ==================== MAIN DOWNLOAD LOOP ====================

def download_region(
    df: pd.DataFrame,
    region: str,
    output_dir: Path,
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    cloud_thresh: int = 20,
    batch_size: int = 50,
    n_workers: int = 4,
    viz_interval: int = 500,
    checkpoint_interval: int = 100
):
    """
    Download S2 patches for a region with tracking and visualization.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading S2 for: {region}")
    logger.info(f"{'='*60}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / 'viz' / region
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # init tracker
    tracker = DownloadTracker(output_dir, region)
    tracker.state['total'] = len(df)
    
    # get coordinates to process
    all_coords = [(row['lon'], row['lat']) for _, row in df.iterrows()]
    pending_coords = [c for c in all_coords if not tracker.is_done(c)]
    
    logger.info(f"Total: {len(all_coords):,}")
    logger.info(f"Already done: {len(all_coords) - len(pending_coords):,}")
    logger.info(f"Pending: {len(pending_coords):,}")
    
    if len(pending_coords) == 0:
        logger.info("All done!")
        return
    
    # process in batches
    n_batches = (len(pending_coords) + batch_size - 1) // batch_size
    all_new_patches = {}
    
    pbar = tqdm(total=len(pending_coords), desc=f"{region} S2 download")
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(pending_coords))
        batch_coords = pending_coords[batch_start:batch_end]
        
        t0 = time.time()
        
        # download batch
        patches, failed = get_s2_batch_getpixels(
            batch_coords,
            start_date=start_date,
            end_date=end_date,
            cloud_thresh=cloud_thresh,
            n_workers=n_workers
        )
        
        elapsed = time.time() - t0
        
        # update tracker
        for coord in patches:
            tracker.mark_done(coord)
            tracker.stats['timestamps'].append(datetime.now().isoformat())
        for coord in failed:
            tracker.mark_failed(coord)
        
        # save patches incrementally
        if patches:
            save_patches_h5(tracker.h5_file, patches)
            all_new_patches.update(patches)
        
        pbar.update(len(batch_coords))
        pbar.set_postfix({
            'success': f"{len(patches)}/{len(batch_coords)}",
            'rate': f"{len(batch_coords)/elapsed:.1f}/s" if elapsed > 0 else "N/A"
        })
        
        # checkpoint
        if (batch_idx + 1) % (checkpoint_interval // batch_size + 1) == 0:
            tracker.save_checkpoint()
        
        # periodic visualization
        total_done = len(tracker.state['completed'])
        if total_done % viz_interval < batch_size or batch_idx == n_batches - 1:
            # acquisition map
            plot_acquisition_map(
                df,
                tracker.state['completed'],
                tracker.state['failed'],
                region,
                viz_dir / f'acquisition_map_{total_done:06d}.png'
            )
            
            # sample patches
            if all_new_patches:
                plot_sample_patches(
                    all_new_patches,
                    region,
                    viz_dir / f'sample_patches_{total_done:06d}.png',
                    n_samples=16
                )
        
        # rate limiting
        time.sleep(0.1)
    
    pbar.close()
    tracker.save_checkpoint()
    
    # final summary
    done, failed, total = tracker.get_progress()
    logger.info(f"\n{region} complete:")
    logger.info(f"  Downloaded: {done:,}/{total:,} ({done/total*100:.1f}%)")
    logger.info(f"  Failed: {failed:,}")
    logger.info(f"  H5 file: {tracker.h5_file}")
    
    # final viz
    plot_acquisition_map(
        df,
        tracker.state['completed'],
        tracker.state['failed'],
        region,
        viz_dir / 'acquisition_map_final.png'
    )


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Download S2 patches for Galileo')
    
    parser.add_argument('--data-dir', type=Path, default=None,
                       help='Directory containing *_grid.parquet files')
    parser.add_argument('--regions', nargs='+', default=None,
                       help='Region names (parquet prefix)')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for S2 patches')
    parser.add_argument('--ee-project', type=str, default='sentinel-downloader-468517',
                       help='GEE project ID')
    
    parser.add_argument('--start-date', default='2024-01-01',
                       help='S2 composite start date')
    parser.add_argument('--end-date', default='2024-12-31',
                       help='S2 composite end date')
    parser.add_argument('--cloud-thresh', type=int, default=20,
                       help='Max cloud percentage')
    
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for parallel download (can go 200+)')
    parser.add_argument('--n-workers', type=int, default=16,
                       help='Parallel workers (can go 32+ with large pool)')
    parser.add_argument('--pool-size', type=int, default=200,
                       help='HTTP connection pool size (can go 500-1000)')
    parser.add_argument('--viz-interval', type=int, default=1000,
                       help='Generate viz every N downloads')
    
    parser.add_argument('--subsample', type=int, default=None,
                       help='Subsample to N points per region (for testing)')

    parser.add_argument('--plot-samples', type=Path, default=None,
                       help='Path to HDF5 file to plot 5 random samples from (skips download)')
    parser.add_argument('--temp-dir', type=Path, default=None,
                       help='Directory to save random sample plots (default: ./tmp_samples)')

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("="*60)
    logger.info("S2 Download for Galileo Embeddings")
    logger.info("="*60)

    # If --plot-samples is provided, just plot samples and exit
    if args.plot_samples:
        logger.info(f"Plotting 5 random samples from: {args.plot_samples}")
        plot_random_samples_to_temp(
            h5_path=args.plot_samples,
            n_samples=5,
            temp_dir=args.temp_dir
        )
        return

    # Validate required args for download mode
    if not args.data_dir or not args.regions or not args.output_dir:
        logger.error("--data-dir, --regions, and --output-dir are required for download mode")
        sys.exit(1)

    # init EE with configured connection pool
    init_ee(args.ee_project, pool_size=args.pool_size)
    
    # process each region
    for region in args.regions:
        parquet_path = args.data_dir / f"{region}_grid.parquet"
        
        if not parquet_path.exists():
            logger.error(f"Parquet not found: {parquet_path}")
            continue
        
        df = pd.read_parquet(parquet_path)
        logger.info(f"\nLoaded {region}: {len(df):,} cells")
        
        if args.subsample:
            df = df.sample(n=min(args.subsample, len(df)), random_state=42)
            logger.info(f"Subsampled to {len(df):,} cells")
        
        download_region(
            df=df,
            region=region,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            cloud_thresh=args.cloud_thresh,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
            viz_interval=args.viz_interval
        )
    
    logger.info("\n" + "="*60)
    logger.info("All regions complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()