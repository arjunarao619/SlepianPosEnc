#!/usr/bin/env python3
"""
OpenBuildings + AlphaEarth Dataset Creation (Generalized)
=========================================================
Creates grid-based building statistics with AlphaEarth embeddings.
Works with ANY OpenBuildings CSV file(s).

Usage:
    python create_dataset.py \
        --csvs region1=/path/to/region1.csv region2=/path/to/region2.csv \
        --output-dir /path/to/output \
        --samples-per-region 30000

Or process all CSVs in a directory:
    python create_dataset.py \
        --csv-dir /path/to/csvs \
        --output-dir /path/to/output
"""
import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import warnings
warnings.filterwarnings('ignore')

import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import wandb

# Geospatial
try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Warning: geopandas not found, some features disabled")

# Cartopy for nice maps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== EARTH ENGINE INITIALIZATION ====================

def initialize_earth_engine(project: str = None):
    """Initialize Google Earth Engine."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        # Test
        test = ee.Number(1).getInfo()
        logger.info("✓ Earth Engine initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"EE initialization with project failed: {e}")
        try:
            logger.info("Attempting default initialization...")
            ee.Initialize()
            logger.info("✓ Earth Engine initialized (default)")
            return True
        except:
            logger.info("Attempting authentication...")
            ee.Authenticate()
            ee.Initialize()
            logger.info("✓ Earth Engine authenticated and initialized")
            return True


# ==================== ALPHAEARTH EXTRACTION ====================

class AlphaEarthExtractor:
    """Extract AlphaEarth embeddings for grid cells."""
    
    BAND_NAMES = [f'A{i:02d}' for i in range(1, 65)]  # A01 through A64
    
    def __init__(self, year: int = 2024, buffer_meters: int = 10):
        self.year = year
        self.buffer_meters = buffer_meters
        
        # Load AlphaEarth collection
        self.collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
        self.image = self._get_embedding_image()
        
        logger.info(f"✓ AlphaEarth extractor initialized (year={year}, buffer={buffer_meters}m)")
    
    def _get_embedding_image(self):
        """Get the AlphaEarth embedding image for the specified year."""
        start_date = ee.Date.fromYMD(self.year, 1, 1)
        end_date = start_date.advance(1, 'year')
        
        year_collection = self.collection.filterDate(start_date, end_date)
        embedding_image = year_collection.mosaic()
        
        return embedding_image
    
    def extract_embedding(self, lon: float, lat: float) -> Optional[np.ndarray]:
        """
        Extract 64-dim AlphaEarth embedding for a single location.
        
        Returns:
            numpy array [64] or None if extraction fails
        """
        try:
            point = ee.Geometry.Point([lon, lat])
            
            if self.buffer_meters > 0:
                region = point.buffer(self.buffer_meters)
                values = self.image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=10,
                    maxPixels=1e9
                )
            else:
                values = self.image.sample(
                    region=point,
                    scale=10,
                    geometries=False
                ).first()
                
                if values is not None:
                    values = values.toDictionary()
            
            result = values.getInfo()
            
            if result and any(v is not None for v in result.values()):
                embedding_vector = []
                for band in self.BAND_NAMES:
                    val = result.get(band, np.nan)
                    embedding_vector.append(val if val is not None else np.nan)
                
                embedding = np.array(embedding_vector, dtype=np.float32)
                
                # Verify embedding quality
                if np.isnan(embedding).sum() > 32:  # More than half NaN = bad
                    return None
                
                return embedding
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Failed to extract embedding at ({lon:.4f}, {lat:.4f}): {e}")
            return None
    
    def extract_batch_parallel(self, coordinates: List[tuple], n_workers: int = 10) -> Dict:
        """
        Extract embeddings for multiple coordinates in parallel.
        
        Args:
            coordinates: List of (lon, lat) tuples
            n_workers: Number of parallel workers
            
        Returns:
            Dictionary mapping (lon, lat) -> embedding array
        """
        results = {}
        failed = []
        
        logger.info(f"Extracting {len(coordinates)} AlphaEarth embeddings with {n_workers} workers...")
        
        def extract_wrapper(coord):
            lon, lat = coord
            emb = self.extract_embedding(lon, lat)
            return (coord, emb)
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(extract_wrapper, coord): coord for coord in coordinates}
            
            with tqdm(total=len(coordinates), desc="AlphaEarth extraction") as pbar:
                for future in as_completed(futures):
                    try:
                        coord, embedding = future.result(timeout=30)
                        if embedding is not None:
                            results[coord] = embedding
                        else:
                            failed.append(coord)
                    except Exception as e:
                        coord = futures[future]
                        logger.debug(f"Error extracting {coord}: {e}")
                        failed.append(coord)
                    
                    pbar.update(1)
        
        logger.info(f"✓ Extracted {len(results)}/{len(coordinates)} embeddings "
                   f"({len(results)/len(coordinates)*100:.1f}% success)")
        
        if failed:
            logger.warning(f"Failed to extract {len(failed)} embeddings")
        
        return results


# ==================== UTILS ====================

def to_raster(values, gy0, gx0, ny, nx, fill=np.nan, dtype=np.float32):
    A = np.full((ny, nx), fill, dtype=dtype)
    A[gy0, gx0] = values
    return A


# ==================== BUILDING GRID CONSTRUCTION ====================

def create_building_grid(csv_path: Path, 
                        region_name: str,
                        grid_size_deg: float = 0.01, 
                        min_buildings: int = 10,
                        lon_col: str = 'longitude',
                        lat_col: str = 'latitude',
                        area_col: str = 'area_in_meters',
                        confidence_col: str = 'confidence') -> pd.DataFrame:
    """
    Aggregate building data into spatial grid cells.
    
    Args:
        csv_path: Path to OpenBuildings CSV
        region_name: Name identifier for this region
        grid_size_deg: Grid cell size in degrees
        min_buildings: Minimum buildings per cell
        lon_col, lat_col, area_col, confidence_col: Column names in CSV
        
    Returns:
        DataFrame with aggregated building statistics per cell
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {region_name}")
    logger.info(f"{'='*60}")
    
    # Load buildings
    logger.info(f"Loading CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)
    logger.info(f"  ✓ Loaded {len(df):,} buildings")
    
    # Verify columns exist
    required_cols = [lon_col, lat_col, area_col, confidence_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    
    # Rename to standard names
    df = df.rename(columns={
        lon_col: 'longitude',
        lat_col: 'latitude',
        area_col: 'area_in_meters',
        confidence_col: 'confidence'
    })
    
    # Basic stats
    logger.info(f"  Lat range: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
    logger.info(f"  Lon range: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")
    logger.info(f"  Area range: [{df['area_in_meters'].min():.2f}, {df['area_in_meters'].max():.2f}] m²")
    logger.info(f"  Confidence: {df['confidence'].mean():.3f} ± {df['confidence'].std():.3f}")
    
    # Create grid bounds
    lon_min = np.floor(df['longitude'].min() / grid_size_deg) * grid_size_deg
    lon_max = np.ceil(df['longitude'].max() / grid_size_deg) * grid_size_deg
    lat_min = np.floor(df['latitude'].min() / grid_size_deg) * grid_size_deg
    lat_max = np.ceil(df['latitude'].max() / grid_size_deg) * grid_size_deg
    
    logger.info(f"\nCreating grid (size={grid_size_deg}°)...")
    lon_edges = np.arange(lon_min, lon_max + grid_size_deg, grid_size_deg)
    lat_edges = np.arange(lat_min, lat_max + grid_size_deg, grid_size_deg)
    
    logger.info(f"  Grid: {len(lon_edges)-1} x {len(lat_edges)-1} cells")
    logger.info(f"  Total: {(len(lon_edges)-1) * (len(lat_edges)-1):,} potential cells")
    
    # Assign buildings to grid cells
    df['grid_lon_idx'] = pd.cut(df['longitude'], lon_edges, labels=False, include_lowest=True)
    df['grid_lat_idx'] = pd.cut(df['latitude'], lat_edges, labels=False, include_lowest=True)
    
    # Remove edge cases
    df = df.dropna(subset=['grid_lon_idx', 'grid_lat_idx'])
    
    # Compute cell centers
    df['cell_lon'] = lon_edges[df['grid_lon_idx'].astype(int)] + grid_size_deg / 2
    df['cell_lat'] = lat_edges[df['grid_lat_idx'].astype(int)] + grid_size_deg / 2
    
    # Aggregate per cell
    logger.info("Aggregating statistics per cell...")
    grid_stats = df.groupby(['grid_lon_idx', 'grid_lat_idx']).agg({
        'area_in_meters': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'confidence': ['mean', 'std'],
        'cell_lon': 'first',
        'cell_lat': 'first',
    }).reset_index()
    
    # Flatten columns
    grid_stats.columns = [
        'grid_lon_idx', 'grid_lat_idx',
        'building_count', 'total_area_m2', 'mean_area_m2', 'std_area_m2', 
        'min_area_m2', 'max_area_m2',
        'mean_confidence', 'std_confidence',
        'lon', 'lat'
    ]
    
    # Fill NaN std with 0 (single building cells)
    grid_stats['std_area_m2'] = grid_stats['std_area_m2'].fillna(0)
    grid_stats['std_confidence'] = grid_stats['std_confidence'].fillna(0)
    
    # Compute derived metrics
    logger.info("Computing derived metrics...")
    
    # Cell area in km² (approximate, varies by latitude)
    cell_area_km2 = (grid_size_deg * 111.32 * np.cos(np.radians(grid_stats['lat']))) * (grid_size_deg * 111.32)
    grid_stats['cell_area_km2'] = cell_area_km2
    
    # Building density (buildings per km²)
    grid_stats['building_density'] = grid_stats['building_count'] / cell_area_km2
    
    # Coverage ratio (building footprint / cell area)
    grid_stats['coverage_ratio'] = (grid_stats['total_area_m2'] / 1e6) / cell_area_km2
    
    # Log-transformed targets (better distributions)
    grid_stats['log_count'] = np.log1p(grid_stats['building_count'])
    grid_stats['log_density'] = np.log1p(grid_stats['building_density'])
    grid_stats['log_total_area'] = np.log1p(grid_stats['total_area_m2'])
    
    # Filter by minimum buildings
    logger.info(f"Filtering cells (min_buildings >= {min_buildings})...")
    logger.info(f"  Before: {len(grid_stats):,} cells")
    grid_stats = grid_stats[grid_stats['building_count'] >= min_buildings].copy()
    logger.info(f"  After: {len(grid_stats):,} cells")
    
    # Add region identifier
    grid_stats['region'] = region_name
    
    logger.info(f"\n✓ Created {len(grid_stats):,} grid cells for {region_name}")
    logger.info(f"  Buildings/cell: {grid_stats['building_count'].mean():.1f} ± "
               f"{grid_stats['building_count'].std():.1f}")
    logger.info(f"  Density: {grid_stats['building_density'].mean():.1f} ± "
               f"{grid_stats['building_density'].std():.1f} buildings/km²")
    logger.info(f"  Confidence: {grid_stats['mean_confidence'].mean():.3f} ± "
               f"{grid_stats['mean_confidence'].std():.3f}")
    
    return grid_stats


# ==================== QUALITY FILTERING ====================

def filter_by_confidence(grid_df: pd.DataFrame,
                        min_confidence: float = 0.7) -> pd.DataFrame:
    """Filter grid cells by building confidence BEFORE embedding extraction."""
    logger.info(f"\n{'='*60}")
    logger.info("CONFIDENCE FILTERING (Pre-Embedding)")
    logger.info(f"{'='*60}")
    
    initial_count = len(grid_df)
    logger.info(f"Initial cells: {initial_count:,}")
    
    mask_confidence = grid_df['mean_confidence'] >= min_confidence
    grid_df_filtered = grid_df[mask_confidence].copy().reset_index(drop=True)
    
    logger.info(f"  Confidence >= {min_confidence}: {len(grid_df_filtered):,} cells "
               f"({len(grid_df_filtered)/initial_count*100:.1f}%)")
    logger.info(f"  Removed: {initial_count - len(grid_df_filtered):,} cells")
    
    return grid_df_filtered


# ==================== STRATIFIED SPATIAL SAMPLING ====================

def stratified_spatial_sample(grid_df: pd.DataFrame,
                              target_samples: int = 30000,
                              n_bins: int = 10,
                              random_seed: int = 42) -> pd.DataFrame:
    """
    Perform stratified spatial sampling to get representative samples.
    """
    logger.info(f"\n{'='*60}")
    logger.info("STRATIFIED SPATIAL SAMPLING")
    logger.info(f"{'='*60}")
    
    initial_count = len(grid_df)
    logger.info(f"Available cells: {initial_count:,}")
    logger.info(f"Target samples: {target_samples:,}")
    
    if initial_count <= target_samples:
        logger.info(f"⚠ Available cells <= target, using all {initial_count:,} cells")
        return grid_df.copy()
    
    rng = np.random.RandomState(random_seed)
    
    # Create spatial bins
    try:
        lon_bins = pd.qcut(grid_df['lon'], q=n_bins, labels=False, duplicates='drop')
        lat_bins = pd.qcut(grid_df['lat'], q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        logger.warning("qcut failed, falling back to cut for binning")
        lon_bins = pd.cut(grid_df['lon'], bins=n_bins, labels=False)
        lat_bins = pd.cut(grid_df['lat'], bins=n_bins, labels=False)
    
    grid_df['spatial_bin'] = lon_bins.astype(str) + '_' + lat_bins.astype(str)
    
    # Count samples per bin
    bin_counts = grid_df['spatial_bin'].value_counts()
    n_bins_populated = len(bin_counts)
    
    logger.info(f"  Spatial grid: {n_bins}x{n_bins} = {n_bins*n_bins} bins")
    logger.info(f"  Populated bins: {n_bins_populated}")
    logger.info(f"  Cells per bin: {bin_counts.mean():.1f} ± {bin_counts.std():.1f}")
    
    # Proportional sampling from each bin
    sampled_indices = []
    
    for bin_id, group in grid_df.groupby('spatial_bin'):
        bin_size = len(group)
        n_samples_bin = int(np.ceil(target_samples * (bin_size / initial_count)))
        n_samples_bin = min(n_samples_bin, bin_size)
        
        if n_samples_bin > 0:
            sample_indices = rng.choice(group.index, size=n_samples_bin, replace=False)
            sampled_indices.extend(sample_indices)
    
    # If we have too many samples, randomly drop excess
    if len(sampled_indices) > target_samples:
        sampled_indices = rng.choice(sampled_indices, size=target_samples, replace=False)
    
    sampled_df = grid_df.loc[sampled_indices].copy().reset_index(drop=True)
    
    # Verify spatial coverage
    sampled_bin_counts = sampled_df['spatial_bin'].value_counts()
    coverage = len(sampled_bin_counts) / n_bins_populated
    
    logger.info(f"\n✓ Sampling complete")
    logger.info(f"  Sampled: {len(sampled_df):,} cells")
    logger.info(f"  Spatial coverage: {len(sampled_bin_counts)}/{n_bins_populated} bins ({coverage*100:.1f}%)")
    
    return sampled_df


# ==================== ALPHAEARTH EMBEDDING INTEGRATION ====================

def add_alphaearth_embeddings(grid_df: pd.DataFrame, 
                              extractor: AlphaEarthExtractor,
                              n_workers: int = 10) -> pd.DataFrame:
    """Add AlphaEarth embeddings to grid cells."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Extracting AlphaEarth embeddings for {len(grid_df):,} cells")
    logger.info(f"{'='*60}")
    
    coordinates = [(row['lon'], row['lat']) for _, row in grid_df.iterrows()]
    
    embeddings_dict = extractor.extract_batch_parallel(coordinates, n_workers=n_workers)
    
    embeddings_list = []
    valid_indices = []
    
    for idx, (_, row) in enumerate(grid_df.iterrows()):
        coord = (row['lon'], row['lat'])
        if coord in embeddings_dict:
            embeddings_list.append(embeddings_dict[coord])
            valid_indices.append(idx)
    
    logger.info(f"Filtering to cells with valid embeddings...")
    logger.info(f"  Before: {len(grid_df):,} cells")
    
    grid_df = grid_df.iloc[valid_indices].copy().reset_index(drop=True)
    valid_embeddings = [embeddings_list[i] for i in valid_indices]
    
    logger.info(f"  After: {len(grid_df):,} cells")
    
    # Drop last dimension (A64 is always NaN) → 63-dim
    logger.info(f"Dropping last NaN dimension (A64)...")
    valid_embeddings_63 = [emb[:-1] for emb in valid_embeddings]
    
    # Store 63-dim embeddings as bytes (parquet-friendly)
    grid_df['embedding'] = [emb.tobytes() for emb in valid_embeddings_63]
    
    # Compute embedding statistics on 63-dim
    embeddings_array = np.stack(valid_embeddings_63, axis=0)
    
    nan_counts = np.isnan(embeddings_array).sum(axis=1)
    grid_df['embedding_nan_count'] = nan_counts
    grid_df['embedding_nan_ratio'] = nan_counts / 63.0  # 63-dim
    
    embeddings_clean = np.nan_to_num(embeddings_array, nan=0.0)
    grid_df['embedding_norm'] = np.linalg.norm(embeddings_clean, axis=1)
    grid_df['embedding_mean'] = np.nanmean(embeddings_array, axis=1)
    grid_df['embedding_std'] = np.nanstd(embeddings_array, axis=1)
    
    logger.info(f"✓ Added AlphaEarth embeddings (63-dim, dropped A64)")
    logger.info(f"  Shape: {embeddings_array.shape}")
    logger.info(f"  Norm: {grid_df['embedding_norm'].mean():.3f} ± "
               f"{grid_df['embedding_norm'].std():.3f}")
    logger.info(f"  NaN ratio: {grid_df['embedding_nan_ratio'].mean():.3f} ± "
               f"{grid_df['embedding_nan_ratio'].std():.3f}")
    
    return grid_df


# ==================== EMBEDDING QUALITY FILTERING ====================

def filter_embedding_quality(grid_df: pd.DataFrame,
                            max_nan_ratio: float = 0.3) -> pd.DataFrame:
    """Filter by embedding quality AFTER extraction."""
    logger.info(f"\n{'='*60}")
    logger.info("EMBEDDING QUALITY FILTERING (Post-Embedding)")
    logger.info(f"{'='*60}")
    
    initial_count = len(grid_df)
    logger.info(f"Initial cells: {initial_count:,}")
    
    mask_embedding = grid_df['embedding_nan_ratio'] <= max_nan_ratio
    grid_df_filtered = grid_df[mask_embedding].copy().reset_index(drop=True)
    
    logger.info(f"  Embedding NaN ratio <= {max_nan_ratio}: {len(grid_df_filtered):,} cells "
               f"({len(grid_df_filtered)/initial_count*100:.1f}%)")
    logger.info(f"  Removed: {initial_count - len(grid_df_filtered):,} cells")
    
    return grid_df_filtered


# ==================== FINAL SAMPLING ====================

def final_sample_to_target(grid_df: pd.DataFrame,
                          target_samples: int = 30000,
                          random_seed: int = 42) -> pd.DataFrame:
    """Final sampling to exact target count if needed."""
    if len(grid_df) <= target_samples:
        logger.info(f"✓ Already at or below target ({len(grid_df):,} <= {target_samples:,})")
        return grid_df
    
    logger.info(f"\nFinal sampling to exactly {target_samples:,} cells...")
    logger.info(f"  Current: {len(grid_df):,} cells")
    
    rng = np.random.RandomState(random_seed)
    sampled_indices = rng.choice(grid_df.index, size=target_samples, replace=False)
    sampled_df = grid_df.loc[sampled_indices].copy().reset_index(drop=True)
    
    logger.info(f"✓ Sampled to exactly {len(sampled_df):,} cells")
    
    return sampled_df


# ==================== DATA SPLITS ====================

def create_splits(grid_df: pd.DataFrame, 
                 train_frac: float = 0.70,
                 val_frac: float = 0.15,
                 test_frac: float = 0.15,
                 random_seed: int = 42) -> pd.DataFrame:
    """Create spatially-stratified train/val/test splits."""
    
    logger.info("\n" + "="*60)
    logger.info("Creating train/val/test splits...")
    logger.info("="*60)
    
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    
    rng = np.random.RandomState(random_seed)
    
    if 'spatial_bin' not in grid_df.columns:
        try:
            lon_bins = pd.qcut(grid_df['lon'], q=5, labels=False, duplicates='drop')
            lat_bins = pd.qcut(grid_df['lat'], q=5, labels=False, duplicates='drop')
            grid_df['spatial_bin'] = lon_bins.astype(str) + '_' + lat_bins.astype(str)
        except:
            grid_df['spatial_bin'] = '0_0'
    
    splits = []
    for bin_id, group in grid_df.groupby('spatial_bin'):
        n = len(group)
        if n < 3:
            splits.extend(['train'] * n)
            continue
        
        indices = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        
        bin_splits = ['train'] * n_train + ['val'] * n_val + ['test'] * (n - n_train - n_val)
        bin_splits = [bin_splits[i] for i in indices]
        splits.extend(bin_splits)
    
    grid_df['split'] = splits
    
    train_count = (grid_df['split'] == 'train').sum()
    val_count = (grid_df['split'] == 'val').sum()
    test_count = (grid_df['split'] == 'test').sum()
    
    logger.info(f"  Train: {train_count:,} ({train_count/len(grid_df)*100:.1f}%)")
    logger.info(f"  Val:   {val_count:,} ({val_count/len(grid_df)*100:.1f}%)")
    logger.info(f"  Test:  {test_count:,} ({test_count/len(grid_df)*100:.1f}%)")
    
    return grid_df


# ==================== METADATA ====================

def save_region_metadata(grid_df: pd.DataFrame, region_name: str, output_dir: Path, grid_size_deg: float):
    """Save region bounding box metadata."""
    metadata = {
        'region': region_name,
        'n_samples': len(grid_df),
        'embedding_dim': 63,  # Add this line
        'bounds': [
            float(grid_df['lon'].min()),
            float(grid_df['lat'].min()),
            float(grid_df['lon'].max()),
            float(grid_df['lat'].max()),
        ],
        'center': [
            float(grid_df['lon'].mean()),
            float(grid_df['lat'].mean()),
        ],
        'grid_size_deg': grid_size_deg,
        'grid_size_km': grid_size_deg * 111.32,
        'statistics': {
            'building_count_mean': float(grid_df['building_count'].mean()),
            'building_count_std': float(grid_df['building_count'].std()),
            'building_density_mean': float(grid_df['building_density'].mean()),
            'building_density_std': float(grid_df['building_density'].std()),
            'mean_confidence_mean': float(grid_df['mean_confidence'].mean()),
            'mean_confidence_std': float(grid_df['mean_confidence'].std()),
            'embedding_norm_mean': float(grid_df['embedding_norm'].mean()),
            'embedding_norm_std': float(grid_df['embedding_norm'].std()),
        },
        'splits': {
            'train': int((grid_df['split'] == 'train').sum()),
            'val': int((grid_df['split'] == 'val').sum()),
            'test': int((grid_df['split'] == 'test').sum()),
        }
    }
    
    metadata_path = output_dir / f"{region_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Saved region metadata: {metadata_path}")
    
    return metadata


# ==================== MAIN ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Create OpenBuildings + AlphaEarth dataset')
    
    # Input - either specify CSVs explicitly or scan a directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csvs', nargs='+', metavar='REGION=PATH',
                           help='Region CSV mappings: region1=/path/to/file1.csv region2=/path/to/file2.csv')
    input_group.add_argument('--csv-dir', type=Path,
                           help='Directory containing OpenBuildings CSV files (will use filename as region name)')
    
    # Output
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for processed datasets')
    
    # Grid parameters
    parser.add_argument('--grid-size-deg', type=float, default=0.01,
                       help='Grid cell size in degrees (default: 0.01 ≈ 1.1km)')
    parser.add_argument('--min-buildings', type=int, default=10,
                       help='Minimum buildings per cell (default: 10)')
    
    # CSV column names (in case they differ)
    parser.add_argument('--lon-col', default='longitude',
                       help='Longitude column name (default: longitude)')
    parser.add_argument('--lat-col', default='latitude',
                       help='Latitude column name (default: latitude)')
    parser.add_argument('--area-col', default='area_in_meters',
                       help='Area column name (default: area_in_meters)')
    parser.add_argument('--confidence-col', default='confidence',
                       help='Confidence column name (default: confidence)')
    
    # Quality filtering
    parser.add_argument('--min-confidence', type=float, default=0.7,
                       help='Minimum mean confidence per cell (default: 0.7)')
    parser.add_argument('--max-nan-ratio', type=float, default=0.3,
                       help='Maximum NaN ratio in embeddings (default: 0.3)')
    
    # Sampling
    parser.add_argument('--samples-per-region', type=int, default=30000,
                       help='Target samples per region (default: 30000)')
    parser.add_argument('--sample-buffer', type=float, default=1.3,
                       help='Buffer ratio for sampling (default: 1.3 = 30%% extra)')
    parser.add_argument('--spatial-bins', type=int, default=10,
                       help='Spatial grid bins for stratified sampling (default: 10)')
    
    # AlphaEarth
    parser.add_argument('--alphaearth-year', type=int, default=2024,
                       help='AlphaEarth year (default: 2024)')
    parser.add_argument('--alphaearth-buffer', type=int, default=10,
                       help='AlphaEarth buffer in meters (default: 10)')
    
    # Processing
    parser.add_argument('--n-workers', type=int, default=8,
                       help='Parallel workers for embedding extraction (default: 8)')
    
    # Splits
    parser.add_argument('--train-frac', type=float, default=0.70)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--test-frac', type=float, default=0.15)
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable WandB logging')
    parser.add_argument('--wandb-project', default='buildings_alphaearth',
                       help='WandB project name')
    parser.add_argument('--ee-project', type=str, required=True,
                   help='Google Earth Engine project ID (required)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("\n" + "="*60)
    logger.info("OpenBuildings + AlphaEarth Dataset Creation")
    logger.info("Generalized Pipeline")
    logger.info("="*60)
    
    # Parse input CSVs
    region_csvs = {}
    if args.csvs:
        # Explicit region=path mappings
        for item in args.csvs:
            if '=' not in item:
                raise ValueError(f"CSV must be in format 'region=path', got: {item}")
            region, path = item.split('=', 1)
            region_csvs[region] = Path(path)
    else:
        # Scan directory
        csv_files = list(args.csv_dir.glob('*.csv'))
        if not csv_files:
            raise ValueError(f"No CSV files found in {args.csv_dir}")
        for csv_file in csv_files:
            region_name = csv_file.stem  # Use filename without extension as region name
            region_csvs[region_name] = csv_file
    
    logger.info(f"\nFound {len(region_csvs)} region(s):")
    for region, path in region_csvs.items():
        logger.info(f"  {region}: {path}")
    
    # Verify all CSVs exist
    for region, path in region_csvs.items():
        if not path.exists():
            raise FileNotFoundError(f"CSV not found for region '{region}': {path}")
    
    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"dataset_{len(region_csvs)}regions_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Initialize Earth Engine
    initialize_earth_engine(args.ee_project)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nOutput directory: {args.output_dir}")
    
    # Initialize AlphaEarth extractor
    extractor = AlphaEarthExtractor(
        year=args.alphaearth_year,
        buffer_meters=args.alphaearth_buffer
    )
    
    # Process each region
    all_regions_data = {}
    target_with_buffer = int(args.samples_per_region * args.sample_buffer)
    
    for region_name, csv_path in region_csvs.items():
        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING REGION: {region_name}")
        logger.info("="*80)
        
        # Step 1: Create building grid
        grid_df = create_building_grid(
            csv_path,
            region_name,
            grid_size_deg=args.grid_size_deg,
            min_buildings=args.min_buildings,
            lon_col=args.lon_col,
            lat_col=args.lat_col,
            area_col=args.area_col,
            confidence_col=args.confidence_col
        )
        
        # Step 2: Filter by confidence
        grid_df = filter_by_confidence(grid_df, min_confidence=args.min_confidence)
        
        # Step 3: Stratified spatial sample
        logger.info(f"\nSampling {target_with_buffer:,} cells ({args.sample_buffer:.1%} buffer)")
        grid_df = stratified_spatial_sample(
            grid_df,
            target_samples=target_with_buffer,
            n_bins=args.spatial_bins,
            random_seed=args.seed
        )
        
        # Step 4: Extract embeddings
        grid_df = add_alphaearth_embeddings(grid_df, extractor, n_workers=args.n_workers)
        
        # Step 5: Filter by embedding quality
        grid_df = filter_embedding_quality(grid_df, max_nan_ratio=args.max_nan_ratio)
        
        # Step 6: Final sample to exact target
        grid_df = final_sample_to_target(
            grid_df,
            target_samples=args.samples_per_region,
            random_seed=args.seed
        )
        
        # Step 7: Create splits
        grid_df = create_splits(
            grid_df,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_seed=args.seed
        )
        
        # Save region data
        output_path = args.output_dir / f"{region_name}_grid.parquet"
        grid_df.to_parquet(output_path, index=False)
        logger.info(f"\n✓ Saved: {output_path}")
        logger.info(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")
        
        # Save metadata
        metadata = save_region_metadata(grid_df, region_name, args.output_dir, args.grid_size_deg)
        
        all_regions_data[region_name] = {
            'df': grid_df,
            'metadata': metadata,
            'path': output_path
        }
    
    # Save combined splits info
    splits_info = {}
    for region_name, data in all_regions_data.items():
        df = data['df']
        splits_info[region_name] = {
            'total': int(len(df)),
            'train': int((df['split'] == 'train').sum()),
            'val': int((df['split'] == 'val').sum()),
            'test': int((df['split'] == 'test').sum()),
        }
    
    splits_path = args.output_dir / "splits.json"
    with open(splits_path, 'w') as f:
        json.dump(splits_info, f, indent=2)
    logger.info(f"\n✓ Saved combined splits: {splits_path}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("DATASET CREATION COMPLETE ✓")
    logger.info("="*60)
    
    total_samples = 0
    for region_name, data in all_regions_data.items():
        df = data['df']
        n_total = len(df)
        n_train = (df['split'] == 'train').sum()
        n_val = (df['split'] == 'val').sum()
        n_test = (df['split'] == 'test').sum()
        
        logger.info(f"\n{region_name}: {n_total:,} cells")
        logger.info(f"  Train: {n_train:,}")
        logger.info(f"  Val:   {n_val:,}")
        logger.info(f"  Test:  {n_test:,}")
        
        total_samples += n_total
    
    logger.info(f"\nTotal samples across all regions: {total_samples:,}")
    logger.info(f"Grid size: {args.grid_size_deg}° (~{args.grid_size_deg*111.32:.2f} km)")
    logger.info(f"Output directory: {args.output_dir}")
    
    if not args.no_wandb:
        wandb.log({
            'total_samples': total_samples,
            'n_regions': len(region_csvs),
            'splits': splits_info
        })
        logger.info(f"WandB: {wandb.run.url}")
        wandb.finish()
    
    logger.info("="*60)


if __name__ == "__main__":
    main()