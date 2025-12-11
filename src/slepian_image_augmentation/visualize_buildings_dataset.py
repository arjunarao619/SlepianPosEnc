#!/usr/bin/env python3
"""
Create better visualizations with wider map views
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Try cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠ cartopy not available, using basic matplotlib")

# Paths
DATA_DIR = Path("/projects/arra4944/SlepianEncoder/datasets/buildings_slepian")
maharashtra_path = DATA_DIR / "maharashtra_grid.parquet"
brazil_path = DATA_DIR / "brazil_grid.parquet"

print("Loading fixed datasets...")
maharashtra_df = pd.read_parquet(maharashtra_path)
brazil_df = pd.read_parquet(brazil_path)

print(f"Maharashtra: {len(maharashtra_df):,} samples")
print(f"Brazil: {len(brazil_df):,} samples")

# Verify embedding dimension
test_emb_maha = np.frombuffer(maharashtra_df.iloc[0]['embedding'], dtype=np.float32)
test_emb_brazil = np.frombuffer(brazil_df.iloc[0]['embedding'], dtype=np.float32)
print(f"\nEmbedding dimensions:")
print(f"  Maharashtra: {len(test_emb_maha)}")
print(f"  Brazil: {len(test_emb_brazil)}")

# ===== BETTER SPATIAL MAPS =====
print("\n" + "="*60)
print("CREATING SPATIAL MAPS")
print("="*60)

if HAS_CARTOPY:
    print("Using Cartopy for geographic context...")
    
    fig = plt.figure(figsize=(20, 10))
    
    for idx, (name, df) in enumerate([('Maharashtra', maharashtra_df), ('Brazil', brazil_df)]):
        # Get bounds with generous padding for context
        lon_min, lon_max = df['lon'].min(), df['lon'].max()
        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        
        # Add 20% padding for geographic context
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        lon_pad = lon_range * 0.2
        lat_pad = lat_range * 0.2
        
        extent = [lon_min - lon_pad, lon_max + lon_pad, 
                  lat_min - lat_pad, lat_max + lat_pad]
        
        print(f"{name} extent: {extent}")
        
        # Building density map
        ax = fig.add_subplot(2, 2, idx*2 + 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add geographic features
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', 
                      edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=':', 
                      edgecolor='gray', alpha=0.5, zorder=1)
        
        # Plot data
        scatter = ax.scatter(df['lon'], df['lat'],
                           c=np.log1p(df['building_density']),
                           s=2, alpha=0.7, cmap='YlOrRd',
                           transform=ccrs.PlateCarree(),
                           zorder=2)
        
        ax.set_title(f'{name} Building Density\n{len(df):,} cells', 
                    fontweight='bold', fontsize=14)
        gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        plt.colorbar(scatter, ax=ax, label='log(buildings/km²)', 
                    shrink=0.7, pad=0.02)
        
        # Building count map
        ax = fig.add_subplot(2, 2, idx*2 + 2, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add geographic features
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', 
                      edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=':', 
                      edgecolor='gray', alpha=0.5, zorder=1)
        
        # Plot data
        scatter = ax.scatter(df['lon'], df['lat'],
                           c=df['building_count'],
                           s=2, alpha=0.7, cmap='viridis',
                           vmin=10, vmax=100,
                           transform=ccrs.PlateCarree(),
                           zorder=2)
        
        ax.set_title(f'{name} Building Count', 
                    fontweight='bold', fontsize=14)
        gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        plt.colorbar(scatter, ax=ax, label='Buildings per cell', 
                    shrink=0.7, pad=0.02)
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / "eda_spatial_maps_wide.png", dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {DATA_DIR / 'eda_spatial_maps_wide.png'}")
    plt.close()

else:
    # Fallback to basic matplotlib
    print("Using basic matplotlib...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    for idx, (name, df) in enumerate([('Maharashtra', maharashtra_df), ('Brazil', brazil_df)]):
        # Density
        ax = axes[idx, 0]
        scatter = ax.scatter(df['lon'], df['lat'],
                           c=np.log1p(df['building_density']),
                           s=3, alpha=0.6, cmap='YlOrRd')
        ax.set_title(f'{name} Building Density\n({len(df):,} cells)', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='log(buildings/km²)')
        ax.grid(alpha=0.3, linestyle='--')
        
        # Count
        ax = axes[idx, 1]
        scatter = ax.scatter(df['lon'], df['lat'],
                           c=df['building_count'],
                           s=3, alpha=0.6, cmap='viridis', vmin=10, vmax=100)
        ax.set_title(f'{name} Building Count', fontweight='bold', fontsize=12)
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Buildings per cell')
        ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / "eda_spatial_maps_wide.png", dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {DATA_DIR / 'eda_spatial_maps_wide.png'}")
    plt.close()

# ===== FIXED DISTRIBUTIONS =====
print("Creating distribution plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

variables = [
    ('building_count', 'Building Count'),
    ('building_density', 'Density (per km²)'),
    ('total_area_m2', 'Total Area (m²)'),
    ('mean_area_m2', 'Mean Area (m²)'),
    ('coverage_ratio', 'Coverage Ratio'),
    ('embedding_norm', 'Embedding Norm (63-dim)'),
]

for idx, (var, label) in enumerate(variables):
    ax = axes[idx // 3, idx % 3]
    
    ax.hist(maharashtra_df[var], bins=50, alpha=0.6, label='Maharashtra', 
           color='blue', density=True)
    ax.hist(brazil_df[var], bins=50, alpha=0.6, label='Brazil', 
           color='red', density=True)
    
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Distribution Comparison (Fixed Embeddings)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(DATA_DIR / "eda_distributions_fixed.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {DATA_DIR / 'eda_distributions_fixed.png'}")
plt.close()

# ===== SUMMARY =====
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for name, df in [('Maharashtra', maharashtra_df), ('Brazil', brazil_df)]:
    print(f"\n{name}:")
    print(f"  Samples: {len(df):,}")
    print(f"  Embedding dim: 63")
    print(f"  Bounds: lon=[{df['lon'].min():.2f}, {df['lon'].max():.2f}], "
          f"lat=[{df['lat'].min():.2f}, {df['lat'].max():.2f}]")
    print(f"  Building count: {df['building_count'].mean():.1f} ± {df['building_count'].std():.1f}")
    print(f"  Building density: {df['building_density'].mean():.1f} ± {df['building_density'].std():.1f} per km²")
    print(f"  Confidence: {df['mean_confidence'].mean():.3f} ± {df['mean_confidence'].std():.3f}")
    print(f"  Embedding norm: {df['embedding_norm'].mean():.3f} ± {df['embedding_norm'].std():.3f}")
    print(f"  Splits: Train={( df['split']=='train').sum():,}, "
          f"Val={(df['split']=='val').sum():,}, Test={(df['split']=='test').sum():,}")

print("\n" + "="*60)
print("✓ All visualizations complete!")
print("="*60)