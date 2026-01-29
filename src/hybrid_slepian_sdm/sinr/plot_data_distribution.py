#!/usr/bin/env python3
"""
Visualize the spatial distribution of iNat training data used by SINR.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def main():
    # Load paths
    with open('paths.json', 'r') as f:
        paths = json.load(f)

    # Load data (just lon/lat columns to save memory)
    print("Loading training data...")
    df = pd.read_csv(
        paths['train'] + 'geo_prior_train.csv',
        usecols=['longitude', 'latitude', 'observed_on']
    )

    print(f"Total observations: {len(df):,}")

    # Extract year range
    df['year'] = df['observed_on'].str[:4].astype(int)
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Observations per year:\n{df['year'].value_counts().sort_index()}")

    lon = df['longitude'].values
    lat = df['latitude'].values

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Global hexbin density
    ax = axes[0]
    hb = ax.hexbin(lon, lat, gridsize=200, cmap='viridis', mincnt=1,
                   bins='log', extent=[-180, 180, -90, 90])
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'iNat Training Data Distribution\n({len(df):,} observations)')
    ax.set_aspect('equal')
    plt.colorbar(hb, ax=ax, label='Count (log scale)')

    # Add coastlines approximation (simple box)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)

    # Right: Histogram of latitude distribution
    ax = axes[1]
    ax.hist(lat, bins=180, orientation='horizontal', color='steelblue', alpha=0.7)
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Count')
    ax.set_title('Latitude Distribution')
    ax.set_ylim(-90, 90)

    plt.tight_layout()
    plt.savefig('inat_data_distribution.png', dpi=150, bbox_inches='tight')
    print("\nSaved: inat_data_distribution.png")

    # Print some stats
    print(f"\nGeographic coverage:")
    print(f"  Longitude: [{lon.min():.2f}, {lon.max():.2f}]")
    print(f"  Latitude:  [{lat.min():.2f}, {lat.max():.2f}]")

    # Check concentration in US/Europe (our Slepian cap regions)
    us_mask = (lon > -130) & (lon < -60) & (lat > 20) & (lat < 55)
    europe_mask = (lon > -15) & (lon < 40) & (lat > 35) & (lat < 70)

    print(f"\nRegional breakdown:")
    print(f"  US region:     {us_mask.sum():,} ({100*us_mask.mean():.1f}%)")
    print(f"  Europe region: {europe_mask.sum():,} ({100*europe_mask.mean():.1f}%)")
    print(f"  Other:         {(~us_mask & ~europe_mask).sum():,} ({100*(~us_mask & ~europe_mask).mean():.1f}%)")

if __name__ == '__main__':
    main()
