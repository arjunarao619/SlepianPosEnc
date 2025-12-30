"""
Visualize Slepian basis on 2D world maps.
Left: Global spherical harmonic mode
Right: Regional Slepian modes (US + Europe) concentrated in spherical caps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import utils

op_dir = 'visualizations/'
os.makedirs(op_dir, exist_ok=True)

slepian_params = {
    'slepian_L_global': 10,
    'slepian_L_regional': 40,
    'slepian_lambda_thresh': 0.1,
    'slepian_grid_resolution': 0.5,
    'slepian_cache_dir': './cache/slepian',
    'slepian_caps': [
        {'name': 'us', 'center': (-95.0, 40.0), 'radius': 25.0},
        {'name': 'europe', 'center': (10.0, 50.0), 'radius': 20.0},
    ]
}

def draw_cap_circle(ax, center_lon, center_lat, radius_deg, **kwargs):
    """Draw a circle representing the spherical cap boundary."""
    theta = np.linspace(0, 2*np.pi, 100)
    radius_rad = np.radians(radius_deg)
    center_lat_rad = np.radians(center_lat)

    lons = []
    lats = []
    for t in theta:
        lat = np.degrees(np.arcsin(
            np.sin(center_lat_rad) * np.cos(radius_rad) +
            np.cos(center_lat_rad) * np.sin(radius_rad) * np.cos(t)
        ))
        dlon = np.degrees(np.arctan2(
            np.sin(t) * np.sin(radius_rad) * np.cos(center_lat_rad),
            np.cos(radius_rad) - np.sin(center_lat_rad) * np.sin(np.radians(lat))
        ))
        lon = center_lon + dlon
        lons.append(lon)
        lats.append(lat)

    ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)

def main():
    # Load Slepian raster
    cache_path = utils.get_slepian_cache_path(slepian_params, slepian_params['slepian_cache_dir'])

    if not os.path.exists(cache_path):
        result = utils.compute_slepian_raster(
            L_global=slepian_params['slepian_L_global'],
            L_regional=slepian_params['slepian_L_regional'],
            caps=slepian_params['slepian_caps'],
            lambda_thresh=slepian_params['slepian_lambda_thresh'],
            grid_resolution=slepian_params['slepian_grid_resolution'],
            cache_path=cache_path, verbose=True)
        raster, meta = result['raster'].numpy(), result['metadata']
    else:
        data = utils.load_slepian_raster(cache_path, verbose=True)
        raster, meta = data['raster'].numpy(), data['metadata']

    global_dim = meta['global_dim']
    regional_dims = meta['regional_dims']
    caps = meta['caps']

    # Split into global SH and regional Slepian features
    global_sh = raster[:, :, :global_dim]
    regional_start = global_dim
    regional = {}
    for cap in caps:
        dim = regional_dims[cap['name']]
        regional[cap['name']] = raster[:, :, regional_start:regional_start + dim]
        regional_start += dim

    # Select modes to display
    sh_mode_idx = 3  # A nice global SH mode

    # Get cap info
    us_cap = next(c for c in caps if c['name'] == 'us')
    europe_cap = next(c for c in caps if c['name'] == 'europe')

    # Use mid-range modes: good structure but still well-concentrated (λ ≈ 0.5-0.7)
    # Roughly 1/3 of the way through the modes
    us_mode_idx = regional_dims['us'] // 3
    europe_mode_idx = regional_dims['europe'] // 3

    print(f"US Slepian modes available: {regional_dims['us']}")
    print(f"Europe Slepian modes available: {regional_dims['europe']}")
    print(f"Displaying US mode {us_mode_idx}, Europe mode {europe_mode_idx} (mid-range, well-concentrated)")

    # Get the mode data
    sh_mode = global_sh[:, :, sh_mode_idx]
    us_slepian = regional['us'][:, :, us_mode_idx]
    europe_slepian = regional['europe'][:, :, europe_mode_idx]

    # Normalize each separately before combining (so both are visible)
    us_slepian = us_slepian / np.abs(us_slepian).max()
    europe_slepian = europe_slepian / np.abs(europe_slepian).max()
    combined_slepian = us_slepian + europe_slepian

    # Normalize to [-1, 1]
    sh_mode = sh_mode / np.abs(sh_mode).max()
    combined_slepian = combined_slepian / np.abs(combined_slepian).max()

    # Create lat/lon grids
    lons = np.linspace(-180, 180, raster.shape[1])
    lats = np.linspace(90, -90, raster.shape[0])
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Create figure with two 2D map projections
    fig = plt.figure(figsize=(14, 5))

    # Left map: Global SH mode
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax1.set_global()
    im1 = ax1.pcolormesh(lon_grid, lat_grid, sh_mode, transform=ccrs.PlateCarree(),
                         cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray')
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')

    # Right map: Combined US + Europe Slepian modes
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())
    ax2.set_global()
    im2 = ax2.pcolormesh(lon_grid, lat_grid, combined_slepian, transform=ccrs.PlateCarree(),
                         cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray')
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')

    # Draw both spherical cap boundaries
    draw_cap_circle(ax2, us_cap['center'][0], us_cap['center'][1],
                    us_cap['radius'], color='red', linewidth=2)
    draw_cap_circle(ax2, europe_cap['center'][0], europe_cap['center'][1],
                    europe_cap['radius'], color='red', linewidth=2)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Amplitude', fontsize=11)

    plt.subplots_adjust(bottom=0.18, wspace=0.08)
    plt.savefig(os.path.join(op_dir, 'slepian_basis_map.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(op_dir, 'slepian_basis_map.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {op_dir}slepian_basis_map.png/pdf")

if __name__ == "__main__":
    main()
