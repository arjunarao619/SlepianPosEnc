"""Slepian encoding functions."""

import torch
import numpy as np
import os
import time
from typing import Dict, List, Optional

from .encoders import encode_loc_sh

try:
    import pyshtools as pysh
    HAVE_PYSH = True
except ImportError:
    HAVE_PYSH = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def compute_slepian_raster(
    L_global: int = 10,
    L_regional: int = 80,
    caps: List[Dict] = None,
    lambda_thresh: float = 0.1,
    grid_resolution: float = 0.5,
    cache_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """Pre-compute Slepian features on a grid for bilinear interpolation.

    this is the faster approach - we evaluate slepian functions once on a dense grid,
    then use bilinear interpolation at runtime. trades memory for speed.
    """
    if not HAVE_PYSH:
        raise ImportError("pyshtools required: pip install pyshtools")

    if caps is None:
        # default caps cover north america and europe - adjust for your study region
        caps = [
            {'name': 'us', 'center': (-95.0, 40.0), 'radius': 25.0},
            {'name': 'europe', 'center': (10.0, 50.0), 'radius': 20.0},
        ]

    t0 = time.time()

    # grid resolution of 0.5 deg gives 720x360 = ~260k points, good balance of accuracy vs memory
    nlon = int(360 / grid_resolution)
    nlat = int(180 / grid_resolution)
    lon_grid = np.linspace(-180 + grid_resolution/2, 180 - grid_resolution/2, nlon)
    lat_grid = np.linspace(90 - grid_resolution/2, -90 + grid_resolution/2, nlat)

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    lon_flat = lon_mesh.flatten().astype(np.float32)
    lat_flat = lat_mesh.flatten().astype(np.float32)
    n_points = len(lon_flat)

    if verbose:
        print(f"Computing Slepian raster: {nlon}x{nlat}, L_global={L_global}, L_regional={L_regional}")

    # global SH gives baseline spatial coverage everywhere on the sphere
    # L_global=0 means slepian-only mode (no global background)
    if L_global > 0:
        lon_norm = torch.tensor(lon_flat / 180.0, dtype=torch.float32)
        lat_norm = torch.tensor(lat_flat / 90.0, dtype=torch.float32)
        coords_norm = torch.stack([lon_norm, lat_norm], dim=1)
        global_features = encode_loc_sh(coords_norm, L=L_global).numpy()
        global_dim = global_features.shape[1]
    else:
        global_features = None
        global_dim = 0

    regional_features_list = []
    regional_dims = {}

    for cap in caps:
        cap_name = cap['name']
        cap_center = cap['center']
        cap_radius = cap['radius']

        # shannon number N ~ (L+1)^2 * (1-cos(theta))/2 tells us how many modes
        # are well-concentrated in the cap. we request 2x shannon to be safe,
        # then filter by eigenvalue
        theta_rad = cap_radius * np.pi / 180.0
        shannon_approx = int((L_regional + 1) ** 2 * (1 - np.cos(theta_rad)) / 2)
        nmax = min(shannon_approx * 2, (L_regional + 1) ** 2)

        # pyshtools computes slepian basis centered at north pole
        slepian = pysh.Slepian.from_cap(theta=cap_radius, lmax=L_regional, nmax=nmax)

        # rotate from north pole to actual cap center - this is the key step
        # that makes the basis region-specific
        slepian.rotate(clat=cap_center[1], clon=cap_center[0], nrot=nmax)

        # eigenvalue = concentration ratio (0 to 1). modes with lambda close to 1
        # have most energy inside the cap. lambda_thresh=0.1 keeps well-concentrated modes
        eigenvalues = slepian.eigenvalues[:nmax]
        keep_mask = eigenvalues > lambda_thresh
        mode_indices = np.where(keep_mask)[0]

        if verbose:
            print(f"  {cap_name}: {keep_mask.sum()}/{nmax} modes kept")

        # pyshtools wants lon in [0, 360], we use [-180, 180] internally
        lon_360 = np.where(lon_flat < 0, lon_flat + 360, lon_flat)
        cap_features = []

        for mode_idx in tqdm(mode_indices, desc=f"  {cap_name}", disable=not verbose, leave=False):
            # convert to SH coeffs, use ortho normalization to match our global SH
            coeffs = slepian.to_shcoeffs(mode_idx).convert(normalization='ortho')
            vals = coeffs.expand(lon=lon_360, lat=lat_flat, degrees=True)
            cap_features.append(vals)

        if cap_features:
            cap_features = np.column_stack(cap_features).astype(np.float32)
        else:
            cap_features = np.zeros((n_points, 0), dtype=np.float32)

        regional_features_list.append(cap_features)
        regional_dims[cap_name] = cap_features.shape[1]

    # concatenate: [global SH | cap1 slepian | cap2 slepian | ...]
    if global_features is not None:
        all_features = [global_features] + regional_features_list
    else:
        all_features = regional_features_list
    combined = np.hstack(all_features).astype(np.float32)

    # standardize each feature to zero mean, unit variance for stable training
    combined = (combined - combined.mean(0, keepdims=True)) / (combined.std(0, keepdims=True) + 1e-8)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    # reshape to [lat, lon, features] for bilinear_interpolate
    raster_tensor = torch.tensor(combined.reshape(nlat, nlon, -1), dtype=torch.float32)

    metadata = {
        'L_global': L_global, 'L_regional': L_regional, 'lambda_thresh': lambda_thresh,
        'grid_resolution': grid_resolution, 'caps': caps, 'global_dim': global_dim,
        'regional_dims': regional_dims, 'total_dim': combined.shape[1],
        'shape': list(raster_tensor.shape), 'n_grid_points': n_points,
    }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({'raster': raster_tensor, 'metadata': metadata}, cache_path)

    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s, shape={list(raster_tensor.shape)}")

    return {'raster': raster_tensor, 'metadata': metadata}


def load_slepian_raster(cache_path: str, verbose: bool = True) -> Dict:
    """Load precomputed Slepian raster from cache."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Slepian cache not found: {cache_path}")

    cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)

    if verbose:
        meta = cache_data.get('metadata', {})
        print(f"Loaded Slepian raster: {list(cache_data['raster'].shape)}, "
              f"L_global={meta.get('L_global')}, L_regional={meta.get('L_regional')}")

    return cache_data


def get_slepian_cache_path(params: Dict, cache_dir: str = './cache/slepian') -> str:
    """Generate cache path from params."""
    L_global = params.get('slepian_L_global', 10)
    L_regional = params.get('slepian_L_regional', 80)
    lambda_thresh = params.get('slepian_lambda_thresh', 0.1)
    grid_res = params.get('slepian_grid_resolution', 0.5)
    caps = params.get('slepian_caps', [])

    caps_str = '_'.join([f"{c['name']}{c['radius']:.0f}" for c in caps])
    cache_name = f"slepian_Lg{L_global}_Lr{L_regional}_lam{lambda_thresh}_res{grid_res}_{caps_str}.pt"
    return os.path.join(cache_dir, cache_name)


def compute_slepian_coefficients(
    L_global: int = 10,
    L_regional: int = 80,
    caps: List[Dict] = None,
    lambda_thresh: float = 0.1,
    cache_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """Compute Slepian SH coefficients for direct evaluation.

    alternative to raster approach - stores the SH coefficients instead of precomputed
    values. more accurate (no interpolation) but slower at runtime since we evaluate
    SH expansion for each query point.
    """
    if not HAVE_PYSH:
        raise ImportError("pyshtools required")

    if caps is None:
        caps = [
            {'name': 'us', 'center': (-95.0, 40.0), 'radius': 25.0},
            {'name': 'europe', 'center': (10.0, 50.0), 'radius': 20.0},
        ]

    if verbose:
        print(f"Computing Slepian coefficients: L_global={L_global}, L_regional={L_regional}")

    regional_coefficients = []
    regional_dims = {}

    for cap in caps:
        cap_name = cap['name']
        cap_center = cap['center']
        cap_radius = cap['radius']

        theta_rad = cap_radius * np.pi / 180.0
        shannon_approx = int((L_regional + 1) ** 2 * (1 - np.cos(theta_rad)) / 2)
        nmax = min(shannon_approx * 2, (L_regional + 1) ** 2)

        slepian = pysh.Slepian.from_cap(theta=cap_radius, lmax=L_regional, nmax=nmax)
        slepian.rotate(clat=cap_center[1], clon=cap_center[0], nrot=nmax)

        eigenvalues = slepian.eigenvalues[:nmax]
        keep_mask = eigenvalues > lambda_thresh
        mode_indices = np.where(keep_mask)[0]

        # store raw SH coeffs - shape is (2, L+1, L+1) for each mode
        # first dim is cosine/sine terms
        cap_coeffs = []
        for mode_idx in mode_indices:
            coeffs = slepian.to_shcoeffs(mode_idx).convert(normalization='ortho')
            cap_coeffs.append(coeffs.coeffs.copy().astype(np.float32))

        regional_coefficients.append(cap_coeffs)
        regional_dims[cap_name] = len(cap_coeffs)

        if verbose:
            print(f"  {cap_name}: {len(cap_coeffs)} modes")

    metadata = {
        'L_global': L_global, 'L_regional': L_regional, 'lambda_thresh': lambda_thresh,
        'caps': caps, 'global_dim': L_global ** 2, 'regional_dims': regional_dims,
        'total_dim': L_global ** 2 + sum(regional_dims.values()), 'encoding_type': 'direct',
    }

    result = {'coefficients': regional_coefficients, 'metadata': metadata}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(result, cache_path)

    return result


def encode_loc_slepian_direct(loc_ip, slepian_data: Dict, device=None):
    """Compute Slepian features directly from SH coefficients.

    slower than raster interpolation but exact - useful for evaluation or
    when you need features at arbitrary high-precision locations.
    """
    if not HAVE_PYSH:
        raise ImportError("pyshtools required")

    meta = slepian_data['metadata']
    L_global = meta['L_global']
    shcoeffs_list = slepian_data['shcoeffs_list']

    if device is None:
        device = loc_ip.device

    # convert from normalized [-1,1] back to degrees
    lon = loc_ip[:, 0].cpu().numpy() * 180.0
    lat = loc_ip[:, 1].cpu().numpy() * 90.0
    lon_360 = np.where(lon < 0, lon + 360, lon)
    N = len(lon)

    if L_global > 0:
        global_feats = encode_loc_sh(loc_ip, L=L_global)
    else:
        global_feats = None

    regional_feats_list = []
    for cap_shcoeffs in shcoeffs_list:
        if not cap_shcoeffs:
            continue
        cap_feats = np.zeros((N, len(cap_shcoeffs)), dtype=np.float32)
        for i, shcoeffs in enumerate(cap_shcoeffs):
            # pyshtools expand() evaluates the SH expansion at given points
            cap_feats[:, i] = shcoeffs.expand(lon=lon_360, lat=lat, degrees=True)
        regional_feats_list.append(torch.tensor(cap_feats, dtype=torch.float32, device=device))

    if regional_feats_list:
        regional_feats = torch.cat(regional_feats_list, dim=1)
        if global_feats is not None:
            return torch.cat([global_feats, regional_feats], dim=1)
        return regional_feats
    return global_feats if global_feats is not None else torch.zeros((N, 0), device=device)


def get_slepian_direct_cache_path(params: Dict, cache_dir: str = './cache/slepian') -> str:
    """Generate cache path for direct Slepian coefficients."""
    L_global = params.get('slepian_L_global', 10)
    L_regional = params.get('slepian_L_regional', 80)
    lambda_thresh = params.get('slepian_lambda_thresh', 0.1)
    caps = params.get('slepian_caps', [])

    caps_str = '_'.join([f"{c['name']}{c['radius']:.0f}" for c in caps])
    return os.path.join(cache_dir, f"slepian_direct_Lg{L_global}_Lr{L_regional}_lam{lambda_thresh}_{caps_str}.pt")
