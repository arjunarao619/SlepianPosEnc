import torch
import numpy as np
import math
import datetime
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory for SH module import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spherical_harmonics_ylm import SH as SH_analytic

# Try to import pyshtools for Slepian functions
try:
    import pyshtools as pysh
    HAVE_PYSH = True
except ImportError:
    HAVE_PYSH = False

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

class CoordEncoder:

    def __init__(self, input_enc, raster=None, L=10, slepian_raster=None, slepian_data=None):
        self.input_enc = input_enc
        self.raster = raster  # Environmental raster
        self.L = L  # SH max degree (only used when input_enc='sh')
        self.slepian_raster = slepian_raster  # Precomputed Slepian feature raster (for 'slepian' mode)
        self.slepian_data = slepian_data  # Slepian coefficients (for 'slepian_direct' mode)

    def encode(self, locs, normalize=False):
        # assumes lon, lat in range [-180, 180] and [-90, 90]
        if normalize:
            locs = normalize_coords(locs)
        if self.input_enc == 'sin_cos':  # sinusoidal encoding
            loc_feats = encode_loc(locs)
        elif self.input_enc == 'env':  # bioclim variables
            loc_feats = bilinear_interpolate(locs, self.raster)
        elif self.input_enc == 'sin_cos_env':  # sinusoidal encoding & bioclim variables
            loc_feats = encode_loc(locs)
            context_feats = bilinear_interpolate(locs, self.raster)
            loc_feats = torch.cat((loc_feats, context_feats), 1)
        elif self.input_enc == 'sh':  # spherical harmonics encoding
            loc_feats = encode_loc_sh(locs, L=self.L)
        elif self.input_enc == 'slepian':  # Slepian encoding via precomputed raster
            loc_feats = bilinear_interpolate(locs, self.slepian_raster)
        elif self.input_enc == 'slepian_env':  # Slepian + environmental features
            slep_feats = bilinear_interpolate(locs, self.slepian_raster)
            env_feats = bilinear_interpolate(locs, self.raster)
            loc_feats = torch.cat((slep_feats, env_feats), 1)
        elif self.input_enc == 'slepian_direct':  # Direct Slepian computation (no interpolation)
            loc_feats = encode_loc_slepian_direct(locs, self.slepian_data)
        elif self.input_enc == 'slepian_direct_env':  # Direct Slepian + environmental features
            slep_feats = encode_loc_slepian_direct(locs, self.slepian_data)
            env_feats = bilinear_interpolate(locs, self.raster)
            loc_feats = torch.cat((slep_feats, env_feats), 1)
        else:
            raise NotImplementedError('Unknown input encoding.')
        return loc_feats

def normalize_coords(locs):
    # locs is in lon {-180, 180}, lat {90, -90}
    # output is in the range [-1, 1]

    locs[:,0] /= 180.0
    locs[:,1] /= 90.0

    return locs

def encode_loc(loc_ip, concat_dim=1):
    # assumes inputs location are in range -1 to 1
    # location is lon, lat
    feats = torch.cat((torch.sin(math.pi*loc_ip), torch.cos(math.pi*loc_ip)), concat_dim)
    return feats


def encode_loc_sh(loc_ip, L=10):
    """
    Encode locations using Spherical Harmonics.

    Args:
        loc_ip: (N, 2) tensor, normalized coords in [-1, 1]
                loc_ip[:, 0] = lon/180, loc_ip[:, 1] = lat/90
        L: Max degree (output dim = L^2)
    Returns:
        (N, L^2) tensor of SH features
    """
    lon_norm = loc_ip[:, 0]  # [-1, 1]
    lat_norm = loc_ip[:, 1]  # [-1, 1]

    # Convert to spherical coordinates
    # phi (azimuth): [-1, 1] -> [0, 2*pi]
    phi = (lon_norm + 1) * math.pi

    # theta (colatitude): lat=+1 (north) -> theta=0, lat=-1 (south) -> theta=pi
    theta = (1 - lat_norm) * (math.pi / 2)

    # Clamp to avoid numerical issues at poles
    theta = torch.clamp(theta, 1e-7, math.pi - 1e-7)

    feats_list = []
    for l in range(L):
        for m in range(-l, l + 1):
            ylm = SH_analytic(m, l, phi, theta)
            if not isinstance(ylm, torch.Tensor):
                ylm = torch.full_like(theta, float(ylm))
            elif ylm.dim() == 0:
                ylm = torch.full_like(theta, ylm.item())
            feats_list.append(ylm)

    feats = torch.stack(feats_list, dim=1)
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


# =============================================================================
# Slepian Encoding Functions
# =============================================================================

def compute_slepian_raster(
    L_global: int = 10,
    L_regional: int = 80,
    caps: List[Dict] = None,
    lambda_thresh: float = 0.1,
    grid_resolution: float = 0.5,
    cache_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Pre-compute Slepian features on a dense grid for fast bilinear interpolation.

    This creates a raster of shape [H, W, C] where:
    - H = 180 / grid_resolution (latitude bins)
    - W = 360 / grid_resolution (longitude bins)
    - C = L_global^2 + sum of regional Slepian modes

    Features = [Global SH (L_global)] + [Cap 1 Slepian] + [Cap 2 Slepian] + ...

    Args:
        L_global: Max degree for global spherical harmonics (baseline features)
        L_regional: Max degree for regional Slepian functions
        caps: List of cap definitions, each with:
              {'name': str, 'center': (lon, lat), 'radius': float (degrees)}
        lambda_thresh: Eigenvalue threshold - keep modes with lambda > thresh
        grid_resolution: Grid spacing in degrees (0.5 = 720x360 grid)
        cache_path: Path to save computed raster (None = don't cache)
        verbose: Print progress information

    Returns:
        Dict with:
        - 'raster': torch.Tensor [H, W, C] of features
        - 'metadata': dict with feature dimensions and configuration
    """
    if not HAVE_PYSH:
        raise ImportError("pyshtools is required for Slepian encoding. "
                          "Install with: pip install pyshtools")

    if caps is None:
        caps = [
            {'name': 'us', 'center': (-95.0, 40.0), 'radius': 25.0},
            {'name': 'europe', 'center': (10.0, 50.0), 'radius': 20.0},
        ]

    t0 = time.time()

    # Create grid coordinates
    nlon = int(360 / grid_resolution)
    nlat = int(180 / grid_resolution)
    lon_grid = np.linspace(-180 + grid_resolution/2, 180 - grid_resolution/2, nlon)
    lat_grid = np.linspace(90 - grid_resolution/2, -90 + grid_resolution/2, nlat)

    # Create meshgrid and flatten
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    lon_flat = lon_mesh.flatten().astype(np.float32)
    lat_flat = lat_mesh.flatten().astype(np.float32)
    n_points = len(lon_flat)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Computing Slepian Feature Raster")
        print(f"{'='*70}")
        print(f"Grid: {nlon} x {nlat} = {n_points:,} points")
        print(f"Resolution: {grid_resolution}°")
        print(f"Global SH: L={L_global} ({L_global**2} features)")
        print(f"Regional Slepian: L={L_regional}")
        print(f"Caps: {[c['name'] for c in caps]}")
        print(f"Eigenvalue threshold: λ > {lambda_thresh}")

    # Step 1: Compute global SH features
    if verbose:
        print(f"\n[1/3] Computing global SH features (L={L_global})...")

    # Normalize coordinates for encode_loc_sh: lon/180, lat/90 -> [-1, 1]
    lon_norm = torch.tensor(lon_flat / 180.0, dtype=torch.float32)
    lat_norm = torch.tensor(lat_flat / 90.0, dtype=torch.float32)
    coords_norm = torch.stack([lon_norm, lat_norm], dim=1)

    global_features = encode_loc_sh(coords_norm, L=L_global).numpy()
    global_dim = global_features.shape[1]

    if verbose:
        print(f"  Global SH shape: {global_features.shape}")

    # Step 2: Compute Slepian features for each cap
    regional_features_list = []
    regional_dims = {}

    if verbose:
        print(f"\n[2/3] Computing regional Slepian features...")

    for cap_idx, cap in enumerate(caps):
        cap_name = cap['name']
        cap_center = cap['center']  # (lon, lat)
        cap_radius = cap['radius']

        if verbose:
            print(f"\n  Cap {cap_idx+1}/{len(caps)}: {cap_name}")
            print(f"    Center: ({cap_center[0]:.1f}°, {cap_center[1]:.1f}°)")
            print(f"    Radius: {cap_radius}°")

        # Estimate Shannon number: N ≈ (L+1)² * (1 - cos(θ)) / 2
        theta_rad = cap_radius * np.pi / 180.0
        shannon_approx = int((L_regional + 1) ** 2 * (1 - np.cos(theta_rad)) / 2)
        # Request more modes than Shannon number, then filter by eigenvalue
        nmax = min(shannon_approx * 2, (L_regional + 1) ** 2)

        if verbose:
            print(f"    Shannon number ≈ {shannon_approx}, requesting {nmax} modes")

        # Create Slepian basis
        t_cap = time.time()
        slepian = pysh.Slepian.from_cap(
            theta=cap_radius,
            lmax=L_regional,
            nmax=nmax
        )

        # Rotate to cap center (from North Pole)
        slepian.rotate(clat=cap_center[1], clon=cap_center[0], nrot=nmax)

        # Get eigenvalues and filter
        eigenvalues = slepian.eigenvalues[:nmax]
        keep_mask = eigenvalues > lambda_thresh
        num_modes_kept = keep_mask.sum()

        if verbose:
            print(f"    Eigenvalues: min={eigenvalues.min():.4f}, max={eigenvalues.max():.4f}")
            print(f"    Modes kept (λ>{lambda_thresh}): {num_modes_kept}/{nmax}")

        # Evaluate Slepian functions at all grid points
        # pyshtools expects lon in [0, 360]
        lon_360 = np.where(lon_flat < 0, lon_flat + 360, lon_flat)

        cap_features = []
        mode_indices = np.where(keep_mask)[0]

        iterator = tqdm(mode_indices, desc=f"    Evaluating {cap_name} modes",
                        disable=not verbose, leave=False)
        for mode_idx in iterator:
            coeffs = slepian.to_shcoeffs(mode_idx)
            # Convert to ortho normalization to match global SH from encode_loc_sh
            coeffs_ortho = coeffs.convert(normalization='ortho')
            vals = coeffs_ortho.expand(lon=lon_360, lat=lat_flat, degrees=True)
            cap_features.append(vals)

        if len(cap_features) > 0:
            cap_features = np.column_stack(cap_features).astype(np.float32)
        else:
            cap_features = np.zeros((n_points, 0), dtype=np.float32)

        regional_features_list.append(cap_features)
        regional_dims[cap_name] = cap_features.shape[1]

        if verbose:
            dt = time.time() - t_cap
            print(f"    Cap features shape: {cap_features.shape}, time: {dt:.1f}s")

    # Step 3: Combine all features
    if verbose:
        print(f"\n[3/3] Combining features...")

    all_features = [global_features] + regional_features_list
    combined_features = np.hstack(all_features).astype(np.float32)

    # Normalize features to zero mean, unit variance
    feature_mean = combined_features.mean(axis=0, keepdims=True)
    feature_std = combined_features.std(axis=0, keepdims=True) + 1e-8
    combined_features = (combined_features - feature_mean) / feature_std

    total_dim = combined_features.shape[1]

    # Reshape to [H, W, C] for bilinear interpolation
    raster = combined_features.reshape(nlat, nlon, total_dim)

    # Handle any NaN/Inf values
    raster = np.nan_to_num(raster, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to tensor
    raster_tensor = torch.tensor(raster, dtype=torch.float32)

    # Create metadata
    metadata = {
        'L_global': L_global,
        'L_regional': L_regional,
        'lambda_thresh': lambda_thresh,
        'grid_resolution': grid_resolution,
        'caps': caps,
        'global_dim': global_dim,
        'regional_dims': regional_dims,
        'total_dim': total_dim,
        'shape': list(raster_tensor.shape),
        'n_grid_points': n_points,
    }

    # Cache to disk if requested
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'raster': raster_tensor,
            'metadata': metadata
        }
        torch.save(cache_data, cache_path)
        if verbose:
            print(f"  Cached to: {cache_path}")

    total_time = time.time() - t0
    if verbose:
        print(f"\n{'='*70}")
        print(f"Slepian raster computation complete!")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Raster shape: {list(raster_tensor.shape)}")
        print(f"  Total features: {total_dim}")
        print(f"    - Global SH: {global_dim}")
        for cap_name, dim in regional_dims.items():
            print(f"    - {cap_name} Slepian: {dim}")
        print(f"  Memory: {raster_tensor.numel() * 4 / 1e6:.1f} MB")
        print(f"{'='*70}\n")

    return {
        'raster': raster_tensor,
        'metadata': metadata
    }


def load_slepian_raster(cache_path: str, verbose: bool = True) -> Dict:
    """
    Load precomputed Slepian raster from cache.

    Args:
        cache_path: Path to cached .pt file
        verbose: Print loading info

    Returns:
        Dict with 'raster' tensor and 'metadata' dict
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Slepian cache not found: {cache_path}")

    cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)

    if verbose:
        meta = cache_data.get('metadata', {})
        print(f"Loaded Slepian raster from {cache_path}")
        print(f"  Shape: {list(cache_data['raster'].shape)}")
        print(f"  L_global={meta.get('L_global', '?')}, "
              f"L_regional={meta.get('L_regional', '?')}, "
              f"total_dim={meta.get('total_dim', '?')}")

    return cache_data


def get_slepian_cache_path(params: Dict, cache_dir: str = './cache/slepian') -> str:
    """
    Generate cache path based on Slepian configuration parameters.

    Args:
        params: Dict with slepian_L_global, slepian_L_regional, slepian_caps, etc.
        cache_dir: Base directory for cache files

    Returns:
        Full path to cache file
    """
    L_global = params.get('slepian_L_global', 10)
    L_regional = params.get('slepian_L_regional', 80)
    lambda_thresh = params.get('slepian_lambda_thresh', 0.1)
    grid_res = params.get('slepian_grid_resolution', 0.5)
    caps = params.get('slepian_caps', [])

    # Build descriptive filename
    caps_str = '_'.join([f"{c['name']}{c['radius']:.0f}" for c in caps])
    cache_name = (f"slepian_Lg{L_global}_Lr{L_regional}_"
                  f"lam{lambda_thresh}_res{grid_res}_{caps_str}.pt")

    return os.path.join(cache_dir, cache_name)


# =============================================================================
# Direct Slepian Computation (no raster interpolation)
# =============================================================================

def compute_slepian_coefficients(
    L_global: int = 10,
    L_regional: int = 80,
    caps: List[Dict] = None,
    lambda_thresh: float = 0.1,
    cache_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Compute Slepian coefficient matrices for direct feature computation.

    Instead of precomputing features on a raster grid, this stores the
    Slepian SH coefficients, allowing exact feature computation at any location.

    Args:
        L_global: Max degree for global spherical harmonics
        L_regional: Max degree for regional Slepian functions
        caps: List of cap definitions
        lambda_thresh: Eigenvalue threshold
        cache_path: Path to save coefficients
        verbose: Print progress

    Returns:
        Dict with:
        - 'coefficients': list of coefficient matrices per cap
        - 'metadata': configuration info
    """
    if not HAVE_PYSH:
        raise ImportError("pyshtools is required for Slepian encoding.")

    if caps is None:
        caps = [
            {'name': 'us', 'center': (-95.0, 40.0), 'radius': 25.0},
            {'name': 'europe', 'center': (10.0, 50.0), 'radius': 20.0},
        ]

    t0 = time.time()
    if verbose:
        print(f"\n{'='*70}")
        print("Computing Slepian Coefficients for Direct Computation")
        print(f"{'='*70}")
        print(f"Global SH: L={L_global} ({L_global**2} features)")
        print(f"Regional Slepian: L={L_regional}")
        print(f"Caps: {[c['name'] for c in caps]}")
        print(f"Eigenvalue threshold: λ > {lambda_thresh}")

    regional_coefficients = []
    regional_dims = {}

    for cap_idx, cap in enumerate(caps):
        cap_name = cap['name']
        cap_center = cap['center']
        cap_radius = cap['radius']

        if verbose:
            print(f"\n  Cap {cap_idx+1}/{len(caps)}: {cap_name}")
            print(f"    Center: ({cap_center[0]:.1f}°, {cap_center[1]:.1f}°)")
            print(f"    Radius: {cap_radius}°")

        # Estimate Shannon number
        theta_rad = cap_radius * np.pi / 180.0
        shannon_approx = int((L_regional + 1) ** 2 * (1 - np.cos(theta_rad)) / 2)
        nmax = min(shannon_approx * 2, (L_regional + 1) ** 2)

        if verbose:
            print(f"    Shannon number ≈ {shannon_approx}, requesting {nmax} modes")

        # Create Slepian basis
        t_cap = time.time()
        slepian = pysh.Slepian.from_cap(
            theta=cap_radius,
            lmax=L_regional,
            nmax=nmax
        )

        # Rotate to cap center
        slepian.rotate(clat=cap_center[1], clon=cap_center[0], nrot=nmax)

        # Get eigenvalues and filter
        eigenvalues = slepian.eigenvalues[:nmax]
        keep_mask = eigenvalues > lambda_thresh
        num_modes_kept = keep_mask.sum()

        if verbose:
            print(f"    Eigenvalues: min={eigenvalues.min():.4f}, max={eigenvalues.max():.4f}")
            print(f"    Modes kept (λ>{lambda_thresh}): {num_modes_kept}/{nmax}")

        # Extract SH coefficients for each kept mode
        # Store as arrays for serialization, reconstruct SHCoeffs at load time
        mode_indices = np.where(keep_mask)[0]
        cap_coeffs = []

        for mode_idx in mode_indices:
            coeffs = slepian.to_shcoeffs(mode_idx)
            # Convert to ortho normalization to match encode_loc_sh
            coeffs_ortho = coeffs.convert(normalization='ortho')
            # Store the coefficient array: shape (2, L+1, L+1)
            cap_coeffs.append(coeffs_ortho.coeffs.copy().astype(np.float32))

        regional_coefficients.append(cap_coeffs)  # List of arrays
        regional_dims[cap_name] = len(cap_coeffs)

        if verbose:
            dt = time.time() - t_cap
            print(f"    Modes extracted: {len(cap_coeffs)}, time: {dt:.1f}s")

    # Compute total dimension
    total_regional_dim = sum(regional_dims.values())
    total_dim = L_global ** 2 + total_regional_dim

    metadata = {
        'L_global': L_global,
        'L_regional': L_regional,
        'lambda_thresh': lambda_thresh,
        'caps': caps,
        'global_dim': L_global ** 2,
        'regional_dims': regional_dims,
        'total_dim': total_dim,
        'encoding_type': 'direct',  # Flag to distinguish from raster
    }

    result = {
        'coefficients': regional_coefficients,  # List of lists of coefficient arrays
        'metadata': metadata
    }

    # Cache to disk
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(result, cache_path)
        if verbose:
            print(f"  Cached to: {cache_path}")

    total_time = time.time() - t0
    if verbose:
        coeff_size = sum(sum(c.nbytes for c in cap) for cap in regional_coefficients) / 1e6
        print(f"\n{'='*70}")
        print(f"Slepian coefficient computation complete!")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Total features: {total_dim}")
        print(f"    - Global SH: {L_global**2}")
        for cap_name, dim in regional_dims.items():
            print(f"    - {cap_name} Slepian: {dim}")
        print(f"  Coefficient storage: {coeff_size:.1f} MB")
        print(f"{'='*70}\n")

    return result


def encode_loc_slepian_direct(loc_ip, slepian_data: Dict, device=None):
    """
    Compute Slepian features directly using pyshtools (no raster interpolation).

    Uses pyshtools' numerically stable SH expansion to evaluate Slepian
    functions at arbitrary locations.

    Args:
        loc_ip: (N, 2) tensor, normalized coords in [-1, 1]
                loc_ip[:, 0] = lon/180, loc_ip[:, 1] = lat/90
        slepian_data: Dict from compute_slepian_coefficients() with 'shcoeffs_list'
        device: torch device

    Returns:
        (N, total_dim) tensor of Slepian features
    """
    if not HAVE_PYSH:
        raise ImportError("pyshtools required for direct Slepian computation")

    meta = slepian_data['metadata']
    L_global = meta['L_global']
    shcoeffs_list = slepian_data['shcoeffs_list']  # List of lists of SHCoeffs objects

    if device is None:
        device = loc_ip.device

    # Convert normalized coords to geographic coords for pyshtools
    # loc_ip: [-1, 1] -> lon: [-180, 180], lat: [-90, 90]
    lon = loc_ip[:, 0].cpu().numpy() * 180.0
    lat = loc_ip[:, 1].cpu().numpy() * 90.0

    # pyshtools expects lon in [0, 360]
    lon_360 = np.where(lon < 0, lon + 360, lon)

    N = len(lon)

    # Step 1: Compute global SH features (L_global is small, analytically stable)
    global_feats = encode_loc_sh(loc_ip, L=L_global)  # (N, L_global^2)

    # Step 2: Evaluate each Slepian mode at query locations using pyshtools
    regional_feats_list = []

    for cap_shcoeffs in shcoeffs_list:
        if len(cap_shcoeffs) == 0:
            continue

        num_modes = len(cap_shcoeffs)
        cap_feats = np.zeros((N, num_modes), dtype=np.float32)

        for mode_idx, shcoeffs in enumerate(cap_shcoeffs):
            # Use pyshtools expand() for numerically stable evaluation
            vals = shcoeffs.expand(lon=lon_360, lat=lat, degrees=True)
            cap_feats[:, mode_idx] = vals

        regional_feats_list.append(torch.tensor(cap_feats, dtype=torch.float32, device=device))

    # Concatenate all features
    if regional_feats_list:
        regional_feats = torch.cat(regional_feats_list, dim=1)
        all_feats = torch.cat([global_feats, regional_feats], dim=1)
    else:
        all_feats = global_feats

    return all_feats


def get_slepian_direct_cache_path(params: Dict, cache_dir: str = './cache/slepian') -> str:
    """Generate cache path for direct Slepian coefficients."""
    L_global = params.get('slepian_L_global', 10)
    L_regional = params.get('slepian_L_regional', 80)
    lambda_thresh = params.get('slepian_lambda_thresh', 0.1)
    caps = params.get('slepian_caps', [])

    caps_str = '_'.join([f"{c['name']}{c['radius']:.0f}" for c in caps])
    cache_name = (f"slepian_direct_Lg{L_global}_Lr{L_regional}_"
                  f"lam{lambda_thresh}_{caps_str}.pt")

    return os.path.join(cache_dir, cache_name)


def bilinear_interpolate(loc_ip, data, remove_nans_raster=True):
    # loc is N x 2 vector, where each row is [lon,lat] entry
    #   each entry spans range [-1,1]
    # data is H x W x C, height x width x channel data matrix
    # op will be N x C matrix of interpolated features

    assert data is not None

    # map to [0,1], then scale to data size
    loc = (loc_ip.clone() + 1) / 2.0
    loc[:,1] = 1 - loc[:,1] # this is because latitude goes from +90 on top to bottom while
                            # longitude goes from -90 to 90 left to right

    assert not torch.any(torch.isnan(loc))

    if remove_nans_raster:
        data[torch.isnan(data)] = 0.0 # replace with mean value (0 is mean post-normalization)

    # cast locations into pixel space
    loc[:, 0] *= (data.shape[1]-1)
    loc[:, 1] *= (data.shape[0]-1)

    loc_int = torch.floor(loc).long()  # integer pixel coordinates
    xx = loc_int[:, 0]
    yy = loc_int[:, 1]
    xx_plus = xx + 1
    xx_plus[xx_plus > (data.shape[1]-1)] = data.shape[1]-1
    yy_plus = yy + 1
    yy_plus[yy_plus > (data.shape[0]-1)] = data.shape[0]-1

    loc_delta = loc - torch.floor(loc)   # delta values
    dx = loc_delta[:, 0].unsqueeze(1)
    dy = loc_delta[:, 1].unsqueeze(1)

    interp_val = data[yy, xx, :]*(1-dx)*(1-dy) + data[yy, xx_plus, :]*dx*(1-dy) + \
                 data[yy_plus, xx, :]*(1-dx)*dy   + data[yy_plus, xx_plus, :]*dx*dy

    return interp_val

def rand_samples(batch_size, device, rand_type='uniform'):
    # randomly sample background locations

    if rand_type == 'spherical':
        rand_loc = torch.rand(batch_size, 2).to(device)
        theta1 = 2.0*math.pi*rand_loc[:, 0]
        theta2 = torch.acos(2.0*rand_loc[:, 1] - 1.0)
        lat = 1.0 - 2.0*theta2/math.pi
        lon = (theta1/math.pi) - 1.0
        rand_loc = torch.cat((lon.unsqueeze(1), lat.unsqueeze(1)), 1)

    elif rand_type == 'uniform':
        rand_loc = torch.rand(batch_size, 2).to(device)*2.0 - 1.0

    return rand_loc

def get_time_stamp():
    cur_time = str(datetime.datetime.now())
    date, time = cur_time.split(' ')
    h, m, s = time.split(':')
    s = s.split('.')[0]
    time_stamp = '{}-{}-{}-{}'.format(date, h, m, s)
    return time_stamp

def coord_grid(grid_size, split_ids=None, split_of_interest=None):
    # generate a grid of locations spaced evenly in coordinate space

    feats = np.zeros((grid_size[0], grid_size[1], 2), dtype=np.float32)
    mg = np.meshgrid(np.linspace(-180, 180, feats.shape[1]), np.linspace(90, -90, feats.shape[0]))
    feats[:, :, 0] = mg[0]
    feats[:, :, 1] = mg[1]
    if split_ids is None or split_of_interest is None:
        # return feats for all locations
        # this will be an N x 2 array
        return feats.reshape(feats.shape[0]*feats.shape[1], 2)
    else:
        # only select a subset of locations
        ind_y, ind_x = np.where(split_ids==split_of_interest)

        # these will be N_subset x 2 in size
        return feats[ind_y, ind_x, :]

def create_spatial_split(raster, mask, train_amt=1.0, cell_size=25):
    # generates a checkerboard style train test split
    # 0 is invalid, 1 is train, and 2 is test
    # c_size is units of pixels
    split_ids = np.ones((raster.shape[0], raster.shape[1]))
    start = cell_size
    for ii in np.arange(0, split_ids.shape[0], cell_size):
        if start == 0:
            start = cell_size
        else:
            start = 0
        for jj in np.arange(start, split_ids.shape[1], cell_size*2):
            split_ids[ii:ii+cell_size, jj:jj+cell_size] = 2
    split_ids = split_ids*mask
    if train_amt < 1.0:
        # take a subset of the data
        tr_y, tr_x = np.where(split_ids==1)
        inds = np.random.choice(len(tr_y), int(len(tr_y)*(1.0-train_amt)), replace=False)
        split_ids[tr_y[inds], tr_x[inds]] = 0
    return split_ids

def average_precision_score_faster(y_true, y_scores):
    # drop in replacement for sklearn's average_precision_score
    # comparable up to floating point differences
    num_positives = y_true.sum()
    inds = np.argsort(y_scores)[::-1]
    y_true_s = y_true[inds]

    false_pos_c = np.cumsum(1.0 - y_true_s)
    true_pos_c = np.cumsum(y_true_s)
    recall = true_pos_c / num_positives
    false_neg = np.maximum(true_pos_c + false_pos_c, np.finfo(np.float32).eps)
    precision = true_pos_c / false_neg

    recall_e = np.hstack((0, recall, 1))
    recall_e = (recall_e[1:] - recall_e[:-1])[:-1]
    map_score = (recall_e*precision).sum()
    return map_score
