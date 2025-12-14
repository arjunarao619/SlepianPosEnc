"""
Utility functions for GP baselines.

Includes:
- Coordinate transformations (lon/lat to normalized planar, 3D Cartesian)
- Inducing point initialization (k-means clustering)
- Median heuristic for lengthscale estimation
"""

import numpy as np
import torch
from typing import Tuple, Optional, Literal


def lonlat_to_cartesian3d(coords: torch.Tensor) -> torch.Tensor:
    """
    Convert (longitude, latitude) in degrees to 3D unit sphere coordinates.

    Args:
        coords: [N, 2] tensor with (lon, lat) in degrees

    Returns:
        [N, 3] tensor with (x, y, z) on unit sphere
    """
    coords_rad = torch.deg2rad(coords)
    lon = coords_rad[:, 0]
    lat = coords_rad[:, 1]

    cos_lon = torch.cos(lon)
    sin_lon = torch.sin(lon)
    cos_lat = torch.cos(lat)
    sin_lat = torch.sin(lat)

    x = cos_lat * cos_lon
    y = cos_lat * sin_lon
    z = sin_lat

    return torch.stack([x, y, z], dim=1)


def normalize_lonlat_planar(
    coords: torch.Tensor,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None
) -> torch.Tensor:
    """
    Normalize (lon, lat) to [0, 1] range for planar GPs.

    Args:
        coords: [N, 2] tensor with (lon, lat) in degrees
        lon_range: Optional (min_lon, max_lon) tuple. If None, uses data range.
        lat_range: Optional (min_lat, max_lat) tuple. If None, uses data range.

    Returns:
        [N, 2] tensor normalized to [0, 1] in each dimension
    """
    if lon_range is None:
        lon_min, lon_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    else:
        lon_min, lon_max = lon_range

    if lat_range is None:
        lat_min, lat_max = coords[:, 1].min().item(), coords[:, 1].max().item()
    else:
        lat_min, lat_max = lat_range

    # Add small epsilon to avoid division by zero
    eps = 1e-6
    lon_norm = (coords[:, 0] - lon_min) / (lon_max - lon_min + eps)
    lat_norm = (coords[:, 1] - lat_min) / (lat_max - lat_min + eps)

    return torch.stack([lon_norm, lat_norm], dim=1)


def geodesic_distance(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic (great-circle) distance between points on the sphere.

    Uses haversine formula for numerical stability.

    Args:
        coords1: [N, 2] tensor with (lon, lat) in degrees
        coords2: [M, 2] tensor with (lon, lat) in degrees

    Returns:
        [N, M] tensor of distances in radians (multiply by Earth radius for km)
    """
    # Convert to radians
    coords1_rad = torch.deg2rad(coords1)
    coords2_rad = torch.deg2rad(coords2)

    lon1, lat1 = coords1_rad[:, 0:1], coords1_rad[:, 1:2]  # [N, 1]
    lon2, lat2 = coords2_rad[:, 0], coords2_rad[:, 1]  # [M]

    # Haversine formula
    dlat = lat2.unsqueeze(0) - lat1  # [N, M]
    dlon = lon2.unsqueeze(0) - lon1  # [N, M]

    a = torch.sin(dlat / 2) ** 2 + \
        torch.cos(lat1) * torch.cos(lat2.unsqueeze(0)) * torch.sin(dlon / 2) ** 2

    # Clamp to avoid numerical issues with arcsin
    a = torch.clamp(a, 0, 1)
    c = 2 * torch.arcsin(torch.sqrt(a))

    return c


def init_inducing_points_kmeans(
    train_coords: torch.Tensor,
    num_inducing: int,
    coord_type: Literal["planar", "spherical"] = "planar",
    random_state: int = 42
) -> torch.Tensor:
    """
    Initialize inducing points via k-means clustering on training coordinates.

    Args:
        train_coords: [N, 2] tensor with (lon, lat) in degrees
        num_inducing: Number of inducing points
        coord_type: "planar" normalizes coords, "spherical" converts to 3D
        random_state: Random seed for k-means

    Returns:
        [num_inducing, 2] or [num_inducing, 3] tensor of inducing points
    """
    from sklearn.cluster import KMeans

    if coord_type == "spherical":
        # Convert to 3D Cartesian for clustering
        coords_3d = lonlat_to_cartesian3d(train_coords).numpy()
        kmeans = KMeans(
            n_clusters=num_inducing,
            random_state=random_state,
            n_init=10
        ).fit(coords_3d)
        # Return 3D inducing points
        return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    else:
        # Normalize to [0, 1] for planar clustering
        coords_norm = normalize_lonlat_planar(train_coords).numpy()
        kmeans = KMeans(
            n_clusters=num_inducing,
            random_state=random_state,
            n_init=10
        ).fit(coords_norm)
        return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)


def init_inducing_points_random(
    train_coords: torch.Tensor,
    num_inducing: int,
    coord_type: Literal["planar", "spherical"] = "planar",
    random_state: int = 42
) -> torch.Tensor:
    """
    Initialize inducing points by random selection from training data.

    Args:
        train_coords: [N, 2] tensor with (lon, lat) in degrees
        num_inducing: Number of inducing points
        coord_type: "planar" normalizes coords, "spherical" converts to 3D
        random_state: Random seed

    Returns:
        [num_inducing, D] tensor of inducing points
    """
    rng = np.random.default_rng(random_state)
    n = len(train_coords)
    indices = rng.choice(n, size=min(num_inducing, n), replace=False)

    selected = train_coords[indices]

    if coord_type == "spherical":
        return lonlat_to_cartesian3d(selected)
    else:
        return normalize_lonlat_planar(selected)


def median_heuristic_lengthscale(
    coords: torch.Tensor,
    coord_type: Literal["planar", "spherical"] = "planar",
    sample_size: int = 5000
) -> float:
    """
    Compute lengthscale using the median heuristic.

    The median heuristic sets the lengthscale to the median of pairwise distances.

    Args:
        coords: [N, 2] tensor with (lon, lat) in degrees
        coord_type: "planar" for normalized Euclidean, "spherical" for geodesic
        sample_size: Number of points to sample (for large datasets)

    Returns:
        Suggested lengthscale value
    """
    n = len(coords)
    if n > sample_size:
        # Subsample for efficiency
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=sample_size, replace=False)
        coords = coords[indices]

    if coord_type == "spherical":
        # Use geodesic distances
        dists = geodesic_distance(coords, coords)
    else:
        # Normalize and use Euclidean
        coords_norm = normalize_lonlat_planar(coords)
        dists = torch.cdist(coords_norm, coords_norm)

    # Get upper triangular (exclude diagonal)
    triu_mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
    pairwise_dists = dists[triu_mask]

    median_dist = torch.median(pairwise_dists).item()

    return median_dist


class CoordinateTransformer:
    """
    Stateful coordinate transformer that stores normalization parameters.

    Use this for consistent transformations between train/val/test sets.
    """

    def __init__(
        self,
        coord_type: Literal["planar", "spherical"] = "planar",
        lon_range: Optional[Tuple[float, float]] = None,
        lat_range: Optional[Tuple[float, float]] = None
    ):
        self.coord_type = coord_type
        self.lon_range = lon_range
        self.lat_range = lat_range
        self._fitted = False

    def fit(self, coords: torch.Tensor) -> "CoordinateTransformer":
        """
        Fit the transformer to training coordinates.

        Args:
            coords: [N, 2] tensor with (lon, lat) in degrees

        Returns:
            self (for chaining)
        """
        if self.lon_range is None:
            self.lon_range = (coords[:, 0].min().item(), coords[:, 0].max().item())
        if self.lat_range is None:
            self.lat_range = (coords[:, 1].min().item(), coords[:, 1].max().item())
        self._fitted = True
        return self

    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Transform coordinates.

        Args:
            coords: [N, 2] tensor with (lon, lat) in degrees

        Returns:
            Transformed coordinates
        """
        if self.coord_type == "spherical":
            return lonlat_to_cartesian3d(coords)
        else:
            if not self._fitted:
                raise ValueError("Transformer must be fitted before transform()")
            return normalize_lonlat_planar(coords, self.lon_range, self.lat_range)

    def fit_transform(self, coords: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        self.fit(coords)
        return self.transform(coords)

    @property
    def output_dim(self) -> int:
        """Output dimensionality after transformation."""
        return 3 if self.coord_type == "spherical" else 2


def prepare_gp_data(
    train_coords: torch.Tensor,
    train_targets: torch.Tensor,
    val_coords: Optional[torch.Tensor] = None,
    val_targets: Optional[torch.Tensor] = None,
    test_coords: Optional[torch.Tensor] = None,
    test_targets: Optional[torch.Tensor] = None,
    coord_type: Literal["planar", "spherical"] = "planar",
    normalize_targets: bool = True
) -> dict:
    """
    Prepare data for GP training with consistent transformations.

    Args:
        train_coords: [N_train, 2] training coordinates (lon, lat) in degrees
        train_targets: [N_train] or [N_train, D] training targets
        val_coords: Optional validation coordinates
        val_targets: Optional validation targets
        test_coords: Optional test coordinates
        test_targets: Optional test targets
        coord_type: "planar" or "spherical" coordinate transformation
        normalize_targets: Whether to normalize targets to [0, 1]

    Returns:
        Dictionary with transformed data and metadata
    """
    # Create transformer and fit on training data
    transformer = CoordinateTransformer(coord_type=coord_type)
    train_x = transformer.fit_transform(train_coords)

    # Transform targets
    train_y = train_targets.clone()
    y_min = y_max = None
    if normalize_targets:
        y_min = train_y.min().item()
        y_max = train_y.max().item()
        train_y = (train_y - y_min) / (y_max - y_min + 1e-8)

    result = {
        'train_x': train_x,
        'train_y': train_y,
        'transformer': transformer,
        'y_min': y_min,
        'y_max': y_max
    }

    if val_coords is not None:
        result['val_x'] = transformer.transform(val_coords)
        if val_targets is not None:
            result['val_y'] = val_targets.clone()
            if normalize_targets:
                result['val_y'] = (result['val_y'] - y_min) / (y_max - y_min + 1e-8)

    if test_coords is not None:
        result['test_x'] = transformer.transform(test_coords)
        if test_targets is not None:
            result['test_y'] = test_targets.clone()
            if normalize_targets:
                result['test_y'] = (result['test_y'] - y_min) / (y_max - y_min + 1e-8)

    return result
