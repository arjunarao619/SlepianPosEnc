"""
Spherical Wavelet Positional Encoder.

Based on FAIREarth implementation. Provides multi-scale spherical wavelet
features for geographic coordinate encoding.
"""
import torch
import numpy as np
import math

from .base_encoder import BaseLocationEncoder
from .get_mhat import spherical_wavelet_family


def fibonacci_sphere(num_points):
    """Generate evenly distributed points on a sphere using Fibonacci spiral."""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)


def cartesian_to_euler(x, y, z):
    """Convert Cartesian coordinates to Euler angles."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # inclination angle (from z-axis down)
    phi = np.arctan2(y, x)    # azimuthal angle (in x-y plane from x-axis)
    psi = phi          # yaw (rotation around z-axis)
    theta = theta      # pitch (rotation around y-axis)
    phi = 0            # roll (rotation around x-axis, set to 0)

    return np.degrees(psi), np.degrees(theta), np.degrees(phi)


def generate_sphere_grid(num_points):
    """Generate rotation grid on sphere using Fibonacci sampling."""
    points = fibonacci_sphere(num_points)
    euler_angles = [cartesian_to_euler(x, y, z) for x, y, z in points]
    return euler_angles


class Wavelets(BaseLocationEncoder):
    """
    Spherical wavelet positional encoder baseline.

    Encodes geographic coordinates using multi-scale spherical wavelets
    with rotations sampled over the sphere.

    Args:
        max_scale: Number of wavelet scales (default: 3)
        max_rotations: Number of rotation samples on sphere (default: 75)
        k_val: Wavelet frequency parameter (default: 6)
        scale_factor: Wavelet scale factor (default: 1.0)
        scale_shift: Scale shift for dilation (default: 1)
        dilation_step: Dilation step size (default: 6)
        wavelet_type: Type of wavelet ('butterfly', 'gabor', 'mexican_hat')

    Input: coords [B, 2] in degrees, (lon, lat)
    Output: features [B, n_features] where n_features = max_scale * max_rotations
    """

    def __init__(
        self,
        max_scale: int = 3,
        max_rotations: int = 75,
        k_val: int = 6,
        scale_factor: float = 1.0,
        scale_shift: int = 1,
        dilation_step: int = 6,
        wavelet_type: str = 'butterfly',
    ):
        super().__init__()

        self.max_scale = max_scale
        self.max_rotations = max_rotations
        self.k_val = k_val
        self.scale_factor = scale_factor
        self.scale_shift = scale_shift
        self.dilation_step = dilation_step
        self.wavelet_type = wavelet_type

        # Embedding dimension: scales * rotations
        self.embedding_dim = self.max_scale * self.max_rotations

        # Precompute rotation grid (Euler angles)
        self.rotation_vals = generate_sphere_grid(self.max_rotations)

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """
        Compute spherical wavelet features.

        Args:
            lonlat: [B, 2] tensor of [lon, lat] in degrees

        Returns:
            features: [B, n_features] spherical wavelet features
        """
        lon = lonlat[:, 0].unsqueeze(1)
        lat = lonlat[:, 1].unsqueeze(1)

        # Convert to spherical coordinates (radians)
        phi = torch.deg2rad(lon + 180)    # azimuth
        theta = torch.deg2rad(90 + lat)   # colatitude-like

        # Compute scales
        scales = [
            2 ** (-(j + self.scale_shift) / self.dilation_step)
            for j in range(0, self.max_scale)
        ]

        Y = []
        for dil_a in scales:
            for alpha_deg, beta_deg, gamma_deg in self.rotation_vals:
                alpha = math.radians(alpha_deg)
                beta = math.radians(beta_deg)
                gamma = math.radians(gamma_deg)

                # Compute wavelet response at this scale and rotation
                w = spherical_wavelet_family(
                    theta, phi,
                    dil_a=dil_a,
                    rot_rho=(alpha, beta, gamma),
                    k=self.k_val,
                    scale_factor=self.scale_factor,
                    wavelet_type=self.wavelet_type,
                ) / (dil_a ** 3)

                Y.append(w)

        # Stack into [B, n_features]
        Y = torch.stack(Y, dim=-1)
        Y = Y.to(dtype=lonlat.dtype, device=lonlat.device)

        return Y
