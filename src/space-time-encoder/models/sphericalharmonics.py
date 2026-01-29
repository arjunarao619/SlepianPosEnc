import sys
import os

import torch
from torch import nn

# Import SH from master file at /projects/arra4944/SlepianPosEnc/src/spherical_harmonics_ylm.py
# From models/ -> space-time-encoder/ -> src/
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from spherical_harmonics_ylm import SH


class SphericalHarmonics(nn.Module):
    """Spherical harmonics spatial encoder using pre-computed analytic equations."""

    def __init__(self, legendre_polys: int = 10, harmonics_calculation="analytic"):
        """
        Args:
            legendre_polys: Number of Legendre polynomial degrees (L).
                           Output dimension = L^2 basis functions.
            harmonics_calculation: Only 'analytic' is supported (uses pre-computed equations).
        """
        super(SphericalHarmonics, self).__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M

        if harmonics_calculation != "analytic":
            raise ValueError(
                f"Only 'analytic' harmonics calculation is supported. Got: {harmonics_calculation}"
            )

        self.SH = SH

    def forward(self, lonlat):
        """
        Compute spherical harmonic features for input coordinates.

        Args:
            lonlat: (N, 2) tensor with [longitude, latitude] in degrees

        Returns:
            (N, L^2) tensor of spherical harmonic features
        """
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # Convert degrees to radians (shift to standard spherical coords)
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)

        Y = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                y = self.SH(m, l, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)
                Y.append(y)

        return torch.stack(Y, dim=-1)
