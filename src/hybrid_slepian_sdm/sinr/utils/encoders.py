"""Positional encoding classes and functions."""

import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from spherical_harmonics_ylm import SH as SH_analytic

from .core import normalize_coords, bilinear_interpolate


class BaselineEncoderAdapter:
    """Adapter for pe_baselines encoders."""

    def __init__(self, encoder, device='cuda', trainable=False):
        self.encoder = encoder.to(device)
        self.device = device
        self.embedding_dim = encoder.embedding_dim
        self.trainable = trainable
        if not trainable:
            self.encoder.eval()

    def encode(self, locs, normalize=False):
        """Encode locs [N,2] in [-1,1] to features."""
        locs_deg = locs.clone().to(self.device)
        locs_deg[:, 0] *= 180.0
        locs_deg[:, 1] *= 90.0

        if self.trainable:
            return self.encoder(locs_deg)
        with torch.no_grad():
            return self.encoder(locs_deg)

    def parameters(self):
        if self.trainable:
            return self.encoder.parameters()
        return iter([])


class PlanarRFFEncoderWrapper(nn.Module):
    """PlanarRFF wrapper with embedding_dim property."""

    def __init__(self, input_dim=2, num_features=2000, lengthscale=0.1,
                 kernel_type='matern52', seed=42):
        super().__init__()
        from pe_baselines.gp.planar_rff import PlanarRFF
        self.rff = PlanarRFF(input_dim=input_dim, num_features=num_features,
                             lengthscale=lengthscale, kernel_type=kernel_type, seed=seed)
        self.embedding_dim = num_features

    def forward(self, coords):
        return self.rff(torch.deg2rad(coords))


class DeepRFFEncoderWrapper(nn.Module):
    """DeepRFFLayer wrapper with embedding_dim property."""

    def __init__(self, input_dim=2, hidden_dim=1000, output_dim=64,
                 lengthscale=1.0, nu=2.5, seed=42):
        super().__init__()
        from pe_baselines.gp.planar_rff import DeepRFFLayer
        self.layer = DeepRFFLayer(input_dim=input_dim, hidden_dim=hidden_dim,
                                  output_dim=output_dim, lengthscale=lengthscale,
                                  nu=nu, seed=seed)
        self.embedding_dim = output_dim

    def forward(self, coords):
        return self.layer(torch.deg2rad(coords))


class CoordEncoder:
    """Main coordinate encoder supporting multiple encoding types."""

    def __init__(self, input_enc, raster=None, L=10, slepian_raster=None,
                 slepian_data=None, baseline_params=None):
        self.input_enc = input_enc
        self.raster = raster
        self.L = L
        self.slepian_raster = slepian_raster
        self.slepian_data = slepian_data
        self.baseline_adapter = None

        if input_enc.startswith('baseline_'):
            self.baseline_adapter = self._create_baseline_encoder(input_enc, baseline_params)

    def _create_baseline_encoder(self, input_enc, params):
        from pe_baselines import (Direct, Wrap, Cartesian3D, Grid, Theory,
                                  SphereC, SphereCPlus, SphereM, SphereMPlus)

        params = params or {}
        enc_type = input_enc.replace('baseline_', '')
        device = params.get('device', 'cuda')

        if enc_type == 'direct':
            encoder = Direct()
        elif enc_type == 'wrap':
            encoder = Wrap()
        elif enc_type == 'cartesian3d':
            encoder = Cartesian3D()
        elif enc_type == 'grid':
            encoder = Grid(frequency_num=params.get('baseline_freq_num', 16))
        elif enc_type == 'theory':
            encoder = Theory(frequency_num=params.get('baseline_freq_num', 16))
        elif enc_type == 'spherec':
            encoder = SphereC(frequency_num=params.get('baseline_freq_num', 16))
        elif enc_type == 'spherecplus':
            encoder = SphereCPlus(frequency_num=params.get('baseline_freq_num', 16))
        elif enc_type == 'spherem':
            encoder = SphereM(frequency_num=params.get('baseline_freq_num', 16))
        elif enc_type == 'spheremplus':
            encoder = SphereMPlus(frequency_num=params.get('baseline_freq_num', 16))
        elif enc_type == 'planar_rff':
            encoder = PlanarRFFEncoderWrapper(
                num_features=params.get('baseline_rff_features', 2000),
                lengthscale=params.get('baseline_rff_lengthscale', 0.1),
                kernel_type=params.get('baseline_rff_kernel', 'matern52'))
        elif enc_type == 'deep_rff':
            encoder = DeepRFFEncoderWrapper(
                hidden_dim=params.get('baseline_deep_rff_hidden', 1000),
                output_dim=params.get('baseline_deep_rff_output', 64),
                lengthscale=params.get('baseline_rff_lengthscale', 1.0),
                nu=params.get('baseline_rff_nu', 2.5))
            return BaselineEncoderAdapter(encoder, device=device, trainable=True)
        else:
            raise NotImplementedError(f'Unknown baseline encoder: {enc_type}')

        return BaselineEncoderAdapter(encoder, device=device, trainable=False)

    def encode(self, locs, normalize=False):
        from .slepian import encode_loc_slepian_direct

        if normalize:
            locs = normalize_coords(locs)

        if self.input_enc.startswith('baseline_'):
            return self.baseline_adapter.encode(locs, normalize=normalize)
        elif self.input_enc == 'sin_cos':
            return encode_loc(locs)
        elif self.input_enc == 'env':
            return bilinear_interpolate(locs, self.raster)
        elif self.input_enc == 'sin_cos_env':
            return torch.cat((encode_loc(locs), bilinear_interpolate(locs, self.raster)), 1)
        elif self.input_enc == 'sh':
            return encode_loc_sh(locs, L=self.L)
        elif self.input_enc == 'slepian':
            return bilinear_interpolate(locs, self.slepian_raster)
        elif self.input_enc == 'slepian_env':
            return torch.cat((bilinear_interpolate(locs, self.slepian_raster),
                              bilinear_interpolate(locs, self.raster)), 1)
        elif self.input_enc == 'slepian_direct':
            return encode_loc_slepian_direct(locs, self.slepian_data)
        elif self.input_enc == 'slepian_direct_env':
            return torch.cat((encode_loc_slepian_direct(locs, self.slepian_data),
                              bilinear_interpolate(locs, self.raster)), 1)
        else:
            raise NotImplementedError('Unknown input encoding.')


def encode_loc(loc_ip, concat_dim=1):
    """Sinusoidal encoding: [sin(pi*x), cos(pi*x)]."""
    return torch.cat((torch.sin(math.pi * loc_ip), torch.cos(math.pi * loc_ip)), concat_dim)


def encode_loc_sh(loc_ip, L=10):
    """Spherical harmonics encoding. Returns (N, L^2) features."""
    lon_norm = loc_ip[:, 0]
    lat_norm = loc_ip[:, 1]

    phi = (lon_norm + 1) * math.pi
    theta = (1 - lat_norm) * (math.pi / 2)
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
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
