import torch
from torch import nn
import numpy as np
import math
from .common import _cal_freq_list

class GridAndSphere(nn.Module):
    def __init__(self, coord_dim=2, frequency_num=16,
                 max_radius=0.01, min_radius=0.00001,
                 freq_init="geometric", name="grid"):
        super(GridAndSphere, self).__init__()
        # Set class name based on variant for cleaner repr
        if name == "grid":
            GridAndSphere.__qualname__ = "Grid"; GridAndSphere.__name__ = "Grid"
        elif name == "spherec":
            GridAndSphere.__qualname__ = "SphereC"; GridAndSphere.__name__ = "SphereC"
        elif name == "spherecplus":
            GridAndSphere.__qualname__ = "SphereCPlus"; GridAndSphere.__name__ = "SphereCPlus"
        elif name == "spherem":
            GridAndSphere.__qualname__ = "SphereM"; GridAndSphere.__name__ = "SphereM"
        elif name == "spheremplus":
            GridAndSphere.__qualname__ = "SphereMPlus"; GridAndSphere.__name__ = "SphereMPlus"

        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.cal_freq_list()
        self.cal_freq_mat()
        self.name = name
        self.embedding_dim = self.cal_embedding_dim()
        self.n_features = self.embedding_dim

    def cal_embedding_dim(self):
        if self.name == "grid":
            return int(4 * self.frequency_num)
        elif self.name == "spherec":
            return int(6 * self.frequency_num)
        elif self.name == "spherecplus":
            return int(12 * self.frequency_num)
        elif self.name == "spherem":
            return int(10 * self.frequency_num)
        elif self.name == "spheremplus":
            return int(16 * self.frequency_num)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        self.freq_mat = np.repeat(freq_mat, 2, axis=1)

    def forward(self, coords: torch.Tensor):
        device, dtype = coords.device, coords.dtype
        N = coords.size(0)

        # Convert to radians for trig functions
        coords_rad = torch.deg2rad(coords)
        coords_np = coords_rad.detach().cpu().numpy()

        # Expand dims for broadcasting with frequency matrix
        coords_mat = np.expand_dims(coords_np[:, None, :], axis=3)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)
        coords_mat = np.repeat(coords_mat, 2, axis=4)
        spr_embeds = coords_mat * self.freq_mat

        if self.name == "grid":
            spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])
            spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])
            out = spr_embeds.reshape(N, -1)

        elif self.name == "spherec":
            lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
            lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)
            lon_sin, lon_cos = np.sin(lon), np.cos(lon)
            lat_sin, lat_cos = np.sin(lat), np.cos(lat)
            spr = np.concatenate([lat_sin, lat_cos * lon_cos, lat_cos * lon_sin], axis=-1)
            out = spr.reshape(N, -1)

        elif self.name == "spherecplus":
            lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
            lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)
            lon_sin, lon_cos = np.sin(lon), np.cos(lon)
            lat_sin, lat_cos = np.sin(lat), np.cos(lat)
            spr = np.concatenate([lat_sin, lat_cos, lon_sin, lon_cos, lat_cos * lon_cos, lat_cos * lon_sin], axis=-1)
            out = spr.reshape(N, -1)

        elif self.name == "spherem":
            lon_single = np.expand_dims(coords_np[:, None, 0, None, None], axis=2)
            lat_single = np.expand_dims(coords_np[:, None, 1, None, None], axis=2)
            lon_single_cos = np.cos(lon_single); lon_single_sin = np.sin(lon_single)
            lat_single_cos = np.cos(lat_single); lat_single_sin = np.sin(lat_single)

            lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
            lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)
            lon_sin, lon_cos = np.sin(lon), np.cos(lon)
            lat_sin, lat_cos = np.sin(lat), np.cos(lat)

            spr = np.concatenate([lat_sin, lat_cos * lon_single_cos, lat_single_cos * lon_cos,
                                  lat_cos * lon_single_sin, lat_single_cos * lon_sin], axis=-1)
            out = spr.reshape(N, -1)

        elif self.name == "spheremplus":
            lon_single = np.expand_dims(coords_np[:, None, 0, None, None], axis=2)
            lat_single = np.expand_dims(coords_np[:, None, 1, None, None], axis=2)
            lon_single_cos = np.cos(lon_single); lon_single_sin = np.sin(lon_single)
            lat_single_cos = np.cos(lat_single); lat_single_sin = np.sin(lat_single)

            lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
            lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)
            lon_sin, lon_cos = np.sin(lon), np.cos(lon)
            lat_sin, lat_cos = np.sin(lat), np.cos(lat)

            spr = np.concatenate([lat_sin, lat_cos, lon_sin, lon_cos,
                                  lat_cos * lon_single_cos, lat_single_cos * lon_cos,
                                  lat_cos * lon_single_sin, lat_single_cos * lon_sin], axis=-1)
            out = spr.reshape(N, -1)

        return torch.from_numpy(out).to(dtype=dtype, device=device)
