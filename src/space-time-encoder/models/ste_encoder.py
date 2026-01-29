# models/ste_encoder.py
import torch
import torch.nn as nn
from .fcnet import FCNet
from .baseline_time import time_embedding_functions
from .sphericalharmonics import SphericalHarmonics

class STEEncoder(nn.Module):
    """Space-Time Encoder combining spherical harmonics (spatial) with temporal encodings.

    Args:
        spatial_L: Spherical harmonics degree (output dim = L^2)
        temporal_type: 'dpss', 'legendre', 'fourier', 'triangle', 'monomial', 'time_copy', 'no_time'
        temporal_dim: Number of temporal basis functions
        combination: How to combine spatial/temporal ('concatenate', 'outer_product', 'hadamard')
        hidden_dim, num_layers, output_dim: FCNet head configuration
        ortho_weight_*: Regularization weights for orthogonality constraints
    """
    def __init__(self,
                 spatial_L=20,
                 temporal_type='legendre',
                 temporal_dim=40,
                 combination='concatenate',
                 hidden_dim=1024,
                 num_layers=4,
                 output_dim=8,
                 ortho_weight=0.0,
                 ortho_weight_space=0.0,
                 ortho_weight_time=0.0,
                 normality_flag=False,
                 ortho_exponent=1,
                 time_grad_penalty_weight=0.0,
                 # Projection size (per-stream)
                 projection_dim=None,
                 # DPSS params (only used if temporal_type='dpss')
                 dpss_N=1460,
                 dpss_NW=20,
                 dpss_optimized=False,
                 dpss_concentration_threshold=0.8,
                 dpss_K_dpss=None):
        super().__init__()

        # Spatial encoder (SH)
        self.spatial_encoder = SphericalHarmonics(
            legendre_polys=spatial_L,
            harmonics_calculation='analytic'
        )
        self.spatial_dim = self.spatial_encoder.embedding_dim

        # Temporal choice
        self.temporal_type = temporal_type
        self.combination = combination
        self.use_time = temporal_type != 'no_time'

        # Temporal encoder
        self.use_dpss = (temporal_type == 'dpss')
        if self.use_dpss:
            max_K = max(1, int(2 * dpss_NW) - 1)
            K_req = min(int(temporal_dim), max_K)
            if temporal_dim > max_K:
                print(f"[STE] DPSS: capping temporal_dim {temporal_dim} → {K_req} (max 2*NW-1).")
            if dpss_optimized:
                from .dpss_time import OptimizedDPSSTimeEncoder
                self.temporal_encoder = OptimizedDPSSTimeEncoder(
                    N=dpss_N, NW=dpss_NW, K_output=K_req,
                    concentration_threshold=dpss_concentration_threshold,
                    K_dpss=(int(dpss_K_dpss) if dpss_K_dpss is not None else max_K),
                )
            else:
                from .dpss_time import DPSSTimeEncoder
                self.temporal_encoder = DPSSTimeEncoder(
                    N=dpss_N, NW=dpss_NW, K=K_req,
                    concentration_threshold=dpss_concentration_threshold
                )
            self.temporal_dim = self.temporal_encoder.output_dim
        elif self.use_time:
            self.temporal_encoder = None
            self.temporal_dim = int(temporal_dim)
        else:
            self.temporal_encoder = None
            self.temporal_dim = 0  # no time stream at all

        # Per-stream projections
        proj_dim = (hidden_dim // 2) if projection_dim is None else int(projection_dim)
        if proj_dim <= 0:
            raise ValueError("projection_dim must be positive.")
        self.space_mlp = FCNet(self.spatial_dim, proj_dim, proj_dim, 2)
        self.time_mlp  = None if not self.use_time else FCNet(self.temporal_dim, proj_dim, proj_dim, 2)

        # Combine projected streams
        if not self.use_time:
            if combination != 'concatenate':
                raise NotImplementedError("With temporal_type='no_time', use combination='concatenate'.")
            combined_dim = proj_dim
        else:
            if combination == 'concatenate':
                combined_dim = 2 * proj_dim
            elif combination == 'outer_product':
                combined_dim = proj_dim * proj_dim
            elif combination == 'hadamard':
                combined_dim = proj_dim
            else:
                raise NotImplementedError(f"Combination {combination} not implemented")
        self.combined_dim = combined_dim
        self.pre_head_norm = nn.LayerNorm(combined_dim, elementwise_affine=False)

        # Prediction head
        self.head = FCNet(self.combined_dim, output_dim, hidden_dim, num_layers)

        # Regularization knobs (final layer α is the one used in Table 2)
        self.ortho_weight = float(ortho_weight)
        self.ortho_weight_space = float(ortho_weight_space)
        self.ortho_weight_time = float(ortho_weight_time)
        self.normality_flag = bool(normality_flag)
        self.ortho_exponent = int(ortho_exponent)
        self.time_grad_penalty_weight = float(time_grad_penalty_weight)

        # Caches for same-pass orthoreg
        self._cache_space_feats = None
        self._cache_time_feats  = None
        self._cache_final_feats = None

        print("[STE] Initialized:")
        print(f"  Spatial: SH L={spatial_L} → {self.spatial_dim} dims")
        print(f"  Temporal: {temporal_type} → {self.temporal_dim} dims (DPSS={self.use_dpss})")
        print(f"  Projections: space→{proj_dim}" + ("" if not self.use_time else f", time→{proj_dim}"))
        print(f"  Combine: {combination} → {combined_dim} dims before head")
        print(f"  Head: hidden_dim={hidden_dim}, num_layers={num_layers}, out={output_dim}")
        print(f"  α(final)={self.ortho_weight}, α(space)={self.ortho_weight_space}, α(time)={self.ortho_weight_time}")

    def encode_time(self, time_coords: torch.Tensor) -> torch.Tensor:
        if not self.use_time:
            # Not used; return empty tensor for clarity
            return time_coords.new_zeros((time_coords.shape[0], 0))
        if self.use_dpss:
            return self.temporal_encoder(time_coords)
        # Analytic families
        batch = time_coords.shape[0]
        out = time_coords.new_zeros((batch, self.temporal_dim))
        if self.temporal_type in time_embedding_functions:
            fn = time_embedding_functions[self.temporal_type]
            t = time_coords.squeeze(-1)
            for k in range(self.temporal_dim):
                out[:, k] = fn(k, t)
            return out
        raise ValueError(f"Unknown temporal encoding: {self.temporal_type}")

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, 3] = [lon, lat, time_norm]
        spatial_coords = coords[:, :2]
        time_coords    = coords[:, 2:3]

        s_raw = self.spatial_encoder(spatial_coords)
        s_feats = self.space_mlp(s_raw)
        self._cache_space_feats = s_feats

        if self.use_time:
            t_raw = self.encode_time(time_coords)
            t_feats = self.time_mlp(t_raw)
            self._cache_time_feats = t_feats
            if self.combination == 'concatenate':
                combined = torch.cat([s_feats, t_feats], dim=1)
            elif self.combination == 'outer_product':
                combined = torch.einsum('bi,bj->bij', s_feats, t_feats).flatten(1)
            elif self.combination == 'hadamard':
                combined = s_feats * t_feats
            else:
                raise NotImplementedError
        else:
            combined = s_feats

        combined = self.pre_head_norm(combined)
        penult = self.head.feats(combined)
        self._cache_final_feats = penult
        out = self.head.class_emb(penult)
        return out

    def get_regularization_loss(self, coords: torch.Tensor) -> torch.Tensor:
        # Final-layer orthogonality (α on last layer), optional per-stream penalties.
        def offdiag(feats):
            if feats is None or feats.numel() == 0:
                return coords.new_zeros(())
            B = feats.shape[0]
            C = (feats.T @ feats) / max(1, B)
            if self.normality_flag:
                I = torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
                P = C - I
            else:
                P = C - torch.diag(torch.diag(C))
            exp = max(1, int(self.ortho_exponent))
            return torch.linalg.norm(P, 'fro') ** exp

        wF, wS, wT = self.ortho_weight, self.ortho_weight_space, self.ortho_weight_time
        if (wF + wS + wT) == 0.0:
            return coords.new_zeros(())

        # Recompute features (no caching side-effects)
        s_raw = self.spatial_encoder(coords[:, :2])
        s_feats = self.space_mlp(s_raw)
        if self.use_time:
            t_raw = self.encode_time(coords[:, 2:3])
            t_feats = self.time_mlp(t_raw)
            if self.combination == 'concatenate':
                comb = torch.cat([s_feats, t_feats], dim=1)
            elif self.combination == 'outer_product':
                comb = torch.einsum('bi,bj->bij', s_feats, t_feats).flatten(1)
            else:
                comb = s_feats * t_feats
        else:
            t_feats = None
            comb = s_feats

        comb = self.pre_head_norm(comb)
        penult = self.head.feats(comb)

        reg = wF * offdiag(penult) + wS * offdiag(s_feats) + wT * offdiag(t_feats)
        return reg
