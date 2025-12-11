# nn/models.py

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .config import MLPConfig, ReSIRENConfig, ResMLPConfig, SIRENConfig, GLUConfig


# =========================
# SIREN building blocks
# =========================

class Sine(nn.Module):
    """Standard SIREN sine activation: sin(omega_0 * x)."""
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class HSine(nn.Module):
    """H-SIREN activation: sin(omega_0 * sinh(2x))."""
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * torch.sinh(2.0 * x))


def siren_init_linear(linear: nn.Linear, omega_0: float, is_first: bool = False):
    """
    SIREN-style initialization.

    First layer:
        w ~ U(-1/in, 1/in)
    Other layers:
        w ~ U(-sqrt(6/in)/omega_0, +sqrt(6/in)/omega_0)
    """
    in_features = linear.in_features
    if is_first:
        bound = 1.0 / in_features
    else:
        bound = math.sqrt(6.0 / in_features) / omega_0

    with torch.no_grad():
        linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.uniform_(-bound, bound)


class ReSirenTrunk(nn.Module):
    """
    Residual SIREN network (ReSIREN) that operates on arbitrary features.

    Layers:
      1. in_dim -> hidden_dim, H-SIREN
      2..D-1.   hidden_dim -> hidden_dim, SIREN
      D.        hidden_dim -> out_dim, identity

    Residual connection in pre-activation space:
      h'_j = (h_j + h'_{j-1}) / 2  for j > 1
      z_{j+1} = A_j(h'_j)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
        depth: int = 16,
        omega_0: float = 30.0,
    ):
        super().__init__()
        assert depth >= 2, "depth must be at least 2 (first + last layer)."

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.omega_0 = omega_0

        linears = []

        # First layer: in_dim -> hidden_dim
        first = nn.Linear(in_dim, hidden_dim)
        siren_init_linear(first, omega_0=omega_0, is_first=True)
        linears.append(first)

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(depth - 2):
            lin = nn.Linear(hidden_dim, hidden_dim)
            siren_init_linear(lin, omega_0=omega_0, is_first=False)
            linears.append(lin)

        # Final layer: hidden_dim -> out_dim
        last = nn.Linear(hidden_dim, out_dim)
        siren_init_linear(last, omega_0=omega_0, is_first=False)
        linears.append(last)

        self.linears = nn.ModuleList(linears)

        self.h_sine = HSine(omega_0=omega_0)
        self.sine = Sine(omega_0=omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_dim) features.
        Returns: (..., out_dim) features.
        """
        z = x
        h_prev = None
        num_layers = len(self.linears)

        for j, linear in enumerate(self.linears, start=1):
            # Pre-activation
            h = linear(z)

            if h_prev is None:
                # First layer: no residual
                h_prime = h
            elif h.shape[-1] == h_prev.shape[-1]:
                # ReSIREN residual: h'_j = (h_j + h'_{j-1}) / 2
                # Only apply when dimensions match (not on last layer if out_dim != hidden_dim)
                h_prime = 0.5 * (h + h_prev)
            else:
                # Dimension mismatch (last layer with out_dim != hidden_dim): no residual
                h_prime = h

            h_prev = h_prime

            # Activation
            if j == 1:
                z = self.h_sine(h_prime)
            elif j < num_layers:
                z = self.sine(h_prime)
            else:
                # Last layer: identity
                z = h_prime

        return z


# =========================
# Simple MLP heads
# =========================

class LocationMLPRegressor(nn.Module):
    """Plain MLP regressor on top of a location encoder."""
    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        out = self.mlp(feats)
        return out.squeeze(-1)


class LocationMLPClassifier(nn.Module):
    """Plain MLP classifier on top of a location encoder."""
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        logits = self.mlp(feats)
        return logits  # shape (B, num_classes)


# =========================
# Residual SIREN heads
# =========================

class ResidualSirenRegressor(nn.Module):
    """
    ReSIREN-based regressor on top of a location encoder.

    Pipeline:
        coords -> encoder -> features -> ReSirenTrunk -> scalar regression output
    """
    def __init__(
        self,
        encoder: nn.Module,
        trunk_hidden_dim: int = 512,
        trunk_out_dim: int = 256,
        trunk_depth: int = 16,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.encoder = encoder

        in_dim = encoder.n_features
        self.trunk = ReSirenTrunk(
            in_dim=in_dim,
            hidden_dim=trunk_hidden_dim,
            out_dim=trunk_out_dim,
            depth=trunk_depth,
            omega_0=omega_0,
        )

        self.head = nn.Linear(trunk_out_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        h = self.trunk(feats)
        out = self.head(h)
        return out.squeeze(-1)


class ResidualSirenClassifier(nn.Module):
    """
    ReSIREN-based classifier on top of a location encoder.

    Pipeline:
        coords -> encoder -> features -> ReSirenTrunk -> logits (num_classes)
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        trunk_hidden_dim: int = 512,
        trunk_out_dim: int = 256,
        trunk_depth: int = 16,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.encoder = encoder

        in_dim = encoder.n_features
        self.trunk = ReSirenTrunk(
            in_dim=in_dim,
            hidden_dim=trunk_hidden_dim,
            out_dim=trunk_out_dim,
            depth=trunk_depth,
            omega_0=omega_0,
        )

        self.head = nn.Linear(trunk_out_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        h = self.trunk(feats)
        logits = self.head(h)
        return logits  # (B, num_classes)


# =========================
# Indexed variants (for cached feature encoders)
# =========================

class IndexedMLPRegressor(nn.Module):
    """
    MLP regressor for encoders that require (coords, indices) input.
    Used with CachedFeatureEncoder.
    """
    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        out = self.mlp(feats)
        return out.squeeze(-1)


class IndexedMLPClassifier(nn.Module):
    """
    MLP classifier for encoders that require (coords, indices) input.
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        logits = self.mlp(feats)
        return logits


class IndexedReSirenRegressor(nn.Module):
    """
    ReSIREN regressor for encoders that require (coords, indices) input.
    """
    def __init__(
        self,
        encoder: nn.Module,
        trunk_hidden_dim: int = 512,
        trunk_out_dim: int = 256,
        trunk_depth: int = 16,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.encoder = encoder

        in_dim = encoder.n_features
        self.trunk = ReSirenTrunk(
            in_dim=in_dim,
            hidden_dim=trunk_hidden_dim,
            out_dim=trunk_out_dim,
            depth=trunk_depth,
            omega_0=omega_0,
        )

        self.head = nn.Linear(trunk_out_dim, 1)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        h = self.trunk(feats)
        out = self.head(h)
        return out.squeeze(-1)


class IndexedReSirenClassifier(nn.Module):
    """
    ReSIREN classifier for encoders that require (coords, indices) input.
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        trunk_hidden_dim: int = 512,
        trunk_out_dim: int = 256,
        trunk_depth: int = 16,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.encoder = encoder

        in_dim = encoder.n_features
        self.trunk = ReSirenTrunk(
            in_dim=in_dim,
            hidden_dim=trunk_hidden_dim,
            out_dim=trunk_out_dim,
            depth=trunk_depth,
            omega_0=omega_0,
        )

        self.head = nn.Linear(trunk_out_dim, num_classes)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        h = self.trunk(feats)
        logits = self.head(h)
        return logits


# =========================
# ResMLP (Residual MLP)
# =========================

class ResBlock(nn.Module):
    """Residual block with ReLU: x + Linear(ReLU(Linear(x)))."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResMLPTrunk(nn.Module):
    """
    Residual MLP trunk.

    Architecture:
        in_dim -> hidden_dim (Linear + LayerNorm + ReLU)
        [ResBlock] × depth
        hidden_dim -> out_dim (Linear)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout=dropout) for _ in range(depth)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class ResMLPRegressor(nn.Module):
    """ResMLP-based regressor on top of a location encoder."""
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = ResMLPTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        h = self.trunk(feats)
        return self.head(h).squeeze(-1)


class ResMLPClassifier(nn.Module):
    """ResMLP-based classifier on top of a location encoder."""
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = ResMLPTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        h = self.trunk(feats)
        return self.head(h)


class IndexedResMLPRegressor(nn.Module):
    """ResMLP regressor for indexed encoders (coords, indices)."""
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = ResMLPTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        h = self.trunk(feats)
        return self.head(h).squeeze(-1)


class IndexedResMLPClassifier(nn.Module):
    """ResMLP classifier for indexed encoders (coords, indices)."""
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = ResMLPTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        h = self.trunk(feats)
        return self.head(h)


# =========================
# SIREN (Standard, no H-SIREN)
# =========================

class SirenTrunk(nn.Module):
    """
    Standard SIREN trunk (no H-SIREN on first layer).

    Uses different omega values for first vs hidden layers as per the original
    SIREN paper (Sitzmann et al.) and Russwurm et al.'s implementation:
        - First layer: omega_0_initial (default 30.0) for high-frequency mapping
        - Hidden layers: omega_0 (default 1.0) for stable gradients

    Architecture:
        Layer 1:      in_dim -> hidden_dim     (Sine activation, omega_0_initial)
        Layer 2..D-1: hidden_dim -> hidden_dim (Sine activation, omega_0)
        Layer D:      hidden_dim -> out_dim    (identity)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        omega_0: float = 1.0,          # For hidden layers (changed from 30.0)
        omega_0_initial: float = 30.0,  # For first layer only
    ):
        super().__init__()
        assert depth >= 2, "depth must be at least 2."

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.omega_0 = omega_0
        self.omega_0_initial = omega_0_initial

        linears = []

        # First layer: in_dim -> hidden_dim
        first = nn.Linear(in_dim, hidden_dim)
        siren_init_linear(first, omega_0=omega_0_initial, is_first=True)
        linears.append(first)

        # Hidden layers: hidden_dim -> hidden_dim
        # Use omega_0 (not omega_0_initial) for init
        for _ in range(depth - 2):
            lin = nn.Linear(hidden_dim, hidden_dim)
            siren_init_linear(lin, omega_0=omega_0, is_first=False)
            linears.append(lin)

        # Final layer: hidden_dim -> out_dim
        last = nn.Linear(hidden_dim, out_dim)
        siren_init_linear(last, omega_0=omega_0, is_first=False)
        linears.append(last)

        self.linears = nn.ModuleList(linears)

        # Separate activations for first layer vs hidden layers
        self.sine_initial = Sine(omega_0=omega_0_initial)
        self.sine = Sine(omega_0=omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        num_layers = len(self.linears)

        for j, linear in enumerate(self.linears, start=1):
            z = linear(z)
            if j == 1:
                # First layer uses omega_0_initial
                z = self.sine_initial(z)
            elif j < num_layers:
                # Hidden layers use omega_0
                z = self.sine(z)
            # Last layer: identity (no activation)

        return z


class SirenRegressor(nn.Module):
    """Standard SIREN regressor on top of a location encoder."""
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        omega_0: float = 1.0,
        omega_0_initial: float = 30.0,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = SirenTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            omega_0=omega_0,
            omega_0_initial=omega_0_initial,
        )
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        h = self.trunk(feats)
        return self.head(h).squeeze(-1)


class SirenClassifier(nn.Module):
    """Standard SIREN classifier on top of a location encoder."""
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        omega_0: float = 1.0,
        omega_0_initial: float = 30.0,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = SirenTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            omega_0=omega_0,
            omega_0_initial=omega_0_initial,
        )
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        h = self.trunk(feats)
        return self.head(h)


class IndexedSirenRegressor(nn.Module):
    """SIREN regressor for indexed encoders (coords, indices)."""
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        omega_0: float = 1.0,
        omega_0_initial: float = 30.0,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = SirenTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            omega_0=omega_0,
            omega_0_initial=omega_0_initial,
        )
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        h = self.trunk(feats)
        return self.head(h).squeeze(-1)


class IndexedSirenClassifier(nn.Module):
    """SIREN classifier for indexed encoders (coords, indices)."""
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 4,
        omega_0: float = 1.0,
        omega_0_initial: float = 30.0,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = SirenTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            omega_0=omega_0,
            omega_0_initial=omega_0_initial,
        )
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        h = self.trunk(feats)
        return self.head(h)


# =========================
# GLU-MLP (Gated Linear Units)
# =========================

class GLUBlock(nn.Module):
    """
    Gated Linear Unit block.

    Computes: sigmoid(Linear_gate(x)) * Linear_value(x)
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        value = self.value(x)
        return self.dropout(gate * value)


class GLUTrunk(nn.Module):
    """
    GLU-based MLP trunk.

    Architecture:
        in_dim -> hidden_dim (GLU block)
        [GLU block] × (depth - 1)
        hidden_dim -> out_dim (Linear)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth

        # Input GLU block
        self.input_block = GLUBlock(in_dim, hidden_dim, dropout=dropout)

        # Hidden GLU blocks
        self.blocks = nn.ModuleList([
            GLUBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(depth - 1)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_block(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class GLURegressor(nn.Module):
    """GLU-based regressor on top of a location encoder."""
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = GLUTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        h = self.trunk(feats)
        return self.head(h).squeeze(-1)


class GLUClassifier(nn.Module):
    """GLU-based classifier on top of a location encoder."""
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = GLUTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords)
        h = self.trunk(feats)
        return self.head(h)


class IndexedGLURegressor(nn.Module):
    """GLU regressor for indexed encoders (coords, indices)."""
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = GLUTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        h = self.trunk(feats)
        return self.head(h).squeeze(-1)


class IndexedGLUClassifier(nn.Module):
    """GLU classifier for indexed encoders (coords, indices)."""
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_features

        self.trunk = GLUTrunk(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(coords, indices)
        h = self.trunk(feats)
        return self.head(h)


# =========================
# Factory
# =========================

def build_location_model(
    encoder: nn.Module,
    task: str = "regression",        # "regression" or "classification"
    arch: str = "mlp",               # "mlp", "resiren", "resmlp", "siren", "glu"
    num_classes: Optional[int] = None,
    # Config objects (take precedence if provided):
    mlp_config: Optional["MLPConfig"] = None,
    resiren_config: Optional["ReSIRENConfig"] = None,
    resmlp_config: Optional["ResMLPConfig"] = None,
    siren_config: Optional["SIRENConfig"] = None,
    glu_config: Optional["GLUConfig"] = None,
    # Legacy individual params (used if config not provided):
    hidden_dim: int = 128,           # for MLP heads
    dropout: float = 0.1,
    resiren_hidden_dim: int = 512,
    resiren_out_dim: int = 256,
    resiren_depth: int = 16,
    resiren_omega0: float = 30.0,
) -> nn.Module:
    """
    Build a location model given an encoder, task type, and architecture name.

    Args:
        encoder: Location encoder module (must have `n_features` attribute).
        task: Either "regression" or "classification".
        arch: One of "mlp", "resiren", "resmlp", "siren", "glu".
        num_classes: Number of classes (required for classification).
        mlp_config: MLPConfig object for MLP architecture.
        resiren_config: ReSIRENConfig object for ReSIREN architecture.
        resmlp_config: ResMLPConfig object for ResMLP architecture.
        siren_config: SIRENConfig object for SIREN architecture.
        glu_config: GLUConfig object for GLU architecture.

    Returns:
        A PyTorch module for location-based prediction.

    Examples:
        # Using config objects
        from nn import MLPConfig, ResMLPConfig, SIRENConfig, GLUConfig
        model = build_location_model(encoder, arch="mlp", mlp_config=MLPConfig(hidden_dim=256))
        model = build_location_model(encoder, arch="resmlp", resmlp_config=ResMLPConfig(depth=4))
        model = build_location_model(encoder, arch="siren", siren_config=SIRENConfig(omega_0=10.0))
        model = build_location_model(encoder, arch="glu", glu_config=GLUConfig(depth=3))
    """
    task = task.lower()
    arch = arch.lower()

    # === MLP ===
    if arch == "mlp":
        if mlp_config is not None:
            _hidden_dim = mlp_config.hidden_dim
            _dropout = mlp_config.dropout
        else:
            _hidden_dim = hidden_dim
            _dropout = dropout

        if task == "regression":
            return LocationMLPRegressor(
                encoder=encoder,
                hidden_dim=_hidden_dim,
                dropout=_dropout,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return LocationMLPClassifier(
                encoder=encoder,
                num_classes=num_classes,
                hidden_dim=_hidden_dim,
                dropout=_dropout,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    # === ReSIREN ===
    elif arch == "resiren":
        if resiren_config is not None:
            _hidden = resiren_config.hidden_dim
            _out = resiren_config.out_dim
            _depth = resiren_config.depth
            _omega = resiren_config.omega_0
        else:
            _hidden = resiren_hidden_dim
            _out = resiren_out_dim
            _depth = resiren_depth
            _omega = resiren_omega0

        if task == "regression":
            return ResidualSirenRegressor(
                encoder=encoder,
                trunk_hidden_dim=_hidden,
                trunk_out_dim=_out,
                trunk_depth=_depth,
                omega_0=_omega,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return ResidualSirenClassifier(
                encoder=encoder,
                num_classes=num_classes,
                trunk_hidden_dim=_hidden,
                trunk_out_dim=_out,
                trunk_depth=_depth,
                omega_0=_omega,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    # === ResMLP ===
    elif arch == "resmlp":
        if resmlp_config is not None:
            _hidden = resmlp_config.hidden_dim
            _out = resmlp_config.out_dim
            _depth = resmlp_config.depth
            _dropout = resmlp_config.dropout
        else:
            # Defaults from ResMLPConfig
            _hidden = 256
            _out = 128
            _depth = 4
            _dropout = dropout

        if task == "regression":
            return ResMLPRegressor(
                encoder=encoder,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                dropout=_dropout,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return ResMLPClassifier(
                encoder=encoder,
                num_classes=num_classes,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                dropout=_dropout,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    # === SIREN (standard, no H-SIREN) ===
    elif arch == "siren":
        if siren_config is not None:
            _hidden = siren_config.hidden_dim
            _out = siren_config.out_dim
            _depth = siren_config.depth
            _omega = siren_config.omega_0
            _omega_initial = getattr(siren_config, 'omega_0_initial', 30.0)
        else:
            # Defaults: omega_0=1.0 for hidden layers, omega_0_initial=30.0 for first layer
            _hidden = 256
            _out = 128
            _depth = 4
            _omega = 1.0
            _omega_initial = 30.0

        if task == "regression":
            return SirenRegressor(
                encoder=encoder,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                omega_0=_omega,
                omega_0_initial=_omega_initial,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return SirenClassifier(
                encoder=encoder,
                num_classes=num_classes,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                omega_0=_omega,
                omega_0_initial=_omega_initial,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    # === GLU ===
    elif arch == "glu":
        if glu_config is not None:
            _hidden = glu_config.hidden_dim
            _out = glu_config.out_dim
            _depth = glu_config.depth
            _dropout = glu_config.dropout
        else:
            # Defaults from GLUConfig
            _hidden = 256
            _out = 128
            _depth = 3
            _dropout = dropout

        if task == "regression":
            return GLURegressor(
                encoder=encoder,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                dropout=_dropout,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return GLUClassifier(
                encoder=encoder,
                num_classes=num_classes,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                dropout=_dropout,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: mlp, resiren, resmlp, siren, glu")


def build_indexed_location_model(
    encoder: nn.Module,
    task: str = "regression",
    arch: str = "mlp",               # "mlp", "resiren", "resmlp", "siren", "glu"
    num_classes: Optional[int] = None,
    mlp_config: Optional["MLPConfig"] = None,
    resiren_config: Optional["ReSIRENConfig"] = None,
    resmlp_config: Optional["ResMLPConfig"] = None,
    siren_config: Optional["SIRENConfig"] = None,
    glu_config: Optional["GLUConfig"] = None,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    resiren_hidden_dim: int = 512,
    resiren_out_dim: int = 256,
    resiren_depth: int = 16,
    resiren_omega0: float = 30.0,
) -> nn.Module:
    """
    Build a location model for encoders that require (coords, indices) input.
    Used with CachedFeatureEncoder or similar indexed encoders.

    Same parameters as build_location_model.
    """
    task = task.lower()
    arch = arch.lower()

    # === MLP ===
    if arch == "mlp":
        if mlp_config is not None:
            _hidden_dim = mlp_config.hidden_dim
            _dropout = mlp_config.dropout
        else:
            _hidden_dim = hidden_dim
            _dropout = dropout

        if task == "regression":
            return IndexedMLPRegressor(
                encoder=encoder,
                hidden_dim=_hidden_dim,
                dropout=_dropout,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return IndexedMLPClassifier(
                encoder=encoder,
                num_classes=num_classes,
                hidden_dim=_hidden_dim,
                dropout=_dropout,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    # === ReSIREN ===
    elif arch == "resiren":
        if resiren_config is not None:
            _hidden = resiren_config.hidden_dim
            _out = resiren_config.out_dim
            _depth = resiren_config.depth
            _omega = resiren_config.omega_0
        else:
            _hidden = resiren_hidden_dim
            _out = resiren_out_dim
            _depth = resiren_depth
            _omega = resiren_omega0

        if task == "regression":
            return IndexedReSirenRegressor(
                encoder=encoder,
                trunk_hidden_dim=_hidden,
                trunk_out_dim=_out,
                trunk_depth=_depth,
                omega_0=_omega,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return IndexedReSirenClassifier(
                encoder=encoder,
                num_classes=num_classes,
                trunk_hidden_dim=_hidden,
                trunk_out_dim=_out,
                trunk_depth=_depth,
                omega_0=_omega,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    # === ResMLP ===
    elif arch == "resmlp":
        if resmlp_config is not None:
            _hidden = resmlp_config.hidden_dim
            _out = resmlp_config.out_dim
            _depth = resmlp_config.depth
            _dropout = resmlp_config.dropout
        else:
            _hidden = 256
            _out = 128
            _depth = 4
            _dropout = dropout

        if task == "regression":
            return IndexedResMLPRegressor(
                encoder=encoder,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                dropout=_dropout,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return IndexedResMLPClassifier(
                encoder=encoder,
                num_classes=num_classes,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                dropout=_dropout,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    # === SIREN ===
    elif arch == "siren":
        if siren_config is not None:
            _hidden = siren_config.hidden_dim
            _out = siren_config.out_dim
            _depth = siren_config.depth
            _omega = siren_config.omega_0
            _omega_initial = getattr(siren_config, 'omega_0_initial', 30.0)
        else:
            # Defaults: omega_0=1.0 for hidden layers, omega_0_initial=30.0 for first layer
            _hidden = 256
            _out = 128
            _depth = 4
            _omega = 1.0
            _omega_initial = 30.0

        if task == "regression":
            return IndexedSirenRegressor(
                encoder=encoder,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                omega_0=_omega,
                omega_0_initial=_omega_initial,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return IndexedSirenClassifier(
                encoder=encoder,
                num_classes=num_classes,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                omega_0=_omega,
                omega_0_initial=_omega_initial,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    # === GLU ===
    elif arch == "glu":
        if glu_config is not None:
            _hidden = glu_config.hidden_dim
            _out = glu_config.out_dim
            _depth = glu_config.depth
            _dropout = glu_config.dropout
        else:
            _hidden = 256
            _out = 128
            _depth = 3
            _dropout = dropout

        if task == "regression":
            return IndexedGLURegressor(
                encoder=encoder,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                dropout=_dropout,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification.")
            return IndexedGLUClassifier(
                encoder=encoder,
                num_classes=num_classes,
                hidden_dim=_hidden,
                out_dim=_out,
                depth=_depth,
                dropout=_dropout,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: mlp, resiren, resmlp, siren, glu")
