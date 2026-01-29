from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .config import MLPConfig, ReSIRENConfig, ResMLPConfig, SIRENConfig, GLUConfig


# SIREN building blocks

class Sine(nn.Module):
    """sin(omega_0 * x)"""
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class HSine(nn.Module):
    """H-SIREN: sin(omega_0 * sinh(2x))"""
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * torch.sinh(2.0 * x))


def siren_init_linear(linear: nn.Linear, omega_0: float, is_first: bool = False):
    """SIREN-style weight init."""
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
    """Residual SIREN with pre-activation averaging: h'_j = (h_j + h'_{j-1}) / 2"""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256,
                 depth: int = 16, omega_0: float = 30.0):
        super().__init__()
        assert depth >= 2

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.omega_0 = omega_0

        linears = []

        first = nn.Linear(in_dim, hidden_dim)
        siren_init_linear(first, omega_0=omega_0, is_first=True)
        linears.append(first)

        for _ in range(depth - 2):
            lin = nn.Linear(hidden_dim, hidden_dim)
            siren_init_linear(lin, omega_0=omega_0, is_first=False)
            linears.append(lin)

        last = nn.Linear(hidden_dim, out_dim)
        siren_init_linear(last, omega_0=omega_0, is_first=False)
        linears.append(last)

        self.linears = nn.ModuleList(linears)
        self.h_sine = HSine(omega_0=omega_0)
        self.sine = Sine(omega_0=omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        h_prev = None
        num_layers = len(self.linears)

        for j, linear in enumerate(self.linears, start=1):
            h = linear(z)

            if h_prev is None:
                h_prime = h
            elif h.shape[-1] == h_prev.shape[-1]:
                h_prime = 0.5 * (h + h_prev)
            else:
                h_prime = h

            h_prev = h_prime

            if j == 1:
                z = self.h_sine(h_prime)
            elif j < num_layers:
                z = self.sine(h_prime)
            else:
                z = h_prime

        return z


# MLP heads

class LocationMLPRegressor(nn.Module):
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
        return self.mlp(feats).squeeze(-1)


class LocationMLPClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
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
        return self.mlp(self.encoder(coords))


# Linear heads

class LinearRegressor(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(encoder.n_features, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.linear(self.encoder(coords)).squeeze(-1)


class LinearClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(encoder.n_features, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.linear(self.encoder(coords))


class IndexedLinearRegressor(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(encoder.n_features, 1)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.linear(self.encoder(coords, indices)).squeeze(-1)


class IndexedLinearClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(encoder.n_features, num_classes)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.linear(self.encoder(coords, indices))


# ReSIREN heads

class ResidualSirenRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, trunk_hidden_dim: int = 512,
                 trunk_out_dim: int = 256, trunk_depth: int = 16, omega_0: float = 30.0):
        super().__init__()
        self.encoder = encoder
        self.trunk = ReSirenTrunk(in_dim=encoder.n_features, hidden_dim=trunk_hidden_dim,
                                   out_dim=trunk_out_dim, depth=trunk_depth, omega_0=omega_0)
        self.head = nn.Linear(trunk_out_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords))).squeeze(-1)


class ResidualSirenClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, trunk_hidden_dim: int = 512,
                 trunk_out_dim: int = 256, trunk_depth: int = 16, omega_0: float = 30.0):
        super().__init__()
        self.encoder = encoder
        self.trunk = ReSirenTrunk(in_dim=encoder.n_features, hidden_dim=trunk_hidden_dim,
                                   out_dim=trunk_out_dim, depth=trunk_depth, omega_0=omega_0)
        self.head = nn.Linear(trunk_out_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords)))


# Indexed MLP/ReSIREN (for cached encoders)

class IndexedMLPRegressor(nn.Module):
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
        return self.mlp(self.encoder(coords, indices)).squeeze(-1)


class IndexedMLPClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
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
        return self.mlp(self.encoder(coords, indices))


class IndexedReSirenRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, trunk_hidden_dim: int = 512,
                 trunk_out_dim: int = 256, trunk_depth: int = 16, omega_0: float = 30.0):
        super().__init__()
        self.encoder = encoder
        self.trunk = ReSirenTrunk(in_dim=encoder.n_features, hidden_dim=trunk_hidden_dim,
                                   out_dim=trunk_out_dim, depth=trunk_depth, omega_0=omega_0)
        self.head = nn.Linear(trunk_out_dim, 1)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords, indices))).squeeze(-1)


class IndexedReSirenClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, trunk_hidden_dim: int = 512,
                 trunk_out_dim: int = 256, trunk_depth: int = 16, omega_0: float = 30.0):
        super().__init__()
        self.encoder = encoder
        self.trunk = ReSirenTrunk(in_dim=encoder.n_features, hidden_dim=trunk_hidden_dim,
                                   out_dim=trunk_out_dim, depth=trunk_depth, omega_0=omega_0)
        self.head = nn.Linear(trunk_out_dim, num_classes)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords, indices)))


# ResMLP

class ResBlock(nn.Module):
    """x + Linear(ReLU(Linear(x)))"""
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
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128,
                 depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class ResMLPRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int = 256, out_dim: int = 128,
                 depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.trunk = ResMLPTrunk(encoder.n_features, hidden_dim, out_dim, depth, dropout)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords))).squeeze(-1)


class ResMLPClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int = 256,
                 out_dim: int = 128, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.trunk = ResMLPTrunk(encoder.n_features, hidden_dim, out_dim, depth, dropout)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords)))


class IndexedResMLPRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int = 256, out_dim: int = 128,
                 depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.trunk = ResMLPTrunk(encoder.n_features, hidden_dim, out_dim, depth, dropout)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords, indices))).squeeze(-1)


class IndexedResMLPClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int = 256,
                 out_dim: int = 128, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.trunk = ResMLPTrunk(encoder.n_features, hidden_dim, out_dim, depth, dropout)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords, indices)))


# Standard SIREN (no H-SIREN)

class SirenTrunk(nn.Module):
    """Standard SIREN with separate omega for first layer (omega_0_initial) vs hidden (omega_0)."""
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128,
                 depth: int = 4, omega_0: float = 1.0, omega_0_initial: float = 30.0):
        super().__init__()
        assert depth >= 2

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.omega_0 = omega_0
        self.omega_0_initial = omega_0_initial

        linears = []

        first = nn.Linear(in_dim, hidden_dim)
        siren_init_linear(first, omega_0=omega_0_initial, is_first=True)
        linears.append(first)

        for _ in range(depth - 2):
            lin = nn.Linear(hidden_dim, hidden_dim)
            siren_init_linear(lin, omega_0=omega_0, is_first=False)
            linears.append(lin)

        last = nn.Linear(hidden_dim, out_dim)
        siren_init_linear(last, omega_0=omega_0, is_first=False)
        linears.append(last)

        self.linears = nn.ModuleList(linears)
        self.sine_initial = Sine(omega_0=omega_0_initial)
        self.sine = Sine(omega_0=omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        num_layers = len(self.linears)

        for j, linear in enumerate(self.linears, start=1):
            z = linear(z)
            if j == 1:
                z = self.sine_initial(z)
            elif j < num_layers:
                z = self.sine(z)

        return z


class SirenRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int = 256, out_dim: int = 128,
                 depth: int = 4, omega_0: float = 1.0, omega_0_initial: float = 30.0):
        super().__init__()
        self.encoder = encoder
        self.trunk = SirenTrunk(encoder.n_features, hidden_dim, out_dim, depth, omega_0, omega_0_initial)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords))).squeeze(-1)


class SirenClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int = 256,
                 out_dim: int = 128, depth: int = 4, omega_0: float = 1.0, omega_0_initial: float = 30.0):
        super().__init__()
        self.encoder = encoder
        self.trunk = SirenTrunk(encoder.n_features, hidden_dim, out_dim, depth, omega_0, omega_0_initial)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords)))


class IndexedSirenRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int = 256, out_dim: int = 128,
                 depth: int = 4, omega_0: float = 1.0, omega_0_initial: float = 30.0):
        super().__init__()
        self.encoder = encoder
        self.trunk = SirenTrunk(encoder.n_features, hidden_dim, out_dim, depth, omega_0, omega_0_initial)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords, indices))).squeeze(-1)


class IndexedSirenClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int = 256,
                 out_dim: int = 128, depth: int = 4, omega_0: float = 1.0, omega_0_initial: float = 30.0):
        super().__init__()
        self.encoder = encoder
        self.trunk = SirenTrunk(encoder.n_features, hidden_dim, out_dim, depth, omega_0, omega_0_initial)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords, indices)))


# GLU-MLP

class GLUBlock(nn.Module):
    """sigmoid(Wx) * Vx"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.sigmoid(self.gate(x)) * self.value(x))


class GLUTrunk(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128,
                 depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth

        self.input_block = GLUBlock(in_dim, hidden_dim, dropout=dropout)
        self.blocks = nn.ModuleList([GLUBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(depth - 1)])
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_block(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class GLURegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int = 256, out_dim: int = 128,
                 depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.trunk = GLUTrunk(encoder.n_features, hidden_dim, out_dim, depth, dropout)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords))).squeeze(-1)


class GLUClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int = 256,
                 out_dim: int = 128, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.trunk = GLUTrunk(encoder.n_features, hidden_dim, out_dim, depth, dropout)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords)))


class IndexedGLURegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int = 256, out_dim: int = 128,
                 depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.trunk = GLUTrunk(encoder.n_features, hidden_dim, out_dim, depth, dropout)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords, indices))).squeeze(-1)


class IndexedGLUClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim: int = 256,
                 out_dim: int = 128, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.trunk = GLUTrunk(encoder.n_features, hidden_dim, out_dim, depth, dropout)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.encoder(coords, indices)))


# Factory functions

def build_location_model(
    encoder: nn.Module,
    task: str = "regression",
    arch: str = "mlp",
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
    """Build a location model. arch: linear, mlp, resiren, resmlp, siren, glu"""
    task = task.lower()
    arch = arch.lower()

    if arch == "linear":
        if task == "regression":
            return LinearRegressor(encoder=encoder)
        elif task == "classification":
            return LinearClassifier(encoder=encoder, num_classes=num_classes)

    elif arch == "mlp":
        _hd = mlp_config.hidden_dim if mlp_config else hidden_dim
        _do = mlp_config.dropout if mlp_config else dropout
        if task == "regression":
            return LocationMLPRegressor(encoder, _hd, _do)
        elif task == "classification":
            return LocationMLPClassifier(encoder, num_classes, _hd, _do)

    elif arch == "resiren":
        if resiren_config:
            _h, _o, _d, _w = resiren_config.hidden_dim, resiren_config.out_dim, resiren_config.depth, resiren_config.omega_0
        else:
            _h, _o, _d, _w = resiren_hidden_dim, resiren_out_dim, resiren_depth, resiren_omega0
        if task == "regression":
            return ResidualSirenRegressor(encoder, _h, _o, _d, _w)
        elif task == "classification":
            return ResidualSirenClassifier(encoder, num_classes, _h, _o, _d, _w)

    elif arch == "resmlp":
        if resmlp_config:
            _h, _o, _d, _do = resmlp_config.hidden_dim, resmlp_config.out_dim, resmlp_config.depth, resmlp_config.dropout
        else:
            _h, _o, _d, _do = 256, 128, 4, dropout
        if task == "regression":
            return ResMLPRegressor(encoder, _h, _o, _d, _do)
        elif task == "classification":
            return ResMLPClassifier(encoder, num_classes, _h, _o, _d, _do)

    elif arch == "siren":
        if siren_config:
            _h, _o, _d, _w, _wi = siren_config.hidden_dim, siren_config.out_dim, siren_config.depth, siren_config.omega_0, getattr(siren_config, 'omega_0_initial', 30.0)
        else:
            _h, _o, _d, _w, _wi = 256, 128, 4, 1.0, 30.0
        if task == "regression":
            return SirenRegressor(encoder, _h, _o, _d, _w, _wi)
        elif task == "classification":
            return SirenClassifier(encoder, num_classes, _h, _o, _d, _w, _wi)

    elif arch == "glu":
        if glu_config:
            _h, _o, _d, _do = glu_config.hidden_dim, glu_config.out_dim, glu_config.depth, glu_config.dropout
        else:
            _h, _o, _d, _do = 256, 128, 3, dropout
        if task == "regression":
            return GLURegressor(encoder, _h, _o, _d, _do)
        elif task == "classification":
            return GLUClassifier(encoder, num_classes, _h, _o, _d, _do)

    else:
        raise ValueError(f"Unknown arch: {arch}")

    raise ValueError(f"Unknown task: {task}")


def build_indexed_location_model(
    encoder: nn.Module,
    task: str = "regression",
    arch: str = "mlp",
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
    """Build a location model for indexed encoders (coords, indices)."""
    task = task.lower()
    arch = arch.lower()

    if arch == "linear":
        if task == "regression":
            return IndexedLinearRegressor(encoder=encoder)
        elif task == "classification":
            return IndexedLinearClassifier(encoder=encoder, num_classes=num_classes)

    elif arch == "mlp":
        _hd = mlp_config.hidden_dim if mlp_config else hidden_dim
        _do = mlp_config.dropout if mlp_config else dropout
        if task == "regression":
            return IndexedMLPRegressor(encoder, _hd, _do)
        elif task == "classification":
            return IndexedMLPClassifier(encoder, num_classes, _hd, _do)

    elif arch == "resiren":
        if resiren_config:
            _h, _o, _d, _w = resiren_config.hidden_dim, resiren_config.out_dim, resiren_config.depth, resiren_config.omega_0
        else:
            _h, _o, _d, _w = resiren_hidden_dim, resiren_out_dim, resiren_depth, resiren_omega0
        if task == "regression":
            return IndexedReSirenRegressor(encoder, _h, _o, _d, _w)
        elif task == "classification":
            return IndexedReSirenClassifier(encoder, num_classes, _h, _o, _d, _w)

    elif arch == "resmlp":
        if resmlp_config:
            _h, _o, _d, _do = resmlp_config.hidden_dim, resmlp_config.out_dim, resmlp_config.depth, resmlp_config.dropout
        else:
            _h, _o, _d, _do = 256, 128, 4, dropout
        if task == "regression":
            return IndexedResMLPRegressor(encoder, _h, _o, _d, _do)
        elif task == "classification":
            return IndexedResMLPClassifier(encoder, num_classes, _h, _o, _d, _do)

    elif arch == "siren":
        if siren_config:
            _h, _o, _d, _w, _wi = siren_config.hidden_dim, siren_config.out_dim, siren_config.depth, siren_config.omega_0, getattr(siren_config, 'omega_0_initial', 30.0)
        else:
            _h, _o, _d, _w, _wi = 256, 128, 4, 1.0, 30.0
        if task == "regression":
            return IndexedSirenRegressor(encoder, _h, _o, _d, _w, _wi)
        elif task == "classification":
            return IndexedSirenClassifier(encoder, num_classes, _h, _o, _d, _w, _wi)

    elif arch == "glu":
        if glu_config:
            _h, _o, _d, _do = glu_config.hidden_dim, glu_config.out_dim, glu_config.depth, glu_config.dropout
        else:
            _h, _o, _d, _do = 256, 128, 3, dropout
        if task == "regression":
            return IndexedGLURegressor(encoder, _h, _o, _d, _do)
        elif task == "classification":
            return IndexedGLUClassifier(encoder, num_classes, _h, _o, _d, _do)

    else:
        raise ValueError(f"Unknown arch: {arch}")

    raise ValueError(f"Unknown task: {task}")
