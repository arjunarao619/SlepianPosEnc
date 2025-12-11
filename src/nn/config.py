# nn/config.py
"""
Configuration dataclasses for neural network architectures.

Usage:
    from nn import MLPConfig, ReSIRENConfig

    # Use defaults
    mlp_cfg = MLPConfig()

    # Override specific values
    resiren_cfg = ReSIRENConfig(depth=8, omega_0=10.0)
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MLPConfig:
    """Configuration for MLP head architecture.

    Architecture:
        in_dim -> hidden_dim (Linear + ReLU + Dropout)
        hidden_dim -> hidden_dim // 2 (Linear + ReLU + Dropout)
        hidden_dim // 2 -> output_dim (Linear)
    """
    hidden_dim: int = 128
    dropout: float = 0.1


@dataclass
class ReSIRENConfig:
    """Configuration for Residual SIREN (ReSIREN) trunk architecture.

    Architecture:
        Layer 1:      in_dim -> hidden_dim     (H-SIREN activation)
        Layer 2..D-1: hidden_dim -> hidden_dim (SIREN activation + residual)
        Layer D:      hidden_dim -> out_dim    (identity)

    The residual connection uses pre-activation averaging:
        h'_j = (h_j + h'_{j-1}) / 2

    Attributes:
        hidden_dim: Width of hidden layers in the trunk.
        out_dim: Output dimension of the trunk (before final head).
        depth: Total number of layers (must be >= 2).
        omega_0: Frequency scaling for sine activations.
    """
    hidden_dim: int = 512
    out_dim: int = 256
    depth: int = 16
    omega_0: float = 30.0


# =============================================================================
# Default instances for convenience
# =============================================================================

MLP_DEFAULT = MLPConfig()
RESIREN_DEFAULT = ReSIRENConfig()


# =============================================================================
# Presets for different use cases
# =============================================================================

# Lightweight configs for quick experiments
MLP_SMALL = MLPConfig(hidden_dim=64, dropout=0.1)
RESIREN_SMALL = ReSIRENConfig(hidden_dim=256, out_dim=128, depth=8, omega_0=30.0)

# Larger configs for final benchmarks
MLP_LARGE = MLPConfig(hidden_dim=256, dropout=0.1)
RESIREN_LARGE = ReSIRENConfig(hidden_dim=1024, out_dim=512, depth=24, omega_0=30.0)

# Geographic-tuned: lower omega_0 for smoother spatial signals
RESIREN_GEO = ReSIRENConfig(hidden_dim=512, out_dim=256, depth=16, omega_0=10.0)


# =============================================================================
# New Architecture Configs
# =============================================================================

@dataclass
class ResMLPConfig:
    """Configuration for Residual MLP architecture.

    Architecture:
        in_dim -> hidden_dim (Linear + LayerNorm + ReLU)
        [ResBlock] × depth
        hidden_dim -> out_dim (Linear)

    ResBlock: x + Linear(ReLU(Linear(x)))

    Attributes:
        hidden_dim: Width of hidden layers.
        out_dim: Output dimension before final head.
        depth: Number of residual blocks.
        dropout: Dropout rate between blocks.
    """
    hidden_dim: int = 256
    out_dim: int = 128
    depth: int = 4
    dropout: float = 0.1


@dataclass
class SIRENConfig:
    """Configuration for standard SIREN architecture (no H-SIREN).

    Architecture:
        Layer 1:      in_dim -> hidden_dim     (Sine activation, omega_0_initial)
        Layer 2..D-1: hidden_dim -> hidden_dim (Sine activation, omega_0)
        Layer D:      hidden_dim -> out_dim    (identity)

    Uses different omega values for first vs hidden layers as per the original
    SIREN paper (Sitzmann et al.) and Russwurm et al.'s implementation:
        - First layer: omega_0_initial (default 30.0) for high-frequency mapping
        - Hidden layers: omega_0 (default 1.0) for stable gradients

    Attributes:
        hidden_dim: Width of hidden layers.
        out_dim: Output dimension before final head.
        depth: Total number of layers (must be >= 2).
        omega_0: Frequency scaling for hidden layer sine activations (default 1.0).
        omega_0_initial: Frequency scaling for first layer (default 30.0).
    """
    hidden_dim: int = 256
    out_dim: int = 128
    depth: int = 4
    omega_0: float = 1.0           # For hidden layers (changed from 30.0)
    omega_0_initial: float = 30.0  # For first layer only


@dataclass
class GLUConfig:
    """Configuration for Gated Linear Unit MLP architecture.

    Architecture:
        in_dim -> hidden_dim (GLU block)
        [GLU block] × depth
        hidden_dim -> out_dim (Linear)

    GLU block: sigmoid(Linear(x)) * Linear(x)

    Attributes:
        hidden_dim: Width of hidden layers.
        out_dim: Output dimension before final head.
        depth: Number of GLU blocks.
        dropout: Dropout rate between blocks.
    """
    hidden_dim: int = 256
    out_dim: int = 128
    depth: int = 3
    dropout: float = 0.1


# Default instances
RESMLP_DEFAULT = ResMLPConfig()
SIREN_DEFAULT = SIRENConfig()
GLU_DEFAULT = GLUConfig()

# Presets for new architectures
RESMLP_SMALL = ResMLPConfig(hidden_dim=128, out_dim=64, depth=2, dropout=0.1)
RESMLP_LARGE = ResMLPConfig(hidden_dim=512, out_dim=256, depth=6, dropout=0.1)

SIREN_SMALL = SIRENConfig(hidden_dim=128, out_dim=64, depth=2, omega_0=1.0, omega_0_initial=30.0)
SIREN_LARGE = SIRENConfig(hidden_dim=512, out_dim=256, depth=8, omega_0=1.0, omega_0_initial=30.0)
SIREN_GEO = SIRENConfig(hidden_dim=256, out_dim=128, depth=4, omega_0=1.0, omega_0_initial=10.0)

GLU_SMALL = GLUConfig(hidden_dim=128, out_dim=64, depth=2, dropout=0.1)
GLU_LARGE = GLUConfig(hidden_dim=512, out_dim=256, depth=5, dropout=0.1)
