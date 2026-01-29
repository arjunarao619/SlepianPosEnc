"""Neural network configuration dataclasses."""

from dataclasses import dataclass


@dataclass
class MLPConfig:
    """MLP head: in -> hidden -> hidden//2 -> out"""
    hidden_dim: int = 128
    dropout: float = 0.1


@dataclass
class ReSIRENConfig:
    """Residual SIREN trunk with pre-activation averaging."""
    hidden_dim: int = 512
    out_dim: int = 256
    depth: int = 16
    omega_0: float = 30.0


@dataclass
class ResMLPConfig:
    """Residual MLP: x + Linear(ReLU(Linear(x)))"""
    hidden_dim: int = 256
    out_dim: int = 128
    depth: int = 4
    dropout: float = 0.1


@dataclass
class SIRENConfig:
    """Standard SIREN with separate omega for first layer."""
    hidden_dim: int = 256
    out_dim: int = 128
    depth: int = 4
    omega_0: float = 1.0           # hidden layers
    omega_0_initial: float = 30.0  # first layer


@dataclass
class GLUConfig:
    """Gated Linear Unit MLP: sigmoid(Wx) * Vx"""
    hidden_dim: int = 256
    out_dim: int = 128
    depth: int = 3
    dropout: float = 0.1


# defaults
MLP_DEFAULT = MLPConfig()
RESIREN_DEFAULT = ReSIRENConfig()
RESMLP_DEFAULT = ResMLPConfig()
SIREN_DEFAULT = SIRENConfig()
GLU_DEFAULT = GLUConfig()

# small versions
MLP_SMALL = MLPConfig(hidden_dim=64)
RESIREN_SMALL = ReSIRENConfig(hidden_dim=256, out_dim=128, depth=8)
RESMLP_SMALL = ResMLPConfig(hidden_dim=128, out_dim=64, depth=2)
SIREN_SMALL = SIRENConfig(hidden_dim=128, out_dim=64, depth=2)
GLU_SMALL = GLUConfig(hidden_dim=128, out_dim=64, depth=2)

# large r versions
MLP_LARGE = MLPConfig(hidden_dim=256)
RESIREN_LARGE = ReSIRENConfig(hidden_dim=1024, out_dim=512, depth=24)
RESMLP_LARGE = ResMLPConfig(hidden_dim=512, out_dim=256, depth=6)
SIREN_LARGE = SIRENConfig(hidden_dim=512, out_dim=256, depth=8)
GLU_LARGE = GLUConfig(hidden_dim=512, out_dim=256, depth=5)

# geo - lower omega for smoother spatial signals
RESIREN_GEO = ReSIRENConfig(hidden_dim=512, out_dim=256, depth=16, omega_0=10.0)
SIREN_GEO = SIRENConfig(hidden_dim=256, out_dim=128, depth=4, omega_0_initial=10.0)
