# nn/__init__.py
"""
Neural network architectures for location encoding.

Exports:
    - build_location_model: Factory function to create models
    - build_indexed_location_model: Factory for indexed encoders
    - Config classes: MLPConfig, ReSIRENConfig, ResMLPConfig, SIRENConfig, GLUConfig
    - Presets for each architecture
"""

from .models import build_location_model, build_indexed_location_model
from .config import (
    # MLP configs
    MLPConfig,
    MLP_DEFAULT,
    MLP_SMALL,
    MLP_LARGE,
    # ReSIREN configs
    ReSIRENConfig,
    RESIREN_DEFAULT,
    RESIREN_SMALL,
    RESIREN_LARGE,
    RESIREN_GEO,
    # ResMLP configs
    ResMLPConfig,
    RESMLP_DEFAULT,
    RESMLP_SMALL,
    RESMLP_LARGE,
    # SIREN configs
    SIRENConfig,
    SIREN_DEFAULT,
    SIREN_SMALL,
    SIREN_LARGE,
    SIREN_GEO,
    # GLU configs
    GLUConfig,
    GLU_DEFAULT,
    GLU_SMALL,
    GLU_LARGE,
)

__all__ = [
    # Factory functions
    "build_location_model",
    "build_indexed_location_model",
    # MLP
    "MLPConfig",
    "MLP_DEFAULT",
    "MLP_SMALL",
    "MLP_LARGE",
    # ReSIREN
    "ReSIRENConfig",
    "RESIREN_DEFAULT",
    "RESIREN_SMALL",
    "RESIREN_LARGE",
    "RESIREN_GEO",
    # ResMLP
    "ResMLPConfig",
    "RESMLP_DEFAULT",
    "RESMLP_SMALL",
    "RESMLP_LARGE",
    # SIREN
    "SIRENConfig",
    "SIREN_DEFAULT",
    "SIREN_SMALL",
    "SIREN_LARGE",
    "SIREN_GEO",
    # GLU
    "GLUConfig",
    "GLU_DEFAULT",
    "GLU_SMALL",
    "GLU_LARGE",
]
