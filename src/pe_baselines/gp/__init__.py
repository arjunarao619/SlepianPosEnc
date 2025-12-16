"""
GP baselines: Exact GP, SVGP, Planar RFF, Spherical RFF/SVGP.
Planar methods use normalized [0,1] coordinates; spherical use geodesic distance.
"""

# Utilities
from .utils import (
    lonlat_to_cartesian3d,
    normalize_lonlat_planar,
    geodesic_distance,
    init_inducing_points_kmeans,
    init_inducing_points_random,
    median_heuristic_lengthscale,
    CoordinateTransformer,
    prepare_gp_data
)

# Exact GP
from .exact_gp import (
    ExactGPRegression,
    ExactGPClassification,
    train_exact_gp_regression,
    train_exact_gp_classification,
    predict_regression,
    predict_classification,
    evaluate_regression,
    evaluate_classification
)

# SVGP
from .svgp import (
    SVGPRegression,
    SVGPClassification,
    train_svgp_regression,
    train_svgp_classification,
    predict_svgp_regression,
    predict_svgp_classification,
    evaluate_svgp_regression,
    evaluate_svgp_classification
)

# Planar RFF
from .planar_rff import (
    PlanarRFF,
    RFFRegression,
    RFFClassification,
    train_rff_regression,
    train_rff_classification,
    evaluate_rff_regression,
    evaluate_rff_classification
)

# Spherical SVGP
from .spherical_svgp import (
    ApproximateSphericalSVGP,
    train_spherical_svgp_regression,
    predict_spherical_svgp,
    evaluate_spherical_svgp
)

# Spherical RFF
from .spherical_rff import (
    SphericalRFF,
    SphericalRFFRegression,
    SphericalRFFClassification,
    train_spherical_rff_regression,
    train_spherical_rff_classification,
    evaluate_spherical_rff_regression,
    evaluate_spherical_rff_classification
)

__all__ = [
    # Utilities
    'lonlat_to_cartesian3d',
    'normalize_lonlat_planar',
    'geodesic_distance',
    'init_inducing_points_kmeans',
    'init_inducing_points_random',
    'median_heuristic_lengthscale',
    'CoordinateTransformer',
    'prepare_gp_data',
    # Exact GP
    'ExactGPRegression',
    'ExactGPClassification',
    'train_exact_gp_regression',
    'train_exact_gp_classification',
    'predict_regression',
    'predict_classification',
    'evaluate_regression',
    'evaluate_classification',
    # SVGP
    'SVGPRegression',
    'SVGPClassification',
    'train_svgp_regression',
    'train_svgp_classification',
    'predict_svgp_regression',
    'predict_svgp_classification',
    'evaluate_svgp_regression',
    'evaluate_svgp_classification',
    # Planar RFF
    'PlanarRFF',
    'RFFRegression',
    'RFFClassification',
    'train_rff_regression',
    'train_rff_classification',
    'evaluate_rff_regression',
    'evaluate_rff_classification',
    # Spherical SVGP
    'ApproximateSphericalSVGP',
    'train_spherical_svgp_regression',
    'predict_spherical_svgp',
    'evaluate_spherical_svgp',
    # Spherical RFF
    'SphericalRFF',
    'SphericalRFFRegression',
    'SphericalRFFClassification',
    'train_spherical_rff_regression',
    'train_spherical_rff_classification',
    'evaluate_spherical_rff_regression',
    'evaluate_spherical_rff_classification',
]
