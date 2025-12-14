"""
Gaussian Process baselines for geographic location encoding.

This module provides GP-based alternatives to learned location encodings,
to answer: "Can a well-tuned GP with appropriate smoothness priors match
or beat learned location encodings?"

Implemented Methods:
--------------------

1. Exact GP (exact_gp.py)
   - ExactGPRegression: Standard GP for regression (up to ~30K points)
   - ExactGPClassification: GP classification via Dirichlet transformation

2. SVGP (svgp.py)
   - SVGPRegression: Sparse Variational GP for large-scale regression
   - SVGPClassification: Sparse Variational GP for multi-class classification

3. Planar RFF (planar_rff.py)
   - PlanarRFF: Random Fourier Features for shift-invariant kernels
   - RFFRegression: Bayesian linear regression on RFF features (closed-form)
   - RFFClassification: Logistic regression on RFF features

4. Spherical SVGP (spherical_svgp.py)
   - SphericalSVGPRegression: SVGP with proper spherical Matern kernel
   - Uses geometric-kernels or geodesic approximation

5. Spherical RFF (spherical_rff.py)
   - SphericalRFF: Random features using Legendre polynomials
   - Based on DRF paper's approach for spherical kernel approximation

Utilities (utils.py):
--------------------
- CoordinateTransformer: Normalize coordinates or convert to 3D
- init_inducing_points_kmeans: K-means based inducing point initialization
- median_heuristic_lengthscale: Data-driven lengthscale estimation
- geodesic_distance: Great-circle distance computation

Usage Example:
-------------
```python
from pe_baselines.gp import (
    train_svgp_regression,
    predict_svgp_regression,
    evaluate_svgp_regression,
    normalize_lonlat_planar
)

# Prepare data
train_x = normalize_lonlat_planar(train_coords)
test_x = normalize_lonlat_planar(test_coords)

# Train SVGP
model, likelihood, _ = train_svgp_regression(
    train_x, train_y,
    num_inducing=500,
    kernel_type='matern52',
    num_epochs=50
)

# Predict
predictions = predict_svgp_regression(model, likelihood, test_x)
metrics = evaluate_svgp_regression(predictions, test_y)
print(f"R2: {metrics['r2']:.4f}")
```
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
