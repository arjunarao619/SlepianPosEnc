"""Spherical SVGP using geometric-kernels for geodesic Matern kernels on S²."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal, Tuple, Dict, List
from tqdm import tqdm

# Check for geometric-kernels
try:
    import geometric_kernels
    from geometric_kernels.spaces import Hypersphere
    from geometric_kernels.kernels import MaternGeometricKernel
    from geometric_kernels.feature_maps import RandomPhaseFeatureMapCompact
    HAVE_GK = True
except ImportError:
    HAVE_GK = False
    print("WARNING: geometric-kernels not found. Install with: pip install geometric-kernels")

try:
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        VariationalStrategy
    )
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import Kernel
    from gpytorch.distributions import MultivariateNormal
    HAVE_GPYTORCH = True
except ImportError:
    HAVE_GPYTORCH = False
    print("WARNING: gpytorch not found. Install with: pip install gpytorch")

from .utils import lonlat_to_cartesian3d


class GeometricSphericalMaternKernel(Kernel if HAVE_GPYTORCH else object):
    """GPyTorch wrapper for geometric-kernels spherical Matern. Inputs: 3D Cartesian on S²."""

    is_stationary = True

    def __init__(
        self,
        nu: float = 2.5,
        num_levels: int = 20,
        **kwargs
    ):
        if not HAVE_GK:
            raise ImportError("geometric-kernels required for spherical kernel")

        super().__init__(**kwargs)

        self.nu = nu
        self.num_levels = num_levels

        # Create the spherical space (2-sphere embedded in R^3)
        self.space = Hypersphere(dim=2)

        # Create the Matern kernel on the sphere
        self.gk_kernel = MaternGeometricKernel(self.space)

        # Register lengthscale as parameter
        self.register_parameter(
            name='raw_lengthscale',
            parameter=torch.nn.Parameter(torch.tensor(0.0))
        )

        # Register outputscale (signal variance)
        self.register_parameter(
            name='raw_outputscale',
            parameter=torch.nn.Parameter(torch.tensor(0.0))
        )

    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.raw_lengthscale)

    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.raw_outputscale)

    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the kernel matrix.

        Args:
            x1: [N, 3] points on unit sphere
            x2: [M, 3] points on unit sphere
            diag: If True, return only diagonal

        Returns:
            [N, M] or [N] kernel matrix
        """
        # geometric-kernels expects params dict with lengthscale and nu
        gk_params = {
            'lengthscale': self.lengthscale.detach().cpu().numpy(),
            'nu': self.nu
        }

        # Compute kernel using geometric-kernels
        # Note: geometric-kernels works with numpy, so we need to convert
        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()

        K = self.gk_kernel.K(gk_params, x1_np, x2_np)
        K = torch.tensor(K, dtype=x1.dtype, device=x1.device)

        # Scale by outputscale
        K = self.outputscale * K

        if diag:
            return K.diag()
        return K


class SphericalSVGPRegression(ApproximateGP if HAVE_GPYTORCH else object):
    """SVGP with spherical Matern kernel using geodesic distance."""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        nu: float = 2.5,
        num_levels: int = 20,
        learn_inducing_locations: bool = True
    ):
        if not HAVE_GPYTORCH or not HAVE_GK:
            raise ImportError("gpytorch and geometric-kernels required")

        num_inducing = inducing_points.shape[0]

        # Variational distribution
        variational_distribution = CholeskyVariationalDistribution(num_inducing)

        # Variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )

        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        self.covar_module = GeometricSphericalMaternKernel(nu=nu, num_levels=num_levels)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class ApproximateSphericalMaternKernel(Kernel if HAVE_GPYTORCH else object):
    """Approximate spherical Matern: standard Matern applied to geodesic distances."""

    is_stationary = True

    def __init__(self, nu: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu

        # Lengthscale parameter (geodesic distance scale)
        self.register_parameter(
            name='raw_lengthscale',
            parameter=torch.nn.Parameter(torch.tensor(0.0))
        )

        # Output scale
        self.register_parameter(
            name='raw_outputscale',
            parameter=torch.nn.Parameter(torch.tensor(0.0))
        )

    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.raw_lengthscale) + 0.01

    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.raw_outputscale)

    def _geodesic_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distances between points on unit sphere."""
        # Cosine of angle = dot product (since points are on unit sphere)
        cos_angle = x1 @ x2.T  # [N, M]

        # Clamp for numerical stability
        cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)

        # Geodesic distance = arccos(cos_angle)
        return torch.arccos(cos_angle)

    def _matern_from_distance(self, dist: torch.Tensor) -> torch.Tensor:
        """Apply Matern function to distances."""
        # Scale by lengthscale
        scaled_dist = dist / self.lengthscale

        if self.nu == 0.5:
            # Exponential kernel
            K = torch.exp(-scaled_dist)
        elif self.nu == 1.5:
            # Matern 3/2
            sqrt3_d = np.sqrt(3) * scaled_dist
            K = (1 + sqrt3_d) * torch.exp(-sqrt3_d)
        elif self.nu == 2.5:
            # Matern 5/2
            sqrt5_d = np.sqrt(5) * scaled_dist
            K = (1 + sqrt5_d + (5/3) * scaled_dist ** 2) * torch.exp(-sqrt5_d)
        else:
            raise ValueError(f"nu must be 0.5, 1.5, or 2.5, got {self.nu}")

        return self.outputscale * K

    def forward(self, x1, x2, diag=False, **params):
        dist = self._geodesic_distance(x1, x2)
        K = self._matern_from_distance(dist)

        if diag:
            return K.diag()

        # Add jitter to diagonal for numerical stability
        if x1.shape[0] == x2.shape[0] and torch.allclose(x1, x2):
            jitter = 1e-4
            K = K + jitter * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)

        return K


class ApproximateSphericalSVGP(ApproximateGP if HAVE_GPYTORCH else object):
    """SVGP with approximate spherical Matern (geodesic distance + standard Matern)."""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        nu: float = 2.5,
        learn_inducing_locations: bool = True
    ):
        if not HAVE_GPYTORCH:
            raise ImportError("gpytorch required")

        num_inducing = inducing_points.shape[0]

        variational_distribution = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )

        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        self.covar_module = ApproximateSphericalMaternKernel(nu=nu)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


def train_spherical_svgp_regression(
    train_coords: torch.Tensor,
    train_y: torch.Tensor,
    num_inducing: int = 500,
    nu: float = 2.5,
    use_geometric_kernels: bool = True,
    learn_inducing_locations: bool = True,
    num_epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[nn.Module, GaussianLikelihood, List[float]]:
    """Train spherical SVGP regression. Returns (model, likelihood, loss_history)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert coordinates to 3D Cartesian on unit sphere
    train_x = lonlat_to_cartesian3d(train_coords).to(device)
    train_y = train_y.to(device)

    # Cap inducing points to training set size
    actual_num_inducing = min(num_inducing, len(train_x))
    if actual_num_inducing < num_inducing:
        print(f"Note: Reduced inducing points from {num_inducing} to {actual_num_inducing} (train size)")

    # Initialize inducing points via k-means on 3D coordinates
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=actual_num_inducing, random_state=seed, n_init=10)
    kmeans.fit(train_x.cpu().numpy())
    inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    # Normalize inducing points to unit sphere
    inducing_points = inducing_points / inducing_points.norm(dim=1, keepdim=True)
    inducing_points = inducing_points.to(device)

    # Create model
    if use_geometric_kernels and HAVE_GK:
        if verbose:
            print("Using geometric-kernels spherical Matern")
        model = SphericalSVGPRegression(
            inducing_points,
            nu=nu,
            learn_inducing_locations=learn_inducing_locations
        ).to(device)
    else:
        if verbose:
            print("Using approximate geodesic Matern")
        model = ApproximateSphericalSVGP(
            inducing_points,
            nu=nu,
            learn_inducing_locations=learn_inducing_locations
        ).to(device)

    likelihood = GaussianLikelihood().to(device)

    # Data loader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=lr)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_x))

    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        iterator = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}") if verbose else loader

        for batch_x, batch_y in iterator:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = -mll(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    return model, likelihood, loss_history


@torch.no_grad()
def predict_spherical_svgp(
    model: nn.Module,
    likelihood: GaussianLikelihood,
    test_coords: torch.Tensor,
    batch_size: int = 2048
) -> Dict[str, torch.Tensor]:
    """Predict with spherical SVGP. Returns dict with 'mean', 'variance'."""
    model.eval()
    likelihood.eval()

    device = next(model.parameters()).device

    # Convert to 3D
    test_x = lonlat_to_cartesian3d(test_coords).to(device)

    means = []
    variances = []

    n = len(test_x)
    for i in range(0, n, batch_size):
        batch_x = test_x[i:i + batch_size]

        with gpytorch.settings.fast_pred_var():
            pred = likelihood(model(batch_x))

        means.append(pred.mean.cpu())
        variances.append(pred.variance.cpu())

    return {
        'mean': torch.cat(means),
        'variance': torch.cat(variances)
    }


def evaluate_spherical_svgp(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None
) -> Dict[str, float]:
    """Evaluate spherical SVGP. Returns dict with MSE, MAE, R2."""
    pred_mean = predictions['mean']

    mse = ((pred_mean - targets) ** 2).mean().item()
    mae = (pred_mean - targets).abs().mean().item()
    ss_res = ((targets - pred_mean) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot).item()

    result = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    if y_min is not None and y_max is not None:
        pred_orig = pred_mean * (y_max - y_min) + y_min
        targets_orig = targets * (y_max - y_min) + y_min
        result['mae_original'] = (pred_orig - targets_orig).abs().mean().item()

    return result
