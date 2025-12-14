"""
Spherical Random Fourier Features using Gegenbauer polynomials.

Based on the approach from:
- Deep Random Features (DRF) paper, Appendix B.2
- Dutordoir et al. "Sparse Gaussian Processes on Discrete Domains"

For a Matern kernel on the sphere, we use the spherical harmonics expansion:
k(x, x') = sum_l a_l * P_l(x · x')

where P_l are Legendre polynomials (Gegenbauer with alpha=1/2) and a_l are
the spectral coefficients of the Matern kernel on S^2.

The random feature approximation samples:
1. Frequency indices l from the spectral distribution
2. Random reference points z on the sphere
3. Evaluates Gegenbauer polynomials at the geodesic distance

This gives features that approximate the spherical Matern kernel in the RKHS.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import eval_legendre, gamma as gamma_fn
from typing import Optional, Literal, Tuple, Dict, List
from tqdm import tqdm

from .utils import lonlat_to_cartesian3d


def matern_spectral_density_sphere(
    l: np.ndarray,
    nu: float,
    lengthscale: float,
    dim: int = 2
) -> np.ndarray:
    """
    Compute the spectral density (eigenvalues) of Matern kernel on the sphere.

    For a Matern-nu kernel on S^d (d-dimensional sphere), the eigenvalues are:
        a_l ~ (2l + d) * (l + d - 1)! / l! * [lengthscale^2 + l(l+d-1)]^(-nu - d/2)

    For S^2 (d=2):
        a_l ~ (2l + 2) / [lengthscale^2 + l(l+1)]^(nu + 1)

    Args:
        l: Array of frequency indices
        nu: Matern smoothness
        lengthscale: Kernel lengthscale
        dim: Sphere dimension (2 for S^2)

    Returns:
        Array of spectral density values (not normalized)
    """
    l = np.asarray(l, dtype=np.float64)

    if dim == 2:
        # S^2: eigenvalues of Laplacian are l(l+1)
        multiplicity = 2 * l + 1  # Degeneracy of spherical harmonics
        laplacian_eigenval = l * (l + 1)
        # Spectral density from Matern formula
        density = multiplicity / (lengthscale ** 2 + laplacian_eigenval) ** (nu + 1)
    else:
        # General sphere (not implemented here for simplicity)
        raise NotImplementedError("Only S^2 is implemented")

    return density


def sample_spherical_frequencies(
    num_features: int,
    nu: float,
    lengthscale: float,
    lmax: int = 50,
    seed: int = 42
) -> np.ndarray:
    """
    Sample frequency indices from the Matern spectral density on S^2.

    Args:
        num_features: Number of frequencies to sample
        nu: Matern smoothness
        lengthscale: Kernel lengthscale
        lmax: Maximum frequency to consider
        seed: Random seed

    Returns:
        [num_features] array of integer frequency indices
    """
    rng = np.random.default_rng(seed)

    # Compute unnormalized spectral density
    l_values = np.arange(0, lmax + 1)
    density = matern_spectral_density_sphere(l_values, nu, lengthscale)

    # Normalize to probability distribution
    probs = density / density.sum()

    # Sample frequencies
    sampled_l = rng.choice(l_values, size=num_features, p=probs)

    return sampled_l


def sample_uniform_sphere(num_points: int, seed: int = 42) -> np.ndarray:
    """
    Sample points uniformly on the unit 2-sphere.

    Args:
        num_points: Number of points to sample
        seed: Random seed

    Returns:
        [num_points, 3] array of points on unit sphere
    """
    rng = np.random.default_rng(seed)

    # Use Gaussian sampling (standard multivariate normal normalized)
    points = rng.standard_normal(size=(num_points, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    return points


class SphericalRFF(nn.Module):
    """
    Spherical Random Fourier Features using Legendre polynomials.

    Approximates Matern kernel on S^2 using random features based on the
    spherical harmonics expansion.

    For each random feature:
    - Sample a frequency l from the spectral density
    - Sample a random reference point z on the sphere
    - Feature = sqrt(weight) * P_l(x · z)

    where P_l is the Legendre polynomial of degree l.

    Args:
        num_features: Number of random features
        nu: Matern smoothness (0.5, 1.5, 2.5)
        lengthscale: Kernel lengthscale (in radians on sphere)
        lmax: Maximum frequency to consider
        trainable_lengthscale: Make lengthscale learnable
        seed: Random seed
    """

    def __init__(
        self,
        num_features: int = 1000,
        nu: float = 2.5,
        lengthscale: float = 0.5,
        lmax: int = 50,
        trainable_lengthscale: bool = False,
        seed: int = 42
    ):
        super().__init__()

        self.num_features = num_features
        self.nu = nu
        self.lmax = lmax

        # Sample frequencies from spectral density
        rng = np.random.default_rng(seed)
        frequencies = sample_spherical_frequencies(
            num_features, nu, lengthscale, lmax, seed
        )
        self.register_buffer('frequencies', torch.tensor(frequencies, dtype=torch.long))

        # Sample random reference points on sphere
        reference_points = sample_uniform_sphere(num_features, seed + 1)
        self.register_buffer(
            'reference_points',
            torch.tensor(reference_points, dtype=torch.float32)
        )

        # Compute feature weights from spectral density
        density = matern_spectral_density_sphere(
            np.arange(lmax + 1), nu, lengthscale
        )
        # Weight for each feature is proportional to the density at that frequency
        # divided by probability of sampling (to debias)
        probs = density / density.sum()
        weights = np.sqrt(density[frequencies] / (num_features * probs[frequencies]))
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

        # Lengthscale parameter
        if trainable_lengthscale:
            self.log_lengthscale = nn.Parameter(
                torch.tensor(np.log(lengthscale), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'log_lengthscale',
                torch.tensor(np.log(lengthscale), dtype=torch.float32)
            )

        # Precompute Legendre polynomial coefficients for each unique l
        self._precompute_legendre()

    def _precompute_legendre(self):
        """Precompute Legendre polynomial values for efficiency."""
        # We'll evaluate Legendre polynomials on the fly using scipy
        # For large-scale use, could precompute via recurrence
        pass

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    def _eval_legendre(self, l: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate Legendre polynomial P_l at values x.

        Args:
            l: Degree of polynomial
            x: Values in [-1, 1] (cosines of angles)

        Returns:
            P_l(x)
        """
        return eval_legendre(l, x)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute spherical random features.

        Args:
            coords: Either [N, 2] (lon, lat in degrees) or [N, 3] (3D Cartesian)

        Returns:
            [N, num_features] feature vector
        """
        # Convert to 3D if needed
        if coords.shape[1] == 2:
            x = lonlat_to_cartesian3d(coords)
        else:
            x = coords

        # Compute cosine of angle between x and reference points
        # cos(angle) = x · z (since both are on unit sphere)
        cos_angles = x @ self.reference_points.T  # [N, num_features]

        # Evaluate Legendre polynomials
        # This is the slow part - need to use scipy for each unique l
        cos_angles_np = cos_angles.detach().cpu().numpy()
        features_np = np.zeros_like(cos_angles_np)

        unique_l = np.unique(self.frequencies.cpu().numpy())
        for l in unique_l:
            mask = (self.frequencies.cpu().numpy() == l)
            if mask.sum() > 0:
                features_np[:, mask] = self._eval_legendre(l, cos_angles_np[:, mask])

        features = torch.tensor(features_np, dtype=coords.dtype, device=coords.device)

        # Apply weights
        features = features * self.weights.unsqueeze(0)

        # Normalize
        features = features / np.sqrt(self.num_features)

        return features


class SphericalRFFRegression(nn.Module):
    """
    Bayesian Linear Regression on spherical RFF features.

    Similar to planar RFF but uses spherical random features.

    Args:
        rff_module: SphericalRFF feature extractor
        prior_precision: Weight prior precision
        noise_precision: Observation noise precision
    """

    def __init__(
        self,
        rff_module: SphericalRFF,
        prior_precision: float = 1.0,
        noise_precision: float = 1.0
    ):
        super().__init__()
        self.rff = rff_module
        self.prior_precision = prior_precision
        self.noise_precision = noise_precision

        self.register_buffer('posterior_mean', None)
        self.register_buffer('posterior_covar', None)

    def fit(self, train_coords: torch.Tensor, train_y: torch.Tensor) -> None:
        """
        Compute closed-form Bayesian linear regression posterior.

        Args:
            train_coords: [N, 2] (lon, lat) or [N, 3] (3D) coordinates
            train_y: [N] training targets
        """
        # Compute features
        Phi = self.rff(train_coords)  # [N, D]
        D = Phi.shape[1]

        alpha = self.prior_precision
        beta = self.noise_precision

        # Posterior precision
        PhiT_Phi = Phi.T @ Phi
        posterior_precision = alpha * torch.eye(D, device=Phi.device, dtype=Phi.dtype) + beta * PhiT_Phi

        # Add jitter for numerical stability (important for large datasets)
        jitter = 1e-4 * torch.trace(posterior_precision) / D
        posterior_precision = posterior_precision + jitter * torch.eye(D, device=Phi.device, dtype=Phi.dtype)

        # Posterior covariance via Cholesky
        try:
            L = torch.linalg.cholesky(posterior_precision)
            posterior_covar = torch.cholesky_inverse(L)
        except RuntimeError:
            # Fallback: use pseudoinverse if Cholesky fails
            print("  Warning: Cholesky failed, using pseudoinverse (less accurate)")
            posterior_covar = torch.linalg.pinv(posterior_precision)

        # Posterior mean
        posterior_mean = beta * posterior_covar @ (Phi.T @ train_y)

        self.posterior_mean = posterior_mean
        self.posterior_covar = posterior_covar

    def predict(self, test_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty.

        Args:
            test_coords: [N, 2] or [N, 3] test coordinates

        Returns:
            Dictionary with 'mean', 'variance'
        """
        if self.posterior_mean is None:
            raise RuntimeError("Model must be fitted before prediction")

        Phi = self.rff(test_coords)

        mean = Phi @ self.posterior_mean

        covar_proj = self.posterior_covar @ Phi.T
        variance = (Phi * covar_proj.T).sum(dim=1) + 1.0 / self.noise_precision

        return {
            'mean': mean,
            'variance': variance
        }


class SphericalRFFClassification(nn.Module):
    """
    Logistic Regression on spherical RFF features for classification.

    Args:
        rff_module: SphericalRFF feature extractor
        num_classes: Number of output classes
        l2_reg: L2 regularization strength
    """

    def __init__(
        self,
        rff_module: SphericalRFF,
        num_classes: int,
        l2_reg: float = 0.01
    ):
        super().__init__()
        self.rff = rff_module
        self.num_classes = num_classes
        self.l2_reg = l2_reg
        self.classifier = nn.Linear(rff_module.num_features, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.rff(coords)
        return self.classifier(features)

    def fit(
        self,
        train_coords: torch.Tensor,
        train_y: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 1024,
        lr: float = 0.01,
        verbose: bool = True
    ) -> list:
        """Train the classifier."""
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(train_coords, train_y.long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.l2_reg)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        return losses

    @torch.no_grad()
    def predict(self, test_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self(test_coords)
        probs = torch.softmax(logits, dim=1)
        predictions = probs.argmax(dim=1)
        return {'probs': probs, 'predictions': predictions}


def train_spherical_rff_regression(
    train_coords: torch.Tensor,
    train_y: torch.Tensor,
    num_features: int = 1000,
    nu: float = 2.5,
    lengthscale: Optional[float] = None,
    lmax: int = 50,
    prior_precision: float = 1.0,
    noise_precision: float = 10.0,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[SphericalRFFRegression, Dict]:
    """
    Train a spherical RFF regression model.

    Args:
        train_coords: [N, 2] (lon, lat) in degrees
        train_y: [N] training targets
        num_features: Number of random features
        nu: Matern smoothness
        lengthscale: Kernel lengthscale (radians). None = estimate from data
        lmax: Maximum frequency
        prior_precision: Weight prior precision
        noise_precision: Observation noise precision
        device: Device to use
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (model, metadata dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_coords = train_coords.to(device)
    train_y = train_y.to(device)

    # Estimate lengthscale from data if not provided
    if lengthscale is None:
        # Convert to 3D and estimate median geodesic distance
        x_3d = lonlat_to_cartesian3d(train_coords[:1000])
        cos_dist = x_3d @ x_3d.T
        cos_dist = torch.clamp(cos_dist, -1, 1)
        geodesic = torch.arccos(cos_dist)
        triu_mask = torch.triu(torch.ones_like(geodesic), diagonal=1).bool()
        lengthscale = torch.median(geodesic[triu_mask]).item()
        if verbose:
            print(f"Using median heuristic lengthscale: {lengthscale:.4f} radians")

    # Create RFF module
    rff = SphericalRFF(
        num_features=num_features,
        nu=nu,
        lengthscale=lengthscale,
        lmax=lmax,
        seed=seed
    ).to(device)

    # Create model
    model = SphericalRFFRegression(
        rff_module=rff,
        prior_precision=prior_precision,
        noise_precision=noise_precision
    ).to(device)

    if verbose:
        print("Computing Bayesian linear regression posterior on spherical RFF...")
    model.fit(train_coords, train_y)

    metadata = {
        'num_features': num_features,
        'nu': nu,
        'lengthscale': lengthscale,
        'lmax': lmax,
        'prior_precision': prior_precision,
        'noise_precision': noise_precision
    }

    return model, metadata


def train_spherical_rff_classification(
    train_coords: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    num_features: int = 1000,
    nu: float = 2.5,
    lengthscale: Optional[float] = None,
    lmax: int = 50,
    l2_reg: float = 0.01,
    num_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[SphericalRFFClassification, Dict]:
    """
    Train a spherical RFF classification model.

    Args:
        train_coords: [N, 2] (lon, lat) in degrees
        train_y: [N] training class labels
        num_classes: Number of classes
        num_features: Number of random features
        nu: Matern smoothness
        lengthscale: Kernel lengthscale (radians)
        lmax: Maximum frequency
        l2_reg: L2 regularization
        num_epochs: Training epochs
        batch_size: Minibatch size
        lr: Learning rate
        device: Device to use
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (model, metadata dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_coords = train_coords.to(device)
    train_y = train_y.to(device)

    # Estimate lengthscale
    if lengthscale is None:
        x_3d = lonlat_to_cartesian3d(train_coords[:1000])
        cos_dist = x_3d @ x_3d.T
        cos_dist = torch.clamp(cos_dist, -1, 1)
        geodesic = torch.arccos(cos_dist)
        triu_mask = torch.triu(torch.ones_like(geodesic), diagonal=1).bool()
        lengthscale = torch.median(geodesic[triu_mask]).item()
        if verbose:
            print(f"Using median heuristic lengthscale: {lengthscale:.4f} radians")

    rff = SphericalRFF(
        num_features=num_features,
        nu=nu,
        lengthscale=lengthscale,
        lmax=lmax,
        seed=seed
    ).to(device)

    model = SphericalRFFClassification(
        rff_module=rff,
        num_classes=num_classes,
        l2_reg=l2_reg
    ).to(device)

    if verbose:
        print("Training logistic regression on spherical RFF features...")
    model.fit(train_coords, train_y, num_epochs=num_epochs, batch_size=batch_size, lr=lr, verbose=verbose)

    metadata = {
        'num_features': num_features,
        'nu': nu,
        'lengthscale': lengthscale,
        'lmax': lmax,
        'l2_reg': l2_reg
    }

    return model, metadata


def evaluate_spherical_rff_regression(
    model: SphericalRFFRegression,
    test_coords: torch.Tensor,
    test_y: torch.Tensor,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None
) -> Dict[str, float]:
    """Evaluate spherical RFF regression."""
    predictions = model.predict(test_coords)
    pred_mean = predictions['mean']

    mse = ((pred_mean - test_y) ** 2).mean().item()
    mae = (pred_mean - test_y).abs().mean().item()
    ss_res = ((test_y - pred_mean) ** 2).sum()
    ss_tot = ((test_y - test_y.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot).item()

    result = {'mse': mse, 'mae': mae, 'r2': r2}

    if y_min is not None and y_max is not None:
        pred_orig = pred_mean * (y_max - y_min) + y_min
        test_orig = test_y * (y_max - y_min) + y_min
        result['mae_original'] = (pred_orig - test_orig).abs().mean().item()

    return result


def evaluate_spherical_rff_classification(
    model: SphericalRFFClassification,
    test_coords: torch.Tensor,
    test_y: torch.Tensor
) -> Dict[str, float]:
    """Evaluate spherical RFF classification."""
    predictions = model.predict(test_coords)
    pred_classes = predictions['predictions']
    accuracy = (pred_classes == test_y).float().mean().item()
    return {'accuracy': accuracy}
