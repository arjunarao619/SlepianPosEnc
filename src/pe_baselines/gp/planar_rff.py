"""
Random Fourier Features (RFF) for fast GP approximation.

Implements the approach from:
- Rahimi & Recht (2007) "Random Features for Large-Scale Kernel Machines"
- Extended to Matern kernels via spectral density sampling

For Matern kernels, frequencies are sampled from the spectral density
which is a scaled Student-t distribution.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import gamma as gamma_fn
from typing import Optional, Literal, Tuple, Dict
from tqdm import tqdm


class PlanarRFF(nn.Module):
    """
    Random Fourier Features for planar coordinates.

    Approximates shift-invariant kernels (RBF, Matern) using random features.

    Args:
        input_dim: Input dimensionality (typically 2 for lon/lat)
        num_features: Number of random features
        lengthscale: Kernel lengthscale (scalar or per-dimension)
        kernel_type: "rbf", "matern32", or "matern52"
        trainable_lengthscale: Whether to make lengthscale a learnable parameter
        seed: Random seed for frequency sampling
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_features: int = 1000,
        lengthscale: float = 0.1,
        kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
        trainable_lengthscale: bool = False,
        seed: int = 42
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_features = num_features
        self.kernel_type = kernel_type

        # Sample frequencies from spectral density
        rng = np.random.default_rng(seed)
        omega = self._sample_frequencies(rng, kernel_type, num_features, input_dim)

        # Store as buffers (not parameters by default)
        self.register_buffer('omega', torch.tensor(omega, dtype=torch.float32))

        # Random phase shifts
        bias = rng.uniform(0, 2 * np.pi, size=num_features)
        self.register_buffer('bias', torch.tensor(bias, dtype=torch.float32))

        # Lengthscale as parameter or buffer
        if trainable_lengthscale:
            self.log_lengthscale = nn.Parameter(
                torch.tensor(np.log(lengthscale), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'log_lengthscale',
                torch.tensor(np.log(lengthscale), dtype=torch.float32)
            )

    def _sample_frequencies(
        self,
        rng: np.random.Generator,
        kernel_type: str,
        num_features: int,
        input_dim: int
    ) -> np.ndarray:
        """
        Sample frequencies from kernel spectral density.

        For RBF: omega ~ N(0, 1)
        For Matern: omega ~ scaled StudentT(2*nu)

        Returns:
            [num_features, input_dim] array of frequencies
        """
        if kernel_type == "rbf":
            # RBF spectral density is Gaussian
            omega = rng.standard_normal(size=(num_features, input_dim))

        elif kernel_type == "matern32":
            # Matern-3/2: nu=1.5, df=3
            # Sample from StudentT(3)
            df = 3
            omega = rng.standard_t(df, size=(num_features, input_dim))
            # Scale: for Matern, spectral density is proportional to (1 + s^2)^(-nu - d/2)
            # The scaling factor sqrt(2*nu) transforms to unit lengthscale
            omega *= np.sqrt(3)

        elif kernel_type == "matern52":
            # Matern-5/2: nu=2.5, df=5
            # Sample from StudentT(5)
            df = 5
            omega = rng.standard_t(df, size=(num_features, input_dim))
            omega *= np.sqrt(5)

        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        return omega

    @property
    def lengthscale(self) -> torch.Tensor:
        """Current lengthscale value."""
        return torch.exp(self.log_lengthscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute random Fourier features.

        Args:
            x: [N, input_dim] input coordinates (should be normalized to [0, 1])

        Returns:
            [N, num_features] feature vector
        """
        # Scale inputs by lengthscale
        # omega is already sampled for unit lengthscale, so divide by lengthscale
        scaled_x = x / self.lengthscale

        # Project: z = x @ omega^T + bias
        projection = torch.matmul(scaled_x, self.omega.T) + self.bias

        # Random features: sqrt(2/D) * cos(projection)
        features = torch.sqrt(torch.tensor(2.0 / self.num_features)) * torch.cos(projection)

        return features


class RFFRegression(nn.Module):
    """
    Bayesian Linear Regression on RFF features.

    For closed-form posterior inference with Gaussian likelihood.

    Args:
        rff_module: PlanarRFF feature extractor
        prior_precision: Precision (inverse variance) of the prior on weights
        noise_precision: Precision of the observation noise
    """

    def __init__(
        self,
        rff_module: PlanarRFF,
        prior_precision: float = 1.0,
        noise_precision: float = 1.0
    ):
        super().__init__()
        self.rff = rff_module
        self.prior_precision = prior_precision
        self.noise_precision = noise_precision

        # Posterior parameters (set during training)
        self.register_buffer('posterior_mean', None)
        self.register_buffer('posterior_covar', None)

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """
        Compute closed-form Bayesian linear regression posterior.

        Args:
            train_x: [N, input_dim] training inputs (normalized)
            train_y: [N] training targets
        """
        # Compute features
        Phi = self.rff(train_x)  # [N, D]
        D = Phi.shape[1]

        # Prior precision matrix: alpha * I
        alpha = self.prior_precision
        beta = self.noise_precision

        # Posterior precision: S_N^{-1} = alpha * I + beta * Phi^T @ Phi
        PhiT_Phi = Phi.T @ Phi
        posterior_precision = alpha * torch.eye(D, device=Phi.device, dtype=Phi.dtype) + beta * PhiT_Phi

        # Add jitter for numerical stability (important for large datasets)
        jitter = 1e-4 * torch.trace(posterior_precision) / D
        posterior_precision = posterior_precision + jitter * torch.eye(D, device=Phi.device, dtype=Phi.dtype)

        # Posterior covariance: S_N = (S_N^{-1})^{-1}
        # Use Cholesky for numerical stability
        try:
            L = torch.linalg.cholesky(posterior_precision)
            posterior_covar = torch.cholesky_inverse(L)
        except RuntimeError:
            # Fallback: use pseudoinverse if Cholesky fails
            print("  Warning: Cholesky failed, using pseudoinverse (less accurate)")
            posterior_covar = torch.linalg.pinv(posterior_precision)

        # Posterior mean: m_N = beta * S_N @ Phi^T @ y
        posterior_mean = beta * posterior_covar @ (Phi.T @ train_y)

        self.posterior_mean = posterior_mean
        self.posterior_covar = posterior_covar

    def predict(self, test_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty.

        Args:
            test_x: [N, input_dim] test inputs (normalized)

        Returns:
            Dictionary with 'mean', 'variance'
        """
        if self.posterior_mean is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Compute features
        Phi = self.rff(test_x)  # [N, D]

        # Predictive mean: y = Phi @ m_N
        mean = Phi @ self.posterior_mean

        # Predictive variance: sigma^2 + Phi @ S_N @ Phi^T (diagonal only)
        # Var = 1/beta + diag(Phi @ S_N @ Phi^T)
        # Compute efficiently: (Phi @ S_N @ Phi^T)_ii = sum_j (Phi_ij * (S_N @ Phi^T)_ji)
        covar_proj = self.posterior_covar @ Phi.T  # [D, N]
        variance = (Phi * covar_proj.T).sum(dim=1) + 1.0 / self.noise_precision

        return {
            'mean': mean,
            'variance': variance
        }


class RFFClassification(nn.Module):
    """
    Logistic Regression on RFF features for classification.

    Uses gradient descent for training (no closed-form solution for logistic).

    Args:
        rff_module: PlanarRFF feature extractor
        num_classes: Number of output classes
        l2_reg: L2 regularization strength
    """

    def __init__(
        self,
        rff_module: PlanarRFF,
        num_classes: int,
        l2_reg: float = 0.01
    ):
        super().__init__()
        self.rff = rff_module
        self.num_classes = num_classes
        self.l2_reg = l2_reg

        # Linear classifier on top of RFF features
        self.classifier = nn.Linear(rff_module.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [N, input_dim] input coordinates

        Returns:
            [N, num_classes] logits
        """
        features = self.rff(x)
        return self.classifier(features)

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 1024,
        lr: float = 0.01,
        verbose: bool = True
    ) -> list:
        """
        Train the classifier using cross-entropy loss.

        Args:
            train_x: [N, input_dim] training inputs (normalized)
            train_y: [N] training class labels
            num_epochs: Number of training epochs
            batch_size: Minibatch size
            lr: Learning rate
            verbose: Print progress

        Returns:
            List of training losses
        """
        device = train_x.device

        # Create data loader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(train_x, train_y.long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
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
    def predict(self, test_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions.

        Args:
            test_x: [N, input_dim] test inputs

        Returns:
            Dictionary with 'probs' and 'predictions'
        """
        logits = self(test_x)
        probs = torch.softmax(logits, dim=1)
        predictions = probs.argmax(dim=1)

        return {
            'probs': probs,
            'predictions': predictions
        }


def train_rff_regression(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_features: int = 1000,
    kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
    lengthscale: Optional[float] = None,
    prior_precision: float = 1.0,
    noise_precision: float = 10.0,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[RFFRegression, Dict]:
    """
    Train an RFF regression model with Bayesian linear regression.

    Args:
        train_x: [N, input_dim] training inputs (should be normalized)
        train_y: [N] training targets
        num_features: Number of random features
        kernel_type: Kernel to approximate
        lengthscale: Kernel lengthscale (None = use median heuristic)
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

    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Estimate lengthscale if not provided
    if lengthscale is None:
        # Median heuristic
        dists = torch.cdist(train_x[:1000], train_x[:1000])
        triu_mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
        lengthscale = torch.median(dists[triu_mask]).item()
        if verbose:
            print(f"Using median heuristic lengthscale: {lengthscale:.4f}")

    # Create RFF module
    rff = PlanarRFF(
        input_dim=train_x.shape[1],
        num_features=num_features,
        lengthscale=lengthscale,
        kernel_type=kernel_type,
        seed=seed
    ).to(device)

    # Create model
    model = RFFRegression(
        rff_module=rff,
        prior_precision=prior_precision,
        noise_precision=noise_precision
    ).to(device)

    # Fit (closed-form)
    if verbose:
        print("Computing Bayesian linear regression posterior...")
    model.fit(train_x, train_y)

    metadata = {
        'num_features': num_features,
        'kernel_type': kernel_type,
        'lengthscale': lengthscale,
        'prior_precision': prior_precision,
        'noise_precision': noise_precision
    }

    return model, metadata


def train_rff_classification(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    num_features: int = 1000,
    kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
    lengthscale: Optional[float] = None,
    l2_reg: float = 0.01,
    num_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[RFFClassification, Dict]:
    """
    Train an RFF classification model with logistic regression.

    Args:
        train_x: [N, input_dim] training inputs (should be normalized)
        train_y: [N] training class labels
        num_classes: Number of classes
        num_features: Number of random features
        kernel_type: Kernel to approximate
        lengthscale: Kernel lengthscale (None = use median heuristic)
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

    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Estimate lengthscale if not provided
    if lengthscale is None:
        dists = torch.cdist(train_x[:1000], train_x[:1000])
        triu_mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
        lengthscale = torch.median(dists[triu_mask]).item()
        if verbose:
            print(f"Using median heuristic lengthscale: {lengthscale:.4f}")

    # Create RFF module
    rff = PlanarRFF(
        input_dim=train_x.shape[1],
        num_features=num_features,
        lengthscale=lengthscale,
        kernel_type=kernel_type,
        seed=seed
    ).to(device)

    # Create model
    model = RFFClassification(
        rff_module=rff,
        num_classes=num_classes,
        l2_reg=l2_reg
    ).to(device)

    # Fit
    if verbose:
        print("Training logistic regression on RFF features...")
    model.fit(train_x, train_y, num_epochs=num_epochs, batch_size=batch_size, lr=lr, verbose=verbose)

    metadata = {
        'num_features': num_features,
        'kernel_type': kernel_type,
        'lengthscale': lengthscale,
        'l2_reg': l2_reg
    }

    return model, metadata


def evaluate_rff_regression(
    model: RFFRegression,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluate RFF regression model.

    Args:
        model: Trained RFFRegression
        test_x: Test inputs (normalized)
        test_y: True targets
        y_min, y_max: For denormalization

    Returns:
        Dictionary with MSE, MAE, R2
    """
    # Ensure test data is on same device as model
    device = model.rff.omega.device
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    predictions = model.predict(test_x)
    pred_mean = predictions['mean']

    mse = ((pred_mean - test_y) ** 2).mean().item()
    mae = (pred_mean - test_y).abs().mean().item()
    ss_res = ((test_y - pred_mean) ** 2).sum()
    ss_tot = ((test_y - test_y.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot).item()

    result = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    if y_min is not None and y_max is not None:
        pred_orig = pred_mean * (y_max - y_min) + y_min
        test_orig = test_y * (y_max - y_min) + y_min
        result['mae_original'] = (pred_orig - test_orig).abs().mean().item()

    return result


def evaluate_rff_classification(
    model: RFFClassification,
    test_x: torch.Tensor,
    test_y: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate RFF classification model.

    Args:
        model: Trained RFFClassification
        test_x: Test inputs
        test_y: True labels

    Returns:
        Dictionary with accuracy
    """
    # Ensure test data is on same device as model
    device = model.rff.omega.device
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    predictions = model.predict(test_x)
    pred_classes = predictions['predictions']
    accuracy = (pred_classes == test_y).float().mean().item()

    return {'accuracy': accuracy}
