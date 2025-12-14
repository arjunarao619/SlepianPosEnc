"""
Sparse Variational Gaussian Process (SVGP) models.

Scales to large datasets (>100K points) via inducing points.
Uses GPyTorch ApproximateGP with VariationalStrategy.
"""

import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
    IndependentMultitaskVariationalStrategy
)
from gpytorch.likelihoods import GaussianLikelihood, SoftmaxLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Literal, Tuple, Dict, List
import numpy as np
from tqdm import tqdm

from .utils import init_inducing_points_kmeans


class SVGPRegression(ApproximateGP):
    """
    Sparse Variational GP for regression.

    Args:
        inducing_points: [M, D] initial inducing point locations
        kernel_type: "rbf", "matern32", or "matern52"
        ard: Use Automatic Relevance Determination
        learn_inducing_locations: Whether to optimize inducing point locations
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
        ard: bool = True,
        learn_inducing_locations: bool = True
    ):
        input_dim = inducing_points.shape[1]
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

        ard_num_dims = input_dim if ard else None

        self.mean_module = ConstantMean()

        if kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == "matern32":
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        elif kernel_type == "matern52":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class SVGPClassification(ApproximateGP):
    """
    Sparse Variational GP for multi-class classification.

    Uses independent GPs for each class with softmax likelihood.

    Args:
        inducing_points: [M, D] initial inducing point locations
        num_classes: Number of output classes
        kernel_type: "rbf", "matern32", or "matern52"
        ard: Use Automatic Relevance Determination
        learn_inducing_locations: Whether to optimize inducing point locations
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        num_classes: int,
        kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
        ard: bool = True,
        learn_inducing_locations: bool = True
    ):
        input_dim = inducing_points.shape[1]
        num_inducing = inducing_points.shape[0]

        self.num_classes = num_classes

        # Variational distribution for each class
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing,
            batch_shape=torch.Size([num_classes])
        )

        # Base variational strategy
        base_variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )

        # Multi-task wrapper
        variational_strategy = IndependentMultitaskVariationalStrategy(
            base_variational_strategy,
            num_tasks=num_classes
        )

        super().__init__(variational_strategy)

        ard_num_dims = input_dim if ard else None

        self.mean_module = ConstantMean(batch_shape=torch.Size([num_classes]))

        if kernel_type == "rbf":
            base_kernel = RBFKernel(
                ard_num_dims=ard_num_dims,
                batch_shape=torch.Size([num_classes])
            )
        elif kernel_type == "matern32":
            base_kernel = MaternKernel(
                nu=1.5,
                ard_num_dims=ard_num_dims,
                batch_shape=torch.Size([num_classes])
            )
        elif kernel_type == "matern52":
            base_kernel = MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                batch_shape=torch.Size([num_classes])
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        self.covar_module = ScaleKernel(
            base_kernel,
            batch_shape=torch.Size([num_classes])
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


def train_svgp_regression(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_inducing: int = 500,
    kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
    ard: bool = True,
    learn_inducing_locations: bool = True,
    num_epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
    inducing_init: Literal["kmeans", "random"] = "kmeans",
    seed: int = 42,
    verbose: bool = True
) -> Tuple[SVGPRegression, GaussianLikelihood, List[float]]:
    """
    Train a Sparse Variational GP regression model.

    Args:
        train_x: [N, D] training inputs (normalized coordinates)
        train_y: [N] training targets
        num_inducing: Number of inducing points
        kernel_type: Kernel to use
        ard: Use ARD lengthscales
        learn_inducing_locations: Optimize inducing point locations
        num_epochs: Number of training epochs
        batch_size: Minibatch size
        lr: Learning rate
        device: Device to train on
        inducing_init: "kmeans" or "random" initialization
        seed: Random seed for initialization
        verbose: Print progress

    Returns:
        Tuple of (model, likelihood, loss_history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cap inducing points to training set size
    actual_num_inducing = min(num_inducing, len(train_x))
    if actual_num_inducing < num_inducing:
        print(f"Note: Reduced inducing points from {num_inducing} to {actual_num_inducing} (train size)")

    # Initialize inducing points
    if inducing_init == "kmeans":
        # Note: train_x should already be normalized
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=actual_num_inducing, random_state=seed, n_init=10)
        kmeans.fit(train_x.cpu().numpy())
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(train_x), size=actual_num_inducing, replace=False)
        inducing_points = train_x[indices].clone()

    inducing_points = inducing_points.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Initialize model and likelihood
    likelihood = GaussianLikelihood().to(device)
    model = SVGPRegression(
        inducing_points,
        kernel_type=kernel_type,
        ard=ard,
        learn_inducing_locations=learn_inducing_locations
    ).to(device)

    # Create data loader
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=lr)

    # Variational ELBO
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


def train_svgp_classification(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    num_inducing: int = 500,
    kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
    ard: bool = True,
    learn_inducing_locations: bool = True,
    num_epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
    inducing_init: Literal["kmeans", "random"] = "kmeans",
    seed: int = 42,
    verbose: bool = True
) -> Tuple[SVGPClassification, SoftmaxLikelihood, List[float]]:
    """
    Train a Sparse Variational GP classification model.

    Args:
        train_x: [N, D] training inputs (normalized coordinates)
        train_y: [N] training class labels (integers)
        num_classes: Number of classes
        num_inducing: Number of inducing points
        kernel_type: Kernel to use
        ard: Use ARD lengthscales
        learn_inducing_locations: Optimize inducing point locations
        num_epochs: Number of training epochs
        batch_size: Minibatch size
        lr: Learning rate
        device: Device to train on
        inducing_init: "kmeans" or "random" initialization
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (model, likelihood, loss_history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cap inducing points to training set size
    actual_num_inducing = min(num_inducing, len(train_x))
    if actual_num_inducing < num_inducing:
        print(f"Note: Reduced inducing points from {num_inducing} to {actual_num_inducing} (train size)")

    # Initialize inducing points
    if inducing_init == "kmeans":
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=actual_num_inducing, random_state=seed, n_init=10)
        kmeans.fit(train_x.cpu().numpy())
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(train_x), size=actual_num_inducing, replace=False)
        inducing_points = train_x[indices].clone()

    inducing_points = inducing_points.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device).long()

    # Initialize model and likelihood
    likelihood = SoftmaxLikelihood(
        num_classes=num_classes,
        num_features=num_classes,  # One GP per class
        mixing_weights=False
    ).to(device)

    model = SVGPClassification(
        inducing_points,
        num_classes=num_classes,
        kernel_type=kernel_type,
        ard=ard,
        learn_inducing_locations=learn_inducing_locations
    ).to(device)

    # Create data loader
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=lr)

    # Variational ELBO
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
def predict_svgp_regression(
    model: SVGPRegression,
    likelihood: GaussianLikelihood,
    test_x: torch.Tensor,
    batch_size: int = 2048
) -> Dict[str, torch.Tensor]:
    """
    Make predictions with a trained SVGP regression model.

    Args:
        model: Trained SVGPRegression
        likelihood: Associated likelihood
        test_x: [N, D] test inputs
        batch_size: Batch size for prediction

    Returns:
        Dictionary with 'mean', 'variance', 'lower', 'upper' tensors
    """
    model.eval()
    likelihood.eval()

    device = next(model.parameters()).device
    test_x = test_x.to(device)

    means = []
    variances = []
    lowers = []
    uppers = []

    n = len(test_x)
    for i in range(0, n, batch_size):
        batch_x = test_x[i:i + batch_size]

        with gpytorch.settings.fast_pred_var():
            pred = likelihood(model(batch_x))
            lower, upper = pred.confidence_region()

        means.append(pred.mean.cpu())
        variances.append(pred.variance.cpu())
        lowers.append(lower.cpu())
        uppers.append(upper.cpu())

    return {
        'mean': torch.cat(means),
        'variance': torch.cat(variances),
        'lower': torch.cat(lowers),
        'upper': torch.cat(uppers)
    }


@torch.no_grad()
def predict_svgp_classification(
    model: SVGPClassification,
    likelihood: SoftmaxLikelihood,
    test_x: torch.Tensor,
    batch_size: int = 2048,
    num_samples: int = 64
) -> Dict[str, torch.Tensor]:
    """
    Make predictions with a trained SVGP classification model.

    Args:
        model: Trained SVGPClassification
        likelihood: Associated likelihood
        test_x: [N, D] test inputs
        batch_size: Batch size for prediction
        num_samples: Number of MC samples for prediction

    Returns:
        Dictionary with 'probs' (class probabilities) and 'predictions' (argmax)
    """
    model.eval()
    likelihood.eval()

    device = next(model.parameters()).device
    test_x = test_x.to(device)

    all_probs = []

    n = len(test_x)
    for i in range(0, n, batch_size):
        batch_x = test_x[i:i + batch_size]

        with gpytorch.settings.num_likelihood_samples(num_samples):
            pred = likelihood(model(batch_x))
            probs = pred.probs.mean(0)  # Average over samples

        all_probs.append(probs.cpu())

    probs = torch.cat(all_probs)
    predictions = probs.argmax(dim=1)

    return {
        'probs': probs,
        'predictions': predictions
    }


def evaluate_svgp_regression(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        predictions: Dictionary from predict_svgp_regression
        targets: True targets
        y_min: Original target minimum (for denormalization)
        y_max: Original target maximum (for denormalization)

    Returns:
        Dictionary with MSE, MAE, R2, and optionally MAE in original scale
    """
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


def evaluate_svgp_classification(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Dictionary from predict_svgp_classification
        targets: True class labels

    Returns:
        Dictionary with accuracy
    """
    pred_classes = predictions['predictions']
    accuracy = (pred_classes == targets).float().mean().item()

    return {
        'accuracy': accuracy
    }
