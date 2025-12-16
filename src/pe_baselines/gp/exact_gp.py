"""Exact GP models for regression and classification using GPyTorch."""

import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood, DirichletClassificationLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from typing import Optional, Literal, Tuple, Dict, List
import numpy as np
from tqdm import tqdm


class ExactGPRegression(ExactGP):
    """Exact GP for regression with configurable kernel (RBF, Matern)."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
        kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
        ard: bool = True
    ):
        if likelihood is None:
            likelihood = GaussianLikelihood()

        super().__init__(train_x, train_y, likelihood)

        input_dim = train_x.shape[1]
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


class ExactGPClassification(ExactGP):
    """Exact GP for multi-class classification using Dirichlet transformation."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        num_classes: int,
        kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
        ard: bool = True
    ):
        # Transform targets for Dirichlet classification
        likelihood = DirichletClassificationLikelihood(
            train_y.long(),
            learn_additional_noise=True
        )

        # Dirichlet approach creates transformed targets
        super().__init__(train_x, likelihood.transformed_targets, likelihood)

        self.num_classes = num_classes
        input_dim = train_x.shape[1]
        ard_num_dims = input_dim if ard else None

        # Multiple output GPs (one per class)
        self.mean_module = ConstantMean(batch_shape=torch.Size([num_classes]))

        if kernel_type == "rbf":
            base_kernel = RBFKernel(
                batch_shape=torch.Size([num_classes]),
                ard_num_dims=ard_num_dims
            )
        elif kernel_type == "matern32":
            base_kernel = MaternKernel(
                nu=1.5,
                batch_shape=torch.Size([num_classes]),
                ard_num_dims=ard_num_dims
            )
        elif kernel_type == "matern52":
            base_kernel = MaternKernel(
                nu=2.5,
                batch_shape=torch.Size([num_classes]),
                ard_num_dims=ard_num_dims
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


def train_exact_gp_regression(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
    ard: bool = True,
    num_iterations: int = 100,
    lr: float = 0.1,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Tuple[ExactGPRegression, GaussianLikelihood, List[float]]:
    """Train an Exact GP regression model. Returns (model, likelihood, loss_history)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Initialize model and likelihood
    likelihood = GaussianLikelihood().to(device)
    model = ExactGPRegression(
        train_x, train_y, likelihood,
        kernel_type=kernel_type, ard=ard
    ).to(device)

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    loss_history = []
    iterator = tqdm(range(num_iterations), desc="Training GP") if verbose else range(num_iterations)

    for i in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and (i + 1) % 20 == 0:
            lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu()
            if isinstance(iterator, tqdm):
                iterator.set_postfix(loss=loss.item(), ls=lengthscales.mean().item())

    return model, likelihood, loss_history


def train_exact_gp_classification(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    kernel_type: Literal["rbf", "matern32", "matern52"] = "matern52",
    ard: bool = True,
    num_iterations: int = 100,
    lr: float = 0.1,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Tuple[ExactGPClassification, DirichletClassificationLikelihood, List[float]]:
    """Train an Exact GP classification model. Returns (model, likelihood, loss_history)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Initialize model
    model = ExactGPClassification(
        train_x, train_y.long(), num_classes,
        kernel_type=kernel_type, ard=ard
    ).to(device)

    likelihood = model.likelihood

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Marginal log likelihood for classification
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    loss_history = []
    iterator = tqdm(range(num_iterations), desc="Training GP Classifier") if verbose else range(num_iterations)

    for i in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and isinstance(iterator, tqdm) and (i + 1) % 20 == 0:
            iterator.set_postfix(loss=loss.item())

    return model, likelihood, loss_history


@torch.no_grad()
def predict_regression(
    model: ExactGPRegression,
    likelihood: GaussianLikelihood,
    test_x: torch.Tensor,
    batch_size: int = 1000
) -> Dict[str, torch.Tensor]:
    """Predict with GP regression. Returns dict with 'mean', 'variance', 'lower', 'upper'."""
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
def predict_classification(
    model: ExactGPClassification,
    likelihood: DirichletClassificationLikelihood,
    test_x: torch.Tensor,
    batch_size: int = 1000
) -> Dict[str, torch.Tensor]:
    """Predict with GP classification. Returns dict with 'probs' and 'predictions'."""
    model.eval()
    likelihood.eval()

    device = next(model.parameters()).device
    test_x = test_x.to(device)

    all_samples = []

    n = len(test_x)
    for i in range(0, n, batch_size):
        batch_x = test_x[i:i + batch_size]

        with gpytorch.settings.fast_pred_var():
            test_dist = model(batch_x)

        # Sample from posterior for classification
        samples = test_dist.sample(torch.Size([256])).exp()
        samples = samples / samples.sum(-2, keepdim=True)
        all_samples.append(samples.cpu())

    samples = torch.cat(all_samples, dim=2)  # [256, num_classes, N]

    # Average over samples and transpose to [N, num_classes]
    probs = samples.mean(0).T  # [N, num_classes]
    predictions = probs.argmax(dim=1)

    return {
        'probs': probs,
        'predictions': predictions
    }


def evaluate_regression(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None
) -> Dict[str, float]:
    """Compute regression metrics (MSE, MAE, R2)."""
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


def evaluate_classification(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor
) -> Dict[str, float]:
    """Compute classification accuracy."""
    pred_classes = predictions['predictions']
    accuracy = (pred_classes == targets).float().mean().item()

    return {
        'accuracy': accuracy
    }
