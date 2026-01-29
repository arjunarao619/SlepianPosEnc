"""Random Fourier Features (RFF) for fast GP approximation on planar coordinates."""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import gamma as gamma_fn
from typing import Optional, Literal, Tuple, Dict
from tqdm import tqdm


class PlanarRFF(nn.Module):
    """Random Fourier Features approximating shift-invariant kernels (RBF, Matern)."""

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
        """Sample frequencies from kernel spectral density."""
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
        """Compute random Fourier features. x: [N, D] -> [N, num_features]."""
        # Scale inputs by lengthscale
        # omega is already sampled for unit lengthscale, so divide by lengthscale
        scaled_x = x / self.lengthscale

        # Project: z = x @ omega^T + bias
        projection = torch.matmul(scaled_x, self.omega.T) + self.bias

        # Random features: sqrt(2/D) * cos(projection)
        features = torch.sqrt(torch.tensor(2.0 / self.num_features)) * torch.cos(projection)

        return features


class RFFRegression(nn.Module):
    """Bayesian linear regression on RFF features (closed-form posterior)."""

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
        """Compute closed-form Bayesian linear regression posterior."""
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
        """Predict with uncertainty. Returns dict with 'mean', 'variance'."""
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
    """Logistic regression on RFF features for classification."""

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
        """Forward pass. Returns [N, num_classes] logits."""
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
        """Train using cross-entropy loss. Returns list of training losses."""
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
        """Predict. Returns dict with 'probs' and 'predictions'."""
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
    """Train RFF regression with Bayesian linear regression. Returns (model, metadata)."""
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
    """Train RFF classification with logistic regression. Returns (model, metadata)."""
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
    """Evaluate RFF regression. Returns dict with MSE, MAE, R2."""
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
    test_y: torch.Tensor,
    batch_size: int = 512
) -> Dict[str, float]:
    """Evaluate RFF classification with batched inference. Returns dict with accuracy."""
    device = model.rff.omega.device
    model.eval()

    all_preds = []
    n = len(test_x)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_x = test_x[i:i + batch_size].to(device)
            preds = model(batch_x).argmax(dim=1)
            all_preds.append(preds.cpu())
            del batch_x

    predictions = torch.cat(all_preds)
    accuracy = (predictions == test_y.cpu()).float().mean().item()

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {'accuracy': accuracy}


# =============================================================================
# Deep RFF Implementation (aligned with DeepRandomFeatures paper)
# =============================================================================

class DeepRFFLayer(nn.Module):
    """
    Deep RFF layer matching the DeepRandomFeatures paper (arxiv 2412.11350).

    Key differences from PlanarRFF:
    - Lengthscale is baked into the spectral sampling (StudentT scale parameter)
    - Amplitude is included in the scaling factor
    - Includes a learnable output layer with weights initialized from N(0,1)

    Architecture: input -> frozen_linear -> cos -> scale -> learnable_linear -> output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lengthscale: float = 1.0,
        nu: float = 2.5,
        amplitude: float = 1.0,
        seed: int = 42
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lengthscale = lengthscale
        self.nu = nu
        self.amplitude = amplitude

        # Initialize layers
        self.hidden_layer = nn.Linear(input_dim, hidden_dim, bias=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=True)

        # Sample weights from spectral density (matching paper's approach)
        torch.manual_seed(seed)

        # StudentT(df=2*nu, scale=1/lengthscale) - paper's approach
        # This bakes lengthscale directly into the weight sampling
        student_t = torch.distributions.StudentT(df=2 * nu, scale=1.0 / lengthscale)
        self.hidden_layer.weight.data = student_t.sample((hidden_dim, input_dim))

        # Bias: Uniform(0, 2*pi)
        self.hidden_layer.bias.data.uniform_(0, 2 * np.pi)

        # Output layer: N(0, 1) initialization (learnable)
        self.output_layer.weight.data.normal_(0, 1)
        self.output_layer.bias.data.zero_()

        # Freeze hidden layer weights (RFF weights should not be trained)
        self.hidden_layer.weight.requires_grad = False
        self.hidden_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass matching paper's implementation."""
        # Project through frozen RFF layer
        x = self.hidden_layer(x)

        # Apply cosine with amplitude scaling (paper's formula)
        # scaling = sqrt(2 * amplitude^2 / hidden_dim)
        scaling = torch.sqrt(torch.tensor(
            2.0 * self.amplitude ** 2 / self.hidden_dim,
            device=x.device, dtype=x.dtype
        ))
        x = scaling * torch.cos(x)

        # Learnable output projection
        x = self.output_layer(x)
        return x


class DeepRFFClassifier(nn.Module):
    """
    Deep RFF classifier with stacked layers and skip connections.

    Matches the DeepSpatiotemporalGPNN architecture from the paper, adapted for classification.

    Architecture:
    - Layer 0: input_dim -> hidden_dim -> bottleneck_dim
    - Layer 1+: (bottleneck_dim + input_dim) -> hidden_dim -> bottleneck_dim
    - Skip connections: concatenate layer output with original input
    - Final: bottleneck_dim -> num_classes
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        num_classes: int,
        num_layers: int = 3,
        lengthscale: float = 1.0,
        nu: float = 2.5,
        amplitude: float = 1.0,
        seed: int = 42
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Build stacked RFF layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer: input_dim -> bottleneck_dim
                layer_input_dim = input_dim
            else:
                # Subsequent layers: (bottleneck_dim + input_dim) -> bottleneck_dim
                # Due to skip connections
                layer_input_dim = bottleneck_dim + input_dim

            layer = DeepRFFLayer(
                input_dim=layer_input_dim,
                hidden_dim=hidden_dim,
                output_dim=bottleneck_dim,
                lengthscale=lengthscale,
                nu=nu,
                amplitude=amplitude,
                seed=seed + i  # Different seed per layer
            )
            self.layers.append(layer)

        # Final classifier
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections."""
        original_input = x

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Add skip connection (except for last layer)
            if i < len(self.layers) - 1:
                x = torch.cat([x, original_input], dim=1)

        # Classification head
        logits = self.classifier(x)
        return logits


def train_deep_rff_classification(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    hidden_dim: int = 1000,
    bottleneck_dim: int = 64,
    num_layers: int = 3,
    lengthscale: Optional[float] = None,
    nu: float = 2.5,
    amplitude: float = 1.0,
    num_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[DeepRFFClassifier, Dict]:
    """
    Train Deep RFF classifier matching the DeepRandomFeatures paper.

    Returns (model, metadata).
    """
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Estimate lengthscale if not provided (median heuristic)
    if lengthscale is None:
        with torch.no_grad():
            sample_size = min(1000, len(train_x))
            dists = torch.cdist(train_x[:sample_size], train_x[:sample_size])
            triu_mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
            lengthscale = torch.median(dists[triu_mask]).item()
        if verbose:
            print(f"Using median heuristic lengthscale: {lengthscale:.4f}")

    # Create model
    model = DeepRFFClassifier(
        input_dim=train_x.shape[1],
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        lengthscale=lengthscale,
        nu=nu,
        amplitude=amplitude,
        seed=seed
    ).to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create data loader
    dataset = TensorDataset(train_x, train_y.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer (only trainable parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    metadata = {
        'hidden_dim': hidden_dim,
        'bottleneck_dim': bottleneck_dim,
        'num_layers': num_layers,
        'lengthscale': lengthscale,
        'nu': nu,
        'amplitude': amplitude,
        'final_loss': losses[-1] if losses else None
    }

    return model, metadata


@torch.no_grad()
def evaluate_deep_rff_classification(
    model: DeepRFFClassifier,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int = 512
) -> Dict[str, float]:
    """
    Evaluate Deep RFF classifier with batched inference and memory cleanup.

    Returns dict with accuracy.
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    n = len(test_x)

    for i in range(0, n, batch_size):
        batch_x = test_x[i:i + batch_size].to(device)
        preds = model(batch_x).argmax(dim=1)
        all_preds.append(preds.cpu())
        del batch_x

    predictions = torch.cat(all_preds)
    accuracy = (predictions == test_y.cpu()).float().mean().item()

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {'accuracy': accuracy}


class DeepRFFRegressor(nn.Module):
    """
    Deep RFF regressor with stacked layers and skip connections.

    Matches the DeepSpatiotemporalGPNN architecture from the paper, adapted for regression.

    Architecture:
    - Layer 0: input_dim -> hidden_dim -> bottleneck_dim
    - Layer 1+: (bottleneck_dim + input_dim) -> hidden_dim -> bottleneck_dim
    - Skip connections: concatenate layer output with original input
    - Final: bottleneck_dim -> 1 (regression output)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        num_layers: int = 3,
        lengthscale: float = 1.0,
        nu: float = 2.5,
        amplitude: float = 1.0,
        seed: int = 42
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_layers = num_layers

        # Build stacked RFF layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer: input_dim -> bottleneck_dim
                layer_input_dim = input_dim
            else:
                # Subsequent layers: (bottleneck_dim + input_dim) -> bottleneck_dim
                # Due to skip connections
                layer_input_dim = bottleneck_dim + input_dim

            layer = DeepRFFLayer(
                input_dim=layer_input_dim,
                hidden_dim=hidden_dim,
                output_dim=bottleneck_dim,
                lengthscale=lengthscale,
                nu=nu,
                amplitude=amplitude,
                seed=seed + i  # Different seed per layer
            )
            self.layers.append(layer)

        # Final regression head (single output)
        self.regressor = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections. Returns [N, 1] predictions."""
        original_input = x

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Add skip connection (except for last layer)
            if i < len(self.layers) - 1:
                x = torch.cat([x, original_input], dim=1)

        # Regression head
        output = self.regressor(x)
        return output


def train_deep_rff_regression(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    hidden_dim: int = 1000,
    bottleneck_dim: int = 64,
    num_layers: int = 3,
    lengthscale: Optional[float] = None,
    nu: float = 2.5,
    amplitude: float = 1.0,
    num_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[DeepRFFRegressor, Dict]:
    """
    Train Deep RFF regressor matching the DeepRandomFeatures paper.

    Returns (model, metadata).
    """
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Ensure train_y is the right shape
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(1)

    # Estimate lengthscale if not provided (median heuristic)
    if lengthscale is None:
        with torch.no_grad():
            sample_size = min(1000, len(train_x))
            dists = torch.cdist(train_x[:sample_size], train_x[:sample_size])
            triu_mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
            lengthscale = torch.median(dists[triu_mask]).item()
        if verbose:
            print(f"Using median heuristic lengthscale: {lengthscale:.4f}")

    # Create model
    model = DeepRFFRegressor(
        input_dim=train_x.shape[1],
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_layers=num_layers,
        lengthscale=lengthscale,
        nu=nu,
        amplitude=amplitude,
        seed=seed
    ).to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create data loader
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer (only trainable parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, MSE Loss: {avg_loss:.6f}")

    metadata = {
        'hidden_dim': hidden_dim,
        'bottleneck_dim': bottleneck_dim,
        'num_layers': num_layers,
        'lengthscale': lengthscale,
        'nu': nu,
        'amplitude': amplitude,
        'final_loss': losses[-1] if losses else None
    }

    return model, metadata


@torch.no_grad()
def evaluate_deep_rff_regression(
    model: DeepRFFRegressor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int = 512,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluate Deep RFF regressor with batched inference and memory cleanup.

    Returns dict with mse, mae, r2 (and mae_original if y_min/y_max provided).
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    n = len(test_x)

    for i in range(0, n, batch_size):
        batch_x = test_x[i:i + batch_size].to(device)
        preds = model(batch_x).squeeze()
        all_preds.append(preds.cpu())
        del batch_x

    predictions = torch.cat(all_preds)
    targets = test_y.cpu()

    # Ensure same shape
    if targets.dim() > 1:
        targets = targets.squeeze()

    # Compute metrics
    mse = ((predictions - targets) ** 2).mean().item()
    mae = (predictions - targets).abs().mean().item()

    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - (ss_res / ss_tot)).item()

    result = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    # Compute MAE in original scale if normalization params provided
    if y_min is not None and y_max is not None:
        pred_orig = predictions * (y_max - y_min) + y_min
        targets_orig = targets * (y_max - y_min) + y_min
        result['mae_original'] = (pred_orig - targets_orig).abs().mean().item()

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
