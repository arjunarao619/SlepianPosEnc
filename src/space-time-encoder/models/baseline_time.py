# models/baseline_time.py
# Matches the original paper implementation exactly
import numpy as np
import torch
from scipy.special import legendre

inv_sqrt2 = 1.0 / np.sqrt(2.0)

# Legendre polynomials (NO orthonormalization - matches paper)
def legendre_fn(degree: int, t: torch.Tensor) -> torch.Tensor:
    t_np = t.detach().cpu().numpy()
    Pn_np = legendre(degree)(t_np)
    return torch.as_tensor(Pn_np, device=t.device, dtype=t.dtype)

level_0_time_embedding_functions = {
    # Baselines (matching paper exactly)
    "no_time":    lambda degree, t: torch.zeros_like(t, dtype=t.dtype, device=t.device),
    "time_copy":  lambda degree, t: t / 2,      # Paper divides by 2
    "monomial":   lambda degree, t: t**degree,

    # Legendre (no orthonormalization factor - matches paper)
    "legendre":   legendre_fn,

    # Fourier: sin(π*degree*t/2)/√2 and cos(π*degree*t/2)/√2 (matches paper)
    "sin": lambda degree, t: torch.sin(torch.pi * degree * t / 2) / np.sqrt(2),
    "cos": lambda degree, t: torch.cos(torch.pi * degree * t / 2) / np.sqrt(2),

    # Triangle variants (matches paper)
    "triangle_1": lambda degree, t: 2 * torch.maximum(
        (degree * t + 1) % 2 - 1,
        2 - (degree * t + 1) % 2 - 1
    ) - 1,
    "triangle_2": lambda degree, t: 2 * torch.maximum(
        (degree * t) % 2 - 1,
        2 - (degree * t) % 2 - 1
    ) - 1,

    # Constant (for completeness)
    "constant": lambda degree, t: torch.ones_like(t, dtype=t.dtype, device=t.device) * inv_sqrt2,
}

level_1_time_embedding_functions = {
    "fourier": lambda degree, t: (
        level_0_time_embedding_functions["sin"](degree // 2, t)
        if degree % 2 == 0
        else level_0_time_embedding_functions["cos"](degree // 2, t)
    ),
    "triangle": lambda degree, t: (
        level_0_time_embedding_functions["triangle_1"](degree // 2, t)
        if degree % 2 == 0
        else level_0_time_embedding_functions["triangle_2"](degree // 2, t)
    ),
}

time_embedding_functions = {
    **level_0_time_embedding_functions,
    **level_1_time_embedding_functions,
}
