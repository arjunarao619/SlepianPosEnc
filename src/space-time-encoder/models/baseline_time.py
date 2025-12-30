# models/baseline_time.py
import numpy as np
import torch
from scipy.special import legendre

# Orthonormal Legendre on [-1, 1]
def legendre_orthonormal(n: int, t: torch.Tensor) -> torch.Tensor:
    t_np = t.detach().cpu().numpy()
    Pn_np = legendre(n)(t_np)
    Pn = torch.as_tensor(Pn_np, device=t.device, dtype=t.dtype)
    return Pn * np.sqrt((2.0 * n + 1.0) / 2.0)

inv_sqrt2 = 1.0 / np.sqrt(2.0)

level_0_time_embedding_functions = {
    # Baselines
    "no_time":    lambda degree, t: torch.zeros_like(t, dtype=t.dtype, device=t.device),  # unused when no_time path is active
    "time_copy":  lambda degree, t: t,          # they duplicate t across dims (non-orthogonal) :contentReference[oaicite:3]{index=3}
    "monomial":   lambda degree, t: t**degree,  # non-orthogonal

    # Orthonormal Legendre
    "legendre":   legendre_orthonormal,

    # Fourier pair with 1/sqrt(2) scaling & πk t / 2 argument (orthogonal) :contentReference[oaicite:4]{index=4}
    "sin": lambda degree, t: torch.sin(torch.pi * (degree + 1) * t / 2) * inv_sqrt2,
    "cos": lambda degree, t: torch.cos(torch.pi * (degree + 1) * t / 2) * inv_sqrt2,

    # Two triangle variants (piecewise linear family, non-orthogonal) :contentReference[oaicite:5]{index=5}
    "triangle_1": lambda degree, t: 2 * torch.maximum(
        ((degree + 1) * t + 1) % 2 - 1,
        2 - ((degree + 1) * t + 1) % 2 - 1
    ) - 1,
    "triangle_2": lambda degree, t: 2 * torch.maximum(
        ((degree + 1) * t) % 2 - 1,
        2 - ((degree + 1) * t) % 2 - 1
    ) - 1,

    # A constant function (normalized on [-1, 1])—not used for the “no time” baseline per paper (§4.1).
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
