"""Sinusoidal embedding extracted from the original QUTCC implementation."""

import math
import torch


def timestep_embedding(timesteps, dim, max_period=10000):
    """Map q values (N,) to float32 sinusoidal features (N,dim) on the same device.

    ``timesteps`` is the upstream parameter name: callers pass quantile levels,
    not epochs or diffusion steps. ``dim`` is feature width; ``max_period`` sets
    the frequency range. An odd width receives one zero-valued padding feature.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
