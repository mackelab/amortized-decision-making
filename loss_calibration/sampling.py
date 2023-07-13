from typing import Callable

import numpy as np
import torch


def inverse_transform_sampling(
    density: Callable,
    num_samples: int = 100,
    x_start: int = -5,
    x_end: int = 5,
    x_intervals: int = 1000,
):
    x_vec = torch.linspace(x_start, x_end, steps=x_intervals)

    density_vals = density(x_vec)
    if density_vals.ndim == 2:  # should only be 1d-otherwise tile doesn't work as expected
        density_vals = density_vals.squeeze()

    cdf_vals = torch.cumsum(density_vals, dim=0)
    cdf_vals = cdf_vals / torch.max(cdf_vals)

    base_samples = torch.rand(num_samples)
    tiled_cdf = cdf_vals.tile(dims=(num_samples, 1))
    tiled_samples = base_samples.tile(dims=(x_intervals, 1))

    dist_fn = torch.abs(tiled_samples - tiled_cdf.T)  # absolute distance between random samples and CDF values
    samples = x_vec[torch.argmin(dist_fn, dim=0)]

    return samples
