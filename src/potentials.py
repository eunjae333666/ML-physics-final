# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
from flax import nnx

class LJPotential(nnx.Module):
    """Lennard-Jones Potential with Soft-Core Modification preventing energy explosion."""
    def __init__(self, sigma=1.0, epsilon=1.0, wall_k=20.0):
        self.sigma = sigma
        self.epsilon = epsilon
        self.wall_k = wall_k

    def __call__(self, x, Z=None, V=None, progress=1.0):
        if x.ndim == 1:
            x = x.reshape(-1, 3)
        N = x.shape[0]
        diff = x[:, None, :] - x[None, :, :]
        r2 = jnp.sum(diff**2, axis=-1)
        mask = 1.0 - jnp.eye(N)
        r2_safe_diag = r2 + (1.0 - mask)

        alpha = jnp.interp(progress, jnp.array([0.0, 0.8]), jnp.array([0.5, 0.001]))
        r6 = (r2_safe_diag / self.sigma**2) ** 3
        denom = alpha + r6
        inv_r6_soft = 1.0 / denom
        lj_matrix = 4.0 * self.epsilon * (inv_r6_soft**2 - inv_r6_soft)

        U_0 = 1000.0
        U_raw = 0.5 * jnp.sum(mask * lj_matrix)
        safe_U_raw = jnp.maximum(U_raw, U_0)
        U_eff = jnp.where(
            U_raw < U_0,
            U_raw,
            U_0 * (1.0 + jnp.log(safe_U_raw / U_0))
        )
        return U_eff
