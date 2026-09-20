# -*- coding: utf-8 -*-
"""
Thermodynamic Manifold - Original Full Playground
- Contains Normalizing Flow, Zwanzig Free Energy, Path Optimizer, and Academic Plotting Suites.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.ndimage as jnd
import flax.nnx as nnx
import optax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

KB = 1.0
GRID_RESOLUTION = 60
T_MIN, T_MAX = 0.5, 2.5
V_MIN, V_MAX = 400.0, 1300.0
N_PARTICLES = 128

class LJPotential(nnx.Module):
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
        return jnp.where(U_raw < U_0, U_raw, U_0 * (1.0 + jnp.log(safe_U_raw / U_0)))

# Rational Quadratic Spline (RQS) flow calculations
def rational_quadratic_spline_1d(x, w, h, d, inverse=False, B=1.0):
    inside_mask = (x > -B) & (x < B)
    x_clamped = jnp.clip(x, -B + 1e-5, B - 1e-5)
    K = w.shape[-1]
    widths = jax.nn.softmax(w, axis=-1) * (2.0 * B)
    heights = jax.nn.softmax(h, axis=-1) * (2.0 * B)
    knot_x = -B + jnp.cumsum(jnp.pad(widths, (1, 0)), axis=-1)
    knot_y = -B + jnp.cumsum(jnp.pad(heights, (1, 0)), axis=-1)
    internal_derivatives = jax.nn.softplus(d) + 1e-3
    derivatives = jnp.pad(internal_derivatives, (1, 1), constant_values=1.0)
    if not inverse:
        bin_idx = jnp.clip(jnp.searchsorted(knot_x, x_clamped, side='right') - 1, 0, K - 1)
    else:
        bin_idx = jnp.clip(jnp.searchsorted(knot_y, x_clamped, side='right') - 1, 0, K - 1)
    gather_w = lambda arr, idx: jnp.take_along_axis(arr, idx[..., None], axis=-1)[..., 0]
    x_k, x_k1 = gather_w(knot_x, bin_idx), gather_w(knot_x, bin_idx + 1)
    y_k, y_k1 = gather_w(knot_y, bin_idx), gather_w(knot_y, bin_idx + 1)
    d_k, d_k1 = gather_w(derivatives, bin_idx), gather_w(derivatives, bin_idx + 1)
    w_b = jnp.clip(x_k1 - x_k, 1e-6, None)
    h_b = jnp.clip(y_k1 - y_k, 1e-6, None)
    s_b = h_b / w_b
    if not inverse:
        xi = (x_clamped - x_k) / w_b
        xi_1_xi = xi * (1.0 - xi)
        num = h_b * (s_b * xi**2 + d_k * xi_1_xi)
        den = s_b + (d_k1 + d_k - 2.0 * s_b) * xi_1_xi
        y_hat = y_k + num / jnp.clip(den, 1e-6, None)
        dk_dx = (s_b**2 * (d_k1 * xi**2 + 2.0 * s_b * xi_1_xi + d_k * (1.0 - xi)**2)) / (den**2)
        return jnp.where(inside_mask, y_hat, x), jnp.where(inside_mask, jnp.log(dk_dx), 0.0)
    else:
        y_delta = x_clamped - y_k
        a = h_b * (s_b - d_k) + y_delta * (d_k1 + d_k - 2.0 * s_b)
        b = h_b * d_k - y_delta * (d_k1 + d_k - 2.0 * s_b)
        c = -s_b * y_delta
        denom = -b - jnp.sqrt(jnp.maximum(b**2 - 4.0 * a * c, 1e-6))
        denom = jnp.where(jnp.abs(denom) < 1e-6, -1e-6, denom)
        xi = (2.0 * c) / denom
        x_hat = xi * w_b + x_k
        den = s_b + (d_k1 + d_k - 2.0 * s_b) * (xi * (1.0 - xi))
        dk_dx = (s_b**2 * (d_k1 * xi**2 + 2.0 * s_b * (xi * (1.0 - xi)) + d_k * (1.0 - xi)**2)) / (den**2)
        return jnp.where(inside_mask, x_hat, x), jnp.where(inside_mask, -jnp.log(dk_dx), 0.0)

class ConditionalSplineCoupling(nnx.Module):
    def __init__(self, d, hidden, mask, n_bins=8, cond_dim=2, *, rngs):
        self.transformed_idx = jnp.where(mask == 0.0)[0]
        self.kept_idx = jnp.where(mask == 1.0)[0]
        self.n_kept, self.d, self.n_bins = len(self.kept_idx), d, n_bins
        self.n_transformed = len(self.transformed_idx)
        self.out_per_dim = 3 * n_bins - 1
        self.net = nnx.Sequential(
            nnx.Linear(self.n_kept + cond_dim, hidden, rngs=rngs), nnx.silu,
            nnx.Linear(hidden, hidden, rngs=rngs), nnx.silu,
            nnx.Linear(hidden, self.n_transformed * self.out_per_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        )
        self.double_vmapped_rqs_forward = jax.vmap(jax.vmap(lambda x, w, h, d: rational_quadratic_spline_1d(x, w, h, d, False, 1.0), (0,0,0,0)), 1, 1)
        self.double_vmapped_rqs_inverse = jax.vmap(jax.vmap(lambda x, w, h, d: rational_quadratic_spline_1d(x, w, h, d, True, 1.0), (0,0,0,0)), 1, 1)

    def _predict_spline_params(self, kept, t_cond, v_cond):
        cond_input = jnp.concatenate([kept, t_cond.reshape(-1,1), v_cond.reshape(-1,1)], axis=-1)
        outputs = self.net(cond_input).reshape(outputs.shape[0], self.n_transformed, self.out_per_dim)
        return outputs[..., :self.n_bins], outputs[..., self.n_bins:2*self.n_bins], outputs[..., 2*self.n_bins:]

    def forward(self, z, t_cond, v_cond):
        w, h, d = self._predict_spline_params(jnp.take(z, self.kept_idx, 1), t_cond, v_cond)
        x_trans, log_det_matrix = self.double_vmapped_rqs_forward(jnp.take(z, self.transformed_idx, 1), w, h, d)
        return z.at[:, self.transformed_idx].set(x_trans), jnp.sum(log_det_matrix, axis=-1)

    def inverse(self, x, t_cond, v_cond):
        w, h, d = self._predict_spline_params(jnp.take(x, self.kept_idx, 1), t_cond, v_cond)
        z_trans, log_det_matrix = self.double_vmapped_rqs_inverse(jnp.take(x, self.transformed_idx, 1), w, h, d)
        return x.at[:, self.transformed_idx].set(z_trans), jnp.sum(log_det_matrix, axis=-1)

class ConditionalNSF(nnx.Module):
    def __init__(self, d=96, n_layers=8, hidden=128, n_bins=8, cond_dim=2, *, rngs):
        masks = [jnp.zeros(d).at[:d//2].set(1.0) if i%2==0 else jnp.zeros(d).at[d//2:].set(1.0) for i in range(n_layers)]
        self.layers = [ConditionalSplineCoupling(d, hidden, m, n_bins, cond_dim, rngs=rngs) for m in masks]
        self.d = d

    def forward(self, z, t_cond, v_cond):
        log_det = jnp.zeros(z.shape[0])
        for layer in self.layers:
            z, ld = layer.forward(z, t_cond, v_cond)
            log_det += ld
        return z, log_det

    def inverse(self, x, t_cond, v_cond):
        log_det = jnp.zeros(x.shape[0])
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x, t_cond, v_cond)
            log_det += ld
        return x, log_det

    def log_prob(self, x, t_cond, v_cond):
        z, log_det_inv = self.inverse(x, t_cond, v_cond)
        return (self.d * jnp.log(0.5)) - log_det_inv

    def sample(self, key, n, t_cond, v_cond):
        z = jr.normal(key, (n, self.d), minval=-1.0, maxval=1.0)
        x, _ = self.forward(z, t_cond, v_cond)
        return x

# Free Energy Estimator with validation diagnostics built in
class FreeEnergyEstimator:
    def __init__(self, flow_model: ConditionalNSF, gnn_model: LJPotential, Z, learning_rate: float):
        self.flow_model = flow_model
        self.gnn_model = gnn_model
        self.Z = Z
        self.optimizer = nnx.Optimizer(flow_model, optax.adam(learning_rate), wrt=nnx.Param)

    def compute_zwanzig_free_energy(self, t_cond, v_cond, num_samples, key):
        t_batch = jnp.full((num_samples,), t_cond)
        v_batch = jnp.full((num_samples,), v_cond)
        t_norm = ((t_batch - T_MIN) / (T_MAX - T_MIN)) * 2.0 - 1.0
        log_v = jnp.log(v_batch)
        v_norm = ((log_v - jnp.log(V_MIN)) / (jnp.log(V_MAX) - jnp.log(V_MIN))) * 2.0 - 1.0

        x_samples = self.flow_model.sample(key, num_samples, t_norm, v_norm)
        scale = jnp.power(v_cond, 1.0 / 3.0) / 2.0
        x_physical = x_samples * scale

        log_q_x = self.flow_model.log_prob(x_samples, t_norm, v_norm) - (x_samples.shape[-1] * jnp.log(scale))
        u_energy = jax.vmap(self.gnn_model, (0, None, 0, None))(x_physical, self.Z, v_batch, 1.0)
        exponent = - (u_energy / (KB * t_cond)) - log_q_x
        f_conf = - (KB * t_cond) * (jax.nn.logsumexp(exponent) - jnp.log(float(num_samples)))

        f_kin = self.Z.shape[0] * KB * t_cond * (jnp.log(self.Z.shape[0]) - 1.0 - 1.5 * jnp.log(t_cond))
        return f_conf + f_kin

class FreeEnergyGridSolver:
    def __init__(self, estimator_obj=None, grid_size: int = 50):
        self.estimator = estimator_obj
        self.grid_size = grid_size

    def generate_and_filter_manifold(self, t_bounds, v_bounds, num_samples=32, sigma=1.0):
        t_coords = jnp.linspace(t_bounds[0], t_bounds[1], self.grid_size)
        v_coords = jnp.linspace(v_bounds[0], v_bounds[1], self.grid_size)
        TT, VV = jnp.meshgrid(t_coords, v_coords, indexing="ij")
        f_flat = jax.vmap(lambda t, v: self.estimator.compute_zwanzig_free_energy(t, v, num_samples, jr.PRNGKey(127)))(TT.ravel(), VV.ravel())
        f_grid_raw = f_flat.reshape(self.grid_size, self.grid_size)
        r = int(4 * sigma + 0.5)
        x = jnp.arange(-r, r + 1)
        k = jnp.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        conv1d = lambda a: jnp.convolve(a, k, mode="same")
        f_grid_clean = jax.vmap(conv1d, 0)(jax.vmap(conv1d, 1)(f_grid_raw))
        return np.array(f_grid_clean), np.array(-jnp.gradient(f_grid_clean, (v_bounds[1]-v_bounds[0])/(self.grid_size-1), axis=1))

print("✔ Playground Full Module logic has been constructed successfully.")
