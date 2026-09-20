# -*- coding: utf-8 -*-
"""
Thermodynamic Manifold Core Module
- JAX/Flax (nnx) based physics-informed Normalizing Flow & Path Optimizer
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.ndimage as jnd
import flax.nnx as nnx
import optax
import numpy as np
import os

# -----------------------------------------------------------------
# 1. Lennard-Jones Potential with Soft-Core Modification
# -----------------------------------------------------------------
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
        U_eff = jnp.where(
            U_raw < U_0,
            U_raw,
            U_0 * (1.0 + jnp.log(safe_U_raw / U_0))
        )
        return U_eff

# -----------------------------------------------------------------
# 2. Rational Quadratic Spline (RQS) Conditional NSF Flow
# -----------------------------------------------------------------
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
    x_k = gather_w(knot_x, bin_idx)
    x_k1 = gather_w(knot_x, bin_idx + 1)
    y_k = gather_w(knot_y, bin_idx)
    y_k1 = gather_w(knot_y, bin_idx + 1)
    d_k = gather_w(derivatives, bin_idx)
    d_k1 = gather_w(derivatives, bin_idx + 1)

    w_b = jnp.clip(x_k1 - x_k, 1e-6, None)
    h_b = jnp.clip(y_k1 - y_k, 1e-6, None)
    s_b = h_b / w_b

    if not inverse:
        xi = (x_clamped - x_k) / w_b
        xi_1_xi = xi * (1.0 - xi)
        num = h_b * (s_b * xi**2 + d_k * xi_1_xi)
        num = jnp.where(jnp.isnan(num), 0.0, num)
        den = s_b + (d_k1 + d_k - 2.0 * s_b) * xi_1_xi
        den = jnp.clip(den, 1e-6, None)
        y_hat = y_k + num / den
        num_deriv = s_b**2 * (d_k1 * xi**2 + 2.0 * s_b * xi_1_xi + d_k * (1.0 - xi)**2)
        den_deriv = (s_b + (d_k1 + d_k - 2.0 * s_b) * xi_1_xi)**2
        dk_dx = num_deriv / den_deriv
        log_det_hat = jnp.log(dk_dx)
        return jnp.where(inside_mask, y_hat, x), jnp.where(inside_mask, log_det_hat, 0.0)
    else:
        y_delta = x_clamped - y_k
        a = h_b * (s_b - d_k) + y_delta * (d_k1 + d_k - 2.0 * s_b)
        b = h_b * d_k - y_delta * (d_k1 + d_k - 2.0 * s_b)
        c = -s_b * y_delta
        rhs = jnp.maximum(b**2 - 4.0 * a * c, 1e-6)
        sqrt_rhs = jnp.sqrt(rhs)
        denom = -b - sqrt_rhs
        denom = jnp.where(jnp.abs(denom) < 1e-6, -1e-6, denom)
        xi = (2.0 * c) / denom
        x_hat = xi * w_b + x_k
        xi_1_xi = xi * (1.0 - xi)
        num_deriv = s_b**2 * (d_k1 * xi**2 + 2.0 * s_b * xi_1_xi + d_k * (1.0 - xi)**2)
        den_deriv = (s_b + (d_k1 + d_k - 2.0 * s_b) * xi_1_xi)**2
        dk_dx = num_deriv / den_deriv
        log_det_hat = -jnp.log(dk_dx)
        return jnp.where(inside_mask, x_hat, x), jnp.where(inside_mask, log_det_hat, 0.0)

class ConditionalSplineCoupling(nnx.Module):
    def __init__(self, d, hidden, mask, n_bins=8, cond_dim=2, *, rngs):
        self.transformed_idx = jnp.where(mask == 0.0)[0]
        self.kept_idx = jnp.where(mask == 1.0)[0]
        self.n_kept = len(self.kept_idx)
        self.d = d
        self.n_bins = n_bins
        self.n_transformed = len(self.transformed_idx)
        self.out_per_dim = 3 * n_bins - 1

        self.net = nnx.Sequential(
            nnx.Linear(self.n_kept + cond_dim, hidden, rngs=rngs), nnx.silu,
            nnx.Linear(hidden, hidden, rngs=rngs), nnx.silu,
            nnx.Linear(hidden, self.n_transformed * self.out_per_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        )
        self.double_vmapped_rqs_forward = jax.vmap(
            jax.vmap(lambda x, w, h, d: rational_quadratic_spline_1d(x, w, h, d, inverse=False, B=1.0), in_axes=(0, 0, 0, 0)),
            in_axes=(1, 1, 1, 1), out_axes=1
        )
        self.double_vmapped_rqs_inverse = jax.vmap(
            jax.vmap(lambda x, w, h, d: rational_quadratic_spline_1d(x, w, h, d, inverse=True, B=1.0), in_axes=(0, 0, 0, 0)),
            in_axes=(1, 1, 1, 1), out_axes=1
        )

    def _predict_spline_params(self, kept, t_cond, v_cond):
        t_c = jnp.reshape(t_cond, (kept.shape[0], 1))
        v_c = jnp.reshape(v_cond, (kept.shape[0], 1))
        cond_input = jnp.concatenate([kept, t_c, v_c], axis=-1)
        outputs = self.net(cond_input)
        outputs = outputs.reshape(outputs.shape[0], self.n_transformed, self.out_per_dim)
        w = outputs[..., :self.n_bins]
        h = outputs[..., self.n_bins:2*self.n_bins]
        d = outputs[..., 2*self.n_bins:]
        return w, h, d

    def forward(self, z, t_cond, v_cond):
        kept_data = jnp.take(z, self.kept_idx, axis=1)
        w, h, d = self._predict_spline_params(kept_data, t_cond, v_cond)
        z_trans = jnp.take(z, self.transformed_idx, axis=1)
        x_trans, log_det_matrix = self.double_vmapped_rqs_forward(z_trans, w, h, d)
        log_det = jnp.sum(log_det_matrix, axis=-1)
        x = z.at[:, self.transformed_idx].set(x_trans)
        return x, log_det

    def inverse(self, x, t_cond, v_cond):
        kept_data = jnp.take(x, self.kept_idx, axis=1)
        w, h, d = self._predict_spline_params(kept_data, t_cond, v_cond)
        x_trans = jnp.take(x, self.transformed_idx, axis=1)
        z_trans, log_det_matrix = self.double_vmapped_rqs_inverse(x_trans, w, h, d)
        log_det = jnp.sum(log_det_matrix, axis=-1)
        z = x.at[:, self.transformed_idx].set(z_trans)
        return z, log_det

class ConditionalNSF(nnx.Module):
    def __init__(self, d=96, n_layers=8, hidden=128, n_bins=8, cond_dim=2, *, rngs):
        masks = []
        for i in range(n_layers):
            mask = jnp.zeros(d)
            if i % 2 == 0:
                mask = mask.at[:d//2].set(1.0)
            else:
                mask = mask.at[d//2:].set(1.0)
            masks.append(mask)
        self.layers = [ConditionalSplineCoupling(d, hidden, m, n_bins, cond_dim, rngs=rngs) for m in masks]
        self.d = d

    def forward(self, z, t_cond, v_cond):
        log_det = jnp.zeros(z.shape[0])
        x = z
        for layer in self.layers:
            x, ld = layer.forward(x, t_cond, v_cond)
            log_det += ld
        return x, log_det

    def inverse(self, x, t_cond, v_cond):
        log_det = jnp.zeros(x.shape[0])
        z = x
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z, t_cond, v_cond)
            log_det += ld
        return z, log_det

    def log_prob(self, x, t_cond, v_cond):
        z, log_det_inv = self.inverse(x, t_cond, v_cond)
        log_pz = self.d * jnp.log(0.5)
        return log_pz - log_det_inv

    def sample(self, key, n, t_cond, v_cond):
        z = jr.normal(key, (n, self.d), minval=-1.0, maxval=1.0)
        x, _ = self.forward(z, t_cond, v_cond)
        return x

# -----------------------------------------------------------------
# 3. Free Energy Estimator with pure JIT core
# -----------------------------------------------------------------
class FreeEnergyEstimator:
    def __init__(self, flow_model: ConditionalNSF, gnn_model: LJPotential, Z, learning_rate: float, KB=1.0):
        self.flow_model = flow_model
        self.gnn_model = gnn_model
        self.Z = Z
        self.KB = KB
        self.optimizer = nnx.Optimizer(flow_model, optax.adam(learning_rate), wrt=nnx.Param)

    def compute_zwanzig_free_energy(self, t_cond, v_cond, num_samples, key, T_MIN=0.5, T_MAX=2.5, V_MIN=400, V_MAX=1300):
        t_batch = jnp.full((num_samples,), t_cond)
        v_batch = jnp.full((num_samples,), v_cond)
        t_norm = ((t_batch - T_MIN) / (T_MAX - T_MIN)) * 2.0 - 1.0
        log_v = jnp.log(v_batch)
        v_norm = ((log_v - jnp.log(V_MIN)) / (jnp.log(V_MAX) - jnp.log(V_MIN))) * 2.0 - 1.0

        x_samples = self.flow_model.sample(key, num_samples, t_norm, v_norm)
        L_box = jnp.power(v_cond, 1.0 / 3.0)
        scale = jnp.squeeze(L_box / 2.0)
        x_physical = x_samples * scale

        log_q_x_raw = self.flow_model.log_prob(x_samples, t_norm, v_norm)
        log_q_x = log_q_x_raw - (x_samples.shape[-1] * jnp.log(scale))
        u_energy = jax.vmap(self.gnn_model, in_axes=(0, None, 0, None))(x_physical, self.Z, v_batch, 1.0)

        u_scaled = u_energy / (self.KB * t_cond)
        exponent = -u_scaled - log_q_x
        f_conf = -(self.KB * t_cond) * (jax.nn.logsumexp(exponent) - jnp.log(float(num_samples)))

        N = self.Z.shape[0]
        t_val = jnp.squeeze(t_cond)
        f_kin = N * self.KB * t_val * (jnp.log(N) - 1.0 - 1.5 * jnp.log(t_val))
        return f_conf + f_kin

    def train_step(self, z, t_cond, v_cond, epoch, n_epochs, T_MIN=0.5, T_MAX=2.5, V_MIN=400, V_MAX=1300):
        progress = epoch / n_epochs
        loss, diag = _pure_jit_train_core(
            self.flow_model, self.optimizer, self.gnn_model,
            self.Z, z, t_cond, v_cond, progress, self.KB, T_MIN, T_MAX, V_MIN, V_MAX
        )
        return loss, diag

@nnx.jit
def _pure_jit_train_core(flow_model, optimizer, gnn_model, Z, z, t_cond, v_cond, progress, KB, T_MIN, T_MAX, V_MIN, V_MAX):
    t_norm = ((t_cond - T_MIN) / (T_MAX - T_MIN)) * 2.0 - 1.0
    log_v = jnp.log(v_cond)
    v_norm = ((log_v - jnp.log(V_MIN)) / (jnp.log(V_MAX) - jnp.log(V_MIN))) * 2.0 - 1.0

    def loss_fn(model):
        x, log_det = model.forward(z, t_norm, v_norm)
        v_cond_flat = jnp.reshape(v_cond, (-1,))
        L_box = jnp.power(v_cond_flat, 1.0 / 3.0)
        scale = L_box / 2.0
        x_physical = x * scale[:, None]
        log_det_physical = log_det + x.shape[-1] * jnp.log(scale)
        u_energy = jax.vmap(gnn_model, in_axes=(0, None, 0, None))(x_physical, Z, v_cond, progress)
        u_scaled = u_energy / (KB * jnp.squeeze(t_cond))
        loss = (u_scaled - log_det_physical).mean()
        return loss, {"u_mean": u_energy.mean(), "u_max": u_energy.max(), "u_min": u_energy.min(), "logdet_mean": log_det_physical.mean(), "loss_u": u_scaled.mean()}

    (loss, diag), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(flow_model)
    optimizer.update(flow_model, grads)
    return loss, diag

# -----------------------------------------------------------------
# 4. Free Energy Grid Solver
# -----------------------------------------------------------------
class FreeEnergyGridSolver:
    def __init__(self, estimator_obj=None, grid_size: int = 50):
        self.estimator = estimator_obj
        self.grid_size = grid_size

    def generate_and_filter_manifold(self, t_bounds: tuple, v_bounds: tuple, num_samples: int = 32, sigma: float = 1.0):
        if self.estimator is None:
            raise ValueError("New manifold requires an estimator instances.")
        t_min, t_max = t_bounds
        v_min, v_max = v_bounds
        t_coords = jnp.linspace(t_min, t_max, self.grid_size)
        v_coords = jnp.linspace(v_min, v_max, self.grid_size)
        TT, VV = jnp.meshgrid(t_coords, v_coords, indexing="ij")
        
        @jax.jit
        def _pure_parallel_scan(t_arr, v_arr):
            return jax.vmap(lambda t, v: self.estimator.compute_zwanzig_free_energy(
                t_cond=t, v_cond=v, num_samples=num_samples, key=jr.PRNGKey(127)
            ))(t_arr, v_arr)

        f_flat = _pure_parallel_scan(TT.ravel(), VV.ravel())
        f_grid_raw = f_flat.reshape(self.grid_size, self.grid_size)

        if sigma <= 0:
            f_grid_clean = f_grid_raw
        else:
            r = int(4 * sigma + 0.5)
            x = jnp.arange(-r, r + 1)
            k = jnp.exp(-0.5 * (x / sigma) ** 2)
            k = k / k.sum()
            conv1d = lambda a: jnp.convolve(a, k, mode="same")
            f_grid_clean = jax.vmap(conv1d, 0)(jax.vmap(conv1d, 1)(f_grid_raw))

        dv = (v_max - v_min) / (self.grid_size - 1)
        pressure_grid = -jnp.gradient(f_grid_clean, dv, axis=1)
        return np.array(f_grid_clean), np.array(pressure_grid)

    def save_to_disk(self, f_grid, p_grid, filename_prefix="ideal_gas_manifold"):
        np.save(f"{filename_prefix}_F.npy", f_grid)
        np.save(f"{filename_prefix}_P.npy", p_grid)

    def load_from_disk(self, filename_prefix="ideal_gas_manifold"):
        f_path = f"{filename_prefix}_F.npy"
        p_path = f"{filename_prefix}_P.npy"
        if os.path.exists(f_path) and os.path.exists(p_path):
            return np.load(f_path), np.load(p_path)
        return None, None

# -----------------------------------------------------------------
# 5. Power Path Optimizer
# -----------------------------------------------------------------
class PowerPathOptimizer(nnx.Module):
    def __init__(
        self, pressure_grid, t_bounds: tuple, v_bounds: tuple,
        num_points: int, learning_rate: float, T_H: float, T_L: float,
        kappa: float = 100.0, C_v: float = 1.5, lambda_smooth: float = 0.5
    ):
        self.pressure_grid = jnp.array(pressure_grid)
        self.grid_size = self.pressure_grid.shape[0]
        self.num_points = num_points
        self.t_min, self.t_max = t_bounds
        self.v_min, self.v_max = v_bounds
        self.T_H, self.T_L = T_H, T_L
        self.kappa, self.C_v = kappa, C_v
        self.lambda_smooth = lambda_smooth

        cycle_center = ((self.t_min + self.t_max) / 2.0, (self.v_min + self.v_max) / 2.0)
        r_t = (self.t_max - self.t_min) * 0.1
        r_v = (self.v_max - self.v_min) * 0.1
        theta = jnp.linspace(0, 2 * jnp.pi, num_points, endpoint=False)

        init_path = jnp.stack([
            cycle_center[0] + r_t * jnp.cos(theta),
            cycle_center[1] + r_v * jnp.sin(theta)
        ], axis=-1)
        self.mid_points = nnx.Param(init_path)
        self.opt = nnx.Optimizer(self, optax.adam(learning_rate), wrt=nnx.Param)

    def compute_pressure(self, t, v):
        ti = (t - self.t_min) / (self.t_max - self.t_min) * (self.grid_size - 1)
        vi = (v - self.v_min) / (self.v_max - self.v_min) * (self.grid_size - 1)
        return jnd.map_coordinates(self.pressure_grid, jnp.stack([ti, vi], axis=0), order=1, mode="nearest")

    def _loss(self, mid):
        path = jnp.concatenate([mid, mid[:1]], axis=0)
        T, V = path[:, 0], path[:, 1]
        P = jax.vmap(self.compute_pressure)(T, V)
        dV = jnp.diff(V)
        dT = jnp.diff(T)
        Pm = 0.5 * (P[:-1] + P[1:])
        W = jnp.sum(Pm * dV)

        Tm = 0.5 * (T[:-1] + T[1:])
        dQ = self.C_v * dT + Pm * dV
        smooth_switch = jax.nn.sigmoid(50.0 * dV)
        T_ext = smooth_switch * self.T_H + (1.0 - smooth_switch) * self.T_L

        tau_relax = 0.5
        dt_thermal = jnp.abs(dQ) / (self.kappa * (jnp.abs(T_ext - Tm) + 1e-3))
        dt_quasistatic = tau_relax * jnp.sqrt((dT/Tm)**2 + (dV/V[:-1])**2 + 1e-8)
        dt = dt_thermal + dt_quasistatic
        tau = jnp.sum(dt)

        diss = 0.001 * jnp.sum((dT**2 + dV**2) / dt)
        extended_path = jnp.concatenate([mid[-1:], mid, mid[:1]], axis=0)
        second = extended_path[2:] - 2 * extended_path[1:-1] + extended_path[:-2]
        smooth = jnp.mean(jnp.sum(second**2, -1))

        power = (W - diss) / jnp.maximum(tau, 1e-6)
        total_loss = -(power - self.lambda_smooth * smooth)
        return total_loss, power

    def step(self):
        (loss, pure_power), grads = nnx.value_and_grad(
            lambda m: m._loss(m.mid_points),
            argnums=nnx.DiffState(0, nnx.Param),
            has_aux=True
        )(self)
        self.opt.update(self, grads)
        return loss, pure_power

    def fit(self, steps):
        losses = []
        for i in range(steps):
            l, pure_p = self.step()
            losses.append(float(l))
        return losses

    def path(self, n=1000):
        p = jnp.concatenate([self.mid_points, self.mid_points[:1]], axis=0)
        t = jnp.linspace(0, 1, p.shape[0])
        td = jnp.linspace(0, 1, n)
        return jnp.stack([
            jnp.interp(td, t, p[:, 0]),
            jnp.interp(td, t, p[:, 1])
        ], axis=-1)
"""
