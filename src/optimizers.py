# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.ndimage as jnd
from flax import nnx
import optax
import numpy as np
import os

class FreeEnergyGridSolver:
    def __init__(self, estimator_obj=None, grid_size: int = 50):
        self.estimator = estimator_obj
        self.grid_size = grid_size

    def generate_and_filter_manifold(self, t_bounds: tuple, v_bounds: tuple, num_samples: int = 32, sigma: float = 1.0):
        if self.estimator is None:
            raise ValueError("New manifold requires an estimator instance.")
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
