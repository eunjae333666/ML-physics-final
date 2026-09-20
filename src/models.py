# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

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
