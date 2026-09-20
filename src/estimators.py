# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from .models import ConditionalNSF
from .potentials import LJPotential

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
