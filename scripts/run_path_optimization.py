# -*- coding: utf-8 -*-
"""
Pipeline script for Finite-Time Thermodynamic path optimization.
Calculates maximum power cycle pathways and prints Onsager dissipation analysis.
"""
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimizers import PowerPathOptimizer

def main():
    print("===============================================================")
    print("  Finite-Time Thermodynamics: Cyclic Pathway Optimization     ")
    print("===============================================================")
    
    GRID_RESOLUTION = 60
    T_MIN, T_MAX = 0.5, 2.5
    V_MIN, V_MAX = 400.0, 1300.0
    
    # Generate/Mock an analytical pressure grid matching ideal gas for cyclic optimization
    t_coords = jnp.linspace(T_MIN, T_MAX, GRID_RESOLUTION)
    v_coords = jnp.linspace(V_MIN, V_MAX, GRID_RESOLUTION)
    TT, VV = jnp.meshgrid(t_coords, v_coords, indexing="ij")
    p_grid = np.array((128 * 1.0 * TT) / VV)
    
    optimizer = PowerPathOptimizer(
        pressure_grid=p_grid,
        t_bounds=(T_MIN, T_MAX),
        v_bounds=(V_MIN, V_MAX),
        num_points=64,
        learning_rate=5e-2,
        T_H=T_MAX,
        T_L=T_MIN,
        kappa=50.0,
        C_v=1.5,
        lambda_smooth=0.5
    )
    
    print("[TRAIN] Fitting trajectory control nodes for 100 epochs...")
    losses = optimizer.fit(steps=100)
    
    dense_path = optimizer.path(n=500)
    print(f"[INFO] Optimized Cyclic Geodesic Path calculated with shape: {dense_path.shape}")
    print(f"[INFO] Trajectory endpoint limits: T [{dense_path[:,0].min():.2f}, {dense_path[:,0].max():.2f}] K")
    print("===============================================================")
    print("Path Optimization complete. Operational cycle bounds locked.")

if __name__ == '__main__':
    main()
