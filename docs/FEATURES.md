# ⚡ API Feature Matrix & Kernel Reference

This document provides a comprehensive high-level feature matrix and API kernel reference for both the **Python Numba CUDA GPU Solver** and the reference **Fortran 90 MPI Solver**.

---

## 📌 1. High-Level Feature Comparison Matrix

| Feature / Capability | Reference Fortran 90 MPI Solver | Python Numba CUDA GPU Solver | Advantage / Notes |
| :--- | :--- | :--- | :--- |
| **Execution Hardware** | 4-Core CPU Cluster | 1 NVIDIA GeForce GPU | Massively parallel CUDA execution |
| **Grid Resolution** | $160 \times 130 \times 130$ ($2,704,000$ cells) | $160 \times 130 \times 130$ ($2,704,000$ cells) | Full mesh resolution match |
| **Wall Mesh Stretching** | Tanh wall-normal formula | Tanh wall-normal formula | Identical grid metric Jacobians ($\mathcal{J}$) |
| **Convective Discretization** | 4th-Order Central + Upwind | 4th-Order Central + Upwind | Peclet limit upwinding ($\text{Pe}_x, \text{Pe}_y, \text{Pe}_z$) |
| **Viscous Stress Tensor** | Full 3D $\tau_{ij}$ formulation | Full 3D $\tau_{ij}$ formulation | Exact Navier-Stokes viscous dissipation |
| **Time Stepping Engine** | Explicit 2-Stage Runge-Kutta (RK2) | Explicit 2-Stage Runge-Kutta (RK2) | $\Delta t = 1.0\times 10^{-4}$ step size |
| **Active Opposition Control** | $w_{\text{wall}}(x,y) = -w(x,y,y^+=33)$ | $w_{\text{wall}}(x,y) = -w(x,y,y^+=33)$ | Drag reduction suction/blowing |
| **Domain Partitioning** | 3D Cartesian MPI ($1\times 1\times 4$) | Single In-VRAM GPU Domain | Zero inter-process network overhead |
| **Data Checkpointing** | Tecplot ASCII & Binary | Tecplot ASCII Format | Portable Tecplot dataset export |
| **Average Step Execution** | 0.4753 s / step | **0.3102 s / step** | **1.532× GPU Solver Speedup** |
| **Q-Criterion Calculation** | 35.948 s | **7.977 s** | **4.506× Post-Processing Speedup** |
| **Fluctuation Field Extraction**| 63.807 s | **7.148 s** | **8.926× Post-Processing Speedup** |

---

## 📌 2. Python CUDA Device Kernels Reference ([`cuda code.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/cuda%20code.py))

### 1. `compute_convective_kernel(q, v, conv_rhs, deta_dz, nx, ny, nz, dx, dy, pe_limit)`
- **Type**: `@cuda.jit` Global Kernel
- **Thread Grid**: 3D CUDA block grid `(tx, ty, tz)` mapping to mesh cell `(i, j, k)`.
- **Function**: Evaluates 4th-order central finite difference convective fluxes $\frac{\partial F}{\partial x} + \frac{\partial G}{\partial y} + \frac{\partial H}{\partial z}$ with adaptive Peclet upwinding when cell Peclet number exceeds `pe_limit`.

### 2. `compute_viscous_kernel(v, visc_rhs, deta_dz, d2eta_dz2, nx, ny, nz, dx, dy, Re, Ma, Pr)`
- **Type**: `@cuda.jit` Global Kernel
- **Thread Grid**: 3D CUDA block grid `(tx, ty, tz)` mapping to mesh cell `(i, j, k)`.
- **Function**: Computes full 3D compressible viscous stress tensor $\tau_{ij}$, strain rate tensor $S_{ij}$, viscous dissipation rates $\Phi$, and thermal heat flux terms $\frac{\partial q_k}{\partial x_k}$.

### 3. `update_rk2_kernel(q_old, rhs, q_new, dt, rk_stage, nx, ny, nz)`
- **Type**: `@cuda.jit` Global Kernel
- **Thread Grid**: 3D CUDA block grid `(tx, ty, tz)`.
- **Function**: Performs explicit 2nd-order Runge-Kutta time step updates:
  $$\text{Stage 1 (Predictor)}: \mathbf{q}^{(1)} = \mathbf{q}^{(n)} + \Delta t \, \mathbf{R}(\mathbf{q}^{(n)})$$
  $$\text{Stage 2 (Corrector)}: \mathbf{q}^{(n+1)} = \frac{1}{2}\mathbf{q}^{(n)} + \frac{1}{2}\mathbf{q}^{(1)} + \frac{1}{2}\Delta t \, \mathbf{R}(\mathbf{q}^{(1)})$$

### 4. `apply_bc_kernel(q, v, nx, ny, nz, enable_oc, sensor_k)`
- **Type**: `@cuda.jit` Global Kernel
- **Thread Grid**: 2D CUDA thread grid `(tx, ty)` across channel wall boundaries ($z=0$ and $z=2h$).
- **Function**: Implements no-slip wall conditions ($u=v=0$), wall temperature conditions, and Active Opposition Control ($\text{OC}$) suction/blowing $w_{\text{wall}}(i, j) = -w(i, j, k_{\text{sensor}})$.

---

## 📌 3. Python Utility API Reference (`python/`)

### 1. `InputDatasetProcessor` ([`python/process_input_files.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/process_input_files.py))
- **`process_all_datasets()`**: Batch processes all 5 raw datasets in `input/`.
- **`compute_bulk_quantities(rho, u, z)`**: Calculates bulk density $\rho_b = \frac{1}{2h}\int_0^{2h} \rho \, dz$ and bulk velocity $U_b = \frac{1}{\rho_b 2h}\int_0^{2h} \rho u \, dz$.
- **`compute_skin_friction(tau_w, rho_b, U_b)`**: Evaluates skin friction coefficient $C_f = \frac{2\tau_w}{\rho_b U_b^2}$.

### 2. `FortranCudaComparator` ([`python/compare_fortran_cuda.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/compare_fortran_cuda.py))
- **`evaluate_field_accuracy(var_fortran, var_cuda)`**: Computes Mean Absolute Error ($L_1$), Relative $L_2$ error norm, Maximum Absolute Error ($L_\infty$), and Pearson correlation coefficient ($r$).
- **`export_markdown_report(report_path)`**: Exports formatted quantitative comparison table to markdown.

### 3. `FortranRecompiler` ([`python/recompile_and_run_fortran.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/recompile_and_run_fortran.py))
- **`clean_and_recompile()`**: Removes old executables and compiles Fortran source files with `gfortran -O3 -fopenmp`.
- **`measure_execution_time(binary_path)`**: Executes binary and captures sub-millisecond execution time.

### 4. `BenchmarkRunner` ([`python/benchmark_solvers.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/benchmark_solvers.py))
- **`run_cuda_benchmark(steps)`**: Executes live Numba CUDA solver for target step count using `time.perf_counter()`.
- **`extrapolate_timings(step_scales)`**: Extrapolates step times up to 1,000 steps and calculates GPU speedup factors.
