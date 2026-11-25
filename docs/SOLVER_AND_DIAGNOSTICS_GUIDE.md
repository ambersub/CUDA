# 🎮 Solver Execution, Checkpointing & Diagnostics Guide

This document provides a comprehensive operational guide for running the **3D Compressible Supersonic Turbulent Channel Flow DNS Solver**, managing its **dual-level checkpointing system**, streaming **high-frequency turbulence diagnostics**, and orchestrating simulations with [`python/cudacomplete.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/cudacomplete.py).

---

## 📌 Table of Contents
1. [🏗️ Dual-Level Checkpointing Architecture](#1-️-dual-level-checkpointing-architecture)
2. [💾 Level 1: Full 3D Volumetric State Checkpoints](#2--level-1-full-3d-volumetric-state-checkpoints)
   - [Memory Layout vs. Tecplot Format](#memory-layout-vs-tecplot-format)
   - [Header Metadata & Step Resumption](#header-metadata--step-resumption)
   - [Source Code Implementation Trace](#source-code-implementation-trace)
3. [📈 Level 2: Real-Time Diagnostic State Persistence](#3--level-2-real-time-diagnostic-state-persistence)
   - [On-the-Fly GPU Reductions](#on-the-fly-gpu-reductions)
   - [Core Diagnostic Streams (`KE.dat`, `rhobulk.dat`, etc.)](#core-diagnostic-streams-kedat-rhobulkdat-etc)
   - [Append-Only Safe Streaming & Fault Tolerance](#append-only-safe-streaming--fault-tolerance)
4. [⚡ Unified Pipeline Orchestration with `cudacomplete.py`](#4--unified-pipeline-orchestration-with-cudacompletepy)
   - [Running Simulations with Periodic Checkpoints](#running-simulations-with-periodic-checkpoints)
   - [Seamless Resumption from Checkpoints](#seamless-resumption-from-checkpoints)
   - [Batch Processing Saved Snapshots](#batch-processing-saved-snapshots)
5. [📖 Hands-On Step-by-Step Tutorial (Modules 1 to 6)](#5--hands-on-step-by-step-tutorial-modules-1-to-6)
   - [Module 1: CUDA Environment Verification](#module-1-cuda-environment-verification)
   - [Module 2: Configuring Physical Parameters (`param.dat`)](#module-2-configuring-physical-parameters-paramdat)
   - [Module 3: Executing Main CUDA Solver (`cuda code.py`)](#module-3-executing-main-cuda-solver-cuda-codepy)
   - [Module 4: Processing Raw Input Datasets (`process_input_files.py`)](#module-4-processing-raw-input-datasets-process_input_filespy)
   - [Module 5: Native Fortran Compilation & Benchmarks](#module-5-native-fortran-compilation--benchmarks)
   - [Module 6: Generating Validation Plots against CKM Data](#module-6-generating-validation-plots-against-ckm-data)

---

## 1. 🏗️ Dual-Level Checkpointing Architecture

Direct Numerical Simulation (DNS) on a $160 \times 130 \times 130$ mesh ($2,704,000$ cells) generates massive data volumes. Each uncompressed 3D ASCII Tecplot checkpoint is approximately **503 MB**. Saving the full state at every time step ($\Delta t = 10^{-4}$) would saturate disk I/O bandwidth and consume terabytes of storage within hours.

To solve this, the solver decouples state persistence into a **two-level checkpointing hierarchy**:

```
                                    GPU Memory (VRAM)
                          Active State: d_q, d_var, d_mu, d_kt
                                            │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
      Level 1: 3D State Checkpoints                 Level 2: Continuous Diagnostics
      (Low Frequency: Periodic / Final)             (High Frequency / Every Step)
                    │                                             │
      benchmark_solvers.py / cudacomplete.py         compute_diagnostics_cuda.py
                    │                                             │
      • Periodic saves: 3D_step100.dat             • Volume & Plane Reductions (GPU)
      • Final state: 3D_cuda.dat                   • Appends to 1D time series:
      • Fluctuation field: fluc.dat                  - KE.dat (Kinetic Energy)
      • Full restart capability                      - rhobulk.dat (Bulk Density)
                                                     - Ubulk.dat (Bulk Velocity)
                                                     - massflow.dat (Mass Flux)
                                                     - tauWall.dat (Wall Shear)
```

1. **Level 1 — Full 3D Volumetric State Checkpoints**:
   - Infrequent, complete snapshots of the entire flow domain ($\sim 503\text{ MB}$ each).
   - Used for full simulation restarts, 3D vortex visualization ($Q$-criterion), and cross-solver parity verification.
2. **Level 2 — Continuous Time-Series Diagnostic Checkpoints**:
   - High-frequency, compact scalar/plane reductions ($\sim \text{few KB}$).
   - Streamed into append-only time-series files (`KE.dat`, `rhobulk.dat`, `Ubulk.dat`, `massflow.dat`, `tauWall.dat`) to capture high-frequency turbulent fluctuations without disk overhead.

---

## 2. 💾 Level 1: Full 3D Volumetric State Checkpoints

### Memory Layout vs. Tecplot Format

* **In GPU Memory (VRAM)**:
  Primitive state array `d_var` has shape `(nx, ny, nz, 7)` in C-contiguous memory:
  $$\mathbf{v} = [\rho, u, v, w, E, T, P]$$
  where $(i, j, k)$ correspond to streamwise $x$, spanwise $y$, and wall-normal $z$.

* **On-Disk File Format (Tecplot POINT Layout)**:
  To maintain 100% binary/ASCII parity with legacy Fortran outputs, checkpoints use the **Tecplot POINT format** with $k \to j \to i$ index ordering:
  ```text
  TITLE ="        2268.1860000000  18963010"
  Variables = "x","y","z","Rho","u","v","w","e","T","p"
  ZONE k=160,j=130,i=130,DATAPACKING=POINT
   0.000000 0.000000 0.000000 1.054516e+00 8.344800e-01 ...
  ```

* **Memory Transformation (`save_solution`)**:
  When a checkpoint is triggered:
  1. `solver.download_primitives()` transfers the 7 primitive fields from GPU VRAM to host RAM (`numpy.ndarray`).
  2. The array is transposed and reassembled into shape `(nz, ny, nx, 10)`:
     ```python
     out_arr = np.empty((nz, ny, nx, 10), dtype=np.float64)
     for k in range(nz):
         out_arr[k, :, :, 0] = self.x1_h[None, :]  # x coordinate
         out_arr[k, :, :, 1] = self.x2_h[:, None]  # y coordinate
         out_arr[k, :, :, 2] = self.x3_h[k]        # z coordinate
         for p in range(7):
             out_arr[k, :, :, 3 + p] = var_h[:, :, k, p].T
     ```
  3. `np.savetxt` streams the flattened array to disk with scientific precision (`%22.15e`).

---

### Header Metadata & Step Resumption

The first line of every Tecplot checkpoint stores the exact simulation state:
```text
TITLE ="        <time_seconds>  <iteration_step>"
```
When resuming (`--resume`):
1. The solver reads line 1: `TITLE =" 2268.1860000000 18963010"`.
2. Regular expressions extract `time = 2268.186` and `step = 18963010`.
3. The solver updates internal clocks (`self.time = time`, `self.step = step`).
4. Time stepping continues from `step + 1` up to `step + requested_steps`.

---

### Source Code Implementation Trace

- **Checkpoint Writer**: `CUDASolver.save_solution()` in [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L1207-L1244)
- **State Ingestion & Parser**: `CUDASolver.read_combined_input()` in [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L295-L345)
- **Periodic Interval Controller**: `run_benchmark()` in [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py#L461-L518) and `run_cudacomplete()` in [`python/cudacomplete.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/cudacomplete.py#L65-L120)

---

## 3. 📈 Level 2: Real-Time Diagnostic State Persistence

Implemented in [`python/compute_diagnostics_cuda.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/compute_diagnostics_cuda.py), Level 2 provides GPU-accelerated spatial reductions that stream directly into append-only time series files.

### Core Diagnostic Streams

| Diagnostic Stream File | Physical Quantity | Mathematical Reduction |
| :--- | :--- | :--- |
| **`output_cuda/rhobulk.dat`** | Bulk Density ($\rho_b$) | $\rho_b(t) = \frac{1}{V}\int_V \rho(\mathbf{x}, t) \, dV$ |
| **`output_cuda/Ubulk.dat`** | Bulk Streamwise Velocity ($U_b$) | $U_b(t) = \frac{1}{\rho_b V}\int_V \rho u \, dV$ |
| **`output_cuda/massflow.dat`** | Total Mass Flow Rate ($\dot{m}$) | $\dot{m}(t) = \int_0^{L_y}\int_0^{2h} \rho u \, dz \, dy$ |
| **`output_cuda/tauWall.dat`** | Wall Shear Stress ($\tau_w$) | $\tau_w(t) = \left. \mu_w \frac{\partial \bar{u}}{\partial z}\right\|_{z=0}$ |
| **`output_cuda/KE.dat`** | Total Kinetic Energy ($E_k$) | $E_k(t) = \frac{1}{2}\int_V \rho (u^2 + v^2 + w^2) \, dV$ |

### Append-Only Safe Streaming & Fault Tolerance
Each record is appended as `(time, step, scalar_value)`:
```text
 2.268186000000000e+03   18963010   1.054516248921443e+00
```
- Because files are opened in append mode (`'a'`), aborted runs do not corrupt existing historical diagnostics.
- Resuming a run simply appends subsequent time steps to the same time-series history without gaps.

---

## 4. ⚡ Unified Pipeline Orchestration with `cudacomplete.py`

[`python/cudacomplete.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/cudacomplete.py) combines solver execution, periodic checkpointing, and GPU diagnostic extraction into a single unified CLI interface.

### Running Simulations with Periodic Checkpoints
From the repository root:
```powershell
python python/cudacomplete.py --steps 500 --save-interval 100
```
- Writes periodic checkpoints: `output_cuda/3D_step100.dat`, `output_cuda/3D_step200.dat`, ...
- Writes the final state to `output_cuda/3D.dat`.
- Automatically streams diagnostics to `output_cuda/*.dat` and produces `output_cuda/benchmark_results.json`.

### Seamless Resumption from Checkpoints
To continue the simulation from the last saved state:
```powershell
python python/cudacomplete.py --steps 500 --save-interval 100 --resume
```
*The solver reads the iteration step and physical time directly from the checkpoint header and continues seamlessly.*

### Batch Processing Saved Snapshots
Compute diagnostics for a single saved snapshot:
```powershell
python python/cudacomplete.py --diagnostics-input output_cuda/3D.dat
```

Process a directory of periodic checkpoints in batch:
```powershell
python python/cudacomplete.py --diagnostics-dir output_cuda --pattern "3D_step*.dat"
```
*(Add `--no-gpu` to fall back to the vectorized CPU NumPy routines if no GPU is available during post-processing).*

---

## 5. 📖 Hands-On Step-by-Step Tutorial (Modules 1 to 6)

### Module 1: CUDA Environment Verification
Confirm your Python environment recognizes the NVIDIA GPU and Numba CUDA JIT compiler:
```python
import numba.cuda

if numba.cuda.is_available():
    dev = numba.cuda.get_current_device()
    print(f"CUDA Device: {dev.name.decode('utf-8')}")
    print(f"Compute Capability: {dev.compute_capability}")
    print(f"Multiprocessors: {dev.MULTIPROCESSOR_COUNT}")
else:
    print("ERROR: CUDA GPU not detected.")
```

---

### Module 2: Configuring Physical Parameters (`param.dat`)
Inspect and adjust runtime physical parameters in [`param.dat`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/param.dat):
```text
160   130   130       ! NX, NY, NZ (Grid dimensions)
3000.0                ! RE (Reynolds number)
1.5                   ! MACH (Mach number)
1.0e-4                ! DT (Time step size)
100                   ! NWRT (Checkpoint write interval)
```

---

### Module 3: Executing Main CUDA Solver (`cuda code.py`)
1. **Standard 100-step simulation**:
   ```powershell
   python "cuda code.py" --input input/3D.dat --steps 100 --output output_cuda/3D_cuda.dat
   ```

2. **Synthetic Demo Simulation (Fast test without loading 500 MB initial dataset)**:
   ```powershell
   python "cuda code.py" --demo --steps 20 --output output_cuda/demo.dat
   ```

---

### Module 4: Processing Raw Input Datasets (`process_input_files.py`)
Process all 5 raw initial condition files in `input/`:
```powershell
python python/process_input_files.py --input-dir input --output-dir output_processed
```
*Generates summary statistics JSON (`input_processing_summary.json`) and 1D profile CSVs.*

---

### Module 5: Native Fortran Compilation & Benchmarks
Compile Fortran source files natively with `gfortran -O3 -fopenmp` and measure CPU vs GPU execution:
```powershell
# Recompile and measure native Fortran execution:
python python/recompile_and_run_fortran.py

# Run live side-by-side benchmark:
python python/benchmark_solvers.py --steps 50
```

---

### Module 6: Generating Validation Plots against CKM Data
Generate the 8 canonical validation figures comparing the CUDA solver against benchmark DNS data:
```powershell
python python/plot_validation.py --cuda-dir output_cuda --output-dir plots/validation_plots
```
*Plots generated:*
- `pmean_valid.png`: Mean pressure profile vs. CKM DNS
- `tmean_valid.png`: Mean temperature profile vs. CKM DNS
- `umean_valid.png`: Mean streamwise velocity profile vs. CKM DNS
- `urms_valid.png`, `vrms_valid.png`, `wrms_valid.png`: Velocity fluctuation RMS profiles
- `tauWall_valid.png`: Wall shear stress time history
- `cf_valid.png`: Skin friction coefficient comparison
