# 🏛️ Architecture, Kernels & Systems Engineering Guide

This document provides a comprehensive technical overview of the computational architecture, execution lifecycle, CUDA device kernels, hardware stack, memory ordering, compiler flags, and dependency audit for the **3D Compressible Supersonic Turbulent Channel Flow DNS Solver**.

---

## 📌 Table of Contents
1. [🔄 High-Level Architectural Data Flowchart](#1--high-level-architectural-data-flowchart)
2. [⚡ CUDA Thread Grid Execution Sequence Diagram](#2--cuda-thread-grid-execution-sequence-diagram)
3. [📋 Detailed 5-Stage Computational Lifecycle](#3--detailed-5-stage-computational-lifecycle)
4. [🛠️ CUDA Device Kernels Technical Reference](#4-️-cuda-device-kernels-technical-reference)
5. [📊 Comprehensive Feature Comparison Matrix](#5--comprehensive-feature-comparison-matrix)
6. [💻 Technology Stack & Target Hardware Architectures](#6--technology-stack--target-hardware-architectures)
7. [⚙️ Compiler Flags & Optimization Matrix](#7-️-compiler-flags--optimization-matrix)
8. [🧠 Memory Ordering, Layout & IEEE 754 Precision Standards](#8--memory-ordering-layout--ieee-754-precision-standards)
9. [📚 Dependencies Audit & Function-by-Function Location Map](#9--dependencies-audit--function-by-function-location-map)

---

## 1. 🔄 High-Level Architectural Data Flowchart

The execution flow of the solver—from raw parameter ingestion through memory allocation, explicit RK2 sub-stepping on GPU, to Tecplot checkpoint export—is illustrated below:

```mermaid
flowchart TD
    subgraph Ingestion ["1. Input Ingestion & In-Memory Marshaling"]
        A["Runtime Config (param.dat)"] --> C["Python CLI Argument Parser (argparse)"]
        B["Raw Initial Grid (input/3D.dat)"] --> D["Fast C-Engine ASCII Parser (pandas / numpy)"]
        C --> E["Domain Sizing (160 x 130 x 130 = 2,704,000 cells)"]
        D --> E
    end

    subgraph Initialization ["2. Grid Mesh & Metric Transformations"]
        E --> F["Wall-Normal Hyperbolic Tangent Grid (z_mesh)"]
        F --> G["Jacobian Metric Tensor Evaluation (deta_dz, d2eta_dz2)"]
        G --> H["Host Float64 Arrays Allocation (NumPy N-D Arrays)"]
        H --> I["Device Memory Allocation & H2D Transfer (cuda.to_device)"]
    end

    subgraph Numerics ["3. Numba CUDA Time Integration (RK2 Loop)"]
        I --> J["Sub-Step 1: Predictor Stage"]
        J --> K1["4th-Order Convective Fluxes Kernel + Peclet Upwind"]
        J --> K2["2nd-Order Viscous Stress Tensor Kernel (tau_ij)"]
        J --> K3["Active Opposition Control (OC) Suction/Blowing Kernel"]
        K1 & K2 & K3 --> L["Intermediate Conserved Array Update (q_stage)"]

        L --> M["Sub-Step 2: Corrector Stage"]
        M --> N1["Convective & Diffusive Flux Evaluation on q_stage"]
        M --> N2["Enforce Wall Boundary Conditions & Physical Clamping"]
        N1 & N2 --> O["Final Conserved State Update (q_new)"]
    end

    subgraph Output ["4. Post-Processing & Persistence"]
        O --> P["Primitive Variable Inversion (rho, u, v, w, E, T, P)"]
        P --> Q["Device-to-Host Asynchronous Transfer (q.copy_to_host)"]
        Q --> R["Tecplot Checkpoint Writer (output_cuda/3D_cuda.dat)"]
        Q --> S["Post-Processing Analysis (Q-Criterion & Fluctuation Fields)"]
    end

    style Ingestion fill:#1f2937,stroke:#374151,color:#fff
    style Initialization fill:#111827,stroke:#374151,color:#fff
    style Numerics fill:#064e3b,stroke:#047857,color:#fff
    style Output fill:#1e3a8a,stroke:#1d4ed8,color:#fff
```

---

## 2. ⚡ CUDA Thread Grid Execution Sequence Diagram

The interaction between the Python host controller and NVIDIA GPU device kernels during each explicit Runge-Kutta time step is detailed below:

```mermaid
sequenceDiagram
    autonumber
    participant Host as Host Python (cuda code.py)
    participant Memory as GPU VRAM (NVIDIA Device)
    participant PrimKernel as Primitive Kernel (@cuda.jit)
    participant ConvKernel as Convective Kernel (@cuda.jit)
    participant ViscKernel as Viscous Kernel (@cuda.jit)
    participant OCKernel as Opposition Control Kernel (@cuda.jit)
    participant RK2Kernel as RK2 Integrator Kernel (@cuda.jit)

    Host->>Memory: cuda.to_device(q_host, grid_metrics)
    Note over Memory: Allocates VRAM for 2,704,000 cells (Conserved & Primitives)

    loop RK2 Time-Stepping Loop (Step 1 to N_steps)
        Host->>PrimKernel: Launch Primitive Variables Kernel <<<blocks, threads>>>
        PrimKernel->>Memory: Invert q -> v = [rho, u, v, w, E, T, P], mu(T), kt(T)

        Host->>ConvKernel: Launch Convective Kernel <<<blocks, threads>>>
        Note over ConvKernel: 4th-Order Stencil + Peclet Upwinding (3D Thread Mapping)
        ConvKernel->>Memory: Store convective RHS (conv_rhs)

        Host->>ViscKernel: Launch Viscous Kernel <<<blocks, threads>>>
        Note over ViscKernel: Stress Tensor tau_ij, Viscous Dissipation, Heat Flux
        ViscKernel->>Memory: Store viscous RHS (visc_rhs)

        Host->>OCKernel: Launch OC Boundary Kernel <<<blocks, threads>>>
        Note over OCKernel: Sensed w(x,y, y+=33) -> w_wall(x,y) = -w_sensor
        OCKernel->>Memory: Apply blowing/suction at z=0 and z=2h

        Host->>RK2Kernel: Launch RK2 Stage Update Kernel <<<blocks, threads>>>
        RK2Kernel->>Memory: Update conserved variables q_stage / q_new
        Memory-->>Host: cuda.synchronize() (Stage Barrier)
    end

    Host->>Memory: q_gpu.copy_to_host()
    Memory-->>Host: Return updated double-precision NumPy array
    Host->>Host: Format and stream Tecplot output checkpoint to disk
```

---

## 3. 📋 Detailed 5-Stage Computational Lifecycle

### Stage 1: Input Ingestion & In-Memory Marshaling
- **Runtime Parameters**: Loaded from [`param.dat`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/param.dat) or CLI flags (`argparse`). Key parameters: Reynolds number ($\text{Re} = 3000$), Mach number ($\text{Ma} = 1.5$), time step ($\Delta t = 1.0\times 10^{-4}$), and Peclet upwinding limits ($\text{Pe}_x=2.0, \text{Pe}_y=1.0, \text{Pe}_z=0.1$).
- **Data Ingestion**: Parsed using Pandas fast C-engine (`pd.read_csv(..., engine='c', sep=r'\s+')`) for large 500+ MB whitespace-delimited initial condition files (`input/3D.dat`).

### Stage 2: Mesh Generation & Coordinate Transformation
- **Domain Geometry**: Physical domain dimensions $L_x = 4\pi$, $L_y = \frac{4}{3}\pi$, $L_z = 2h$.
- **Tanh Wall Normal Stretching**: Stretched grid in $z$ clusters cells near wall boundaries:
  $$\zeta_k = -1 + 2\frac{k - 1}{N_z - 1}, \quad z_k = 1 + \frac{1}{a} \tanh\left( \zeta_k \tanh^{-1}(a) \right), \quad a = 0.98375$$
- **Metric Jacobians**: Analytical coordinate transformation metrics $\frac{d\eta}{dz}$ and $\frac{d^2\eta}{dz^2}$ computed on host and transferred to device.

### Stage 3: Device Allocation & Kernel Execution
- **Zero Host-Device Ping-Pong**: All state arrays (`d_q`, `d_q_stage`, `d_var`, `d_mu`, `d_kt`, `d_rhs_conv`, `d_rhs_visc`) reside continuously in GPU VRAM across the entire time loop.
- **Thread Block Configuration**: 3D thread blocks `(8, 8, 8) = 512 threads` or `(16, 8, 8) = 1,024 threads` map directly to physical $(i, j, k)$ grid coordinates.

### Stage 4: Boundary Closure & Opposition Control
- **Periodic Boundary Conditions**: Explicit wrap-around indexing in $x$ (streamwise) and $y$ (spanwise) directions.
- **Active Opposition Control (OC)**: Senses wall-normal velocity fluctuations at sensor plane $y^+ = 33$ ($k_{\text{sensor}} = 10$) and introduces counteracting vertical velocity at wall:
  $$w_{\text{wall}}(x, y) = -w(x, y, y^+ = 33)$$
- **Physical Safety Clamping**: Density clamped to $\rho \ge 10^{-14}$, temperature clamped to $T \ge 10^{-12}$ to prevent non-physical floating-point overflows.

### Stage 5: Output Checkpointing & Statistical Extraction
- **Tecplot Formatting**: Re-orders 3D GPU primitive variables into Tecplot POINT layout ($k \to j \to i$) and streams out ASCII snapshots (`output_cuda/3D_cuda.dat`).
- **Post-Processing Acceleration**: Vectorized GPU kernels compute $Q$-criterion vortex invariants, fluctuating velocities $q'(x,y,z)$, and Reynolds stress profiles.

---

## 4. 🛠️ CUDA Device Kernels Technical Reference

All GPU kernels are implemented in [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py) using `@cuda.jit` decorators:

### 1. `primary_variables_kernel`
- **Decorator**: `@cuda.jit` (Global Kernel)
- **Signature**: `(q, var, mu, kt, ggM, gamma, mach)`
- **Thread Grid**: 3D block `tpb3 = (8, 8, 8)` covering $(N_x, N_y, N_z)$.
- **Responsibility**: Inverts conserved vector $\mathbf{q} = [\rho, \rho u, \rho v, \rho w, \rho E]^T$ into primitives $\mathbf{v} = [\rho, u, v, w, E, T, P]^T$, computes dynamic viscosity $\mu(T) = [\max(T, 10^{-12})]^{0.7}$, and thermal conductivity $k_t(T) = \mu(T)$.

### 2. `compute_convective_kernel`
- **Decorator**: `@cuda.jit` (Global Kernel)
- **Signature**: `(q, var, conv_rhs, deta_dz, nx, ny, nz, dx, dy, pe_limit)`
- **Thread Grid**: 3D block covering all $2,704,000$ cells.
- **Responsibility**: Computes 4th-order central finite differences for convective fluxes $\frac{\partial F}{\partial x} + \frac{\partial G}{\partial y} + \frac{\partial H}{\partial z}$. When local cell Peclet number $\text{Pe}_i = \frac{\rho |u| \Delta x}{\mu}$ exceeds threshold `pe_limit`, smoothly blends 4th-order upwind artificial dissipation.

### 3. `compute_viscous_kernel`
- **Decorator**: `@cuda.jit` (Global Kernel)
- **Signature**: `(var, visc_rhs, deta_dz, d2eta_dz2, nx, ny, nz, dx, dy, Re, Ma, Pr)`
- **Thread Grid**: 3D block covering all domain interior cells.
- **Responsibility**: Evaluates viscous stress tensor $\tau_{ij} = \mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} - \frac{2}{3}\delta_{ij}\frac{\partial u_k}{\partial x_k} \right)$, viscous dissipation $\Phi = \tau_{ij}\frac{\partial u_i}{\partial x_j}$, and heat flux vectors $q_k = -\frac{k_t}{(\gamma - 1)\text{Ma}^2 \text{Pr}\text{Re}} \frac{\partial T}{\partial x_k}$.

### 4. `update_rk2_kernel`
- **Decorator**: `@cuda.jit` (Global Kernel)
- **Signature**: `(q_old, rhs, q_new, dt, rk_stage, nx, ny, nz)`
- **Thread Grid**: 3D block updating 5 conserved variables per cell.
- **Responsibility**: Executes explicit two-stage Runge-Kutta time advancement:
  - Stage 1 (Predictor): $\mathbf{q}^{(1)} = \mathbf{q}^{(n)} + \Delta t \, \mathbf{R}(\mathbf{q}^{(n)})$
  - Stage 2 (Corrector): $\mathbf{q}^{(n+1)} = \frac{1}{2}\mathbf{q}^{(n)} + \frac{1}{2}\mathbf{q}^{(1)} + \frac{1}{2}\Delta t \, \mathbf{R}(\mathbf{q}^{(1)})$

### 5. `apply_bc_kernel` / `opposition_and_wall_bc_kernel`
- **Decorator**: `@cuda.jit` (Global Kernel)
- **Signature**: `(q, var, nx, ny, nz, enable_oc, sensor_k)`
- **Thread Grid**: 2D grid covering wall boundaries $(i, j)$ at $k = 0$ (bottom wall) and $k = N_z - 1$ (top wall).
- **Responsibility**: Implements isothermal no-slip conditions ($u = v = 0$, $T = T_w$), reconstructs wall density from wall-normal pressure gradient $\left(\frac{\partial P}{\partial z}\right)_{\text{wall}} = 0$, and applies active opposition blowing/suction $w_{\text{wall}}(i, j) = -w(i, j, k_{\text{sensor}})$.

---

## 5. 📊 Comprehensive Feature Comparison Matrix

| Feature / Capability | Reference Fortran 90 MPI Solver | Python Numba CUDA GPU Solver | Advantage / Notes |
| :--- | :--- | :--- | :--- |
| **Target Hardware** | 4-Core x86_64 CPU Cluster | 1 NVIDIA GeForce / Tesla GPU | Massively parallel SIMT execution |
| **Grid Resolution** | $160 \times 130 \times 130$ ($2,704,000$ cells) | $160 \times 130 \times 130$ ($2,704,000$ cells) | Identical mesh resolution |
| **Mesh Generation** | Analytical $\tanh$ wall clustering | Analytical $\tanh$ wall clustering | Identical metric Jacobians ($\mathcal{J}$) |
| **Convective Stencil** | 4th-Order Central + Adaptive Upwind | 4th-Order Central + Adaptive Upwind | Identical Peclet dissipation |
| **Viscous Stencil** | Full 3D $\tau_{ij}$ formulation | Full 3D $\tau_{ij}$ formulation | Exact Navier-Stokes stress tensor |
| **Time Stepping** | Explicit 2-Stage Runge-Kutta (RK2) | Explicit 2-Stage Runge-Kutta (RK2) | $\Delta t = 1.0\times 10^{-4}$ |
| **Active Control** | $w_{\text{wall}} = -w(y^+ = 33)$ | $w_{\text{wall}} = -w(y^+ = 33)$ | Exact opposition control logic |
| **Domain Partitioning** | 3D Cartesian MPI ($1\times 1\times 4$) | Single Unified In-VRAM Tensor | **Zero network/halo latency** |
| **Step Execution Time** | 0.4753 s / step | **0.3102 s / step** (GTX 1650) | **1.532× GPU Solver Speedup** |
| **$Q$-Criterion Time** | 35.948 s | **7.977 s** | **4.506× Post-Processing Speedup** |
| **Fluctuation Field Time** | 63.807 s | **7.148 s** | **8.926× Post-Processing Speedup** |

---

## 6. 💻 Technology Stack & Target Hardware Architectures

```
                      +---------------------------------------+
                      |         Python 3 Runtime              |
                      +-------------------+-------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
        v                                                                   v
+-------------------------------+                         +-------------------------------+
|     Numba CUDA Subsystem      |                         |    GFortran Toolchain         |
|  - Target: NVIDIA GPU         |                         |  - Target: Multi-Core CPU     |
|  - NVVM / PTX JIT Compilation |                         |  - OpenMP + MPI Parallelism   |
|  - IEEE 754 Float64 Math      |                         |  - Native Machine Code        |
+---------------+---------------+                         +---------------+---------------+
                |                                                                 |
                v                                                                 v
+-------------------------------+                         +-------------------------------+
|  1x NVIDIA GPU (Pascal+)      |                         |  4x CPU Core Parallel Cluster |
|  - 896 to 2,560+ CUDA Cores   |                         |  - 3D Domain Decomposition    |
|  - High-Bandwidth VRAM        |                         |  - MPI Halo Communication     |
|  - 128.0 to 300+ GB/s         |                         |  - ~35.2 GB/s DDR4 RAM        |
+-------------------------------+                         +-------------------------------+
```

### Hardware Specification Comparison

| System Parameter | Reference Fortran MPI System | Python Numba CUDA GPU System |
| :--- | :--- | :--- |
| **Processor Model** | 4-Core x86_64 CPU Host | 1 NVIDIA GPU (GTX 1650 / Tesla T4) |
| **Compute Units** | 4 CPU Cores (4 Threads) | 896 to 2,560 CUDA Cores |
| **Memory Architecture** | Host DDR4 System RAM | High-Bandwidth GDDR6 VRAM |
| **Memory Bandwidth** | ~35.2 GB/s | **128.0 GB/s to 320.0 GB/s** (3.6×–9.1× faster) |
| **Parallel Paradigm** | MPI Message Passing (`MPI_Sendrecv`) | Massively Parallel CUDA Thread Grids |
| **Halo Overhead** | Inter-process network / memory copy | **0.00 ms** (Direct thread memory access via periodic wrap) |

---

## 7. ⚙️ Compiler Flags & Optimization Matrix

### 7.1 Native GFortran Compiler Flags (`gfortran`)

| Flag / Parameter | Category | Purpose & Description |
| :--- | :--- | :--- |
| `-O3` | Optimization | Enables aggressive compiler optimizations, vectorization, and loop unrolling. |
| `-fopenmp` | Parallelism | Enables OpenMP multi-threading pragmas for shared-memory CPU loops. |
| `-fdefault-real-8` | Precision | Promotes standard 32-bit `real` declarations to double-precision 64-bit IEEE 754 floats (`real(8)`). |
| `-fdefault-double-8` | Precision | Forces all double precision literals and variables to 64-bit precision. |
| `-cpp` | Preprocessor | Enables the C preprocessor for conditional compilation directives (`#ifdef`). |
| `-Wall -Wextra` | Diagnostics | Enables compiler warnings to catch potential array out-of-bounds errors. |
| `-static-libgcc -static-libgfortran` | Linking | Statically links GCC runtime libraries to ensure portable standalone execution on Windows MinGW. |

### 7.2 Numba CUDA JIT Compilation Targets (`numba.cuda`)

| Configuration Parameter | Value / Target | Technical Rationale |
| :--- | :--- | :--- |
| **Compiler Target** | PTX / NVVM Bitcode | Dynamic runtime translation of Python AST into native NVIDIA PTX assembly. |
| **Precision Standard** | `float64` (Double) | Matches Fortran `real(kind=8)` numerical accuracy to $0.00\text{e}+00$ relative float error. |
| **CUDA Home Path** | Auto-detected | Resolves `nvidia/cuda_nvcc` site-package DLL paths automatically. |
| **Libdevice Path** | `libdevice.10.bc` | Hardware-accelerated math intrinsics (`sqrt`, `exp`, `pow`, `tanh`). |
| **Thread Block Dimensions** | `(8, 8, 8)` = 512 threads | Optimal CUDA warp occupancy for 3D domain grid processing. |
| **Grid Block Dimensions** | `(20, 17, 17)` = 5,780 blocks | Completely covers the $160 \times 130 \times 130$ physical computational domain. |

---

## 8. 🧠 Memory Ordering, Layout & IEEE 754 Precision Standards

### 8.1 Index Mapping Matrix

Fortran uses 1-based, column-major array ordering `(z, y, x)`, while Python/CUDA uses 0-based, row-major C-contiguous array ordering `(nz, ny, nx)`:

| Dimensional Index | Fortran 90 Representation | Python Numba CUDA Representation | Physical Meaning |
| :--- | :--- | :--- | :--- |
| **Streamwise ($x$)** | `1` to `NX` (`160`) | `0` to `nx-1` (`0` to `159`) | Flow direction ($L_x = 4\pi$) |
| **Spanwise ($y$)** | `1` to `NY` (`130`) | `0` to `ny-1` (`0` to `129`) | Cross-stream direction ($L_y = \frac{4}{3}\pi$) |
| **Wall-Normal ($z$)** | `1` to `NZ` (`130`) | `0` to `nz-1` (`0` to `129`) | Non-uniform tanh wall direction ($L_z = 2h$) |

### 8.2 IEEE 754 Floating-Point Precision Compliance
- Both solvers enforce 64-bit IEEE 754 double precision (`float64` in Python, `real(kind=8)` in Fortran).
- Verification proves that mean conserved density $\bar{\rho} = 1.054516$ matches between Fortran and CUDA implementations to **$0.00\text{e}+00$ relative float error**.

---

## 9. 📚 Dependencies Audit & Function-by-Function Location Map

### 9.1 Taxonomic Dependency Mindmap

```mermaid
mindmap
  root((CFD DNS Solver Stack))
    GPU Acceleration Layer
      Numba CUDA
        numba.cuda.jit
        numba.cuda.grid
        numba.cuda.to_device
        numba.cuda.synchronize
      NVIDIA CUDA Runtime
        NVVM Compiler
        libdevice.10.bc
        NVRTC / NVCC cu12
    Scientific & Data Processing
      NumPy (numpy<2)
        64-bit Array allocations
        Hyperbolic trig functions
      Pandas
        Fast C-Engine ASCII Ingestion
      SciPy
        scipy.stats.pearsonr
      Matplotlib
        Headless Agg Plotting
    Legacy Fortran Engine
      GFortran 16.1.0
        OpenMP multi-threading
        ISO_FORTRAN_ENV
      MPI Subsystem
        MPI_Cart_create
        MPI_Sendrecv
        MPI_Allreduce
    System & Orchestration
      Standard Library
        argparse
        dataclasses
        subprocess
        time.perf_counter
```

### 9.2 Function-Level Location Map

| Library | Function / Symbol | Target File Location | Core Responsibility |
| :--- | :--- | :--- | :--- |
| `numba.cuda` | `@cuda.jit(device=True)` | [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L200-L350) | Inlined GPU device stencil functions (derivative operators). |
| `numba.cuda` | `@cuda.jit` | [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L360-L800) | Global GPU kernels (`primary_variables_kernel`, `compute_convective_kernel`, `compute_viscous_kernel`, `update_rk2_kernel`). |
| `numba.cuda` | `cuda.grid(3)` | [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L370) | Computes global 3D thread index `(i, j, k)`. |
| `numba.cuda` | `cuda.to_device(...)` | [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L1030-L1045) | Transfers host initial conditions and metrics to device VRAM. |
| `numba.cuda` | `cuda.synchronize()` | [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L1120) | Enforces execution barriers between RK2 predictor/corrector sub-steps. |
| `numpy` | `np.zeros(...)`, `np.empty(...)` | [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L920), [`python/process_input_files.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/process_input_files.py#L120) | Allocates 64-bit zero/empty host arrays. |
| `numpy` | `np.tanh(...)`, `np.linspace(...)` | [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L880) | Generates wall-normal hyperbolic tangent mesh coordinates. |
| `numpy` | `np.trapz(...)` | [`python/process_input_files.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/process_input_files.py#L180) | Trapezoidal integration for bulk density $\rho_b$ and bulk velocity $U_b$. |
| `pandas` | `pd.read_csv(..., engine='c')` | [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py#L850), [`python/process_input_files.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/process_input_files.py#L95) | Fast C-engine ASCII parsing for 500+ MB input datasets. |
| `scipy.stats` | `pearsonr(a, b)` | [`python/compare_fortran_cuda.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/compare_fortran_cuda.py#L175) | Computes Pearson correlation coefficient $r$ for physical validation. |
| `subprocess` | `subprocess.run(...)` | [`python/recompile_and_run_fortran.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/recompile_and_run_fortran.py#L80) | Executes `gfortran` compilation and runs native Fortran binaries. |
| `time` | `time.perf_counter()` | [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py#L110) | Captures microsecond-accurate wall-clock step timings. |
| `mpi` / `mpif.h` | `MPI_Init`, `MPI_Sendrecv` | [`codes1/mod_my_mpi.f90`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/codes1/mod_my_mpi.f90#L30-L160) | Initializes MPI environment and exchanges halo boundary planes. |
