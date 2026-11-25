# 📁 Complete Codebase Reference & Fortran Manual

This document provides an exhaustive reference of the entire repository structure, file specifications, exposed symbols, function-level dependencies, and the complete technical manual for the legacy Fortran 90 solver subsystem.

---

## 📌 Table of Contents
1. [🌳 Master Repository Directory Map](#1-️-master-repository-directory-map)
2. [🐍 Python Subsystem & Main Solver Specifications](#2--python-subsystem--main-solver-specifications)
3. [⚙️ Setup, Diagnostics & Configuration Files](#3-️-setup-diagnostics--configuration-files)
4. [📂 Data Files & Output Directories](#4--data-files--output-directories)
5. [📜 Fortran 90 Subsystem & Legacy Modules Manual](#5--fortran-90-subsystem--legacy-modules-manual)
   - [Core Simulation Modules (`codes1/`)](#core-simulation-modules-codes1)
   - [Post-Processing & Statistical Tools (`codes/`)](#post-processing--statistical-tools-codes)
   - [GNUplot Visualization Scripts (`.txt`)](#gnuplot-visualization-scripts-txt)
   - [Sampling Control Files (`.dat`)](#sampling-control-files-dat)
   - [Build System (`makefile_3D`)](#build-system-makefile_3d)
6. [📖 Data Dictionary: Conserved, Primitive & Metric Variables](#6--data-dictionary-conserved-primitive--metric-variables)

---

## 1. 🌳 Master Repository Directory Map

```
.
├── docs/                                  # Central Technical Documentation Suite (7 Core Pillars)
│   ├── README.md                          # Master Landing Page & Documentation Portal
│   ├── SETUP_AND_CLOUD_GUIDE.md           # Local & Colab Setup, GPU Environments, Troubleshooting
│   ├── ARCHITECTURE_AND_KERNELS.md        # Architecture, Flowcharts, JIT Kernels, Tech Stack
│   ├── SOLVER_AND_DIAGNOSTICS_GUIDE.md     # Solver Operations, Checkpointing, cudacomplete.py, Tutorial
│   ├── BENCHMARK_AND_VALIDATION_REPORT.md # CPU vs GPU Benchmarks, Error Norms, CKM Validation
│   ├── CODEBASE_REFERENCE.md              # Exhaustive File Directory, Fortran Manual, Data Dictionary
│   └── MATHEMATICAL_FORMULATION.md        # Complete Navier-Stokes Math, Metrics, Stencils, Code Traces
│
├── python/                                # Python Solver, Diagnostics & Analysis Modules
│   ├── benchmark_solvers.py               # Live Dynamic GPU vs CPU Fortran Benchmark Suite
│   ├── compare_fortran_cuda.py            # Quantitative Field Accuracy & Error Norm Evaluator
│   ├── compare_processing_times.py        # Post-Processing Execution Time Benchmarking Tool
│   ├── compute_diagnostics_cuda.py        # High-Frequency GPU Diagnostic Stream Engine
│   ├── cudacomplete.py                    # All-in-One Solver, Checkpointing & Diagnostics Pipeline
│   ├── plot_validation.py                 # Matplotlib Publication-Quality Validation Plotter
│   ├── process_input_files.py             # Bulk 5-Dataset Ingestion & Statistical Summary Module
│   └── recompile_and_run_fortran.py       # Live Fortran Source Compiler & Timing Runner
│
├── setup/                                 # Environment Setup & Diagnostics Suite
│   ├── check_setup.py                     # Multi-Layer Automated System & GPU Verification
│   ├── requirements.txt                   # Strictly Pinned Dependency Specification (numpy<2)
│   ├── param.dat.example                  # Template Physical Simulation Parameters File
│   ├── setup_windows.bat                  # 1-Click Windows Batch Setup Script
│   ├── setup_windows.ps1                  # Windows PowerShell Setup Script
│   └── setup_linux.sh                     # Linux / Ubuntu / WSL2 Automated Bash Setup Script
│
├── codes1/                                # Modernized Fortran 90 DNS Solver Modules
│   ├── main_v2.0_08_oc+v_st_3.f90         # Main Fortran MPI Driver Program
│   ├── mod_comdata.f90                    # Global Variables, Mesh Metrics, Physical Parameters
│   ├── mod_var_dec.f90                    # Array Declarations & Simulation Constants
│   ├── mod_bc.f90                         # Boundary Condition Update Routines
│   ├── mod_control.f90                    # Active Opposition Flow Control (OC) Module
│   ├── mod_conv.f90                       # 4th-Order Convective Flux Stencil Discretization
│   ├── mod_diff.f90                       # Viscous Stress Tensor & Heat Flux Evaluation
│   ├── mod_in_out.f90                     # Snapshot Checkpoint Reading & Writing
│   ├── mod_my_mpi.f90                     # 3D Cartesian Domain Decomposition & Halo Exchanges
│   └── mod_stats.f90                      # On-the-Fly Turbulence Statistics & Favre Averages
│
├── codes/                                 # Fortran 90 Post-Processing Analysis Routines
│   ├── avgTauWall.f90                     # Wall Shear Stress Statistical Processor
│   ├── avgTauWallT.f90                    # Wall Heat Flux & Temperature Gradient Processor
│   ├── avg_Cf.f90                         # Skin Friction Coefficient ($C_f$) Averaging
│   ├── avg_Favre.f90                      # Favre-Averaged Compressible Turbulence Statistics
│   ├── avg_TKE.f90                        # Turbulent Kinetic Energy & Reynolds Stress Profiles
│   ├── fluc.f90                           # Velocity & Temperature Fluctuation Field Evaluator
│   ├── qInvariant.f90                     # Vortex Q-Criterion Second Invariant Evaluator
│   └── makefile_3D                        # GFortran Compilation Makefile for 3D Solver
│
├── input/                                 # 3D Initial Condition & Statistical Datasets
│   ├── 3D.dat                             # Full 3D Flow Field Checkpoint (527 MB, 2,704,000 cells)
│   ├── Prime3D.dat                        # 3D Velocity Fluctuation Snapshot (527 MB)
│   ├── PrimeUV.dat                        # 2D Cross-Covariance Profile Dataset (127 MB)
│   ├── RMSu2v2w2t2uvvt.dat                # 1D RMS Profile Dataset (15 KB, 119 wall points)
│   └── rhouvwtpmu.dat                     # 1D Mean Profile Dataset (15 KB, 119 wall points)
│
├── output_cuda/                           # Checkpoints & Diagnostics Generated by CUDA Solver
│   ├── 3D_cuda.dat                        # Final Full 3D Tecplot Solution Snapshot
│   ├── fluc.dat                           # 3D Instantaneous Fluctuation Field Snapshot
│   ├── rhobulk.dat, Ubulk.dat, etc.       # High-Frequency Diagnostic Time Series
│   └── benchmark_results.json             # Dynamic Execution Performance Metrics
│
├── output_fortran/                        # Reference Solution Checkpoints from Fortran Solver
│   └── 3D_fortran.dat                     # Reference Fortran Solution Snapshot
│
├── output_processed/                      # Extracted CSV Profiles & JSON Summaries
│   ├── input_processing_summary.json      # Bulk Statistical Properties of Input Datasets
│   ├── extracted_mean_profiles.csv        # Wall-Normal 1D Mean Profile Distribution
│   └── extracted_rms_profiles.csv         # Wall-Normal 1D RMS Profile Distribution
│
├── output_comparison/                     # Field Accuracy Verification Reports
│   ├── comparison_report.md               # Cell-by-Cell Error Metrics Table ($L_1, L_2, L_\infty, r$)
│   └── comparison_summary.json            # Machine-Readable Verification Error Norms
│
├── plots/                                 # Generated Publication Validation Figures
│   └── validation_plots/*.png             # 8 Canonical Validation Figures vs CKM DNS Data
│
├── cuda code.py                           # Main Standalone 3D Compressible CUDA Solver (1,351 lines)
└── param.dat                              # Runtime Physical Simulation Parameters File
```

---

## 2. 🐍 Python Subsystem & Main Solver Specifications

### 1. [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py)
- **Role**: Primary standalone 3D Compressible Navier-Stokes DNS solver accelerated on an NVIDIA GPU using Numba CUDA.
- **Lines**: 1,351 lines | **Language**: Python 3 (Numba CUDA JIT)
- **Key Functions / Kernels**:
  - `primary_variables_kernel`: Inverts conserved state $\mathbf{q} \to \mathbf{v}$, computes $\mu(T)$ and $k_t(T)$.
  - `compute_convective_kernel`: 4th-order central convective flux differences with adaptive Peclet artificial dissipation.
  - `compute_viscous_kernel`: Full 3D viscous stress tensor $\tau_{ij}$, strain rate $S_{ij}$, dissipation $\Phi$, and heat flux $q_k$.
  - `update_rk2_kernel`: Explicit Runge-Kutta 2-stage conserved variable updater.
  - `opposition_and_wall_bc_kernel`: Active opposition blowing/suction at $z=0, 2h$ based on sensor plane $y^+=33$.
  - `save_solution`: Streams 3D volumetric fields in Tecplot POINT layout.
- **Dependencies**: `numba.cuda`, `numpy` (pinned `<2`), `argparse`, `math`, `pandas`.

### 2. [`python/cudacomplete.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/cudacomplete.py)
- **Role**: High-level orchestrator integrating solver execution, periodic checkpointing, fault-tolerant resumption, and GPU diagnostic extraction.
- **CLI Options**: `--steps`, `--save-interval`, `--resume`, `--diagnostics-input`, `--diagnostics-dir`, `--pattern`, `--no-gpu`.

### 3. [`python/compute_diagnostics_cuda.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/compute_diagnostics_cuda.py)
- **Role**: High-frequency streaming engine that executes 3D volume and 2D plane GPU reductions to track bulk density, bulk velocity, wall shear stress, and kinetic energy without I/O bottlenecks.

### 4. [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py)
- **Role**: Live performance benchmarking harness measuring real wall-clock time per step (`time.perf_counter`) and extrapolating speedups up to 1,000 steps.

### 5. [`python/compare_fortran_cuda.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/compare_fortran_cuda.py)
- **Role**: Cell-by-cell numerical validation tool calculating Mean Absolute Error ($L_1$), Relative $L_2$ error norm, Maximum Absolute Error ($L_\infty$), and Pearson correlation coefficient ($r$) across all $2,704,000$ cells.

### 6. [`python/plot_validation.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/plot_validation.py)
- **Role**: Headless Matplotlib visualization utility generating the 8 canonical validation figures comparing CUDA solver profiles against benchmark CKM DNS literature data.

### 7. [`python/process_input_files.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/process_input_files.py)
- **Role**: High-speed batch processor parsing all 5 initial condition datasets in `input/` and extracting statistical profiles.

### 8. [`python/recompile_and_run_fortran.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/recompile_and_run_fortran.py)
- **Role**: Build harness that cleans old binaries, compiles Fortran 90 source modules with `gfortran -O3 -fopenmp`, and measures native CPU execution speed.

---

## 3. ⚙️ Setup, Diagnostics & Configuration Files

- [`setup/check_setup.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/setup/check_setup.py): 6-stage automated pre-flight diagnostic script validating OS, Python, packages, GPU hardware, micro CUDA JIT kernel, and directory structure.
- [`setup/requirements.txt`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/setup/requirements.txt): Pinned dependencies enforcing `"numpy<2"` to prevent Numba CUDA breakage.
- [`setup/param.dat.example`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/setup/param.dat.example): Template configuration file.
- [`param.dat`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/param.dat): Active physical simulation parameters.

---

## 4. 📂 Data Files & Output Directories

- `input/3D.dat`: 527 MB initial flow condition on $160 \times 130 \times 130$ mesh ($2.7\text{M}$ cells).
- `input/Prime3D.dat`: 527 MB velocity fluctuation snapshot.
- `input/PrimeUV.dat`: 127 MB 2D cross-covariance dataset.
- `input/RMSu2v2w2t2uvvt.dat`: 15 KB 1D RMS fluctuation profile.
- `input/rhouvwtpmu.dat`: 15 KB 1D mean flow field profile.
- `output_cuda/`: Output directory for GPU checkpoints, diagnostic files, and benchmark JSON.
- `output_fortran/`: Reference output from Fortran MPI solver.
- `output_comparison/`: Quantitative comparison reports and error norm metrics.
- `plots/validation_plots/`: Generated PNG figures.

---

## 5. 📜 Fortran 90 Subsystem & Legacy Modules Manual

The repository contains a complete, production-grade Fortran 90 implementation of the compressible turbulent channel flow solver in [`codes1/`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/codes1) and analysis routines in [`codes/`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/codes).

### Core Simulation Modules (`codes1/`)

| Module File | Program / Module Name | Purpose & Algorithmic Role |
| :--- | :--- | :--- |
| `main_v2.0_08_oc+v_st_3.f90` | `program chanCompr` | Main simulation driver coordinating explicit RK2 stages, MPI halo exchanges, active control, and I/O. |
| `mod_comdata.f90` | `module comdata` | Reads `param.dat`, allocates 3D mesh arrays, evaluates $\tanh$ grid stretching, and computes metric derivatives. |
| `mod_var_dec.f90` | `module var_dec` | Declares conserved state buffers (`var`, `solV`), physical constants ($\text{Re}, \text{Ma}, \text{Pr}, \gamma$), and boundary arrays. |
| `mod_bc.f90` | `module bc` | Implements isothermal no-slip wall conditions and computes wall derivatives. |
| `mod_control.f90` | `module control` | Implements Active Opposition Flow Control ($w_{\text{wall}} = -w(y^+ = 33)$). |
| `mod_conv.f90` | `module conv` | 4th-order convective flux discretization with adaptive Peclet number limiter. |
| `mod_diff.f90` | `module diff` | Evaluates viscous stress tensor $\tau_{ij}$, heat fluxes, and viscous dissipation rates. |
| `mod_in_out.f90` | `module in_out` | Handles initial condition parsing (`read_input_files`) and Tecplot snapshot writes (`write_output_files`). |
| `mod_my_mpi.f90` | `module my_mpi` | Manages 3D Cartesian domain decomposition (`MPI_Cart_create`) and non-blocking boundary exchanges (`MPI_Sendrecv`). |
| `mod_stats.f90` | `module stats` | Computes Favre averages, Reynolds stresses, skin friction $C_f$, and TKE budgets. |

---

### Post-Processing & Statistical Tools (`codes/`)

| Program File | Target Executable | Analysis Capability |
| :--- | :--- | :--- |
| `fluc.f90` | `fluc_win.exe` | Computes 3D fluctuating fields $q'(x, y, z) = q - \bar{q}(z)$ for velocity, density, and temperature. |
| `qInvariant.f90` | `qInvariant_win.exe` | Computes the second invariant of the velocity gradient tensor ($Q$-criterion) for vortex visualization. |
| `avg_v1.7.f90` | `avg_v1.7_win.exe` | Computes Reynolds and Favre spatial/temporal averages across homogeneous $x\text{--}y$ planes. |
| `avgTauWall.f90` | `avgTauWall_win.exe` | Averages wall shear stress $\tau_w$ and computes skin friction coefficient $C_f$. |
| `avg_TKE.f90` | `avg_TKE_win.exe` | Evaluates Turbulent Kinetic Energy profiles and individual Reynolds stress tensor components. |
| `combine.f90` | `combine_win.exe` | Merges partitioned MPI sub-domain output files into a single consolidated Tecplot file. |

---

### GNUplot Visualization Scripts (`.txt`)

- `codes/cmdGNUplot.txt`: Generates mean density, velocity, temperature, and pressure profile comparisons against CKM DNS data.
- `codes/cmdGNUplot2.txt`: Plots RMS velocity fluctuations ($u'_{rms}, v'_{rms}, w'_{rms}$) and wall shear stress history.
- `codes/cmdGNUplot_TKE.txt`: Plots Turbulent Kinetic Energy (TKE) budget terms (production, dissipation, diffusion).
- `codes/cmdGNUplot_TKE_Fav.txt`: Plots Favre-averaged TKE budget profiles.

---

### Sampling Control Files (`.dat`)

Control files in `codes/` specify time-averaging sampling intervals:
- `AoNDT.dat`: Sampling parameters for general statistical averaging.
- `AoNDT_Cf.dat`: Sampling configuration for skin friction coefficient tracking.
- `AoNDT_Favre.dat`: Sampling parameters for Favre-averaged statistics.
- `AoNDT_v1.7_TKE_ALL.dat`: Sampling parameters for full TKE budget averaging.
- `plotInfo.dat`: Axis ranges, tick formatting, and plot label definitions.

---

### Build System (`makefile_3D`)

The legacy solver uses GNU Make to build with GFortran and OpenMPI:
```makefile
FC = mpif90
FFLAGS = -O3 -fopenmp -fdefault-real-8 -fdefault-double-8 -cpp
OBJS = mod_var_dec.o mod_comdata.o mod_my_mpi.o mod_bc.o mod_control.o \
       mod_conv.o mod_diff.o mod_stats.o mod_in_out.o main_v2.0_08_oc+v_st_3.o

chanCompr: $(OBJS)
	$(FC) $(FFLAGS) -o $@ $(OBJS)
```

---

## 6. 📖 Data Dictionary: Conserved, Primitive & Metric Variables

| Variable Symbol | Code Name | Units / Scaling | Physical Definition |
| :--- | :--- | :--- | :--- |
| $\rho$ | `rho`, `q[..., 0]` | $\rho / \rho_\infty$ | Non-dimensional fluid density |
| $u$ | `u`, `var[..., 1]` | $u / c_\infty$ | Streamwise velocity component |
| $v$ | `v`, `var[..., 2]` | $v / c_\infty$ | Spanwise velocity component |
| $w$ | `w`, `var[..., 3]` | $w / c_\infty$ | Wall-normal velocity component |
| $E$ | `e`, `var[..., 4]` | $E / (c_\infty^2)$ | Total specific energy: $E = e_{\text{int}} + \frac{1}{2}(u^2+v^2+w^2)$ |
| $T$ | `T`, `var[..., 5]` | $T / T_\infty$ | Static temperature: $T = E - \Gamma_{gM} \cdot k$ |
| $P$ | `p`, `var[..., 6]` | $P / (\rho_\infty c_\infty^2)$ | Static pressure: $P = \frac{\rho T}{\gamma \text{Ma}^2}$ |
| $\mu$ | `mu` | $\mu / \mu_\infty$ | Dynamic molecular viscosity: $\mu(T) = T^{0.7}$ |
| $k_t$ | `kt` | $k_t / k_{t,\infty}$ | Thermal conductivity: $k_t(T) = \mu(T)$ |
| $\tau_{ij}$ | `tau_ij` | $\tau / (\rho_\infty c_\infty^2)$ | Viscous shear stress tensor |
| $\Phi$ | `dissip` | $\Phi$ | Viscous dissipation rate |
| $\frac{d\eta}{dz}$ | `deta_dz` | $1 / \text{length}$ | First metric derivative of wall-normal coordinate stretching |
| $\frac{d^2\eta}{dz^2}$ | `d2eta_dz2` | $1 / \text{length}^2$ | Second metric derivative of wall-normal coordinate stretching |
| $C_f$ | `cf` | Non-dimensional | Skin friction coefficient: $C_f = \frac{2\tau_w}{\rho_b U_b^2}$ |
| $\tau_w$ | `tau_w` | $\tau / (\rho_\infty c_\infty^2)$ | Wall shear stress: $\left.\mu_w \frac{\partial u}{\partial z}\right\|_{z=0}$ |
| $u_\tau$ | `u_tau` | $u / c_\infty$ | Friction velocity: $u_\tau = \sqrt{\tau_w / \rho_w}$ |
| $z^+$ | `z_plus` | Wall units | Viscous wall distance: $z^+ = z u_\tau / \nu_w$ |
