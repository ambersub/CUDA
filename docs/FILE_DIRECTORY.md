# 📁 Exhaustive File Directory & Module Reference

This document provides a comprehensive file-by-file specification of every source file, script, dataset, and configuration file in the project repository.

---

## 📌 1. Repository Directory Structure Map

```
.
├── docs/                                  # Central Technical Documentation Suite
│   ├── README.md                          # Documentation Hub Landing Page
│   ├── PROJECT_FLOW.md                    # Architectural Flow, Sequence Diagrams, Data Lifecycle
│   ├── FILE_DIRECTORY.md                  # File-by-File Reference & Library Breakdown (this file)
│   ├── LIBRARIES_REPORT.md                # Comprehensive Library Audit & Mapping Matrix
│   ├── TECH_STACK.md                      # Technical Stack Breakdown, Compiler Flags, Build Systems
│   ├── Techchoice.md                      # Architectural Technical Choices & Rationale
│   ├── TUTORIAL.md                        # Hands-on User Tutorial & Code Examples
│   ├── FEATURES.md                        # High-Level API & CUDA Kernel Feature Matrix
│   └── FULL_BENCHMARK_REPORT.md           # Master Benchmark Report & Field Error Metrics
│
├── python/                                # Primary Python Solver & Utility Modules
│   ├── process_input_files.py             # Bulk 5-Dataset Ingestion & Statistical Summary Module
│   ├── compare_fortran_cuda.py            # Quantitative Field Accuracy & Error Norm Evaluator
│   ├── recompile_and_run_fortran.py       # Live Fortran Source Compiler & Timing Runner
│   ├── benchmark_solvers.py               # Live Dynamic GPU vs CPU Fortran Benchmark Suite
│   └── compare_processing_times.py        # Post-Processing Execution Time Benchmarking Tool
│
├── codes1/                                # Native Fortran 90 DNS Solver Source Modules (Modernized)
│   ├── mod_comdata.f90                    # Global Variables, Mesh Metrics, Physical Parameters Module
│   ├── mod_bc.f90                         # Boundary Condition Functions & Wall Clamping
│   ├── mod_conv.f90                       # 4th-Order Convective Flux Stencil Kernels
│   ├── mod_diff.f90                       # 2nd-Order Viscous Stress Tensor & Heat Flux Kernels
│   ├── mod_control.f90                    # Opposition Control (OC) Blowing/Suction Implementation
│   ├── mod_stats.f90                      # Turbulence Statistics, Favre Averaging & Correlations
│   ├── mod_in_out.f90                     # Binary / ASCII Snapshot Checkpoint Reading & Writing
│   ├── mod_my_mpi.f90                     # 3D Domain Decomposition & MPI Halo Exchange Routines
│   ├── main_v2.0_08_oc+v_st_3.f90         # Main Fortran MPI Driver Program
│   └── ...                                # Analysis routines & GNUplot script templates
│
├── codes/                                 # Fortran 90 Post-Processing Analysis Routines
│   ├── comdata.f90                        # Common Data Structures Module
│   ├── fluc.f90                           # Velocity & Temperature Fluctuation Field Evaluator
│   ├── qInvariant.f90                     # Vortex Q-Criterion Invariant Detector
│   ├── prime.f90                          # Turbulent Prime Field Extractor
│   ├── avg_v1.7.f90                       # Spanwise/Streamwise Averaging Processor
│   ├── combine.f90                        # Multi-processor Output Merger Utility
│   ├── avgTauWall.f90                     # Wall Shear Stress Processor
│   └── avgTauWallT.f90                    # Wall Heat Flux & Temperature Gradient Processor
│
├── input/                                 # 3D Initial Condition & Statistical Datasets
│   ├── 3D.dat                             # Primary 3D Snapshot Grid (527 MB, 2,704,000 cells)
│   ├── Prime3D.dat                        # 3D Turbulent Velocity Fluctuation Snapshot (527 MB)
│   ├── PrimeUV.dat                        # 2D Cross-Covariance Profile Dataset (127 MB)
│   ├── RMSu2v2w2t2uvvt.dat                # 1D RMS Profile Dataset (15 KB, 119 wall points)
│   └── rhouvwtpmu.dat                     # 1D Mean Flow Field Profile Dataset (15 KB, 119 wall points)
│
├── output_cuda/                           # Exported CUDA GPU Tecplot Checkpoint Outputs
│   └── 3D_cuda.dat                        # Exported 3D Solution State Snapshot
│
├── output_fortran/                        # Fortran MPI Reference Solution Outputs
│   └── 3D_fortran.dat                     # Reference Fortran Solution Snapshot
│
├── output_processed/                      # Processed Datasets & Summary Output Folder
│   ├── input_processing_summary.json      # Summary Statistics JSON for all 5 Input Datasets
│   ├── extracted_mean_profiles.csv        # Extracted 1D Mean Profiles CSV
│   └── extracted_rms_profiles.csv         # Extracted 1D RMS Profiles CSV
│
├── output_comparison/                     # Fortran vs CUDA Verification Reports
│   ├── comparison_report.md               # Field Accuracy Verification Markdown Report
│   ├── comparison_summary.json            # Error Norm Metrics Summary JSON
│   └── recompiled_fortran_timings.json    # Live Re-compiled Fortran Performance JSON
│
├── cuda code.py                           # Main Python Numba CUDA DNS Solver (1,351 lines)
├── param.dat                              # Runtime Physical Simulation Parameters Config File
├── terminal.dat                           # Reference Run Execution Log of Fortran MPI Solver
├── README.md                              # Main Project Landing Page
├── HANDOFF.md                             # Project Handoff Summary Document
├── FULL_BENCHMARK_REPORT.md               # Comprehensive Benchmark Data Report
├── TUTORIAL.md                            # Comprehensive User Guide
├── benchmark_solvers.py                   # Root CLI Wrapper for python/benchmark_solvers.py
├── compare_fortran_cuda.py                # Root CLI Wrapper for python/compare_fortran_cuda.py
├── compare_processing_times.py            # Root CLI Wrapper for python/compare_processing_times.py
├── process_input_files.py                 # Root CLI Wrapper for python/process_input_files.py
└── recompile_and_run_fortran.py           # Root CLI Wrapper for python/recompile_and_run_fortran.py
```

---

## 📌 2. Detailed File Specifications & Library Breakdown

### 2.1 Root Main Solver & Configuration Files

#### 1. [`cuda code.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/cuda%20code.py)
- **Lines**: 1,351 lines | **Size**: ~40.4 KB
- **Core Responsibility**: Main 3D Compressible Supersonic Navier-Stokes DNS Solver accelerated on single NVIDIA GPU using Python and Numba CUDA. Implements wall-normal tanh mesh generation, 4th-order central convective flux stencils with Peclet upwinding, 2nd-order viscous stress tensor evaluation, active opposition flow control ($\text{OC}$), and explicit 2-stage Runge-Kutta (RK2) time integration.
- **Exposed Symbols / Functions**:
  - `compute_convective_kernel`: Device JIT kernel for 4th-order convective fluxes with Peclet upwinding.
  - `compute_viscous_kernel`: Device JIT kernel for viscous stress tensor $\tau_{ij}$ and heat flux vectors.
  - `update_rk2_kernel`: Device JIT kernel for explicit RK2 conserved variable updates.
  - `apply_bc_kernel`: Boundary condition kernel implementing active opposition control at $z=0, 2h$.
  - `parse_input_file`: Ingestion routine for raw whitespace-delimited ASCII input grids.
  - `run_solver`: Main driver function executing initialization, GPU memory allocation, RK2 sub-stepping, and Tecplot export.
- **Libraries Used In This File**:
  - `numba.cuda` (`cuda`, `@cuda.jit`, `float64`, `int32`): GPU JIT compiler and thread block manager.
  - `numpy` (`np`): Host array creation, vector operations, and memory allocations.
  - `argparse`: Command line interface parsing (`--input`, `--steps`, `--output`, `--demo`).
  - `math`: Hardware scalar mathematical operations (`math.sqrt`, `math.exp`, `math.tanh`).
  - `os`, `sys`, `site`, `glob`, `re`, `time`: System integration, CUDA path auto-resolution, and execution timing.

---

#### 2. [`param.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/param.dat)
- **Lines**: ~35 lines | **Size**: ~1.1 KB
- **Core Responsibility**: Runtime simulation config specifying grid resolution ($160 \times 130 \times 130$), flow Mach number ($\text{Ma} = 1.5$), Reynolds number ($\text{Re} = 3000$), time step ($\Delta t = 1.0 \times 10^{-4}$), domain dimensions ($L_x = 4\pi, L_y = \frac{4}{3}\pi, L_z = 2$), and Peclet upwinding limit thresholds ($\text{Pe}_x=2.0, \text{Pe}_y=1.0, \text{Pe}_z=0.1$).
- **Libraries Used**: ASCII plain-text config parsed via standard Python file I/O and `mod_in_out.f90`.

---

#### 3. [`terminal.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/terminal.dat)
- **Lines**: ~150 lines | **Size**: ~3.9 KB
- **Core Responsibility**: Reference execution log of the original Fortran 90 solver running with 4 MPI processes ($1 \times 1 \times 4$ domain topology) on step range $18,963,000$ to $18,963,050$.
- **Libraries Used**: Fortran MPI stdout logging output.

---

### 2.2 Python Utility Pipeline (`python/`)

#### 4. [`python/process_input_files.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/process_input_files.py)
- **Lines**: 494 lines | **Size**: ~17.8 KB
- **Core Responsibility**: Ingestion, validation, and statistical analysis module for all 5 input datasets in `input/`. Calculates bulk density $\rho_b$, bulk velocity $U_b$, wall shear stress $\tau_w$, skin friction $C_f$, turbulence kinetic energy ($k$), and exports profile summaries to JSON and CSV formats.
- **Exposed Symbols**: `InputDatasetProcessor`, `process_3d_dat`, `process_prime3d_dat`, `process_primeuv_dat`, `process_rms_dat`, `process_mean_dat`, `main`.
- **Libraries Used In This File**:
  - `pandas`: High-performance dataset parsing and CSV generation (`pd.DataFrame`, `engine='c'`).
  - `numpy`: Statistical calculations (`np.mean`, `np.std`, `np.trapz`).
  - `json`: Structured output export (`json.dump`).
  - `argparse`, `os`, `sys`, `pathlib`, `logging`, `time`: CLI interface, path management, and logging.

---

#### 5. [`python/compare_fortran_cuda.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/compare_fortran_cuda.py)
- **Lines**: 379 lines | **Size**: ~14.8 KB
- **Core Responsibility**: Comprehensive verification module evaluating field accuracy metrics ($L_1$ mean absolute error, relative $L_2$ norm error, $L_\infty$ maximum error, and Pearson correlation $r$) between reference Fortran MPI outputs and Numba CUDA outputs.
- **Exposed Symbols**: `FortranCudaComparator`, `calculate_error_norms`, `generate_markdown_report`, `main`.
- **Libraries Used In This File**:
  - `numpy`: Matrix difference computations and array indexing.
  - `scipy.stats`: Pearson correlation evaluation (`scipy.stats.pearsonr`).
  - `json`, `argparse`, `os`, `sys`, `time`: Report persistence and execution control.

---

#### 6. [`python/recompile_and_run_fortran.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/recompile_and_run_fortran.py)
- **Lines**: 218 lines | **Size**: ~8.6 KB
- **Core Responsibility**: Automation script for clean-rebuilding all native Fortran binaries (`avg_v1.7`, `fluc`, `prime`, `qInvariant`) using `gfortran`, removing stale outputs, executing test runs, and recording precise wall-clock execution timings.
- **Exposed Symbols**: `FortranRecompiler`, `compile_fortran_file`, `run_binary_and_time`, `main`.
- **Libraries Used In This File**:
  - `subprocess`: Live process execution and stdout/stderr capture (`subprocess.run`).
  - `time`: Sub-millisecond wall-clock timing (`time.perf_counter`).
  - `json`, `argparse`, `os`, `sys`: Workflow management and report export.

---

#### 7. [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/benchmark_solvers.py)
- **Lines**: 217 lines | **Size**: ~8.2 KB
- **Core Responsibility**: Dynamic execution benchmark harness comparing real-time per-step speeds of the Numba CUDA GPU solver against the reference 4-core Fortran MPI solver across arbitrary step counts.
- **Exposed Symbols**: `BenchmarkRunner`, `run_cuda_benchmark`, `extrapolate_timings`, `main`.
- **Libraries Used In This File**:
  - `time`: High-precision timer (`time.perf_counter`).
  - `subprocess`, `argparse`, `json`, `sys`, `os`: Process execution and timing analysis.

---

#### 8. [`python/compare_processing_times.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/compare_processing_times.py)
- **Lines**: 259 lines | **Size**: ~9.1 KB
- **Core Responsibility**: Benchmarking module for post-processing analysis routines (Vortex $Q$-criterion and velocity fluctuation fields), measuring speedup factors of vectorized Python/CUDA processors over re-compiled native Fortran binaries.
- **Exposed Symbols**: `PostProcessBenchmark`, `benchmark_q_criterion`, `benchmark_fluc`, `main`.
- **Libraries Used In This File**:
  - `numpy`: Vectorized field tensor transformations.
  - `time`, `subprocess`, `json`, `argparse`, `os`, `sys`: Benchmark logging and orchestration.

---

### 2.3 Fortran DNS Solver & Post-Processing Source Modules (`codes1/` & `codes/`)

#### 9. [`codes1/mod_comdata.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_comdata.f90)
- **Lines**: 305 lines | **Size**: ~8.4 KB
- **Core Responsibility**: Central Fortran 90 data module declaring global domain constants, flow parameters ($\text{Re}, \text{Ma}, \text{Pr}, \gamma$), grid dimensions, metric arrays, and conserved state arrays.
- **Exposed Symbols**: `module mod_comdata`, `nx`, `ny`, `nz`, `q`, `v`, `deta_dz`, `d2eta_dz2`.
- **Libraries Used**: Fortran 90 Standard `ISO_FORTRAN_ENV`.

---

#### 10. [`codes1/mod_bc.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_bc.f90)
- **Lines**: 474 lines | **Size**: ~19.8 KB
- **Core Responsibility**: Boundary condition module implementing periodic boundary conditions in $x$ and $y$, wall boundary conditions (no-slip, isothermal/adiabatic wall temperature), and high-order 3rd/4th-order one-sided derivative stencils near boundary walls.
- **Exposed Symbols**: `subroutine boundary_conditions`, `subroutine enforce_wall_bc`.
- **Libraries Used**: Fortran 90 Standard Math intrinsics.

---

#### 11. [`codes1/mod_conv.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_conv.f90)
- **Lines**: 194 lines | **Size**: ~7.8 KB
- **Core Responsibility**: Convective flux discretization module evaluating 4th-order central finite difference derivative operators and Peclet-number adaptive upwinding.
- **Exposed Symbols**: `subroutine convective_fluxes`, `subroutine peclet_upwind`.
- **Libraries Used**: Fortran 90 Standard Array operations.

---

#### 12. [`codes1/mod_diff.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_diff.f90)
- **Lines**: 285 lines | **Size**: ~9.5 KB
- **Core Responsibility**: Viscous flux discretization module evaluating full 3D compressible stress tensor $\tau_{ij}$, strain rate tensors, thermal conductivity fluxes, and energy dissipation rates.
- **Exposed Symbols**: `subroutine viscous_fluxes`, `subroutine compute_stress_tensor`.
- **Libraries Used**: Fortran 90 Standard Math intrinsics.

---

#### 13. [`codes1/mod_control.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_control.f90)
- **Lines**: 48 lines | **Size**: ~1.0 KB
- **Core Responsibility**: Active opposition flow control ($\text{OC}$) module evaluating wall blowing and suction velocity profiles $w_{\text{wall}}(x,y) = -w(x,y,y^+=33)$ to suppress wall turbulence and skin friction drag.
- **Exposed Symbols**: `subroutine active_opposition_control`.
- **Libraries Used**: Fortran 90 Standard.

---

#### 14. [`codes1/mod_stats.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_stats.f90)
- **Lines**: 1,669 lines | **Size**: ~61.7 KB
- **Core Responsibility**: Extensive statistical post-processing module computing Reynolds-averaged and Favre-averaged flow fields, turbulent kinetic energy budgets ($\text{TKE}$), velocity cross-covariances $\langle u'v' \rangle$, and spanwise/streamwise spatial correlations.
- **Exposed Symbols**: `subroutine compute_turbulence_stats`, `subroutine favre_average`, `subroutine tke_budget`.
- **Libraries Used**: Fortran 90 Standard Math intrinsics & array algebra.

---

#### 15. [`codes1/mod_in_out.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_in_out.f90)
- **Lines**: 169 lines | **Size**: ~4.4 KB
- **Core Responsibility**: I/O handler reading input grid files ([`input/3D.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/input/3D.dat)) and exporting binary/ASCII solution checkpoint files (`3D001.dat`–`3D008.dat`).
- **Exposed Symbols**: `subroutine read_restart`, `subroutine write_solution`.
- **Libraries Used**: Fortran 90 Unformatted / Formatted I/O statements.

---

#### 16. [`codes1/mod_my_mpi.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_my_mpi.f90)
- **Lines**: 208 lines | **Size**: ~8.6 KB
- **Core Responsibility**: MPI domain decomposition module setting up 3D Cartesian process grids (`MPI_Cart_create`), performing halo exchanges (`MPI_Sendrecv`), and executing global statistics reductions (`MPI_Allreduce`).
- **Exposed Symbols**: `subroutine init_mpi`, `subroutine halo_exchange`, `subroutine global_sum`.
- **Libraries Used**: Fortran `mpi` module (`use mpi` / `mpif.h`).

---

#### 17. [`codes/fluc.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes/fluc.f90)
- **Lines**: 92 lines | **Size**: ~2.8 KB
- **Core Responsibility**: Standalone post-processing program extracting turbulent velocity fluctuation components ($u', v', w'$) and temperature fluctuations ($T'$) from 3D flow snapshots.
- **Exposed Symbols**: `program fluc`.
- **Libraries Used**: Fortran 90 Standard file I/O & math intrinsics.

---

#### 18. [`codes/qInvariant.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes/qInvariant.f90)
- **Lines**: 99 lines | **Size**: ~2.8 KB
- **Core Responsibility**: Standalone post-processing program computing the Second Invariant of the Velocity Gradient Tensor ($Q$-criterion) for 3D coherent vortex structure identification.
- **Exposed Symbols**: `program qInvariant`.
- **Libraries Used**: Fortran 90 Standard file I/O & math intrinsics.

---

#### 19. [`codes/prime.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes/prime.f90)
- **Lines**: 100 lines | **Size**: ~2.3 KB
- **Core Responsibility**: Standalone post-processing program extracting prime RMS fluctuation fields from 3D flow fields.
- **Exposed Symbols**: `program prime`.
- **Libraries Used**: Fortran 90 Standard.

---

#### 20. [`codes/avg_v1.7.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes/avg_v1.7.f90)
- **Lines**: 291 lines | **Size**: ~7.5 KB
- **Core Responsibility**: Planar averaging program computing homogeneous $x$-$y$ plane averages for mean velocity profiles $\bar{u}(z)$, mean density $\bar{\rho}(z)$, and mean temperature $\bar{T}(z)$.
- **Exposed Symbols**: `program avg_v1_7`.
- **Libraries Used**: Fortran 90 Standard.

---

### 2.4 Summary Matrix of All Files

| File Path | Lines | Size | Language / Format | Primary Function |
| :--- | :--- | :--- | :--- | :--- |
| [`cuda code.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/cuda%20code.py) | 1,351 | 40.4 KB | Python / Numba CUDA | Main Numba CUDA 3D Navier-Stokes DNS Solver |
| [`param.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/param.dat) | 35 | 1.1 KB | ASCII Text Data | Physics & Mesh Simulation Setup Parameters |
| [`terminal.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/terminal.dat) | 150 | 3.9 KB | ASCII Log File | Fortran MPI 4-Core Execution Reference Log |
| [`python/process_input_files.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/process_input_files.py) | 494 | 17.8 KB | Python 3 | Ingestion & Summary Statistics for 5 Input Datasets |
| [`python/compare_fortran_cuda.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/compare_fortran_cuda.py) | 379 | 14.8 KB | Python 3 | Quantifies $L_1, L_2, L_\infty$ Error Norms & Pearson $r$ |
| [`python/recompile_and_run_fortran.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/recompile_and_run_fortran.py) | 218 | 8.6 KB | Python 3 | Clean Re-compilation & Timed Execution of Fortran Binaries |
| [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/benchmark_solvers.py) | 217 | 8.2 KB | Python 3 | Dynamic GPU vs Fortran MPI Step Time Benchmark |
| [`python/compare_processing_times.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/python/compare_processing_times.py) | 259 | 9.1 KB | Python 3 | Post-Processing Speedup Benchmark (Q-Criterion, Fluc) |
| [`codes1/mod_comdata.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_comdata.f90) | 305 | 8.4 KB | Fortran 90 | Global Variable Declarations & Mesh Metrics |
| [`codes1/mod_bc.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_bc.f90) | 474 | 19.8 KB | Fortran 90 | Periodic & Wall Boundary Condition Routines |
| [`codes1/mod_conv.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_conv.f90) | 194 | 7.8 KB | Fortran 90 | 4th-Order Convective Fluxes & Upwinding |
| [`codes1/mod_diff.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_diff.f90) | 285 | 9.5 KB | Fortran 90 | 2nd-Order Viscous Stress Tensor & Heat Fluxes |
| [`codes1/mod_control.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_control.f90) | 48 | 1.0 KB | Fortran 90 | Opposition Control (OC) Wall Suction/Blowing |
| [`codes1/mod_stats.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_stats.f90) | 1,669 | 61.7 KB | Fortran 90 | Turbulence Statistics & Favre Averaging Engine |
| [`codes1/mod_in_out.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_in_out.f90) | 169 | 4.4 KB | Fortran 90 | Snapshot I/O Checkpoint File Handler |
| [`codes1/mod_my_mpi.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes1/mod_my_mpi.f90) | 208 | 8.6 KB | Fortran 90 | 3D Cartesian MPI Domain Decomposition & Halos |
| [`codes/fluc.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes/fluc.f90) | 92 | 2.8 KB | Fortran 90 | Standalone Fluctuation Field Extractor |
| [`codes/qInvariant.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes/qInvariant.f90) | 99 | 2.8 KB | Fortran 90 | Standalone Vortex $Q$-Criterion Calculator |
| [`codes/prime.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes/prime.f90) | 100 | 2.3 KB | Fortran 90 | Standalone Turbulent Prime Fluctuation Extractor |
| [`codes/avg_v1.7.f90`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/codes/avg_v1.7.f90) | 291 | 7.5 KB | Fortran 90 | Homogeneous Planar Averaging Program |
