# 3D Compressible Turbulent Channel Flow DNS Solver — Documentation Hub

Welcome to the central documentation portal for the **3D Compressible Supersonic Wall-Bounded Turbulent Channel Flow Direct Numerical Simulation (DNS) Solver** ($\text{Re}_\tau = 3000, \text{Ma} = 1.5$, $160 \times 130 \times 130 = 2,704,000$ cells).

This repository contains both the reference **Fortran 90 + OpenMP/MPI** solver and an accelerated **Python + Numba CUDA GPU** solver equipped with active opposition flow control ($\text{OC}$) and high-frequency streaming diagnostics.

---

## 🏛️ The 7 Core Documentation Pillars

All technical documentation is organized inside the [`docs/`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs) directory into seven comprehensive, modular guides:

| Document | Primary Focus | Key Contents |
| :--- | :--- | :--- |
| **[`docs/SETUP_AND_CLOUD_GUIDE.md`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs/SETUP_AND_CLOUD_GUIDE.md)** | **Installation & Cloud** | Cross-platform local setup (Windows, Linux, WSL2), Google Colab 1-click cells (T4 GPU), pinned dependencies (`numpy<2`), environment paths, and troubleshooting matrix. |
| **[`docs/ARCHITECTURE_AND_KERNELS.md`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs/ARCHITECTURE_AND_KERNELS.md)** | **Systems & Kernels** | Architectural Mermaid flowchart, CUDA thread execution sequence diagram, 5-stage lifecycle, CUDA kernel specifications, memory layouts, compiler flags, and dependency audit. |
| **[`docs/SOLVER_AND_DIAGNOSTICS_GUIDE.md`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs/SOLVER_AND_DIAGNOSTICS_GUIDE.md)** | **Operations & Tutorial** | Dual-level checkpointing (3D Tecplot snapshots + live diagnostic streaming), fault-tolerant resumption (`--resume`), unified orchestration with `cudacomplete.py`, and 6-module hands-on tutorial. |
| **[`docs/BENCHMARK_AND_VALIDATION_REPORT.md`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs/BENCHMARK_AND_VALIDATION_REPORT.md)** | **Data & Benchmarks** | CPU vs GPU benchmark tables across 1 to 1,000 steps, 165 s time savings, cell-by-cell error metrics ($L_1, L_2, L_\infty, r$), post-processing speedups, and CKM benchmark validation. |
| **[`docs/CODEBASE_REFERENCE.md`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs/CODEBASE_REFERENCE.md)** | **Codebase & Manual** | Complete repository directory tree, per-file line counts, exposed symbols, Fortran 90 manual (`codes1/` & `codes/`), build system (`makefile_3D`), and physical data dictionary. |
| **[`docs/MATHEMATICAL_FORMULATION.md`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs/MATHEMATICAL_FORMULATION.md)** | **Governing Mathematics** | Navier-Stokes in curvilinear coordinates, thermodynamic relations, $\tanh$ grid metrics, 4th-order stencils, Peclet upwinding, viscous stress tensor, and exact code line-by-line traces. |
| **[`docs/commands.md`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs/commands.md)** | **CLI Commands Cheatsheet** | Complete command catalog and arguments reference for running `cudacomplete.py` and `benchmark_solvers.py`. |
| **[`docs/README.md`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs/README.md)** | **Documentation Hub** | Master index, repository navigation, executive overview, and 3-step quick start guide (this file). |

---

## ⚡ Quick Start: 3 Steps to Execution

### Step 1: Pre-Flight Environment Setup & Verification
Set up the Python 3.10–3.12 virtual environment and run the automated diagnostics:
```powershell
# Windows 1-click automated setup:
powershell -ExecutionPolicy Bypass -File .\setup\setup_windows.ps1

# Or run diagnostic manually:
python setup/check_setup.py
```
*(On Google Colab, see [Google Colab Setup Guide](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/docs/SETUP_AND_CLOUD_GUIDE.md#4-️-google-colab-cloud-execution-guide) for 1-click copy-paste execution).*

### Step 2: Run the GPU Solver with Periodic Checkpoints
Execute 100 Runge-Kutta time steps on the 2.7M-cell mesh with periodic checkpointing:
```powershell
python python/cudacomplete.py --steps 100 --save-interval 50
```
- Periodic state snapshots: `output_cuda/3D_step50.dat`, `output_cuda/3D_step100.dat`
- Continuous diagnostic streams: `output_cuda/rhobulk.dat`, `output_cuda/Ubulk.dat`, `output_cuda/tauWall.dat`

### Step 3: Resume or Extract Diagnostics
Resume seamlessly from the latest saved checkpoint:
```powershell
python python/cudacomplete.py --steps 100 --save-interval 50 --resume
```
Or generate publication-quality validation figures compared against CKM DNS data:
```powershell
python python/plot_validation.py --cuda-dir output_cuda --output-dir plots/validation_plots
```

---

## 🚀 Key Performance Highlights

- **GPU Acceleration**: Delivers **0.3102 s / step** on a modest NVIDIA GTX 1650 (vs. 0.4753 s / step on 4 CPU cores), saving **165.10 seconds** per 1,000 RK2 steps (**34.73% time reduction**). On modern datacenter GPUs (Tesla T4 / A100), per-step execution scales to $< 0.03\text{ s/step}$.
- **Post-Processing Acceleration**: Up to **4.51× speedup** for 3D $Q$-criterion vortex extraction and **8.92× speedup** for turbulent fluctuation field processing.
- **Numerical Parity**: Full 64-bit IEEE 754 precision matching Fortran mean conserved density to **$0.00\text{e}+00$ relative float error** and relative $L_2$ error $< 0.92\%$.
- **Active Flow Control**: Closed-loop opposition control at $y^+ = 33$ delivering $\sim 18\%\text{--}22\%$ turbulent drag reduction.
