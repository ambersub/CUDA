# 3D Compressible Turbulent Channel Flow DNS Solver — Documentation Hub

Welcome to the central documentation hub for the **3D Compressible Supersonic Wall-Bounded Turbulent Channel Flow Direct Numerical Simulation (DNS) Solver**. This repository houses both the original **Fortran 90 + MPI** parallel implementation and a high-performance **Python + Numba CUDA GPU** solver with active opposition flow control ($\text{OC}$).

---

## 📌 Executive Summary of Documented Artifacts

The complete technical documentation suite is organized inside the [`docs/`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/docs) directory:

```
docs/
├── README.md                  # Central documentation hub & landing page (this file)
├── PROJECT_FLOW.md            # Architectural flow, Mermaid sequence/flowcharts, CUDA execution lifecycle
├── FILE_DIRECTORY.md          # File-by-file specifications & explicit library lists for every file
├── LIBRARIES_REPORT.md        # Comprehensive library audit & function-level mapping matrix
├── TECH_STACK.md              # Technical stack breakdown, compiler flags, and build systems
├── Techchoice.md              # Technical choices: CUDA Numba vs Fortran MPI, memory layouts & precision
├── TUTORIAL.md                # Complete 6-module hands-on user guide & code examples
├── FEATURES.md                # High-level API feature matrix & function reference
└── FULL_BENCHMARK_REPORT.md   # Comprehensive benchmark report, field error metrics & 1000-step analysis
```

---

## 📑 Detailed Breakdown of Documentation Modules

### 1. 🔄 Architectural Flow & Execution Lifecycle ([`docs/PROJECT_FLOW.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/docs/PROJECT_FLOW.md))
- **Mermaid Flowchart**: End-to-end visual mapping from `param.dat` & `input/3D.dat` ingestion $\rightarrow$ grid mesh generation $\rightarrow$ primitive variable conversion $\rightarrow$ RK2 sub-stepping $\rightarrow$ boundary control $\rightarrow$ Tecplot output export.
- **Mermaid Sequence Diagram**: CUDA kernel execution sequence, thread block mapping `(tx, ty, tz)`, GPU device memory allocations, and synchronization barriers.
- **5-Step Physical & Numerical Lifecycle**:
  1. Input Ingestion & Mesh Metric Generation (Tanh stretching in wall-normal $z$, metrics $\mathcal{J}, \eta_z, \eta_{zz}$).
  2. Primitive Variable Conversion & Boundary Condition Enforcement (Active Opposition Control blowing/suction at $z=0, 2h$).
  3. Convective & Diffusive Flux Stencil Evaluation (4th-order central + Peclet-based upwind + full 3D viscous stress tensor $\tau_{ij}$).
  4. RK2 Explicit Time Integration & Sub-stepping (Step 1 predictor & Step 2 corrector).
  5. Post-Processing & Checkpoint Export (Tecplot `.dat` output format, Q-Criterion vortex detection, Fluctuation RMS fields).
- **Explicit Library Annotations**: Annotated every execution phase with the exact Fortran module, C++ runtime header, or Python scientific package driving it.

---

### 2. 📁 Exhaustive File Directory & Module Reference ([`docs/FILE_DIRECTORY.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/docs/FILE_DIRECTORY.md))
- **Complete Repository Tree**: Detailed structural map of all directories (`cuda code.py`, `python/`, `codes/`, `codes1/`, `input/`, `output/`).
- **File Specifications**: Line counts, file sizes, core responsibilities, and exposed symbols for every single file in the project.
- **Libraries Used In This File**: Explicit breakdown of imported/included libraries for **each individual file**:
  - Main GPU Solver: [`cuda code.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/cuda%20code.py) (1,351 lines)
  - Python Utility Pipeline: `process_input_files.py` (494 lines), `compare_fortran_cuda.py` (379 lines), `recompile_and_run_fortran.py` (218 lines), `benchmark_solvers.py` (217 lines), `compare_processing_times.py` (259 lines).
  - Fortran Solver Modules (`codes1/`): `mod_comdata.f90`, `mod_bc.f90`, `mod_conv.f90`, `mod_diff.f90`, `mod_control.f90`, `mod_stats.f90`, `mod_in_out.f90`, `mod_my_mpi.f90`, `main_v2.0_08_oc+v_st_3.f90`.
  - Fortran Post-Processing Routines (`codes/`): `fluc.f90`, `qInvariant.f90`, `prime.f90`, `avg_v1.7.f90`, `combine.f90`, `avgTauWall.f90`, `avgTauWallT.f90`.
  - Config & Data Files: `param.dat`, `terminal.dat`, and all 5 raw datasets in `input/`.

---

### 3. 📚 Dependencies & Libraries Audit ([`docs/LIBRARIES_REPORT.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/docs/LIBRARIES_REPORT.md))
- **Mermaid Mindmap Diagram**: Taxonomic breakdown of CUDA Numba, Scientific Python stack, Fortran 90 standard library, MPI, OpenMP, and build tools.
- **Specification Matrix**: Comprehensive table detailing library names, include/import statements, category, license, and core solver responsibilities.
- **File-by-File & Function-by-Function Location Map**: Exhaustive mapping table connecting every module (`numba.cuda`, `numpy`, `pandas`, `scipy.stats`, `matplotlib`, `argparse`, `dataclasses`, `struct`, `gfortran`, `mpi`) directly to its exact file path and function invocations.

---

### 4. 🛠️ Tech Stack & Key Technical Decisions ([`docs/TECH_STACK.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/docs/TECH_STACK.md) & [`docs/Techchoice.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/docs/Techchoice.md))
- **Compiler Flags & Build System Matrix**: Exhaustive breakdown of flags across platforms (`gfortran -O3 -fopenmp -fdefault-real-8`, Numba CUDA NVVM / JIT target flags).
- **Why We Selected Python + Numba CUDA Over Fortran MPI**:
  - Elimination of inter-node MPI message-passing latency (`MPI_Sendrecv`).
  - Single-GPU unified memory indexing eliminating halo buffers.
  - Zero IPC serialization overhead, yielding **1.532× GPU solver speedup** and up to **8.92× post-processing speedup**.
- **Numerical Precision & IEEE 754 Compliance**:
  - Full 64-bit double precision floating-point arithmetic (`float64` / `real(kind=8)`).
  - Exact match of field mean quantities to $0.00\text{e}+00$ relative float error and relative $L_2$ error $< 0.92\%$.

---

### 5. 📖 Complete Hands-On Tutorial ([`docs/TUTORIAL.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/docs/TUTORIAL.md))
- **Prerequisites & Installation**: Environment setup for Numba CUDA, NVCC, NVRTC, and GFortran 16.1.0 on Windows.
- **6 Step-by-Step Code Modules**:
  1. Module 1: Environment Verification & CUDA Device Discovery
  2. Module 2: Executing the Python Numba CUDA DNS Solver (`cuda code.py`)
  3. Module 3: Processing & Summarizing Raw Input Datasets (`process_input_files.py`)
  4. Module 4: Live Re-compilation & Execution of Native Fortran Binaries (`recompile_and_run_fortran.py`)
  5. Module 5: Running the Live Solver Benchmark Suite (`benchmark_solvers.py`)
  6. Module 6: Post-Processing Analysis (Q-Criterion & Velocity Fluctuation Fields)
- **Troubleshooting & FAQ**: Windows DLL initialization, `CUDA_HOME` auto-detection, float precision verification, Tecplot format parsing.

---

### 6. ⚡ High-Level Feature Reference ([`docs/FEATURES.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/docs/FEATURES.md))
- **Feature Matrix**: Side-by-side capability comparison between Fortran 90 MPI and Python Numba CUDA solvers.
- **API & Kernel Reference**: Detailed descriptions of CUDA JIT device kernels (`compute_convective_kernel`, `compute_viscous_kernel`, `update_rk2_kernel`, `apply_bc_kernel`) and Python utility classes (`InputProcessor`, `FortranCudaComparator`, `BenchmarkRunner`).

---

### 7. 📊 Benchmark Metrics & Master Data Report ([`docs/FULL_BENCHMARK_REPORT.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/docs/FULL_BENCHMARK_REPORT.md))
- Comprehensive performance comparison on the $160 \times 130 \times 130$ ($2,704,000$ cells) DNS domain.
- **Solver Step Timings**:
  - Fortran MPI: **0.4753 s / step** (4 CPU Cores)
  - Python Numba CUDA: **0.3102 s / step** (1 NVIDIA GPU)
  - Time saved over 1,000 steps: **165.10 seconds saved** (**34.73% time reduction**, **1.532× speedup**)
- **Post-Processing Timings**:
  - Vortex $Q$-Criterion: **4.51× GPU speedup** (35.95s Fortran vs 7.98s CUDA)
  - Fluctuation Fields: **8.92× GPU speedup** (63.81s Fortran vs 7.15s CUDA)
- **Field-by-Field Error Norms**: $L_1$, relative $L_2$, $L_\infty$, and Pearson correlation coefficients ($r > 0.992$ for key physical variables).

---

## 🚀 Quick Execution Cheatsheet

```powershell
# 1. Process all 5 raw input datasets in input/
python python/process_input_files.py --input-dir input --output-dir output_processed

# 2. Re-compile native Fortran source code from scratch & calculate live execution timings
python python/recompile_and_run_fortran.py

# 3. Run live dynamic execution benchmark (Fortran MPI vs Python CUDA)
python python/benchmark_solvers.py --steps 50

# 4. Run Main Python CUDA DNS Solver for 100 RK2 time steps
python "cuda code.py" --steps 100 --output output_cuda/3D_cuda.dat

# 5. Compare Fortran and CUDA solution field outputs and quantitative error metrics
python python/compare_fortran_cuda.py --fortran-dir output_fortran --cuda-dir output_cuda --output-dir output_comparison
```
