# Project Handoff Documentation: 3D Compressible CFD Solver

**Repository**: [`https://github.com/ambersub/CUDA`](https://github.com/ambersub/CUDA)  
**Branch**: `main`  
**Latest Commit**: `41feda7` ("refortran computing")  
**Target Flow Case**: 3D Compressible Supersonic Turbulent Channel Flow ($\text{Re} = 3000$, $\text{Ma} = 1.5$, $160 \times 130 \times 130 = 2,704,000$ cells)  
**Master Data Report**: [`FULL_BENCHMARK_REPORT.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/FULL_BENCHMARK_REPORT.md)

---

## 1. 1000-Step Solver Execution Time & Speedup Comparison

Below is the dynamic execution performance comparison between the **Fortran 90 MPI Solver** (4 CPU Cores) and the **Python Numba CUDA GPU Solver** (1 NVIDIA GPU) extrapolated across iteration step scales up to **1,000 steps**:

### Performance Metrics for Main Flow Solver

| Metric / Step Scale | Fortran 90 MPI Solver (4 CPU Cores) | Python Numba CUDA Solver (1 GPU) | Time Saved on GPU ($\Delta T$) | GPU Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Time per Single Step** | **0.4753 s / step** | **0.3102 s / step** | **0.1651 s saved / step** | **1.532× GPU Speedup** |
| **Time per 10 RK2 Steps** | **4.753 s** | **3.102 s** | **1.651 s saved** | **1.532× GPU Speedup** |
| **Time per 100 RK2 Steps** | **47.53 s** | **31.02 s** | **16.51 s saved** | **1.532× GPU Speedup** |
| **Time per 1,000 RK2 Steps** | **475.30 s** *(7 min 55.3 s)* | **310.20 s** *(5 min 10.2 s)* | **165.10 s saved** *(2 min 45.1 s)* | **1.532× GPU Speedup** |

### Time Difference Summary for 1,000 Steps
- **Fortran 90 MPI Execution Time (1,000 steps)**: **475.30 seconds** ($7.92\text{ minutes}$)
- **Python CUDA GPU Execution Time (1,000 steps)**: **310.20 seconds** ($5.17\text{ minutes}$)
- **Net Time Saved on GPU**: **165.10 seconds** ($2.75\text{ minutes}$ or **$34.73\%$ time reduction**)

---

## 2. Post-Processing Analysis Routine Timings

Re-compiled Fortran binaries vs vectorized CUDA / Python data processing routines on full 3D snapshot fields ($2,704,000$ cells):

| Analysis Task | Re-compiled Fortran Binary (`gfortran 16.1.0`) | CUDA / Vectorized Python Processor | Time Difference Saved | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Vortex $Q$-Criterion (`qInvariant`)** | **35.948 seconds** | **7.977 seconds** | **27.971 seconds saved** | **4.51× GPU Speedup** |
| **Fluctuation Field (`fluc`)** | **63.807 seconds** | **7.148 seconds** | **56.659 seconds saved** | **8.92× GPU Speedup** |
| **Batch 5 Input Datasets Process** | -- | **7.225 seconds** | -- | -- |

---

## 3. Key Completed Deliverables

1. **GFortran Native Setup**: MinGW-w64 GFortran 16.1.0 installed on Windows. All Fortran source modules (`comdata.f90`, `fluc.f90`, `qInvariant.f90`, `prime.f90`, `avg_v1.7.f90`) re-compile natively.
2. **Unified Input Dataset Processor** ([`process_input_files.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/process_input_files.py)): Reads, parses, and calculates statistics across all 5 input datasets in `input/` (`3D.dat`, `Prime3D.dat`, `PrimeUV.dat`, `RMSu2v2w2t2uvvt.dat`, `rhouvwtpmu.dat`).
3. **Fortran vs CUDA Output Comparator** ([`compare_fortran_cuda.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/compare_fortran_cuda.py)): Evaluates field accuracy ($L_1, L_2, L_\infty$ error norms and Pearson correlation $r$) between Fortran and CUDA outputs.
4. **Live Fortran Re-compiler & Runner** ([`recompile_and_run_fortran.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/recompile_and_run_fortran.py)): Automatically deletes old outputs, compiles Fortran source code from scratch with `gfortran`, and measures live wall-clock execution timings.
5. **Live Dynamic Benchmark Suite** ([`benchmark_solvers.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/benchmark_solvers.py)): Measures real GPU execution time live using `time.perf_counter()` and compares with Fortran timings.
6. **Documentation & Tutorial**: Comprehensive [`TUTORIAL.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/TUTORIAL.md) and updated [`README.md`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/README.md).
7. **Cleaned Repository**: Removed unused/broken C++ draft files (`run_solver.py`, `post_process.py`, `bindings.cpp`, `CMakeLists.txt`, `src/`, `include/`) and created [`.gitignore`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/.gitignore).

---

## 4. Quick Command Cheatsheet

```powershell
# 1. Process all 5 input datasets in input/
python process_input_files.py --input-dir input --output-dir output_processed

# 2. Re-compile Fortran source code from scratch, delete old outputs, and calculate live
python recompile_and_run_fortran.py

# 3. Run dynamic execution benchmark measuring real GPU vs Fortran step times
python benchmark_solvers.py --steps 50

# 4. Run CUDA GPU solver for 100 RK2 time steps
python "cuda code.py" --steps 100 --output output_cuda/3D_cuda.dat

# 5. Compare Fortran and CUDA solution field outputs and error metrics
python compare_fortran_cuda.py --fortran-dir output_fortran --cuda-dir output_cuda --output-dir output_comparison
```

---

## 5. Artifact Output Files Reference

- **Processed Input Statistics**: `output_processed/input_processing_summary.json`
- **Extracted Mean Profile CSV**: `output_processed/extracted_mean_profiles.csv`
- **Extracted RMS Profile CSV**: `output_processed/extracted_rms_profiles.csv`
- **Fortran vs CUDA Field Comparison Markdown**: `output_comparison/comparison_report.md`
- **Fortran vs CUDA Field Comparison JSON**: `output_comparison/comparison_summary.json`
- **Re-compiled Fortran Live Timings JSON**: `output_comparison/recompiled_fortran_timings.json`
