# 📖 Complete Hands-On Tutorial & User Guide

Welcome to the comprehensive, module-by-module user guide for the **3D Compressible Supersonic Turbulent Channel Flow DNS Solver**. This tutorial covers installation, execution of the main Python CUDA solver, batch data processing, native Fortran re-compilation, live execution benchmarking, and post-processing field analysis.

---

## 📋 Prerequisites & Installation

### System Requirements
- **Operating System**: Windows 10/11 64-bit or Linux (x86_64).
- **GPU Hardware**: NVIDIA CUDA-capable GPU (Compute Capability 3.5+, e.g., GTX 1650 or higher).
- **Fortran Compiler**: MinGW-w64 `gfortran` 16.1.0 or higher.
- **Python Version**: Python 3.9 – 3.12.

### Python Environment Installation
Execute the following PowerShell command to install required scientific and CUDA packages:

```powershell
pip install numpy numba numba-cuda nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 cuda-bindings<13 pandas scipy matplotlib
```

---

## 🛠️ Module 1: Environment Setup & CUDA Device Verification

Verify that your Python installation detects the NVIDIA GPU and Numba CUDA runtime properly:

```python
import numba.cuda

# Check CUDA availability
if numba.cuda.is_available():
    device = numba.cuda.get_current_device()
    print(f"✅ CUDA Hardware Detected: {device.name.decode('utf-8')}")
    print(f"   Compute Capability : {device.compute_capability}")
    print(f"   Total MultiProcessors: {device.MULTIPROCESSOR_COUNT}")
else:
    print("❌ No CUDA GPU detected. Please verify NVIDIA graphics drivers.")
```

---

## 🚀 Module 2: Executing the Main Python Numba CUDA DNS Solver

The main GPU solver script is [`cuda code.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/cuda%20code.py). It automatically reads parameter settings from [`param.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/param.dat) and 3D initial condition grid from [`input/3D.dat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/input/3D.dat).

### Command Options

1. **Standard Solver Run (100 Time Steps)**:
   ```powershell
   python "cuda code.py" --steps 100 --output output_cuda/3D_cuda.dat
   ```

2. **Custom Input Dataset & Higher Step Count**:
   ```powershell
   python "cuda code.py" --input input/3D.dat --steps 500 --output output_cuda/3D_cuda.dat
   ```

3. **Synthetic Demo Mode (Validation Test without 500MB file load)**:
   ```powershell
   python "cuda code.py" --demo --steps 20 --output output_cuda/demo.dat
   ```

---

## 📊 Module 3: Processing & Summarizing Raw Input Datasets

To process, compute statistics, and extract profile summaries from all 5 raw datasets in `input/` (`3D.dat`, `Prime3D.dat`, `PrimeUV.dat`, `RMSu2v2w2t2uvvt.dat`, `rhouvwtpmu.dat`), execute:

```powershell
python python/process_input_files.py --input-dir input --output-dir output_processed
```

### Generated Artifacts
- `output_processed/input_processing_summary.json`: Detailed JSON report containing bulk density $\rho_b$, bulk velocity $U_b$, wall shear stress $\tau_w$, skin friction $C_f$, and peak turbulence kinetic energy $k_{\text{max}}$.
- `output_processed/extracted_mean_profiles.csv`: Extracted 1D mean velocity $\bar{u}(z)$, temperature $\bar{T}(z)$, and density $\bar{\rho}(z)$ profile CSV.
- `output_processed/extracted_rms_profiles.csv`: Extracted 1D RMS fluctuation profile CSV.

---

## ⚙️ Module 4: Live Re-compilation & Execution of Native Fortran Binaries

To clean-rebuild all Fortran source modules (`codes1/` and `codes/`) natively using `gfortran`, delete stale outputs, and measure wall-clock compilation and step timings:

```powershell
python python/recompile_and_run_fortran.py
```

### What This Script Does
1. Compiles `codes/fluc.f90` $\rightarrow$ `codes/fluc_win.exe`.
2. Compiles `codes/qInvariant.f90` $\rightarrow$ `codes/qInvariant_win.exe`.
3. Compiles `codes/prime.f90` $\rightarrow$ `codes/prime_win.exe`.
4. Compiles `codes/avg_v1.7.f90` $\rightarrow$ `codes/avg_v1.7_win.exe`.
5. Executes native binaries and exports timing statistics to `output_comparison/recompiled_fortran_timings.json`.

---

## ⚡ Module 5: Running Live Solver Benchmarks & Verification

### 1. Live Step Time Benchmark (Fortran MPI vs Python Numba CUDA)
Measure real GPU per-step execution time live and extrapolate speedup performance across 1, 10, 50, 100, 500, and 1,000 steps:

```powershell
python python/benchmark_solvers.py --steps 50
```

### 2. Quantitative Field Accuracy Verification
Compare Fortran reference outputs (`output_fortran/3D_fortran.dat`) against CUDA GPU outputs (`output_cuda/3D_cuda.dat`) to evaluate $L_1, L_2, L_\infty$ error norms and Pearson correlation coefficients ($r$):

```powershell
python python/compare_fortran_cuda.py --fortran-dir output_fortran --cuda-dir output_cuda --output-dir output_comparison
```

---

## 🔬 Module 6: Post-Processing Field Analysis

Compare execution speeds of post-processing analysis routines (Vortex $Q$-Criterion and Velocity Fluctuation fields) between re-compiled Fortran binaries and vectorized Python/CUDA processors:

```powershell
python python/compare_processing_times.py
```

### Code Example: Vortex $Q$-Criterion Tensor Computation

```python
import numpy as np

def compute_q_criterion_3d(u, v, w, dx, dy, dz):
    """
    Computes Second Invariant of Velocity Gradient Tensor Q across 3D flow field.
    Q = 0.5 * (||Omega||^2 - ||S||^2)
    """
    # Compute velocity gradient tensor components
    dudx, dudy, dudz = np.gradient(u, dx, dy, dz)
    dvdx, dvdy, dvdz = np.gradient(v, dx, dy, dz)
    dwdx, dwdy, dwdz = np.gradient(w, dx, dy, dz)
    
    # Strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    S11 = dudx; S22 = dvdy; S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)
    
    # Rotation rate tensor Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
    O12 = 0.5 * (dudy - dvdx)
    O13 = 0.5 * (dudz - dwdx)
    O23 = 0.5 * (dvdz - dwdy)
    
    norm_S_sq = 2 * (S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2))
    norm_O_sq = 4 * (O12**2 + O13**2 + O23**2)
    
    Q = 0.5 * (norm_O_sq - norm_S_sq)
    return Q
```

---

## ❓ Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: Numba CUDA throws `CudaSupportError: No CUDA GPUs found`
- **Solution**: Ensure your system has an active NVIDIA graphics card with updated display drivers (`nvidia-smi`). If using PyTorch/Conda environment, verify `CUDA_HOME` is set properly in environment variables.

### Q2: `gfortran` command is not recognized on Windows
- **Solution**: Install MinGW-w64 (GFortran 16.1.0+) and add `C:\mingw64\bin` to your system `PATH` environment variable.

### Q3: Memory Error when loading `input/3D.dat`
- **Solution**: The 3D grid file is ~527 MB ASCII text. Ensure your system has at least 8 GB Host RAM free. Use Pandas C-engine (`pd.read_csv(..., engine='c')`) which is 1.72× faster than standard Python file loaders.
