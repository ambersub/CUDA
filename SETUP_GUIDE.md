# 🚀 Cross-Device Setup & Installation Guide

This guide provides complete instructions for cloning, configuring, and verifying the **3D Compressible Turbulent Channel Flow DNS Solver** on a new device (Windows 10/11, Ubuntu/Linux, HPC Cluster, or WSL2).

---

## 📌 Table of Contents
1. [⚡ Quick Start (Automated 1-Click Setup)](#1--quick-start-automated-1-click-setup)
2. [🖥️ Hardware & Software Requirements](#2-️-hardware--software-requirements)
3. [📋 Step-by-Step Setup Walkthrough](#3--step-by-step-setup-walkthrough)
   - [Step 1: Clone Repository & Transfer Datasets](#step-1-clone-repository--transfer-datasets)
   - [Step 2: Verify NVIDIA Drivers](#step-2-verify-nvidia-drivers)
   - [Step 3: Setup Python Environment](#step-3-setup-python-environment)
   - [Step 4: Install Dependencies](#step-4-install-dependencies)
   - [Step 5: Setup GFortran Compiler (Optional)](#step-5-setup-gfortran-compiler-optional)
   - [Step 6: Run Automated Diagnostics](#step-6-run-automated-diagnostics)
   - [Step 7: Run Sanity Verification Simulation](#step-7-run-sanity-verification-simulation)
4. [🛠️ Verification Checklist](#4-️-verification-checklist)
5. [❓ Troubleshooting & Known Solutions](#5--troubleshooting--known-solutions)

---

## 1. ⚡ Quick Start (Automated 1-Click Setup)

Automated setup scripts are included in the repository to create the virtual environment, install dependencies, configure default parameters, and verify GPU acceleration automatically.

### On Windows (Command Prompt or PowerShell)
Double-click [`setup_windows.bat`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/setup_windows.bat) or execute in PowerShell:
```powershell
.\setup_windows.bat
# or in PowerShell:
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

### On Linux / Ubuntu / WSL2
```bash
chmod +x setup_linux.sh
./setup_linux.sh
```

---

## 2. 🖥️ Hardware & Software Requirements

| Component | Minimum Requirement | Recommended Specification |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GPU with Compute Capability $\ge 6.0$ (Pascal or newer) | NVIDIA GeForce GTX 1650, RTX 20/30/40/50 series, A100, V100, T4 |
| **GPU VRAM** | 4 GB Dedicated VRAM | 6 GB+ Dedicated VRAM (for $160 \times 130 \times 130$ DNS grid) |
| **NVIDIA Driver** | R535+ (CUDA 12 compatible) | Latest Game Ready / Studio / Data Center Driver |
| **Operating System** | Windows 10/11 (64-bit) or Ubuntu 20.04+ / Linux / WSL2 | Windows 11 (64-bit) or Ubuntu 22.04 LTS |
| **Host System RAM** | 8 GB RAM | 16 GB+ RAM (for parsing 500 MB ASCII initial datasets) |
| **Python** | Python 3.10, 3.11, or 3.12 (64-bit) | Python 3.12 (64-bit) |
| **Fortran Compiler** | *(Optional)* GFortran 10+ | MinGW-w64 GFortran 16.1.0 or GCC GFortran |

---

## 3. 📋 Step-by-Step Setup Walkthrough

### Step 1: Clone Repository & Transfer Datasets

1. **Clone the repository on the new device**:
   ```bash
   git clone https://github.com/ambersub/CUDA.git
   cd CUDA
   ```

2. **Initialize Simulation Config (`param.dat`)**:
   If `param.dat` is not present, create it from the provided template:
   ```bash
   # Windows:
   copy param.dat.example param.dat

   # Linux:
   cp param.dat.example param.dat
   ```

3. **Transfer Initial Condition Datasets**:
   > ⚠️ **Important**: Large raw flow dataset files (`input/3D.dat`, 527 MB) are ignored by git to keep repository operations lightweight.
   - **Option A (Full DNS Simulation)**: Copy the [`input/`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/input) folder from your primary device or shared drive into `CUDA/input/`.
   - **Option B (Synthetic Demo Test)**: If you do not have `3D.dat` immediately available, you can still test the solver using `--demo` mode (see Step 7).

---

### Step 2: Verify NVIDIA Drivers

Ensure the NVIDIA GPU driver is active and recognized:
```bash
nvidia-smi
```
*Expected output: Displays Driver Version (e.g. `576.52`) and CUDA Version (e.g. `12.x`).*

---

### Step 3: Setup Python Environment

Always use a 64-bit Python virtual environment:

#### Windows:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```
*(If you encounter a PowerShell script execution error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

#### Linux / WSL2:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 4: Install Dependencies

Install the tested scientific computing and Numba CUDA stack via [`requirements.txt`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/requirements.txt):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### What this installs:
- **`numpy<2.0.0`**: Multidimensional grid operations and array allocations.
- **`numba` & `numba-cuda`**: High-performance Python-to-PTX GPU JIT compilation.
- **`nvidia-cuda-nvcc-cu12` & `nvidia-cuda-nvrtc-cu12`**: In-tree CUDA compiler wheels providing `libdevice.10.bc` and runtime libraries without requiring a heavy standalone system CUDA SDK.
- **`pandas`**: Fast C-engine ASCII dataset parser.
- **`scipy`**: Pearson correlation statistics.
- **`matplotlib`**: Law-of-the-wall and turbulence profile validation plots.
- **`tqdm`**: Real-time simulation progress tracking.

---

### Step 5: Setup GFortran Compiler (Optional)

> ℹ️ **Note**: GFortran is only required if you want to re-compile the legacy Fortran 90 CPU solver or execute the CPU-vs-GPU speedup benchmark routines. The Python CUDA GPU solver runs completely standalone.

#### Windows:
Install MinGW-w64 GFortran in one command using Windows Package Manager (WinGet):
```powershell
winget install BrechtSanders.WinLibs.MCF.UCRT
```
*After installation, restart your terminal so `gfortran` is in your `PATH`.*

#### Linux / Ubuntu / WSL2:
```bash
sudo apt update && sudo apt install -y gfortran build-essential
```

---

### Step 6: Run Automated Diagnostics

Run [`check_setup.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/check_setup.py) to automatically validate all 6 components of your new device:

```bash
python check_setup.py
```

#### What the diagnostic checks:
1. **OS & Python**: Confirms 64-bit architecture and Python version.
2. **Package Matrix**: Verifies all required Python libraries.
3. **GPU Hardware**: Detects NVIDIA device name, compute capability (SM), and VRAM.
4. **Micro CUDA JIT**: Runs an in-memory 1,024-thread GPU vector addition kernel to confirm JIT compilation works end-to-end.
5. **Fortran Compiler**: Identifies `gfortran` toolchain presence.
6. **Workspace Structure**: Auto-creates required output folders (`output_cuda/`, `output_comparison/`, `output_processed/`, `plots/`).

*Expected result:*
```text
===========================================================================
 Summary & Verification Verdict
===========================================================================
 [SUCCESS] ALL REQUIRED SYSTEMS GO! This device is ready to execute simulations.
```

---

### Step 7: Run Sanity Verification Simulation

Test GPU execution with a fast synthetic initial condition test (does not require `input/3D.dat`):

```bash
python "cuda code.py" --demo --steps 20 --output output_cuda/demo.dat
```

*Expected output:*
```text
CUDA device: NVIDIA GeForce GTX 1650
Using synthetic demo initial condition.
step=      20  time=2.000000e-03  mean(rho)=1.000000e+00
CUDA solver finished: executed 20 steps in ~4.5 s
```

---

## 4. 🛠️ Verification Checklist

When transferring to a new device, check off each step:

- [ ] Repository cloned: `git clone https://github.com/ambersub/CUDA.git`
- [ ] `nvidia-smi` displays active NVIDIA GPU and driver version $\ge 535$
- [ ] Python 3.10–3.12 (64-bit) installed
- [ ] Virtual environment created and activated (`.venv`)
- [ ] `pip install -r requirements.txt` completed with 0 errors
- [ ] `param.dat` exists (copied from `param.dat.example`)
- [ ] `python check_setup.py` returns `[SUCCESS]`
- [ ] `python "cuda code.py" --demo --steps 20` runs and finishes successfully
- [ ] *(If running full simulations)* `input/3D.dat` copied into `input/` folder
- [ ] *(Optional)* `gfortran --version` verified

---

## 5. ❓ Troubleshooting & Known Solutions

### Problem 1: `Numba CUDA cannot find libdevice.10.bc`
- **Cause**: Numba cannot locate the math bitcode library installed by pip wheels.
- **Solution**: The solver [`cuda code.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/cuda%20code.py) and [`check_setup.py`](file:///c:/Users/Amber/Downloads/uc_Re3k_Ma15/check_setup.py) automatically scan `site-packages/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc` and set `NUMBA_CUDA_LIBDEVICE`. If needed manually, set in PowerShell:
  ```powershell
  $env:NUMBA_CUDA_LIBDEVICE = (python -c "import site, os; print(os.path.join(site.getsitepackages()[0], 'nvidia', 'cuda_nvcc', 'nvvm', 'libdevice', 'libdevice.10.bc'))")
  ```

### Problem 2: `UserWarning: cuda-bindings was built for CUDA major version 13...`
- **Cause**: NVIDIA recently published CUDA 13 bindings wheels, but your GPU driver is CUDA 12.
- **Solution**: Set environment variable `CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1` (already embedded in solver scripts) or pin in pip:
  ```bash
  pip install "cuda-bindings<13"
  ```

### Problem 3: `FileNotFoundError: param.dat`
- **Cause**: `.gitignore` excludes `*.dat` files by default.
- **Solution**: Copy `param.dat.example` to `param.dat`:
  ```bash
  copy param.dat.example param.dat
  ```

### Problem 4: `OutOfMemoryError: CUDA out of memory`
- **Cause**: The full 3D simulation domain ($160 \times 130 \times 130 = 2,704,000$ cells) requires approximately 3.2 GB of GPU VRAM.
- **Solution**: Close background applications using GPU memory (web browsers, 3D modeling tools) or run on a GPU with 4 GB or more dedicated VRAM.

### Problem 5: `Execution of scripts is disabled on this system` (PowerShell)
- **Cause**: Windows default PowerShell execution policy blocks `.venv\Scripts\Activate.ps1`.
- **Solution**: Enable script execution for your session:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

---

## 🎯 Production Execution Commands

Once setup is complete, you are ready to execute:

```powershell
# 1. Run dynamic live benchmark (measures real GPU step rate live):
python python/benchmark_solvers.py --steps 50

# 2. Run full 3D CUDA GPU simulation for 100 steps:
python "cuda code.py" --steps 100 --output output_cuda/3D_cuda.dat

# 3. Process all input datasets:
python python/process_input_files.py --input-dir input --output-dir output_processed

# 4. Compare Fortran vs CUDA outputs:
python python/compare_fortran_cuda.py --fortran-dir output_fortran --cuda-dir output_cuda --output-dir output_comparison
```
