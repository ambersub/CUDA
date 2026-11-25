# 🚀 Cross-Device Setup & Cloud Execution Guide

This comprehensive guide provides end-to-end installation, configuration, and verification procedures for the **3D Compressible Supersonic Turbulent Channel Flow DNS CUDA Solver**. It covers local workstations (Windows 10/11, Ubuntu/Linux, WSL2, HPC clusters) and cloud environments (Google Colab with NVIDIA GPUs).

All automated configuration scripts and templates are housed in the [`setup/`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/setup) directory.

---

## 📌 Table of Contents
1. [⚡ Quick Start (Automated 1-Click Local Setup)](#1--quick-start-automated-1-click-local-setup)
2. [🖥️ Hardware & Software Requirements](#2-️-hardware--software-requirements)
3. [📋 Step-by-Step Local Setup Walkthrough](#3--step-by-step-local-setup-walkthrough)
   - [Step 1: Clone Repository & Dataset Placement](#step-1-clone-repository--dataset-placement)
   - [Step 2: Verify NVIDIA Drivers](#step-2-verify-nvidia-drivers)
   - [Step 3: Setup Python 3.10–3.12 Virtual Environment](#step-3-setup-python-310312-virtual-environment)
   - [Step 4: Install Dependencies with Version Pins](#step-4-install-dependencies-with-version-pins)
   - [Step 5: Setup GFortran Toolchain (Optional)](#step-5-setup-gfortran-toolchain-optional)
   - [Step 6: Run Automated Diagnostics](#step-6-run-automated-diagnostics)
   - [Step 7: Run Sanity Verification Simulation](#step-7-run-sanity-verification-simulation)
4. [☁️ Google Colab Cloud Execution Guide](#4-️-google-colab-cloud-execution-guide)
   - [Colab Runtime Prerequisites](#colab-runtime-prerequisites)
   - [1-Click All-in-One Setup & Verification Cell](#1-click-all-in-one-setup--verification-cell)
   - [Step-by-Step Colab Execution Cells (Cells 1 to 8)](#step-by-step-colab-execution-cells-cells-1-to-8)
5. [🛠️ Pre-Flight Verification Checklist](#5-️-pre-flight-verification-checklist)
6. [❓ Troubleshooting & Known Solutions](#6--troubleshooting--known-solutions)

---

## 1. ⚡ Quick Start (Automated 1-Click Local Setup)

Automated scripts create the virtual environment, install the exact pinned CUDA stack, initialize configuration files, and verify GPU acceleration automatically.

### On Windows (PowerShell or Command Prompt)
Run from the repository root:
```powershell
# PowerShell:
powershell -ExecutionPolicy Bypass -File .\setup\setup_windows.ps1

# Or Command Prompt / double-click:
.\setup\setup_windows.bat
```

### On Linux / Ubuntu / WSL2
From the repository root:
```bash
chmod +x setup/setup_linux.sh
./setup/setup_linux.sh
```

---

## 2. 🖥️ Hardware & Software Requirements

| Component | Minimum Requirement | Recommended Specification |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GPU with Compute Capability $\ge 6.0$ (Pascal or newer) | NVIDIA GeForce GTX 1650, RTX 20/30/40/50 series, Tesla T4, V100, A100 |
| **GPU VRAM** | 4 GB Dedicated VRAM | 6 GB+ Dedicated VRAM (for full $160 \times 130 \times 130 = 2.7\text{M}$ grid) |
| **NVIDIA Driver** | R535+ (CUDA 12 compatible) | Latest Game Ready / Studio / Data Center Driver (CUDA 12.x+) |
| **Operating System** | Windows 10/11 (64-bit) or Ubuntu 20.04+ / Linux / WSL2 | Windows 11 (64-bit) or Ubuntu 22.04 / 24.04 LTS |
| **Host System RAM** | 8 GB RAM | 16 GB+ RAM (for parsing 500 MB ASCII initial datasets) |
| **Python** | Python 3.10, 3.11, or 3.12 (64-bit) | Python 3.12 (64-bit) |
| **Fortran Compiler** | *(Optional)* GFortran 10+ | MinGW-w64 GFortran 16.1.0 (Windows) or GCC GFortran (Linux) |

---

## 3. 📋 Step-by-Step Local Setup Walkthrough

### Step 1: Clone Repository & Dataset Placement

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ambersub/CUDA.git uc_Re3k_Ma15
   cd uc_Re3k_Ma15
   ```

2. **Initialize Simulation Configuration (`param.dat`)**:
   ```bash
   # Windows:
   copy setup\param.dat.example param.dat

   # Linux / WSL2:
   cp --update=none setup/param.dat.example param.dat
   ```

3. **Dataset Setup**:
   - **Full DNS Run**: Ensure `input/3D.dat` (502.9 MB) is present in `input/`.
   - **Quick Testing / CI**: If large datasets are not present, test immediately with `--demo` mode without downloading `input/3D.dat`.

---

### Step 2: Verify NVIDIA Drivers

Ensure the NVIDIA GPU driver is active and recognized:
```bash
nvidia-smi
```
*Expected output: Displays Driver Version (e.g. `535.x`–`580.x`) and CUDA Version (e.g. `12.x`).*

---

### Step 3: Setup Python 3.10–3.12 Virtual Environment

```bash
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / Ubuntu / WSL2:
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 4: Install Dependencies with Version Pins

> ⚠️ **CRITICAL DEPENDENCY RULE**: `numpy<2` is strictly required. NumPy 2.x removed `np.row_stack`, which causes Numba CUDA JIT compilation to fail with `AttributeError: module 'numpy' has no attribute 'row_stack'`.

Install the full dependency stack:
```bash
python -m pip install --upgrade pip
pip install "numpy<2" numba numba-cuda \
    nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
    "cuda-bindings<13" pandas scipy matplotlib matplotlib-inline tqdm
```

Alternatively, install from the provided requirements file:
```bash
pip install -r setup/requirements.txt
```

---

### Step 5: Setup GFortran Toolchain (Optional)

> ℹ️ **Note**: GFortran is only required if you want to re-compile the legacy Fortran 90 CPU solver or execute the CPU-vs-GPU speedup benchmark routines. The Python CUDA GPU solver runs completely standalone.

#### Windows (via WinGet):
```powershell
winget install BrechtSanders.WinLibs.MCF.UCRT
```
*After installation, restart your terminal to reload `PATH`.*

#### Linux / Ubuntu / WSL2:
```bash
sudo apt update && sudo apt install -y gfortran build-essential
```

---

### Step 6: Run Automated Diagnostics

Execute [`setup/check_setup.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/setup/check_setup.py) to validate all 6 system layers:
```bash
python setup/check_setup.py
```

The diagnostic performs:
1. **OS & Python**: Architecture (64-bit required) and Python version.
2. **Package Matrix**: Verifies `numpy<2`, `numba`, `numba.cuda`, `pandas`, `scipy`, `matplotlib`.
3. **GPU Hardware**: Identifies device name, compute capability (SM), and VRAM capacity.
4. **Micro CUDA JIT**: Runs an in-memory 1,024-thread GPU vector addition kernel to confirm JIT compilation works end-to-end.
5. **Toolchain Check**: Checks for `gfortran` presence.
6. **Workspace Structure**: Auto-creates missing directories (`output_cuda/`, `output_comparison/`, `output_processed/`, `plots/`).

---

### Step 7: Run Sanity Verification Simulation

Test GPU execution with a fast synthetic initial condition test (does not require `input/3D.dat`):
```bash
python "cuda code.py" --demo --steps 20 --output output_cuda/demo.dat
```

*Expected output:*
```text
CUDA device: NVIDIA GeForce GTX 1650 (or similar)
Using synthetic demo initial condition.
step=      20  time=2.000000e-03  mean(rho)=1.000000e+00
CUDA solver finished: executed 20 steps in ~4.5 s
```

---

## 4. ☁️ Google Colab Cloud Execution Guide

### Colab Runtime Prerequisites

1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Upload the project folder `uc_Re3k_Ma15` to your Google Drive at `My Drive/uc_Re3k_Ma15`.
3. **Switch to GPU Runtime**:
   - Menu: **Runtime** → **Change runtime type**.
   - Hardware accelerator: Select **T4 GPU**.
   - Click **Save**.

---

### 1-Click All-in-One Setup & Verification Cell

Paste and run this single cell in Colab to mount Drive, create an isolated Python 3.12 environment, install the CUDA runtime, set library paths, and run diagnostics:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/uc_Re3k_Ma15

# 1. Install Python 3.12 and venv tools
!apt-get update -y -qq && apt-get install -y -qq python3.12 python3.12-venv python3.12-dev

# 2. Create isolated Python 3.12 virtual environment on container disk
!python3.12 -m venv /content/py312_env
!/content/py312_env/bin/pip install --upgrade pip -q

# 3. Install dependencies (numpy<2 is strictly required for Numba CUDA)
!/content/py312_env/bin/pip install -q "numpy<2" numba numba-cuda \
    nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
    "cuda-bindings<13" pandas scipy matplotlib matplotlib-inline tqdm

# 4. Initialize param.dat if missing
!cp --update=none setup/param.dat.example param.dat

# 5. Run diagnostic verification suite with CUDA library paths exported
!LD_LIBRARY_PATH=/content/py312_env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/content/py312_env/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH \
 MPLBACKEND=Agg /content/py312_env/bin/python setup/check_setup.py
```

---

### Step-by-Step Colab Execution Cells (Cells 1 to 8)

For interactive workflows, execute these individual cells in sequence:

#### Cell 1: Mount Drive & Navigate
```python
from google.colab import drive
import os

drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/uc_Re3k_Ma15')
print(f"Current working directory: {os.getcwd()}")
!ls -lh param.dat "cuda code.py" input/3D.dat
```

#### Cell 2: Check GPU Hardware Allocation
```python
!nvidia-smi
```

#### Cell 3: Setup Virtual Environment & Install Packages
```python
!apt-get update -y -qq && apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
!python3.12 -m venv /content/py312_env
!/content/py312_env/bin/pip install --upgrade pip -q
!/content/py312_env/bin/pip install -q "numpy<2" numba numba-cuda \
    nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
    "cuda-bindings<13" pandas scipy matplotlib matplotlib-inline tqdm
print("\n[+] Python 3.12 environment and CUDA dependencies installed successfully.")
```

#### Cell 4: Initialize Config & Run Diagnostics
```python
!cp --update=none setup/param.dat.example param.dat

%env LD_LIBRARY_PATH=/content/py312_env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/content/py312_env/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
%env MPLBACKEND=Agg

!/content/py312_env/bin/python setup/check_setup.py
```

#### Cell 5: Run Main CUDA GPU Solver
```python
# Run 100 explicit RK2 time-steps on the 2.7M-cell mesh
!/content/py312_env/bin/python "cuda code.py" --input input/3D.dat --steps 100 --output output_cuda/3D_cuda.dat
```
> 💡 *Synthetic Demo Alternative*: `!/content/py312_env/bin/python "cuda code.py" --demo --steps 20 --output output_cuda/demo.dat`

#### Cell 6: Run Live Benchmark & Turbulence Statistics
```python
!/content/py312_env/bin/python python/benchmark_solvers.py --steps 50 --output-dir output_cuda
```

#### Cell 7: Quantitative Accuracy Verification (Fortran vs CUDA)
```python
!/content/py312_env/bin/python python/compare_fortran_cuda.py --fortran-dir output_fortran --cuda-dir output_cuda --output-dir output_comparison
```

#### Cell 8: Generate & Display Publication Validation Plots
```python
# Generate validation plots from output_cuda/ checkpoints
!/content/py312_env/bin/python python/plot_validation.py --cuda-dir output_cuda --output-dir plots/validation_plots

# Display validation plots inline
from IPython.display import Image, display
display(Image('plots/validation_plots/pmean_valid.png'))
display(Image('plots/validation_plots/tmean_valid.png'))
```

---

## 5. 🛠️ Pre-Flight Verification Checklist

- [ ] Repository cloned and working directory set to root
- [ ] `nvidia-smi` detects active NVIDIA GPU and driver $\ge 535$
- [ ] Python 3.10–3.12 64-bit environment active
- [ ] Pinned dependencies installed (`numpy<2` strictly enforced)
- [ ] `param.dat` exists in repository root
- [ ] `python setup/check_setup.py` outputs `[SUCCESS] ALL REQUIRED SYSTEMS GO!`
- [ ] `python "cuda code.py" --demo --steps 20` finishes successfully
- [ ] `input/3D.dat` verified (for full DNS simulations)

---

## 6. ❓ Troubleshooting & Known Solutions

| Issue / Error | Root Cause | Exact Solution |
| :--- | :--- | :--- |
| `AttributeError: module 'numpy' has no attribute 'row_stack'` | NumPy 2.x installed. Numba CUDA expects `np.row_stack` (deprecated in NumPy 2.0). | Force NumPy 1.x: `pip install "numpy<2"`. |
| `Failure finding libcudart.so.12` | Runtime CUDA shared library missing from system library search path. | Install `nvidia-cuda-runtime-cu12` and export:<br>`export LD_LIBRARY_PATH=$VENV_DIR/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH` |
| `ValueError: Key backend: 'module://matplotlib_inline...' is not a valid value` | Colab sets default `MPLBACKEND` to `matplotlib_inline`, missing inside custom venvs. | Install `matplotlib-inline` and set headless backend:<br>`export MPLBACKEND=Agg` |
| `cp: warning: behavior of -n is non-portable` | Ubuntu 24.04 GNU `cp` deprecated `-n` in favor of `--update=none`. | Use `cp --update=none source dest`. |
| `Numba CUDA cannot find libdevice.10.bc` | Numba cannot find NVVM bitcode math library. | Set environment variable: `NUMBA_CUDA_LIBDEVICE` to the path of `nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc`. |
| `UserWarning: cuda-bindings was built for CUDA 13...` | CUDA 13 bindings wheel installed on CUDA 12 driver. | Pin bindings: `pip install "cuda-bindings<13"`, or set `CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1`. |
| `OutOfMemoryError: CUDA out of memory` | Full domain ($2.7\text{M}$ cells) requires $\sim 3.2\text{ GB}$ VRAM. | Close browser GPU tabs/background processes, or ensure GPU has $\ge 4\text{ GB}$ dedicated VRAM. |
| `Execution of scripts is disabled on this system` (PowerShell) | Windows default security policy blocks script execution. | Run in PowerShell: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. |
