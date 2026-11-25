# ⌨️ Complete CLI Commands Reference Guide

This reference document compiles all runnable command-line invocations, operational modes, and parameter configurations for the two primary simulation drivers:
1. [`python/cudacomplete.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/cudacomplete.py) — All-in-One Simulation, Checkpointing & High-Frequency Diagnostics Orchestrator
2. [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py) — Live Execution Benchmark, Fortran Parity & Turbulence Statistics Engine

---

## 📌 Table of Contents
1. [⚙️ Execution Environments & Path Notes](#1-️-execution-environments--path-notes)
2. [⚡ Commands for `python/cudacomplete.py`](#2--commands-for-pythoncudacompletepy)
   - [A. Solver Execution & Simulation Runs](#a-solver-execution--simulation-runs)
   - [B. Diagnostic Extraction on Saved Snapshots](#b-diagnostic-extraction-on-saved-snapshots)
   - [C. Complete CLI Arguments Reference](#c-complete-cli-arguments-reference)
3. [⏱️ Commands for `python/benchmark_solvers.py`](#3-️-commands-for-pythonbenchmark_solverspy)
   - [A. Benchmarking & Performance Runs](#a-benchmarking--performance-runs)
   - [B. Custom Destination Outputs](#b-custom-destination-outputs)
   - [C. Complete CLI Arguments Reference](#c-complete-cli-arguments-reference-1)
4. [💡 Workflow Selection Guide: Which Script to Use?](#4--workflow-selection-guide-which-script-to-use)

---

## 1. ⚙️ Execution Environments & Path Notes

Prepend the appropriate Python executable based on your active operating environment:

* **Windows PowerShell (with activated `.venv`)**:
  ```powershell
  python python/cudacomplete.py [flags]
  ```
* **Windows PowerShell (explicit virtual environment binary)**:
  ```powershell
  .\.venv\Scripts\python.exe python/cudacomplete.py [flags]
  ```
* **Linux / WSL2 (with activated environment)**:
  ```bash
  python3 python/cudacomplete.py [flags]
  ```
* **Google Colab (isolated container venv)**:
  ```bash
  !LD_LIBRARY_PATH=/content/py312_env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/content/py312_env/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH \
   /content/py312_env/bin/python python/cudacomplete.py [flags]
  ```

---

## 2. ⚡ Commands for `python/cudacomplete.py`

[`python/cudacomplete.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/cudacomplete.py) is the primary driver for simulation production. It advances the simulation through explicit RK2 stages, streams scalar reductions (`KE.dat`, `rhobulk.dat`, `Ubulk.dat`, `massflow.dat`, `tauWall.dat`) at every step, and writes periodic 3D state checkpoints.

### A. Solver Execution & Simulation Runs

#### 1. Standard Quick Run (50 steps default)
Advances the flow field by 50 time steps, writes the final state to `output_cuda/3D.dat`, and appends diagnostics:
```powershell
python python/cudacomplete.py --steps 50
```

#### 2. Production Run with Periodic Checkpointing
Advances the simulation for 10,000 steps, generating resumable intermediate snapshots (`3D_step1000.dat`, `3D_step2000.dat`, ...) every 1,000 steps:
```powershell
python python/cudacomplete.py --steps 10000 --save-interval 1000 --output output_cuda/3D.dat
```

#### 3. Seamless Resumption from Last Checkpoint
Resumes execution from `output_cuda/3D.dat` (retaining the checkpoint's iteration index and physical time) and continues for an additional 5,000 steps:
```powershell
python python/cudacomplete.py --steps 5000 --save-interval 1000 --resume
```

#### 4. High-Throughput Simulation (Skip Heavy 527 MB `fluc.dat`)
Generates 1D mean and RMS profiles without incurring the disk I/O cost of writing the large 3D fluctuation field:
```powershell
python python/cudacomplete.py --steps 2000 --save-interval 500 --no-fluc
```

#### 5. Pure Solver Timing Run (No Checkpoint Writing)
Measures raw GPU execution speed with full streaming diagnostics, skipping 3D disk writes:
```powershell
python python/cudacomplete.py --steps 500 --no-save
```

---

### B. Diagnostic Extraction on Saved Snapshots

#### 6. Single-Snapshot Diagnostic Extraction
Computes volume and wall reductions for an existing 3D Tecplot snapshot without running time steps:
```powershell
python python/cudacomplete.py --diagnostics-input output_cuda/3D.dat
```

#### 7. Batch Diagnostic Extraction Across Checkpoints
Scans a target directory, processes all checkpoints matching `--pattern`, and appends each state to the continuous time series files:
```powershell
python python/cudacomplete.py --diagnostics-dir output_cuda --pattern "3D_step*.dat"
```

#### 8. CPU Fallback Diagnostics
Extracts diagnostics using vectorized NumPy operations when an NVIDIA GPU is not available on the post-processing node:
```powershell
python python/cudacomplete.py --diagnostics-dir output_cuda --no-gpu
```

#### 9. Standalone Turbulence Statistics Extraction
Calculates fluctuating velocity fields $q'(x, y, z)$ and wall-normal profiles ($\bar{u}(z), u'_{rms}(z), k(z)$) from an existing snapshot:
```powershell
python python/cudacomplete.py --post-only --input output_cuda/3D.dat --mean-file output/mean.dat
```

---

### C. Complete CLI Arguments Reference (`cudacomplete.py`)

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--steps` | `int` | `50` | Total number of GPU solver steps to execute. |
| `--input` | `path` | `input/3D.dat` | Path to starting 3D flow state snapshot. |
| `--output` | `path` | `output_cuda/3D.dat` | Destination path for the final 3D solution field. |
| `--mean-file` | `path` | `output/mean.dat` | Reference 1D mean profile for fluctuation calculation. |
| `--save-interval`, `--checkpoint-interval` | `int` | `None` | Save a resumable checkpoint every $N$ steps (`3D_stepN.dat`). |
| `--resume` | `flag` | `False` | Resume from `--output` checkpoint instead of `input/3D.dat`. |
| `--post-only` | `flag` | `False` | Skip solver steps; compute fluctuations and profiles only. |
| `--no-save` | `flag` | `False` | Disable writing the final 3D output checkpoint to disk. |
| `--no-fluc` | `flag` | `False` | Disable exporting the large 3D fluctuation field `fluc.dat`. |
| `--no-stats` | `flag` | `False` | Disable exporting 1D profiles (`rms.dat`, `mean.dat`, `turb_KE.dat`). |
| `--diagnostics-input`, `-i` | `path` | `None` | Compute diagnostics for one existing 3D snapshot. |
| `--diagnostics-dir`, `-d` | `dir` | `None` | Process a directory of snapshots matching `--pattern`. |
| `--pattern` | `str` | `"3D_step*.dat"` | Filename glob pattern used with `--diagnostics-dir`. |
| `--output-dir` | `dir` | `output_cuda` | Destination directory for diagnostic time-series files. |
| `--no-gpu` | `flag` | `False` | Fall back to CPU vectorized routines for diagnostic reduction. |

---

## 3. ⏱️ Commands for `python/benchmark_solvers.py`

[`python/benchmark_solvers.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py) is tailored for benchmarking, performance profiling, timing comparisons against the legacy 4-core Fortran CPU solver, and export of machine-readable benchmark JSON summaries.

### A. Benchmarking & Performance Runs

#### 1. Standard Benchmark Run (50 Steps)
Measures dynamic wall-clock execution time per step using `time.perf_counter()`, compares against Fortran baseline (0.4753 s/step), and outputs summary metrics:
```powershell
python python/benchmark_solvers.py --steps 50
```

#### 2. Quick Pipeline Sanity Benchmark (10 Steps)
Rapidly validates CUDA JIT compilation and device memory allocation without running long iterations:
```powershell
python python/benchmark_solvers.py --steps 10
```

#### 3. Long Benchmark with Checkpointing
Benchmarks 1,000 steps with intermediate checkpoints saved every 250 steps:
```powershell
python python/benchmark_solvers.py --steps 1000 --save-interval 250 --output output_cuda/3D_cuda.dat
```

#### 4. Resuming Benchmark from Checkpoint
Resumes timing and state progression from a previously saved checkpoint:
```powershell
python python/benchmark_solvers.py --steps 500 --resume --output output_cuda/3D_cuda.dat
```

#### 5. Zero-I/O Kernel Benchmark
Eliminates all disk write operations to measure pure mathematical kernel throughput:
```powershell
python python/benchmark_solvers.py --steps 200 --no-save --no-fluc
```

---

### B. Custom Destination Outputs

#### 6. Explicit Custom File Routing
Routes every individual output file to specific destination names:
```powershell
python python/benchmark_solvers.py --steps 100 `
  --output output_cuda/custom_3D.dat `
  --fluc-output output_cuda/custom_fluc.dat `
  --rms-output output_cuda/custom_rms.dat `
  --mean-output output_cuda/custom_mean.dat `
  --tke-output output_cuda/custom_tke.dat
```

#### 7. Post-Processing Verification
Calculates error metrics and turbulence statistics from an existing solution file:
```powershell
python python/benchmark_solvers.py --post-only --input output_cuda/3D.dat --mean-file output/mean.dat
```

---

### C. Complete CLI Arguments Reference (`benchmark_solvers.py`)

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--steps` | `int` | `50` | Number of GPU steps to benchmark. |
| `--input` | `path` | `input/3D.dat` | Path to starting 3D checkpoint. |
| `--output` | `path` | `output_cuda/3D.dat` | Path to save final 3D solution field. |
| `--mean-file` | `path` | `output/mean.dat` | 1D wall-normal mean profile used for fluctuations. |
| `--save-interval` | `int` | `None` | Save intermediate checkpoint every $N$ steps (`3D_stepN.dat`). |
| `--resume` | `flag` | `False` | Resume from existing checkpoint in `--output`. |
| `--post-only` | `flag` | `False` | Skip solver steps; calculate statistics from `--input` only. |
| `--no-save` | `flag` | `False` | Disable saving the 3D output checkpoint. |
| `--no-fluc` | `flag` | `False` | Disable exporting the 3D fluctuation field `fluc.dat`. |
| `--no-stats` | `flag` | `False` | Disable writing 1D profiles (`rms.dat`, `mean.dat`, `turb_KE.dat`). |
| `--fluc-output` | `path` | `output_cuda/fluc.dat` | Custom output path for 3D fluctuation field. |
| `--rms-output` | `path` | `output_cuda/rms.dat` | Custom output path for 1D RMS profile. |
| `--mean-output` | `path` | `output_cuda/mean.dat` | Custom output path for 1D mean profile. |
| `--tke-output` | `path` | `output_cuda/turb_KE.dat`| Custom output path for 1D TKE profile. |

---

## 4. 💡 Workflow Selection Guide: Which Script to Use?

* **Use `python/cudacomplete.py` for standard DNS simulations**:
  - Automatically manages continuous time-series diagnostic streaming (`KE.dat`, `rhobulk.dat`, `Ubulk.dat`, `massflow.dat`, `tauWall.dat`).
  - Supports periodic resumable checkpoints (`--save-interval`).
  - Capable of batch post-processing existing directories of checkpoints (`--diagnostics-dir`).
* **Use `python/benchmark_solvers.py` for performance profiling**:
  - Focuses on wall-clock timing accuracy and throughput benchmarking.
  - Automatically compares GPU execution rates against the Fortran 90 4-core CPU baseline.
  - Generates machine-readable `benchmark_results.json` performance logs.
