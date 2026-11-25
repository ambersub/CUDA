# Handoff: Fortran-to-CUDA/Python CFD Solver & Analysis Pipeline

## 1. Project Summary

The legacy Fortran 90 MPI supersonic turbulent channel flow solver (`codes/` v1.9 uncontrolled and `codes1/` v2.0 opposition-controlled/actuated) and its post-processing/plotting tools have been replicated in Python/Numba CUDA across four phases, with automated numerical parity verification against compiled `gfortran`.

- **Environment**: Windows 11 (AMD64), Python 3.12.10 (`.venv`), NumPy 1.26.4, Pandas 3.0.6, SciPy 1.17.1, Matplotlib 3.11.2, Numba 0.68.0 (`numba_cuda`), NVIDIA GeForce GTX 1650 (4 GB VRAM, SM 7.5), GNU Fortran 16.1.0 (`gfortran`).
- **Verification Status**: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` -> **Ran 22 tests in 77.811s — OK**; `.\.venv\Scripts\python.exe setup\check_setup.py --test-gpu` -> **ALL REQUIRED SYSTEMS GO!**

---

## 2. Phase-by-Phase Architecture & Parity Matrix

| Phase | Fortran Source(s) | Python / CUDA Target(s) | Key Capabilities & Parity |
|---|---|---|---|
| **Phase 1: Core Solver & BCs** | `codes/main_v1.9_unc.f90`, `codes1/mod_var_dec.f90`, `mod_conv.f90`, `mod_diff.f90`, `mod_bc.f90`, `mod_control.f90`, `mod_in_out.f90` | [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py) | `param.dat` v1.9/v2.0 parsing, CLI override precedence, `cosine`/`tanh` auto-detection, 2nd/3rd-order wall stencils, periodic halo wrapping, convective Peclet (`Re`, `jac[k]`), race-free OC + wall pressure/state BCs, dynamic GPU `tau_lower/tau_upper`/`d_rho_mean`, `resOp=2` component restart loading (`domInfo.txt` + `3Dnnn.dat`). |
| **Phase 2: Live Diagnostics, Stats & Checkpoints** | `codes1/mod_stats.f90`, `codes/main_v1.9_unc.f90` (`write_output_files`, `Diagnostic_files`, `tauWall_write`) | [`python/compute_diagnostics_cuda.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/compute_diagnostics_cuda.py), [`cuda code.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/cuda%20code.py), [`python/benchmark_solvers.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/benchmark_solvers.py), [`python/cudacomplete.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/cudacomplete.py) | Zero-3D-copy GPU reductions (`LiveStatsWorkspace`, `compute_live_step_diagnostics`, `compute_spatial_stats_gpu`) + CPU fallback (`compute_spatial_stats_cpu`), streaming `KE.dat`, `rhobulk.dat`, `Ubulk.dat`, `massflow.dat`, `FFT.dat`, `tauWall.dat`, `moment1st/2nd/Joint`, `moment1stT/2ndT/JointT`, `Omega1st/2nd`, `Cf1st/2nd/Joint`, `Favre1st/2nd/Joint`, plus `"full"` mode streams (`power_OC`, `power_STWSV`, `power`, `uv_slip`, `tauWall_Lxt1/2`, `moment80`, `momentxP*`, `TKE_xt_*`, `xz/*`, `NU/*`). Rolling combined checkpoints (`3D.dat` <-> `3D_.dat`) and partitioned component checkpoints (`domInfo.txt` + `3Dnnn.dat`), with JIT-warmup isolation in benchmark runners. |
| **Phase 3: Post-Processing Suite** | `codes/avg_v1.7.f90`, `avgTauWall.f90`, `combine.f90`, `fluc.f90`, `qInvariant.f90`; `codes1/avg_VD.f90`, `avg_FIK.f90`, `avg_v1.7_TKE_ALL.f90`, `avg_Favre.f90`, `avg_Omega.f90`, `avg_Cf.f90`, `qInvariant_Omega.f90` | [`python/postprocess_cuda.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/postprocess_cuda.py) | Restart-overlap deduplication (`deduplicate_time_series`), temporal averaging (`mean.dat`, `rms.dat`, `corr.dat`, `AoNDT.dat`, `tauWallAvg.dat`, extended TKE/Favre/Omega/Cf averages), Van Driest & Trettel-Larsson velocity transformations (`VD.dat`), FIK skin-friction decomposition (`FIK.dat`), component stitching (`combine_component_checkpoints`), 3D fluctuations (`fluc.dat`), and GPU/CPU Q-criterion & vorticity (`kernel_q_invariant`, `qInvariant.dat`). |
| **Phase 4: Validation Plotting & End-to-End Pipeline** | `codes/cmdGNUplot*.txt`, `codes1/cmdGNUplot*.txt`, CKM benchmark validation | [`python/plot_validation.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/python/plot_validation.py), [`tests/test_phase4.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/tests/test_phase4.py) | Headless Matplotlib (`Agg`) generation of CKM validation plots (`KE_valid.png`, `massflow_valid.png`, `rhobulk_valid.png`, `tauWall_valid.png`, `pmean_valid.png`, `rhomean_valid.png`, `tmean_valid.png`, `Trms_valid.png`) directly from live + post-processed CUDA outputs. |

---

## 3. Key Commands

### Run the CUDA Solver with Live Stats & Rolling Checkpoints
```powershell
# Combined checkpoint alternation (3D.dat <-> 3D_.dat) & standard live stats
.\.venv\Scripts\python.exe "cuda code.py" --steps 1000 --diag-freq 10 --istat 100 --stats-dir output_cuda --stats-mode standard --checkpoint-dir output_cuda --output-freq 1000

# Synthetic demo run with full mod_stats streams and partitioned component checkpoints
.\.venv\Scripts\python.exe "cuda code.py" --demo --nx 16 --ny 12 --nz 40 --oc 0 --steps 10 --diag-freq 2 --istat 2 --stats-dir output_cuda/demo_stats --stats-mode full --checkpoint-dir output_cuda/demo_ckpt --output-freq 10 --checkpoint-components
```

### Run Benchmark Runners (Warmup-Isolated)
```powershell
.\.venv\Scripts\python.exe python\benchmark_solvers.py --steps 50 --stats-dir output_cuda --stats-mode standard --diag-freq 10 --checkpoint-dir output_cuda --output-freq 50
.\.venv\Scripts\python.exe python\cudacomplete.py --steps 50 --stats-dir output_cuda --stats-mode standard --diag-freq 10
```

### Run Phase 3 Post-Processing
```powershell
# Stitch partitioned checkpoints, temporal-average with restart dedup, compute VD & FIK, 3D fluc, and GPU Q-invariant
.\.venv\Scripts\python.exe python\postprocess_cuda.py --stats-dir output_cuda --out-dir output_cuda --combine output_cuda/domInfo.txt --combine-out output_cuda/3D_combined.dat --avg --vd-fik --snapshot output_cuda/3D_combined.dat --fluc --q-invariant
```

### Run Phase 4 Validation Plotting
```powershell
.\.venv\Scripts\python.exe python\plot_validation.py --data-dir output_cuda --solver cuda --full-time --output-dir plots/validation_plots
```

### Run Full Test Suite & Environment Check
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe setup\check_setup.py --test-gpu
```

---

## 4. Test Suite Inventory

- [`tests/test_phase1.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/tests/test_phase1.py) + [`tests/fortran_ref_parity.f90`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/tests/fortran_ref_parity.f90): 14 tests covering `param.dat` parsing, CLI overrides, `cosine`/`tanh` mesh & stencil polynomial exactness, `resOp=1/2` restart loading, OC sensor & `amp_v` decoupling, `check_blown`, and compiled `gfortran` RK2 solver parity (`RelL2 < 1e-12`).
- [`tests/test_phase2.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/tests/test_phase2.py) + [`tests/fortran_stats_parity.f90`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/tests/fortran_stats_parity.f90): 5 tests covering GPU-vs-CPU spatial/diagnostic parity (`minimal`/`standard`/`full`), compiled `gfortran` `mod_stats.f90` parity (`RelL2 < 1e-12`, ASCII `F15.7` `< 2e-7`), live cadence streaming, rolling `3D.dat <-> 3D_.dat` and component `domInfo.txt` round-trips, and benchmark runner JIT-warmup isolation.
- [`tests/test_phase3.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/tests/test_phase3.py): 2 tests covering multi-restart deduplication, `avg_v1.7`/`avgTauWall`, `VD.dat`, `FIK.dat`, `combine`, `fluc.dat`, and GPU-vs-CPU `qInvariant.dat`.
- [`tests/test_phase4.py`](file:///c:/Users/Amber/Desktop/uc_Re3k_Ma15/tests/test_phase4.py): 1 end-to-end pipeline test (`CUDASolver` -> live stats + component checkpoint -> `postprocess_cuda.py` -> `plot_validation.py` PNG generation).

---

## 5. Operational Notes & Caveats

- **Coarse `cosine` + `wall_order=2` with `enable_oc=1`**: `SolverConfig` defaults to `enable_oc=1, opposition_distance=33` (for `codes1/` `nz >= 160` `tanh` production runs). When running uncontrolled `codes/` cases (`mesh_type="cosine"`, `wall_order=2`) or coarse grids (`nz=40`), pass `--oc 0` (or `enable_oc=0`).
- **Component checkpoint layout**: `checkpoint_components=True` writes `domInfo.txt` and `3D001.dat..3Dnnn.dat` using a balanced `(px, py, pz)` partition (`_choose_component_grid`) directly readable by `read_component_inputs()` and `combine_component_checkpoints()`.
- **Unused Fortran legacy stub**: `codes1/mod_stats.f90:vel_corr` contains an unimplemented `stop` stub in Fortran and is intentionally omitted.
