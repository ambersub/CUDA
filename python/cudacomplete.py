#!/usr/bin/env python3
"""Combined CUDA solver, checkpointing, turbulence statistics, and diagnostics.

Run solver steps with resumable periodic checkpoints, compute integrated
diagnostics, or post-process a single or batch of existing Tecplot snapshots.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd

# Suppress minor version warnings for clean terminal output
os.environ.setdefault("CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING", "1")
warnings.filterwarnings("ignore")

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import compute_diagnostics_cuda as cdc

calculate_diagnostics_cuda = cdc.calculate_diagnostics_cuda
append_diagnostic_record = cdc.append_diagnostic_record
process_3d_snapshot = cdc.process_3d_snapshot
process_snapshot_series = cdc.process_snapshot_series
compute_wall_coefficients = cdc.compute_wall_coefficients


# ---------------------------------------------------------------------------
# Mean Profile Ingestion & Statistics Helpers
# ---------------------------------------------------------------------------

def load_mean_profile(
    mean_filepath: Optional[str] = "output/mean.dat",
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Reads 1D wall-normal mean profile from output/mean.dat (or user-specified path).
    Expected columns:
      0: z, 1: rho, 2: u, 3: v, 4: w, 5: E, 6: T, 7: P
    Returns:
      (mean_data_array, resolved_file_path)
    """
    candidates = []
    if mean_filepath:
        candidates.append(mean_filepath)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    candidates.extend([
        "output/mean.dat",
        os.path.join(repo_root, "output", "mean.dat"),
        "../output/mean.dat",
        "mean.dat",
        os.path.join(repo_root, "output_fortran", "mean.dat"),
        "output_fortran/mean.dat",
    ])

    resolved = None
    for cand in candidates:
        if cand and os.path.isfile(cand):
            resolved = os.path.abspath(cand)
            break

    if not resolved:
        print(f"    [!] Warning: Mean profile file '{mean_filepath}' not found in output directory.")
        return None, None

    try:
        # Load numeric lines using pandas engine
        df = pd.read_csv(
            resolved,
            sep=r"\s+",
            header=None,
            comment="#",
            engine="c",
            dtype=np.float64,
        )
        df = df.dropna()
        arr = df.to_numpy(dtype=np.float64)
        if arr.shape[1] < 8:
            print(f"    [!] Warning: '{resolved}' has {arr.shape[1]} columns, expected at least 8 (z, rho, u, v, w, E, T, P).")
            return None, None

        print(f"    [+] Successfully read mean profile from: {resolved}")
        print(f"        Grid Points: {arr.shape[0]} wall-normal levels | Columns: z, rho, u, v, w, E, T, P")
        return arr, resolved
    except Exception as e:
        print(f"    [!] Error reading mean profile from '{resolved}': {e}")
        return None, None


def save_fluc_file(
    filename: str,
    var_prime: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    time_val: float,
    nstep_val: int,
) -> float:
    """
    Exports the 3D fluctuation field to Tecplot format matching Fortran codes/fluc.f90:
    Header:
      TITLE ="<time> <nstep>"
      Variables ="x","y","z","rho_prime","U_prime","V_prime","W_prime","E_prime","T_prime","P_prime"
      ZONE k=nz,j=ny,i=nx,DATAPACKING=POINT
    """
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    nx, ny, nz = var_prime.shape[0], var_prime.shape[1], var_prime.shape[2]
    out_arr = np.empty((nz, ny, nx, 10), dtype=np.float64)

    # Reorder into (nz, ny, nx, 10) matching Fortran k -> j -> i convention
    for k in range(nz):
        out_arr[k, :, :, 0] = x1[None, :]
        out_arr[k, :, :, 1] = x2[:, None]
        out_arr[k, :, :, 2] = x3[k]
        for p in range(7):
            out_arr[k, :, :, 3 + p] = var_prime[:, :, k, p].T

    flat_data = out_arr.reshape(-1, 10)
    header = (
        f'TITLE ="{time_val:20.10f} {nstep_val:9d}"\n'
        f'Variables ="x","y","z","rho_prime","U_prime","V_prime","W_prime","E_prime","T_prime","P_prime"\n'
        f'ZONE k={nz},j={ny},i={nx},DATAPACKING=POINT'
    )
    fmt = " %10.6f %10.6f %10.6f " + " ".join(["%22.15e"] * 7)
    np.savetxt(filename, flat_data, fmt=fmt, header=header, comments="")
    return time.perf_counter() - t0


def compute_and_save_new_results(
    var_h: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    time_val: float,
    nstep_val: int,
    mean_data: np.ndarray,
    mean_source_path: str,
    output_dir: str = "output_cuda",
    save_fluc: bool = True,
    save_stats: bool = True,
    fluc_output: Optional[str] = None,
    rms_output: Optional[str] = None,
    mean_output: Optional[str] = None,
    tke_output: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Uses the reference mean profile (output/mean.dat) together with the simulated 3D flow state
    to obtain and export all new results:
      1. Fluctuating field: q' = q - mean_q(z)
      2. 3D fluctuation checkpoint: fluc.dat (matching codes/fluc.f90)
      3. 1D RMS turbulence profile: rms.dat (matching output/rms.dat)
      4. 1D New simulated mean profile: mean.dat (matching output/mean.dat)
      5. 1D Turbulent Kinetic Energy & Reynolds stresses: turb_KE.dat
      6. Quantitative error norms and correlation against the reference mean profile
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    nx, ny, nz, nvar = var_h.shape
    mean_z = mean_data[:, 0]
    mean_vals = mean_data[:, 1:8]  # cols 1 to 7: rho, u, v, w, E, T, P

    # 1. Match wall-normal layers (with interpolation fallback if grid size differs)
    if mean_data.shape[0] == nz:
        ref_mean = mean_vals
    else:
        print(f"    [!] Interpolating mean profile from {mean_data.shape[0]} points to {nz} simulation points...")
        ref_mean = np.empty((nz, 7), dtype=np.float64)
        for p in range(7):
            ref_mean[:, p] = np.interp(x3, mean_z, mean_vals[:, p])

    # Broadcast reference mean to (1, 1, nz, 7)
    ref_mean_bcast = ref_mean[None, None, :, :]

    # 2. Compute 3D Instantaneous Fluctuations (matching codes/fluc.f90)
    var_prime = var_h - ref_mean_bcast

    # 3. Compute 1D Simulated Mean Profile: <q>_{xy}
    sim_mean = np.mean(var_h, axis=(0, 1))  # (nz, 7)

    # 4. Compute 1D RMS Fluctuations Profile: sqrt(<(q - mean_ref)^2>_{xy})
    sim_rms = np.sqrt(np.mean(var_prime ** 2, axis=(0, 1)))  # (nz, 7)

    # 5. Compute Reynolds Stresses & Turbulent Kinetic Energy (TKE)
    # var indices: 0: rho, 1: u, 2: v, 3: w, 4: E, 5: T, 6: P
    uu = np.mean(var_prime[:, :, :, 1] ** 2, axis=(0, 1))
    vv = np.mean(var_prime[:, :, :, 2] ** 2, axis=(0, 1))
    ww = np.mean(var_prime[:, :, :, 3] ** 2, axis=(0, 1))
    uv = np.mean(var_prime[:, :, :, 1] * var_prime[:, :, :, 2], axis=(0, 1))
    vt = np.mean(var_prime[:, :, :, 2] * var_prime[:, :, :, 5], axis=(0, 1))
    tke = 0.5 * (uu + vv + ww)

    # 6. Quantitative Accuracy Comparison: Simulated Mean vs Reference Mean
    var_names = ["rho", "u", "v", "w", "E", "T", "P"]
    var_metrics = {}

    for p, name in enumerate(var_names):
        cur_sim = sim_mean[:, p]
        cur_ref = ref_mean[:, p]
        diff = cur_sim - cur_ref

        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        max_err = float(np.max(np.abs(diff)))
        norm_ref = float(np.linalg.norm(cur_ref))
        rel_l2 = float(np.linalg.norm(diff) / (norm_ref if norm_ref > 1e-12 else 1.0))

        # Pearson correlation coefficient
        if np.std(cur_sim) > 1e-12 and np.std(cur_ref) > 1e-12:
            r_val = float(np.corrcoef(cur_sim, cur_ref)[0, 1])
        else:
            r_val = 1.0

        var_metrics[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "MaxError": max_err,
            "RelativeL2": rel_l2,
            "Pearson_r": r_val,
        }

    # 7. File Exports
    created_files = {}

    # A. 3D Fluctuation File (fluc.dat)
    actual_fluc_file = fluc_output or os.path.join(output_dir, "fluc.dat")
    if save_fluc:
        print(f"\n[+] Saving 3D fluctuation field to '{actual_fluc_file}'...")
        t_fluc = save_fluc_file(
            actual_fluc_file, var_prime, x1, x2, x3, time_val, nstep_val
        )
        fluc_size_mb = os.path.getsize(actual_fluc_file) / (1024 * 1024)
        print(f"    Fluctuation checkpoint saved: {actual_fluc_file} ({fluc_size_mb:.1f} MB in {t_fluc:.2f} s)")
        created_files["fluc_file"] = os.path.abspath(actual_fluc_file)
        created_files["fluc_size_mb"] = fluc_size_mb

    if save_stats:
        # B. 1D RMS Turbulence Profile (rms.dat)
        actual_rms_file = rms_output or os.path.join(output_dir, "rms.dat")
        rms_data = np.column_stack([x3, sim_rms])
        fmt_8 = " %22.18f" * 8
        np.savetxt(actual_rms_file, rms_data, fmt=fmt_8)
        created_files["rms_file"] = os.path.abspath(actual_rms_file)
        print(f"    Saved RMS turbulence profile: {actual_rms_file}")

        # C. 1D New Simulated Mean Profile (mean.dat)
        actual_mean_file = mean_output or os.path.join(output_dir, "mean.dat")
        mean_out_data = np.column_stack([x3, sim_mean])
        np.savetxt(actual_mean_file, mean_out_data, fmt=fmt_8)
        created_files["mean_file"] = os.path.abspath(actual_mean_file)
        print(f"    Saved simulated mean profile: {actual_mean_file}")

        # D. 1D Turbulent Kinetic Energy & Reynolds Stresses (turb_KE.dat & meanTKE.dat)
        actual_tke_file = tke_output or os.path.join(output_dir, "turb_KE.dat")
        tke_data = np.column_stack([x3, uu, vv, ww, uv, vt, tke])
        fmt_7 = " %22.18f" * 7
        np.savetxt(actual_tke_file, tke_data, fmt=fmt_7)
        created_files["tke_file"] = os.path.abspath(actual_tke_file)

        # Also write meanTKE.dat alias matching Fortran
        mean_tke_file = os.path.join(output_dir, "meanTKE.dat")
        np.savetxt(mean_tke_file, tke_data, fmt=fmt_7)
        created_files["mean_tke_file"] = os.path.abspath(mean_tke_file)
        print(f"    Saved TKE & Reynolds stress profiles: {actual_tke_file}")

    return {
        "mean_source_file": mean_source_path,
        "metrics": var_metrics,
        "created_files": created_files,
        "grid_shape": [nx, ny, nz],
        "time_val": time_val,
        "nstep_val": nstep_val,
    }


# ---------------------------------------------------------------------------
# Benchmark Runner Implementation
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Orchestrator for CUDA GPU solver live execution and mean profile post-processing."""

    def __init__(
        self,
        steps: int = 50,
        input_file: Optional[str] = None,
        output_file: Optional[str] = "output_cuda/3D.dat",
        mean_file: Optional[str] = "output/mean.dat",
        save_output: bool = True,
        save_interval: Optional[int] = None,
        save_fluc: bool = True,
        save_stats: bool = True,
        resume: bool = False,
        post_only: bool = False,
        fluc_output: Optional[str] = None,
        rms_output: Optional[str] = None,
        mean_output: Optional[str] = None,
        tke_output: Optional[str] = None,
    ):
        self.steps = steps
        self.input_file = input_file
        self.output_file = output_file
        self.mean_file = mean_file
        self.save_output = save_output
        self.save_interval = save_interval
        self.save_fluc = save_fluc
        self.save_stats = save_stats
        self.resume = resume
        self.post_only = post_only
        self.fluc_output = fluc_output
        self.rms_output = rms_output
        self.mean_output = mean_output
        self.tke_output = tke_output

    def run_cuda_benchmark(self) -> Dict[str, Any]:
        return measure_cuda_performance(
            num_steps=self.steps,
            input_file=self.input_file,
            output_file=self.output_file,
            mean_file=self.mean_file,
            save_output=self.save_output,
            save_interval=self.save_interval,
            save_fluc=self.save_fluc,
            save_stats=self.save_stats,
            resume=self.resume,
            post_only=self.post_only,
            fluc_output=self.fluc_output,
            rms_output=self.rms_output,
            mean_output=self.mean_output,
            tke_output=self.tke_output,
        )

    def run(self) -> Dict[str, Any]:
        return run_benchmark(
            num_cuda_steps=self.steps,
            input_file=self.input_file,
            output_file=self.output_file,
            mean_file=self.mean_file,
            save_output=self.save_output,
            save_interval=self.save_interval,
            save_fluc=self.save_fluc,
            save_stats=self.save_stats,
            resume=self.resume,
            post_only=self.post_only,
            fluc_output=self.fluc_output,
            rms_output=self.rms_output,
            mean_output=self.mean_output,
            tke_output=self.tke_output,
        )


def measure_cuda_performance(
    num_steps: int = 50,
    input_file: Optional[str] = None,
    output_file: Optional[str] = "output_cuda/3D.dat",
    mean_file: Optional[str] = "output/mean.dat",
    save_output: bool = True,
    save_interval: Optional[int] = None,
    save_fluc: bool = True,
    save_stats: bool = True,
    resume: bool = False,
    post_only: bool = False,
    fluc_output: Optional[str] = None,
    rms_output: Optional[str] = None,
    mean_output: Optional[str] = None,
    tke_output: Optional[str] = None,
    diagnostics_use_gpu: bool = True,
    param_file: str = "param.dat",
    solver_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Dynamically executes the CUDA GPU solver live using time.perf_counter().
    Reads reference mean profile from output/mean.dat, computes fluctuations and
    turbulence statistics, and exports checkpoints and profiles.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be zero or greater.")
    if save_interval is not None and save_interval < 1:
        raise ValueError("save_interval must be a positive number of steps.")

    # Locate cuda code.py
    cuda_script = "cuda code.py"
    if not os.path.exists(cuda_script) and os.path.exists("../cuda code.py"):
        cuda_script = "../cuda code.py"

    if not os.path.exists(cuda_script):
        raise FileNotFoundError(f"Could not locate '{cuda_script}'")

    # Dynamically import module from file
    spec = importlib.util.spec_from_file_location("cuda_code_module", cuda_script)
    cuda_mod = importlib.util.module_from_spec(spec)
    sys.modules["cuda_code_module"] = cuda_mod
    spec.loader.exec_module(cuda_mod)

    # Determine input path (handle --resume if requested)
    target_input = input_file
    if resume and not target_input:
        if output_file and os.path.exists(output_file):
            target_input = output_file
            print(f"    [+] Resuming from existing checkpoint: {output_file}")
        elif os.path.exists("output_cuda/3D_cuda.dat"):
            target_input = "output_cuda/3D_cuda.dat"
            print(f"    [+] Resuming from existing checkpoint: {target_input}")
        else:
            print(f"    [!] --resume specified but no checkpoint found in '{output_file}'. Falling back to default initial condition.")

    overrides = dict(solver_overrides or {})
    overrides["steps"] = num_steps
    live_stats_dir = overrides.pop("stats_dir", None)
    live_ckpt_dir = overrides.pop("checkpoint_dir", None)
    cfg, q0, time_init, nstep_init, loaded_source = cuda_mod.configure_and_load_state(
        param_path=param_file,
        input_file=target_input,
        demo=False,
        overrides=overrides,
        fallback_to_demo=True,
        default_nx=130,
        default_ny=130,
        default_nz=160,
        default_re=3000.0,
        default_mach=1.5,
        default_dt=1.0e-4,
    )
    if loaded_source == "demo":
        print("    Using synthetic demo state.")
    else:
        print(f"    Loaded input snapshot: {loaded_source} (Grid: {cfg.nx}x{cfg.ny}x{cfg.nz}, t={time_init:.4f}, step={nstep_init})")

    solver = cuda_mod.CUDASolver(cfg, q0)
    solver.time_value = time_init
    solver.nstep = nstep_init

    steps_completed = 0
    interrupted = False
    elapsed = 0.0

    if not post_only and num_steps > 0:
        print(f"\n[+] Initializing CUDA GPU Solver ({num_steps} RK2 steps requested)...")
        # Warm up CUDA JIT with live stats/checkpoints detached so warmup never pollutes outputs.
        print("    Warming up Numba CUDA JIT kernels...")
        solver.step()
        cuda_mod.cuda.synchronize()
        solver.d_q.copy_to_device(np.ascontiguousarray(q0))
        solver.reset_runtime_buffers()
        solver.time_value = time_init
        solver.nstep = nstep_init
        solver.cfg.stats_dir = live_stats_dir
        solver.cfg.checkpoint_dir = live_ckpt_dir

        # Time live execution loop
        t_start = time.perf_counter()
        log_freq = max(1, min(50, num_steps // 10)) if num_steps > 10 else 1

        try:
            for step_idx in range(1, num_steps + 1):
                solver.step()
                steps_completed += 1

                # Save periodic checkpoints if requested
                if save_interval is not None and steps_completed % save_interval == 0:
                    base_name, ext = os.path.splitext(output_file) if output_file else ("output_cuda/3D", ".dat")
                    ckpt_path = f"{base_name}_step{solver.nstep}{ext}"
                    print(f"\n    [Checkpoint] Saving intermediate state at step {solver.nstep} to '{ckpt_path}'...")
                    solver.save_solution(ckpt_path)

                # Live progress output
                if step_idx % log_freq == 0 or step_idx == num_steps:
                    cuda_mod.cuda.synchronize()
                    cur_elapsed = time.perf_counter() - t_start
                    cur_rate = steps_completed / max(cur_elapsed, 1e-6)
                    ms_step = (cur_elapsed / steps_completed) * 1000.0
                    rem_steps = num_steps - steps_completed
                    eta_sec = rem_steps / cur_rate if cur_rate > 0 else 0
                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s" if eta_sec >= 60 else f"{eta_sec:.1f}s"
                    pct = (steps_completed / num_steps) * 100.0
                    print(
                        f"    [Step {steps_completed:d}/{num_steps:d}] ({pct:5.1f}%) | "
                        f"t={solver.time_value:.6e} | {ms_step:.2f} ms/step ({cur_rate:.1f} steps/s) | "
                        f"Elapsed: {cur_elapsed:.1f}s | ETA: {eta_str}",
                        flush=True
                    )
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[!] Execution interrupted by user at step {steps_completed}/{num_steps}.")
        finally:
            cuda_mod.cuda.synchronize()
            t_end = time.perf_counter()

        elapsed = t_end - t_start
    else:
        print("\n[+] Post-processing mode: Skipping time integration steps.")

    actual_steps = max(steps_completed, 1)
    step_time = elapsed / actual_steps if steps_completed > 0 else 0.0
    throughput = actual_steps / max(elapsed, 1e-6) if steps_completed > 0 else 0.0

    if steps_completed > 0:
        status_str = f"interrupted after {steps_completed}" if interrupted else f"completed {steps_completed}"
        print(f"\n    GPU Solver {status_str} steps in {elapsed:.3f} s ({step_time*1000:.2f} ms/step, {throughput:.1f} steps/s)")

    # Save output 3D solution checkpoint if requested
    saved_checkpoint_path = None
    saved_file_size_mb = 0.0
    if save_output and output_file and (steps_completed > 0 or post_only):
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        print(f"\n[+] Saving CUDA 3D solution flow state checkpoint to '{output_file}'...")
        t_save_0 = time.perf_counter()
        solver.save_solution(output_file)
        t_save = time.perf_counter() - t_save_0
        if os.path.exists(output_file):
            saved_checkpoint_path = os.path.abspath(output_file)
            saved_file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"    Checkpoint saved successfully: {output_file} ({saved_file_size_mb:.1f} MB in {t_save:.2f} s)")

    # -----------------------------------------------------------------------
    # Ingest Reference Mean Profile and Obtain New Results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" INGESTING REFERENCE MEAN PROFILE & COMPUTING NEW RESULTS")
    print("=" * 80)

    mean_data, mean_source = load_mean_profile(mean_file)
    results_from_mean = None

    if mean_data is not None:
        var_h = solver.download_primitives()
        out_target_dir = os.path.dirname(output_file) if output_file else "output_cuda"

        results_from_mean = compute_and_save_new_results(
            var_h=var_h,
            x1=solver.x1_h,
            x2=solver.x2_h,
            x3=solver.x3_h,
            time_val=solver.time_value,
            nstep_val=solver.nstep,
            mean_data=mean_data,
            mean_source_path=mean_source,
            output_dir=out_target_dir or "output_cuda",
            save_fluc=save_fluc,
            save_stats=save_stats,
            fluc_output=fluc_output,
            rms_output=rms_output,
            mean_output=mean_output,
            tke_output=tke_output,
        )

    print("\n" + "=" * 80)
    print(" EXECUTING CUDA FLOW DIAGNOSTICS (KE, massflow, rhobulk, Ubulk, tauWall)")
    print("=" * 80)
    var_h = solver.download_primitives()
    out_target_dir = os.path.dirname(output_file) if output_file else "output_cuda"
    rho_cur, u_cur = var_h[..., 0], var_h[..., 1]
    v_cur, w_cur, temperature_cur = var_h[..., 2], var_h[..., 3], var_h[..., 5]
    dy_cur = (
        float(solver.x2_h[1] - solver.x2_h[0])
        if len(solver.x2_h) > 1
        else (4.0 * math.pi / 3.0) / (cfg.ny - 1)
    )
    diag_results = calculate_diagnostics_cuda(
        rho=rho_cur,
        u=u_cur,
        v=v_cur,
        w=w_cur,
        T=temperature_cur,
        z_coords=solver.x3_h,
        dy=dy_cur,
        reynolds=cfg.reynolds,
        use_gpu=diagnostics_use_gpu,
        dx=solver.dx,
    )
    append_diagnostic_record(out_target_dir or "output_cuda", solver.time_value, diag_results)
    print(f"    [+] Flow diagnostics computed for t={solver.time_value:.6f}:")
    print(f"        - Kinetic Energy (KE) : {diag_results['KE']:.7f}")
    print(f"        - Bulk Density (rho_b): {[round(x, 5) for x in diag_results['rhobulk']]}")
    print(f"        - Bulk Velocity (U_b) : {[round(x, 5) for x in diag_results['Ubulk']]}")
    print(f"        - Bulk Mass Flow      : {[round(x, 5) for x in diag_results['massflow']]}")
    print(
        f"        - Wall Shear Stress   : Bottom={diag_results['tau_bottom']:.6f}, "
        f"Top={diag_results['tau_top']:.6f}, Avg={diag_results['tau_avg']:.6f}"
    )
    if results_from_mean is not None:
        results_from_mean["flow_diagnostics"] = diag_results

    # Save complete benchmark & statistical analysis summary JSON
    summary_dict = {
        "num_steps_requested": num_steps,
        "num_steps_completed": steps_completed,
        "interrupted": interrupted,
        "total_time_sec": elapsed,
        "step_time_sec": step_time,
        "steps_per_sec": throughput,
        "ms_per_step": step_time * 1000.0,
        "final_time_value": solver.time_value,
        "final_step_index": solver.nstep,
        "grid_shape": [cfg.nx, cfg.ny, cfg.nz],
        "total_grid_points": cfg.nx * cfg.ny * cfg.nz,
        "output_solution_file": saved_checkpoint_path,
        "output_file_size_mb": saved_file_size_mb,
        "checkpoint_interval_steps": save_interval,
        "mean_profile_source": mean_source,
        "statistics_results": results_from_mean,
    }

    results_json_path = os.path.join(os.path.dirname(output_file) if output_file else "output_cuda", "benchmark_results.json")
    try:
        with open(results_json_path, "w") as f:
            json.dump(summary_dict, f, indent=2)
        print(f"\n[+] Full benchmark results saved to JSON: {results_json_path}")
    except Exception as e:
        print(f"    [!] Could not save JSON report: {e}")

    return summary_dict


def run_benchmark(
    num_cuda_steps: int = 50,
    input_file: Optional[str] = None,
    output_file: Optional[str] = "output_cuda/3D.dat",
    mean_file: Optional[str] = "output/mean.dat",
    save_output: bool = True,
    save_interval: Optional[int] = None,
    save_fluc: bool = True,
    save_stats: bool = True,
    resume: bool = False,
    post_only: bool = False,
    fluc_output: Optional[str] = None,
    rms_output: Optional[str] = None,
    mean_output: Optional[str] = None,
    tke_output: Optional[str] = None,
    diagnostics_use_gpu: bool = True,
    param_file: str = "param.dat",
    solver_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Runs the CUDA GPU solver live, ingests mean profile, and outputs new results."""
    print("=" * 80)
    print(" 3D Compressible CFD Solver - CUDA GPU Benchmark & Statistics Suite")
    print("=" * 80)

    summary = measure_cuda_performance(
        num_steps=num_cuda_steps,
        input_file=input_file,
        output_file=output_file,
        mean_file=mean_file,
        save_output=save_output,
        save_interval=save_interval,
        save_fluc=save_fluc,
        save_stats=save_stats,
        resume=resume,
        post_only=post_only,
        fluc_output=fluc_output,
        rms_output=rms_output,
        mean_output=mean_output,
        tke_output=tke_output,
        diagnostics_use_gpu=diagnostics_use_gpu,
        param_file=param_file,
        solver_overrides=solver_overrides,
    )

    t_cuda = summary["step_time_sec"]
    throughput = summary["steps_per_sec"]

    print("\n" + "=" * 80)
    print(" BENCHMARK & TURBULENCE STATISTICS SUMMARY")
    print("=" * 80)
    print(f" Steps Executed                   : {summary['num_steps_completed']}")
    print(f" Total Wall-Clock Execution Time  : {summary['total_time_sec']:.3f} s")
    if summary["num_steps_completed"] > 0:
        print(f" Average Step Time                : {t_cuda:.4f} s/step ({t_cuda*1000:.2f} ms/step)")
        print(f" Throughput                       : {throughput:.2f} steps/second")
    if summary.get("output_solution_file"):
        print(f" 3D Solution Checkpoint           : {summary['output_solution_file']} ({summary.get('output_file_size_mb', 0):.1f} MB)")
    if summary.get("mean_profile_source"):
        print(f" Reference Mean Profile Read From : {summary['mean_profile_source']}")

    stats = summary.get("statistics_results")
    if stats and "metrics" in stats:
        print("\n Quantitative Comparison: Simulated State vs Reference Mean Profile:")
        print("-" * 80)
        print(f" {'Variable':<10} {'MAE':<12} {'RMSE':<12} {'Max Error':<12} {'Rel L2 (%)':<12} {'Pearson r':<10}")
        print("-" * 80)
        for var_name, m in stats["metrics"].items():
            print(
                f" {var_name:<10} {m['MAE']:<12.4e} {m['RMSE']:<12.4e} "
                f"{m['MaxError']:<12.4e} {m['RelativeL2']*100:<12.4f}% {m['Pearson_r']:<10.4f}"
            )
        print("-" * 80)

    if stats and "created_files" in stats:
        print("\n Generated Result Files (Derived from output/mean.dat):")
        for desc, fpath in stats["created_files"].items():
            if not desc.endswith("_mb"):
                print(f"  [+] {desc:<16}: {fpath}")

    print("=" * 80)
    return summary


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the CUDA solver and diagnostics, or post-process existing 3D snapshots."
    )
    p.add_argument(
        "--steps", type=_nonnegative_int, default=50,
        help="Total solver steps to execute in this run (default: 50)",
    )
    p.add_argument("--input", default=None, help="Input solution checkpoint path (default: input/3D.dat)")
    p.add_argument(
        "--output",
        default="output_cuda/3D.dat",
        help="Output Tecplot solution file path (default: output_cuda/3D.dat)",
    )
    p.add_argument(
        "--mean-file",
        default="output/mean.dat",
        help="Path to 1D wall-normal reference mean profile (default: output/mean.dat)",
    )
    p.add_argument(
        "--save-interval", "--checkpoint-interval",
        type=_positive_int,
        default=None,
        help="Save a resumable 3D checkpoint after each N completed steps (e.g. 1000)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from existing checkpoint in --output (default: output_cuda/3D.dat) instead of restarting from input/3D.dat",
    )
    p.add_argument(
        "--post-only",
        action="store_true",
        help="Run post-processing and statistics calculation only (skips solver time steps)",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving the 3D output solution checkpoint (timing only)",
    )
    p.add_argument(
        "--no-fluc",
        action="store_true",
        help="Disable exporting large 3D fluctuation field fluc.dat (compute 1D profiles only)",
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable exporting 1D statistics profiles (rms.dat, mean.dat, turb_KE.dat)",
    )
    p.add_argument(
        "--fluc-output",
        default=None,
        help="Custom destination for 3D fluctuation field fluc.dat (default: output_cuda/fluc.dat)",
    )
    p.add_argument(
        "--rms-output",
        default=None,
        help="Custom destination for 1D RMS profile rms.dat (default: output_cuda/rms.dat)",
    )
    p.add_argument(
        "--mean-output",
        default=None,
        help="Custom destination for 1D mean profile mean.dat (default: output_cuda/mean.dat)",
    )
    p.add_argument(
        "--tke-output",
        default=None,
        help="Custom destination for 1D TKE profile turb_KE.dat (default: output_cuda/turb_KE.dat)",
    )
    p.add_argument(
        "--diagnostics-input", "-i", default=None,
        help="Compute diagnostics for one existing Tecplot 3D snapshot (without running solver).",
    )
    p.add_argument(
        "--diagnostics-dir", "-d", default=None,
        help="Process a directory of existing 3D Tecplot snapshots (without running solver).",
    )
    p.add_argument(
        "--pattern", default="3D_step*.dat",
        help="Snapshot filename pattern used with --diagnostics-dir.",
    )
    p.add_argument(
        "--output-dir", default="output_cuda",
        help="Output directory for diagnostics when processing existing snapshots.",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Use CPU diagnostics instead of CUDA kernels (solver mode still requires CUDA).",
    )
    p.add_argument("--param", default="param.dat", help="Path to param.dat (default: param.dat)")
    p.add_argument("--mesh", choices=("auto", "cosine", "tanh"), default=None, help="Wall-normal grid stretching")
    p.add_argument("--wall-order", type=int, choices=(2, 3), default=None, help="Wall derivative order (2=codes/, 3=codes1/)")
    p.add_argument("--oc", type=int, choices=(0, 1), default=None, help="1=enable opposition control, 0=disable")
    p.add_argument("--amp-v", type=float, default=None, help="Spanwise wall oscillation amplitude")
    p.add_argument("--kappa", type=float, default=None, help="Actuation streamwise wavenumber")
    p.add_argument("--omega", type=float, default=None, help="Actuation angular frequency")
    p.add_argument("--dynamic-source", dest="dynamic_source", action="store_true", default=None, help="Enable dynamic GPU source updates")
    p.add_argument("--no-dynamic-source", dest="dynamic_source", action="store_false", help="Disable dynamic GPU source updates")
    p.add_argument("--stats-dir", default=None, help="Directory to stream live Fortran-parity .dat stats during stepping")
    p.add_argument("--stats-mode", choices=("none", "minimal", "standard", "full"), default=None, help="Live statistics tier (default: standard)")
    p.add_argument("--diag-freq", type=_positive_int, default=None, help="Live Diagnostic_files/PowSpec cadence in steps (default: 10)")
    p.add_argument("--checkpoint-dir", default=None, help="Directory for rolling 3D.dat/3D_.dat or component checkpoints")
    p.add_argument("--output-freq", type=_positive_int, default=None, help="Rolling checkpoint cadence in steps (default: 1000)")
    p.add_argument("--checkpoint-components", action="store_true", default=None, help="Write rolling checkpoints as domInfo.txt + 3Dnnn.dat")
    return p.parse_args()


def main():
    args = parse_args()
    if args.diagnostics_input and args.diagnostics_dir:
        raise SystemExit("Use only one of --diagnostics-input or --diagnostics-dir.")
    if args.diagnostics_input:
        diagnostics = process_3d_snapshot(
            args.diagnostics_input,
            output_dir=args.output_dir,
            append=True,
            use_gpu=not args.no_gpu,
        )
        if diagnostics is None:
            raise SystemExit(1)
        print(f"    Time          : {diagnostics['time']:.6f}")
        print(f"    Kinetic Energy: {diagnostics['KE']:.7f}")
        print(f"    Bulk Density  : {diagnostics['rhobulk']}")
        print(f"    Bulk Velocity : {diagnostics['Ubulk']}")
        print(f"    Bulk Mass Flow: {diagnostics['massflow']}")
        print(
            f"    Wall Shear    : Bottom={diagnostics['tau_bottom']:.7f}, "
            f"Top={diagnostics['tau_top']:.7f}, Avg={diagnostics['tau_avg']:.7f}"
        )
        return
    if args.diagnostics_dir:
        processed = process_snapshot_series(
            args.diagnostics_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            use_gpu=not args.no_gpu,
        )
        if processed == 0:
            raise SystemExit(1)
        return

    solver_overrides = {
        k: getattr(args, k)
        for k in (
            "mesh",
            "wall_order",
            "oc",
            "amp_v",
            "kappa",
            "omega",
            "dynamic_source",
            "stats_dir",
            "stats_mode",
            "diag_freq",
            "checkpoint_dir",
            "output_freq",
            "checkpoint_components",
        )
    }
    run_benchmark(
        num_cuda_steps=args.steps,
        input_file=args.input,
        output_file=args.output,
        mean_file=args.mean_file,
        save_output=not args.no_save,
        save_interval=args.save_interval,
        save_fluc=not args.no_fluc,
        save_stats=not args.no_stats,
        resume=args.resume,
        post_only=args.post_only,
        fluc_output=args.fluc_output,
        rms_output=args.rms_output,
        mean_output=args.mean_output,
        tke_output=args.tke_output,
        diagnostics_use_gpu=not args.no_gpu,
        param_file=args.param,
        solver_overrides=solver_overrides,
    )


if __name__ == "__main__":
    main()
