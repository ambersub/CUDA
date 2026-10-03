#!/usr/bin/env python3
"""
Dynamic Execution & Performance Benchmark Suite for Fortran MPI and Python Numba CUDA Solvers.

Measures real wall-clock execution time for both solvers dynamically without hardcoded values:
  1. CUDA GPU Solver: Executes CUDASolver or cuda code.py live for N steps with high-resolution time.perf_counter().
  2. Fortran MPI Solver: Attempts live execution via mpiexec/gfortran/ifort if available, or dynamically extracts real-time CPU step logs from terminal.dat.

Author: Google DeepMind Antigravity Pair Programmer
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, Any, Optional


def measure_cuda_performance(num_steps: int = 50, input_file: Optional[str] = None) -> Dict[str, float]:
    """
    Dynamically measures real CUDA GPU solver execution time using time.perf_counter().
    """
    print(f"\n[+] Running CUDA GPU Solver Live ({num_steps} RK2 steps)...")

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

    # Determine input path
    target_input = input_file
    if not target_input:
        for cand in ["input/3D.dat", "3D.dat", "../input/3D.dat"]:
            if os.path.exists(cand):
                target_input = cand
                break

    cfg = cuda_mod.SolverConfig(
        nx=130, ny=130, nz=160,
        reynolds=3000.0, mach=1.5, dt=1.0e-4,
        max_steps=num_steps, statistic_frequency=num_steps
    )

    if target_input and os.path.exists(target_input):
        q0, time_init, nstep_init = cuda_mod.read_combined_input(target_input, cfg)
        print(f"    Loaded input snapshot: {target_input}")
    else:
        q0 = cuda_mod.demo_initial_condition(cfg)
        time_init, nstep_init = 0.0, 0
        print("    Using synthetic demo state.")

    solver = cuda_mod.CUDASolver(cfg, q0)
    solver.time_value = time_init
    solver.nstep = nstep_init

    # Warmup step to initialize Numba CUDA JIT kernels
    print("    Warming up Numba CUDA JIT kernels...")
    solver.step()

    # Time live execution loop
    cuda_mod.cuda.synchronize()
    t_start = time.perf_counter()

    for _ in range(num_steps):
        solver.step()

    cuda_mod.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    step_time = elapsed / max(num_steps, 1)
    throughput = max(num_steps, 1) / elapsed

    print(f"    Completed {num_steps} GPU steps in {elapsed:.3f} s ({step_time*1000:.2f} ms/step, {throughput:.1f} steps/s)")

    return {
        "num_steps": num_steps,
        "total_time_sec": elapsed,
        "step_time_sec": step_time,
        "steps_per_sec": throughput,
        "ms_per_step": step_time * 1000.0
    }


def measure_fortran_performance(terminal_log: str = "terminal.dat") -> Dict[str, Any]:
    """
    Dynamically measures Fortran MPI execution time:
      - Attempts live execution if mpiexec/gfortran/ifort is present.
      - Otherwise dynamically extracts real-time CPU step logs from terminal.dat.
    """
    print("\n[+] Evaluating Fortran MPI Solver Performance...")

    # Check if mpiexec or compiler is present for live run
    mpiexec_path = shutil.which("mpiexec") or shutil.which("mpirun")
    live_executed = False
    timing_logs = []

    if mpiexec_path and os.path.exists("codes/main_v1.9_unc.x"):
        print("    Found mpiexec & Fortran binary. Attempting live Fortran MPI execution...")
        try:
            t0 = time.perf_counter()
            proc = subprocess.run([mpiexec_path, "-np", "4", "./codes/main_v1.9_unc.x"], capture_output=True, text=True, timeout=15)
            t1 = time.perf_counter()
            if proc.returncode == 0:
                live_executed = True
                print(f"    Live Fortran execution completed in {t1 - t0:.3f} s")
                # Parse stdout for timing lines
                for line in proc.stdout.splitlines():
                    m = re.search(r'CPU Time:\s*([0-9\.]+)\s*sec for 10 iters', line)
                    if m:
                        timing_logs.append(float(m.group(1)))
        except Exception as e:
            print(f"    Live Fortran run skipped: {e}")

    # Fallback/dynamic extraction from terminal.dat log file if live run didn't produce timing logs
    if not timing_logs:
        target_log = terminal_log
        if not os.path.exists(target_log) and os.path.exists("../terminal.dat"):
            target_log = "../terminal.dat"

        if os.path.exists(target_log):
            print(f"    Parsing real-time execution logs from '{target_log}'...")
            with open(target_log, 'r') as f:
                for line in f:
                    m = re.search(r'CPU Time:\s*([0-9\.]+)\s*sec for 10 iters', line)
                    if m:
                        timing_logs.append(float(m.group(1)))

    if timing_logs:
        avg_10_iter = float(sum(timing_logs) / len(timing_logs))
        step_time = avg_10_iter / 10.0
        throughput = 1.0 / step_time if step_time > 0 else 0.0
        print(f"    Extracted {len(timing_logs)} iteration timing samples.")
        print(f"    Fortran Mean Time per 10 steps: {avg_10_iter:.3f} s ({step_time*1000:.2f} ms/step, {throughput:.1f} steps/s)")
    else:
        print("    [WARNING] No Fortran timing logs found. Returning 0.0 sec.")
        avg_10_iter, step_time, throughput = 0.0, 0.0, 0.0

    return {
        "live_executed": live_executed,
        "sample_count": len(timing_logs),
        "ten_iter_time_sec": avg_10_iter,
        "step_time_sec": step_time,
        "steps_per_sec": throughput,
        "ms_per_step": step_time * 1000.0
    }


def run_benchmark(num_cuda_steps: int = 50, input_file: Optional[str] = None) -> Dict[str, Any]:
    """Runs full dynamic benchmark measuring real execution times for both solvers."""
    print("=" * 80)
    print(" 3D Compressible CFD Solver Dynamic Performance Benchmark")
    print("=" * 80)

    # Measure CUDA GPU live
    cuda_perf = measure_cuda_performance(num_steps=num_cuda_steps, input_file=input_file)

    # Measure Fortran MPI
    fortran_perf = measure_fortran_performance()

    t_fort = fortran_perf["step_time_sec"]
    t_cuda = cuda_perf["step_time_sec"]

    speedup = (t_fort / t_cuda) if (t_cuda > 0 and t_fort > 0) else 1.0
    pct_reduction = ((t_fort - t_cuda) / t_fort * 100.0) if t_fort > 0 else 0.0

    print("\n" + "=" * 80)
    print(" DYNAMIC PERFORMANCE BENCHMARK RESULT")
    print("=" * 80)
    print(f" Fortran MPI Solver Execution Time : {t_fort:.4f} s/step ({t_fort*10:.3f} s per 10 steps)")
    print(f" Python CUDA GPU Solver Execution Time: {t_cuda:.4f} s/step ({t_cuda*10:.3f} s per 10 steps)")
    print(f" Real GPU Speedup Factor              : {speedup:.2f}x faster on GPU")
    print(f" Wall-Clock Time Reduction            : {pct_reduction:.1f}% faster")
    print("=" * 80)

    return {
        "cuda_performance": cuda_perf,
        "fortran_performance": fortran_perf,
        "benchmark_summary": {
            "fortran_step_time_sec": t_fort,
            "cuda_step_time_sec": t_cuda,
            "speedup_factor": speedup,
            "percent_time_reduction": pct_reduction
        }
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run Dynamic Benchmark for Fortran MPI vs CUDA GPU Solver")
    p.add_argument("--steps", type=int, default=50, help="Number of GPU steps to measure live")
    p.add_argument("--input", default=None, help="Input solution checkpoint path")
    return p.parse_args()


def main():
    args = parse_args()
    run_benchmark(num_cuda_steps=args.steps, input_file=args.input)


if __name__ == "__main__":
    main()
