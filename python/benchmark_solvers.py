#!/usr/bin/env python3
"""
Live CUDA GPU Solver Performance Benchmark and 3D Flow Field Checkpoint Generator.

Measures real wall-clock execution time for the Numba CUDA GPU solver:
  1. Executes CUDASolver (cuda code.py) live on the GPU for N RK2 time steps.
  2. Dynamically logs progress, step rates, and elapsed times.
  3. Saves the final resulting 3D flow state solution checkpoint (default: output_cuda/3D.dat).
  4. Supports checkpoint resumption (--resume) if stopped and restarted.

Note: Solution comparison and verification against Fortran is performed separately
      via python/compare_fortran_cuda.py when both solvers are run on the same machine.

Author: Google DeepMind Antigravity Pair Programmer
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import warnings
from typing import Dict, Any, Optional

# Suppress minor version warnings for clean terminal output
os.environ.setdefault("CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING", "1")
warnings.filterwarnings("ignore")


class BenchmarkRunner:
    """Orchestrator for CUDA GPU solver live execution and 3D checkpoint saving."""

    def __init__(
        self,
        steps: int = 50,
        input_file: Optional[str] = None,
        output_file: Optional[str] = "output_cuda/3D.dat",
        save_output: bool = True,
        save_interval: Optional[int] = None,
        resume: bool = False,
    ):
        self.steps = steps
        self.input_file = input_file
        self.output_file = output_file
        self.save_output = save_output
        self.save_interval = save_interval
        self.resume = resume

    def run_cuda_benchmark(self) -> Dict[str, Any]:
        return measure_cuda_performance(
            num_steps=self.steps,
            input_file=self.input_file,
            output_file=self.output_file,
            save_output=self.save_output,
            save_interval=self.save_interval,
            resume=self.resume,
        )

    def run(self) -> Dict[str, Any]:
        return run_benchmark(
            num_cuda_steps=self.steps,
            input_file=self.input_file,
            output_file=self.output_file,
            save_output=self.save_output,
            save_interval=self.save_interval,
            resume=self.resume,
        )


def measure_cuda_performance(
    num_steps: int = 50,
    input_file: Optional[str] = None,
    output_file: Optional[str] = "output_cuda/3D.dat",
    save_output: bool = True,
    save_interval: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Dynamically executes the CUDA GPU solver live using high-resolution time.perf_counter().
    Saves the computed 3D flow state checkpoint to output_file (default: output_cuda/3D.dat).
    """
    print(f"\n[+] Initializing CUDA GPU Solver ({num_steps} RK2 steps requested)...")

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
        print(f"    Loaded input snapshot: {target_input} (Grid: {cfg.nx}x{cfg.ny}x{cfg.nz}, t={time_init:.4f}, step={nstep_init})")
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
    steps_completed = 0
    interrupted = False

    # Log frequency: every 10% or at most 50 steps
    log_freq = max(1, min(50, num_steps // 10)) if num_steps > 10 else 1

    try:
        for step_idx in range(1, num_steps + 1):
            solver.step()
            steps_completed += 1

            # Save periodic checkpoints if requested
            if save_interval and steps_completed % save_interval == 0:
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
    actual_steps = max(steps_completed, 1)
    step_time = elapsed / actual_steps
    throughput = actual_steps / max(elapsed, 1e-6)

    status_str = f"interrupted after {steps_completed}" if interrupted else f"completed {steps_completed}"
    print(f"\n    GPU Solver {status_str} steps in {elapsed:.3f} s ({step_time*1000:.2f} ms/step, {throughput:.1f} steps/s)")

    # Save output solution checkpoint file if enabled
    saved_checkpoint_path = None
    saved_file_size_mb = 0.0
    if save_output and output_file and steps_completed > 0:
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

    return {
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
    }


def run_benchmark(
    num_cuda_steps: int = 50,
    input_file: Optional[str] = None,
    output_file: Optional[str] = "output_cuda/3D.dat",
    save_output: bool = True,
    save_interval: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Runs the CUDA GPU solver live and saves the output 3D.dat file."""
    print("=" * 80)
    print(" 3D Compressible CFD Solver - Live CUDA GPU Solver Execution")
    print("=" * 80)

    # Measure CUDA GPU live and save 3D.dat
    cuda_perf = measure_cuda_performance(
        num_steps=num_cuda_steps,
        input_file=input_file,
        output_file=output_file,
        save_output=save_output,
        save_interval=save_interval,
        resume=resume,
    )

    t_cuda = cuda_perf["step_time_sec"]
    throughput = cuda_perf["steps_per_sec"]

    print("\n" + "=" * 80)
    print(" CUDA GPU EXECUTION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f" Steps Executed                   : {cuda_perf['num_steps_completed']}")
    print(f" Total Wall-Clock Execution Time  : {cuda_perf['total_time_sec']:.3f} s")
    print(f" Average Step Time                : {t_cuda:.4f} s/step ({t_cuda*1000:.2f} ms/step)")
    print(f" Throughput                       : {throughput:.2f} steps/second")
    if cuda_perf.get("output_solution_file"):
        print(f" Output Solution Checkpoint       : {cuda_perf['output_solution_file']} ({cuda_perf.get('output_file_size_mb', 0):.1f} MB)")
    print("=" * 80)
    print(" Note: Output verification and comparison against Fortran should be performed")
    print("       via python/compare_fortran_cuda.py when both codes are run on this device.")
    print("=" * 80)

    return cuda_perf


def parse_args():
    p = argparse.ArgumentParser(
        description="Run CUDA GPU Solver Benchmark and Save 3D.dat Solution Checkpoint"
    )
    p.add_argument("--steps", type=int, default=50, help="Number of GPU steps to execute (default: 50)")
    p.add_argument("--input", default=None, help="Input solution checkpoint path (default: input/3D.dat)")
    p.add_argument(
        "--output",
        default="output_cuda/3D.dat",
        help="Output Tecplot solution file path (default: output_cuda/3D.dat)",
    )
    p.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Interval for saving intermediate checkpoints during long runs (e.g. 1000)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from existing checkpoint in --output (default: output_cuda/3D.dat) instead of restarting from input/3D.dat",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving the 3D output solution checkpoint (timing only)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_benchmark(
        num_cuda_steps=args.steps,
        input_file=args.input,
        output_file=args.output,
        save_output=not args.no_save,
        save_interval=args.save_interval,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
