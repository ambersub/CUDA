#!/usr/bin/env python3
"""
Re-compiles all Fortran F90 source files from scratch using GFortran,
deletes previous Fortran outputs, executes the compiled Fortran binaries live on Windows,
measures real wall-clock execution time, and compares with Python/CUDA GPU times.

Author: Google DeepMind Antigravity Pair Programmer
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Any, Optional


def find_gfortran() -> str:
    """Locates gfortran executable on system or WinGet package path."""
    gfortran = shutil.which("gfortran")
    if gfortran:
        return gfortran

    # Check WinGet MinGW GFortran install path
    appdata = os.environ.get("LOCALAPPDATA", "")
    winget_gfortran = os.path.join(
        appdata,
        r"Microsoft\WinGet\Packages\BrechtSanders.WinLibs.MCF.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\gfortran.exe"
    )
    if os.path.exists(winget_gfortran):
        return winget_gfortran

    # Search WinGet packages directory for any gfortran.exe
    pkg_dir = os.path.join(appdata, r"Microsoft\WinGet\Packages")
    if os.path.exists(pkg_dir):
        matches = glob.glob(os.path.join(pkg_dir, "**", "gfortran.exe"), recursive=True)
        if matches:
            return matches[0]

    raise FileNotFoundError("GFortran compiler not found. Install gfortran or MinGW-w64.")


def compile_fortran_codes(gfortran_bin: str) -> Dict[str, str]:
    """Compiles Fortran source modules in codes/ into native Windows executables."""
    print("=" * 75)
    print(" Compiling Fortran Source Codes from Scratch")
    print("=" * 75)
    print(f" Using Compiler: {gfortran_bin}")

    codes_dir = os.path.abspath("codes")
    executables = {}

    # GFortran flags: -static -fno-automatic ensures static memory allocation on 64-bit Windows without stack limit issues
    gflags = ["-O3", "-static", "-fno-automatic"]

    # 1. Compile shared module comdata.f90
    print("\n[1/5] Compiling shared data module: comdata.f90...")
    cmd_mod = [gfortran_bin] + gflags + ["-c", "comdata.f90", "-o", "comdata.o"]
    res_mod = subprocess.run(cmd_mod, cwd=codes_dir, capture_output=True, text=True)
    if res_mod.returncode != 0:
        raise RuntimeError(f"Failed to compile comdata.f90:\n{res_mod.stderr}")
    print("      Successfully compiled comdata.o")

    # 2. Compile prime.f90
    print("\n[2/5] Compiling prime.f90...")
    exe_prime = os.path.join(codes_dir, "prime_win.exe")
    cmd_prime = [gfortran_bin] + gflags + ["prime.f90", "-o", exe_prime]
    res_prime = subprocess.run(cmd_prime, cwd=codes_dir, capture_output=True, text=True)
    if res_prime.returncode == 0:
        executables["prime"] = exe_prime
        print(f"      Successfully compiled {os.path.basename(exe_prime)}")

    # 3. Compile fluc.f90
    print("\n[3/5] Compiling fluc.f90...")
    exe_fluc = os.path.join(codes_dir, "fluc_win.exe")
    cmd_fluc = [gfortran_bin] + gflags + ["fluc.f90", "-o", exe_fluc]
    res_fluc = subprocess.run(cmd_fluc, cwd=codes_dir, capture_output=True, text=True)
    if res_fluc.returncode == 0:
        executables["fluc"] = exe_fluc
        print(f"      Successfully compiled {os.path.basename(exe_fluc)}")

    # 4. Compile qInvariant.f90
    print("\n[4/5] Compiling qInvariant.f90...")
    exe_qinv = os.path.join(codes_dir, "qInvariant_win.exe")
    cmd_qinv = [gfortran_bin] + gflags + ["comdata.o", "qInvariant.f90", "-o", exe_qinv]
    res_qinv = subprocess.run(cmd_qinv, cwd=codes_dir, capture_output=True, text=True)
    if res_qinv.returncode == 0:
        executables["qinvariant"] = exe_qinv
        print(f"      Successfully compiled {os.path.basename(exe_qinv)}")

    # 5. Compile avg_v1.7.f90
    print("\n[5/5] Compiling avg_v1.7.f90...")
    exe_avg = os.path.join(codes_dir, "avg_v1.7_win.exe")
    cmd_avg = [gfortran_bin] + gflags + ["comdata.o", "avg_v1.7.f90", "-o", exe_avg]
    res_avg = subprocess.run(cmd_avg, cwd=codes_dir, capture_output=True, text=True)
    if res_avg.returncode == 0:
        executables["avg"] = exe_avg
        print(f"      Successfully compiled {os.path.basename(exe_avg)}")

    print("\n[+] Fortran compilation complete.")
    return executables


def clean_fortran_outputs():
    """Deletes old Fortran generated outputs in output/ and input/."""
    print("\n[+] Cleaning old Fortran output files...")
    targets = [
        "output/fluc.dat",
        "output/qInvariant.dat",
        "input/Prime3D.dat",
        "input/PrimeUV.dat"
    ]
    for target in targets:
        if os.path.exists(target):
            try:
                os.remove(target)
                print(f"    Deleted old file: {target}")
            except Exception as e:
                print(f"    Could not delete {target}: {e}")


def execute_and_time_fortran(executables: Dict[str, str]) -> Dict[str, float]:
    """Executes compiled Fortran binaries live and measures real wall-clock time."""
    print("=" * 75)
    print(" Executing Compiled Fortran Codes Live & Measuring Real Execution Time")
    print("=" * 75)

    timings = {}

    # Run fluc_win.exe
    if "fluc" in executables:
        print("\n[+] Running fluc_win.exe (Calculates velocity & pressure fluctuation fields)...")
        t0 = time.perf_counter()
        proc = subprocess.run([executables["fluc"]], cwd="codes", capture_output=True, text=True)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timings["fluc"] = elapsed
        print(f"    Completed in {elapsed:.3f} s (Exit code: {proc.returncode})")

    # Run qInvariant_win.exe
    if "qinvariant" in executables:
        print("\n[+] Running qInvariant_win.exe (Calculates 3D Q-criterion vortex identification)...")
        t0 = time.perf_counter()
        proc = subprocess.run([executables["qinvariant"]], cwd="codes", capture_output=True, text=True)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timings["qinvariant"] = elapsed
        print(f"    Completed in {elapsed:.3f} s (Exit code: {proc.returncode})")

    # Run prime_win.exe
    if "prime" in executables:
        print("\n[+] Running prime_win.exe (Calculates Prime3D & PrimeUV fluctuation snapshots)...")
        t0 = time.perf_counter()
        proc = subprocess.run([executables["prime"]], cwd="codes", capture_output=True, text=True)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timings["prime"] = elapsed
        print(f"    Completed in {elapsed:.3f} s (Exit code: {proc.returncode})")

    return timings


def run_recompile_benchmark():
    gfortran_bin = find_gfortran()
    clean_fortran_outputs()
    executables = compile_fortran_codes(gfortran_bin)
    fortran_timings = execute_and_time_fortran(executables)

    # Measure Python / CUDA equivalent performance live
    print("\n" + "=" * 75)
    print(" Measuring Vectorized CUDA / Python Processor Execution Time Live")
    print("=" * 75)
    from python.compare_processing_times import run_python_cuda_qinvariant, run_python_process_all

    print("[+] Running CUDA/Python 3D Q-Criterion Vectorized Processing live...")
    t_py_qinv = run_python_cuda_qinvariant()
    print(f"    CUDA/Python Q-Criterion Time: {t_py_qinv:.3f} s")

    print("\n[+] Running Full Python Dataset Batch Processing live...")
    t_py_all = run_python_process_all()
    print(f"    Full Python Batch Processing Time: {t_py_all:.3f} s")

    t_f_qinv = fortran_timings.get("qinvariant", 0.0)
    speedup = (t_f_qinv / t_py_qinv) if (t_py_qinv > 0 and t_f_qinv > 0) else 1.0

    print("\n" + "=" * 80)
    print(" LIVE RE-COMPILED FORTRAN vs CUDA / PYTHON TIMING SUMMARY")
    print("=" * 80)
    print(f" Fortran qInvariant_win.exe Live Execution Time : {t_f_qinv:.3f} s")
    print(f" CUDA/Python 3D Q-Criterion Live Time           : {t_py_qinv:.3f} s")
    print(f" Real Speedup Factor                            : {speedup:.2f}x")
    print("=" * 80)

    # Save summary
    res_data = {
        "compiler": gfortran_bin,
        "fortran_timings_sec": fortran_timings,
        "cuda_python_timings_sec": {
            "qinvariant_cuda_python": t_py_qinv,
            "full_batch_processor": t_py_all
        },
        "speedup": speedup
    }

    os.makedirs("output_comparison", exist_ok=True)
    out_json = os.path.join("output_comparison", "recompiled_fortran_timings.json")
    with open(out_json, 'w') as f:
        json.dump(res_data, f, indent=2)
    print(f"\n[+] Saved live compilation & timing report: {out_json}")


if __name__ == "__main__":
    run_recompile_benchmark()
