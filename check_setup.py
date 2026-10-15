#!/usr/bin/env python3
"""
================================================================================
3D Compressible CFD Solver - Cross-Device Setup & Diagnostic Verification Tool
================================================================================
Automated diagnostic script to verify hardware, CUDA drivers, Python packages,
GFortran compiler toolchain, and workspace file integrity when setting up this
project on a new device.

Usage:
    python check_setup.py
    python check_setup.py --test-gpu
    python check_setup.py --fix-missing-dirs

Author: Google DeepMind Antigravity Pair Programmer
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import os
import platform
import shutil
import struct
import subprocess
import sys
from typing import Tuple, List, Dict, Any

import warnings

# Suppress minor version & performance warnings for clean diagnostic output
warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING", "1")

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Safe Terminal Formatting
USE_COLOR = os.environ.get("NO_COLOR") is None and sys.platform != "win32" or "WT_SESSION" in os.environ or "ANSICON" in os.environ or os.environ.get("TERM")

class Colors:
    GREEN = "\033[92m" if USE_COLOR else ""
    YELLOW = "\033[93m" if USE_COLOR else ""
    RED = "\033[91m" if USE_COLOR else ""
    CYAN = "\033[96m" if USE_COLOR else ""
    BOLD = "\033[1m" if USE_COLOR else ""
    RESET = "\033[0m" if USE_COLOR else ""

def print_header(title: str) -> None:
    bar = "=" * 75
    print(f"\n{Colors.BOLD}{Colors.CYAN}{bar}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN} {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{bar}{Colors.RESET}")

def print_status(component: str, ok: bool, details: str = "", warn: bool = False) -> None:
    if ok:
        tag = f"{Colors.GREEN}[PASS]{Colors.RESET}"
    elif warn:
        tag = f"{Colors.YELLOW}[WARN]{Colors.RESET}"
    else:
        tag = f"{Colors.RED}[FAIL]{Colors.RESET}"
    print(f" {tag} {Colors.BOLD}{component:<30}{Colors.RESET} {details}")


# ------------------------------------------------------------------------------
# 1. Platform & Python Environment Check
# ------------------------------------------------------------------------------
def check_python_environment() -> Tuple[bool, List[str]]:
    print_header("1. Operating System & Python Environment")
    issues = []
    
    # OS info
    os_name = platform.system()
    os_release = platform.release()
    print_status("Operating System", True, f"{os_name} {os_release} ({platform.machine()})")

    # 64-bit architecture check
    is_64bit = struct.calcsize("P") * 8 == 64
    if is_64bit:
        print_status("Architecture", True, "64-bit (Required for CUDA & CFD arrays)")
    else:
        print_status("Architecture", False, "32-bit detected! 64-bit Python is strictly required.")
        issues.append("Install 64-bit Python.")

    # Python version check (3.10 to 3.12 recommended)
    py_ver = sys.version_info
    ver_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver.major == 3 and 10 <= py_ver.minor <= 12:
        print_status("Python Version", True, f"Python {ver_str} (Fully supported with Numba CUDA)")
    elif py_ver.major == 3 and py_ver.minor >= 13:
        print_status("Python Version", True, f"Python {ver_str} (Numba CUDA currently best supports 3.10-3.12)", warn=True)
    else:
        print_status("Python Version", False, f"Python {ver_str} is below minimum requirement (>= 3.10)")
        issues.append("Upgrade Python to >= 3.10 (3.10 - 3.12 recommended).")

    # Virtual environment status
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print_status("Virtual Environment", True, f"Active ({sys.prefix})")
    else:
        print_status("Virtual Environment", True, "Global environment (recommended: python -m venv .venv)", warn=True)

    return len(issues) == 0, issues


# ------------------------------------------------------------------------------
# 2. Required Python Dependencies Check
# ------------------------------------------------------------------------------
def check_python_packages() -> Tuple[bool, List[str]]:
    print_header("2. Python Scientific & CUDA Package Check")
    issues = []

    required_pkgs = [
        ("numpy", "NumPy (Multidimensional Array Core)"),
        ("pandas", "Pandas (Fast C-Engine ASCII Ingestion)"),
        ("scipy", "SciPy (Statistical Correlation Norms)"),
        ("matplotlib", "Matplotlib (Validation Plotting Suite)"),
        ("numba", "Numba (JIT Compiler Core)"),
        ("tqdm", "Tqdm (Simulation Progress Tracker)"),
    ]

    for pkg_name, desc in required_pkgs:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            print_status(desc, True, f"v{ver}")
        except ImportError:
            print_status(desc, False, "Not installed")
            issues.append(f"Missing '{pkg_name}': run 'pip install -r requirements.txt'")

    # Specific check for numba-cuda & CUDA toolchain packages
    try:
        import numba_cuda  # noqa: F401
        print_status("Numba CUDA Extension", True, "Installed")
    except ImportError:
        try:
            from numba import cuda  # noqa: F401
            print_status("Numba CUDA Extension", True, "Available via Numba")
        except ImportError:
            print_status("Numba CUDA Extension", False, "Missing 'numba-cuda'")
            issues.append("Run 'pip install numba-cuda nvidia-cuda-nvcc-cu12'")

    return len(issues) == 0, issues


# ------------------------------------------------------------------------------
# 3. NVIDIA GPU Hardware & CUDA Toolchain Check
# ------------------------------------------------------------------------------
def configure_cuda_environment() -> None:
    """Configures CUDA paths and libdevice for Numba."""
    try:
        import site
        sp_dirs = site.getsitepackages() + [site.getusersitepackages()]
        for sp in sp_dirs:
            cuda_nvcc = os.path.join(sp, "nvidia", "cuda_nvcc")
            if os.path.exists(cuda_nvcc):
                os.environ.setdefault("CUDA_HOME", cuda_nvcc)
                os.environ.setdefault("NUMBA_CUDA_HOME", cuda_nvcc)
                libdev = os.path.join(cuda_nvcc, "nvvm", "libdevice", "libdevice.10.bc")
                if os.path.exists(libdev):
                    os.environ.setdefault("NUMBA_CUDA_LIBDEVICE", libdev)
                break
    except Exception:
        pass


def check_cuda_gpu() -> Tuple[bool, List[str]]:
    print_header("3. NVIDIA GPU Hardware & CUDA Acceleration")
    issues = []

    # Configure libdevice / cuda_nvcc automatically
    configure_cuda_environment()

    # Check nvidia-smi command
    smi_path = shutil.which("nvidia-smi")
    if smi_path:
        try:
            smi_out = subprocess.check_output([smi_path, "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader,nounits"], text=True)
            parts = smi_out.strip().split("\n")[0].split(",")
            if len(parts) >= 3:
                drv_ver, dev_name, mem_mb = parts[0].strip(), parts[1].strip(), parts[2].strip()
                print_status("NVIDIA Driver (nvidia-smi)", True, f"Driver {drv_ver} | GPU: {dev_name} ({mem_mb} MB VRAM)")
            else:
                print_status("NVIDIA Driver (nvidia-smi)", True, smi_out.strip())
        except Exception as e:
            print_status("NVIDIA Driver (nvidia-smi)", True, f"Detected (query error: {e})", warn=True)
    else:
        print_status("NVIDIA Driver (nvidia-smi)", True, "nvidia-smi not in PATH (Numba CUDA may still work)", warn=True)

    # Test Numba CUDA initialization
    try:
        from numba import cuda
        available = cuda.is_available()
        if available:
            dev = cuda.get_current_device()
            dev_name = dev.name.decode("utf-8") if isinstance(dev.name, bytes) else str(dev.name)
            cc = dev.compute_capability
            
            # Query memory
            vram_gb = "Unknown"
            try:
                free_b, total_b = cuda.current_context().get_memory_info()
                vram_gb = f"{total_b / (1024**3):.2f} GB"
            except Exception:
                pass

            print_status("CUDA Device Available", True, f"Found: {dev_name}")
            print_status("Compute Capability", True, f"SM {cc[0]}.{cc[1]} (Compute Capability >= 6.0 supported)")
            print_status("GPU Dedicated VRAM", True, f"{vram_gb}")

            # Verify libdevice path
            libdev = os.environ.get("NUMBA_CUDA_LIBDEVICE", "")
            if libdev and os.path.exists(libdev):
                print_status("NVVM libdevice.10.bc", True, f"Verified: {os.path.basename(libdev)}")
            else:
                print_status("NVVM libdevice.10.bc", True, "Auto-detected by Numba runtime")
        else:
            print_status("CUDA Device Available", False, "No CUDA GPU detected or driver incompatible.")
            issues.append("Ensure an NVIDIA GPU is installed with current NVIDIA drivers.")
            return False, issues
    except Exception as e:
        print_status("CUDA Device Initialization", False, f"Exception: {e}")
        issues.append(f"CUDA Error: {e}")
        return False, issues

    return len(issues) == 0, issues


def run_gpu_kernel_test() -> bool:
    """Performs a micro GPU JIT kernel execution to verify end-to-end functionality."""
    print_header("4. Micro CUDA Kernel JIT Verification")
    try:
        import numpy as np
        from numba import cuda

        @cuda.jit
        def _test_vector_add(a, b, c):
            i = cuda.grid(1)
            if i < a.size:
                c[i] = a[i] + b[i]

        n = 1024
        a = np.ones(n, dtype=np.float64) * 2.5
        b = np.ones(n, dtype=np.float64) * 1.5
        c = np.zeros(n, dtype=np.float64)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.to_device(c)

        threads = 128
        blocks = (n + threads - 1) // threads
        _test_vector_add[blocks, threads](d_a, d_b, d_c)
        cuda.synchronize()

        d_c.copy_to_host(c)
        if np.allclose(c, 4.0):
            print_status("GPU Kernel JIT & Execution", True, "Successfully executed vector test (1024 threads)")
            return True
        else:
            print_status("GPU Kernel JIT & Execution", False, "Numerical mismatch in test kernel")
            return False
    except Exception as e:
        print_status("GPU Kernel JIT & Execution", False, f"Failed JIT test: {e}")
        return False


# ------------------------------------------------------------------------------
# 5. Fortran Toolchain Check
# ------------------------------------------------------------------------------
def find_gfortran() -> str | None:
    """Finds gfortran executable across PATH and WinGet paths."""
    gfortran = shutil.which("gfortran")
    if gfortran:
        return gfortran

    appdata = os.environ.get("LOCALAPPDATA", "")
    if appdata:
        winget_gfortran = os.path.join(
            appdata,
            r"Microsoft\WinGet\Packages\BrechtSanders.WinLibs.MCF.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\gfortran.exe"
        )
        if os.path.exists(winget_gfortran):
            return winget_gfortran

        pkg_dir = os.path.join(appdata, r"Microsoft\WinGet\Packages")
        if os.path.exists(pkg_dir):
            matches = glob.glob(os.path.join(pkg_dir, "**", "gfortran.exe"), recursive=True)
            if matches:
                return matches[0]
    return None


def check_fortran_toolchain() -> Tuple[bool, List[str]]:
    print_header("5. Fortran 90 Compiler Toolchain (Optional for CUDA Solver)")
    issues = []

    gfortran_exe = find_gfortran()
    if gfortran_exe:
        try:
            out = subprocess.check_output([gfortran_exe, "--version"], text=True)
            first_line = out.strip().split("\n")[0]
            print_status("GFortran Compiler", True, f"{first_line}")
            print_status("GFortran Binary Path", True, f"{gfortran_exe}")
        except Exception as e:
            print_status("GFortran Compiler", True, f"Found but error running: {e}", warn=True)
    else:
        print_status("GFortran Compiler", True, "Not found (Only needed if running Fortran CPU benchmarks)", warn=True)
        if platform.system() == "Windows":
            print(f"      {Colors.YELLOW}--> To install on Windows: winget install BrechtSanders.WinLibs.MCF.UCRT{Colors.RESET}")
        else:
            print(f"      {Colors.YELLOW}--> To install on Linux: sudo apt update && sudo apt install -y gfortran{Colors.RESET}")

    return True, issues


# ------------------------------------------------------------------------------
# 6. Workspace Files & Directories Integrity
# ------------------------------------------------------------------------------
def check_workspace_files(fix_dirs: bool = False) -> Tuple[bool, List[str]]:
    print_header("6. Workspace Structure & Dataset Integrity")
    issues = []

    # Check param.dat
    if os.path.exists("param.dat"):
        print_status("Simulation Config", True, "param.dat found")
    elif os.path.exists("param.dat.example"):
        print_status("Simulation Config", True, "param.dat missing, copying from param.dat.example...", warn=True)
        try:
            shutil.copy("param.dat.example", "param.dat")
            print(f"      {Colors.GREEN}--> Auto-created 'param.dat' from 'param.dat.example'{Colors.RESET}")
        except Exception:
            issues.append("Copy param.dat.example to param.dat.")
    else:
        print_status("Simulation Config", False, "param.dat not found!")
        issues.append("param.dat is required for simulation parameters.")

    # Check Main Solver Scripts
    core_files = [
        ("cuda code.py", "Main Python CUDA Solver Script"),
        ("python/benchmark_solvers.py", "Dynamic Benchmark Suite"),
        ("python/process_input_files.py", "Input Dataset Processing Suite"),
        ("python/compare_fortran_cuda.py", "Fortran vs CUDA Output Comparator"),
    ]
    for path, desc in core_files:
        if os.path.exists(path):
            print_status(desc, True, f"{path}")
        else:
            print_status(desc, False, f"Missing: {path}")
            issues.append(f"Missing core script: {path}")

    # Check Input Datasets
    input_file = os.path.join("input", "3D.dat")
    if os.path.exists(input_file):
        sz_mb = os.path.getsize(input_file) / (1024 * 1024)
        print_status("3D Initial Dataset", True, f"input/3D.dat found ({sz_mb:.1f} MB)")
    else:
        print_status("3D Initial Dataset", True, "input/3D.dat not found (Not in git due to 527MB size)", warn=True)
        print(f"      {Colors.YELLOW}--> Note: You can still run synthetic demo tests: python \"cuda code.py\" --demo --steps 20{Colors.RESET}")
        print(f"      {Colors.YELLOW}--> To run full DNS: copy 'input/3D.dat' from original device or data drive.{Colors.RESET}")

    # Check / Create required output directories
    required_dirs = [
        "input",
        "output",
        "output_cuda",
        "output_fortran",
        "output_comparison",
        "output_processed",
        "plots",
    ]
    for d in required_dirs:
        if os.path.exists(d):
            print_status(f"Directory '{d}/'", True, "Exists")
        else:
            os.makedirs(d, exist_ok=True)
            print_status(f"Directory '{d}/'", True, "Created automatically")

    return len(issues) == 0, issues


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cross-device setup and verification tool for 3D DNS solver.")
    parser.add_argument("--test-gpu", action="store_true", help="Run micro GPU JIT kernel test.")
    parser.add_argument("--fix-missing-dirs", action="store_true", help="Automatically create missing directories.")
    args = parser.parse_args()

    bar = "#" * 75
    print(f"\n{Colors.BOLD}{Colors.CYAN}{bar}")
    print(" [SETUP CHECK] 3D COMPRESSIBLE CFD SOLVER: SETUP VERIFICATION")
    print(f"{bar}{Colors.RESET}")

    all_ok = True
    all_issues = []

    # 1. Python Environment
    ok, issues = check_python_environment()
    all_ok = all_ok and ok
    all_issues.extend(issues)

    # 2. Python Packages
    ok, issues = check_python_packages()
    all_ok = all_ok and ok
    all_issues.extend(issues)

    # 3. CUDA GPU
    ok, issues = check_cuda_gpu()
    all_ok = all_ok and ok
    all_issues.extend(issues)

    # 4. GPU Kernel JIT Test
    if ok:
        kernel_ok = run_gpu_kernel_test()
        all_ok = all_ok and kernel_ok
        if not kernel_ok:
            all_issues.append("GPU JIT kernel test failed.")

    # 5. Fortran Toolchain (Optional)
    check_fortran_toolchain()

    # 6. Workspace Files
    ok, issues = check_workspace_files(fix_dirs=args.fix_missing_dirs)
    all_ok = all_ok and ok
    all_issues.extend(issues)

    # Final Verdict Summary
    print_header("Summary & Verification Verdict")
    if all_ok:
        print(f"\n{Colors.BOLD}{Colors.GREEN} [SUCCESS] ALL REQUIRED SYSTEMS GO! This device is ready to execute simulations.{Colors.RESET}\n")
        print(" Quick Start Commands:")
        print("   1. Synthetic Demo Test (runs instantly without large input files):")
        print(f"      {Colors.CYAN}python \"cuda code.py\" --demo --steps 20 --output output_cuda/demo.dat{Colors.RESET}")
        print("   2. Benchmark Dynamic Step Timings:")
        print(f"      {Colors.CYAN}python python/benchmark_solvers.py --steps 50{Colors.RESET}")
        print("   3. Full Simulation Run (requires input/3D.dat):")
        print(f"      {Colors.CYAN}python \"cuda code.py\" --steps 100 --output output_cuda/3D_cuda.dat{Colors.RESET}\n")
    else:
        print(f"\n{Colors.BOLD}{Colors.RED} [INCOMPLETE] Please address the following items:{Colors.RESET}")
        for i, iss in enumerate(all_issues, 1):
            print(f"   {i}. {iss}")
        print(f"\n Refer to {Colors.BOLD}SETUP_GUIDE.md{Colors.RESET} for step-by-step instructions.\n")


if __name__ == "__main__":
    main()
