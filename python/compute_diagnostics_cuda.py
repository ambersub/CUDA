#!/usr/bin/env python3
"""
GPU-Accelerated CFD Diagnostic Suite for 3D Compressible Supersonic Channel Flow.

Calculates key flow diagnostics directly on NVIDIA GPU using Numba CUDA:
  1. Kinetic Energy (KE.dat):
     Volume-averaged total kinetic energy: mean(u^2 + v^2 + w^2)
  2. Bulk Density (rhobulk.dat):
     Plane-averaged bulk density across 5 streamwise monitoring stations (x1..x5)
  3. Bulk Velocity (Ubulk.dat):
     Plane-averaged streamwise bulk velocity across the 5 monitoring stations
  4. Bulk Mass Flow Rate (massflow.dat):
     Plane-averaged bulk mass flux rho*u across the 5 monitoring stations
  5. Wall Shear Stress (tauWall.dat):
     Average shear stress at bottom wall (z=0) and top wall (z=2):
       tau_bottom = mean(mu_wall * dudz_bottom) / Re
       tau_top    = mean(mu_wall * dudz_top) / Re

Supports:
  - Direct in-memory computation from CUDASolver device arrays (zero-copy live diagnostics).
  - Post-processing from a single 3D snapshot file (e.g. output_cuda/3D.dat).
  - Batch processing over a time series of 3D snapshots (e.g. output_cuda/3D_step*.dat).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import numba
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Numba CUDA Kernels
# ---------------------------------------------------------------------------

if CUDA_AVAILABLE:
    @cuda.jit
    def kernel_kinetic_energy(u, v, w, out_ke, nx, ny, nz):
        """Computes point-wise kinetic energy: u^2 + v^2 + w^2."""
        i, j, k = cuda.grid(3)
        if i < nx and j < ny and k < nz:
            out_ke[i, j, k] = u[i, j, k] ** 2 + v[i, j, k] ** 2 + w[i, j, k] ** 2

    @cuda.jit
    def kernel_plane_quads(rho, u, plane_idx, out_rho_quad, out_mass_quad, out_u_quad, ny, nz):
        """Computes 2D bilinear quad elements on a given y-z plane."""
        j, k = cuda.grid(2)
        if j < ny - 1 and k < nz - 1:
            p = plane_idx
            r00 = rho[p, j, k]
            r01 = rho[p, j + 1, k]
            r10 = rho[p, j, k + 1]
            r11 = rho[p, j + 1, k + 1]

            u00 = u[p, j, k]
            u01 = u[p, j + 1, k]
            u10 = u[p, j, k + 1]
            u11 = u[p, j + 1, k + 1]

            out_rho_quad[j, k] = 0.25 * (r00 + r01 + r10 + r11)
            out_mass_quad[j, k] = 0.25 * (r00 * u00 + r01 * u01 + r10 * u10 + r11 * u11)
            out_u_quad[j, k] = 0.25 * (u00 + u01 + u10 + u11)

    @cuda.jit
    def kernel_wall_shear(u, T, f0, f1, f2, b0, b1, b2, invRe, out_tau_bot, out_tau_top, nx, ny, nz):
        """Computes bottom and top wall shear stress across the (x, y) wall surfaces."""
        i, j = cuda.grid(2)
        if i < nx and j < ny:
            # Bottom wall (k=0)
            mu_bot = T[i, j, 0] ** 0.7
            dudz_bot = f0 * u[i, j, 0] + f1 * u[i, j, 1] + f2 * u[i, j, 2]
            out_tau_bot[i, j] = mu_bot * dudz_bot * invRe

            # Top wall (k=nz-1)
            mu_top = T[i, j, nz - 1] ** 0.7
            dudz_top = b0 * u[i, j, nz - 1] + b1 * u[i, j, nz - 2] + b2 * u[i, j, nz - 3]
            out_tau_top[i, j] = mu_top * dudz_top * invRe


# ---------------------------------------------------------------------------
# Asymmetric Boundary Finite-Difference Stencils
# ---------------------------------------------------------------------------

def compute_wall_coefficients(z_coords: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Computes 2nd-order asymmetric finite-difference coefficients for dudz at wall boundaries
    matching Fortran main_v1.9_unc.f90 lines 56-90:
      Bottom: f1d2c0, f1d2c1, f1d2c2
      Top:    b1d2c0, b1d2c1, b1d2c2
    """
    # Bottom wall (k=0)
    dz1 = z_coords[1] - z_coords[0]
    alpha_bot = (z_coords[2] - z_coords[1]) / dz1
    tmpR1_bot = dz1 * alpha_bot * (alpha_bot + 1.0)
    f0 = (1.0 - (alpha_bot + 1.0) ** 2) / tmpR1_bot
    f1 = ((alpha_bot + 1.0) ** 2) / tmpR1_bot
    f2 = -1.0 / tmpR1_bot

    # Top wall (k=nz-1)
    dzN = z_coords[-1] - z_coords[-2]
    alpha_top = (z_coords[-2] - z_coords[-3]) / dzN
    tmpR1_top = dzN * alpha_top * (alpha_top + 1.0)
    b0 = ((alpha_top + 1.0) ** 2 - 1.0) / tmpR1_top
    b1 = -((alpha_top + 1.0) ** 2) / tmpR1_top
    b2 = 1.0 / tmpR1_top

    return f0, f1, f2, b0, b1, b2


# ---------------------------------------------------------------------------
# Diagnostics Calculator Core
# ---------------------------------------------------------------------------

def calculate_diagnostics_cuda(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    z_coords: np.ndarray,
    dy: float,
    planes: Optional[List[int]] = None,
    reynolds: float = 3000.0,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Computes all diagnostic metrics (KE, rhobulk, Ubulk, massflow, tauWall)
    using CUDA GPU acceleration where available.

    Inputs:
      rho, u, v, w, T: 3D numpy arrays shape (nx, ny, nz)
      z_coords: 1D array length nz
      dy: grid spacing in y
      planes: list of 5 streamwise plane indices (default: [4, 9, 69, 99, 124])
      reynolds: Reynolds number (default: 3000.0)
    """
    nx, ny, nz = u.shape
    if planes is None:
        # Default 5 streamwise monitoring stations from Fortran plane=[5,10,70,100,125] (1-based)
        planes = [min(idx, nx - 1) for idx in [4, 9, 69, 99, 124]]

    Ayz = z_coords[-1] * (dy * (ny - 1))
    dz_arr = np.diff(z_coords)
    invRe = 1.0 / reynolds
    f0, f1, f2, b0, b1, b2 = compute_wall_coefficients(z_coords)

    can_gpu = use_gpu and CUDA_AVAILABLE

    if can_gpu:
        try:
            # 1. Kinetic Energy on GPU
            tpb3 = (8, 8, 8)
            bpg3 = ((nx + 7) // 8, (ny + 7) // 8, (nz + 7) // 8)
            d_u = cuda.to_device(np.ascontiguousarray(u, dtype=np.float64))
            d_v = cuda.to_device(np.ascontiguousarray(v, dtype=np.float64))
            d_w = cuda.to_device(np.ascontiguousarray(w, dtype=np.float64))
            d_ke = cuda.device_array((nx, ny, nz), dtype=np.float64)

            kernel_kinetic_energy[bpg3, tpb3](d_u, d_v, d_w, d_ke, nx, ny, nz)
            cuda.synchronize()
            ke_val = float(np.mean(d_ke.copy_to_host()))

            # 2. Wall Shear Stress on GPU
            tpb2 = (16, 16)
            bpg2 = ((nx + 15) // 16, (ny + 15) // 16)
            d_T = cuda.to_device(np.ascontiguousarray(T, dtype=np.float64))
            d_tau_bot = cuda.device_array((nx, ny), dtype=np.float64)
            d_tau_top = cuda.device_array((nx, ny), dtype=np.float64)

            kernel_wall_shear[bpg2, tpb2](
                d_u, d_T, f0, f1, f2, b0, b1, b2, invRe, d_tau_bot, d_tau_top, nx, ny, nz
            )
            cuda.synchronize()
            tau_bot = float(np.mean(d_tau_bot.copy_to_host()))
            tau_top = float(np.mean(d_tau_top.copy_to_host()))

            # 3. Streamwise Plane Reductions (rhobulk, Ubulk, massflow)
            d_rho = cuda.to_device(np.ascontiguousarray(rho, dtype=np.float64))
            tpb_plane = (16, 16)
            bpg_plane = ((ny - 1 + 15) // 16, (nz - 1 + 15) // 16)

            d_rq = cuda.device_array((ny - 1, nz - 1), dtype=np.float64)
            d_mq = cuda.device_array((ny - 1, nz - 1), dtype=np.float64)
            d_uq = cuda.device_array((ny - 1, nz - 1), dtype=np.float64)

            dA = dy * dz_arr[None, :]  # shape (1, nz-1) broadcast with (ny-1, nz-1)

            rhobulk_list = []
            ubulk_list = []
            massflow_list = []

            for p in planes:
                kernel_plane_quads[bpg_plane, tpb_plane](
                    d_rho, d_u, p, d_rq, d_mq, d_uq, ny, nz
                )
                cuda.synchronize()
                rq_h = d_rq.copy_to_host()
                mq_h = d_mq.copy_to_host()
                uq_h = d_uq.copy_to_host()

                rhobulk_list.append(float(np.sum(rq_h * dA) / Ayz))
                massflow_list.append(float(np.sum(mq_h * dA) / Ayz))
                ubulk_list.append(float(np.sum(uq_h * dA) / Ayz))

            return {
                "KE": ke_val,
                "rhobulk": rhobulk_list,
                "Ubulk": ubulk_list,
                "massflow": massflow_list,
                "tau_bottom": tau_bot,
                "tau_top": tau_top,
                "tau_avg": (tau_bot - tau_top) / 2.0,
            }
        except Exception as e:
            print(f"    [!] CUDA diagnostic kernel fallback to CPU vectorization: {e}")

    # CPU Vectorized Fallback
    ke_val = float(np.mean(u ** 2 + v ** 2 + w ** 2))

    mu_bot = T[:, :, 0] ** 0.7
    dudz_bot = f0 * u[:, :, 0] + f1 * u[:, :, 1] + f2 * u[:, :, 2]
    tau_bot = float(np.mean(mu_bot * dudz_bot * invRe))

    mu_top = T[:, :, -1] ** 0.7
    dudz_top = b0 * u[:, :, -1] + b1 * u[:, :, -2] + b2 * u[:, :, -3]
    tau_top = float(np.mean(mu_top * dudz_top * invRe))

    rhobulk_list = []
    ubulk_list = []
    massflow_list = []
    dA = dy * dz_arr[None, :]

    for p in planes:
        r_p = rho[p, :, :]
        u_p = u[p, :, :]
        rq = 0.25 * (r_p[:-1, :-1] + r_p[1:, :-1] + r_p[:-1, 1:] + r_p[1:, 1:])
        uq = 0.25 * (u_p[:-1, :-1] + u_p[1:, :-1] + u_p[:-1, 1:] + u_p[1:, 1:])
        mq = 0.25 * (
            r_p[:-1, :-1] * u_p[:-1, :-1]
            + r_p[1:, :-1] * u_p[1:, :-1]
            + r_p[:-1, 1:] * u_p[:-1, 1:]
            + r_p[1:, 1:] * u_p[1:, 1:]
        )
        rhobulk_list.append(float(np.sum(rq * dA) / Ayz))
        massflow_list.append(float(np.sum(mq * dA) / Ayz))
        ubulk_list.append(float(np.sum(uq * dA) / Ayz))

    return {
        "KE": ke_val,
        "rhobulk": rhobulk_list,
        "Ubulk": ubulk_list,
        "massflow": massflow_list,
        "tau_bottom": tau_bot,
        "tau_top": tau_top,
        "tau_avg": (tau_bot - tau_top) / 2.0,
    }


# ---------------------------------------------------------------------------
# Output File Appender (Exact Fortran Formatting)
# ---------------------------------------------------------------------------

def append_diagnostic_record(output_dir: Union[str, Path], time_val: float, diag: Dict[str, Any]) -> None:
    """
    Appends a new record to KE.dat, rhobulk.dat, Ubulk.dat, massflow.dat, and tauWall.dat.
    Formatting matches Fortran formats 108, 109, and 110 exactly.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. KE.dat: format(2(1X,F15.7))
    with open(out_dir / "KE.dat", "a", encoding="utf-8") as f:
        f.write(f" {time_val:15.7f} {diag['KE']:15.7f}\n")

    # 2. rhobulk.dat: format(6(1X,F15.7))
    with open(out_dir / "rhobulk.dat", "a", encoding="utf-8") as f:
        line = f" {time_val:15.7f}" + "".join(f" {v:15.7f}" for v in diag["rhobulk"])
        f.write(line + "\n")

    # 3. Ubulk.dat: format(6(1X,F15.7))
    with open(out_dir / "Ubulk.dat", "a", encoding="utf-8") as f:
        line = f" {time_val:15.7f}" + "".join(f" {v:15.7f}" for v in diag["Ubulk"])
        f.write(line + "\n")

    # 4. massflow.dat: format(6(1X,F15.7))
    with open(out_dir / "massflow.dat", "a", encoding="utf-8") as f:
        line = f" {time_val:15.7f}" + "".join(f" {v:15.7f}" for v in diag["massflow"])
        f.write(line + "\n")

    # 5. tauWall.dat: format(3(1X,F15.7))
    with open(out_dir / "tauWall.dat", "a", encoding="utf-8") as f:
        f.write(f" {time_val:15.7f} {diag['tau_bottom']:15.7f} {diag['tau_top']:15.7f}\n")


# ---------------------------------------------------------------------------
# Ingestion & Dispatch Functions
# ---------------------------------------------------------------------------

def process_3d_snapshot(
    snapshot_path: Union[str, Path],
    output_dir: Union[str, Path] = "output_cuda",
    append: bool = True,
    use_gpu: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Reads a 3D Tecplot snapshot (e.g. 3D.dat or 3D_step*.dat), extracts time and flow fields,
    computes diagnostics via CUDA, and appends to target .dat files.
    """
    path = Path(snapshot_path)
    if not path.is_file():
        print(f"    [!] Error: Snapshot file not found: {path}")
        return None

    # Parse header for time and grid dimensions
    time_val = 0.0
    nstep_val = 0
    nx, ny, nz = 130, 130, 160

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            m_title = re.search(r'TITLE\s*=\s*"([^"]+)"', line, re.IGNORECASE)
            if m_title:
                tokens = m_title.group(1).split()
                if len(tokens) >= 1:
                    try:
                        time_val = float(tokens[0])
                        if len(tokens) >= 2:
                            nstep_val = int(tokens[1])
                    except ValueError:
                        pass
            m_k = re.search(r'k\s*=\s*(\d+)', line, re.IGNORECASE)
            m_j = re.search(r'j\s*=\s*(\d+)', line, re.IGNORECASE)
            m_i = re.search(r'i\s*=\s*(\d+)', line, re.IGNORECASE)
            if m_k and m_j and m_i:
                nz = int(m_k.group(1))
                ny = int(m_j.group(1))
                nx = int(m_i.group(1))

    expected_cells = nx * ny * nz

    try:
        df = pd.read_csv(
            path,
            skiprows=3,
            sep=r"\s+",
            header=None,
            comment="#",
            on_bad_lines="skip",
            engine="c",
            dtype=np.float64,
        )
        if df.shape[0] < expected_cells:
            print(f"    [!] Snapshot '{path.name}' incomplete: {df.shape[0]} rows (expected {expected_cells}).")
            return None

        arr = df.iloc[:expected_cells, :10].to_numpy()
        # Data format: ordered k -> j -> i
        arr_3d = arr.reshape(nz, ny, nx, 10)

        # Transpose to (nx, ny, nz) matching solver indexing:
        # arr_3d axis 0 is k (z), axis 1 is j (y), axis 2 is i (x)
        x_coords = arr_3d[0, 0, :, 0]
        y_coords = arr_3d[0, :, 0, 1]
        z_coords = arr_3d[:, 0, 0, 2]

        dy = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else (4.0 * math.pi / 3.0) / (ny - 1)

        rho = np.transpose(arr_3d[..., 3], (2, 1, 0))
        u   = np.transpose(arr_3d[..., 4], (2, 1, 0))
        v   = np.transpose(arr_3d[..., 5], (2, 1, 0))
        w   = np.transpose(arr_3d[..., 6], (2, 1, 0))
        T   = np.transpose(arr_3d[..., 8], (2, 1, 0))

        # Check for NaN divergence in snapshots
        if np.isnan(u[0, 0, 0]) or np.isnan(rho[0, 0, 0]):
            print(f"    [!] Skipping snapshot '{path.name}': contains NaN values.")
            return None

        diag = calculate_diagnostics_cuda(rho, u, v, w, T, z_coords, dy, use_gpu=use_gpu)
        diag["time"] = time_val
        diag["nstep"] = nstep_val
        diag["source"] = str(path)

        if append:
            append_diagnostic_record(output_dir, time_val, diag)

        return diag
    except Exception as e:
        print(f"    [!] Error processing snapshot '{path.name}': {e}")
        return None


def process_snapshot_series(
    directory: Union[str, Path] = "output_cuda",
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "3D*.dat",
    use_gpu: bool = True,
) -> int:
    """
    Finds and processes all 3D snapshot files in a directory sorted chronologically by step/time.
    Creates fresh or appends to diagnostic time series files.
    """
    src_dir = Path(directory)
    dest_dir = Path(output_dir) if output_dir else src_dir

    all_files = list(src_dir.glob(pattern))
    if not all_files:
        print(f"    [INFO] No snapshots found matching '{pattern}' in '{src_dir}'.")
        return 0

    # Extract step number from filename for natural chronological sorting
    def get_step(p: Path) -> int:
        m = re.search(r'step(\d+)', p.name)
        if m:
            return int(m.group(1))
        m2 = re.search(r'3D(\d+)', p.name)
        if m2:
            return int(m2.group(1))
        return 0

    all_files.sort(key=lambda p: (get_step(p), p.name))
    print(f"\n[+] Batch processing {len(all_files)} 3D snapshot(s) on CUDA from '{src_dir}'...")

    processed_count = 0
    t0 = time.time()

    for p in all_files:
        sys.stdout.write(f"    Processing {p.name:<25}... ")
        sys.stdout.flush()
        diag = process_3d_snapshot(p, output_dir=dest_dir, append=True, use_gpu=use_gpu)
        if diag:
            sys.stdout.write(f"DONE (t={diag['time']:.4f}, KE={diag['KE']:.5f}, tau_w={diag['tau_avg']:.6f})\n")
            processed_count += 1
        else:
            sys.stdout.write("SKIPPED\n")

    print(f"[+] Diagnostic batch processing finished: {processed_count} files in {time.time() - t0:.2f} s.")
    print(f"    Diagnostic files updated in: {dest_dir.resolve()}")
    return processed_count


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUDA-Accelerated CFD Diagnostic Extractor for 3D Channel Flow Snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", default=None,
                        help="Path to single 3D snapshot file (e.g. output_cuda/3D.dat).")
    parser.add_argument("--dir", "-d", default=None,
                        help="Directory containing 3D snapshots to batch process (e.g. output_cuda).")
    parser.add_argument("--pattern", default="3D_step*.dat",
                        help="Filename glob pattern when using --dir.")
    parser.add_argument("--output-dir", "-o", default="output_cuda",
                        help="Destination directory for output .dat files (KE.dat, massflow.dat, etc.).")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU kernels and run on CPU vectorized fallback.")

    args = parser.parse_args()
    use_gpu = not args.no_gpu and CUDA_AVAILABLE

    if args.input:
        print(f"\n[+] Computing diagnostics for single snapshot: {args.input}")
        diag = process_3d_snapshot(args.input, output_dir=args.output_dir, append=True, use_gpu=use_gpu)
        if diag:
            print(f"    Time          : {diag['time']:.6f}")
            print(f"    Kinetic Energy: {diag['KE']:.7f}")
            print(f"    Bulk Density  : {diag['rhobulk']}")
            print(f"    Bulk Velocity : {diag['Ubulk']}")
            print(f"    Bulk Mass Flow: {diag['massflow']}")
            print(f"    Wall Shear    : Bottom={diag['tau_bottom']:.7f}, Top={diag['tau_top']:.7f}, Avg={diag['tau_avg']:.7f}")
    elif args.dir:
        process_snapshot_series(args.dir, output_dir=args.output_dir, pattern=args.pattern, use_gpu=use_gpu)
    else:
        # Default: process output_cuda/3D.dat if it exists, or batch process snapshots
        default_single = Path("output_cuda/3D.dat")
        if default_single.is_file():
            print(f"[+] Processing default snapshot: {default_single}")
            process_3d_snapshot(default_single, output_dir=args.output_dir, append=True, use_gpu=use_gpu)
        else:
            process_snapshot_series("output_cuda", output_dir=args.output_dir, pattern=args.pattern, use_gpu=use_gpu)


if __name__ == "__main__":
    main()
