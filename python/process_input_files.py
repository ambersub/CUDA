#!/usr/bin/env python3
"""
Comprehensive Input File Processing Script for 3D Compressible CFD Solver.

Processes all input datasets in input/:
  1. 3D.dat               : 3D Tecplot instantaneous flow state snapshot (x, y, z, rho, u, v, w, e, T, p)
  2. Prime3D.dat          : 3D Tecplot fluctuation fields (x, y, z, rho', u', v', w', e', T', p')
  3. PrimeUV.dat          : 2D/3D fluctuation components (u', v')
  4. RMSu2v2w2t2uvvt.dat  : 1D wall-normal RMS turbulence profiles (z, u_rms^2, v_rms^2, w_rms^2, T_rms^2, uv_rms, vt_rms)
  5. rhouvwtpmu.dat       : 1D wall-normal mean flow profiles (z, rho_bar, u_bar, v_bar, w_bar, T_bar, p_bar, mu_bar)

Author: Google DeepMind Antigravity Pair Programmer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility & Parsing Helpers
# ---------------------------------------------------------------------------

def parse_tecplot_header(filepath: str) -> Tuple[int, int, int, int, list[str]]:
    """
    Scans Tecplot header lines to extract zone dimensions (nz, ny, nx),
    header line offset, and variable names.
    """
    nx, ny, nz = 130, 130, 160
    header_lines = 0
    variables = []

    with open(filepath, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                header_lines += 1
                continue

            # Check if line contains numbers (start of data)
            # If the first token can be converted to float, header is done
            tokens = line_str.split()
            if tokens and is_float(tokens[0]):
                break

            header_lines += 1

            if "ZONE" in line_str.upper():
                # Extract k=..., j=..., i=...
                for part in line_str.replace(',', ' ').split():
                    if part.lower().startswith('k='):
                        nz = int(part.split('=')[1])
                    elif part.lower().startswith('j='):
                        ny = int(part.split('=')[1])
                    elif part.lower().startswith('i='):
                        nx = int(part.split('=')[1])

            if "VARIABLES" in line_str.upper():
                # Extract variable strings inside quotes
                import re
                vars_found = re.findall(r'"([^"]*)"', line_str)
                if vars_found:
                    variables = vars_found

    return header_lines, nz, ny, nx, variables


def is_float(val_str: str) -> bool:
    try:
        float(val_str)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Individual File Processors
# ---------------------------------------------------------------------------

def process_3d_dat(filepath: str) -> Dict[str, Any]:
    """Processes input/3D.dat (Full 3D flow state snapshot)."""
    t0 = time.time()
    print(f"\n[1/5] Processing 3D Snapshot: {filepath}")

    header_lines, nz, ny, nx, vars_found = parse_tecplot_header(filepath)
    expected_points = nx * ny * nz
    print(f"      Header offset: {header_lines} lines | Grid dimensions: Nz={nz}, Ny={ny}, Nx={nx} ({expected_points:,} cells)")

    # Read data efficiently using pandas C-engine
    df = pd.read_csv(
        filepath,
        skiprows=header_lines,
        sep=r'\s+',
        header=None,
        engine='c',
        dtype=np.float64
    )

    # Validate shape
    if df.shape[1] < 10:
        raise ValueError(f"Expected 10 columns in 3D.dat, found {df.shape[1]}")

    # Columns: x, y, z, rho, u, v, w, e, T, p
    cols = ["x", "y", "z", "rho", "u", "v", "w", "e", "T", "p"]
    df = df.iloc[:, :10]
    df.columns = cols

    # Domain bounds
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()

    # Calculate mean statistics across homogeneous (x-y) planes for each z level
    # Since DATAPACKING=POINT in k -> j -> i order, reshape array: (nz, ny, nx, 10)
    arr = df.to_numpy().reshape(nz, ny, nx, 10)

    z_coords = arr[:, 0, 0, 2]
    rho_mean = arr[:, :, :, 3].mean(axis=(1, 2))
    u_mean = arr[:, :, :, 4].mean(axis=(1, 2))
    v_mean = arr[:, :, :, 5].mean(axis=(1, 2))
    w_mean = arr[:, :, :, 6].mean(axis=(1, 2))
    T_mean = arr[:, :, :, 8].mean(axis=(1, 2))
    p_mean = arr[:, :, :, 9].mean(axis=(1, 2))

    # Bulk flow properties
    rho_bulk = np.mean(rho_mean)
    u_bulk = np.mean(u_mean)

    # Wall shear stress estimation at z=0 (bottom wall)
    dz_wall = z_coords[1] - z_coords[0]
    dudz_wall = (u_mean[1] - u_mean[0]) / dz_wall
    # Viscosity mu ~ T^0.7 (Sutherland estimate or Re ref=3000)
    re_ref = 3000.0
    tau_wall = (1.0 / re_ref) * dudz_wall
    cf = 2.0 * tau_wall / (rho_bulk * u_bulk**2 + 1e-15)

    elapsed = time.time() - t0
    print(f"      Completed in {elapsed:.2f} s")
    print(f"      Domain bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")
    print(f"      Bulk Density (rho_b): {rho_bulk:.6f} | Bulk Velocity (U_b): {u_bulk:.6f}")
    print(f"      Wall Shear Stress (tau_w): {tau_wall:.6e} | Skin Friction (Cf): {cf:.6e}")

    return {
        "file": os.path.basename(filepath),
        "grid": {"nx": nx, "ny": ny, "nz": nz, "total_cells": expected_points},
        "bounds": {
            "x": [float(x_min), float(x_max)],
            "y": [float(y_min), float(y_max)],
            "z": [float(z_min), float(z_max)]
        },
        "bulk_properties": {
            "rho_bulk": float(rho_bulk),
            "u_bulk": float(u_bulk),
            "tau_wall": float(tau_wall),
            "skin_friction_Cf": float(cf)
        },
        "mean_profiles": {
            "z": z_coords.tolist(),
            "rho_mean": rho_mean.tolist(),
            "u_mean": u_mean.tolist(),
            "T_mean": T_mean.tolist(),
            "p_mean": p_mean.tolist()
        }
    }


def process_prime3d_dat(filepath: str) -> Dict[str, Any]:
    """Processes input/Prime3D.dat (3D fluctuation dataset)."""
    t0 = time.time()
    print(f"\n[2/5] Processing Prime3D Fluctuation Dataset: {filepath}")

    header_lines, nz, ny, nx, vars_found = parse_tecplot_header(filepath)
    print(f"      Header offset: {header_lines} lines | Dimensions: Nz={nz}, Ny={ny}, Nx={nx}")

    df = pd.read_csv(
        filepath,
        skiprows=header_lines,
        sep=r'\s+',
        header=None,
        engine='c',
        dtype=np.float64
    )

    cols = ["x", "y", "z", "rho_prime", "u_prime", "v_prime", "w_prime", "e_prime", "T_prime", "p_prime"]
    df = df.iloc[:, :10]
    df.columns = cols

    arr = df.to_numpy().reshape(nz, ny, nx, 10)
    z_coords = arr[:, 0, 0, 2]

    # Calculate RMS profiles: sqrt(mean(q'^2)) across x-y planes
    u_prime = arr[:, :, :, 4]
    v_prime = arr[:, :, :, 5]
    w_prime = arr[:, :, :, 6]
    T_prime = arr[:, :, :, 8]

    u_rms = np.sqrt(np.mean(u_prime**2, axis=(1, 2)))
    v_rms = np.sqrt(np.mean(v_prime**2, axis=(1, 2)))
    w_rms = np.sqrt(np.mean(w_prime**2, axis=(1, 2)))
    T_rms = np.sqrt(np.mean(T_prime**2, axis=(1, 2)))
    uv_rms = np.mean(u_prime * v_prime, axis=(1, 2))

    # TKE profile: 0.5 * (u_rms^2 + v_rms^2 + w_rms^2)
    tke = 0.5 * (u_rms**2 + v_rms**2 + w_rms**2)

    peak_u_rms_idx = np.argmax(u_rms)
    peak_u_rms_z = z_coords[peak_u_rms_idx]
    peak_u_rms_val = u_rms[peak_u_rms_idx]

    elapsed = time.time() - t0
    print(f"      Completed in {elapsed:.2f} s")
    print(f"      Peak Streamwise Turbulence Intensity (u_rms): {peak_u_rms_val:.6f} at z = {peak_u_rms_z:.4f}")
    print(f"      Max TKE (Turbulent Kinetic Energy): {np.max(tke):.6f}")

    return {
        "file": os.path.basename(filepath),
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "turbulence_statistics": {
            "max_u_rms": float(peak_u_rms_val),
            "max_u_rms_location_z": float(peak_u_rms_z),
            "max_v_rms": float(np.max(v_rms)),
            "max_w_rms": float(np.max(w_rms)),
            "max_T_rms": float(np.max(T_rms)),
            "max_tke": float(np.max(tke))
        },
        "rms_profiles": {
            "z": z_coords.tolist(),
            "u_rms": u_rms.tolist(),
            "v_rms": v_rms.tolist(),
            "w_rms": w_rms.tolist(),
            "T_rms": T_rms.tolist(),
            "uv_rms": uv_rms.tolist(),
            "tke": tke.tolist()
        }
    }


def process_primeuv_dat(filepath: str) -> Dict[str, Any]:
    """Processes input/PrimeUV.dat (2D fluctuation component fields)."""
    t0 = time.time()
    print(f"\n[3/5] Processing PrimeUV Fluctuation Dataset: {filepath}")

    # File has 2 columns: u', v'
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        engine='c',
        dtype=np.float64
    )

    num_rows = len(df)
    u_prime = df.iloc[:, 0].to_numpy()
    v_prime = df.iloc[:, 1].to_numpy()

    u_mean, u_std = float(np.mean(u_prime)), float(np.std(u_prime))
    v_mean, v_std = float(np.mean(v_prime)), float(np.std(v_prime))
    uv_cov = float(np.mean(u_prime * v_prime))

    elapsed = time.time() - t0
    print(f"      Completed in {elapsed:.2f} s ({num_rows:,} data rows)")
    print(f"      u' Fluctuation: Mean={u_mean:.6e}, StdDev={u_std:.6f}")
    print(f"      v' Fluctuation: Mean={v_mean:.6e}, StdDev={v_std:.6f}")
    print(f"      Covariance <u'v'>: {uv_cov:.6e}")

    return {
        "file": os.path.basename(filepath),
        "total_rows": num_rows,
        "u_prime_stats": {"mean": u_mean, "std": u_std, "min": float(np.min(u_prime)), "max": float(np.max(u_prime))},
        "v_prime_stats": {"mean": v_mean, "std": v_std, "min": float(np.min(v_prime)), "max": float(np.max(v_prime))},
        "covariance_uv": uv_cov
    }


def process_rms_profile_dat(filepath: str) -> Dict[str, Any]:
    """Processes input/RMSu2v2w2t2uvvt.dat (1D wall-normal RMS statistics profile)."""
    t0 = time.time()
    print(f"\n[4/5] Processing 1D RMS Profile Dataset: {filepath}")

    # Columns: z, u_rms^2, v_rms^2, w_rms^2, T_rms^2, uv_rms, vt_rms
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        engine='c',
        dtype=np.float64
    )

    cols = ["z", "u2_rms", "v2_rms", "w2_rms", "T2_rms", "uv_rms", "vt_rms"]
    df = df.iloc[:, :7]
    df.columns = cols

    z = df["z"].to_numpy()
    u_rms = np.sqrt(np.maximum(df["u2_rms"].to_numpy(), 0.0))
    v_rms = np.sqrt(np.maximum(df["v2_rms"].to_numpy(), 0.0))
    w_rms = np.sqrt(np.maximum(df["w2_rms"].to_numpy(), 0.0))
    T_rms = np.sqrt(np.maximum(df["T2_rms"].to_numpy(), 0.0))

    peak_u = float(np.max(u_rms))
    peak_u_z = float(z[np.argmax(u_rms)])

    elapsed = time.time() - t0
    print(f"      Completed in {elapsed:.2f} s ({len(z)} wall-normal grid points)")
    print(f"      Peak u_rms: {peak_u:.6f} at z = {peak_u_z:.4f}")
    print(f"      Peak v_rms: {np.max(v_rms):.6f} | Peak w_rms: {np.max(w_rms):.6f} | Peak T_rms: {np.max(T_rms):.6f}")

    return {
        "file": os.path.basename(filepath),
        "num_points": len(z),
        "peaks": {
            "u_rms_max": peak_u,
            "u_rms_max_z": peak_u_z,
            "v_rms_max": float(np.max(v_rms)),
            "w_rms_max": float(np.max(w_rms)),
            "T_rms_max": float(np.max(T_rms)),
            "min_uv_rms": float(np.min(df["uv_rms"]))
        },
        "profiles": df.to_dict(orient="list")
    }


def process_mean_profile_dat(filepath: str) -> Dict[str, Any]:
    """Processes input/rhouvwtpmu.dat (1D wall-normal mean flow profile)."""
    t0 = time.time()
    print(f"\n[5/5] Processing 1D Mean Profile Dataset: {filepath}")

    # Columns: z, rho_bar, u_bar, v_bar, w_bar, T_bar, p_bar, mu_bar
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        engine='c',
        dtype=np.float64
    )

    cols = ["z", "rho", "u", "v", "w", "T", "p", "mu"]
    df = df.iloc[:, :8]
    df.columns = cols

    z = df["z"].to_numpy()
    rho = df["rho"].to_numpy()
    u = df["u"].to_numpy()
    v = df["v"].to_numpy()
    w = df["w"].to_numpy()
    T = df["T"].to_numpy()
    p = df["p"].to_numpy()
    mu = df["mu"].to_numpy()

    # Calculate friction velocity u_tau and law of the wall coordinates
    dz = z[1] - z[0]
    dudz_wall = (u[1] - u[0]) / dz
    tau_wall = mu[0] * dudz_wall / 3000.0  # scaled by Re=3000
    u_tau = np.sqrt(tau_wall / (rho[0] + 1e-15))
    nu_wall = (mu[0] / 3000.0) / rho[0]
    z_plus = z * u_tau / (nu_wall + 1e-15)
    u_plus = u / (u_tau + 1e-15)

    # Estimate Log-law parameters in 30 <= z+ <= 100
    log_mask = (z_plus >= 30.0) & (z_plus <= 100.0)
    if np.any(log_mask):
        log_z = np.log(z_plus[log_mask])
        slope, intercept = np.polyfit(log_z, u_plus[log_mask], 1)
        kappa = 1.0 / slope if slope != 0 else 0.41
        B = intercept
    else:
        kappa, B = 0.41, 5.2

    elapsed = time.time() - t0
    print(f"      Completed in {elapsed:.2f} s ({len(z)} wall-normal grid points)")
    print(f"      Friction Velocity (u_tau): {u_tau:.6f} | Wall viscosity (nu_w): {nu_wall:.6e}")
    print(f"      Log-law fit: u+ = (1/{kappa:.3f})*ln(z+) + {B:.2f}")

    return {
        "file": os.path.basename(filepath),
        "num_points": len(z),
        "wall_units": {
            "tau_wall": float(tau_wall),
            "u_tau": float(u_tau),
            "nu_wall": float(nu_wall),
            "kappa_fit": float(kappa),
            "B_fit": float(B)
        },
        "profiles": {
            "z": z.tolist(),
            "z_plus": z_plus.tolist(),
            "u_plus": u_plus.tolist(),
            "rho": rho.tolist(),
            "u": u.tolist(),
            "T": T.tolist(),
            "p": p.tolist()
        }
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def process_all_inputs(input_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Orchestrates processing of all 5 input datasets in input_dir."""
    print("=" * 75)
    print(f" Processing CFD Solver Input Dataset Directory: {os.path.abspath(input_dir)}")
    print("=" * 75)

    results = {}

    # File mapping
    file_map = {
        "3D.dat": process_3d_dat,
        "Prime3D.dat": process_prime3d_dat,
        "PrimeUV.dat": process_primeuv_dat,
        "RMSu2v2w2t2uvvt.dat": process_rms_profile_dat,
        "rhouvwtpmu.dat": process_mean_profile_dat,
    }

    for fname, func in file_map.items():
        fpath = os.path.join(input_dir, fname)
        if os.path.exists(fpath):
            try:
                results[fname] = func(fpath)
            except Exception as e:
                print(f"  [ERROR] Failed to process {fname}: {e}")
                results[fname] = {"error": str(e)}
        else:
            print(f"  [WARNING] File '{fname}' not found in {input_dir}")

    # Export json summary if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "input_processing_summary.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[+] Saved consolidated JSON summary report: {json_path}")

        # Also export CSV profiles for extracted mean and RMS profiles
        if "3D.dat" in results and "mean_profiles" in results["3D.dat"]:
            mean_df = pd.DataFrame(results["3D.dat"]["mean_profiles"])
            csv_mean_path = os.path.join(output_dir, "extracted_mean_profiles.csv")
            mean_df.to_csv(csv_mean_path, index=False)
            print(f"[+] Saved extracted mean profile CSV: {csv_mean_path}")

        if "Prime3D.dat" in results and "rms_profiles" in results["Prime3D.dat"]:
            rms_df = pd.DataFrame(results["Prime3D.dat"]["rms_profiles"])
            csv_rms_path = os.path.join(output_dir, "extracted_rms_profiles.csv")
            rms_df.to_csv(csv_rms_path, index=False)
            print(f"[+] Saved extracted RMS profile CSV: {csv_rms_path}")

    print("\n" + "=" * 75)
    print(" INPUT FILES PROCESSING COMPLETE")
    print("=" * 75)
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Process 3D Compressible Channel Flow Input Datasets")
    p.add_argument("--input-dir", default="input", help="Path to input directory containing .dat files")
    p.add_argument("--output-dir", default="output_processed", help="Path to export JSON/CSV reports")
    p.add_argument("--file", default=None, help="Process a single specific input file")
    return p.parse_args()


def main():
    args = parse_args()
    if args.file:
        fpath = os.path.join(args.input_dir, args.file) if not os.path.isabs(args.file) else args.file
        base = os.path.basename(fpath)
        if base == "3D.dat":
            res = process_3d_dat(fpath)
        elif base == "Prime3D.dat":
            res = process_prime3d_dat(fpath)
        elif base == "PrimeUV.dat":
            res = process_primeuv_dat(fpath)
        elif base == "RMSu2v2w2t2uvvt.dat":
            res = process_rms_profile_dat(fpath)
        elif base == "rhouvwtpmu.dat":
            res = process_mean_profile_dat(fpath)
        else:
            print(f"Unknown input file format: {base}")
            sys.exit(1)
        print("\nResult Summary:\n", json.dumps(res, indent=2, default=str))
    else:
        process_all_inputs(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
