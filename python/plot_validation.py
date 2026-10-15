#!/usr/bin/env python3
"""
Validation and Benchmark Plotting Suite for 3D Compressible Supersonic Channel Flow.

Generates high-resolution validation plots comparing CFD solver simulation results
(from Fortran 90 MPI 'output/' and Python Numba CUDA 'output_cuda/') against the
canonical Coleman, Kim & Moser (CKM) supersonic turbulent channel flow benchmark:

  1. Kinetic Energy (KE_valid.png):
     Evolution of kinetic energy over non-dimensional time t (e.g. t ~ 940 to 1420),
     exhibiting steady upward trend with minor fluctuations as flow evolves.
  2. Mass Flow Rate (massflow_valid.png):
     Bulk mass flow rate (massb) over time t, showing slight oscillations around an
     increasing trend line (plane-averaged bulk flow).
  3. Mean Pressure (pmean_valid.png):
     Mean pressure profile across wall-normal coordinate z comparing Present Solver
     against CKM benchmark, demonstrating symmetrical qualitative behavior across channel.
  4. Bulk Density (rhobulk_valid.png):
     Fluctuations in bulk density (rho_b) over time t, staying tightly bounded around 1.006.
  5. Mean Density (rhomean_valid.png):
     Mean density (rho_mean) distribution along z, high density near walls (z=0 and z=2)
     and flat core region, showing excellent agreement with CKM benchmark.
  6. Wall Shear Stress (tauWall_valid.png):
     Shear stress (tau_w) at Bottom Wall, Top Wall, and their Average over time t,
     demonstrating stable boundary layer conditions.
  7. Mean Temperature (tmean_valid.png):
     Mean temperature (T_mean) across z compared with CKM reference, exhibiting near-perfect
     overlap showing symmetric core heating.
  8. Root-Mean-Square Temperature (Trms_valid.png):
     RMS temperature fluctuations (T_rms) across z, capturing peak thermal fluctuations
     near walls and lower fluctuations in channel center, closely matching CKM benchmark.

Features:
  - Supports Fortran ('output/'), CUDA ('output_cuda/'), and side-by-side solver comparisons.
  - Automatically extracts 1D mean & RMS profiles from 3D snapshots (e.g. 3D_cuda.dat) if
    mean.dat / rms.dat files are still being processed.
  - Graceful handling of in-progress solver runs: skips missing files with informative status
    messages without crashing.
  - Publication-quality Matplotlib styling matching classical gnuplot enhanced benchmarks.

Author: Google DeepMind Antigravity Pair Programmer
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for reliable headless execution and thread safety
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Global Styling & Matplotlib Configuration
# ---------------------------------------------------------------------------

def configure_matplotlib_style(use_latex: bool = False) -> None:
    """Configures Matplotlib to match classical publication / gnuplot benchmarks."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 13,
        "font.weight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "lines.linewidth": 1.5,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "legend.handlelength": 2.0,
    })


# ---------------------------------------------------------------------------
# Data Loading & Processing Utilities
# ---------------------------------------------------------------------------

def is_float(val: str) -> bool:
    """Check if string token can be converted to float."""
    try:
        float(val)
        return True
    except ValueError:
        return False


def load_time_series_data(
    filepath: Union[str, Path],
    col_time: int = 0,
    col_val: Optional[Union[int, List[int]]] = 1,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    stride: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Loads large time-series datasets (KE.dat, massflow.dat, rhobulk.dat, tauWall.dat).
    Filters by optional time bounds [tmin, tmax] and optional stride for efficiency.
    """
    path = Path(filepath)
    if not path.is_file():
        return None

    try:
        # Load with pandas fast C engine
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64)
        if df.empty or df.shape[1] <= col_time:
            return None

        # Filter by time bounds if requested
        if tmin is not None:
            df = df[df.iloc[:, col_time] >= tmin]
        if tmax is not None:
            df = df[df.iloc[:, col_time] <= tmax]

        # Apply stride if requested and dataset is large
        if stride > 1 and len(df) > stride:
            df = df.iloc[::stride].copy()

        return df
    except Exception as e:
        print(f"    [WARNING] Error reading time-series file '{path}': {e}")
        return None


def load_1d_profiles(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Loads 1D wall-normal profile files (mean.dat, rms.dat, corr.dat).
    Expected columns for mean.dat:
      0: z, 1: rho, 2: u, 3: v, 4: w, 5: e, 6: T, 7: p
    Expected columns for rms.dat:
      0: z, 1: rho_rms, 2: u_rms, 3: v_rms, 4: w_rms, 5: e_rms, 6: T_rms, 7: p_rms
    """
    path = Path(filepath)
    if not path.is_file():
        return None

    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64)
        return df if not df.empty else None
    except Exception as e:
        print(f"    [WARNING] Error reading 1D profile file '{path}': {e}")
        return None


def load_ckm_mean_benchmark(benchmark_file: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Loads Coleman, Kim & Moser (CKM) mean flow benchmark data (input/rhouvwtpmu.dat).
    Columns:
      0: z, 1: rho, 2: u, 3: v, 4: w, 5: T, 6: p, 7: mu
    """
    path = Path(benchmark_file)
    if not path.is_file():
        return None

    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64)
        return df
    except Exception as e:
        print(f"    [WARNING] Error reading CKM mean benchmark '{path}': {e}")
        return None


def load_ckm_rms_benchmark(benchmark_file: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Loads Coleman, Kim & Moser (CKM) RMS fluctuation benchmark data (input/RMSu2v2w2t2uvvt.dat).
    Columns:
      0: z, 1: u_rms, 2: v_rms, 3: w_rms, 4: T_rms, 5: uv, 6: vt
    """
    path = Path(benchmark_file)
    if not path.is_file():
        return None

    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64)
        return df
    except Exception as e:
        print(f"    [WARNING] Error reading CKM RMS benchmark '{path}': {e}")
        return None


def extract_profiles_from_3d_snapshot(
    snapshot_path: Union[str, Path],
    nz: int = 160,
    ny: int = 130,
    nx: int = 130
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Extracts 1D wall-normal mean profiles and RMS fluctuations directly from a
    3D instantaneous flow snapshot (e.g. output_cuda/3D_cuda.dat or input/3D.dat).
    Used as an automatic fallback if mean.dat / rms.dat have not yet been produced.
    
    Returns:
      (mean_df, rms_df) matching the exact column layout of mean.dat and rms.dat.
    """
    path = Path(snapshot_path)
    if not path.is_file():
        return None, None

    print(f"    [INFO] Extracting 1D profiles from 3D snapshot: {path.name}...")
    try:
        # Determine header offset
        header_offset = 0
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    header_offset += 1
                    continue
                tokens = s.split()
                if tokens and is_float(tokens[0]):
                    break
                header_offset += 1

        # Load 3D data: cols [x, y, z, rho, u, v, w, e, T, p]
        df_3d = pd.read_csv(path, skiprows=header_offset, sep=r"\s+", header=None, engine="c", dtype=np.float64).iloc[:, :10]
        expected_cells = nz * ny * nx
        if len(df_3d) < expected_cells:
            print(f"    [WARNING] 3D file has {len(df_3d)} rows, expected {expected_cells}. Snapshot might be incomplete.")
            return None, None

        arr = df_3d.to_numpy()[:expected_cells].reshape(nz, ny, nx, 10)
        z_coords = arr[:, 0, 0, 2]  # Wall-normal coordinate z

        # Calculate mean over x (dim 2) and y (dim 1)
        # Columns in 3D: 3: rho, 4: u, 5: v, 6: w, 7: e, 8: T, 9: p
        rho_mean = arr[:, :, :, 3].mean(axis=(1, 2))
        u_mean = arr[:, :, :, 4].mean(axis=(1, 2))
        v_mean = arr[:, :, :, 5].mean(axis=(1, 2))
        w_mean = arr[:, :, :, 6].mean(axis=(1, 2))
        e_mean = arr[:, :, :, 7].mean(axis=(1, 2))
        T_mean = arr[:, :, :, 8].mean(axis=(1, 2))
        p_mean = arr[:, :, :, 9].mean(axis=(1, 2))

        mean_df = pd.DataFrame({
            0: z_coords,
            1: rho_mean,
            2: u_mean,
            3: v_mean,
            4: w_mean,
            5: e_mean,
            6: T_mean,
            7: p_mean
        })

        # Calculate RMS fluctuations: sqrt(mean((q - q_bar)^2))
        # Wall-normal broadcast shapes: (nz, 1, 1)
        rho_rms = np.sqrt(np.mean((arr[:, :, :, 3] - rho_mean[:, None, None])**2, axis=(1, 2)))
        u_rms = np.sqrt(np.mean((arr[:, :, :, 4] - u_mean[:, None, None])**2, axis=(1, 2)))
        v_rms = np.sqrt(np.mean((arr[:, :, :, 5] - v_mean[:, None, None])**2, axis=(1, 2)))
        w_rms = np.sqrt(np.mean((arr[:, :, :, 6] - w_mean[:, None, None])**2, axis=(1, 2)))
        e_rms = np.sqrt(np.mean((arr[:, :, :, 7] - e_mean[:, None, None])**2, axis=(1, 2)))
        T_rms = np.sqrt(np.mean((arr[:, :, :, 8] - T_mean[:, None, None])**2, axis=(1, 2)))
        p_rms = np.sqrt(np.mean((arr[:, :, :, 9] - p_mean[:, None, None])**2, axis=(1, 2)))

        rms_df = pd.DataFrame({
            0: z_coords,
            1: rho_rms,
            2: u_rms,
            3: v_rms,
            4: w_rms,
            5: e_rms,
            6: T_rms,
            7: p_rms
        })

        print(f"    [+] Successfully extracted 1D profiles across {nz} wall-normal points.")
        return mean_df, rms_df
    except Exception as e:
        print(f"    [WARNING] Failed to extract profiles from 3D snapshot '{path}': {e}")
        return None, None


# ---------------------------------------------------------------------------
# Core Validation Plotting Functions (The 8 Specified Target Figures)
# ---------------------------------------------------------------------------

def plot_kinetic_energy(
    sources: Dict[str, Union[str, Path]],
    output_filepath: Union[str, Path],
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    stride: int = 1,
) -> bool:
    """
    Plot 1: Kinetic Energy (KE_valid.png)
    Shows the evolution of kinetic energy over non-dimensional time t, exhibiting
    a steady upward trend with minor fluctuations as the flow evolves.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted_any = False

    # Plot styling palette for single or multiple solvers
    styles = [
        {"color": "black", "linestyle": "-", "linewidth": 1.5, "label": ""},
        {"color": "#0055d4", "linestyle": "--", "linewidth": 1.5, "label": "CUDA Solver"},
    ]

    idx = 0
    for label, dir_path in sources.items():
        ke_file = Path(dir_path) / "KE.dat"
        df = load_time_series_data(ke_file, col_time=0, col_val=1, tmin=tmin, tmax=tmax, stride=stride)
        if df is None:
            continue

        style = styles[idx % len(styles)]
        plot_label = label if len(sources) > 1 else ""
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], label=plot_label)
        plotted_any = True
        idx += 1

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xlabel("t")
    ax.set_ylabel("Kinetic Energy")
    if len(sources) > 1 and any(ax.get_legend_handles_labels()[1]):
        ax.legend(loc="upper left")

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_mass_flow_rate(
    sources: Dict[str, Union[str, Path]],
    output_filepath: Union[str, Path],
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    stride: int = 1,
) -> bool:
    """
    Plot 2: Mass Flow Rate (massflow_valid.png)
    Displays the bulk mass flow rate (massb) over time t, showing slight oscillations
    around an increasing trend line. Calculated by averaging the 5 wall-parallel planes.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted_any = False

    styles = [
        {"color": "black", "linestyle": "-", "linewidth": 1.5},
        {"color": "#0055d4", "linestyle": "--", "linewidth": 1.5},
    ]

    idx = 0
    for label, dir_path in sources.items():
        mass_file = Path(dir_path) / "massflow.dat"
        df = load_time_series_data(mass_file, col_time=0, tmin=tmin, tmax=tmax, stride=stride)
        if df is None:
            continue

        time_arr = df.iloc[:, 0]
        # Average columns 1 to min(6, df.shape[1]) representing 5 monitored sampling planes
        num_planes = min(5, df.shape[1] - 1)
        mass_bulk = df.iloc[:, 1:1 + num_planes].mean(axis=1)

        style = styles[idx % len(styles)]
        plot_label = label if len(sources) > 1 else ""
        ax.plot(time_arr, mass_bulk, color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], label=plot_label)
        plotted_any = True
        idx += 1

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xlabel("t")
    ax.set_ylabel(r"$\mathrm{mass}_b$")
    if len(sources) > 1 and any(ax.get_legend_handles_labels()[1]):
        ax.legend(loc="upper left")

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_mean_pressure(
    sources: Dict[str, Union[str, Path]],
    ckm_mean_file: Union[str, Path],
    output_filepath: Union[str, Path],
) -> bool:
    """
    Plot 3: Mean Pressure (pmean_valid.png)
    Compares the mean pressure profile across the wall-normal coordinate z between
    the Present Solver and the CKM benchmark, showing similar symmetrical qualitative
    behavior across the channel height.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted_any = False

    # 1. Present Solver(s)
    solver_styles = [
        {"marker": "+", "color": "black", "linestyle": "None", "label": "Present Solver", "ms": 5},
        {"marker": "x", "color": "#0055d4", "linestyle": "None", "label": "CUDA Solver", "ms": 5},
    ]

    idx = 0
    for label, dir_path in sources.items():
        mean_file = Path(dir_path) / "mean.dat"
        df_mean = load_1d_profiles(mean_file)

        # Fallback to 3D snapshot extraction if mean.dat is missing
        if df_mean is None:
            for snap_name in ["3D_cuda.dat", "3D.dat", "3D001.dat"]:
                snap_path = Path(dir_path) / snap_name
                if snap_path.is_file():
                    df_mean, _ = extract_profiles_from_3d_snapshot(snap_path)
                    break

        if df_mean is None:
            continue

        style = solver_styles[idx % len(solver_styles)]
        plot_label = label if len(sources) > 1 else style["label"]
        # In mean.dat: col 0 is z, col 7 is p_mean
        z_pts = df_mean.iloc[:, 0]
        p_pts = df_mean.iloc[:, 7]
        ax.plot(z_pts, p_pts, marker=style["marker"], color=style["color"],
                linestyle=style["linestyle"], label=plot_label, markersize=style["ms"])
        plotted_any = True
        idx += 1

    # 2. CKM Benchmark Reference: col 0 is z, col 6 is p_mean
    df_ckm = load_ckm_mean_benchmark(ckm_mean_file)
    if df_ckm is not None:
        ax.plot(df_ckm.iloc[:, 0], df_ckm.iloc[:, 6], color="black", linestyle="-",
                linewidth=1.8, label="CKM")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xlabel("z")
    ax.set_ylabel(r"$P_{\mathrm{mean}}$")
    ax.legend(loc="upper right")

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_bulk_density(
    sources: Dict[str, Union[str, Path]],
    output_filepath: Union[str, Path],
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    stride: int = 1,
) -> bool:
    """
    Plot 4: Bulk Density (rhobulk_valid.png)
    Plots fluctuations in bulk density (rho_b) over time t, staying tightly bounded around 1.006.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted_any = False

    styles = [
        {"color": "black", "linestyle": "-", "linewidth": 1.5},
        {"color": "#0055d4", "linestyle": "--", "linewidth": 1.5},
    ]

    idx = 0
    for label, dir_path in sources.items():
        rho_file = Path(dir_path) / "rhobulk.dat"
        df = load_time_series_data(rho_file, col_time=0, tmin=tmin, tmax=tmax, stride=stride)
        if df is None:
            continue

        time_arr = df.iloc[:, 0]
        num_planes = min(5, df.shape[1] - 1)
        rho_bulk = df.iloc[:, 1:1 + num_planes].mean(axis=1)

        style = styles[idx % len(styles)]
        plot_label = label if len(sources) > 1 else ""
        ax.plot(time_arr, rho_bulk, color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], label=plot_label)
        plotted_any = True
        idx += 1

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xlabel("t")
    ax.set_ylabel(r"$\rho_b$")
    if len(sources) > 1 and any(ax.get_legend_handles_labels()[1]):
        ax.legend(loc="upper right")

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_mean_density(
    sources: Dict[str, Union[str, Path]],
    ckm_mean_file: Union[str, Path],
    output_filepath: Union[str, Path],
) -> bool:
    """
    Plot 5: Mean Density (rhomean_valid.png)
    Displays the mean density (rho_mean) distribution along z, showing high density near
    the walls (z=0 and z=2) and a flat core region, with excellent agreement between
    the Present Solver and CKM.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted_any = False

    solver_styles = [
        {"marker": "+", "color": "black", "linestyle": "None", "label": "Present Solver", "ms": 5},
        {"marker": "x", "color": "#0055d4", "linestyle": "None", "label": "CUDA Solver", "ms": 5},
    ]

    idx = 0
    for label, dir_path in sources.items():
        mean_file = Path(dir_path) / "mean.dat"
        df_mean = load_1d_profiles(mean_file)

        if df_mean is None:
            for snap_name in ["3D_cuda.dat", "3D.dat", "3D001.dat"]:
                snap_path = Path(dir_path) / snap_name
                if snap_path.is_file():
                    df_mean, _ = extract_profiles_from_3d_snapshot(snap_path)
                    break

        if df_mean is None:
            continue

        style = solver_styles[idx % len(solver_styles)]
        plot_label = label if len(sources) > 1 else style["label"]
        # In mean.dat: col 0 is z, col 1 is rho_mean
        z_pts = df_mean.iloc[:, 0]
        rho_pts = df_mean.iloc[:, 1]
        ax.plot(z_pts, rho_pts, marker=style["marker"], color=style["color"],
                linestyle=style["linestyle"], label=plot_label, markersize=style["ms"])
        plotted_any = True
        idx += 1

    # CKM Benchmark: col 0 is z, col 1 is rho_mean
    df_ckm = load_ckm_mean_benchmark(ckm_mean_file)
    if df_ckm is not None:
        ax.plot(df_ckm.iloc[:, 0], df_ckm.iloc[:, 1], color="black", linestyle="-",
                linewidth=1.8, label="CKM")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xlabel("z")
    ax.set_ylabel(r"$\rho_{\mathrm{mean}}$")
    ax.legend(loc="upper right")

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_wall_shear_stress(
    sources: Dict[str, Union[str, Path]],
    output_filepath: Union[str, Path],
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    stride: int = 1,
) -> bool:
    """
    Plot 6: Wall Shear Stress (tauWall_valid.png)
    Tracks the shear stress (tau_w) at the Bottom Wall, Top Wall, and their Average
    over time t, demonstrating stable boundary layer conditions.
      - Bottom Wall: col 1
      - Top Wall: col 2 (negative wall normal)
      - Average: (tau_bottom - tau_top) / 2.0
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted_any = False

    for label, dir_path in sources.items():
        tau_file = Path(dir_path) / "tauWall.dat"
        df = load_time_series_data(tau_file, col_time=0, tmin=tmin, tmax=tmax, stride=stride)
        if df is None:
            continue

        time_arr = df.iloc[:, 0]
        tau_bottom = df.iloc[:, 1]
        tau_top = df.iloc[:, 2]
        tau_avg = (tau_bottom - tau_top) / 2.0

        prefix = f"{label} " if len(sources) > 1 else ""
        # Classical gnuplot color scheme: red solid, bright green dashed, blue dotted
        ax.plot(time_arr, tau_bottom, color="red", linestyle="-", linewidth=1.6, label=f"{prefix}Bottom Wall")
        ax.plot(time_arr, tau_top, color="#00dd00", linestyle="--", linewidth=1.6, label=f"{prefix}Top Wall")
        ax.plot(time_arr, tau_avg, color="blue", linestyle=":", linewidth=1.8, label=f"{prefix}Average")
        plotted_any = True
        break  # For tauWall, single clear solver presentation avoids line clutter

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xlabel("t")
    ax.set_ylabel(r"$\tau_w$")
    ax.legend(loc="center right")

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_mean_temperature(
    sources: Dict[str, Union[str, Path]],
    ckm_mean_file: Union[str, Path],
    output_filepath: Union[str, Path],
) -> bool:
    """
    Plot 7: Mean Temperature (tmean_valid.png)
    Compares the mean temperature (T_mean) across z with the CKM reference,
    exhibiting near-perfect overlap showing symmetric core heating.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted_any = False

    solver_styles = [
        {"marker": "+", "color": "black", "linestyle": "None", "label": "Present Solver", "ms": 5},
        {"marker": "x", "color": "#0055d4", "linestyle": "None", "label": "CUDA Solver", "ms": 5},
    ]

    idx = 0
    for label, dir_path in sources.items():
        mean_file = Path(dir_path) / "mean.dat"
        df_mean = load_1d_profiles(mean_file)

        if df_mean is None:
            for snap_name in ["3D_cuda.dat", "3D.dat", "3D001.dat"]:
                snap_path = Path(dir_path) / snap_name
                if snap_path.is_file():
                    df_mean, _ = extract_profiles_from_3d_snapshot(snap_path)
                    break

        if df_mean is None:
            continue

        style = solver_styles[idx % len(solver_styles)]
        plot_label = label if len(sources) > 1 else style["label"]
        # In mean.dat: col 0 is z, col 6 is T_mean
        z_pts = df_mean.iloc[:, 0]
        T_pts = df_mean.iloc[:, 6]
        ax.plot(z_pts, T_pts, marker=style["marker"], color=style["color"],
                linestyle=style["linestyle"], label=plot_label, markersize=style["ms"])
        plotted_any = True
        idx += 1

    # CKM Benchmark: col 0 is z, col 5 is T_mean
    df_ckm = load_ckm_mean_benchmark(ckm_mean_file)
    if df_ckm is not None:
        ax.plot(df_ckm.iloc[:, 0], df_ckm.iloc[:, 5], color="black", linestyle="-",
                linewidth=1.8, label="CKM")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xlabel("z")
    ax.set_ylabel(r"$T_{\mathrm{mean}}$")
    ax.legend(loc="lower right")

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_rms_temperature(
    sources: Dict[str, Union[str, Path]],
    ckm_rms_file: Union[str, Path],
    output_filepath: Union[str, Path],
) -> bool:
    """
    Plot 8: Root-Mean-Square Temperature (Trms_valid.png)
    Plots the RMS temperature fluctuations (T_rms) across z, capturing peak thermal
    fluctuations near the walls and lower fluctuations in the channel center,
    closely matching the CKM benchmark.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted_any = False

    solver_styles = [
        {"marker": "+", "color": "black", "linestyle": "None", "label": "Present Solver", "ms": 5},
        {"marker": "x", "color": "#0055d4", "linestyle": "None", "label": "CUDA Solver", "ms": 5},
    ]

    idx = 0
    for label, dir_path in sources.items():
        rms_file = Path(dir_path) / "rms.dat"
        df_rms = load_1d_profiles(rms_file)

        if df_rms is None:
            for snap_name in ["3D_cuda.dat", "3D.dat", "3D001.dat"]:
                snap_path = Path(dir_path) / snap_name
                if snap_path.is_file():
                    _, df_rms = extract_profiles_from_3d_snapshot(snap_path)
                    break

        if df_rms is None:
            continue

        style = solver_styles[idx % len(solver_styles)]
        plot_label = label if len(sources) > 1 else style["label"]
        # In rms.dat: col 0 is z, col 6 is T_rms
        z_pts = df_rms.iloc[:, 0]
        T_rms = df_rms.iloc[:, 6]
        ax.plot(z_pts, T_rms, marker=style["marker"], color=style["color"],
                linestyle=style["linestyle"], label=plot_label, markersize=style["ms"])
        plotted_any = True
        idx += 1

    # CKM Benchmark: col 0 is z, col 4 is T_rms
    df_ckm = load_ckm_rms_benchmark(ckm_rms_file)
    if df_ckm is not None:
        ax.plot(df_ckm.iloc[:, 0], df_ckm.iloc[:, 4], color="black", linestyle="-",
                linewidth=1.8, label="CKM")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xlabel("z")
    ax.set_ylabel(r"$T_{\mathrm{rms}}$")
    ax.legend(loc="upper right")

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Auxiliary Validation Plots (Velocity profiles & RMS from cmdGNUplot.txt)
# ---------------------------------------------------------------------------

def plot_mean_velocity_components(
    sources: Dict[str, Union[str, Path]],
    ckm_mean_file: Union[str, Path],
    output_dir: Union[str, Path],
) -> List[str]:
    """Generates umean_valid.png, vmean_valid.png, and wmean_valid.png."""
    out_dir = Path(output_dir)
    generated = []

    comp_info = [
        {"col_s": 2, "col_c": 2, "name": "u", "ylabel": r"$u_{\mathrm{mean}}$", "loc": "lower right"},
        {"col_s": 3, "col_c": 3, "name": "v", "ylabel": r"$v_{\mathrm{mean}}$", "loc": "upper right"},
        {"col_s": 4, "col_c": 4, "name": "w", "ylabel": r"$w_{\mathrm{mean}}$", "loc": "upper right"},
    ]

    df_ckm = load_ckm_mean_benchmark(ckm_mean_file)

    for info in comp_info:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        plotted = False

        for label, dir_path in sources.items():
            mean_file = Path(dir_path) / "mean.dat"
            df_m = load_1d_profiles(mean_file)
            if df_m is None:
                continue
            ax.plot(df_m.iloc[:, 0], df_m.iloc[:, info["col_s"]], marker="+", color="black",
                    linestyle="None", label="Present Solver", markersize=5)
            plotted = True
            break

        if df_ckm is not None:
            ax.plot(df_ckm.iloc[:, 0], df_ckm.iloc[:, info["col_c"]], color="black",
                    linestyle="-", linewidth=1.8, label="CKM")
            plotted = True

        if plotted:
            ax.set_xlabel("z")
            ax.set_ylabel(info["ylabel"])
            ax.legend(loc=info["loc"])
            fname = f"{info['name']}mean_valid.png"
            p = out_dir / fname
            fig.savefig(p)
            generated.append(str(p))
        plt.close(fig)

    return generated


# ---------------------------------------------------------------------------
# Orchestration Suite
# ---------------------------------------------------------------------------

class ValidationPlotter:
    """
    Main manager for parsing CFD outputs and generating all 8 canonical validation figures.
    """

    def __init__(
        self,
        fortran_dir: Union[str, Path] = "output",
        cuda_dir: Union[str, Path] = "output_cuda",
        benchmark_dir: Union[str, Path] = "input",
        output_dir: Union[str, Path] = "plots/validation_plots",
        solver_mode: str = "auto",
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        stride: int = 1,
    ):
        self.fortran_dir = Path(fortran_dir)
        self.cuda_dir = Path(cuda_dir)
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)
        self.solver_mode = solver_mode.lower()
        self.tmin = tmin
        self.tmax = tmax
        self.stride = stride

        self.ckm_mean_file = self.benchmark_dir / "rhouvwtpmu.dat"
        self.ckm_rms_file = self.benchmark_dir / "RMSu2v2w2t2uvvt.dat"

        configure_matplotlib_style()

    def resolve_sources(self) -> Dict[str, Path]:
        """Resolves target data sources according to requested solver mode."""
        sources: Dict[str, Path] = {}

        if self.solver_mode == "fortran":
            if self.fortran_dir.is_dir():
                sources["Present Solver"] = self.fortran_dir
            else:
                print(f"[!] Warning: Fortran directory '{self.fortran_dir}' does not exist.")

        elif self.solver_mode == "cuda":
            if self.cuda_dir.is_dir():
                sources["CUDA Solver"] = self.cuda_dir
            else:
                print(f"[!] Warning: CUDA directory '{self.cuda_dir}' does not exist.")

        elif self.solver_mode in ["compare", "both"]:
            if self.fortran_dir.is_dir():
                sources["Present Solver"] = self.fortran_dir
            if self.cuda_dir.is_dir():
                sources["CUDA Solver"] = self.cuda_dir

        else:  # 'auto'
            # Default to Present Solver (Fortran) for canonical validation benchmarks
            if self.fortran_dir.is_dir():
                sources["Present Solver"] = self.fortran_dir
            elif self.cuda_dir.is_dir():
                sources["CUDA Solver"] = self.cuda_dir

        return sources

    def generate_all_plots(self, selected_plots: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executes plotting pipeline for the requested figures.
        Returns a dictionary summarizing execution results and generated paths.
        """
        sources = self.resolve_sources()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 90)
        print(" CFD Validation & Benchmark Plot Generation Suite")
        print("=" * 90)
        print(f" Output Directory   : {self.output_dir.resolve()}")
        print(f" CKM Benchmark Path : {self.benchmark_dir.resolve()}")
        print(f" Active Data Sources:")
        for label, p in sources.items():
            print(f"   - {label:<16}: {p.resolve()} (Exists: {p.is_dir()})")
        if self.tmin is not None or self.tmax is not None:
            print(f" Time Range Filter  : t in [{self.tmin if self.tmin else 'min'}, {self.tmax if self.tmax else 'max'}]")
        print("=" * 90)

        results: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": {k: str(v) for k, v in sources.items()},
            "generated_plots": [],
            "pending_or_skipped": [],
        }

        # Target 8 validation plots specifications
        targets = [
            ("KE_valid.png", "Kinetic Energy Evolution", lambda out: plot_kinetic_energy(
                sources, out, tmin=self.tmin, tmax=self.tmax, stride=self.stride
            )),
            ("massflow_valid.png", "Bulk Mass Flow Rate (massb)", lambda out: plot_mass_flow_rate(
                sources, out, tmin=self.tmin, tmax=self.tmax, stride=self.stride
            )),
            ("pmean_valid.png", "Mean Pressure Profile (Pmean)", lambda out: plot_mean_pressure(
                sources, self.ckm_mean_file, out
            )),
            ("rhobulk_valid.png", "Bulk Density Fluctuations (rho_b)", lambda out: plot_bulk_density(
                sources, out, tmin=self.tmin, tmax=self.tmax, stride=self.stride
            )),
            ("rhomean_valid.png", "Mean Density Profile (rho_mean)", lambda out: plot_mean_density(
                sources, self.ckm_mean_file, out
            )),
            ("tauWall_valid.png", "Wall Shear Stress (tau_w: Top/Bottom/Avg)", lambda out: plot_wall_shear_stress(
                sources, out, tmin=self.tmin, tmax=self.tmax, stride=self.stride
            )),
            ("tmean_valid.png", "Mean Temperature Profile (Tmean)", lambda out: plot_mean_temperature(
                sources, self.ckm_mean_file, out
            )),
            ("Trms_valid.png", "RMS Temperature Fluctuations (Trms)", lambda out: plot_rms_temperature(
                sources, self.ckm_rms_file, out
            )),
        ]

        print(f"\n[+] Processing {len(targets)} Canonical Validation Plots:")
        for idx, (filename, desc, plot_func) in enumerate(targets, 1):
            if selected_plots and filename.lower() not in [p.lower() for p in selected_plots]:
                continue

            out_file = self.output_dir / filename
            sys.stdout.write(f"  [{idx}/8] Generating {filename:<19} ({desc})... ")
            sys.stdout.flush()

            try:
                success = plot_func(out_file)
                if success:
                    print(f"SUCCESS -> Saved ({out_file.stat().st_size / 1024:.1f} KB)")
                    results["generated_plots"].append({
                        "filename": filename,
                        "description": desc,
                        "filepath": str(out_file),
                        "size_bytes": out_file.stat().st_size
                    })
                else:
                    print("PENDING / SKIPPED (Required data files still being processed)")
                    results["pending_or_skipped"].append({
                        "filename": filename,
                        "description": desc,
                        "reason": "Data file not found or empty (solver computation in progress)"
                    })
            except Exception as e:
                print(f"ERROR: {e}")
                results["pending_or_skipped"].append({
                    "filename": filename,
                    "description": desc,
                    "error": str(e)
                })

        print("\n" + "=" * 90)
        print(" Validation Plotting Execution Summary")
        print("=" * 90)
        print(f" Plots Successfully Created : {len(results['generated_plots'])} / {len(targets)}")
        print(f" Pending / Waiting on Solver: {len(results['pending_or_skipped'])} / {len(targets)}")
        print("=" * 90)

        return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive Validation & Benchmark Plot Generator for Supersonic Channel Flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None,
                        help="Primary solver output directory. Overrides --fortran-dir / --cuda-dir if provided.")
    parser.add_argument("--fortran-dir", default="output",
                        help="Fortran 90 MPI solver output directory.")
    parser.add_argument("--cuda-dir", default="output_cuda",
                        help="Python Numba CUDA solver output directory.")
    parser.add_argument("--benchmark-dir", default="input",
                        help="Directory containing CKM reference benchmark data.")
    parser.add_argument("--output-dir", default="plots/validation_plots",
                        help="Destination directory to save generated PNG plots.")
    parser.add_argument("--solver", choices=["fortran", "cuda", "compare", "both", "auto"], default="auto",
                        help="Solver dataset(s) to plot: fortran, cuda, compare (both overlaid), or auto.")
    parser.add_argument("--tmin", type=float, default=940.0,
                        help="Minimum non-dimensional time cutoff for time-series plots (e.g. 940.0).")
    parser.add_argument("--tmax", type=float, default=1420.0,
                        help="Maximum non-dimensional time cutoff for time-series plots (e.g. 1420.0).")
    parser.add_argument("--full-time", action="store_true",
                        help="Plot full available time history without [940, 1420] cutoff.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Subsampling stride for high-frequency time series (1 = full data, 5 = every 5th pt).")
    parser.add_argument("--plots", nargs="*", default=None,
                        help="Optional subset of plots to generate (e.g., --plots KE_valid.png pmean_valid.png).")
    parser.add_argument("--include-auxiliary", action="store_true",
                        help="Also generate auxiliary velocity profiles (umean, vmean, wmean).")

    args = parser.parse_args()

    # Time bounds
    if args.full_time:
        tmin, tmax = None, None
    else:
        tmin, tmax = args.tmin, args.tmax

    # If --data-dir is explicitly given, override directories
    fortran_dir = args.data_dir if args.data_dir else args.fortran_dir
    cuda_dir = args.data_dir if args.data_dir else args.cuda_dir

    # 1. Primary validation run (Present Solver vs CKM benchmark)
    plotter = ValidationPlotter(
        fortran_dir=fortran_dir,
        cuda_dir=cuda_dir,
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        solver_mode=args.solver,
        tmin=tmin,
        tmax=tmax,
        stride=args.stride,
    )

    results = plotter.generate_all_plots(selected_plots=args.plots)

    # 2. If running in auto mode and CUDA data is available, also generate CUDA and comparison suites
    cuda_path = Path(cuda_dir)
    cuda_has_data = cuda_path.is_dir() and any(cuda_path.iterdir())
    if args.solver == "auto" and cuda_has_data and not args.data_dir:
        print("\n" + "=" * 90)
        print(" Additional Suite: CUDA Solver Validation & Comparison")
        print("=" * 90)
        
        # CUDA against CKM
        cuda_plotter = ValidationPlotter(
            fortran_dir=fortran_dir,
            cuda_dir=cuda_dir,
            benchmark_dir=args.benchmark_dir,
            output_dir=Path(args.output_dir) / "cuda",
            solver_mode="cuda",
            tmin=tmin,
            tmax=tmax,
            stride=args.stride,
        )
        cuda_plotter.generate_all_plots(selected_plots=args.plots)

        # 3-way Comparison (Fortran vs CUDA vs CKM)
        compare_plotter = ValidationPlotter(
            fortran_dir=fortran_dir,
            cuda_dir=cuda_dir,
            benchmark_dir=args.benchmark_dir,
            output_dir=Path(args.output_dir) / "comparison",
            solver_mode="compare",
            tmin=tmin,
            tmax=tmax,
            stride=args.stride,
        )
        compare_plotter.generate_all_plots(selected_plots=args.plots)

    if args.include_auxiliary:
        print("\n[+] Generating Auxiliary Velocity Profiles (umean, vmean, wmean)...")
        sources = plotter.resolve_sources()
        aux_files = plot_mean_velocity_components(sources, plotter.ckm_mean_file, plotter.output_dir)
        for af in aux_files:
            print(f"    Saved: {af}")


if __name__ == "__main__":
    main()
