#!/usr/bin/env python3
"""
MPI -> CUDA Python port of the supplied 3-D compressible channel-flow solver.

Target:
    Python 3 + Numba CUDA + one NVIDIA GPU.

The original Fortran program uses MPI for:
  1) 3-D Cartesian domain decomposition
  2) halo exchange with MPI_Sendrecv
  3) MPI_Reduce / MPI_Bcast for statistics
  4) MPI_Gather for domain information

In this single-GPU version, the whole physical domain is resident on the GPU.
CUDA threads replace MPI ranks. x/y periodicity is handled with modular
neighbor indexing, so MPI halo exchange is removed. GPU reductions replace
MPI_Reduce where a scalar/global statistic is needed.

This file keeps the original solver organization:
    read input -> mesh -> primitive variables -> RK2:
    convective fluxes -> diffusive fluxes -> source/RHS -> boundary control.

The statistics/output routines are intentionally kept as hooks because the
uploaded project contains many separate analysis routines in mod_stats.f90.
The main flow solver is the part ported here and can be extended with the
same kernels/reduction pattern.

Input format supported:
  Combined input (resOp=1): whitespace separated rows
      x y z rho u v w E T P
  after optional header lines. The parser scans all numeric rows and takes
  the last 10 values in each data row.

Example:
    python mpi_to_cuda_solver.py --input ../input.dat --steps 1000
    python mpi_to_cuda_solver.py --demo --steps 20

Requirements:
    pip install numpy numba
    CUDA-capable NVIDIA GPU + compatible NVIDIA driver
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import time
from dataclasses import dataclass

# Automatically set CUDA paths for Numba on Windows if needed
try:
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
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

import numpy as np
import warnings
from numba import cuda, float64, int32
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
try:
    from numba.cuda.errors import NumbaPerformanceWarning as CudaNumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=CudaNumbaPerformanceWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Original constants
# ---------------------------------------------------------------------------

GAMMA = 1.4
INV_PR = 1.0 / 0.71
HALF = 0.5
TWO_THIRDS = 2.0 / 3.0
ONE_TWELFTH = 1.0 / 12.0
EP = 5.0

# Conserved variables:
# q[...,0] = rho
# q[...,1] = rho*u
# q[...,2] = rho*v
# q[...,3] = rho*w
# q[...,4] = rho*E
NQ = 5

# Primitive variables:
# v[...,0] = rho
# v[...,1] = u
# v[...,2] = v
# v[...,3] = w
# v[...,4] = E
# v[...,5] = T
# v[...,6] = P
NV = 7


@dataclass
class SolverConfig:
    nx: int = 192
    ny: int = 130
    nz: int = 160

    reynolds: float = 1000.0
    mach: float = 0.2
    dt: float = 1.0e-4

    peclet_x: float = 1.0
    peclet_y: float = 1.0
    peclet_z: float = 1.0

    # Wall shear statistics for channel forcing (updated dynamically on GPU
    # when dynamic_source is True).
    tau_lower: float = 0.0
    tau_upper: float = 0.0

    # Opposition-control and actuation parameters from mod_control/mod_var_dec.
    opposition_distance: int = 33
    actuation_wavenumber: float = 1.0
    actuation_omega: float = 0.5
    actuation_amplitude: float = 0.0
    actuation_beta: float = 0.0
    actuation_amplitude_ls: float = 0.0

    # 1 = enable opposition control, 0 = no actuation.
    enable_oc: int = 1

    # Solver/input options from param.dat and codes/ vs codes1/ variants.
    description: str = "unc"
    restart_option: int = 1
    topo: tuple[int, int, int] = (1, 1, 8)
    debug_op: int = 0
    mesh_type: str = "auto"   # "auto", "cosine", or "tanh"
    wall_order: int = 3       # 2 = codes/ 2nd-order, 3 = codes1/ 3rd-order
    dynamic_source: bool = True
    defer_source: bool = False

    max_steps: int = 100
    statistic_frequency: int = 100
    diag_frequency: int = 10
    output_frequency: int = 1000

    # Phase 2 live statistics & rolling checkpoint controls.
    stats_dir: str | None = None
    stats_mode: str = "standard"   # "none", "minimal", "standard", "full"
    checkpoint_dir: str | None = None
    checkpoint_components: bool = False

    @property
    def inv_re(self) -> float:
        return 1.0 / self.reynolds

    @property
    def ggM(self) -> float:
        return GAMMA * (GAMMA - 1.0) * self.mach * self.mach


def cuda_check() -> None:
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. Install an NVIDIA driver and a "
            "Numba-compatible CUDA environment."
        )
    dev_name = cuda.get_current_device().name
    if isinstance(dev_name, bytes):
        dev_name = dev_name.decode("utf-8")
    print("CUDA device:", dev_name)


def compute_z_grid(nz: int, mesh_type: str):
    """Computes wall-normal coordinate, metric Jacobian, and inverse Jacobian."""
    if mesh_type not in ("cosine", "tanh"):
        raise ValueError(f"Unsupported mesh_type '{mesh_type}'. Use 'cosine' or 'tanh'.")

    xi = np.linspace(0.0, 2.0, nz).astype(np.float64)
    if mesh_type == "cosine":
        x3 = 1.0 - np.cos(0.5 * xi * math.pi)
        jac = 0.5 * math.pi * np.sin(0.5 * xi * math.pi)
        inv_jac = np.zeros_like(jac)
        inv_jac[1:-1] = 1.0 / jac[1:-1]
    else:
        tanh_half_ep = math.tanh(0.5 * EP)
        tanh_arg = np.tanh(0.5 * EP * (xi - 1.0))
        x3 = 1.0 + tanh_arg / tanh_half_ep
        x3[0] = 0.0
        jac = 0.5 * EP * (1.0 - tanh_arg ** 2) / tanh_half_ep
        inv_jac = np.zeros_like(jac)
        inv_jac[1:] = 1.0 / jac[1:]
    return x3, jac, inv_jac


def detect_mesh_type(z_coords: np.ndarray, tol: float = 1.0e-4) -> str:
    """Identifies whether loaded wall-normal coordinates follow cosine or tanh stretching."""
    z_arr = np.asarray(z_coords, dtype=np.float64)
    z_cos, _, _ = compute_z_grid(len(z_arr), "cosine")
    z_tanh, _, _ = compute_z_grid(len(z_arr), "tanh")
    err_cos = float(np.max(np.abs(z_arr - z_cos)))
    err_tanh = float(np.max(np.abs(z_arr - z_tanh)))
    best_type = "cosine" if err_cos <= err_tanh else "tanh"
    best_err = min(err_cos, err_tanh)
    if best_err > tol:
        print(
            f"Warning: loaded z coordinates differ from '{best_type}' grid by max {best_err:.3e} "
            f"(cos err={err_cos:.3e}, tanh err={err_tanh:.3e})."
        )
    else:
        print(f"Auto-detected mesh stretching: {best_type} (max |dz|={best_err:.3e})")
    return best_type


def compute_wall_stencil_coeffs(z: np.ndarray) -> dict[str, np.ndarray]:
    """Computes 2nd- and 3rd-order one-sided wall derivative coefficients (Coefficients())."""
    n = len(z)
    # 2nd-order 3-point 1st derivative (f1d2c*, b1d2c*)
    dz1 = z[1] - z[0]
    alpha_f = (z[2] - z[1]) / dz1
    den_f = dz1 * alpha_f * (alpha_f + 1.0)
    low2 = np.array(
        [
            (1.0 - (alpha_f + 1.0) ** 2) / den_f,
            ((alpha_f + 1.0) ** 2) / den_f,
            -1.0 / den_f,
        ],
        dtype=np.float64,
    )

    dzN = z[n - 1] - z[n - 2]
    alpha_b = (z[n - 2] - z[n - 3]) / dzN
    den_b = dzN * alpha_b * (alpha_b + 1.0)
    high2 = np.array(
        [
            ((alpha_b + 1.0) ** 2 - 1.0) / den_b,
            -((alpha_b + 1.0) ** 2) / den_b,
            1.0 / den_b,
        ],
        dtype=np.float64,
    )

    # Bottom 4-point intervals
    d21 = z[1] - z[0]
    d31 = z[2] - z[0]
    d32 = z[2] - z[1]
    d41 = z[3] - z[0]
    d42 = z[3] - z[1]
    d43 = z[3] - z[2]

    low3 = np.array(
        [
            (d31 * d21 + d31 * d41 + d21 * d41) / ((-d21) * (-d31) * (-d41)),
            (d41 * d31) / (d21 * (-d32) * (-d42)),
            (d41 * d21) / (d31 * d32 * (-d43)),
            (d31 * d21) / (d41 * d42 * d43),
        ],
        dtype=np.float64,
    )

    low2_d2 = np.array(
        [
            (2.0 * (-d21 - d31 - d41)) / ((-d21) * (-d31) * (-d41)),
            (2.0 * (-d31 - d41)) / (d21 * (-d32) * (-d42)),
            (2.0 * (-d21 - d41)) / (d31 * d32 * (-d43)),
            (2.0 * (-d21 - d31)) / (d41 * d42 * d43),
        ],
        dtype=np.float64,
    )

    # Top 4-point intervals
    b12 = z[n - 1] - z[n - 2]
    b13 = z[n - 1] - z[n - 3]
    b23 = z[n - 2] - z[n - 3]
    b14 = z[n - 1] - z[n - 4]
    b24 = z[n - 2] - z[n - 4]
    b34 = z[n - 3] - z[n - 4]

    high3 = np.array(
        [
            (b13 * b12 + b13 * b14 + b12 * b14) / (b12 * b13 * b14),
            (b14 * b13) / ((-b12) * b23 * b24),
            (b14 * b12) / ((-b13) * (-b23) * b34),
            (b13 * b12) / ((-b14) * (-b24) * (-b34)),
        ],
        dtype=np.float64,
    )

    high2_d2 = np.array(
        [
            (2.0 * (b12 + b13 + b14)) / (b12 * b13 * b14),
            (2.0 * (b13 + b14)) / ((-b12) * b23 * b24),
            (2.0 * (b12 + b14)) / ((-b13) * (-b23) * b34),
            (2.0 * (b12 + b13)) / ((-b14) * (-b24) * (-b34)),
        ],
        dtype=np.float64,
    )

    wall_dz = np.array([d21, d32, b12, b23], dtype=np.float64)
    return {
        "low2": low2,
        "high2": high2,
        "low2_d2": low2_d2,
        "high2_d2": high2_d2,
        "low3": low3,
        "high3": high3,
        "wall_dz": wall_dz,
    }


def make_mesh(cfg: SolverConfig):
    """Equivalent of setupMesh()/mod_comdata:setupMesh()."""
    x1 = np.linspace(0.0, 4.0 * math.pi, cfg.nx).astype(np.float64)
    x2 = np.linspace(0.0, 4.0 * math.pi / 3.0, cfg.ny).astype(np.float64)
    xi = np.linspace(0.0, 2.0, cfg.nz).astype(np.float64)

    dx = float(x1[1] - x1[0])
    dy = float(x2[1] - x2[0])
    dxi = float(xi[1] - xi[0])

    resolved_mesh = "tanh" if cfg.mesh_type == "auto" else cfg.mesh_type
    x3, jac, inv_jac = compute_z_grid(cfg.nz, resolved_mesh)
    coeffs = compute_wall_stencil_coeffs(x3)

    return x1, x2, x3, jac, inv_jac, dx, dy, dxi, coeffs["low3"], coeffs["high3"]


def write_mesh_tecplot(path: str, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> None:
    """Writes Mesh.dat in Tecplot POINT format matching Fortran setupMesh() when debugOp=1."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    nx, ny, nz = len(x1), len(x2), len(x3)
    X, Y, Z = np.meshgrid(x1, x2, x3, indexing="ij")
    pts = np.stack(
        [X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)],
        axis=-1,
    ).reshape(-1, 3)
    header = (
        'TITLE ="Mesh"\n'
        'Variables ="x","y","z"\n'
        f'ZONE k={nz},j={ny},i={nx},DATAPACKING="POINT"'
    )
    np.savetxt(path, pts, fmt="%8.4f %8.4f %8.4f", header=header, comments="")
    print(f"Wrote debug mesh file: {path}")


def load_param_dat(path: str, cfg: SolverConfig) -> str | None:
    """Parses runtime and control parameters from param.dat (v1.9 and v2.0 formats)."""
    if not os.path.exists(path):
        return None
    input_path = None
    topo = list(cfg.topo)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = [s.strip() for s in line.split(":", 1)]
            key_l = key.lower()
            if not val:
                continue
            if "brief description" in key_l or "case description" in key_l:
                cfg.description = val
            elif "restart state" in key_l or "restart option" in key_l:
                cfg.restart_option = int(val.split()[0])
            elif "no of mpi process in x-direction" in key_l:
                topo[0] = int(val.split()[0])
            elif "no of mpi process in y-direction" in key_l:
                topo[1] = int(val.split()[0])
            elif "no of mpi process in z-direction" in key_l:
                topo[2] = int(val.split()[0])
            elif "process topology" in key_l:
                parts = [int(tok) for tok in val.split()[:3]]
                if len(parts) == 3:
                    topo = parts
            elif "reynolds number" in key_l:
                cfg.reynolds = float(val)
            elif "mach number" in key_l:
                cfg.mach = float(val)
            elif "timestep" in key_l:
                cfg.dt = float(val)
            elif "frequency of calculating spatial statistics" in key_l:
                cfg.statistic_frequency = int(val)
            elif "peclet in x direction" in key_l:
                cfg.peclet_x = float(val)
            elif "peclet in y direction" in key_l:
                cfg.peclet_y = float(val)
            elif "peclet in z direction" in key_l:
                cfg.peclet_z = float(val)
            elif "debug state" in key_l or "debug option" in key_l:
                cfg.debug_op = int(val.split()[0])
            elif "relative path of initial solution" in key_l:
                input_path = val
            elif "kappa" in key_l:
                cfg.actuation_wavenumber = float(val)
            elif "omega" in key_l:
                cfg.actuation_omega = float(val)
            elif "amp_v" in key_l:
                cfg.actuation_amplitude = float(val)
            elif "beta" in key_l:
                cfg.actuation_beta = float(val)
            elif "amp_ls" in key_l:
                cfg.actuation_amplitude_ls = float(val)
    cfg.topo = (topo[0], topo[1], topo[2])
    return input_path


def _resolve_input_path(path: str, extra_candidates: tuple[str, ...] = ()) -> str:
    """Resolves a relative Fortran path (e.g. ../input/3D.dat) from the repo root."""
    candidates = [path]
    cleaned = path.replace("\\", "/").lstrip("./")
    if cleaned != path:
        candidates.append(cleaned)
    if path.startswith("../") or path.startswith("..\\"):
        candidates.append(path[3:])
    alt = os.path.basename(path)
    candidates.extend([os.path.join("input", alt), os.path.join("output", alt), alt])
    candidates.extend(extra_candidates)
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Input file '{path}' not found.")


def _parse_tecplot_header(path: str) -> tuple[float, int, int | None, int | None, int | None]:
    time_val = 0.0
    nstep_val = 0
    nx_h = ny_h = nz_h = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        l1 = f.readline()
        _ = f.readline()
        l3 = f.readline()
    m_title = re.search(r'TITLE\s*=\s*"([^"]+)"', l1, re.IGNORECASE)
    if m_title:
        parts = m_title.group(1).split()
        if len(parts) >= 2:
            try:
                time_val = float(parts[0])
                nstep_val = int(parts[1])
            except ValueError:
                pass
    m_k = re.search(r"k\s*=\s*(\d+)", l3, re.IGNORECASE)
    m_j = re.search(r"j\s*=\s*(\d+)", l3, re.IGNORECASE)
    m_i = re.search(r"i\s*=\s*(\d+)", l3, re.IGNORECASE)
    if m_k and m_j and m_i:
        nz_h = int(m_k.group(1))
        ny_h = int(m_j.group(1))
        nx_h = int(m_i.group(1))
    return time_val, nstep_val, nx_h, ny_h, nz_h


def _assemble_q_from_grid_data(grid_data: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    """Auto-detects wall-normal mesh stretching and converts (nz, ny, nx, >=10) primitives to q."""
    p = grid_data[..., -10:]
    if cfg.mesh_type == "auto":
        cfg.mesh_type = detect_mesh_type(p[:, 0, 0, 2])

    rho = p[..., 3]
    u = p[..., 4]
    vv = p[..., 5]
    w = p[..., 6]
    e = p[..., 7]

    q = np.empty((cfg.nx, cfg.ny, cfg.nz, NQ), dtype=np.float64)
    q[..., 0] = rho.transpose(2, 1, 0)
    q[..., 1] = (rho * u).transpose(2, 1, 0)
    q[..., 2] = (rho * vv).transpose(2, 1, 0)
    q[..., 3] = (rho * w).transpose(2, 1, 0)
    q[..., 4] = (rho * e).transpose(2, 1, 0)
    return q


def _read_tecplot_point_block(path: str, expected_rows: int) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(
        path,
        skiprows=3,
        sep=r"\s+",
        header=None,
        comment="T",
        on_bad_lines="skip",
        engine="c",
    )
    df = df.dropna().iloc[:expected_rows]
    data = df.to_numpy(dtype=np.float64)
    if data.shape[0] < expected_rows:
        raise ValueError(
            f"Input file '{path}' contains {data.shape[0]} valid rows; expected {expected_rows}."
        )
    return data


def read_combined_input(path: str, cfg: SolverConfig):
    """
    Reads Tecplot formatted combined CFD input file (resOp=1, e.g. input/3D.dat).
    Automatically parses header dimensions (nx, ny, nz), detects mesh stretching,
    and returns conserved state q plus (time, nstep).
    """
    resolved_path = _resolve_input_path(path)
    time_val, nstep_val, nx_h, ny_h, nz_h = _parse_tecplot_header(resolved_path)
    if nx_h and ny_h and nz_h:
        cfg.nx, cfg.ny, cfg.nz = nx_h, ny_h, nz_h

    expected = cfg.nx * cfg.ny * cfg.nz
    print(f"Loading input '{resolved_path}' (Grid: {cfg.nx}x{cfg.ny}x{cfg.nz}, Expected points: {expected})...")
    t0 = time.time()
    data = _read_tecplot_point_block(resolved_path, expected)
    print(f"Input loaded in {time.time() - t0:.2f} s.")

    grid_data = data.reshape(cfg.nz, cfg.ny, cfg.nx, -1)
    q = _assemble_q_from_grid_data(grid_data, cfg)
    return q, time_val, nstep_val


def read_component_inputs(dom_info_path: str, cfg: SolverConfig):
    """
    Reads partitioned MPI component files (resOp=2) using domInfo.txt and
    3D001.dat..3Dnnn.dat, matching mod_in_out.f90:case(2) and combine.f90.
    """
    resolved_dom = _resolve_input_path(
        dom_info_path, extra_candidates=("output/domInfo.txt", "domInfo.txt")
    )
    base_dir = os.path.dirname(resolved_dom) or "."

    with open(resolved_dom, "r", encoding="utf-8") as f:
        dims = [int(tok) for tok in f.readline().split()]
        nproc = int(f.readline().strip().split()[0])
        bounds = []
        for _ in range(nproc):
            vals = [int(tok) for tok in f.readline().split()]
            if len(vals) != 6:
                raise ValueError(f"Malformed bounds row in '{resolved_dom}': {vals}")
            bounds.append(vals)

    cfg.nx, cfg.ny, cfg.nz = dims[0], dims[1], dims[2]
    grid_data = np.empty((cfg.nz, cfg.ny, cfg.nx, 10), dtype=np.float64)
    time_val = 0.0
    nstep_val = 0

    print(
        f"Loading {nproc} component restart files using '{resolved_dom}' "
        f"(Grid: {cfg.nx}x{cfg.ny}x{cfg.nz})..."
    )
    t0 = time.time()
    for rank, (bx, ex, by, ey, bz, ez) in enumerate(bounds):
        comp_path = os.path.join(base_dir, f"3D{rank + 1:03d}.dat")
        if not os.path.exists(comp_path):
            raise FileNotFoundError(f"Missing component restart file: '{comp_path}'")
        t_r, step_r, _, _, _ = _parse_tecplot_header(comp_path)
        if rank == 0:
            time_val, nstep_val = t_r, step_r
        lx, ly, lz = ex - bx + 1, ey - by + 1, ez - bz + 1
        block = _read_tecplot_point_block(comp_path, lx * ly * lz)
        block = block[..., -10:].reshape(lz, ly, lx, 10)
        grid_data[bz - 1 : ez, by - 1 : ey, bx - 1 : ex, :] = block

    print(f"Component files loaded in {time.time() - t0:.2f} s.")
    q = _assemble_q_from_grid_data(grid_data, cfg)
    return q, time_val, nstep_val


def demo_initial_condition(cfg: SolverConfig):
    """Small/self-contained starting field for testing the CUDA port."""
    x1, x2, x3, *_ = make_mesh(cfg)
    X, Y, Z = np.meshgrid(x1, x2, x3, indexing="ij")

    rho = np.ones_like(X)
    u = np.ones_like(X)
    v = 0.01 * np.sin(X)
    w = np.zeros_like(X)

    T = np.ones_like(X)
    # Same algebra as the original program:
    # E = T + ggM/2 * (u^2+v^2+w^2)
    E = T + 0.5 * cfg.ggM * (u * u + v * v + w * w)

    q = np.empty((cfg.nx, cfg.ny, cfg.nz, NQ), dtype=np.float64)
    q[..., 0] = rho
    q[..., 1] = rho * u
    q[..., 2] = rho * v
    q[..., 3] = rho * w
    q[..., 4] = rho * E
    return q


@cuda.jit
def primary_variables_kernel(q, var, mu, kt, ggM, gamma, mach, init_all):
    """
    Equivalent of find_primary_variable().

    When init_all == 1, converts all (0..nx-1, 0..ny-1, 0..nz-1) cells so
    restarted or synthetic wall values are populated. During RK2 substeps
    (init_all == 0), updates interior fluid cells (1..nx-2, 1..ny-2, 1..nz-2)
    and only refreshes wall viscosity/conductivity from current T.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]

    if i >= nx or j >= ny or k >= nz:
        return

    if init_all == 0 and (k == 0 or k == nz - 1):
        mu[i, j, k] = max(var[i, j, k, 5], 1.0e-12) ** 0.7
        kt[i, j, k] = mu[i, j, k]
        return

    if init_all == 0 and (i == 0 or i == nx - 1 or j == 0 or j == ny - 1):
        return

    rho = max(q[i, j, k, 0], 1.0e-14)
    u = q[i, j, k, 1] / rho
    v = q[i, j, k, 2] / rho
    w = q[i, j, k, 3] / rho
    E = q[i, j, k, 4] / rho

    kinetic = 0.5 * (u * u + v * v + w * w)
    T = E - ggM * kinetic
    P = rho * T / (gamma * mach * mach)

    var[i, j, k, 0] = rho
    var[i, j, k, 1] = u
    var[i, j, k, 2] = v
    var[i, j, k, 3] = w
    var[i, j, k, 4] = E
    var[i, j, k, 5] = T
    var[i, j, k, 6] = P

    mu[i, j, k] = max(T, 1.0e-12) ** 0.7
    kt[i, j, k] = mu[i, j, k]


@cuda.jit(device=True)
def _copy_state_cell(q, var, mu, kt, dst_i, dst_j, src_i, src_j, k):
    for c in range(NQ):
        q[dst_i, dst_j, k, c] = q[src_i, src_j, k, c]
    for c in range(NV):
        var[dst_i, dst_j, k, c] = var[src_i, src_j, k, c]
    mu[dst_i, dst_j, k] = mu[src_i, src_j, k]
    kt[dst_i, dst_j, k] = kt[src_i, src_j, k]


@cuda.jit
def sync_halos_x_kernel(q, var, mu, kt, k_low, k_high):
    """Copies x periodic halo faces (nx-2 -> 0 and 1 -> nx-1) to match MPI halos."""
    j, k = cuda.grid(2)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
    if j < 1 or j >= ny - 1 or k < k_low or k > k_high or k >= nz:
        return
    _copy_state_cell(q, var, mu, kt, 0, j, nx - 2, j, k)
    _copy_state_cell(q, var, mu, kt, nx - 1, j, 1, j, k)


@cuda.jit
def sync_halos_y_kernel(q, var, mu, kt, k_low, k_high):
    """Copies y periodic halo faces (ny-2 -> 0 and 1 -> ny-1) across all x indices."""
    i, k = cuda.grid(2)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
    if i >= nx or k < k_low or k > k_high or k >= nz:
        return
    _copy_state_cell(q, var, mu, kt, i, 0, i, ny - 2, k)
    _copy_state_cell(q, var, mu, kt, i, ny - 1, i, 1, k)


@cuda.jit(device=True)
def _upwind3_flux(a, b, qm2, qm1, q0, qp1, qp2):
    """Third-order sign-switched biased/upwind interface flux difference (unscaled by 1/(8*h))."""
    if a >= 0.0 and b >= 0.0:
        return (
            a * (6.0 * q0 + 3.0 * qp1 - qm1)
            - b * (6.0 * qm1 + 3.0 * q0 - qm2)
        ) / 8.0
    if a < 0.0 and b < 0.0:
        return (
            a * (6.0 * qp1 + 3.0 * q0 - qp2)
            - b * (6.0 * q0 + 3.0 * qm1 - qp1)
        ) / 8.0
    if a < 0.0 and b >= 0.0:
        return (
            a * (6.0 * qp1 + 3.0 * q0 - qp2)
            - b * (6.0 * qm1 + 3.0 * q0 - qm2)
        ) / 8.0
    return (
        a * (6.0 * q0 + 3.0 * qp1 - qm1)
        - b * (6.0 * q0 + 3.0 * qm1 - qp1)
    ) / 8.0


@cuda.jit
def convective_kernel(
    q,
    var,
    mu,
    F,
    G,
    H,
    dx,
    dy,
    dxi,
    jac,
    inv_jac,
    reynolds,
    px_limit,
    py_limit,
    pz_limit,
):
    """
    GPU version of calc_convective_terms().

    Operates strictly over interior physical points 1..nx-2, 1..ny-2, 1..nz-2
    (matching Fortran bn(k):en(k)) with periodic wrapping modulo (nx-2, ny-2).
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]

    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
        return

    nx_int = nx - 2
    ny_int = ny - 2

    im2 = (i - 3 + nx_int) % nx_int + 1
    im1 = (i - 2 + nx_int) % nx_int + 1
    ip1 = (i % nx_int) + 1
    ip2 = ((i + 1) % nx_int) + 1

    jm2 = (j - 3 + ny_int) % ny_int + 1
    jm1 = (j - 2 + ny_int) % ny_int + 1
    jp1 = (j % ny_int) + 1
    jp2 = ((j + 1) % ny_int) + 1

    km1 = k - 1
    kp1 = k + 1
    km2 = max(k - 2, 0)
    kp2 = min(k + 2, nz - 1)

    mu_cur = max(mu[i, j, k], 1.0e-14)

    # ------------------------- x direction ------------------------------
    pe_x = reynolds * abs(q[i, j, k, 1]) * dx / mu_cur
    use_upwind_x = (pe_x > px_limit) or (i == 1) or (i == nx - 2)
    a_x = 0.5 * (var[i, j, k, 1] + var[ip1, j, k, 1])
    b_x = 0.5 * (var[i, j, k, 1] + var[im1, j, k, 1])

    for c in range(NQ):
        if use_upwind_x:
            F[i, j, k, c] = (
                _upwind3_flux(
                    a_x,
                    b_x,
                    q[im2, j, k, c],
                    q[im1, j, k, c],
                    q[i, j, k, c],
                    q[ip1, j, k, c],
                    q[ip2, j, k, c],
                )
                / dx
            )
        else:
            F[i, j, k, c] = (
                -var[ip2, j, k, 1] * q[ip2, j, k, c]
                + 8.0 * var[ip1, j, k, 1] * q[ip1, j, k, c]
                - 8.0 * var[im1, j, k, 1] * q[im1, j, k, c]
                + var[im2, j, k, 1] * q[im2, j, k, c]
            ) / (12.0 * dx)

    # ------------------------- y direction ------------------------------
    pe_y = reynolds * abs(q[i, j, k, 2]) * dy / mu_cur
    use_upwind_y = (pe_y > py_limit) or (j == 1) or (j == ny - 2)
    a_y = 0.5 * (var[i, j, k, 2] + var[i, jp1, k, 2])
    b_y = 0.5 * (var[i, j, k, 2] + var[i, jm1, k, 2])

    for c in range(NQ):
        if use_upwind_y:
            G[i, j, k, c] = (
                _upwind3_flux(
                    a_y,
                    b_y,
                    q[i, jm2, k, c],
                    q[i, jm1, k, c],
                    q[i, j, k, c],
                    q[i, jp1, k, c],
                    q[i, jp2, k, c],
                )
                / dy
            )
        else:
            G[i, j, k, c] = (
                -var[i, jp2, k, 2] * q[i, jp2, k, c]
                + 8.0 * var[i, jp1, k, 2] * q[i, jp1, k, c]
                - 8.0 * var[i, jm1, k, 2] * q[i, jm1, k, c]
                + var[i, jm2, k, 2] * q[i, jm2, k, c]
            ) / (12.0 * dy)

    # ------------------------- z direction ------------------------------
    pe_z = reynolds * abs(q[i, j, k, 3]) * dxi * jac[k] / mu_cur
    a_z = 0.5 * (var[i, j, k, 3] + var[i, j, kp1, 3])
    b_z = 0.5 * (var[i, j, k, 3] + var[i, j, km1, 3])

    for c in range(NQ):
        if k < 2 or k > nz - 3:
            # Same active choice as H_central_2D() in mod_conv.f90.
            H[i, j, k, c] = (
                (
                    var[i, j, kp1, 3] * q[i, j, kp1, c]
                    - var[i, j, km1, 3] * q[i, j, km1, c]
                )
                * 0.5
                * inv_jac[k]
                / dxi
            )
        elif pe_z > pz_limit:
            H[i, j, k, c] = (
                _upwind3_flux(
                    a_z,
                    b_z,
                    q[i, j, km2, c],
                    q[i, j, km1, c],
                    q[i, j, k, c],
                    q[i, j, kp1, c],
                    q[i, j, kp2, c],
                )
                * inv_jac[k]
                / dxi
            )
        else:
            H[i, j, k, c] = (
                (
                    -var[i, j, kp2, 3] * q[i, j, kp2, c]
                    + 8.0 * var[i, j, kp1, 3] * q[i, j, kp1, c]
                    - 8.0 * var[i, j, km1, 3] * q[i, j, km1, c]
                    + var[i, j, km2, 3] * q[i, j, km2, c]
                )
                * inv_jac[k]
                / (12.0 * dxi)
            )


@cuda.jit(device=True)
def _deriv_z_var(var, i, j, k, nz, comp, inv_dxi, inv_jac_k, wall_order, low2, high2, low3, high3):
    """Wall-normal derivative supporting both 2nd- and 3rd-order one-sided wall stencils."""
    if k == 0:
        if wall_order == 2:
            return (
                low2[0] * var[i, j, 0, comp]
                + low2[1] * var[i, j, 1, comp]
                + low2[2] * var[i, j, 2, comp]
            )
        return (
            low3[0] * var[i, j, 0, comp]
            + low3[1] * var[i, j, 1, comp]
            + low3[2] * var[i, j, 2, comp]
            + low3[3] * var[i, j, 3, comp]
        )
    if k == nz - 1:
        if wall_order == 2:
            return (
                high2[0] * var[i, j, nz - 1, comp]
                + high2[1] * var[i, j, nz - 2, comp]
                + high2[2] * var[i, j, nz - 3, comp]
            )
        return (
            high3[0] * var[i, j, nz - 1, comp]
            + high3[1] * var[i, j, nz - 2, comp]
            + high3[2] * var[i, j, nz - 3, comp]
            + high3[3] * var[i, j, nz - 4, comp]
        )
    return 0.5 * (var[i, j, k + 1, comp] - var[i, j, k - 1, comp]) * inv_dxi * inv_jac_k


@cuda.jit
def diffusive_kernel(
    var,
    mu,
    kt,
    fD,
    gD,
    hD,
    inv_dx,
    inv_dy,
    inv_dxi,
    inv_jac,
    inv_re,
    inv_pr,
    ggM,
    gamma,
    wall_order,
    low2,
    high2,
    low3,
    high3,
):
    """
    GPU version of calc_diffusive_terms_2D() / calc_diffusive_terms_3D().

    Runs over 1..nx-2, 1..ny-2, 0..nz-1 (bs(3):es(3) so wall hD is ready for k=1 and k=nz-2).
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]

    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k >= nz:
        return

    nx_int = nx - 2
    ny_int = ny - 2
    im1 = (i - 2 + nx_int) % nx_int + 1
    ip1 = (i % nx_int) + 1
    jm1 = (j - 2 + ny_int) % ny_int + 1
    jp1 = (j % ny_int) + 1

    u = var[i, j, k, 1]
    vv = var[i, j, k, 2]
    w = var[i, j, k, 3]
    p = var[i, j, k, 6]
    muv = mu[i, j, k]
    ktv = kt[i, j, k]

    dudx = 0.5 * (var[ip1, j, k, 1] - var[im1, j, k, 1]) * inv_dx
    dudy = 0.5 * (var[i, jp1, k, 1] - var[i, jm1, k, 1]) * inv_dy

    dvdx = 0.5 * (var[ip1, j, k, 2] - var[im1, j, k, 2]) * inv_dx
    dvdy = 0.5 * (var[i, jp1, k, 2] - var[i, jm1, k, 2]) * inv_dy

    dwdx = 0.5 * (var[ip1, j, k, 3] - var[im1, j, k, 3]) * inv_dx
    dwdy = 0.5 * (var[i, jp1, k, 3] - var[i, jm1, k, 3]) * inv_dy

    dTdx = 0.5 * (var[ip1, j, k, 5] - var[im1, j, k, 5]) * inv_dx
    dTdy = 0.5 * (var[i, jp1, k, 5] - var[i, jm1, k, 5]) * inv_dy

    inv_jac_k = inv_jac[k]
    dudz = _deriv_z_var(var, i, j, k, nz, 1, inv_dxi, inv_jac_k, wall_order, low2, high2, low3, high3)
    dvdz = _deriv_z_var(var, i, j, k, nz, 2, inv_dxi, inv_jac_k, wall_order, low2, high2, low3, high3)
    dwdz = _deriv_z_var(var, i, j, k, nz, 3, inv_dxi, inv_jac_k, wall_order, low2, high2, low3, high3)
    dTdz = _deriv_z_var(var, i, j, k, nz, 5, inv_dxi, inv_jac_k, wall_order, low2, high2, low3, high3)

    divg = dudx + dvdy + dwdz

    diss_f = ggM * inv_re * (
        TWO_THIRDS * muv * u * divg
        - 2.0 * muv * u * dudx
        - muv * vv * (dvdx + dudy)
        - muv * w * (dwdx + dudz)
    )
    diss_g = ggM * inv_re * (
        -muv * u * (dvdx + dudy)
        + TWO_THIRDS * muv * vv * divg
        - 2.0 * muv * vv * dvdy
        - muv * w * (dwdy + dvdz)
    )
    diss_h = ggM * inv_re * (
        -muv * u * (dwdx + dudz)
        - muv * vv * (dwdy + dvdz)
        + TWO_THIRDS * muv * w * divg
        - 2.0 * muv * w * dwdz
    )

    fD[i, j, k, 0] = 0.0
    fD[i, j, k, 1] = p + TWO_THIRDS * muv * inv_re * divg - 2.0 * muv * inv_re * dudx
    fD[i, j, k, 2] = -muv * inv_re * (dvdx + dudy)
    fD[i, j, k, 3] = -muv * inv_re * (dwdx + dudz)
    fD[i, j, k, 4] = -gamma * ktv * inv_re * inv_pr * dTdx + ggM * u * p + diss_f

    gD[i, j, k, 0] = 0.0
    gD[i, j, k, 1] = -muv * inv_re * (dvdx + dudy)
    gD[i, j, k, 2] = p + TWO_THIRDS * muv * inv_re * divg - 2.0 * muv * inv_re * dvdy
    gD[i, j, k, 3] = -muv * inv_re * (dwdy + dvdz)
    gD[i, j, k, 4] = -gamma * ktv * inv_re * inv_pr * dTdy + ggM * vv * p + diss_g

    hD[i, j, k, 0] = 0.0
    hD[i, j, k, 1] = -muv * inv_re * (dwdx + dudz)
    hD[i, j, k, 2] = -muv * inv_re * (dwdy + dvdz)
    hD[i, j, k, 3] = p + TWO_THIRDS * muv * inv_re * divg - 2.0 * muv * inv_re * dwdz
    hD[i, j, k, 4] = -gamma * ktv * inv_re * inv_pr * dTdz + ggM * w * p + diss_h


@cuda.jit
def rhs_and_rk2_kernel(
    q,
    q_stage,
    var,
    F,
    G,
    H,
    fD,
    gD,
    hD,
    rho_mean,
    dx,
    dy,
    dxi,
    inv_jac,
    dt,
    ggM,
    tau_lower,
    tau_upper,
    iteration,
):
    """
    Equivalent of calc_FHD_and_srcV() plus the two RK2 updates.
    Executes only over interior points 1..nx-2, 1..ny-2, 1..nz-2 (bn(k):en(k)).
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = q.shape[0], q.shape[1], q.shape[2]

    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
        return

    nx_int = nx - 2
    ny_int = ny - 2
    im1 = (i - 2 + nx_int) % nx_int + 1
    ip1 = (i % nx_int) + 1
    jm1 = (j - 2 + ny_int) % ny_int + 1
    jp1 = (j % ny_int) + 1
    km1 = k - 1
    kp1 = k + 1

    body = 0.5 * (tau_lower - tau_upper) * rho_mean[k]

    for c in range(NQ):
        div_diff = (
            0.5 * (fD[ip1, j, k, c] - fD[im1, j, k, c]) / dx
            + 0.5 * (gD[i, jp1, k, c] - gD[i, jm1, k, c]) / dy
            + 0.5 * (hD[i, j, kp1, c] - hD[i, j, km1, c]) / dxi * inv_jac[k]
        )

        conv = F[i, j, k, c] + G[i, j, k, c] + H[i, j, k, c]

        source = 0.0
        if c == 1:
            source = body
        elif c == 4:
            source = ggM * var[i, j, k, 1] * body

        R = source - (conv + div_diff)

        if iteration == 1:
            q_stage[i, j, k, c] = q[i, j, k, c]
            q[i, j, k, c] = q[i, j, k, c] + 0.5 * dt * R
        else:
            q[i, j, k, c] = q_stage[i, j, k, c] + dt * R


@cuda.jit
def opposition_control_kernel(q, var, opposition_distance, enable_oc):
    """
    GPU version of mod_control:OC().
    Applies wall-normal opposition control at 1-based plane y_d (0-based y_d - 1)
    and top mirror nz - y_d, or enforces zero wall w when enable_oc == 0.
    """
    i, j = cuda.grid(2)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1:
        return

    if enable_oc:
        src_b = min(max(opposition_distance - 1, 1), nz - 2)
        src_t = min(max(nz - opposition_distance, 1), nz - 2)

        w_q_b = -q[i, j, src_b, 3]
        w_q_t = -q[i, j, src_t, 3]

        var[i, j, 0, 3] = w_q_b / max(var[i, j, 0, 0], 1.0e-14)
        q[i, j, 0, 3] = w_q_b

        var[i, j, nz - 1, 3] = w_q_t / max(var[i, j, nz - 1, 0], 1.0e-14)
        q[i, j, nz - 1, 3] = w_q_t
    else:
        var[i, j, 0, 3] = 0.0
        q[i, j, 0, 3] = 0.0
        var[i, j, nz - 1, 3] = 0.0
        q[i, j, nz - 1, 3] = 0.0


@cuda.jit
def wall_pressure_bc_kernel(
    var,
    mu,
    rhow_prev,
    inv_dx,
    inv_dy,
    inv_re,
    inv_dt,
    wall_order,
    low2,
    high2,
    low2_d2,
    high2_d2,
    low3,
    high3,
    wall_dz,
):
    """
    GPU version of the wall-normal momentum pressure BC in bc_update_2D()/bc_update_3D().
    Reads neighbor wall velocities computed prior to wall state overwrite (race-free 2nd pass).
    """
    i, j = cuda.grid(2)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1:
        return

    nx_int = nx - 2
    ny_int = ny - 2
    im1 = (i - 2 + nx_int) % nx_int + 1
    ip1 = (i % nx_int) + 1
    jm1 = (j - 2 + ny_int) % ny_int + 1
    jp1 = (j % ny_int) + 1

    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy

    for wall_idx in range(2):
        if wall_idx == 0:
            k0, s = 0, 1
            if wall_order == 2:
                c0, c1, c2, c3 = low2[0], low2[1], low2[2], 0.0
                d2c0, d2c1, d2c2, d2c3 = low2_d2[0], low2_d2[1], low2_d2[2], low2_d2[3]
            else:
                c0, c1, c2, c3 = low3[0], low3[1], low3[2], low3[3]
                d2c0 = d2c1 = d2c2 = d2c3 = 0.0
            d1, d2 = wall_dz[0], wall_dz[1]
        else:
            k0, s = nz - 1, -1
            if wall_order == 2:
                c0, c1, c2, c3 = high2[0], high2[1], high2[2], 0.0
                d2c0, d2c1, d2c2, d2c3 = high2_d2[0], high2_d2[1], high2_d2[2], high2_d2[3]
            else:
                c0, c1, c2, c3 = high3[0], high3[1], high3[2], high3[3]
                d2c0 = d2c1 = d2c2 = d2c3 = 0.0
            d1, d2 = wall_dz[2], wall_dz[3]

        k1 = k0 + s
        k2 = k0 + 2 * s
        k3 = k0 + 3 * s

        du_dz_ip1 = (
            c0 * var[ip1, j, k0, 1]
            + c1 * var[ip1, j, k1, 1]
            + c2 * var[ip1, j, k2, 1]
            + c3 * var[ip1, j, k3, 1]
        )
        du_dz_im1 = (
            c0 * var[im1, j, k0, 1]
            + c1 * var[im1, j, k1, 1]
            + c2 * var[im1, j, k2, 1]
            + c3 * var[im1, j, k3, 1]
        )
        dudxdz = (mu[ip1, j, k0] * du_dz_ip1 - mu[im1, j, k0] * du_dz_im1) * 0.5 * inv_dx

        dudzdx = (
            (
                c0 * mu[i, j, k0] * var[ip1, j, k0, 1]
                + c1 * mu[i, j, k1] * var[ip1, j, k1, 1]
                + c2 * mu[i, j, k2] * var[ip1, j, k2, 1]
                + c3 * mu[i, j, k3] * var[ip1, j, k3, 1]
            )
            - (
                c0 * mu[i, j, k0] * var[im1, j, k0, 1]
                + c1 * mu[i, j, k1] * var[im1, j, k1, 1]
                + c2 * mu[i, j, k2] * var[im1, j, k2, 1]
                + c3 * mu[i, j, k3] * var[im1, j, k3, 1]
            )
        ) * 0.5 * inv_dx

        dv_dz_jp1 = (
            c0 * var[i, jp1, k0, 2]
            + c1 * var[i, jp1, k1, 2]
            + c2 * var[i, jp1, k2, 2]
            + c3 * var[i, jp1, k3, 2]
        )
        dv_dz_jm1 = (
            c0 * var[i, jm1, k0, 2]
            + c1 * var[i, jm1, k1, 2]
            + c2 * var[i, jm1, k2, 2]
            + c3 * var[i, jm1, k3, 2]
        )
        dvdydz = (mu[i, jp1, k0] * dv_dz_jp1 - mu[i, jm1, k0] * dv_dz_jm1) * 0.5 * inv_dy

        dvdzdy = (
            (
                c0 * mu[i, j, k0] * var[i, jp1, k0, 2]
                + c1 * mu[i, j, k1] * var[i, jp1, k1, 2]
                + c2 * mu[i, j, k2] * var[i, jp1, k2, 2]
                + c3 * mu[i, j, k3] * var[i, jp1, k3, 2]
            )
            - (
                c0 * mu[i, j, k0] * var[i, jm1, k0, 2]
                + c1 * mu[i, j, k1] * var[i, jm1, k1, 2]
                + c2 * mu[i, j, k2] * var[i, jm1, k2, 2]
                + c3 * mu[i, j, k3] * var[i, jm1, k3, 2]
            )
        ) * 0.5 * inv_dy

        w0 = var[i, j, k0, 3]
        w1 = var[i, j, k1, 3]
        w2 = var[i, j, k2, 3]
        w3 = var[i, j, k3, 3]

        drhouwdx = (
            var[ip1, j, k0, 0] * var[ip1, j, k0, 1] * var[ip1, j, k0, 3]
            - var[im1, j, k0, 0] * var[im1, j, k0, 1] * var[im1, j, k0, 3]
        ) * 0.5 * inv_dx
        d2wdx2 = mu[i, j, k0] * (var[ip1, j, k0, 3] + var[im1, j, k0, 3] - 2.0 * w0) * inv_dx2

        drhovwdy = (
            var[i, jp1, k0, 0] * var[i, jp1, k0, 2] * var[i, jp1, k0, 3]
            - var[i, jm1, k0, 0] * var[i, jm1, k0, 2] * var[i, jm1, k0, 3]
        ) * 0.5 * inv_dy
        d2wdy2 = mu[i, j, k0] * (var[i, jp1, k0, 3] + var[i, jm1, k0, 3] - 2.0 * w0) * inv_dy2

        if wall_order == 2:
            mud2wdz2 = mu[i, j, k0] * (d2c0 * w0 + d2c1 * w1 + d2c2 * w2 + d2c3 * w3)
            dwdz_w = c0 * w0 + c1 * w1 + c2 * w2
            dmudz_w = c0 * mu[i, j, k0] + c1 * mu[i, j, k1] + c2 * mu[i, j, k2]
            dwdzdmudz = dwdz_w * dmudz_w
        else:
            mud2wdz2 = (
                mu[i, j, k1] * (w2 - w1) / d2 - mu[i, j, k0] * (w1 - w0) / d1
            ) / d1
            dwdzdmudz = 0.0

        drhow2dz = (
            c0 * var[i, j, k0, 0] * w0 * w0
            + c1 * var[i, j, k1, 0] * w1 * w1
            + c2 * var[i, j, k2, 0] * w2 * w2
            + c3 * var[i, j, k3, 0] * w3 * w3
        )
        rhow_cur = var[i, j, k0, 0] * w0
        drhowdt = (rhow_cur - rhow_prev[i, j, wall_idx]) * inv_dt
        rhow_prev[i, j, wall_idx] = rhow_cur

        lhs = (
            inv_re
            * (
                -dudxdz
                - dvdydz
                - d2wdx2
                - d2wdy2
                + TWO_THIRDS * (dudzdx + dvdzdy - 2.0 * (mud2wdz2 + dwdzdmudz))
            )
            + drhow2dz
            + drhouwdx
            + drhovwdy
            + drhowdt
        )

        var[i, j, k0, 6] = -(
            lhs
            + c1 * var[i, j, k1, 6]
            + c2 * var[i, j, k2, 6]
            + c3 * var[i, j, k3, 6]
        ) / c0


@cuda.jit
def wall_state_bc_kernel(
    q,
    var,
    mu,
    kt,
    time_value,
    x1,
    ggM,
    gamma,
    mach,
    kappa,
    omega,
    amp_v,
    enable_oc,
):
    """
    Updates isothermal non-slip/actuated wall primitive and conserved state
    without overwriting opposition-controlled wall-normal velocity.
    """
    i, j = cuda.grid(2)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1:
        return

    act = amp_v * math.sin(kappa * x1[i] - omega * time_value)

    for wall_idx in range(2):
        k = 0 if wall_idx == 0 else nz - 1
        w_val = var[i, j, k, 3] if enable_oc else 0.0

        var[i, j, k, 1] = 0.0
        var[i, j, k, 2] = act
        var[i, j, k, 3] = w_val
        var[i, j, k, 5] = 1.0
        var[i, j, k, 4] = 1.0 + 0.5 * ggM * (act * act + w_val * w_val)
        rho = gamma * mach * mach * var[i, j, k, 6]
        var[i, j, k, 0] = rho

        mu[i, j, k] = 1.0
        kt[i, j, k] = 1.0

        q[i, j, k, 0] = rho
        q[i, j, k, 1] = 0.0
        q[i, j, k, 2] = rho * act
        if not enable_oc:
            q[i, j, k, 3] = 0.0
        q[i, j, k, 4] = rho * var[i, j, k, 4]


@cuda.jit
def wall_shear_kernel(var, mu, low2, high2, inv_dx, inv_re, out_taus):
    """GPU reduction for bottom and top wall shear stress matching tauWall_write()."""
    i, j = cuda.grid(2)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1:
        return

    nx_int = nx - 2
    im1 = (i - 2 + nx_int) % nx_int + 1
    ip1 = (i % nx_int) + 1

    dwdx_l = mu[i, j, 0] * (var[ip1, j, 0, 3] - var[im1, j, 0, 3]) * 0.5 * inv_dx
    dudz_l = mu[i, j, 0] * (
        low2[0] * var[i, j, 0, 1]
        + low2[1] * var[i, j, 1, 1]
        + low2[2] * var[i, j, 2, 1]
    )
    cuda.atomic.add(out_taus, 0, (dwdx_l + dudz_l) * inv_re)

    dwdx_u = mu[i, j, nz - 1] * (var[ip1, j, nz - 1, 3] - var[im1, j, nz - 1, 3]) * 0.5 * inv_dx
    dudz_u = mu[i, j, nz - 1] * (
        high2[0] * var[i, j, nz - 1, 1]
        + high2[1] * var[i, j, nz - 2, 1]
        + high2[2] * var[i, j, nz - 3, 1]
    )
    cuda.atomic.add(out_taus, 1, (dwdx_u + dudz_u) * inv_re)


@cuda.jit
def xy_rho_sum_kernel(var, out_rho_sum):
    """GPU x-y planar density sum per k level matching stat1Spc(:,1) computation."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
    if i < nx and j < ny and k < nz:
        cuda.atomic.add(out_rho_sum, k, var[i, j, k, 0])


@cuda.jit
def finite_check_kernel(var, flag):
    """Equivalent of check_blown()."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
    if i < nx and j < ny and k < nz:
        if not (var[i, j, k, 0] > 0.0) or not (var[i, j, k, 5] > 0.0):
            cuda.atomic.max(flag, 0, 1)


@cuda.jit
def sum_scalar_kernel(a, out):
    """Small GPU reduction building block replacing MPI_Reduce for a scalar."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = a.shape
    if i < nx and j < ny and k < nz:
        cuda.atomic.add(out, 0, a[i, j, k])


def blocks_2d(nx: int, ny: int, threads=(16, 8)):
    return (
        (nx + threads[0] - 1) // threads[0],
        (ny + threads[1] - 1) // threads[1],
    )


def blocks_3d(shape, threads=(8, 8, 8)):
    return (
        (shape[0] + threads[0] - 1) // threads[0],
        (shape[1] + threads[1] - 1) // threads[1],
        (shape[2] + threads[2] - 1) // threads[2],
    )


def _import_diagnostics_module():
    """Imports python/compute_diagnostics_cuda.py lazily without cwd sensitivity."""
    py_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python")
    if py_dir not in sys.path and os.path.isdir(py_dir):
        sys.path.insert(0, py_dir)
    import compute_diagnostics_cuda as cdc

    return cdc


def _write_tecplot_subzone(
    filename: str,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    var_h: np.ndarray,
    time_val: float,
    nstep_val: int,
    bx: int = 1,
    ex: int | None = None,
    by: int = 1,
    ey: int | None = None,
    bz: int = 1,
    ez: int | None = None,
) -> None:
    """Writes a full or rank-partitioned Tecplot POINT snapshot matching mod_in_out.f90."""
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    nx_g, ny_g, nz_g = var_h.shape[:3]
    ex = nx_g if ex is None else ex
    ey = ny_g if ey is None else ey
    ez = nz_g if ez is None else ez

    lx, ly, lz = ex - bx + 1, ey - by + 1, ez - bz + 1
    out_arr = np.empty((lz, ly, lx, 10), dtype=np.float64)
    x1_s = x1[bx - 1 : ex]
    x2_s = x2[by - 1 : ey]
    for kk, k in enumerate(range(bz - 1, ez)):
        out_arr[kk, :, :, 0] = x1_s[None, :]
        out_arr[kk, :, :, 1] = x2_s[:, None]
        out_arr[kk, :, :, 2] = x3[k]
        out_arr[kk, :, :, 3:] = var_h[bx - 1 : ex, by - 1 : ey, k, :].transpose(1, 0, 2)

    header = (
        f'TITLE ="{time_val:20.10f} {nstep_val:9d}"\n'
        f'Variables = "x","y","z","Rho","u","v","w","e","T","p"\n'
        f'ZONE k={lz},j={ly},i={lx},DATAPACKING=POINT'
    )
    fmt = " %10.6f %10.6f %10.6f " + " ".join(["%22.15e"] * 7)
    np.savetxt(filename, out_arr.reshape(-1, 10), fmt=fmt, header=header, comments="")


def _mpi_axis_partitions(n: int, parts: int) -> list[tuple[int, int]]:
    """Reproduces 1-based inclusive bounds from mod_my_mpi.f90 lines 37-55."""
    q = n // parts
    cutoff = parts - (n % parts)
    bounds: list[tuple[int, int]] = []
    cur = 1
    for r in range(parts):
        sz = q + 1 if r >= cutoff else q
        bounds.append((cur, cur + sz - 1))
        cur += sz
    return bounds


def _choose_component_grid(nx: int, ny: int, nz: int) -> tuple[int, int, int]:
    """Selects a balanced (px, py, pz) subdomain grid with non-empty partitions."""
    px = 2 if nx >= 8 else 1
    py = 3 if ny >= 24 else (2 if ny >= 8 else 1)
    pz = 2 if nz >= 8 else 1
    return px, py, pz


def _write_component_checkpoint(
    out_dir: str,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    var_h: np.ndarray,
    time_val: float,
    nstep_val: int,
    grid_parts: tuple[int, int, int] | None = None,
) -> str:
    """Writes domInfo.txt and partitioned 3Dnnn.dat snapshots readable via read_component_inputs()."""
    os.makedirs(out_dir, exist_ok=True)
    nx, ny, nz = var_h.shape[:3]
    px, py, pz = grid_parts or _choose_component_grid(nx, ny, nz)
    xb = _mpi_axis_partitions(nx, px)
    yb = _mpi_axis_partitions(ny, py)
    zb = _mpi_axis_partitions(nz, pz)
    bounds = [
        (bx, ex, by, ey, bz, ez)
        for bx, ex in xb
        for by, ey in yb
        for bz, ez in zb
    ]
    dom_path = os.path.join(out_dir, "domInfo.txt")
    with open(dom_path, "w", encoding="utf-8") as f:
        f.write(f" {nx:6d} {ny:6d} {nz:6d}\n")
        f.write(f" {len(bounds):6d}\n")
        for bx, ex, by, ey, bz, ez in bounds:
            f.write(f" {bx:6d} {ex:6d} {by:6d} {ey:6d} {bz:6d} {ez:6d}\n")
    for rank, (bx, ex, by, ey, bz, ez) in enumerate(bounds):
        comp_path = os.path.join(out_dir, f"3D{rank + 1:03d}.dat")
        _write_tecplot_subzone(
            comp_path,
            x1,
            x2,
            x3,
            var_h,
            time_val,
            nstep_val,
            bx=bx,
            ex=ex,
            by=by,
            ey=ey,
            bz=bz,
            ez=ez,
        )
    return dom_path


class CUDASolver:
    """Single-GPU replacement for the MPI solver."""

    def __init__(self, cfg: SolverConfig, q_host: np.ndarray):
        if q_host.shape != (cfg.nx, cfg.ny, cfg.nz, NQ):
            raise ValueError(f"q_host shape must be {(cfg.nx, cfg.ny, cfg.nz, NQ)}")

        self.cfg = cfg
        if cfg.wall_order not in (2, 3):
            raise ValueError(f"wall_order must be 2 or 3, got {cfg.wall_order}")
        if cfg.stats_mode not in ("none", "minimal", "standard", "full"):
            raise ValueError(
                f"stats_mode must be 'none', 'minimal', 'standard', or 'full', got {cfg.stats_mode!r}"
            )

        (
            x1,
            x2,
            x3,
            jac,
            inv_jac,
            dx,
            dy,
            dxi,
            low3,
            high3,
        ) = make_mesh(cfg)
        coeffs = compute_wall_stencil_coeffs(x3)

        self.x1_h = x1
        self.x2_h = x2
        self.x3_h = x3

        self.dx = dx
        self.dy = dy
        self.dxi = dxi

        if cfg.debug_op == 1:
            write_mesh_tecplot("Mesh.dat", x1, x2, x3)

        self.d_q = cuda.to_device(np.ascontiguousarray(q_host))
        self.d_q_stage = cuda.device_array_like(self.d_q)

        shape4 = (cfg.nx, cfg.ny, cfg.nz, NV)
        shape5 = (cfg.nx, cfg.ny, cfg.nz, NQ)

        self.d_var = cuda.device_array(shape4, dtype=np.float64)
        self.d_mu = cuda.device_array((cfg.nx, cfg.ny, cfg.nz), dtype=np.float64)
        self.d_kt = cuda.device_array((cfg.nx, cfg.ny, cfg.nz), dtype=np.float64)

        self.d_F = cuda.device_array(shape5, dtype=np.float64)
        self.d_G = cuda.device_array(shape5, dtype=np.float64)
        self.d_H = cuda.device_array(shape5, dtype=np.float64)

        self.d_fD = cuda.device_array(shape5, dtype=np.float64)
        self.d_gD = cuda.device_array(shape5, dtype=np.float64)
        self.d_hD = cuda.device_array(shape5, dtype=np.float64)

        self.d_x1 = cuda.to_device(x1)
        self.d_jac = cuda.to_device(jac)
        self.d_inv_jac = cuda.to_device(inv_jac)
        self.d_low2 = cuda.to_device(coeffs["low2"])
        self.d_high2 = cuda.to_device(coeffs["high2"])
        self.d_low2_d2 = cuda.to_device(coeffs["low2_d2"])
        self.d_high2_d2 = cuda.to_device(coeffs["high2_d2"])
        self.d_low3 = cuda.to_device(low3)
        self.d_high3 = cuda.to_device(high3)
        self.d_wall_dz = cuda.to_device(coeffs["wall_dz"])

        self.d_rhow_prev = cuda.to_device(np.zeros((cfg.nx, cfg.ny, 2), dtype=np.float64))
        self.d_rho_mean = cuda.to_device(np.ones(cfg.nz, dtype=np.float64))
        self.d_rho_sum_buf = cuda.device_array(cfg.nz, dtype=np.float64)
        self.d_tau_buf = cuda.device_array(2, dtype=np.float64)
        self.d_bad = cuda.to_device(np.array([0], dtype=np.int32))

        self.tpb3 = (8, 8, 8)
        self.bpg3 = blocks_3d((cfg.nx, cfg.ny, cfg.nz), self.tpb3)

        self.tpb2 = (16, 8)
        self.bpg2 = blocks_2d(cfg.nx, cfg.ny, self.tpb2)
        self.bpg_yz = blocks_2d(cfg.ny, cfg.nz, self.tpb2)
        self.bpg_xz = blocks_2d(cfg.nx, cfg.nz, self.tpb2)

        self.time_value = 0.0
        self.nstep = 0
        self._stats_workspace = None
        self._checkpoint_toggle = False

        self.reset_runtime_buffers()

    def reset_runtime_buffers(self):
        """Re-initializes derived primitives, periodic halos, rhow history, and channel forcing."""
        self.d_rhow_prev.copy_to_device(np.zeros((self.cfg.nx, self.cfg.ny, 2), dtype=np.float64))
        self.d_rho_mean.copy_to_device(np.ones(self.cfg.nz, dtype=np.float64))
        self.update_primitive(init_all=True)
        self.sync_halos(0, self.cfg.nz - 1)
        if self.cfg.dynamic_source and not self.cfg.defer_source:
            self.update_source_terms()

    def update_primitive(self, init_all: bool = False):
        c = self.cfg
        primary_variables_kernel[self.bpg3, self.tpb3](
            self.d_q,
            self.d_var,
            self.d_mu,
            self.d_kt,
            c.ggM,
            GAMMA,
            c.mach,
            1 if init_all else 0,
        )

    def sync_halos(self, k_low: int = 0, k_high: int | None = None):
        if k_high is None:
            k_high = self.cfg.nz - 1
        sync_halos_x_kernel[self.bpg_yz, self.tpb2](
            self.d_q, self.d_var, self.d_mu, self.d_kt, k_low, k_high
        )
        sync_halos_y_kernel[self.bpg_xz, self.tpb2](
            self.d_q, self.d_var, self.d_mu, self.d_kt, k_low, k_high
        )

    def sync_wall_halos(self):
        self.sync_halos(0, 0)
        self.sync_halos(self.cfg.nz - 1, self.cfg.nz - 1)

    def update_source_terms(self):
        """Computes dynamic tau_lower, tau_upper, and span-averaged rho_mean(z) on GPU."""
        c = self.cfg
        c.tau_lower, c.tau_upper = self.measure_wall_shear()

        self.d_rho_sum_buf.copy_to_device(np.zeros(c.nz, dtype=np.float64))
        xy_rho_sum_kernel[self.bpg3, self.tpb3](self.d_var, self.d_rho_sum_buf)
        rho_mean_h = self.d_rho_sum_buf.copy_to_host() / float(c.nx * c.ny)
        self.d_rho_mean.copy_to_device(rho_mean_h)

    def convective(self):
        c = self.cfg
        convective_kernel[self.bpg3, self.tpb3](
            self.d_q,
            self.d_var,
            self.d_mu,
            self.d_F,
            self.d_G,
            self.d_H,
            self.dx,
            self.dy,
            self.dxi,
            self.d_jac,
            self.d_inv_jac,
            c.reynolds,
            c.peclet_x,
            c.peclet_y,
            c.peclet_z,
        )

    def diffusive(self):
        c = self.cfg
        diffusive_kernel[self.bpg3, self.tpb3](
            self.d_var,
            self.d_mu,
            self.d_kt,
            self.d_fD,
            self.d_gD,
            self.d_hD,
            1.0 / self.dx,
            1.0 / self.dy,
            1.0 / self.dxi,
            self.d_inv_jac,
            c.inv_re,
            INV_PR,
            c.ggM,
            GAMMA,
            c.wall_order,
            self.d_low2,
            self.d_high2,
            self.d_low3,
            self.d_high3,
        )

    def rk2_substep(self, iteration: int):
        c = self.cfg
        rhs_and_rk2_kernel[self.bpg3, self.tpb3](
            self.d_q,
            self.d_q_stage,
            self.d_var,
            self.d_F,
            self.d_G,
            self.d_H,
            self.d_fD,
            self.d_gD,
            self.d_hD,
            self.d_rho_mean,
            self.dx,
            self.dy,
            self.dxi,
            self.d_inv_jac,
            c.dt,
            c.ggM,
            c.tau_lower,
            c.tau_upper,
            iteration,
        )

    def apply_bc_and_oc(self):
        c = self.cfg
        opposition_control_kernel[self.bpg2, self.tpb2](
            self.d_q,
            self.d_var,
            c.opposition_distance,
            c.enable_oc,
        )
        self.sync_wall_halos()

        wall_pressure_bc_kernel[self.bpg2, self.tpb2](
            self.d_var,
            self.d_mu,
            self.d_rhow_prev,
            1.0 / self.dx,
            1.0 / self.dy,
            c.inv_re,
            1.0 / c.dt,
            c.wall_order,
            self.d_low2,
            self.d_high2,
            self.d_low2_d2,
            self.d_high2_d2,
            self.d_low3,
            self.d_high3,
            self.d_wall_dz,
        )

        wall_state_bc_kernel[self.bpg2, self.tpb2](
            self.d_q,
            self.d_var,
            self.d_mu,
            self.d_kt,
            self.time_value,
            self.d_x1,
            c.ggM,
            GAMMA,
            c.mach,
            c.actuation_wavenumber,
            c.actuation_omega,
            c.actuation_amplitude,
            c.enable_oc,
        )
        self.sync_wall_halos()

    def check_blown(self):
        self.d_bad.copy_to_device(np.array([0], dtype=np.int32))
        finite_check_kernel[self.bpg3, self.tpb3](self.d_var, self.d_bad)
        flag = int(self.d_bad.copy_to_host()[0])
        if flag:
            try:
                self.save_solution("debug/3D001.dat")
            except Exception as err:
                print(f"Warning: failed to write debug/3D001.dat: {err}")
            raise FloatingPointError(
                f"Unphysical solution detected at step {self.nstep}"
            )

    def measure_wall_shear(self) -> tuple[float, float]:
        """Computes instantaneous bottom/top area-averaged wall shear on GPU."""
        c = self.cfg
        self.d_tau_buf.copy_to_device(np.zeros(2, dtype=np.float64))
        wall_shear_kernel[self.bpg2, self.tpb2](
            self.d_var,
            self.d_mu,
            self.d_low2,
            self.d_high2,
            1.0 / self.dx,
            c.inv_re,
            self.d_tau_buf,
        )
        taus = self.d_tau_buf.copy_to_host()
        nxy_int = float((c.nx - 2) * (c.ny - 2)) if c.nx > 2 and c.ny > 2 else float(c.nx * c.ny)
        return float(taus[0] / nxy_int), float(taus[1] / nxy_int)

    def _get_stats_workspace(self):
        if self._stats_workspace is None:
            cdc = _import_diagnostics_module()
            self._stats_workspace = cdc.LiveStatsWorkspace.allocate(
                self.cfg.nx, self.cfg.ny, self.cfg.nz, self.x3_h
            )
        return self._stats_workspace

    def compute_diagnostics(self) -> dict:
        """Computes live Diagnostic_files() and PowSpec() scalars directly from GPU d_var."""
        cdc = _import_diagnostics_module()
        ws = self._get_stats_workspace()
        return cdc.compute_live_step_diagnostics(
            self.d_var,
            self.x2_h,
            self.x3_h,
            self.dy,
            ws,
            tau_lower=self.cfg.tau_lower,
            tau_upper=self.cfg.tau_upper,
        )

    def compute_spatial_stats(
        self,
        mode: str | None = None,
        tau_lower: float | None = None,
        tau_upper: float | None = None,
    ) -> dict:
        """Computes wall-normal spatial/Favre/TKE/Omega/Cf stats directly on GPU."""
        cdc = _import_diagnostics_module()
        ws = self._get_stats_workspace()
        if tau_lower is None or tau_upper is None:
            tl_m, tu_m = self.measure_wall_shear()
            if tau_lower is None:
                tau_lower = tl_m
            if tau_upper is None:
                tau_upper = tu_m
        return cdc.compute_spatial_stats_gpu(
            self.d_var,
            self.d_mu,
            self.x3_h,
            self.d_inv_jac,
            self.d_low2,
            self.d_high2,
            self.dx,
            self.dy,
            self.dxi,
            self.cfg.reynolds,
            ws,
            tau_lower=tau_lower,
            tau_upper=tau_upper,
            mode=mode or self.cfg.stats_mode,
        )

    def emit_live_outputs(self) -> None:
        """Appends live Fortran-parity statistics streams according to diag/stat cadences."""
        c = self.cfg
        if not c.stats_dir or c.stats_mode == "none":
            return
        cdc = _import_diagnostics_module()
        do_diag = c.diag_frequency > 0 and self.nstep % c.diag_frequency == 0
        do_stat = c.statistic_frequency > 0 and self.nstep % c.statistic_frequency == 0

        if do_diag:
            diag = self.compute_diagnostics()
            cdc.append_powspec_record(c.stats_dir, self.time_value, diag["fft"])
            cdc.append_diagnostic_record(c.stats_dir, self.time_value, diag, include_tau=False)

        if do_stat:
            if c.dynamic_source:
                tau_bot, tau_top = c.tau_lower, c.tau_upper
            else:
                tau_bot, tau_top = self.measure_wall_shear()
            stats = self.compute_spatial_stats(
                mode=c.stats_mode, tau_lower=tau_bot, tau_upper=tau_top
            )
            cdc.append_spatial_stats(c.stats_dir, self.time_value, stats, mode=c.stats_mode)

    def save_rolling_checkpoint(self) -> str:
        """Writes alternating 3D.dat/3D_.dat or partitioned domInfo.txt+3Dnnn.dat checkpoints."""
        c = self.cfg
        if not c.checkpoint_dir:
            raise ValueError("checkpoint_dir is not configured")
        os.makedirs(c.checkpoint_dir, exist_ok=True)
        if c.checkpoint_components:
            var_h = self.download_primitives()
            _write_component_checkpoint(
                c.checkpoint_dir,
                self.x1_h,
                self.x2_h,
                self.x3_h,
                var_h,
                self.time_value,
                self.nstep,
            )
            return os.path.join(c.checkpoint_dir, "domInfo.txt")

        name = "3D_.dat" if self._checkpoint_toggle else "3D.dat"
        self._checkpoint_toggle = not self._checkpoint_toggle
        out_path = os.path.join(c.checkpoint_dir, name)
        self.save_solution(out_path)
        return out_path

    def step(self):
        """
        Original main-loop sequence:

          nstep = nstep + 1
          time  = time + dt
          do iter=1,2
             calc_convective_terms
             calc_diffusive_terms_2D/3D
             transfer_FHD
             calc_FHD_and_srcV
             transfer_2p_solV
             find_primary_variable
             transfer_1p_var
             OC
             bc_update_2D/3D
             transfer_xyline
          enddo
        """
        self.nstep += 1
        self.time_value += self.cfg.dt

        for iteration in (1, 2):
            self.convective()
            self.diffusive()
            self.rk2_substep(iteration)
            self.update_primitive(init_all=False)
            self.sync_halos(1, self.cfg.nz - 2)
            self.apply_bc_and_oc()

        if (
            self.cfg.dynamic_source
            and self.cfg.statistic_frequency > 0
            and self.nstep % self.cfg.statistic_frequency == 0
        ):
            self.update_source_terms()

        self.emit_live_outputs()
        if (
            self.cfg.checkpoint_dir
            and self.cfg.output_frequency > 0
            and self.nstep % self.cfg.output_frequency == 0
        ):
            self.save_rolling_checkpoint()

    def download_solution(self):
        return self.d_q.copy_to_host()

    def download_primitives(self):
        return self.d_var.copy_to_host()

    def save_solution(self, filename: str):
        """Exports the current 3D flow state to Tecplot format matching mod_in_out.f90."""
        var_h = self.download_primitives()
        print(f"Writing solution checkpoint to '{filename}'...")
        t0 = time.time()
        _write_tecplot_subzone(
            filename,
            self.x1_h,
            self.x2_h,
            self.x3_h,
            var_h,
            self.time_value,
            self.nstep,
        )
        print(f"Saved solution in {time.time() - t0:.2f} s.")

    def run(self):
        start = time.perf_counter()
        steps_done = 0

        for _ in range(self.cfg.max_steps):
            self.step()
            steps_done += 1

            if self.nstep % 10 == 0:
                self.check_blown()

            if self.nstep % self.cfg.statistic_frequency == 0 or steps_done == self.cfg.max_steps:
                cuda.synchronize()
                rho_mean = float(self.d_var[:, :, :, 0].copy_to_host().mean())
                print(
                    f"step={self.nstep:8d}  time={self.time_value:.6e}  "
                    f"mean(rho)={rho_mean:.6e}  tau=({self.cfg.tau_lower:.6e}, {self.cfg.tau_upper:.6e})"
                )

        cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(
            f"CUDA solver finished: executed {steps_done} steps (step index {self.nstep}, t={self.time_value:.6e}) "
            f"in {elapsed:.3f} s ({elapsed/max(steps_done,1):.6e} s/step)"
        )


def add_common_solver_args(p: argparse.ArgumentParser) -> None:
    """Adds solver physics/discretization override flags shared across entry points."""
    p.add_argument("--param", default="param.dat", help="path to param.dat (default: param.dat)")
    p.add_argument("--mesh", choices=("auto", "cosine", "tanh"), default=None, help="wall-normal grid stretching")
    p.add_argument("--wall-order", type=int, choices=(2, 3), default=None, help="2=codes/ 2nd-order, 3=codes1/ 3rd-order")
    p.add_argument("--oc", type=int, choices=(0, 1), default=None, help="1=enable opposition control, 0=disable")
    p.add_argument("--amp-v", type=float, default=None, help="spanwise wall oscillation amplitude")
    p.add_argument("--kappa", type=float, default=None, help="actuation streamwise wavenumber")
    p.add_argument("--omega", type=float, default=None, help="actuation angular frequency")
    p.add_argument("--dynamic-source", dest="dynamic_source", action="store_true", default=None, help="enable GPU wall shear/rho_mean source updates")
    p.add_argument("--no-dynamic-source", dest="dynamic_source", action="store_false", help="disable dynamic GPU source updates (use fixed tau)")
    p.add_argument("--stats-dir", default=None, help="directory to stream live Fortran-parity statistics (.dat)")
    p.add_argument("--stats-mode", choices=("none", "minimal", "standard", "full"), default=None, help="live statistics tier (default: standard)")
    p.add_argument("--diag-freq", type=int, default=None, help="Diagnostic_files/PowSpec step cadence (default: 10)")
    p.add_argument("--checkpoint-dir", default=None, help="directory for rolling 3D.dat/3D_.dat or partitioned checkpoints")
    p.add_argument("--output-freq", type=int, default=None, help="rolling checkpoint step cadence (default: 1000)")
    p.add_argument("--checkpoint-components", action="store_true", default=None, help="write rolling checkpoints as domInfo.txt + 3Dnnn.dat component files")


def parse_args():
    p = argparse.ArgumentParser(
        description="CUDA Python conversion of the supplied MPI CFD solver"
    )

    add_common_solver_args(p)
    p.add_argument("--input", default=None, help="combined input file or domInfo.txt for resOp=2")
    p.add_argument("--output", default=None, help="output checkpoint file path")
    p.add_argument("--demo", action="store_true", help="run a synthetic test case")

    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--nz", type=int, default=None)

    p.add_argument("--re", type=float, default=None)
    p.add_argument("--mach", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)

    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--istat", type=int, default=None)
    p.add_argument("--res-op", type=int, choices=(1, 2), default=None, help="1=combined 3D.dat, 2=component domInfo.txt")
    p.add_argument("--debug", type=int, choices=(0, 1), default=None, help="1=write Mesh.dat")

    p.add_argument("--tau-lower", type=float, default=None)
    p.add_argument("--tau-upper", type=float, default=None)
    p.add_argument("--defer-source", action="store_true", help="defer initial tau update until step %% istat == 0 (matches cold-start Fortran)")

    return p.parse_args()


def _apply_overrides_dict(cfg: SolverConfig, overrides: dict | None) -> None:
    if not overrides:
        return
    key_map = {
        "nx": "nx",
        "ny": "ny",
        "nz": "nz",
        "re": "reynolds",
        "reynolds": "reynolds",
        "mach": "mach",
        "dt": "dt",
        "steps": "max_steps",
        "max_steps": "max_steps",
        "istat": "statistic_frequency",
        "statistic_frequency": "statistic_frequency",
        "diag_freq": "diag_frequency",
        "diag_frequency": "diag_frequency",
        "output_freq": "output_frequency",
        "output_frequency": "output_frequency",
        "res_op": "restart_option",
        "restart_option": "restart_option",
        "debug": "debug_op",
        "debug_op": "debug_op",
        "mesh": "mesh_type",
        "mesh_type": "mesh_type",
        "wall_order": "wall_order",
        "oc": "enable_oc",
        "enable_oc": "enable_oc",
        "amp_v": "actuation_amplitude",
        "actuation_amplitude": "actuation_amplitude",
        "kappa": "actuation_wavenumber",
        "omega": "actuation_omega",
        "tau_lower": "tau_lower",
        "tau_upper": "tau_upper",
        "stats_dir": "stats_dir",
        "stats_mode": "stats_mode",
        "checkpoint_dir": "checkpoint_dir",
    }
    for key, attr in key_map.items():
        val = overrides.get(key)
        if val is not None:
            setattr(cfg, attr, val)

    dyn = overrides.get("dynamic_source")
    if dyn is not None:
        cfg.dynamic_source = bool(dyn)
    elif overrides.get("tau_lower") is not None or overrides.get("tau_upper") is not None:
        cfg.dynamic_source = False

    if overrides.get("defer_source"):
        cfg.defer_source = True

    ckpt_comp = overrides.get("checkpoint_components")
    if ckpt_comp is not None:
        cfg.checkpoint_components = bool(ckpt_comp)


def configure_and_load_state(
    param_path: str = "param.dat",
    input_file: str | None = None,
    demo: bool = False,
    overrides: dict | None = None,
    fallback_to_demo: bool = False,
    default_nx: int = 192,
    default_ny: int = 130,
    default_nz: int = 160,
    default_re: float = 1000.0,
    default_mach: float = 0.2,
    default_dt: float = 1.0e-4,
):
    """Loads param.dat, applies explicit overrides, and reads or builds initial q0."""
    cfg = SolverConfig(
        nx=default_nx,
        ny=default_ny,
        nz=default_nz,
        reynolds=default_re,
        mach=default_mach,
        dt=default_dt,
    )
    param_input = load_param_dat(param_path, cfg) if param_path else None
    _apply_overrides_dict(cfg, overrides)
    if cfg.statistic_frequency <= 0:
        cfg.statistic_frequency = max(cfg.max_steps, 1)

    if demo:
        return cfg, demo_initial_condition(cfg), 0.0, 0, "demo"

    if (input_file and os.path.basename(input_file).lower() == "dominfo.txt") or (
        cfg.restart_option == 2 and not input_file
    ):
        dom_path = input_file or "output/domInfo.txt"
        q0, time_init, nstep_init = read_component_inputs(dom_path, cfg)
        return cfg, q0, time_init, nstep_init, dom_path

    candidate = input_file or param_input
    if not candidate:
        for cand in ("input/3D.dat", "3D.dat", "../input/3D.dat"):
            if os.path.exists(cand):
                candidate = cand
                break

    if candidate:
        try:
            q0, time_init, nstep_init = read_combined_input(candidate, cfg)
            return cfg, q0, time_init, nstep_init, candidate
        except FileNotFoundError:
            if not fallback_to_demo or input_file:
                raise

    if fallback_to_demo:
        return cfg, demo_initial_condition(cfg), 0.0, 0, "demo"
    raise SystemExit("No input file found. Provide --input INPUT_FILE or use --demo.")


def main():
    args = parse_args()
    cuda_check()

    overrides = {
        k: getattr(args, k)
        for k in (
            "nx",
            "ny",
            "nz",
            "re",
            "mach",
            "dt",
            "steps",
            "istat",
            "diag_freq",
            "output_freq",
            "res_op",
            "debug",
            "mesh",
            "wall_order",
            "oc",
            "amp_v",
            "kappa",
            "omega",
            "tau_lower",
            "tau_upper",
            "dynamic_source",
            "defer_source",
            "stats_dir",
            "stats_mode",
            "checkpoint_dir",
            "checkpoint_components",
        )
    }
    cfg, q0, time_init, nstep_init, source_label = configure_and_load_state(
        param_path=args.param,
        input_file=args.input,
        demo=args.demo,
        overrides=overrides,
        fallback_to_demo=False,
    )
    if source_label == "demo":
        print("Using synthetic demo initial condition.")
    else:
        print(f"Loaded CFD input: {source_label} (t={time_init:.4f}, step={nstep_init})")

    solver = CUDASolver(cfg, q0)
    solver.time_value = time_init
    solver.nstep = nstep_init
    solver.run()

    if args.output:
        solver.save_solution(args.output)


if __name__ == "__main__":
    main()
