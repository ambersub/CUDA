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
from numba import cuda, float64, int32


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

    # Original source term requires wall shear statistics. Set these when
    # the full statistics module is ported.
    tau_lower: float = 0.0
    tau_upper: float = 0.0

    # Opposition-control parameters from mod_control/mod_var_dec.
    opposition_distance: int = 33
    actuation_wavenumber: float = 1.0
    actuation_omega: float = 0.5
    actuation_amplitude: float = 0.0

    # 1 = enable opposition control, 0 = no actuation.
    enable_oc: int = 1

    max_steps: int = 100
    statistic_frequency: int = 100
    output_frequency: int = 1000

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


def make_mesh(cfg: SolverConfig):
    """Equivalent of mod_comdata:setupMesh()."""
    x1 = np.linspace(0.0, 4.0 * math.pi, cfg.nx).astype(np.float64)
    x2 = np.linspace(0.0, 4.0 * math.pi / 3.0, cfg.ny).astype(np.float64)
    xi = np.linspace(0.0, 2.0, cfg.nz).astype(np.float64)

    dx = x1[1] - x1[0]
    dy = x2[1] - x2[0]
    dxi = xi[1] - xi[0]

    x3 = 1.0 + np.tanh(0.5 * EP * (xi - 1.0)) / np.tanh(0.5 * EP)
    x3[0] = 0.0

    jac = (
        0.5 * EP
        * (1.0 - np.tanh(0.5 * EP * (xi - 1.0)) ** 2)
        / np.tanh(0.5 * EP)
    )
    inv_jac = np.zeros_like(jac)
    inv_jac[1:] = 1.0 / jac[1:]

    # One-sided third-order derivative coefficients in the original solver.
    # They are generated using the same expressions in Coefficients().
    def coeff_low_3(z):
        d21 = z[1] - z[0]
        d31 = z[2] - z[0]
        d32 = z[2] - z[1]
        d41 = z[3] - z[0]
        d42 = z[3] - z[1]
        d43 = z[3] - z[2]

        c0 = (d31 * d21 + d31 * d41 + d21 * d41) / (
            (-d21) * (-d31) * (-d41)
        )
        c1 = (d41 * d31) / (d21 * (-d32) * (-d42))
        c2 = (d41 * d21) / (d31 * d32 * (-d43))
        c3 = (d31 * d21) / (d41 * d42 * d43)
        return np.array([c0, c1, c2, c3], dtype=np.float64)

    def coeff_high_3(z):
        n = len(z)
        d12 = z[n - 1] - z[n - 2]
        d13 = z[n - 1] - z[n - 3]
        d23 = z[n - 2] - z[n - 3]
        d14 = z[n - 1] - z[n - 4]
        d24 = z[n - 2] - z[n - 4]
        d34 = z[n - 3] - z[n - 4]

        c0 = (d13 * d12 + d13 * d14 + d12 * d14) / (d12 * d13 * d14)
        c1 = (d14 * d13) / ((-d12) * d23 * d24)
        c2 = (d14 * d12) / ((-d13) * (-d23) * d34)
        c3 = (d13 * d12) / ((-d14) * (-d24) * (-d34))
        return np.array([c0, c1, c2, c3], dtype=np.float64)

    low3 = coeff_low_3(x3)
    high3 = coeff_high_3(x3)

    return x1, x2, x3, jac, inv_jac, dx, dy, dxi, low3, high3


def load_param_dat(path: str, cfg: SolverConfig) -> str | None:
    """Parses runtime parameters from param.dat if available."""
    if not os.path.exists(path):
        return None
    input_path = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = [s.strip() for s in line.split(":", 1)]
            if "Reynolds Number" in key:
                cfg.reynolds = float(val)
            elif "Mach Number" in key:
                cfg.mach = float(val)
            elif "Timestep" in key:
                cfg.dt = float(val)
            elif "Frequency of Calculating Spatial Statistics" in key:
                cfg.statistic_frequency = int(val)
            elif "Peclet in x direction" in key:
                cfg.peclet_x = float(val)
            elif "Peclet in y direction" in key:
                cfg.peclet_y = float(val)
            elif "Peclet in z direction" in key:
                cfg.peclet_z = float(val)
            elif "Relative path of initial solution" in key:
                input_path = val
    return input_path


def read_combined_input(path: str, cfg: SolverConfig):
    """
    Reads Tecplot formatted combined CFD input file (e.g., input/3D.dat).
    Automatically parses header for grid dimensions (nx, ny, nz) and state variables.
    """
    import pandas as pd

    resolved_path = path
    if not os.path.exists(resolved_path):
        alt = os.path.basename(path)
        for cand in [path, os.path.join("input", alt), alt]:
            if os.path.exists(cand):
                resolved_path = cand
                break

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Input file '{path}' not found.")

    time_val = 0.0
    nstep_val = 0

    with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
        l1 = f.readline()
        l2 = f.readline()
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

        m_k = re.search(r'k\s*=\s*(\d+)', l3, re.IGNORECASE)
        m_j = re.search(r'j\s*=\s*(\d+)', l3, re.IGNORECASE)
        m_i = re.search(r'i\s*=\s*(\d+)', l3, re.IGNORECASE)
        if m_k and m_j and m_i:
            cfg.nz = int(m_k.group(1))
            cfg.ny = int(m_j.group(1))
            cfg.nx = int(m_i.group(1))

    expected = cfg.nx * cfg.ny * cfg.nz
    print(f"Loading input '{resolved_path}' (Grid: {cfg.nx}x{cfg.ny}x{cfg.nz}, Expected points: {expected})...")

    t0 = time.time()
    df = pd.read_csv(
        resolved_path,
        skiprows=3,
        sep=r"\s+",
        header=None,
        comment="T",
        on_bad_lines="skip",
        engine="c",
    )

    df = df.dropna().iloc[:expected]
    data = df.to_numpy(dtype=np.float64)
    if data.shape[0] < expected:
        raise ValueError(
            f"Input file contains {data.shape[0]} valid rows; expected {expected} for grid {cfg.nx}x{cfg.ny}x{cfg.nz}."
        )

    print(f"Input loaded in {time.time() - t0:.2f} s.")

    data = data.reshape(cfg.nz, cfg.ny, cfg.nx, -1)
    p = data[..., -10:]

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
def primary_variables_kernel(q, var, mu, kt, ggM, gamma, mach):
    """
    Equivalent of find_primary_variable().

    q, var layout:
        [i, j, k, component]

    CUDA replacement for MPI decomposition:
        every GPU thread owns one (i,j,k) cell.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]

    if i < nx and j < ny and k < nz:
        rho = q[i, j, k, 0]
        rho = max(rho, 1.0e-14)

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
    inv_jac,
    inv_re,
    px_limit,
    py_limit,
    pz_limit,
):
    """
    GPU version of calc_convective_terms().

    The four sign combinations of the original biased/upwind form are
    evaluated per cell. x and y are periodic exactly at the index level,
    replacing MPI halo exchange on periodic directions.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]

    if i >= nx or j >= ny or k >= nz:
        return

    im2 = (i - 2) % nx
    im1 = (i - 1) % nx
    ip1 = (i + 1) % nx
    ip2 = (i + 2) % nx

    jm2 = (j - 2) % ny
    jm1 = (j - 1) % ny
    jp1 = (j + 1) % ny
    jp2 = (j + 2) % ny

    km1 = max(k - 1, 0)
    kp1 = min(k + 1, nz - 1)
    km2 = max(k - 2, 0)
    kp2 = min(k + 2, nz - 1)

    # ------------------------- x direction ------------------------------
    pe = dx * abs(q[i, j, k, 1]) / max(mu[i, j, k], 1.0e-14)
    use_upwind = (pe > px_limit) or (i == 0) or (i == nx - 1)

    for c in range(NQ):
        if use_upwind:
            a = 0.5 * (var[i, j, k, 1] + var[ip1, j, k, 1])
            b = 0.5 * (var[i, j, k, 1] + var[im1, j, k, 1])

            if a >= 0.0 and b >= 0.0:
                val = (
                    a * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[ip1, j, k, c]
                        - q[im1, j, k, c]
                    )
                    - b * (
                        6.0 * q[im1, j, k, c]
                        + 3.0 * q[i, j, k, c]
                        - q[im2, j, k, c]
                    )
                ) / 8.0
            elif a < 0.0 and b < 0.0:
                val = (
                    a * (
                        6.0 * q[ip1, j, k, c]
                        + 3.0 * q[i, j, k, c]
                        - q[ip2, j, k, c]
                    )
                    - b * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[im1, j, k, c]
                        - q[ip1, j, k, c]
                    )
                ) / 8.0
            elif a < 0.0 and b >= 0.0:
                val = (
                    a * (
                        6.0 * q[ip1, j, k, c]
                        + 3.0 * q[i, j, k, c]
                        - q[ip2, j, k, c]
                    )
                    - b * (
                        6.0 * q[im1, j, k, c]
                        + 3.0 * q[i, j, k, c]
                        - q[im2, j, k, c]
                    )
                ) / 8.0
            else:
                val = (
                    a * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[ip1, j, k, c]
                        - q[im1, j, k, c]
                    )
                    - b * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[im1, j, k, c]
                        - q[ip1, j, k, c]
                    )
                ) / 8.0
            F[i, j, k, c] = val / dx
        else:
            F[i, j, k, c] = (
                -var[ip2, j, k, 1] * q[ip2, j, k, c]
                + 8.0 * var[ip1, j, k, 1] * q[ip1, j, k, c]
                - 8.0 * var[im1, j, k, 1] * q[im1, j, k, c]
                + var[im2, j, k, 1] * q[im2, j, k, c]
            ) / (12.0 * dx)

    # ------------------------- y direction ------------------------------
    pe = dy * abs(q[i, j, k, 2]) / max(mu[i, j, k], 1.0e-14)
    use_upwind = (pe > py_limit) or (j == 0) or (j == ny - 1)

    for c in range(NQ):
        if use_upwind:
            a = 0.5 * (var[i, j, k, 2] + var[i, jp1, k, 2])
            b = 0.5 * (var[i, j, k, 2] + var[i, jm1, k, 2])

            if a >= 0.0 and b >= 0.0:
                val = (
                    a * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[i, jp1, k, c]
                        - q[i, jm1, k, c]
                    )
                    - b * (
                        6.0 * q[i, jm1, k, c]
                        + 3.0 * q[i, j, k, c]
                        - q[i, jm2, k, c]
                    )
                ) / 8.0
            elif a < 0.0 and b < 0.0:
                val = (
                    a * (
                        6.0 * q[i, jp1, k, c]
                        + 3.0 * q[i, j, k, c]
                        - q[i, jp2, k, c]
                    )
                    - b * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[i, jm1, k, c]
                        - q[i, jp1, k, c]
                    )
                ) / 8.0
            elif a < 0.0 and b >= 0.0:
                val = (
                    a * (
                        6.0 * q[i, jp1, k, c]
                        + 3.0 * q[i, j, k, c]
                        - q[i, jp2, k, c]
                    )
                    - b * (
                        6.0 * q[i, jm1, k, c]
                        + 3.0 * q[i, j, k, c]
                        - q[i, jm2, k, c]
                    )
                ) / 8.0
            else:
                val = (
                    a * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[i, jp1, k, c]
                        - q[i, jm1, k, c]
                    )
                    - b * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[i, jm1, k, c]
                        - q[i, jp1, k, c]
                    )
                ) / 8.0
            G[i, j, k, c] = val / dy
        else:
            G[i, j, k, c] = (
                -var[i, jp2, k, 2] * q[i, jp2, k, c]
                + 8.0 * var[i, jp1, k, 2] * q[i, jp1, k, c]
                - 8.0 * var[i, jm1, k, 2] * q[i, jm1, k, c]
                + var[i, jm2, k, 2] * q[i, jm2, k, c]
            ) / (12.0 * dy)

    # ------------------------- z direction ------------------------------
    pe = dxi * abs(q[i, j, k, 3]) * inv_jac[k] / max(mu[i, j, k], 1.0e-14)

    for c in range(NQ):
        if k < 2 or k > nz - 3:
            # Same active choice as H_central_2D() in the supplied source.
            H[i, j, k, c] = (
                var[i, j, kp1, 3] * q[i, j, kp1, c]
                - var[i, j, km1, 3] * q[i, j, km1, c]
            ) * 0.5 * inv_jac[k] / dxi
        elif pe > pz_limit:
            a = 0.5 * (var[i, j, k, 3] + var[i, j, kp1, 3])
            b = 0.5 * (var[i, j, k, 3] + var[i, j, km1, 3])

            if a >= 0.0 and b >= 0.0:
                val = (
                    a * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[i, j, kp1, c]
                        - q[i, j, km1, c]
                    )
                    - b * (
                        6.0 * q[i, j, km1, c]
                        + 3.0 * q[i, j, k, c]
                        - q[i, j, km2, c]
                    )
                ) / 8.0
            elif a < 0.0 and b < 0.0:
                val = (
                    a * (
                        6.0 * q[i, j, kp1, c]
                        + 3.0 * q[i, j, k, c]
                        - q[i, j, kp2, c]
                    )
                    - b * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[i, j, km1, c]
                        - q[i, j, kp1, c]
                    )
                ) / 8.0
            elif a < 0.0 and b >= 0.0:
                val = (
                    a * (
                        6.0 * q[i, j, kp1, c]
                        + 3.0 * q[i, j, k, c]
                        - q[i, j, kp2, c]
                    )
                    - b * (
                        6.0 * q[i, j, km1, c]
                        + 3.0 * q[i, j, k, c]
                        - q[i, j, km2, c]
                    )
                ) / 8.0
            else:
                val = (
                    a * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[i, j, kp1, c]
                        - q[i, j, km1, c]
                    )
                    - b * (
                        6.0 * q[i, j, k, c]
                        + 3.0 * q[i, j, km1, c]
                        - q[i, j, kp1, c]
                    )
                ) / 8.0
            H[i, j, k, c] = val * inv_jac[k] / dxi
        else:
            H[i, j, k, c] = (
                -var[i, j, kp2, 3] * q[i, j, kp2, c]
                + 8.0 * var[i, j, kp1, 3] * q[i, j, kp1, c]
                - 8.0 * var[i, j, km1, 3] * q[i, j, km1, c]
                + var[i, j, km2, 3] * q[i, j, km2, c]
            ) * inv_jac[k] / (12.0 * dxi)


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
    low3,
    high3,
):
    """
    GPU version of calc_diffusive_terms_3D().

    Central differences in x/y and third-order one-sided z derivatives at
    the two physical walls, matching the supplied Coefficients() pattern.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]

    if i >= nx or j >= ny or k >= nz:
        return

    im1 = (i - 1) % nx
    ip1 = (i + 1) % nx
    jm1 = (j - 1) % ny
    jp1 = (j + 1) % ny

    u = var[i, j, k, 1]
    vv = var[i, j, k, 2]
    w = var[i, j, k, 3]
    temp = var[i, j, k, 5]
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

    if k == 0:
        dudz = (
            low3[0] * var[i, j, 0, 1]
            + low3[1] * var[i, j, 1, 1]
            + low3[2] * var[i, j, 2, 1]
            + low3[3] * var[i, j, 3, 1]
        )
        dvdz = (
            low3[0] * var[i, j, 0, 2]
            + low3[1] * var[i, j, 1, 2]
            + low3[2] * var[i, j, 2, 2]
            + low3[3] * var[i, j, 3, 2]
        )
        dwdz = (
            low3[0] * var[i, j, 0, 3]
            + low3[1] * var[i, j, 1, 3]
            + low3[2] * var[i, j, 2, 3]
            + low3[3] * var[i, j, 3, 3]
        )
        dTdz = (
            low3[0] * var[i, j, 0, 5]
            + low3[1] * var[i, j, 1, 5]
            + low3[2] * var[i, j, 2, 5]
            + low3[3] * var[i, j, 3, 5]
        )
    elif k == nz - 1:
        dudz = (
            high3[0] * var[i, j, nz - 1, 1]
            + high3[1] * var[i, j, nz - 2, 1]
            + high3[2] * var[i, j, nz - 3, 1]
            + high3[3] * var[i, j, nz - 4, 1]
        )
        dvdz = (
            high3[0] * var[i, j, nz - 1, 2]
            + high3[1] * var[i, j, nz - 2, 2]
            + high3[2] * var[i, j, nz - 3, 2]
            + high3[3] * var[i, j, nz - 4, 2]
        )
        dwdz = (
            high3[0] * var[i, j, nz - 1, 3]
            + high3[1] * var[i, j, nz - 2, 3]
            + high3[2] * var[i, j, nz - 3, 3]
            + high3[3] * var[i, j, nz - 4, 3]
        )
        dTdz = (
            high3[0] * var[i, j, nz - 1, 5]
            + high3[1] * var[i, j, nz - 2, 5]
            + high3[2] * var[i, j, nz - 3, 5]
            + high3[3] * var[i, j, nz - 4, 5]
        )
    else:
        dudz = 0.5 * (
            var[i, j, k + 1, 1] - var[i, j, k - 1, 1]
        ) * inv_dxi * inv_jac[k]
        dvdz = 0.5 * (
            var[i, j, k + 1, 2] - var[i, j, k - 1, 2]
        ) * inv_dxi * inv_jac[k]
        dwdz = 0.5 * (
            var[i, j, k + 1, 3] - var[i, j, k - 1, 3]
        ) * inv_dxi * inv_jac[k]
        dTdz = 0.5 * (
            var[i, j, k + 1, 5] - var[i, j, k - 1, 5]
        ) * inv_dxi * inv_jac[k]

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
        -muv * vv * (dwdy + dvdz)
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
    src,
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

    Original:
        R = src - (F1+F2+G1+G2+H1+H2)

    Here:
        F1/G1/H1 = convective fluxes
        F2/G2/H2 = divergence of diffusive fluxes
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = q.shape[0], q.shape[1], q.shape[2]

    if i >= nx or j >= ny or k >= nz:
        return

    im1 = (i - 1) % nx
    ip1 = (i + 1) % nx
    jm1 = (j - 1) % ny
    jp1 = (j + 1) % ny
    km1 = max(k - 1, 0)
    kp1 = min(k + 1, nz - 1)

    # srcV from the supplied code. stat1Spc(k,1) is replaced here by 1.0;
    # the wall-statistics kernel can feed the actual value later.
    body = 0.5 * (tau_lower - tau_upper)

    for c in range(NQ):
        div_diff = (
            0.5 * (fD[ip1, j, k, c] - fD[im1, j, k, c]) / dx
            + 0.5 * (gD[i, jp1, k, c] - gD[i, jm1, k, c]) / dy
            + 0.5
            * (
                hD[i, j, kp1, c] - hD[i, j, km1, c]
            )
            / dxi
            * inv_jac[k]
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
def opposition_and_wall_bc_kernel(
    q,
    var,
    time_value,
    x1,
    ggM,
    gamma,
    mach,
    opposition_distance,
    kappa,
    omega,
    amp_v,
    enable_oc,
):
    """
    GPU version of the active part of OC() + the wall velocity part of
    bc_update_3D().

    Bottom/top walls:
      u = 0
      v = A sin(k*x - omega*t) when control is enabled
      T = 1
      pressure relation -> rho
      conserved variables rebuilt from primitive variables
    """
    i, j = cuda.grid(2)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]

    if i >= nx or j >= ny:
        return

    if enable_oc:
        kd = opposition_distance
        if kd < 1:
            kd = 1
        if kd > nz - 1:
            kd = nz - 1

        # OC acts on the wall-normal velocity component (w in the CFD state).
        # The uploaded source uses y_d=33.
        src_b = max(0, kd)
        src_t = min(nz - 1, nz - 1 - kd)

        var[i, j, 0, 3] = -q[i, j, src_b, 3] / max(var[i, j, 0, 0], 1.0e-14)
        var[i, j, nz - 1, 3] = -q[i, j, src_t, 3] / max(
            var[i, j, nz - 1, 0], 1.0e-14
        )

    act = amp_v * math.sin(kappa * x1[i] - omega * time_value) if enable_oc else 0.0

    for k in (0, nz - 1):
        var[i, j, k, 1] = 0.0
        var[i, j, k, 2] = act
        var[i, j, k, 3] = 0.0
        var[i, j, k, 5] = 1.0

        # Pressure is left at its current value; rho comes from the EOS relation.
        var[i, j, k, 0] = (
            gamma * mach * mach * var[i, j, k, 6] / var[i, j, k, 5]
        )
        var[i, j, k, 4] = (
            var[i, j, k, 5]
            + 0.5 * ggM
            * (
                var[i, j, k, 1] ** 2
                + var[i, j, k, 2] ** 2
                + var[i, j, k, 3] ** 2
            )
        )

        rho = var[i, j, k, 0]
        q[i, j, k, 0] = rho
        q[i, j, k, 1] = rho * var[i, j, k, 1]
        q[i, j, k, 2] = rho * var[i, j, k, 2]
        q[i, j, k, 3] = rho * var[i, j, k, 3]
        q[i, j, k, 4] = rho * var[i, j, k, 4]


@cuda.jit
def finite_check_kernel(var, flag):
    """Equivalent of check_blown()."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
    if i < nx and j < ny and k < nz:
        if var[i, j, k, 0] <= 0.0 or var[i, j, k, 5] <= 0.0:
            cuda.atomic.max(flag, 0, 1)


@cuda.jit
def sum_scalar_kernel(a, out):
    """Small GPU reduction building block replacing MPI_Reduce for a scalar."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = a.shape
    if i < nx and j < ny and k < nz:
        cuda.atomic.add(out, 0, a[i, j, k])


def blocks_3d(shape, threads=(8, 8, 8)):
    return (
        (shape[0] + threads[0] - 1) // threads[0],
        (shape[1] + threads[1] - 1) // threads[1],
        (shape[2] + threads[2] - 1) // threads[2],
    )


class CUDASolver:
    """Single-GPU replacement for the MPI solver."""

    def __init__(self, cfg: SolverConfig, q_host: np.ndarray):
        if q_host.shape != (cfg.nx, cfg.ny, cfg.nz, NQ):
            raise ValueError(f"q_host shape must be {(cfg.nx, cfg.ny, cfg.nz, NQ)}")

        self.cfg = cfg
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

        self.x1_h = x1
        self.x2_h = x2
        self.x3_h = x3

        self.dx = dx
        self.dy = dy
        self.dxi = dxi

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
        self.d_inv_jac = cuda.to_device(inv_jac)
        self.d_low3 = cuda.to_device(low3)
        self.d_high3 = cuda.to_device(high3)

        self.d_bad = cuda.to_device(np.array([0], dtype=np.int32))

        self.tpb3 = (8, 8, 8)
        self.bpg3 = blocks_3d((cfg.nx, cfg.ny, cfg.nz), self.tpb3)

        self.tpb2 = (32, 8)
        self.bpg2 = (
            (cfg.nx + self.tpb2[0] - 1) // self.tpb2[0],
            (cfg.ny + self.tpb2[1] - 1) // self.tpb2[1],
        )

        self.time_value = 0.0
        self.nstep = 0

        self.update_primitive()

    def update_primitive(self):
        c = self.cfg
        primary_variables_kernel[self.bpg3, self.tpb3](
            self.d_q,
            self.d_var,
            self.d_mu,
            self.d_kt,
            c.ggM,
            GAMMA,
            c.mach,
        )

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
            self.d_inv_jac,
            c.inv_re,
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
            self.d_F,  # source array is unused; retained for API symmetry
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
        opposition_and_wall_bc_kernel[self.bpg2, self.tpb2](
            self.d_q,
            self.d_var,
            self.time_value,
            self.d_x1,
            c.ggM,
            GAMMA,
            c.mach,
            c.opposition_distance,
            c.actuation_wavenumber,
            c.actuation_omega,
            c.actuation_amplitude,
            c.enable_oc,
        )

    def check_blown(self):
        self.d_bad.copy_to_device(np.array([0], dtype=np.int32))
        finite_check_kernel[self.bpg3, self.tpb3](self.d_var, self.d_bad)
        flag = int(self.d_bad.copy_to_host()[0])
        if flag:
            raise FloatingPointError(
                f"Unphysical solution detected at step {self.nstep}"
            )

    def step(self):
        """
        Original main-loop sequence:

          do iter=1,2
             calc_convective_terms
             calc_diffusive_terms_3D
             transfer_FHD
             calc_FHD_and_srcV
             transfer_2p_solV
             find_primary_variable
             transfer_1p_var
             OC
             bc_update_3D
             transfer_xyline
          enddo
        """
        for iteration in (1, 2):
            self.convective()
            self.diffusive()
            self.rk2_substep(iteration)
            self.update_primitive()
            self.apply_bc_and_oc()

        self.nstep += 1
        self.time_value += self.cfg.dt

    def download_solution(self):
        return self.d_q.copy_to_host()

    def download_primitives(self):
        return self.d_var.copy_to_host()

    def save_solution(self, filename: str):
        """Exports the current 3D flow state to Tecplot format matching mod_in_out.f90."""
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        var_h = self.download_primitives()

        cfg = self.cfg
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz

        print(f"Writing solution checkpoint to '{filename}'...")
        t0 = time.time()

        # Reorder into (nz, ny, nx, 10) matching Fortran output layout (k -> j -> i)
        out_arr = np.empty((nz, ny, nx, 10), dtype=np.float64)
        for k in range(nz):
            out_arr[k, :, :, 0] = self.x1_h[None, :]
            out_arr[k, :, :, 1] = self.x2_h[:, None]
            out_arr[k, :, :, 2] = self.x3_h[k]
            out_arr[k, :, :, 3] = var_h[:, :, k, 0].T
            out_arr[k, :, :, 4] = var_h[:, :, k, 1].T
            out_arr[k, :, :, 5] = var_h[:, :, k, 2].T
            out_arr[k, :, :, 6] = var_h[:, :, k, 3].T
            out_arr[k, :, :, 7] = var_h[:, :, k, 4].T
            out_arr[k, :, :, 8] = var_h[:, :, k, 5].T
            out_arr[k, :, :, 9] = var_h[:, :, k, 6].T

        flat_data = out_arr.reshape(-1, 10)

        header = (
            f'TITLE ="{self.time_value:20.10f} {self.nstep:9d}"\n'
            f'Variables = "x","y","z","Rho","u","v","w","e","T","p"\n'
            f'ZONE k={nz},j={ny},i={nx},DATAPACKING=POINT'
        )
        fmt = " %10.6f %10.6f %10.6f " + " ".join(["%22.15e"] * 7)
        np.savetxt(filename, flat_data, fmt=fmt, header=header, comments="")
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
                    f"mean(rho)={rho_mean:.6e}"
                )

        cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(
            f"CUDA solver finished: executed {steps_done} steps (step index {self.nstep}, t={self.time_value:.6e}) "
            f"in {elapsed:.3f} s ({elapsed/max(steps_done,1):.6e} s/step)"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="CUDA Python conversion of the supplied MPI CFD solver"
    )

    p.add_argument("--input", default=None, help="combined input file")
    p.add_argument("--output", default=None, help="output checkpoint file path")
    p.add_argument("--demo", action="store_true", help="run a synthetic test case")

    p.add_argument("--nx", type=int, default=192)
    p.add_argument("--ny", type=int, default=130)
    p.add_argument("--nz", type=int, default=160)

    p.add_argument("--re", type=float, default=1000.0)
    p.add_argument("--mach", type=float, default=0.2)
    p.add_argument("--dt", type=float, default=1.0e-4)

    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--istat", type=int, default=100)

    p.add_argument("--tau-lower", type=float, default=0.0)
    p.add_argument("--tau-upper", type=float, default=0.0)

    p.add_argument("--oc", type=int, choices=(0, 1), default=1)
    p.add_argument("--amp-v", type=float, default=0.0)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--omega", type=float, default=0.5)

    return p.parse_args()


def main():
    args = parse_args()
    cuda_check()

    cfg = SolverConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        reynolds=args.re,
        mach=args.mach,
        dt=args.dt,
        max_steps=args.steps,
        statistic_frequency=args.istat,
        tau_lower=args.tau_lower,
        tau_upper=args.tau_upper,
        enable_oc=args.oc,
        actuation_amplitude=args.amp_v,
        actuation_wavenumber=args.kappa,
        actuation_omega=args.omega,
    )

    param_input = load_param_dat("param.dat", cfg)
    input_file = args.input or param_input

    # Fallback search if default path is relative or not explicitly given
    if not input_file and not args.demo:
        for cand in ["input/3D.dat", "3D.dat", "../input/3D.dat"]:
            if os.path.exists(cand):
                input_file = cand
                break

    time_init = 0.0
    nstep_init = 0

    if input_file and not args.demo:
        q0, time_init, nstep_init = read_combined_input(input_file, cfg)
        print(f"Loaded combined CFD input: {input_file} (t={time_init:.4f}, step={nstep_init})")
    elif args.demo:
        q0 = demo_initial_condition(cfg)
        print("Using synthetic demo initial condition.")
    else:
        raise SystemExit("No input file found. Provide --input INPUT_FILE or use --demo.")

    solver = CUDASolver(cfg, q0)
    solver.time_value = time_init
    solver.nstep = nstep_init
    solver.run()

    if args.output:
        solver.save_solution(args.output)


if __name__ == "__main__":
    main()
