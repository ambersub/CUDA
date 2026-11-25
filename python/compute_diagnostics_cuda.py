#!/usr/bin/env python3
"""
GPU-Accelerated CFD Diagnostic & Statistics Suite for 3D Compressible Channel Flow.

Covers both high-frequency diagnostics (KE, rhobulk, Ubulk, massflow, FFT, tauWall)
and spatial/turbulence statistics streams matching codes1/mod_stats.f90 and
codes/main_v1.9_unc.f90:
  - Diagnostic_files(): KE.dat, rhobulk.dat, Ubulk.dat, massflow.dat
  - PowSpec(): FFT.dat
  - tauWall_write(): tauWall.dat (including dwdx contribution over interior (nx-2)*(ny-2))
  - write_stats() / TKE_stats(): moment1st.dat, moment2nd.dat, momentJoint.dat,
                                 moment1stT.dat, moment2ndT.dat, momentJointT.dat
  - Omega_stats(): Omega1st.dat, Omega2nd.dat
  - Cf_stats(): Cf1st.dat, Cf2nd.dat, CfJoint.dat
  - Favre(): Favre1st.dat, Favre2nd.dat, FavreJoint.dat
  - Extended ("full") streams: power_OC.dat, power_STWSV.dat, power.dat, uv_slip.dat,
                               tauWall_Lxt1/2.dat, moment80.dat, momentxP*.dat,
                               TKE_xt_*.dat, xz/moment*_xz.dat, NU/moment*.dat
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import numba
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


TWO_THIRDS = 2.0 / 3.0
# 0-based (a, b) indices matching mod_stats.f90 joint correlations:
# rhow, rhoT, uv, uw, vw, vT, Tw, Pw, PT
JOINT_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 3),
    (0, 5),
    (1, 2),
    (1, 3),
    (2, 3),
    (2, 5),
    (5, 3),
    (6, 3),
    (6, 5),
)


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
    def kernel_wall_shear(
        u,
        w,
        T,
        f0,
        f1,
        f2,
        b0,
        b1,
        b2,
        inv_dx,
        invRe,
        out_tau_bot,
        out_tau_top,
        nx,
        ny,
        nz,
    ):
        """Computes bottom/top wall shear stress (dwdx + dudz)*mu/Re on GPU."""
        i, j = cuda.grid(2)
        if i >= nx or j >= ny:
            return

        if nx > 2:
            nx_int = nx - 2
            im1 = (i - 2 + nx_int) % nx_int + 1 if (1 <= i <= nx - 2) else max(i - 1, 0)
            ip1 = (i % nx_int) + 1 if (1 <= i <= nx - 2) else min(i + 1, nx - 1)
        else:
            im1 = max(i - 1, 0)
            ip1 = min(i + 1, nx - 1)

        mu_bot = T[i, j, 0] ** 0.7
        dwdx_bot = 0.5 * (w[ip1, j, 0] - w[im1, j, 0]) * inv_dx
        dudz_bot = f0 * u[i, j, 0] + f1 * u[i, j, 1] + f2 * u[i, j, 2]
        out_tau_bot[i, j] = mu_bot * (dwdx_bot + dudz_bot) * invRe

        mu_top = T[i, j, nz - 1] ** 0.7
        dwdx_top = 0.5 * (w[ip1, j, nz - 1] - w[im1, j, nz - 1]) * inv_dx
        dudz_top = b0 * u[i, j, nz - 1] + b1 * u[i, j, nz - 2] + b2 * u[i, j, nz - 3]
        out_tau_top[i, j] = mu_top * (dwdx_top + dudz_top) * invRe

    @cuda.jit
    def kernel_ke_jk_sum(var, out_sum):
        """Reduces kinetic energy u^2 + v^2 + w^2 across x inside each (j, k) thread."""
        j, k = cuda.grid(2)
        nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
        if j >= ny or k >= nz:
            return
        acc = 0.0
        for i in range(nx):
            u = var[i, j, k, 1]
            v = var[i, j, k, 2]
            w = var[i, j, k, 3]
            acc += u * u + v * v + w * w
        cuda.atomic.add(out_sum, 0, acc)

    @cuda.jit
    def kernel_plane_bulk_sums(var, planes, d_dz, dy, out_sums):
        """
        Computes bilinear area-weighted plane integrals of rho, u, rho*u for the
        5 monitoring stations directly from device var (shape (nx, ny, nz, 7)).
        """
        k, p_idx = cuda.grid(2)
        ny, nz = var.shape[1], var.shape[2]
        if p_idx >= planes.shape[0] or k >= nz - 1:
            return
        p = planes[p_idx]
        if p <= 1 or p >= var.shape[0] - 2:
            return

        r_acc = 0.0
        u_acc = 0.0
        m_acc = 0.0
        w_area = 0.25 * dy * d_dz[k]

        for j in range(ny - 1):
            r00 = var[p, j, k, 0]
            r01 = var[p, j + 1, k, 0]
            r10 = var[p, j, k + 1, 0]
            r11 = var[p, j + 1, k + 1, 0]

            u00 = var[p, j, k, 1]
            u01 = var[p, j + 1, k, 1]
            u10 = var[p, j, k + 1, 1]
            u11 = var[p, j + 1, k + 1, 1]

            r_acc += r00 + r01 + r10 + r11
            u_acc += u00 + u01 + u10 + u11
            m_acc += r00 * u00 + r01 * u01 + r10 * u10 + r11 * u11

        cuda.atomic.add(out_sums, (p_idx, 0), r_acc * w_area)
        cuda.atomic.add(out_sums, (p_idx, 1), u_acc * w_area)
        cuda.atomic.add(out_sums, (p_idx, 2), m_acc * w_area)

    @cuda.jit
    def kernel_moments_planar(var, out_moments):
        """
        Computes 1st moments (0..6), 2nd moments (7..13), 9 joint moments (14..22),
        and triple velocity moments u^2*w, v^2*w, w^3 (23..25) summed along x,
        reduced via atomicAdd per (j, k) thread -> out_moments[nz, 26].
        """
        j, k = cuda.grid(2)
        nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
        if j >= ny or k >= nz:
            return

        m0 = m1 = m2 = m3 = m4 = m5 = m6 = 0.0
        s0 = s1 = s2 = s3 = s4 = s5 = s6 = 0.0
        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = 0.0
        t0 = t1 = t2 = 0.0

        for i in range(nx):
            v0 = var[i, j, k, 0]
            v1 = var[i, j, k, 1]
            v2 = var[i, j, k, 2]
            v3 = var[i, j, k, 3]
            v4 = var[i, j, k, 4]
            v5 = var[i, j, k, 5]
            v6 = var[i, j, k, 6]

            m0 += v0; m1 += v1; m2 += v2; m3 += v3; m4 += v4; m5 += v5; m6 += v6
            s0 += v0 * v0; s1 += v1 * v1; s2 += v2 * v2; s3 += v3 * v3
            s4 += v4 * v4; s5 += v5 * v5; s6 += v6 * v6

            c0 += v0 * v3
            c1 += v0 * v5
            c2 += v1 * v2
            c3 += v1 * v3
            c4 += v2 * v3
            c5 += v2 * v5
            c6 += v5 * v3
            c7 += v6 * v3
            c8 += v6 * v5

            t0 += v1 * v1 * v3
            t1 += v2 * v2 * v3
            t2 += v3 * v3 * v3

        cuda.atomic.add(out_moments, (k, 0), m0)
        cuda.atomic.add(out_moments, (k, 1), m1)
        cuda.atomic.add(out_moments, (k, 2), m2)
        cuda.atomic.add(out_moments, (k, 3), m3)
        cuda.atomic.add(out_moments, (k, 4), m4)
        cuda.atomic.add(out_moments, (k, 5), m5)
        cuda.atomic.add(out_moments, (k, 6), m6)
        cuda.atomic.add(out_moments, (k, 7), s0)
        cuda.atomic.add(out_moments, (k, 8), s1)
        cuda.atomic.add(out_moments, (k, 9), s2)
        cuda.atomic.add(out_moments, (k, 10), s3)
        cuda.atomic.add(out_moments, (k, 11), s4)
        cuda.atomic.add(out_moments, (k, 12), s5)
        cuda.atomic.add(out_moments, (k, 13), s6)
        cuda.atomic.add(out_moments, (k, 14), c0)
        cuda.atomic.add(out_moments, (k, 15), c1)
        cuda.atomic.add(out_moments, (k, 16), c2)
        cuda.atomic.add(out_moments, (k, 17), c3)
        cuda.atomic.add(out_moments, (k, 18), c4)
        cuda.atomic.add(out_moments, (k, 19), c5)
        cuda.atomic.add(out_moments, (k, 20), c6)
        cuda.atomic.add(out_moments, (k, 21), c7)
        cuda.atomic.add(out_moments, (k, 22), c8)
        cuda.atomic.add(out_moments, (k, 23), t0)
        cuda.atomic.add(out_moments, (k, 24), t1)
        cuda.atomic.add(out_moments, (k, 25), t2)

    @cuda.jit
    def kernel_favre_planar(var, out_favre):
        """
        Computes Favre-weighted planar sums matching mod_stats.f90:Favre():
          0..6  : sum(rho * var[p])
          7..13 : sum((rho * var[p])^2)
          14..22: sum(rho * joint_9)
        """
        j, k = cuda.grid(2)
        nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
        if j >= ny or k >= nz:
            return

        f0 = f1 = f2 = f3 = f4 = f5 = f6 = 0.0
        s0 = s1 = s2 = s3 = s4 = s5 = s6 = 0.0
        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = 0.0

        for i in range(nx):
            rho = var[i, j, k, 0]
            v1 = var[i, j, k, 1]
            v2 = var[i, j, k, 2]
            v3 = var[i, j, k, 3]
            v4 = var[i, j, k, 4]
            v5 = var[i, j, k, 5]
            v6 = var[i, j, k, 6]

            r0 = rho * rho
            r1 = rho * v1
            r2 = rho * v2
            r3 = rho * v3
            r4 = rho * v4
            r5 = rho * v5
            r6 = rho * v6

            f0 += r0; f1 += r1; f2 += r2; f3 += r3; f4 += r4; f5 += r5; f6 += r6
            s0 += r0 * r0; s1 += r1 * r1; s2 += r2 * r2; s3 += r3 * r3
            s4 += r4 * r4; s5 += r5 * r5; s6 += r6 * r6

            c0 += r0 * v3
            c1 += r0 * v5
            c2 += r1 * v2
            c3 += r1 * v3
            c4 += r2 * v3
            c5 += r2 * v5
            c6 += r5 * v3
            c7 += r6 * v3
            c8 += r6 * v5

        cuda.atomic.add(out_favre, (k, 0), f0)
        cuda.atomic.add(out_favre, (k, 1), f1)
        cuda.atomic.add(out_favre, (k, 2), f2)
        cuda.atomic.add(out_favre, (k, 3), f3)
        cuda.atomic.add(out_favre, (k, 4), f4)
        cuda.atomic.add(out_favre, (k, 5), f5)
        cuda.atomic.add(out_favre, (k, 6), f6)
        cuda.atomic.add(out_favre, (k, 7), s0)
        cuda.atomic.add(out_favre, (k, 8), s1)
        cuda.atomic.add(out_favre, (k, 9), s2)
        cuda.atomic.add(out_favre, (k, 10), s3)
        cuda.atomic.add(out_favre, (k, 11), s4)
        cuda.atomic.add(out_favre, (k, 12), s5)
        cuda.atomic.add(out_favre, (k, 13), s6)
        cuda.atomic.add(out_favre, (k, 14), c0)
        cuda.atomic.add(out_favre, (k, 15), c1)
        cuda.atomic.add(out_favre, (k, 16), c2)
        cuda.atomic.add(out_favre, (k, 17), c3)
        cuda.atomic.add(out_favre, (k, 18), c4)
        cuda.atomic.add(out_favre, (k, 19), c5)
        cuda.atomic.add(out_favre, (k, 20), c6)
        cuda.atomic.add(out_favre, (k, 21), c7)
        cuda.atomic.add(out_favre, (k, 22), c8)

    @cuda.jit(device=True)
    def _tke_dz_comp(var, i, j, k, nz, comp, inv_dxi, inv_jac_k, low2, high2):
        if k == 0:
            return low2[0] * var[i, j, 0, comp] + low2[1] * var[i, j, 1, comp] + low2[2] * var[i, j, 2, comp]
        if k == nz - 1:
            return (
                high2[0] * var[i, j, nz - 1, comp]
                + high2[1] * var[i, j, nz - 2, comp]
                + high2[2] * var[i, j, nz - 3, comp]
            )
        return 0.5 * (var[i, j, k + 1, comp] - var[i, j, k - 1, comp]) * inv_dxi * inv_jac_k

    @cuda.jit
    def kernel_tke_grad(
        var,
        mu,
        low2,
        high2,
        inv_dx,
        inv_dy,
        inv_dxi,
        inv_jac,
        inv_re,
        out_tke,
    ):
        """
        Computes interior (1..nx-2, 1..ny-2) derivative and viscous-stress sums per k
        matching mod_stats.f90:TKE_stats() -> out_tke[nz, 25].
        """
        j, k = cuda.grid(2)
        nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
        if j < 1 or j >= ny - 1 or k >= nz:
            return

        nx_int = nx - 2
        ny_int = ny - 2
        jm1 = (j - 2 + ny_int) % ny_int + 1
        jp1 = (j % ny_int) + 1
        inv_jk = inv_jac[k]

        m0 = m1 = m2 = m3 = m4 = m5 = m6 = m7 = 0.0
        s0 = s1 = s2 = s3 = s4 = s5 = s6 = s7 = 0.0
        c0 = c1 = c2 = c4 = c5 = c6 = c8 = c9 = c10 = 0.0

        for i in range(1, nx - 1):
            im1 = (i - 2 + nx_int) % nx_int + 1
            ip1 = (i % nx_int) + 1

            u = var[i, j, k, 1]
            v = var[i, j, k, 2]
            w = var[i, j, k, 3]
            p = var[i, j, k, 6]
            muv = mu[i, j, k]

            dudx = 0.5 * (var[ip1, j, k, 1] - var[im1, j, k, 1]) * inv_dx
            dwdx = 0.5 * (var[ip1, j, k, 3] - var[im1, j, k, 3]) * inv_dx
            dvdy = 0.5 * (var[i, jp1, k, 2] - var[i, jm1, k, 2]) * inv_dy
            dwdy = 0.5 * (var[i, jp1, k, 3] - var[i, jm1, k, 3]) * inv_dy

            dudz = _tke_dz_comp(var, i, j, k, nz, 1, inv_dxi, inv_jk, low2, high2)
            dvdz = _tke_dz_comp(var, i, j, k, nz, 2, inv_dxi, inv_jk, low2, high2)
            dwdz = _tke_dz_comp(var, i, j, k, nz, 3, inv_dxi, inv_jk, low2, high2)

            tauxz = muv * (dwdx + dudz) * inv_re
            tauyz = muv * (dwdy + dvdz) * inv_re
            tauzz = muv * (2.0 * dwdz - TWO_THIRDS * (dudx + dvdy + dwdz)) * inv_re

            m0 += dudx; m1 += dudz; m2 += tauxz; m3 += dvdz
            m4 += dwdz; m5 += dvdy; m6 += tauyz; m7 += tauzz

            s0 += dudx * dudx; s1 += dudz * dudz; s2 += tauxz * tauxz; s3 += dvdz * dvdz
            s4 += dwdz * dwdz; s5 += dvdy * dvdy; s6 += tauyz * tauyz; s7 += tauzz * tauzz

            c0 += p * dudx
            c1 += u * tauxz
            c2 += tauxz * dudz
            c4 += p * dvdy
            c5 += v * tauyz
            c6 += tauyz * dvdz
            c8 += p * dwdz
            c9 += w * tauzz
            c10 += tauzz * dwdz

        cuda.atomic.add(out_tke, (k, 0), m0)
        cuda.atomic.add(out_tke, (k, 1), m1)
        cuda.atomic.add(out_tke, (k, 2), m2)
        cuda.atomic.add(out_tke, (k, 3), m3)
        cuda.atomic.add(out_tke, (k, 4), m4)
        cuda.atomic.add(out_tke, (k, 5), m5)
        cuda.atomic.add(out_tke, (k, 6), m6)
        cuda.atomic.add(out_tke, (k, 7), m7)
        cuda.atomic.add(out_tke, (k, 8), s0)
        cuda.atomic.add(out_tke, (k, 9), s1)
        cuda.atomic.add(out_tke, (k, 10), s2)
        cuda.atomic.add(out_tke, (k, 11), s3)
        cuda.atomic.add(out_tke, (k, 12), s4)
        cuda.atomic.add(out_tke, (k, 13), s5)
        cuda.atomic.add(out_tke, (k, 14), s6)
        cuda.atomic.add(out_tke, (k, 15), s7)
        cuda.atomic.add(out_tke, (k, 16), c0)
        cuda.atomic.add(out_tke, (k, 17), c1)
        cuda.atomic.add(out_tke, (k, 18), c2)
        cuda.atomic.add(out_tke, (k, 19), c4)
        cuda.atomic.add(out_tke, (k, 20), c5)
        cuda.atomic.add(out_tke, (k, 21), c6)
        cuda.atomic.add(out_tke, (k, 22), c8)
        cuda.atomic.add(out_tke, (k, 23), c9)
        cuda.atomic.add(out_tke, (k, 24), c10)

    @cuda.jit
    def kernel_omega_cf_grad(var, x3, inv_dx, inv_dy, out_ocf):
        """
        Computes vorticity sums and Cf joint correlations matching
        mod_stats.f90:Omega_stats() and Cf_stats() -> out_ocf[nz, 8].
        """
        j, k = cuda.grid(2)
        nx, ny, nz = var.shape[0], var.shape[1], var.shape[2]
        if j < 1 or j >= ny - 1 or k >= nz:
            return

        nx_int = nx - 2
        ny_int = ny - 2
        jm1 = (j - 2 + ny_int) % ny_int + 1
        jp1 = (j % ny_int) + 1

        if k == nz - 1:
            ka, kb = k, k - 1
        else:
            ka, kb = k + 1, k
        inv_dz = 1.0 / (x3[ka] - x3[kb])

        ox_s = oy_s = oz_s = 0.0
        ox2_s = oy2_s = oz2_s = 0.0
        woy_s = voz_s = 0.0

        for i in range(1, nx - 1):
            im1 = (i - 2 + nx_int) % nx_int + 1
            ip1 = (i % nx_int) + 1

            dudz = (var[i, j, ka, 1] - var[i, j, kb, 1]) * inv_dz
            dvdz = (var[i, j, ka, 2] - var[i, j, kb, 2]) * inv_dz
            dvdx = 0.5 * (var[ip1, j, k, 2] - var[im1, j, k, 2]) * inv_dx
            dwdx = 0.5 * (var[ip1, j, k, 3] - var[im1, j, k, 3]) * inv_dx
            dudy = 0.5 * (var[i, jp1, k, 1] - var[i, jm1, k, 1]) * inv_dy
            dwdy = 0.5 * (var[i, jp1, k, 3] - var[i, jm1, k, 3]) * inv_dy

            ox = dwdy - dvdz
            oy = dwdx - dudz
            oz = dvdx - dudy

            ox_s += ox; oy_s += oy; oz_s += oz
            ox2_s += ox * ox; oy2_s += oy * oy; oz2_s += oz * oz
            woy_s += var[i, j, k, 3] * (-0.5 * oy)
            voz_s += var[i, j, k, 2] * (0.5 * oz)

        cuda.atomic.add(out_ocf, (k, 0), ox_s)
        cuda.atomic.add(out_ocf, (k, 1), oy_s)
        cuda.atomic.add(out_ocf, (k, 2), oz_s)
        cuda.atomic.add(out_ocf, (k, 3), ox2_s)
        cuda.atomic.add(out_ocf, (k, 4), oy2_s)
        cuda.atomic.add(out_ocf, (k, 5), oz2_s)
        cuda.atomic.add(out_ocf, (k, 6), woy_s)
        cuda.atomic.add(out_ocf, (k, 7), voz_s)


# ---------------------------------------------------------------------------
# Mesh / Stencil Utilities
# ---------------------------------------------------------------------------

def default_monitoring_planes(nx: int) -> List[int]:
    """
    Matches Fortran plane indices:
      - codes/main_v1.9_unc.f90: [5, 10, 70, 100, 125] (1-based, nx <= 130)
      - codes1/mod_var_dec.f90: [15, 50, 80, 130, 175] (1-based, nx > 130)
    """
    raw = [4, 9, 69, 99, 124] if nx <= 130 else [14, 49, 79, 129, 174]
    hi = max(nx - 2, 1)
    return [min(max(idx, 1), hi) for idx in raw]


def compute_wall_coefficients(z_coords: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Computes 2nd-order asymmetric finite-difference coefficients for dudz at wall boundaries
    matching Fortran main_v1.9_unc.f90 lines 56-90:
      Bottom: f1d2c0, f1d2c1, f1d2c2
      Top:    b1d2c0, b1d2c1, b1d2c2
    """
    dz1 = z_coords[1] - z_coords[0]
    alpha_bot = (z_coords[2] - z_coords[1]) / dz1
    tmpR1_bot = dz1 * alpha_bot * (alpha_bot + 1.0)
    f0 = (1.0 - (alpha_bot + 1.0) ** 2) / tmpR1_bot
    f1 = ((alpha_bot + 1.0) ** 2) / tmpR1_bot
    f2 = -1.0 / tmpR1_bot

    dzN = z_coords[-1] - z_coords[-2]
    alpha_top = (z_coords[-2] - z_coords[-3]) / dzN
    tmpR1_top = dzN * alpha_top * (alpha_top + 1.0)
    b0 = ((alpha_top + 1.0) ** 2 - 1.0) / tmpR1_top
    b1 = -((alpha_top + 1.0) ** 2) / tmpR1_top
    b2 = 1.0 / tmpR1_top

    return f0, f1, f2, b0, b1, b2


def _default_inv_jac(z_coords: np.ndarray) -> np.ndarray:
    """Reconstructs interior dxi / dz inverse metric Jacobian from z_coords."""
    nz = len(z_coords)
    dxi = 2.0 / max(nz - 1, 1)
    inv_jac = np.zeros(nz, dtype=np.float64)
    if nz > 2:
        dz_central = (z_coords[2:] - z_coords[:-2]) / (2.0 * dxi)
        mask = np.abs(dz_central) > 1.0e-12
        inv_jac[1:-1][mask] = 1.0 / dz_central[mask]
    return inv_jac


@dataclass
class LiveStatsWorkspace:
    """Reusable GPU reduction buffers for zero-3D-copy live diagnostics and stats."""

    nx: int
    ny: int
    nz: int
    planes: np.ndarray
    d_planes: Any
    d_dz: Any
    d_x3: Any
    d_ke_sum: Any
    d_bulk_sums: Any
    d_moments: Any
    d_favre: Any
    d_tke: Any
    d_ocf: Any

    @classmethod
    def allocate(
        cls,
        nx: int,
        ny: int,
        nz: int,
        z_coords: np.ndarray,
        planes: Optional[List[int]] = None,
    ) -> "LiveStatsWorkspace":
        p_arr = np.asarray(planes or default_monitoring_planes(nx), dtype=np.int32)
        dz_arr = np.diff(np.asarray(z_coords, dtype=np.float64))
        return cls(
            nx=nx,
            ny=ny,
            nz=nz,
            planes=p_arr,
            d_planes=cuda.to_device(p_arr),
            d_dz=cuda.to_device(dz_arr),
            d_x3=cuda.to_device(np.ascontiguousarray(z_coords, dtype=np.float64)),
            d_ke_sum=cuda.device_array(1, dtype=np.float64),
            d_bulk_sums=cuda.device_array((len(p_arr), 3), dtype=np.float64),
            d_moments=cuda.device_array((nz, 26), dtype=np.float64),
            d_favre=cuda.device_array((nz, 23), dtype=np.float64),
            d_tke=cuda.device_array((nz, 25), dtype=np.float64),
            d_ocf=cuda.device_array((nz, 8), dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Diagnostics Calculator Core
# ---------------------------------------------------------------------------

def _wall_shear_cpu(
    u: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    f0: float,
    f1: float,
    f2: float,
    b0: float,
    b1: float,
    b2: float,
    inv_dx: float,
    inv_re: float,
) -> Tuple[float, float]:
    nx, ny, _ = u.shape
    if nx > 2 and ny > 2:
        ip1 = np.roll(np.arange(1, nx - 1), -1)
        im1 = np.roll(np.arange(1, nx - 1), 1)
        mu_bot = T[1:-1, 1:-1, 0] ** 0.7
        dwdx_bot = 0.5 * (w[ip1, 1:-1, 0] - w[im1, 1:-1, 0]) * inv_dx
        dudz_bot = f0 * u[1:-1, 1:-1, 0] + f1 * u[1:-1, 1:-1, 1] + f2 * u[1:-1, 1:-1, 2]
        tau_bot = float(np.mean(mu_bot * (dwdx_bot + dudz_bot) * inv_re))

        mu_top = T[1:-1, 1:-1, -1] ** 0.7
        dwdx_top = 0.5 * (w[ip1, 1:-1, -1] - w[im1, 1:-1, -1]) * inv_dx
        dudz_top = b0 * u[1:-1, 1:-1, -1] + b1 * u[1:-1, 1:-1, -2] + b2 * u[1:-1, 1:-1, -3]
        tau_top = float(np.mean(mu_top * (dwdx_top + dudz_top) * inv_re))
    else:
        dw_bot = np.gradient(w[:, :, 0], axis=0) * inv_dx if nx > 1 else 0.0
        dw_top = np.gradient(w[:, :, -1], axis=0) * inv_dx if nx > 1 else 0.0
        tau_bot = float(np.mean((T[:, :, 0] ** 0.7) * (dw_bot + f0 * u[:, :, 0] + f1 * u[:, :, 1] + f2 * u[:, :, 2]) * inv_re))
        tau_top = float(np.mean((T[:, :, -1] ** 0.7) * (dw_top + b0 * u[:, :, -1] + b1 * u[:, :, -2] + b2 * u[:, :, -3]) * inv_re))
    return tau_bot, tau_top


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
    dx: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes all diagnostic metrics (KE, rhobulk, Ubulk, massflow, tauWall)
    matching Diagnostic_files() and tauWall_write().
    """
    nx, ny, nz = u.shape
    if planes is None:
        planes = default_monitoring_planes(nx)
    if dx is None:
        dx = (4.0 * math.pi) / max(nx - 1, 1)

    Ayz = z_coords[-1] * (dy * (ny - 1))
    dz_arr = np.diff(z_coords)
    invRe = 1.0 / reynolds
    inv_dx = 1.0 / dx if dx > 0 else 0.0
    f0, f1, f2, b0, b1, b2 = compute_wall_coefficients(z_coords)

    if use_gpu and CUDA_AVAILABLE:
        try:
            tpb3 = (8, 8, 8)
            bpg3 = ((nx + 7) // 8, (ny + 7) // 8, (nz + 7) // 8)
            d_u = cuda.to_device(np.ascontiguousarray(u, dtype=np.float64))
            d_v = cuda.to_device(np.ascontiguousarray(v, dtype=np.float64))
            d_w = cuda.to_device(np.ascontiguousarray(w, dtype=np.float64))
            d_ke = cuda.device_array((nx, ny, nz), dtype=np.float64)

            kernel_kinetic_energy[bpg3, tpb3](d_u, d_v, d_w, d_ke, nx, ny, nz)
            cuda.synchronize()
            ke_val = float(np.mean(d_ke.copy_to_host()))

            tpb2 = (16, 16)
            bpg2 = ((nx + 15) // 16, (ny + 15) // 16)
            d_T = cuda.to_device(np.ascontiguousarray(T, dtype=np.float64))
            d_tau_bot = cuda.device_array((nx, ny), dtype=np.float64)
            d_tau_top = cuda.device_array((nx, ny), dtype=np.float64)

            kernel_wall_shear[bpg2, tpb2](
                d_u, d_w, d_T, f0, f1, f2, b0, b1, b2, inv_dx, invRe, d_tau_bot, d_tau_top, nx, ny, nz
            )
            cuda.synchronize()
            tb_h = d_tau_bot.copy_to_host()
            tt_h = d_tau_top.copy_to_host()
            if nx > 2 and ny > 2:
                tau_bot = float(np.mean(tb_h[1:-1, 1:-1]))
                tau_top = float(np.mean(tt_h[1:-1, 1:-1]))
            else:
                tau_bot = float(np.mean(tb_h))
                tau_top = float(np.mean(tt_h))

            d_rho = cuda.to_device(np.ascontiguousarray(rho, dtype=np.float64))
            tpb_plane = (16, 16)
            bpg_plane = ((ny - 1 + 15) // 16, (nz - 1 + 15) // 16)
            d_rq = cuda.device_array((ny - 1, nz - 1), dtype=np.float64)
            d_mq = cuda.device_array((ny - 1, nz - 1), dtype=np.float64)
            d_uq = cuda.device_array((ny - 1, nz - 1), dtype=np.float64)
            dA = dy * dz_arr[None, :]

            rhobulk_list, ubulk_list, massflow_list = [], [], []
            for p in planes:
                kernel_plane_quads[bpg_plane, tpb_plane](d_rho, d_u, p, d_rq, d_mq, d_uq, ny, nz)
                cuda.synchronize()
                rhobulk_list.append(float(np.sum(d_rq.copy_to_host() * dA) / Ayz))
                massflow_list.append(float(np.sum(d_mq.copy_to_host() * dA) / Ayz))
                ubulk_list.append(float(np.sum(d_uq.copy_to_host() * dA) / Ayz))

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

    ke_val = float(np.mean(u ** 2 + v ** 2 + w ** 2))
    tau_bot, tau_top = _wall_shear_cpu(u, w, T, f0, f1, f2, b0, b1, b2, inv_dx, invRe)

    rhobulk_list, ubulk_list, massflow_list = [], [], []
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


def compute_live_step_diagnostics(
    d_var: Any,
    x2: np.ndarray,
    x3: np.ndarray,
    dy: float,
    workspace: LiveStatsWorkspace,
    tau_lower: float = 0.0,
    tau_upper: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes Diagnostic_files() and PowSpec() scalars directly from GPU d_var
    with sub-kilobyte device-to-host copies (no full 3D snapshot download).
    """
    nx, ny, nz = workspace.nx, workspace.ny, workspace.nz
    workspace.d_ke_sum.copy_to_device(np.zeros(1, dtype=np.float64))
    tpb2 = (16, 8)
    bpg_jk = ((ny + tpb2[0] - 1) // tpb2[0], (nz + tpb2[1] - 1) // tpb2[1])
    kernel_ke_jk_sum[bpg_jk, tpb2](d_var, workspace.d_ke_sum)

    n_planes = len(workspace.planes)
    workspace.d_bulk_sums.copy_to_device(np.zeros((n_planes, 3), dtype=np.float64))
    bpg_p = ((max(nz - 1, 1) + tpb2[0] - 1) // tpb2[0], (n_planes + tpb2[1] - 1) // tpb2[1])
    kernel_plane_bulk_sums[bpg_p, tpb2](
        d_var, workspace.d_planes, workspace.d_dz, dy, workspace.d_bulk_sums
    )

    ke_val = float(workspace.d_ke_sum.copy_to_host()[0]) / float(nx * ny * nz)
    bulk_h = workspace.d_bulk_sums.copy_to_host()
    Ayz = float(x3[-1] * x2[-1]) if len(x2) > 1 and len(x3) > 1 else 1.0

    # PowSpec(): 1-based var(bn(1)+5, bn(2)+5, k=(16,36,52), 2)
    i_fft = min(6, max(nx - 2, 0))
    j_fft = min(6, max(ny - 2, 0))
    k_fft = [min(k_1b - 1, nz - 1) for k_1b in (16, 36, 52)]
    u_line = d_var[i_fft, j_fft, :, 1].copy_to_host()
    fft_vals = [float(u_line[k_idx]) for k_idx in k_fft]

    return {
        "KE": ke_val,
        "rhobulk": [float(v / Ayz) for v in bulk_h[:, 0]],
        "Ubulk": [float(v / Ayz) for v in bulk_h[:, 1]],
        "massflow": [float(v / Ayz) for v in bulk_h[:, 2]],
        "fft": fft_vals,
        "tau_bottom": float(tau_lower),
        "tau_top": float(tau_upper),
        "tau_avg": float(0.5 * (tau_lower - tau_upper)),
    }


# ---------------------------------------------------------------------------
# Phase 2 Statistics Kernels, CPU Reference & Full mod_stats Parity
# ---------------------------------------------------------------------------

def _moments_and_favre_from_raw(
    raw_moments: np.ndarray,
    raw_favre: np.ndarray,
    nx: int,
    ny: int,
) -> Dict[str, np.ndarray]:
    """Post-scales planar sum arrays (nz, 26) and (nz, 23) matching mod_stats.f90."""
    nxy = float(nx * ny)
    m_avg = raw_moments / nxy
    stat1 = m_avg[:, 0:7].copy()
    stat2 = m_avg[:, 7:14].copy()
    corr = m_avg[:, 14:23].copy()
    rho_mean = stat1[:, 0]
    rho_safe = np.where(np.abs(rho_mean) > 1.0e-14, rho_mean, 1.0)
    triple_uvw = 0.5 * rho_mean[:, None] * m_avg[:, 23:26]

    f_avg = raw_favre / nxy
    fav1 = f_avg[:, 0:7] / rho_safe[:, None]
    fav2 = f_avg[:, 7:14] / (rho_safe[:, None] ** 2)
    fav_corr = f_avg[:, 14:23] / rho_safe[:, None]
    return {
        "stat1Spc": stat1,
        "stat2Spc": stat2,
        "corrSpc": corr,
        "triple_uvw": triple_uvw,
        "Fav_stat1Spc": fav1,
        "Fav_stat2Spc": fav2,
        "Fav_corrSpc": fav_corr,
    }


def _assemble_tke_omega_cf(
    stat1Spc: np.ndarray,
    stat2Spc: np.ndarray,
    triple_uvw: np.ndarray,
    raw_tke: np.ndarray,
    raw_ocf: np.ndarray,
    nx: int,
    ny: int,
) -> Dict[str, np.ndarray]:
    nxy_2 = float(max((nx - 2) * (ny - 2), 1))
    tke_avg = raw_tke / nxy_2
    stat1SpcT = tke_avg[:, 0:8].copy()
    stat2SpcT = tke_avg[:, 8:16].copy()

    corrSpcT = np.empty((stat1Spc.shape[0], 12), dtype=np.float64)
    # Interior (nXY_2) gradient/stress correlations
    corrSpcT[:, 0] = tke_avg[:, 16]
    corrSpcT[:, 1] = tke_avg[:, 17]
    corrSpcT[:, 2] = tke_avg[:, 18]
    corrSpcT[:, 3] = triple_uvw[:, 0]
    corrSpcT[:, 4] = tke_avg[:, 19]
    corrSpcT[:, 5] = tke_avg[:, 20]
    corrSpcT[:, 6] = tke_avg[:, 21]
    corrSpcT[:, 7] = triple_uvw[:, 1]
    corrSpcT[:, 8] = tke_avg[:, 22]
    corrSpcT[:, 9] = tke_avg[:, 23]
    corrSpcT[:, 10] = tke_avg[:, 24]
    corrSpcT[:, 11] = triple_uvw[:, 2]

    ocf_avg = raw_ocf / nxy_2
    omega1 = ocf_avg[:, 0:3].copy()
    omega2 = ocf_avg[:, 3:6].copy()

    cf1 = np.empty((stat1Spc.shape[0], 7), dtype=np.float64)
    cf2 = np.empty((stat1Spc.shape[0], 7), dtype=np.float64)
    cf1[:, 0:4] = stat1Spc[:, 0:4]
    cf2[:, 0:4] = stat2Spc[:, 0:4]
    cf1[:, 4] = 0.5 * omega1[:, 0]
    cf1[:, 5] = -0.5 * omega1[:, 1]
    cf1[:, 6] = 0.5 * omega1[:, 2]
    cf2[:, 4:7] = 0.25 * omega2
    cf_corr = ocf_avg[:, 6:8].copy()

    return {
        "stat1SpcT": stat1SpcT,
        "stat2SpcT": stat2SpcT,
        "corrSpcT": corrSpcT,
        "omega1": omega1,
        "omega2": omega2,
        "cf1": cf1,
        "cf2": cf2,
        "cf_corr": cf_corr,
    }


def _moments_along_axes(
    arr: np.ndarray, axes: Union[int, Tuple[int, ...]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes (stat1[7], stat2[7], corr[9]) along specified reduction axes."""
    stat1 = np.mean(arr, axis=axes)
    stat2 = np.mean(arr * arr, axis=axes)
    corr = np.stack([np.mean(arr[..., a] * arr[..., b], axis=axes) for a, b in JOINT_PAIRS], axis=-1)
    return stat1, stat2, corr


def _compute_extra_full_streams(
    var: np.ndarray,
    mu: np.ndarray,
    x3: np.ndarray,
    dx: float,
    dy: float,
    dxi: float,
    inv_jac: np.ndarray,
    low2: np.ndarray,
    high2: np.ndarray,
    inv_re: float,
    stat1Spc: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes the remaining optional mod_stats.f90 streams ("full" mode):
      power_OC, power_STWSV, power_uc, slip, tauWall_Lxt,
      moment80 (write_stats_wMax), write_stats_xAvg, TKE_stats_xt, stats_xz, NU_stats.
    """
    nx, ny, nz, _ = var.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dxi = 1.0 / dxi

    # 1. power_OC, power_STWSV, power_uc over 1..nx-2, 1..ny-2
    v_b = var[1:-1, 1:-1, 0, :]
    v_t = var[1:-1, 1:-1, -1, :]
    mu_b = mu[1:-1, 1:-1, 0]
    mu_t = mu[1:-1, 1:-1, -1]

    pow_oc_l = float(np.mean(np.abs(v_b[..., 6] * v_b[..., 3]) + 0.5 * np.abs(v_b[..., 0] * (v_b[..., 3] ** 3))))
    pow_oc_u = float(np.mean(np.abs(v_t[..., 6] * v_t[..., 3]) + 0.5 * np.abs(v_t[..., 0] * (v_t[..., 3] ** 3))))

    dvdz_b = low2[0] * var[1:-1, 1:-1, 0, 2] + low2[1] * var[1:-1, 1:-1, 1, 2] + low2[2] * var[1:-1, 1:-1, 2, 2]
    dvdz_t = high2[0] * var[1:-1, 1:-1, -1, 2] + high2[1] * var[1:-1, 1:-1, -2, 2] + high2[2] * var[1:-1, 1:-1, -3, 2]
    pow_stv_l = float(np.mean(v_b[..., 2] * mu_b * dvdz_b * inv_re))
    pow_stv_u = float(np.mean(v_t[..., 2] * mu_t * dvdz_t * inv_re))

    dudz_b = low2[0] * var[1:-1, 1:-1, 0, 1] + low2[1] * var[1:-1, 1:-1, 1, 1] + low2[2] * var[1:-1, 1:-1, 2, 1]
    dudz_t = high2[0] * var[1:-1, 1:-1, -1, 1] + high2[1] * var[1:-1, 1:-1, -2, 1] + high2[2] * var[1:-1, 1:-1, -3, 1]
    pow_uc_l = float(np.mean(mu_b * dudz_b * inv_re))
    pow_uc_u = float(np.mean(mu_t * dudz_t * inv_re))

    # 2. tauWall_Lxt(): bottom wall shear averaged along x (1..nx-2) for every j (0..ny-1)
    ip1 = np.roll(np.arange(1, nx - 1), -1)
    im1 = np.roll(np.arange(1, nx - 1), 1)
    dwdx_b = 0.5 * (var[ip1, :, 0, 3] - var[im1, :, 0, 3]) * inv_dx
    dudz_b_all = low2[0] * var[1:-1, :, 0, 1] + low2[1] * var[1:-1, :, 1, 1] + low2[2] * var[1:-1, :, 2, 1]
    tau_loc_b = mu[1:-1, :, 0] * (dwdx_b + dudz_b_all) * inv_re
    tau_lxt1 = np.mean(tau_loc_b, axis=0)
    tau_lxt2 = np.mean(tau_loc_b * tau_loc_b, axis=0)

    # 3. slip(): [stat1Spc(0, u), stat1Spc(0, v)]
    slip_uv = np.array([stat1Spc[0, 1], stat1Spc[0, 2]], dtype=np.float64)

    # 4. write_stats_wMax(): k=80 (1-based -> 79, clamped for smaller grids)
    k_wmax = min(79, nz - 1)
    moment80 = np.mean(var[:, :, k_wmax, :], axis=0)  # (ny, 7)

    # 5. write_stats_xAvg(): x-averaged (ny, nz) stats
    l_s1, l_s2, _ = _moments_along_axes(var, axes=0)
    l_corr = np.stack(
        [
            np.mean(var[..., 1] * var[..., 3], axis=0),  # uw
            np.mean(var[..., 5] * var[..., 3], axis=0),  # Tw
        ],
        axis=-1,
    )

    # 6. TKE_stats_xt(): (ny, nz) TKE gradients using exact mod_stats.f90:752-765 j-bounds
    var_int = var[1:-1, :, :, :]  # (nx-2, ny, nz, 7)
    mu_int = mu[1:-1, :, :]
    nx_i = var_int.shape[0]
    ip = np.roll(np.arange(nx_i), -1)
    im = np.roll(np.arange(nx_i), 1)

    dudx_xt = 0.5 * (var_int[ip, :, :, 1] - var_int[im, :, :, 1]) * inv_dx
    dwdx_xt = 0.5 * (var_int[ip, :, :, 3] - var_int[im, :, :, 3]) * inv_dx

    # z-derivatives (f1d2 at k=0, b1d2 at k=nz-1, central interior)
    dudz_xt = np.empty((nx_i, ny, nz), dtype=np.float64)
    dvdz_xt = np.empty((nx_i, ny, nz), dtype=np.float64)
    dwdz_xt = np.empty((nx_i, ny, nz), dtype=np.float64)
    for comp_idx, target in ((1, dudz_xt), (2, dvdz_xt), (3, dwdz_xt)):
        target[:, :, 0] = (
            low2[0] * var_int[:, :, 0, comp_idx]
            + low2[1] * var_int[:, :, 1, comp_idx]
            + low2[2] * var_int[:, :, 2, comp_idx]
        )
        target[:, :, -1] = (
            high2[0] * var_int[:, :, -1, comp_idx]
            + high2[1] * var_int[:, :, -2, comp_idx]
            + high2[2] * var_int[:, :, -3, comp_idx]
        )
        target[:, :, 1:-1] = (
            0.5 * (var_int[:, :, 2:, comp_idx] - var_int[:, :, :-2, comp_idx]) * inv_dxi * inv_jac[None, None, 1:-1]
        )

    # y-derivatives matching mod_stats.f90:752-761 (one-sided at j=0, j=ny-1)
    dwdy_xt = np.empty((nx_i, ny, nz), dtype=np.float64)
    dvdy_xt = np.empty((nx_i, ny, nz), dtype=np.float64)
    dwdy_xt[:, 0, :] = (var_int[:, 1, :, 3] - var_int[:, 0, :, 3]) * inv_dy
    dvdy_xt[:, 0, :] = (var_int[:, 1, :, 2] - var_int[:, 0, :, 2]) * inv_dy
    dwdy_xt[:, -1, :] = (var_int[:, -1, :, 3] - var_int[:, -2, :, 3]) * inv_dy
    dvdy_xt[:, -1, :] = (var_int[:, -1, :, 2] - var_int[:, -2, :, 2]) * inv_dy
    dwdy_xt[:, 1:-1, :] = 0.5 * (var_int[:, 2:, :, 3] - var_int[:, :-2, :, 3]) * inv_dy
    dvdy_xt[:, 1:-1, :] = 0.5 * (var_int[:, 2:, :, 2] - var_int[:, :-2, :, 2]) * inv_dy

    tauxz_xt = mu_int * (dwdx_xt + dudz_xt) * inv_re
    tauyz_xt = mu_int * (dwdy_xt + dvdz_xt) * inv_re
    tauzz_xt = mu_int * (2.0 * dwdz_xt - TWO_THIRDS * (dudx_xt + dvdy_xt + dwdz_xt)) * inv_re

    temp2_xt = np.stack(
        [dudx_xt, dudz_xt, tauxz_xt, dvdz_xt, dwdz_xt, dvdy_xt, tauyz_xt, tauzz_xt],
        axis=-1,
    )
    xt_s1, xt_s2, xt_corr = _moments_along_axes(var, axes=0)
    xt_st1 = np.mean(temp2_xt, axis=0)
    xt_st2 = np.mean(temp2_xt * temp2_xt, axis=0)
    rho_jk = xt_s1[None, :, :, 0]
    xt_corrs = np.stack(
        [
            np.mean(var_int[..., 6] * dudx_xt, axis=0),
            np.mean(var_int[..., 1] * tauxz_xt, axis=0),
            np.mean(tauxz_xt * dudz_xt, axis=0),
            np.mean(0.5 * rho_jk * (var[..., 1] ** 2) * var[..., 3], axis=0),
            np.mean(var_int[..., 6] * dvdy_xt, axis=0),
            np.mean(var_int[..., 2] * tauyz_xt, axis=0),
            np.mean(tauyz_xt * dvdz_xt, axis=0),
            np.mean(0.5 * rho_jk * (var[..., 2] ** 2) * var[..., 3], axis=0),
            np.mean(var_int[..., 6] * dwdz_xt, axis=0),
            np.mean(var_int[..., 3] * tauzz_xt, axis=0),
            np.mean(tauzz_xt * dwdz_xt, axis=0),
            np.mean(0.5 * rho_jk * (var[..., 3] ** 3), axis=0),
        ],
        axis=-1,
    )

    # 7. stats_xz(): (nx, nz) stats averaged over y
    xz_s1, xz_s2, xz_corr = _moments_along_axes(var, axes=1)

    # 8. NU_stats(): q in (70, 80, 90, 100, 110, 120) -> 0-based q-1 when present
    nu_slices: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for q in (70, 80, 90, 100, 110, 120):
        idx_q = min(q - 1, nx - 1)
        nu_slices[q] = _moments_along_axes(var[idx_q, :, :, :], axes=0)

    return {
        "power_OC": (pow_oc_l, pow_oc_u, 0.5 * (pow_oc_l + pow_oc_u)),
        "power_STWSV": (pow_stv_l, pow_stv_u, abs(pow_stv_l) + abs(pow_stv_u)),
        "power_uc": (pow_uc_l, pow_uc_u, 0.5 * (pow_uc_l + pow_uc_u)),
        "slip": slip_uv,
        "tauWall_Lxt": (tau_lxt1, tau_lxt2),
        "moment80": moment80,
        "xAvg": (l_s1, l_s2, l_corr),
        "TKE_xt": (xt_s1, xt_s2, xt_corr, xt_st1, xt_st2, xt_corrs),
        "xz": (xz_s1, xz_s2, xz_corr),
        "NU": nu_slices,
    }


def compute_spatial_stats_cpu(
    var: np.ndarray,
    mu: np.ndarray,
    x3: np.ndarray,
    inv_jac: Optional[np.ndarray] = None,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    dxi: Optional[float] = None,
    low2: Optional[np.ndarray] = None,
    high2: Optional[np.ndarray] = None,
    reynolds: float = 3000.0,
    mode: str = "standard",
) -> Dict[str, Any]:
    """CPU reference computation for all Phase 2 spatial/turbulence statistics."""
    nx, ny, nz, _ = var.shape
    dx = (4.0 * math.pi) / max(nx - 1, 1) if dx is None else dx
    dy = (4.0 * math.pi / 3.0) / max(ny - 1, 1) if dy is None else dy
    dxi = 2.0 / max(nz - 1, 1) if dxi is None else dxi
    if inv_jac is None:
        inv_jac = _default_inv_jac(x3)
    if low2 is None or high2 is None:
        f0, f1, f2, b0, b1, b2 = compute_wall_coefficients(x3)
        low2 = np.array([f0, f1, f2], dtype=np.float64)
        high2 = np.array([b0, b1, b2], dtype=np.float64)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dxi = 1.0 / dxi
    inv_re = 1.0 / reynolds

    tau_l, tau_u = _wall_shear_cpu(
        var[..., 1], var[..., 3], var[..., 5],
        low2[0], low2[1], low2[2], high2[0], high2[1], high2[2],
        inv_dx, inv_re,
    )

    stat1Spc, stat2Spc, corrSpc = _moments_along_axes(var, axes=(0, 1))
    rho_mean = stat1Spc[:, 0]
    rho_safe = np.where(np.abs(rho_mean) > 1.0e-14, rho_mean, 1.0)
    triple_uvw = 0.5 * rho_mean[:, None] * np.stack(
        [
            np.mean((var[..., 1] ** 2) * var[..., 3], axis=(0, 1)),
            np.mean((var[..., 2] ** 2) * var[..., 3], axis=(0, 1)),
            np.mean(var[..., 3] ** 3, axis=(0, 1)),
        ],
        axis=-1,
    )

    rho_grid = var[..., 0:1]
    fav1 = np.mean(rho_grid * var, axis=(0, 1)) / rho_safe[:, None]
    fav2 = np.mean((rho_grid * var) ** 2, axis=(0, 1)) / (rho_safe[:, None] ** 2)
    fav_corr = np.stack(
        [np.mean(var[..., 0] * var[..., a] * var[..., b], axis=(0, 1)) / rho_safe for a, b in JOINT_PAIRS],
        axis=-1,
    )

    out: Dict[str, Any] = {
        "tau_lower": tau_l,
        "tau_upper": tau_u,
        "stat1Spc": stat1Spc,
        "stat2Spc": stat2Spc,
        "corrSpc": corrSpc,
        "Fav_stat1Spc": fav1,
        "Fav_stat2Spc": fav2,
        "Fav_corrSpc": fav_corr,
    }
    if mode == "minimal":
        return out

    # Interior 1..nx-2, 1..ny-2 gradients for TKE_stats, Omega_stats, Cf_stats
    v_in = var[1:-1, 1:-1, :, :]
    mu_in = mu[1:-1, 1:-1, :]
    nx_i = v_in.shape[0]
    ny_i = v_in.shape[1]

    dudx = 0.5 * (np.roll(v_in[..., 1], -1, axis=0) - np.roll(v_in[..., 1], 1, axis=0)) * inv_dx
    dvdx = 0.5 * (np.roll(v_in[..., 2], -1, axis=0) - np.roll(v_in[..., 2], 1, axis=0)) * inv_dx
    dwdx = 0.5 * (np.roll(v_in[..., 3], -1, axis=0) - np.roll(v_in[..., 3], 1, axis=0)) * inv_dx

    dudy = 0.5 * (np.roll(v_in[..., 1], -1, axis=1) - np.roll(v_in[..., 1], 1, axis=1)) * inv_dy
    dvdy = 0.5 * (np.roll(v_in[..., 2], -1, axis=1) - np.roll(v_in[..., 2], 1, axis=1)) * inv_dy
    dwdy = 0.5 * (np.roll(v_in[..., 3], -1, axis=1) - np.roll(v_in[..., 3], 1, axis=1)) * inv_dy

    dudz = np.empty((nx_i, ny_i, nz), dtype=np.float64)
    dvdz = np.empty((nx_i, ny_i, nz), dtype=np.float64)
    dwdz = np.empty((nx_i, ny_i, nz), dtype=np.float64)
    for comp_idx, arr_d in ((1, dudz), (2, dvdz), (3, dwdz)):
        arr_d[:, :, 0] = (
            low2[0] * v_in[:, :, 0, comp_idx]
            + low2[1] * v_in[:, :, 1, comp_idx]
            + low2[2] * v_in[:, :, 2, comp_idx]
        )
        arr_d[:, :, -1] = (
            high2[0] * v_in[:, :, -1, comp_idx]
            + high2[1] * v_in[:, :, -2, comp_idx]
            + high2[2] * v_in[:, :, -3, comp_idx]
        )
        arr_d[:, :, 1:-1] = (
            0.5 * (v_in[:, :, 2:, comp_idx] - v_in[:, :, :-2, comp_idx]) * inv_dxi * inv_jac[None, None, 1:-1]
        )

    tauxz = mu_in * (dwdx + dudz) * inv_re
    tauyz = mu_in * (dwdy + dvdz) * inv_re
    tauzz = mu_in * (2.0 * dwdz - TWO_THIRDS * (dudx + dvdy + dwdz)) * inv_re

    locT = np.stack([dudx, dudz, tauxz, dvdz, dwdz, dvdy, tauyz, tauzz], axis=-1)
    stat1SpcT = np.mean(locT, axis=(0, 1))
    stat2SpcT = np.mean(locT * locT, axis=(0, 1))

    p_in = v_in[..., 6]
    u_in = v_in[..., 1]
    v_c = v_in[..., 2]
    w_in = v_in[..., 3]
    corrSpcT = np.stack(
        [
            np.mean(p_in * dudx, axis=(0, 1)),
            np.mean(u_in * tauxz, axis=(0, 1)),
            np.mean(tauxz * dudz, axis=(0, 1)),
            triple_uvw[:, 0],
            np.mean(p_in * dvdy, axis=(0, 1)),
            np.mean(v_c * tauyz, axis=(0, 1)),
            np.mean(tauyz * dvdz, axis=(0, 1)),
            triple_uvw[:, 1],
            np.mean(p_in * dwdz, axis=(0, 1)),
            np.mean(w_in * tauzz, axis=(0, 1)),
            np.mean(tauzz * dwdz, axis=(0, 1)),
            triple_uvw[:, 2],
        ],
        axis=-1,
    )

    # Omega & Cf forward/backward z-differences
    dz_step = np.empty(nz, dtype=np.float64)
    dz_step[:-1] = x3[1:] - x3[:-1]
    dz_step[-1] = x3[-1] - x3[-2]
    dudz_f = np.empty((nx_i, ny_i, nz), dtype=np.float64)
    dvdz_f = np.empty((nx_i, ny_i, nz), dtype=np.float64)
    dudz_f[:, :, :-1] = (v_in[:, :, 1:, 1] - v_in[:, :, :-1, 1]) / dz_step[None, None, :-1]
    dudz_f[:, :, -1] = (v_in[:, :, -1, 1] - v_in[:, :, -2, 1]) / dz_step[-1]
    dvdz_f[:, :, :-1] = (v_in[:, :, 1:, 2] - v_in[:, :, :-1, 2]) / dz_step[None, None, :-1]
    dvdz_f[:, :, -1] = (v_in[:, :, -1, 2] - v_in[:, :, -2, 2]) / dz_step[-1]

    ox = dwdy - dvdz_f
    oy = dwdx - dudz_f
    oz = dvdx - dudy
    omega_arr = np.stack([ox, oy, oz], axis=-1)
    omega1 = np.mean(omega_arr, axis=(0, 1))
    omega2 = np.mean(omega_arr * omega_arr, axis=(0, 1))

    cf1 = np.empty((nz, 7), dtype=np.float64)
    cf2 = np.empty((nz, 7), dtype=np.float64)
    cf1[:, 0:4] = stat1Spc[:, 0:4]
    cf2[:, 0:4] = stat2Spc[:, 0:4]
    cf1[:, 4] = 0.5 * omega1[:, 0]
    cf1[:, 5] = -0.5 * omega1[:, 1]
    cf1[:, 6] = 0.5 * omega1[:, 2]
    cf2[:, 4:7] = 0.25 * omega2
    cf_corr = np.stack(
        [np.mean(w_in * (-0.5 * oy), axis=(0, 1)), np.mean(v_c * (0.5 * oz), axis=(0, 1))],
        axis=-1,
    )

    out.update(
        stat1SpcT=stat1SpcT,
        stat2SpcT=stat2SpcT,
        corrSpcT=corrSpcT,
        omega1=omega1,
        omega2=omega2,
        cf1=cf1,
        cf2=cf2,
        cf_corr=cf_corr,
    )
    if mode == "full":
        out["extra"] = _compute_extra_full_streams(
            var, mu, x3, dx, dy, dxi, inv_jac, low2, high2, inv_re, stat1Spc
        )
    return out


def compute_spatial_stats_gpu(
    d_var: Any,
    d_mu: Any,
    x3: np.ndarray,
    d_inv_jac: Any,
    d_low2: Any,
    d_high2: Any,
    dx: float,
    dy: float,
    dxi: float,
    reynolds: float,
    workspace: LiveStatsWorkspace,
    tau_lower: Optional[float] = None,
    tau_upper: Optional[float] = None,
    mode: str = "standard",
) -> Dict[str, Any]:
    """Computes spatial/Favre/TKE/Omega/Cf statistics directly from GPU d_var."""
    nx, ny, nz = workspace.nx, workspace.ny, workspace.nz
    if tau_lower is None or tau_upper is None:
        low2_h = d_low2.copy_to_host()
        high2_h = d_high2.copy_to_host()
        u_b = d_var[:, :, 0:3, 1].copy_to_host()
        w_b = d_var[:, :, 0:1, 3].copy_to_host()
        mu_b = d_mu[:, :, 0].copy_to_host()
        u_t = d_var[:, :, nz - 3 : nz, 1].copy_to_host()
        w_t = d_var[:, :, nz - 1 : nz, 3].copy_to_host()
        mu_t = d_mu[:, :, nz - 1].copy_to_host()
        inv_re_val = 1.0 / reynolds
        inv_dx_val = 1.0 / dx
        if nx > 2 and ny > 2:
            ip1 = np.roll(np.arange(1, nx - 1), -1)
            im1 = np.roll(np.arange(1, nx - 1), 1)
            dwdx_b = 0.5 * (w_b[ip1, 1:-1, 0] - w_b[im1, 1:-1, 0]) * inv_dx_val
            dudz_b = (
                low2_h[0] * u_b[1:-1, 1:-1, 0]
                + low2_h[1] * u_b[1:-1, 1:-1, 1]
                + low2_h[2] * u_b[1:-1, 1:-1, 2]
            )
            tl_calc = float(np.mean(mu_b[1:-1, 1:-1] * (dwdx_b + dudz_b) * inv_re_val))

            dwdx_t = 0.5 * (w_t[ip1, 1:-1, 0] - w_t[im1, 1:-1, 0]) * inv_dx_val
            dudz_t = (
                high2_h[0] * u_t[1:-1, 1:-1, 2]
                + high2_h[1] * u_t[1:-1, 1:-1, 1]
                + high2_h[2] * u_t[1:-1, 1:-1, 0]
            )
            tu_calc = float(np.mean(mu_t[1:-1, 1:-1] * (dwdx_t + dudz_t) * inv_re_val))
        else:
            tl_calc = 0.0
            tu_calc = 0.0
        if tau_lower is None:
            tau_lower = tl_calc
        if tau_upper is None:
            tau_upper = tu_calc

    tpb2 = (16, 8)
    bpg_all = ((ny + tpb2[0] - 1) // tpb2[0], (nz + tpb2[1] - 1) // tpb2[1])

    workspace.d_moments.copy_to_device(np.zeros((nz, 26), dtype=np.float64))
    kernel_moments_planar[bpg_all, tpb2](d_var, workspace.d_moments)

    workspace.d_favre.copy_to_device(np.zeros((nz, 23), dtype=np.float64))
    if mode != "minimal":
        kernel_favre_planar[bpg_all, tpb2](d_var, workspace.d_favre)

    raw_moments = workspace.d_moments.copy_to_host()
    raw_favre = workspace.d_favre.copy_to_host() if mode != "minimal" else np.zeros((nz, 23), dtype=np.float64)
    out = _moments_and_favre_from_raw(raw_moments, raw_favre, nx, ny)
    out["tau_lower"] = float(tau_lower)
    out["tau_upper"] = float(tau_upper)

    if mode == "minimal":
        return out

    inv_re = 1.0 / reynolds
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dxi = 1.0 / dxi

    workspace.d_tke.copy_to_device(np.zeros((nz, 25), dtype=np.float64))
    kernel_tke_grad[bpg_all, tpb2](
        d_var, d_mu, d_low2, d_high2, inv_dx, inv_dy, inv_dxi, d_inv_jac, inv_re, workspace.d_tke
    )

    workspace.d_ocf.copy_to_device(np.zeros((nz, 8), dtype=np.float64))
    kernel_omega_cf_grad[bpg_all, tpb2](d_var, workspace.d_x3, inv_dx, inv_dy, workspace.d_ocf)

    raw_tke = workspace.d_tke.copy_to_host()
    raw_ocf = workspace.d_ocf.copy_to_host()
    out.update(
        _assemble_tke_omega_cf(
            out["stat1Spc"], out["stat2Spc"], out["triple_uvw"], raw_tke, raw_ocf, nx, ny
        )
    )

    if mode == "full":
        var_h = d_var.copy_to_host()
        mu_h = d_mu.copy_to_host()
        inv_jac_h = d_inv_jac.copy_to_host()
        low2_h = d_low2.copy_to_host()
        high2_h = d_high2.copy_to_host()
        out["extra"] = _compute_extra_full_streams(
            var_h, mu_h, x3, dx, dy, dxi, inv_jac_h, low2_h, high2_h, inv_re, out["stat1Spc"]
        )
    return out


# ---------------------------------------------------------------------------
# Output File Appenders (Exact Fortran Formatting)
# ---------------------------------------------------------------------------

def _append_f15_row(path: Path, time_val: float, values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.asarray(values, dtype=np.float64).ravel()
    with path.open("a", encoding="utf-8") as f:
        f.write(f" {time_val:15.7f}")
        for val in flat:
            f.write(f" {float(val):15.7f}")
        f.write("\n")


def append_powspec_record(output_dir: Union[str, Path], time_val: float, fft_vals: List[float]) -> None:
    """Appends time + 3 sampled u values to FFT.dat matching PowSpec()."""
    _append_f15_row(Path(output_dir) / "FFT.dat", time_val, fft_vals)


def append_diagnostic_record(
    output_dir: Union[str, Path],
    time_val: float,
    diag: Dict[str, Any],
    include_tau: bool = True,
) -> None:
    """
    Appends new records to KE.dat, rhobulk.dat, Ubulk.dat, massflow.dat, and
    optionally tauWall.dat (matching Fortran formats 108, 109, 110).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _append_f15_row(out_dir / "KE.dat", time_val, [diag["KE"]])
    _append_f15_row(out_dir / "rhobulk.dat", time_val, diag["rhobulk"])
    _append_f15_row(out_dir / "Ubulk.dat", time_val, diag["Ubulk"])
    _append_f15_row(out_dir / "massflow.dat", time_val, diag["massflow"])
    if include_tau:
        _append_f15_row(out_dir / "tauWall.dat", time_val, [diag["tau_bottom"], diag["tau_top"]])


def append_spatial_stats(
    output_dir: Union[str, Path],
    time_val: float,
    stats: Dict[str, Any],
    mode: str = "standard",
) -> None:
    """Appends spatial, TKE, Omega, Cf, Favre, and optional extended mod_stats.f90 files."""
    if mode == "none":
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _append_f15_row(out_dir / "tauWall.dat", time_val, [stats["tau_lower"], stats["tau_upper"]])
    _append_f15_row(out_dir / "moment1st.dat", time_val, stats["stat1Spc"])
    _append_f15_row(out_dir / "moment2nd.dat", time_val, stats["stat2Spc"])
    _append_f15_row(out_dir / "momentJoint.dat", time_val, stats["corrSpc"])

    if mode == "minimal":
        return

    _append_f15_row(out_dir / "moment1stT.dat", time_val, stats["stat1SpcT"])
    _append_f15_row(out_dir / "moment2ndT.dat", time_val, stats["stat2SpcT"])
    _append_f15_row(out_dir / "momentJointT.dat", time_val, stats["corrSpcT"])

    _append_f15_row(out_dir / "Omega1st.dat", time_val, stats["omega1"])
    _append_f15_row(out_dir / "Omega2nd.dat", time_val, stats["omega2"])

    _append_f15_row(out_dir / "Cf1st.dat", time_val, stats["cf1"])
    _append_f15_row(out_dir / "Cf2nd.dat", time_val, stats["cf2"])
    _append_f15_row(out_dir / "CfJoint.dat", time_val, stats["cf_corr"])

    _append_f15_row(out_dir / "Favre1st.dat", time_val, stats["Fav_stat1Spc"])
    _append_f15_row(out_dir / "Favre2nd.dat", time_val, stats["Fav_stat2Spc"])
    _append_f15_row(out_dir / "FavreJoint.dat", time_val, stats["Fav_corrSpc"])

    if mode == "full" and "extra" in stats:
        ex = stats["extra"]
        _append_f15_row(out_dir / "power_OC.dat", time_val, ex["power_OC"])
        _append_f15_row(out_dir / "power_STWSV.dat", time_val, ex["power_STWSV"])
        _append_f15_row(out_dir / "power.dat", time_val, ex["power_uc"])
        _append_f15_row(out_dir / "uv_slip.dat", time_val, ex["slip"])

        tau_lxt1, tau_lxt2 = ex["tauWall_Lxt"]
        _append_f15_row(out_dir / "tauWall_Lxt1.dat", time_val, tau_lxt1)
        _append_f15_row(out_dir / "tauWall_Lxt2.dat", time_val, tau_lxt2)
        _append_f15_row(out_dir / "moment80.dat", time_val, ex["moment80"])

        l_s1, l_s2, l_corr = ex["xAvg"]
        # Loop k outer, j inner matching Fortran mod_stats.f90:1028-1033 -> transpose (nz, ny, cols)
        _append_f15_row(out_dir / "momentxP.dat", time_val, l_s1.transpose(1, 0, 2))
        _append_f15_row(out_dir / "momentxP2.dat", time_val, l_s2.transpose(1, 0, 2))
        _append_f15_row(out_dir / "momentxPJoint.dat", time_val, l_corr.transpose(1, 0, 2))

        xt_s1, xt_s2, xt_corr, xt_st1, xt_st2, xt_corrs = ex["TKE_xt"]
        _append_f15_row(out_dir / "TKE_xt_1st.dat", time_val, xt_s1.transpose(1, 0, 2))
        _append_f15_row(out_dir / "TKE_xt_2nd.dat", time_val, xt_s2.transpose(1, 0, 2))
        _append_f15_row(out_dir / "TKE_xt_Joint.dat", time_val, xt_corr.transpose(1, 0, 2))
        _append_f15_row(out_dir / "TKE_xt_1stT.dat", time_val, xt_st1.transpose(1, 0, 2))
        _append_f15_row(out_dir / "TKE_xt_2ndT.dat", time_val, xt_st2.transpose(1, 0, 2))
        _append_f15_row(out_dir / "TKE_xt_JointT.dat", time_val, xt_corrs.transpose(1, 0, 2))

        xz_s1, xz_s2, xz_corr = ex["xz"]
        _append_f15_row(out_dir / "xz" / "moment1st_xz.dat", time_val, xz_s1.transpose(1, 0, 2))
        _append_f15_row(out_dir / "xz" / "moment2nd_xz.dat", time_val, xz_s2.transpose(1, 0, 2))
        _append_f15_row(out_dir / "xz" / "momentJoint_xz.dat", time_val, xz_corr.transpose(1, 0, 2))

        for q, (ns1, ns2, nc) in ex["NU"].items():
            _append_f15_row(out_dir / "NU" / f"moment1st{q:03d}.dat", time_val, ns1)
            _append_f15_row(out_dir / "NU" / f"moment2nd{q:03d}.dat", time_val, ns2)
            _append_f15_row(out_dir / "NU" / f"momentJoint{q:03d}.dat", time_val, nc)


# ---------------------------------------------------------------------------
# Ingestion & Dispatch Functions
# ---------------------------------------------------------------------------

def process_3d_snapshot(
    snapshot_path: Union[str, Path],
    output_dir: Union[str, Path] = "output_cuda",
    append: bool = True,
    use_gpu: bool = True,
    reynolds: float = 3000.0,
) -> Optional[Dict[str, Any]]:
    """
    Reads a 3D Tecplot snapshot (e.g. 3D.dat or 3D_step*.dat), extracts time and flow fields,
    computes diagnostics via CUDA, and appends to target .dat files.
    """
    path = Path(snapshot_path)
    if not path.is_file():
        print(f"    [!] Error: Snapshot file not found: {path}")
        return None

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
        arr_3d = arr.reshape(nz, ny, nx, 10)

        x_coords = arr_3d[0, 0, :, 0]
        y_coords = arr_3d[0, :, 0, 1]
        z_coords = arr_3d[:, 0, 0, 2]

        dx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else (4.0 * math.pi) / (nx - 1)
        dy = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else (4.0 * math.pi / 3.0) / (ny - 1)

        rho = np.transpose(arr_3d[..., 3], (2, 1, 0))
        u   = np.transpose(arr_3d[..., 4], (2, 1, 0))
        v   = np.transpose(arr_3d[..., 5], (2, 1, 0))
        w   = np.transpose(arr_3d[..., 6], (2, 1, 0))
        T   = np.transpose(arr_3d[..., 8], (2, 1, 0))

        if np.isnan(u[0, 0, 0]) or np.isnan(rho[0, 0, 0]):
            print(f"    [!] Skipping snapshot '{path.name}': contains NaN values.")
            return None

        diag = calculate_diagnostics_cuda(
            rho, u, v, w, T, z_coords, dy, reynolds=reynolds, use_gpu=use_gpu, dx=dx
        )
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
        default_single = Path("output_cuda/3D.dat")
        if default_single.is_file():
            print(f"[+] Processing default snapshot: {default_single}")
            process_3d_snapshot(default_single, output_dir=args.output_dir, append=True, use_gpu=use_gpu)
        else:
            process_snapshot_series("output_cuda", output_dir=args.output_dir, pattern=args.pattern, use_gpu=use_gpu)


if __name__ == "__main__":
    main()
