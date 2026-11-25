#!/usr/bin/env python3
"""
CUDA/Python Post-Processing Suite (Phase 3) replicating codes/ and codes1/ utilities:
  - Temporal averaging with restart-overlap deduplication (avg_v1.7.f90, avgTauWall.f90,
    avg_v1.7_TKE_ALL.f90, avg_Favre.f90, avg_Omega.f90, avg_Cf.f90)
  - Compressible channel scaling & identities (avg_VD.f90, avg_FIK.f90)
  - 3D field post-processors: component combine (combine.f90), fluctuations (fluc.f90),
    and GPU/CPU Q-criterion & vorticity (qInvariant.f90, qInvariant_Omega.f90)
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numba import cuda

CUDA_AVAILABLE = bool(cuda.is_available())

# Joint correlation pairs matching mod_stats.f90: (a, b) 0-based in [rho, u, v, w, E, T, P]
_JOINT_PAIR_INDICES: Tuple[Tuple[int, int], ...] = (
    (0, 3),  # rho * w
    (0, 5),  # rho * T
    (1, 2),  # u * v
    (1, 3),  # u * w
    (2, 3),  # v * w
    (2, 5),  # v * T
    (3, 5),  # w * T
    (3, 6),  # w * P
    (5, 6),  # T * P
)


def _load_solver_module():
    root = Path(__file__).resolve().parent.parent
    cuda_script = root / "cuda code.py"
    spec = importlib.util.spec_from_file_location("postproc_cuda_code", str(cuda_script))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["postproc_cuda_code"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Time-Series Deduplication & Temporal Averaging (avg_v1.7, avgTauWall, etc.)
# ---------------------------------------------------------------------------

def deduplicate_time_series(
    data: np.ndarray,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    stride: int = 1,
) -> np.ndarray:
    """
    Removes overlapping restart intervals where time goes backwards (matching avg_v1.7.f90)
    and optionally filters by [tmin, tmax] and stride.
    """
    arr = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if arr.size == 0:
        return arr

    keep_indices: List[int] = []
    max_time_seen = -math.inf
    for idx in range(arr.shape[0]):
        t = float(arr[idx, 0])
        if t > max_time_seen:
            keep_indices.append(idx)
            max_time_seen = t

    clean = arr[keep_indices]
    if tmin is not None:
        clean = clean[clean[:, 0] >= tmin]
    if tmax is not None:
        clean = clean[clean[:, 0] <= tmax]
    if stride > 1 and clean.shape[0] > 0:
        clean = clean[::stride]
    return clean


def _load_clean_table(
    path: Path,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    stride: int = 1,
) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64)
    return deduplicate_time_series(raw, tmin=tmin, tmax=tmax, stride=stride)


def _infer_z_coords(
    nz: int,
    stats_dir: Path,
    mesh_type: str = "auto",
) -> np.ndarray:
    mean_cand = stats_dir / "mean.dat"
    if mean_cand.exists():
        try:
            m = np.atleast_2d(np.loadtxt(mean_cand, dtype=np.float64))
            if m.shape[0] == nz and m.shape[1] >= 8:
                return m[:, 0].copy()
        except Exception:
            pass
    cuda_mod = _load_solver_module()
    m_type = "tanh" if mesh_type == "auto" else mesh_type
    z, _, _ = cuda_mod.compute_z_grid(nz, m_type)
    return z


def temporal_average_tauwall(
    stats_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    stride: int = 1,
    reynolds: float = 3000.0,
    rho_wall: float = 1.0,
) -> Dict[str, Any]:
    """Replicates codes/avgTauWall.f90: reads tauWall.dat -> writes tauWallAvg.dat + scaling."""
    sdir = Path(stats_dir)
    odir = Path(out_dir) if out_dir is not None else sdir
    odir.mkdir(parents=True, exist_ok=True)

    tau_tbl = _load_clean_table(sdir / "tauWall.dat", tmin=tmin, tmax=tmax, stride=stride)
    if tau_tbl.shape[0] == 0:
        raise ValueError(f"No valid rows in '{sdir / 'tauWall.dat'}'")

    t = tau_tbl[:, 0]
    tau_l = tau_tbl[:, 1]
    tau_u = tau_tbl[:, 2]
    tau_avg = 0.5 * (tau_l - tau_u)

    out_path = odir / "tauWallAvg.dat"
    with out_path.open("w", encoding="utf-8") as f:
        for k in range(len(t)):
            f.write(f" {t[k]:15.7f} {tau_l[k]:15.7f} {tau_u[k]:15.7f} {tau_avg[k]:15.7f}\n")

    tau_mean = float(np.mean(tau_avg))
    u_tau = math.sqrt(abs(tau_mean) / max(rho_wall, 1.0e-14))
    re_tau = rho_wall * u_tau * reynolds
    return {
        "tauWallAvg_path": str(out_path),
        "tau_mean": tau_mean,
        "u_tau": u_tau,
        "Re_tau": re_tau,
        "n_samples": int(len(t)),
    }


def temporal_average_moments(
    stats_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    x3: Optional[np.ndarray] = None,
    mesh_type: str = "tanh",
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    stride: int = 1,
) -> Dict[str, Any]:
    """
    Replicates codes/avg_v1.7.f90 (plus extended TKE/Favre/Omega/Cf temporal averages):
    reads moment1st/2nd/Joint.dat, deduplicates restarts, and writes mean.dat, rms.dat, corr.dat.
    """
    sdir = Path(stats_dir)
    odir = Path(out_dir) if out_dir is not None else sdir
    odir.mkdir(parents=True, exist_ok=True)

    m1 = _load_clean_table(sdir / "moment1st.dat", tmin=tmin, tmax=tmax, stride=stride)
    m2 = _load_clean_table(sdir / "moment2nd.dat", tmin=tmin, tmax=tmax, stride=stride)
    mj = _load_clean_table(sdir / "momentJoint.dat", tmin=tmin, tmax=tmax, stride=stride)
    n_t = min(m1.shape[0], m2.shape[0], mj.shape[0])
    if n_t == 0:
        raise ValueError(f"No matching moment rows found in '{sdir}'")

    m1 = m1[:n_t]
    m2 = m2[:n_t]
    mj = mj[:n_t]

    if (m1.shape[1] - 1) % 7 != 0:
        raise ValueError(f"Unexpected moment1st.dat column count: {m1.shape[1]}")
    nz = (m1.shape[1] - 1) // 7
    z = np.asarray(x3, dtype=np.float64) if x3 is not None else _infer_z_coords(nz, sdir, mesh_type)

    stat1_spc = m1[:, 1:].reshape(n_t, nz, 7)
    stat2_spc = m2[:, 1:].reshape(n_t, nz, 7)
    corr_spc = mj[:, 1:].reshape(n_t, nz, 9)

    stat1 = np.mean(stat1_spc, axis=0)
    stat2 = np.mean(stat2_spc, axis=0)
    rms = np.sqrt(np.maximum(stat2 - stat1 * stat1, 0.0))
    corr_mean = np.mean(corr_spc, axis=0)

    covar = np.empty((nz, 9), dtype=np.float64)
    for col_idx, (a, b) in enumerate(_JOINT_PAIR_INDICES):
        covar[:, col_idx] = corr_mean[:, col_idx] - stat1[:, a] * stat1[:, b]

    mean_path = odir / "mean.dat"
    rms_path = odir / "rms.dat"
    corr_path = odir / "corr.dat"
    np.savetxt(mean_path, np.column_stack([z, stat1]), fmt="%22.15e")
    np.savetxt(rms_path, np.column_stack([z, rms]), fmt="%22.15e")
    np.savetxt(corr_path, np.column_stack([z, covar]), fmt=" %10.5f" * 10)

    t_start, t_end = float(m1[0, 0]), float(m1[-1, 0])
    (odir / "AoNDT.dat").write_text(
        f"Averaged b/w nondimensional time {t_start:10.5f} to {t_end:10.5f}\n"
        f"Samples: {n_t} (nz={nz})\n",
        encoding="utf-8",
    )

    result: Dict[str, Any] = {
        "z": z,
        "mean": stat1,
        "rms": rms,
        "covar": covar,
        "n_samples": int(n_t),
        "t_start": t_start,
        "t_end": t_end,
        "created_files": {
            "mean": str(mean_path),
            "rms": str(rms_path),
            "corr": str(corr_path),
        },
    }

    # Optional extended stream temporal averages (codes1/avg_v1.7_TKE_ALL, avg_Favre, avg_Omega, avg_Cf)
    extended_specs = (
        ("moment1stT.dat", 8, "meanT.dat"),
        ("moment2ndT.dat", 8, "rmsT_raw.dat"),
        ("momentJointT.dat", 12, "corrT.dat"),
        ("Favre1st.dat", 7, "Favre_mean.dat"),
        ("Favre2nd.dat", 7, "Favre_rms_raw.dat"),
        ("FavreJoint.dat", 9, "Favre_corr.dat"),
        ("Omega1st.dat", 3, "Omega_mean.dat"),
        ("Omega2nd.dat", 3, "Omega_rms_raw.dat"),
        ("Cf1st.dat", 7, "Cf_mean.dat"),
        ("Cf2nd.dat", 7, "Cf_rms_raw.dat"),
        ("CfJoint.dat", 2, "Cf_corr.dat"),
    )
    for src_name, ncols, dst_name in extended_specs:
        src_p = sdir / src_name
        if src_p.exists():
            tbl = _load_clean_table(src_p, tmin=tmin, tmax=tmax, stride=stride)
            if tbl.shape[0] > 0 and tbl.shape[1] == 1 + nz * ncols:
                avg_arr = np.mean(tbl[:, 1:].reshape(tbl.shape[0], nz, ncols), axis=0)
                dst_p = odir / dst_name
                np.savetxt(dst_p, np.column_stack([z, avg_arr]), fmt="%22.15e")
                result["created_files"][dst_name] = str(dst_p)

    return result


# ---------------------------------------------------------------------------
# Van Driest / Trettel-Larsson & FIK Skin-Friction Identity (avg_VD, avg_FIK)
# ---------------------------------------------------------------------------

def compute_vd_and_tl(
    z: np.ndarray,
    mean: np.ndarray,
    reynolds: float = 3000.0,
    u_tau: Optional[float] = None,
    re_tau: Optional[float] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Replicates codes1/avg_VD.f90: Van Driest (U_VD+) and Trettel-Larsson (U_TL+)
    velocity transformations over the lower half-channel.
    """
    nz = len(z)
    half = max(nz // 2, 2)
    rho = mean[:, 0]
    u = mean[:, 1]
    T = mean[:, 5]
    mu_mean = np.maximum(T, 1.0e-14) ** 0.7

    if u_tau is None or re_tau is None:
        dz0 = max(z[1] - z[0], 1.0e-12)
        dudz_w = (u[1] - u[0]) / dz0
        tau_w = abs(mu_mean[0] * dudz_w / reynolds)
        u_tau_eff = math.sqrt(max(tau_w / max(rho[0], 1.0e-14), 1.0e-16))
        re_tau_eff = float(rho[0] * u_tau_eff * reynolds / max(mu_mean[0], 1.0e-14))
    else:
        u_tau_eff = float(u_tau)
        re_tau_eff = float(re_tau)

    vd = np.zeros(half, dtype=np.float64)
    tl = np.zeros(half, dtype=np.float64)
    for k in range(1, half):
        km = k - 1
        du = u[k] - u[km]
        dz = max(z[k] - z[km], 1.0e-14)
        rho_fac = math.sqrt(max(rho[km] / max(rho[0], 1.0e-14), 0.0))
        vd[k] = vd[km] + rho_fac * du
        tl_corr = (
            1.0
            + (0.5 / max(rho[km], 1.0e-14)) * ((rho[k] - rho[km]) / dz) * z[km]
            - (1.0 / max(mu_mean[km], 1.0e-14)) * ((mu_mean[k] - mu_mean[km]) / dz) * z[km]
        )
        tl[k] = tl[km] + rho_fac * tl_corr * du

    z_plus = z[:half] * re_tau_eff
    u_vd_plus = vd / max(u_tau_eff, 1.0e-14)
    u_tl_plus = tl / max(u_tau_eff, 1.0e-14)

    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(op, np.column_stack([z_plus, u_vd_plus, u_tl_plus]), fmt=" %22.15e")

    return {
        "z_plus": z_plus,
        "u_VD_plus": u_vd_plus,
        "u_TL_plus": u_tl_plus,
        "u_tau": u_tau_eff,
        "Re_tau": re_tau_eff,
    }


def compute_fik(
    z: np.ndarray,
    mean: np.ndarray,
    rms: np.ndarray,
    covar: np.ndarray,
    reynolds: float = 3000.0,
    out_path: Optional[Union[str, Path]] = None,
) -> Dict[str, np.ndarray]:
    """
    Replicates codes1/avg_FIK.f90: Fukagata-Iwamoto-Kasagi (FIK) skin-friction decomposition
    over the lower half-channel.
    """
    nz = len(z)
    half = max(nz // 2, 2)
    u = mean[:, 1]
    mu_mean = np.maximum(mean[:, 5], 1.0e-14) ** 0.7
    mu_rms = np.maximum(rms[:, 5], 0.0) ** 0.7
    uw = covar[:, 3]  # <u'w'>

    cf1 = np.zeros(half, dtype=np.float64)
    cf2 = np.zeros(half, dtype=np.float64)
    cf3 = np.zeros(half, dtype=np.float64)
    cf1_loc = np.zeros(half, dtype=np.float64)
    cf2_loc = np.zeros(half, dtype=np.float64)
    cf3_loc = np.zeros(half, dtype=np.float64)

    for k in range(half - 1):
        dz = z[k + 1] - z[k]
        w_f = 1.0 - z[k]
        cf1_loc[k] = w_f * uw[k] * dz
        if k == 0:
            dudz = (u[1] - u[0]) / max(dz, 1.0e-14)
            durms_dz = (rms[1, 1] - rms[0, 1]) / max(dz, 1.0e-14)
        else:
            dz2 = max(z[k + 1] - z[k - 1], 1.0e-14)
            dudz = (u[k + 1] - u[k - 1]) / dz2
            durms_dz = (rms[k + 1, 1] - rms[k - 1, 1]) / dz2

        cf2_loc[k] = w_f * (mu_mean[k] / max(mu_mean[0], 1.0e-14) - 1.0) * dudz * dz
        cf3_loc[k] = w_f * mu_rms[k] * durms_dz * dz

        cf1[k] = cf1_loc[k] + (cf1[k - 1] if k > 0 else 0.0)
        cf2[k] = cf2_loc[k] + (cf2[k - 1] if k > 0 else 0.0)
        cf3[k] = cf3_loc[k] + (cf3[k - 1] if k > 0 else 0.0)

    # Carry final point cleanly
    cf1[-1] = cf1[-2]
    cf2[-1] = cf2[-2]
    cf3[-1] = cf3[-2]

    cols = np.column_stack(
        [
            z[:half],
            -6.0 * cf1,
            6.0 * cf2 / reynolds,
            6.0 * cf3 / reynolds,
            -6.0 * cf1_loc,
            6.0 * cf2_loc / reynolds,
            6.0 * cf3_loc / reynolds,
        ]
    )
    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(op, cols, fmt=" %22.15e")

    return {
        "z": z[:half],
        "cf_turb": -6.0 * cf1,
        "cf_visc_mean": 6.0 * cf2 / reynolds,
        "cf_visc_rms": 6.0 * cf3 / reynolds,
        "table": cols,
    }


# ---------------------------------------------------------------------------
# 3D Post-Processors: Combine, Fluctuations, and Q-Invariant (GPU + CPU)
# ---------------------------------------------------------------------------

def combine_component_checkpoints(
    dom_info_path: Union[str, Path],
    output_path: Union[str, Path],
) -> str:
    """Replicates codes/combine.f90: stitches domInfo.txt + 3Dnnn.dat directly into one 3D.dat."""
    cuda_mod = _load_solver_module()
    dom_p = Path(dom_info_path)
    base_dir = dom_p.parent
    with dom_p.open("r", encoding="utf-8") as f:
        dims = [int(tok) for tok in f.readline().split()]
        nproc = int(f.readline().strip().split()[0])
        bounds = [[int(tok) for tok in f.readline().split()] for _ in range(nproc)]

    nx, ny, nz = dims[0], dims[1], dims[2]
    grid_data = np.empty((nz, ny, nx, 10), dtype=np.float64)
    time_val, nstep_val = 0.0, 0
    for rank, (bx, ex, by, ey, bz, ez) in enumerate(bounds):
        comp_path = base_dir / f"3D{rank + 1:03d}.dat"
        t_r, step_r, _, _, _ = cuda_mod._parse_tecplot_header(str(comp_path))
        if rank == 0:
            time_val, nstep_val = t_r, step_r
        lx, ly, lz = ex - bx + 1, ey - by + 1, ez - bz + 1
        block = cuda_mod._read_tecplot_point_block(str(comp_path), lx * ly * lz)
        grid_data[bz - 1 : ez, by - 1 : ey, bx - 1 : ex, :] = block[..., -10:].reshape(lz, ly, lx, 10)

    x1 = grid_data[0, 0, :, 0].copy()
    x2 = grid_data[0, :, 0, 1].copy()
    x3 = grid_data[:, 0, 0, 2].copy()
    var = np.transpose(grid_data[..., 3:10], (2, 1, 0, 3)).copy()
    cuda_mod._write_tecplot_subzone(str(output_path), x1, x2, x3, var, time_val, nstep_val)
    return str(output_path)


def _read_3d_tecplot_primitives(snapshot_path: Union[str, Path]) -> Dict[str, Any]:
    """Reads a Tecplot POINT 3D snapshot into (x1, x2, x3, var[nx,ny,nz,7], time, nstep)."""
    import pandas as pd

    cuda_mod = _load_solver_module()
    p = Path(snapshot_path)
    time_val, nstep_val, nx, ny, nz = cuda_mod._parse_tecplot_header(str(p))
    if not (nx and ny and nz):
        raise ValueError(f"Could not parse ZONE dimensions from '{p}'")

    expected = nx * ny * nz
    df = pd.read_csv(
        p,
        skiprows=3,
        sep=r"\s+",
        header=None,
        comment="T",
        on_bad_lines="skip",
        engine="c",
    )
    arr = df.dropna().iloc[:expected, :10].to_numpy(dtype=np.float64)
    if arr.shape[0] < expected:
        raise ValueError(f"Snapshot '{p}' has {arr.shape[0]} rows; expected {expected}")

    arr_3d = arr.reshape(nz, ny, nx, 10)
    x1 = arr_3d[0, 0, :, 0].copy()
    x2 = arr_3d[0, :, 0, 1].copy()
    x3 = arr_3d[:, 0, 0, 2].copy()
    var = np.transpose(arr_3d[..., 3:10], (2, 1, 0, 3)).copy()
    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "var": var,
        "time": time_val,
        "nstep": nstep_val,
    }


def compute_fluc_3d(
    snapshot_path: Union[str, Path],
    mean_path: Union[str, Path],
    output_path: Union[str, Path],
) -> str:
    """Replicates codes/fluc.f90: subtracts 1D mean(z) from 3D snapshot and writes fluc.dat."""
    snap = _read_3d_tecplot_primitives(snapshot_path)
    x1, x2, x3, var = snap["x1"], snap["x2"], snap["x3"], snap["var"]
    mean_tbl = np.atleast_2d(np.loadtxt(mean_path, dtype=np.float64))
    nz = len(x3)
    if mean_tbl.shape[0] != nz or mean_tbl.shape[1] < 8:
        raise ValueError(f"mean.dat shape {mean_tbl.shape} incompatible with nz={nz}")

    fluc = var - mean_tbl[:, 1:8][None, None, :, :]
    nx, ny, _ = var.shape[:3]
    out_arr = np.empty((nz, ny, nx, 10), dtype=np.float64)
    for k in range(nz):
        out_arr[k, :, :, 0] = x1[None, :]
        out_arr[k, :, :, 1] = x2[:, None]
        out_arr[k, :, :, 2] = x3[k]
        out_arr[k, :, :, 3:] = fluc[:, :, k, :].transpose(1, 0, 2)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f'TITLE ="{snap["time"]:20.10f} {snap["nstep"]:9d}"\n'
        f'Variables ="x","y","z","rho_prime","U_prime","V_prime","W_prime","E_prime","T_prime","P_prime"\n'
        f'ZONE k={nz},j={ny},i={nx},DATAPACKING="POINT"'
    )
    fmt = " %10.6f %10.6f %10.6f " + " ".join(["%22.15e"] * 7)
    np.savetxt(out_p, out_arr.reshape(-1, 10), fmt=fmt, header=header, comments="")
    return str(out_p)


if CUDA_AVAILABLE:

    @cuda.jit
    def kernel_q_invariant(vel, fluc_vel, x1, x2, x3, out_omega, out_q):
        """
        Computes vorticity omega = curl(vel) and Q-criterion on stretched grid
        matching codes1/qInvariant_Omega.f90.
        """
        i, j, k = cuda.grid(3)
        nx, ny, nz = vel.shape[0], vel.shape[1], vel.shape[2]
        if i >= nx or j >= ny or k >= nz:
            return

        # Vorticity using one-sided forward (backward at upper boundary) differences
        if i == nx - 1:
            dvdx_o = (vel[i, j, k, 1] - vel[i - 1, j, k, 1]) / (x1[i] - x1[i - 1])
            dwdx_o = (vel[i, j, k, 2] - vel[i - 1, j, k, 2]) / (x1[i] - x1[i - 1])
        else:
            dvdx_o = (vel[i + 1, j, k, 1] - vel[i, j, k, 1]) / (x1[i + 1] - x1[i])
            dwdx_o = (vel[i + 1, j, k, 2] - vel[i, j, k, 2]) / (x1[i + 1] - x1[i])

        if j == ny - 1:
            dudy_o = (vel[i, j, k, 0] - vel[i, j - 1, k, 0]) / (x2[j] - x2[j - 1])
            dwdy_o = (vel[i, j, k, 2] - vel[i, j - 1, k, 2]) / (x2[j] - x2[j - 1])
        else:
            dudy_o = (vel[i, j + 1, k, 0] - vel[i, j, k, 0]) / (x2[j + 1] - x2[j])
            dwdy_o = (vel[i, j + 1, k, 2] - vel[i, j, k, 2]) / (x2[j + 1] - x2[j])

        if k == nz - 1:
            dudz_o = (vel[i, j, k, 0] - vel[i, j, k - 1, 0]) / (x3[k] - x3[k - 1])
            dvdz_o = (vel[i, j, k, 1] - vel[i, j, k - 1, 1]) / (x3[k] - x3[k - 1])
        else:
            dudz_o = (vel[i, j, k + 1, 0] - vel[i, j, k, 0]) / (x3[k + 1] - x3[k])
            dvdz_o = (vel[i, j, k + 1, 1] - vel[i, j, k, 1]) / (x3[k + 1] - x3[k])

        out_omega[i, j, k, 0] = dwdy_o - dvdz_o
        out_omega[i, j, k, 1] = dwdx_o - dudz_o
        out_omega[i, j, k, 2] = dvdx_o - dudy_o

        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
            out_q[i, j, k] = 0.0
            return

        dx_p = x1[i + 1] - x1[i]
        dx_m = x1[i] - x1[i - 1]
        dy_p = x2[j + 1] - x2[j]
        dy_m = x2[j] - x2[j - 1]
        dz_p = x3[k + 1] - x3[k]
        dz_m = x3[k] - x3[k - 1]

        j11 = ((dx_m / dx_p) * (fluc_vel[i + 1, j, k, 0] - fluc_vel[i, j, k, 0]) + (dx_p / dx_m) * (fluc_vel[i, j, k, 0] - fluc_vel[i - 1, j, k, 0])) / (dx_p + dx_m)
        j21 = ((dx_m / dx_p) * (fluc_vel[i + 1, j, k, 1] - fluc_vel[i, j, k, 1]) + (dx_p / dx_m) * (fluc_vel[i, j, k, 1] - fluc_vel[i - 1, j, k, 1])) / (dx_p + dx_m)
        j31 = ((dx_m / dx_p) * (fluc_vel[i + 1, j, k, 2] - fluc_vel[i, j, k, 2]) + (dx_p / dx_m) * (fluc_vel[i, j, k, 2] - fluc_vel[i - 1, j, k, 2])) / (dx_p + dx_m)

        j12 = ((dy_m / dy_p) * (fluc_vel[i, j + 1, k, 0] - fluc_vel[i, j, k, 0]) + (dy_p / dy_m) * (fluc_vel[i, j, k, 0] - fluc_vel[i, j - 1, k, 0])) / (dy_p + dy_m)
        j22 = ((dy_m / dy_p) * (fluc_vel[i, j + 1, k, 1] - fluc_vel[i, j, k, 1]) + (dy_p / dy_m) * (fluc_vel[i, j, k, 1] - fluc_vel[i, j - 1, k, 1])) / (dy_p + dy_m)
        j32 = ((dy_m / dy_p) * (fluc_vel[i, j + 1, k, 2] - fluc_vel[i, j, k, 2]) + (dy_p / dy_m) * (fluc_vel[i, j, k, 2] - fluc_vel[i, j - 1, k, 2])) / (dy_p + dy_m)

        j13 = ((dz_m / dz_p) * (fluc_vel[i, j, k + 1, 0] - fluc_vel[i, j, k, 0]) + (dz_p / dz_m) * (fluc_vel[i, j, k, 0] - fluc_vel[i, j, k - 1, 0])) / (dz_p + dz_m)
        j23 = ((dz_m / dz_p) * (fluc_vel[i, j, k + 1, 1] - fluc_vel[i, j, k, 1]) + (dz_p / dz_m) * (fluc_vel[i, j, k, 1] - fluc_vel[i, j, k - 1, 1])) / (dz_p + dz_m)
        j33 = ((dz_m / dz_p) * (fluc_vel[i, j, k + 1, 2] - fluc_vel[i, j, k, 2]) + (dz_p / dz_m) * (fluc_vel[i, j, k, 2] - fluc_vel[i, j, k - 1, 2])) / (dz_p + dz_m)

        s12 = 0.5 * (j12 + j21)
        s13 = 0.5 * (j13 + j31)
        s23 = 0.5 * (j23 + j32)
        as1 = j11 * j11 + j22 * j22 + j33 * j33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)

        r12 = 0.5 * (j12 - j21)
        r13 = 0.5 * (j13 - j31)
        r23 = 0.5 * (j23 - j32)
        as2 = -2.0 * (r12 * r12 + r13 * r13 + r23 * r23)

        p_inv = -(j11 + j22 + j33)
        out_q[i, j, k] = 0.5 * (p_inv * p_inv - as1 - as2) - (p_inv * p_inv) / 3.0


def _q_invariant_cpu(
    vel: np.ndarray,
    fluc_vel: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = vel.shape[:3]
    omega = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    q_crit = np.zeros((nx, ny, nz), dtype=np.float64)

    dx_fwd = np.empty(nx, dtype=np.float64)
    dx_fwd[:-1] = x1[1:] - x1[:-1]
    dx_fwd[-1] = x1[-1] - x1[-2]
    dy_fwd = np.empty(ny, dtype=np.float64)
    dy_fwd[:-1] = x2[1:] - x2[:-1]
    dy_fwd[-1] = x2[-1] - x2[-2]
    dz_fwd = np.empty(nz, dtype=np.float64)
    dz_fwd[:-1] = x3[1:] - x3[:-1]
    dz_fwd[-1] = x3[-1] - x3[-2]

    dvdx = np.empty((nx, ny, nz), dtype=np.float64)
    dwdx = np.empty((nx, ny, nz), dtype=np.float64)
    dvdx[:-1] = (vel[1:, :, :, 1] - vel[:-1, :, :, 1]) / dx_fwd[:-1, None, None]
    dvdx[-1] = (vel[-1, :, :, 1] - vel[-2, :, :, 1]) / dx_fwd[-1]
    dwdx[:-1] = (vel[1:, :, :, 2] - vel[:-1, :, :, 2]) / dx_fwd[:-1, None, None]
    dwdx[-1] = (vel[-1, :, :, 2] - vel[-2, :, :, 2]) / dx_fwd[-1]

    dudy = np.empty((nx, ny, nz), dtype=np.float64)
    dwdy = np.empty((nx, ny, nz), dtype=np.float64)
    dudy[:, :-1] = (vel[:, 1:, :, 0] - vel[:, :-1, :, 0]) / dy_fwd[None, :-1, None]
    dudy[:, -1] = (vel[:, -1, :, 0] - vel[:, -2, :, 0]) / dy_fwd[-1]
    dwdy[:, :-1] = (vel[:, 1:, :, 2] - vel[:, :-1, :, 2]) / dy_fwd[None, :-1, None]
    dwdy[:, -1] = (vel[:, -1, :, 2] - vel[:, -2, :, 2]) / dy_fwd[-1]

    dudz = np.empty((nx, ny, nz), dtype=np.float64)
    dvdz = np.empty((nx, ny, nz), dtype=np.float64)
    dudz[:, :, :-1] = (vel[:, :, 1:, 0] - vel[:, :, :-1, 0]) / dz_fwd[None, None, :-1]
    dudz[:, :, -1] = (vel[:, :, -1, 0] - vel[:, :, -2, 0]) / dz_fwd[-1]
    dvdz[:, :, :-1] = (vel[:, :, 1:, 1] - vel[:, :, :-1, 1]) / dz_fwd[None, None, :-1]
    dvdz[:, :, -1] = (vel[:, :, -1, 1] - vel[:, :, -2, 1]) / dz_fwd[-1]

    omega[..., 0] = dwdy - dvdz
    omega[..., 1] = dwdx - dudz
    omega[..., 2] = dvdx - dudy

    dx_p = (x1[2:] - x1[1:-1])[:, None, None]
    dx_m = (x1[1:-1] - x1[:-2])[:, None, None]
    dy_p = (x2[2:] - x2[1:-1])[None, :, None]
    dy_m = (x2[1:-1] - x2[:-2])[None, :, None]
    dz_p = (x3[2:] - x3[1:-1])[None, None, :]
    dz_m = (x3[1:-1] - x3[:-2])[None, None, :]

    fv = fluc_vel
    J = np.empty((nx - 2, ny - 2, nz - 2, 3, 3), dtype=np.float64)
    for c in range(3):
        J[..., c, 0] = (
            (dx_m / dx_p) * (fv[2:, 1:-1, 1:-1, c] - fv[1:-1, 1:-1, 1:-1, c])
            + (dx_p / dx_m) * (fv[1:-1, 1:-1, 1:-1, c] - fv[:-2, 1:-1, 1:-1, c])
        ) / (dx_p + dx_m)
        J[..., c, 1] = (
            (dy_m / dy_p) * (fv[1:-1, 2:, 1:-1, c] - fv[1:-1, 1:-1, 1:-1, c])
            + (dy_p / dy_m) * (fv[1:-1, 1:-1, 1:-1, c] - fv[1:-1, :-2, 1:-1, c])
        ) / (dy_p + dy_m)
        J[..., c, 2] = (
            (dz_m / dz_p) * (fv[1:-1, 1:-1, 2:, c] - fv[1:-1, 1:-1, 1:-1, c])
            + (dz_p / dz_m) * (fv[1:-1, 1:-1, 1:-1, c] - fv[1:-1, 1:-1, :-2, c])
        ) / (dz_p + dz_m)

    S = 0.5 * (J + np.swapaxes(J, -1, -2))
    R = 0.5 * (J - np.swapaxes(J, -1, -2))
    as1 = np.sum(S * S, axis=(-1, -2))
    as2 = -np.sum(R * R, axis=(-1, -2))
    p_inv = -(J[..., 0, 0] + J[..., 1, 1] + J[..., 2, 2])
    q_crit[1:-1, 1:-1, 1:-1] = 0.5 * (p_inv * p_inv - as1 - as2) - (p_inv * p_inv) / 3.0
    return q_crit, omega


def compute_q_invariant(
    snapshot_path: Union[str, Path],
    output_path: Union[str, Path],
    fluc_path: Optional[Union[str, Path]] = None,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Replicates codes1/qInvariant_Omega.f90 (or codes/qInvariant.f90 when fluc_path is None)."""
    snap = _read_3d_tecplot_primitives(snapshot_path)
    x1, x2, x3, var = snap["x1"], snap["x2"], snap["x3"], snap["var"]
    vel = np.ascontiguousarray(var[..., 1:4], dtype=np.float64)

    if fluc_path is not None:
        fluc_snap = _read_3d_tecplot_primitives(fluc_path)
        fluc_vel = np.ascontiguousarray(fluc_snap["var"][..., 1:4], dtype=np.float64)
    else:
        fluc_vel = vel

    nx, ny, nz = vel.shape[:3]
    if use_gpu and CUDA_AVAILABLE:
        d_vel = cuda.to_device(vel)
        d_fvel = cuda.to_device(fluc_vel)
        d_x1 = cuda.to_device(np.ascontiguousarray(x1, dtype=np.float64))
        d_x2 = cuda.to_device(np.ascontiguousarray(x2, dtype=np.float64))
        d_x3 = cuda.to_device(np.ascontiguousarray(x3, dtype=np.float64))
        d_omega = cuda.device_array((nx, ny, nz, 3), dtype=np.float64)
        d_q = cuda.device_array((nx, ny, nz), dtype=np.float64)
        tpb = (8, 8, 8)
        bpg = ((nx + 7) // 8, (ny + 7) // 8, (nz + 7) // 8)
        kernel_q_invariant[bpg, tpb](d_vel, d_fvel, d_x1, d_x2, d_x3, d_omega, d_q)
        cuda.synchronize()
        q_crit = d_q.copy_to_host()
        omega = d_omega.copy_to_host()
    else:
        q_crit, omega = _q_invariant_cpu(vel, fluc_vel, x1, x2, x3)

    out_arr = np.empty((nz, ny, nx, 8), dtype=np.float64)
    for k in range(nz):
        out_arr[k, :, :, 0] = x1[None, :]
        out_arr[k, :, :, 1] = x2[:, None]
        out_arr[k, :, :, 2] = x3[k]
        out_arr[k, :, :, 3] = q_crit[:, :, k].T
        out_arr[k, :, :, 4:7] = omega[:, :, k, :].transpose(1, 0, 2)
        out_arr[k, :, :, 7] = var[:, :, k, 1].T

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    header = (
        'TITLE ="qInvariant"\n'
        'Variables = "x","y","z","Q","Omega_x","Omega_y","Omega_z","u"\n'
        f'ZONE i={nx},j={ny},k={nz},DATAPACKING="POINT"'
    )
    fmt = " %10.6f %10.6f %10.6f " + " ".join(["%22.15e"] * 5)
    np.savetxt(out_p, out_arr.reshape(-1, 8), fmt=fmt, header=header, comments="")
    return {"output_path": str(out_p), "Q": q_crit, "omega": omega}


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 CUDA/Python Post-Processing Suite")
    p.add_argument("--stats-dir", default="output_cuda", help="Directory with live .dat streams")
    p.add_argument("--out-dir", default=None, help="Output directory (defaults to --stats-dir)")
    p.add_argument("--reynolds", type=float, default=3000.0, help="Reynolds number")
    p.add_argument("--mesh", choices=("auto", "cosine", "tanh"), default="tanh", help="Mesh type if mean.dat absent")
    p.add_argument("--tmin", type=float, default=None, help="Minimum time filter")
    p.add_argument("--tmax", type=float, default=None, help="Maximum time filter")
    p.add_argument("--stride", type=int, default=1, help="Time-series stride")
    p.add_argument("--avg", action="store_true", help="Compute temporal averages (mean, rms, corr, tauWallAvg)")
    p.add_argument("--vd-fik", action="store_true", help="Compute Van Driest/Trettel-Larsson (VD.dat) and FIK.dat")
    p.add_argument("--combine", default=None, help="Path to domInfo.txt to combine into --combine-out")
    p.add_argument("--combine-out", default="output_cuda/3D_combined.dat", help="Combined 3D.dat destination")
    p.add_argument("--snapshot", default=None, help="3D snapshot for --fluc or --q-invariant")
    p.add_argument("--fluc", action="store_true", help="Write 3D fluc.dat from --snapshot and mean.dat")
    p.add_argument("--q-invariant", action="store_true", help="Write 3D qInvariant.dat from --snapshot")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU fallback for 3D Q-invariant")
    args = p.parse_args()

    sdir = Path(args.stats_dir)
    odir = Path(args.out_dir) if args.out_dir else sdir

    if args.combine:
        combine_component_checkpoints(args.combine, args.combine_out)

    mom_res = None
    tau_res = None
    if args.avg or args.vd_fik:
        if (sdir / "tauWall.dat").exists():
            tau_res = temporal_average_tauwall(
                sdir, odir, tmin=args.tmin, tmax=args.tmax, stride=args.stride, reynolds=args.reynolds
            )
        mom_res = temporal_average_moments(
            sdir, odir, mesh_type=args.mesh, tmin=args.tmin, tmax=args.tmax, stride=args.stride
        )

    if args.vd_fik and mom_res is not None:
        compute_vd_and_tl(
            mom_res["z"],
            mom_res["mean"],
            reynolds=args.reynolds,
            u_tau=tau_res["u_tau"] if tau_res else None,
            re_tau=tau_res["Re_tau"] if tau_res else None,
            out_path=odir / "VD.dat",
        )
        compute_fik(
            mom_res["z"],
            mom_res["mean"],
            mom_res["rms"],
            mom_res["covar"],
            reynolds=args.reynolds,
            out_path=odir / "FIK.dat",
        )

    if args.snapshot:
        if args.fluc:
            compute_fluc_3d(args.snapshot, odir / "mean.dat", odir / "fluc.dat")
        if args.q_invariant:
            fluc_p = (odir / "fluc.dat") if (odir / "fluc.dat").exists() else None
            compute_q_invariant(args.snapshot, odir / "qInvariant.dat", fluc_path=fluc_p, use_gpu=not args.no_gpu)


if __name__ == "__main__":
    main()
