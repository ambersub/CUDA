"""Unit, GPU-vs-CPU, rolling-checkpoint, and compiled Fortran parity tests for Phase 2."""

from __future__ import annotations

import importlib.util
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CUDA_SCRIPT = REPO_ROOT / "cuda code.py"
BENCHMARK_SCRIPT = REPO_ROOT / "python" / "benchmark_solvers.py"
COMPLETE_SCRIPT = REPO_ROOT / "python" / "cudacomplete.py"
DIAG_SCRIPT = REPO_ROOT / "python" / "compute_diagnostics_cuda.py"
FORTRAN_STATS_SRC = Path(__file__).resolve().parent / "fortran_stats_parity.f90"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cuda_mod = _load_module("phase2_cuda_code", CUDA_SCRIPT)
cdc = _load_module("phase2_diag_cuda", DIAG_SCRIPT)
CUDA_AVAILABLE = bool(cuda_mod.cuda.is_available())
GFORTRAN_PATH = shutil.which("gfortran")


def _build_rich_channel_state(
    cfg: cuda_mod.SolverConfig,
    *,
    nonzero_wall: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constructs a periodic smooth channel state for Phase 2 parity and streaming tests."""
    x1, x2, x3, _jac, inv_jac, dx, dy, dxi, *_ = cuda_mod.make_mesh(cfg)
    X, Y, Z = np.meshgrid(x1, x2, x3, indexing="ij")
    rho = 1.0 + 0.05 * np.sin(0.5 * X) * np.cos(1.5 * Y) * np.sin(0.5 * math.pi * Z)
    u = (Z * (2.0 - Z)) * (1.0 + 0.09 * np.sin(X) * np.cos(1.5 * Y))
    if nonzero_wall:
        # Non-zero wall v, w and non-orthogonal y modes exercise dwdx, power_*, and cf_corr.
        v = (0.04 * np.cos(X) + 0.03 * np.sin(0.5 * math.pi * Z)) * (np.sin(X + 1.5 * Y) + 0.5 * np.cos(1.5 * Y))
        w = (0.03 * np.cos(X) + 0.025 * np.sin(math.pi * Z)) * (np.sin(1.5 * Y) + 0.6 * np.cos(1.5 * Y))
    else:
        v = 0.04 * np.sin(0.5 * math.pi * Z) * np.sin(X + 1.5 * Y)
        w = 0.03 * np.sin(math.pi * Z) * np.cos(X) * np.sin(1.5 * Y)
    T = 1.0 + 0.04 * np.sin(0.5 * math.pi * Z)
    E = T + 0.5 * cfg.ggM * (u * u + v * v + w * w)
    P = rho * T / (cuda_mod.GAMMA * cfg.mach * cfg.mach)

    var = np.stack([rho, u, v, w, E, T, P], axis=-1).astype(np.float64)
    mu = (T ** 0.7).astype(np.float64)
    q = np.empty((cfg.nx, cfg.ny, cfg.nz, 5), dtype=np.float64)
    q[..., 0] = rho
    q[..., 1] = rho * u
    q[..., 2] = rho * v
    q[..., 3] = rho * w
    q[..., 4] = rho * E
    return x1, x2, x3, inv_jac, var, mu, q


def _enforce_periodic_halos(arr: np.ndarray) -> np.ndarray:
    """Applies exact interior periodic wrapping on x (0, nx-1) and y (0, ny-1)."""
    out = arr.copy()
    out[0, ...] = out[-2, ...]
    out[-1, ...] = out[1, ...]
    out[:, 0, ...] = out[:, -2, ...]
    out[:, -1, ...] = out[:, 1, ...]
    return out


def _write_tecplot_snapshot(
    path: Path, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, var: np.ndarray, t: float, step: int
) -> None:
    nx, ny, nz = len(x1), len(x2), len(x3)
    out = np.empty((nz, ny, nx, 10), dtype=np.float64)
    for k in range(nz):
        out[k, :, :, 0] = x1[None, :]
        out[k, :, :, 1] = x2[:, None]
        out[k, :, :, 2] = x3[k]
        out[k, :, :, 3:] = var[:, :, k, :].transpose(1, 0, 2)
    header = (
        f'TITLE ="{t:20.10f} {step:9d}"\n'
        f'Variables = "x","y","z","Rho","u","v","w","e","T","p"\n'
        f"ZONE k={nz},j={ny},i={nx},DATAPACKING=POINT"
    )
    fmt = " %10.6f %10.6f %10.6f " + " ".join(["%22.15e"] * 7)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, out.reshape(-1, 10), fmt=fmt, header=header, comments="")


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1.0e-15))


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA GPU required")
class TestPhase2DiagnosticsAndStatsGPUvsCPU(unittest.TestCase):
    """Verifies GPU zero-3D-copy diagnostics/statistics match CPU reference across all tiers."""

    def test_gpu_vs_cpu_live_diagnostics_and_all_spatial_modes(self):
        cfg = cuda_mod.SolverConfig(
            nx=128, ny=10, nz=64, reynolds=3000.0, mach=1.5, dt=1.0e-4, mesh_type="tanh"
        )
        x1, x2, x3, inv_jac, var, mu, _q = _build_rich_channel_state(cfg)
        var = _enforce_periodic_halos(var)
        mu = _enforce_periodic_halos(mu)
        coeffs = cuda_mod.compute_wall_stencil_coeffs(x3)
        dx = float(x1[1] - x1[0])
        dy = float(x2[1] - x2[0])
        dxi = 2.0 / float(cfg.nz - 1)

        d_var = cuda_mod.cuda.to_device(np.ascontiguousarray(var))
        d_mu = cuda_mod.cuda.to_device(np.ascontiguousarray(mu))
        d_inv_jac = cuda_mod.cuda.to_device(np.ascontiguousarray(inv_jac))
        d_low2 = cuda_mod.cuda.to_device(coeffs["low2"])
        d_high2 = cuda_mod.cuda.to_device(coeffs["high2"])
        ws = cdc.LiveStatsWorkspace.allocate(cfg.nx, cfg.ny, cfg.nz, x3)

        # 1. Full stats: GPU with auto wall-shear vs CPU reference
        stats_gpu = cdc.compute_spatial_stats_gpu(
            d_var, d_mu, x3, d_inv_jac, d_low2, d_high2, dx, dy, dxi, cfg.reynolds, ws, mode="full"
        )
        stats_cpu = cdc.compute_spatial_stats_cpu(
            var,
            mu,
            x3,
            inv_jac=inv_jac,
            dx=dx,
            dy=dy,
            dxi=dxi,
            low2=coeffs["low2"],
            high2=coeffs["high2"],
            reynolds=cfg.reynolds,
            mode="full",
        )

        self.assertAlmostEqual(stats_gpu["tau_lower"], stats_cpu["tau_lower"], places=12)
        self.assertAlmostEqual(stats_gpu["tau_upper"], stats_cpu["tau_upper"], places=12)

        for key in (
            "stat1Spc",
            "stat2Spc",
            "corrSpc",
            "Fav_stat1Spc",
            "Fav_stat2Spc",
            "Fav_corrSpc",
            "stat1SpcT",
            "stat2SpcT",
            "corrSpcT",
            "omega1",
            "omega2",
            "cf1",
            "cf2",
            "cf_corr",
        ):
            self.assertLess(_rel_l2(stats_gpu[key], stats_cpu[key]), 1.0e-12, msg=key)

        ex_g, ex_c = stats_gpu["extra"], stats_cpu["extra"]
        for key in ("power_OC", "power_STWSV", "power_uc", "slip", "moment80"):
            self.assertLess(_rel_l2(ex_g[key], ex_c[key]), 1.0e-12, msg=key)
        for key in ("tauWall_Lxt", "xAvg", "TKE_xt", "xz"):
            for idx, (arr_g, arr_c) in enumerate(zip(ex_g[key], ex_c[key])):
                self.assertLess(_rel_l2(arr_g, arr_c), 1.0e-12, msg=f"{key}[{idx}]")
        for q_idx in (70, 80, 90, 100, 110, 120):
            for col_i, (arr_g, arr_c) in enumerate(zip(ex_g["NU"][q_idx], ex_c["NU"][q_idx])):
                self.assertLess(_rel_l2(arr_g, arr_c), 1.0e-12, msg=f"NU[{q_idx}][{col_i}]")

        # 2. Minimal tier omits extended arrays while keeping basic moments.
        stats_min = cdc.compute_spatial_stats_gpu(
            d_var, d_mu, x3, d_inv_jac, d_low2, d_high2, dx, dy, dxi, cfg.reynolds, ws, mode="minimal"
        )
        self.assertNotIn("stat1SpcT", stats_min)
        self.assertNotIn("extra", stats_min)
        self.assertLess(_rel_l2(stats_min["stat1Spc"], stats_cpu["stat1Spc"]), 1.0e-12)

        # 3. Live zero-3D-copy step diagnostics vs snapshot calculate_diagnostics_cuda (GPU & CPU)
        diag_live = cdc.compute_live_step_diagnostics(
            d_var, x2, x3, dy, ws, tau_lower=stats_gpu["tau_lower"], tau_upper=stats_gpu["tau_upper"]
        )
        diag_snap_gpu = cdc.calculate_diagnostics_cuda(
            var[..., 0],
            var[..., 1],
            var[..., 2],
            var[..., 3],
            var[..., 5],
            x3,
            dy,
            reynolds=cfg.reynolds,
            use_gpu=True,
            dx=dx,
        )
        diag_snap_cpu = cdc.calculate_diagnostics_cuda(
            var[..., 0],
            var[..., 1],
            var[..., 2],
            var[..., 3],
            var[..., 5],
            x3,
            dy,
            reynolds=cfg.reynolds,
            use_gpu=False,
            dx=dx,
        )
        for d_ref in (diag_snap_gpu, diag_snap_cpu):
            self.assertAlmostEqual(diag_live["KE"], d_ref["KE"], places=12)
            self.assertAlmostEqual(diag_live["tau_bottom"], d_ref["tau_bottom"], places=12)
            self.assertAlmostEqual(diag_live["tau_top"], d_ref["tau_top"], places=12)
            self.assertAlmostEqual(diag_live["tau_avg"], d_ref["tau_avg"], places=12)
            for k in ("rhobulk", "Ubulk", "massflow"):
                self.assertLess(_rel_l2(diag_live[k], d_ref[k]), 1.0e-12, msg=k)
        expected_fft = [float(var[6, 6, k1b - 1, 1]) for k1b in (16, 36, 52)]
        self.assertLess(_rel_l2(diag_live["fft"], expected_fft), 1.0e-14)


@unittest.skipUnless(CUDA_AVAILABLE and GFORTRAN_PATH, "CUDA GPU and gfortran required")
class TestPhase2FortranStatsParity(unittest.TestCase):
    """Compiled gfortran mod_stats.f90 reference vs CUDA live stats and ASCII outputs."""

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.exe_path = Path(cls._tmp_dir.name) / "fortran_stats_parity.exe"
        subprocess.run(
            [GFORTRAN_PATH, "-O2", str(FORTRAN_STATS_SRC), "-o", str(cls.exe_path)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()

    def test_fortran_mod_stats_and_dat_formatting_parity(self):
        cfg = cuda_mod.SolverConfig(
            nx=128, ny=10, nz=64, reynolds=3000.0, mach=1.5, dt=1.0e-4, mesh_type="tanh"
        )
        x1, x2, x3, inv_jac, var, mu, _q = _build_rich_channel_state(cfg)
        var = _enforce_periodic_halos(var)
        mu = _enforce_periodic_halos(mu)
        coeffs = cuda_mod.compute_wall_stencil_coeffs(x3)
        dx = float(x1[1] - x1[0])
        dy = float(x2[1] - x2[0])
        dxi = 2.0 / float(cfg.nz - 1)
        time_val = 1.25
        planes_1b = np.array([p + 1 for p in cdc.default_monitoring_planes(cfg.nx)], dtype=np.int32)

        in_bin = Path(self._tmp_dir.name) / "stats_in.bin"
        out_bin = Path(self._tmp_dir.name) / "stats_out.bin"
        with open(in_bin, "wb") as f:
            f.write(struct.pack("<3i", cfg.nx, cfg.ny, cfg.nz))
            f.write(struct.pack("<5d", dx, dy, dxi, cfg.reynolds, time_val))
            planes_1b.tofile(f)
            np.asarray(x1, dtype=np.float64).tofile(f)
            np.asarray(x2, dtype=np.float64).tofile(f)
            np.asarray(x3, dtype=np.float64).tofile(f)
            np.asarray(inv_jac, dtype=np.float64).tofile(f)
            np.asarray(coeffs["low2"], dtype=np.float64).tofile(f)
            np.asarray(coeffs["high2"], dtype=np.float64).tofile(f)
            np.ascontiguousarray(var.transpose(3, 2, 1, 0), dtype=np.float64).tofile(f)
            np.ascontiguousarray(mu.transpose(2, 1, 0), dtype=np.float64).tofile(f)

        subprocess.run([str(self.exe_path), str(in_bin), str(out_bin)], check=True)

        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        with open(out_bin, "rb") as f:
            ke_f = struct.unpack("<1d", f.read(8))[0]
            rhobulk_f = np.fromfile(f, dtype=np.float64, count=5)
            ubulk_f = np.fromfile(f, dtype=np.float64, count=5)
            massflow_f = np.fromfile(f, dtype=np.float64, count=5)
            fft_f = np.fromfile(f, dtype=np.float64, count=3)
            tau_l_f, tau_u_f = struct.unpack("<2d", f.read(16))

            def rd(shape):
                return np.fromfile(f, dtype=np.float64, count=int(np.prod(shape))).reshape(shape, order="F")

            f_ref = {
                "stat1Spc": rd((nz, 7)),
                "stat2Spc": rd((nz, 7)),
                "corrSpc": rd((nz, 9)),
                "Fav_stat1Spc": rd((nz, 7)),
                "Fav_stat2Spc": rd((nz, 7)),
                "Fav_corrSpc": rd((nz, 9)),
                "stat1SpcT": rd((nz, 8)),
                "stat2SpcT": rd((nz, 8)),
                "corrSpcT": rd((nz, 12)),
                "omega1": rd((nz, 3)),
                "omega2": rd((nz, 3)),
                "cf1": rd((nz, 7)),
                "cf2": rd((nz, 7)),
                "cf_corr": rd((nz, 2)),
            }

        d_var = cuda_mod.cuda.to_device(np.ascontiguousarray(var))
        d_mu = cuda_mod.cuda.to_device(np.ascontiguousarray(mu))
        d_inv_jac = cuda_mod.cuda.to_device(np.ascontiguousarray(inv_jac))
        d_low2 = cuda_mod.cuda.to_device(coeffs["low2"])
        d_high2 = cuda_mod.cuda.to_device(coeffs["high2"])
        ws = cdc.LiveStatsWorkspace.allocate(nx, ny, nz, x3)

        stats_gpu = cdc.compute_spatial_stats_gpu(
            d_var, d_mu, x3, d_inv_jac, d_low2, d_high2, dx, dy, dxi, cfg.reynolds, ws, mode="full"
        )
        diag_gpu = cdc.compute_live_step_diagnostics(
            d_var, x2, x3, dy, ws, tau_lower=stats_gpu["tau_lower"], tau_upper=stats_gpu["tau_upper"]
        )

        self.assertAlmostEqual(diag_gpu["KE"], ke_f, places=12)
        self.assertAlmostEqual(stats_gpu["tau_lower"], tau_l_f, places=12)
        self.assertAlmostEqual(stats_gpu["tau_upper"], tau_u_f, places=12)
        self.assertLess(_rel_l2(diag_gpu["rhobulk"], rhobulk_f), 1.0e-12)
        self.assertLess(_rel_l2(diag_gpu["Ubulk"], ubulk_f), 1.0e-12)
        self.assertLess(_rel_l2(diag_gpu["massflow"], massflow_f), 1.0e-12)
        self.assertLess(_rel_l2(diag_gpu["fft"], fft_f), 1.0e-14)
        for key, ref_arr in f_ref.items():
            self.assertLess(_rel_l2(stats_gpu[key], ref_arr), 1.0e-12, msg=f"Fortran parity failed on {key}")

        # Verify ASCII F15.7 files match Fortran output values to ~2e-7 rounding tolerance.
        stats_dir = Path(self._tmp_dir.name) / "ascii_stats"
        cdc.append_powspec_record(stats_dir, time_val, diag_gpu["fft"])
        cdc.append_diagnostic_record(stats_dir, time_val, diag_gpu, include_tau=False)
        cdc.append_spatial_stats(stats_dir, time_val, stats_gpu, mode="full")

        expected_rows = {
            "KE.dat": np.array([time_val, ke_f]),
            "FFT.dat": np.concatenate([[time_val], fft_f]),
            "rhobulk.dat": np.concatenate([[time_val], rhobulk_f]),
            "Ubulk.dat": np.concatenate([[time_val], ubulk_f]),
            "massflow.dat": np.concatenate([[time_val], massflow_f]),
            "tauWall.dat": np.array([time_val, tau_l_f, tau_u_f]),
            "moment1st.dat": np.concatenate([[time_val], f_ref["stat1Spc"].ravel()]),
            "moment2nd.dat": np.concatenate([[time_val], f_ref["stat2Spc"].ravel()]),
            "momentJoint.dat": np.concatenate([[time_val], f_ref["corrSpc"].ravel()]),
            "Favre1st.dat": np.concatenate([[time_val], f_ref["Fav_stat1Spc"].ravel()]),
            "Favre2nd.dat": np.concatenate([[time_val], f_ref["Fav_stat2Spc"].ravel()]),
            "FavreJoint.dat": np.concatenate([[time_val], f_ref["Fav_corrSpc"].ravel()]),
            "moment1stT.dat": np.concatenate([[time_val], f_ref["stat1SpcT"].ravel()]),
            "moment2ndT.dat": np.concatenate([[time_val], f_ref["stat2SpcT"].ravel()]),
            "momentJointT.dat": np.concatenate([[time_val], f_ref["corrSpcT"].ravel()]),
            "Omega1st.dat": np.concatenate([[time_val], f_ref["omega1"].ravel()]),
            "Omega2nd.dat": np.concatenate([[time_val], f_ref["omega2"].ravel()]),
            "Cf1st.dat": np.concatenate([[time_val], f_ref["cf1"].ravel()]),
            "Cf2nd.dat": np.concatenate([[time_val], f_ref["cf2"].ravel()]),
            "CfJoint.dat": np.concatenate([[time_val], f_ref["cf_corr"].ravel()]),
        }
        for fname, ref_row in expected_rows.items():
            parsed = np.loadtxt(stats_dir / fname, dtype=np.float64)
            self.assertEqual(parsed.shape, ref_row.shape, msg=fname)
            self.assertLess(np.max(np.abs(parsed - ref_row)), 2.0e-7, msg=fname)

        for full_file in (
            "power_OC.dat",
            "power_STWSV.dat",
            "power.dat",
            "uv_slip.dat",
            "tauWall_Lxt1.dat",
            "tauWall_Lxt2.dat",
            "moment80.dat",
            "momentxP.dat",
            "momentxP2.dat",
            "momentxPJoint.dat",
            "TKE_xt_1st.dat",
            "TKE_xt_2nd.dat",
            "TKE_xt_Joint.dat",
            "TKE_xt_1stT.dat",
            "TKE_xt_2ndT.dat",
            "TKE_xt_JointT.dat",
            "xz/moment1st_xz.dat",
            "xz/moment2nd_xz.dat",
            "xz/momentJoint_xz.dat",
            "NU/moment1st070.dat",
            "NU/moment2nd100.dat",
            "NU/momentJoint120.dat",
        ):
            fpath = stats_dir / full_file
            self.assertTrue(fpath.exists(), msg=f"Missing full-mode output {full_file}")
            vals = np.loadtxt(fpath, dtype=np.float64)
            self.assertTrue(np.all(np.isfinite(vals)), msg=full_file)


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA GPU required")
class TestPhase2StreamingCheckpointsAndRunners(unittest.TestCase):
    """Tests live solver cadences, rolling/component checkpoints, and warmup isolation."""

    def test_live_streaming_cadences_modes_and_fixed_tau(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Standard mode, dynamic_source=True
            stats_std = tmp_path / "std"
            cfg = cuda_mod.SolverConfig(
                nx=16,
                ny=12,
                nz=60,
                reynolds=3000.0,
                mach=1.5,
                dt=1.0e-4,
                mesh_type="tanh",
                wall_order=3,
                max_steps=10,
                diag_frequency=2,
                statistic_frequency=5,
                stats_dir=str(stats_std),
                stats_mode="standard",
            )
            *_, q0 = _build_rich_channel_state(cfg, nonzero_wall=False)
            solver = cuda_mod.CUDASolver(cfg, q0)
            solver.run()

            for diag_name in ("KE.dat", "rhobulk.dat", "Ubulk.dat", "massflow.dat", "FFT.dat"):
                rows = np.atleast_2d(np.loadtxt(stats_std / diag_name, dtype=np.float64))
                self.assertEqual(rows.shape[0], 5, msg=diag_name)
                np.testing.assert_allclose(rows[:, 0], [2e-4, 4e-4, 6e-4, 8e-4, 1e-3], rtol=0, atol=1e-7)

            for stat_name in ("tauWall.dat", "moment1st.dat", "moment1stT.dat", "Omega1st.dat", "Cf1st.dat", "Favre1st.dat"):
                rows = np.atleast_2d(np.loadtxt(stats_std / stat_name, dtype=np.float64))
                self.assertEqual(rows.shape[0], 2, msg=stat_name)
                np.testing.assert_allclose(rows[:, 0], [5e-4, 1e-3], rtol=0, atol=1e-7)

            # Minimal mode omits TKE/Omega/Cf/Favre; dynamic_source=False still measures instantaneous tauWall.
            stats_min = tmp_path / "min"
            cfg_min = cuda_mod.SolverConfig(
                nx=16,
                ny=12,
                nz=60,
                reynolds=3000.0,
                mach=1.5,
                dt=1.0e-4,
                mesh_type="cosine",
                wall_order=2,
                max_steps=4,
                diag_frequency=2,
                statistic_frequency=4,
                dynamic_source=False,
                tau_lower=0.002,
                tau_upper=-0.002,
                stats_dir=str(stats_min),
                stats_mode="minimal",
            )
            solver_min = cuda_mod.CUDASolver(cfg_min, q0)
            solver_min.run()
            self.assertTrue((stats_min / "moment1st.dat").exists())
            tau_row = np.loadtxt(stats_min / "tauWall.dat", dtype=np.float64)
            self.assertNotAlmostEqual(tau_row[1], 0.0)
            self.assertFalse((stats_min / "moment1stT.dat").exists())
            self.assertFalse((stats_min / "Favre1st.dat").exists())

            # stats_mode="none" produces no output files even if stats_dir is set.
            stats_none = tmp_path / "none"
            cfg_none = cuda_mod.SolverConfig(
                nx=16,
                ny=12,
                nz=60,
                max_steps=4,
                diag_frequency=2,
                statistic_frequency=2,
                stats_dir=str(stats_none),
                stats_mode="none",
            )
            cuda_mod.CUDASolver(cfg_none, q0).run()
            self.assertFalse(stats_none.exists())

    def test_rolling_combined_and_component_checkpoints_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 1. Alternating 3D.dat <-> 3D_.dat every 3 steps
            ckpt_dir = tmp_path / "rolling_combined"
            cfg = cuda_mod.SolverConfig(
                nx=12,
                ny=10,
                nz=40,
                reynolds=3000.0,
                mach=1.5,
                dt=1.0e-4,
                mesh_type="cosine",
                wall_order=2,
                enable_oc=0,
                output_frequency=3,
                checkpoint_dir=str(ckpt_dir),
                checkpoint_components=False,
            )
            *_, q0 = _build_rich_channel_state(cfg, nonzero_wall=False)
            solver = cuda_mod.CUDASolver(cfg, q0)

            for _ in range(3):
                solver.step()
            self.assertTrue((ckpt_dir / "3D.dat").exists())
            self.assertFalse((ckpt_dir / "3D_.dat").exists())
            t3, s3 = cuda_mod._parse_tecplot_header(str(ckpt_dir / "3D.dat"))[:2]
            self.assertEqual(s3, 3)

            for _ in range(3):
                solver.step()
            self.assertTrue((ckpt_dir / "3D_.dat").exists())
            t6, s6 = cuda_mod._parse_tecplot_header(str(ckpt_dir / "3D_.dat"))[:2]
            self.assertEqual(s6, 6)
            self.assertAlmostEqual(t6, 6.0e-4)

            for _ in range(3):
                solver.step()
            q9_gpu = solver.download_solution()
            cfg_read = cuda_mod.SolverConfig(nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, mach=cfg.mach)
            q9_disk, t9, s9 = cuda_mod.read_combined_input(str(ckpt_dir / "3D.dat"), cfg_read)
            self.assertEqual(s9, 9)
            self.assertAlmostEqual(t9, 9.0e-4)
            self.assertLess(_rel_l2(q9_disk, q9_gpu), 1.0e-12)

            # 2. Component domInfo.txt + 3Dnnn.dat round-trip via read_component_inputs()
            comp_dir = tmp_path / "rolling_comp"
            cfg_comp = cuda_mod.SolverConfig(
                nx=12,
                ny=10,
                nz=40,
                reynolds=3000.0,
                mach=1.5,
                dt=1.0e-4,
                mesh_type="tanh",
                wall_order=3,
                enable_oc=0,
                output_frequency=4,
                checkpoint_dir=str(comp_dir),
                checkpoint_components=True,
            )
            solver_comp = cuda_mod.CUDASolver(cfg_comp, q0)
            for _ in range(4):
                solver_comp.step()
            self.assertTrue((comp_dir / "domInfo.txt").exists())
            self.assertTrue((comp_dir / "3D001.dat").exists())
            cfg_reload = cuda_mod.SolverConfig(mach=cfg_comp.mach)
            q_comp, t_comp, s_comp = cuda_mod.read_component_inputs(
                str(comp_dir / "domInfo.txt"), cfg_reload
            )
            self.assertEqual((cfg_reload.nx, cfg_reload.ny, cfg_reload.nz), (12, 10, 40))
            self.assertEqual(s_comp, 4)
            self.assertAlmostEqual(t_comp, 4.0e-4)
            self.assertLess(_rel_l2(q_comp, solver_comp.download_solution()), 1.0e-12)

    def test_runner_warmup_isolation_and_cli_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = cuda_mod.SolverConfig(
                nx=12, ny=12, nz=40, reynolds=3000.0, mach=1.5, dt=1.0e-4, mesh_type="cosine"
            )
            x1, x2, x3, _invj, var, _mu, _q = _build_rich_channel_state(cfg, nonzero_wall=False)
            inp_path = tmp_path / "input_3D.dat"
            _write_tecplot_snapshot(inp_path, x1, x2, x3, var, t=5.0, step=52)

            mean_path = tmp_path / "mean.dat"
            prof = np.zeros((cfg.nz, 8), dtype=np.float64)
            prof[:, 0] = x3
            for idx in range(7):
                prof[:, idx + 1] = var[:, :, :, idx].mean(axis=(0, 1))
            np.savetxt(mean_path, prof, fmt="%22.15e")

            param_path = tmp_path / "param.dat"
            param_path.write_text((REPO_ROOT / "param.dat").read_text(encoding="utf-8"), encoding="utf-8")

            for label, script_path in (("benchmark", BENCHMARK_SCRIPT), ("cudacomplete", COMPLETE_SCRIPT)):
                run_dir = tmp_path / label
                stats_dir = run_dir / "stats"
                ckpt_dir = run_dir / "ckpt"
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--steps", "4",
                    "--param", str(param_path),
                    "--input", str(inp_path),
                    "--mean-file", str(mean_path),
                    "--output", str(run_dir / "out" / "3D.dat"),
                    "--no-save",
                    "--no-fluc",
                    "--stats-dir", str(stats_dir),
                    "--stats-mode", "standard",
                    "--diag-freq", "2",
                    "--checkpoint-dir", str(ckpt_dir),
                    "--output-freq", "4",
                    "--rms-output", str(run_dir / "out" / "rms.dat"),
                    "--mean-output", str(run_dir / "out" / "mean.dat"),
                    "--tke-output", str(run_dir / "out" / "tke.dat"),
                ]
                proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, msg=f"{label} failed:\n{proc.stdout}\n{proc.stderr}")

                # Warmup step must NOT append a duplicate row: expect exact 2 diag rows (t=5.0002, 5.0004).
                ke_rows = np.atleast_2d(np.loadtxt(stats_dir / "KE.dat", dtype=np.float64))
                fft_rows = np.atleast_2d(np.loadtxt(stats_dir / "FFT.dat", dtype=np.float64))
                self.assertEqual(ke_rows.shape[0], 2, msg=f"{label} KE rows mismatch")
                self.assertEqual(fft_rows.shape[0], 2, msg=f"{label} FFT rows mismatch")
                np.testing.assert_allclose(ke_rows[:, 0], [5.0002, 5.0004], rtol=0, atol=1e-7)
                self.assertTrue((ckpt_dir / "3D.dat").exists())
                ckpt_t, ckpt_step = cuda_mod._parse_tecplot_header(str(ckpt_dir / "3D.dat"))[:2]
                self.assertEqual(ckpt_step, 56)
                self.assertAlmostEqual(ckpt_t, 5.0004)

            # End-to-end cuda code.py CLI --demo with full stats & component checkpoints
            cli_stats = tmp_path / "cli_stats"
            cli_ckpt = tmp_path / "cli_ckpt"
            proc_cli = subprocess.run(
                [
                    sys.executable,
                    str(CUDA_SCRIPT),
                    "--demo",
                    "--param", str(param_path),
                    "--nx", "16",
                    "--ny", "12",
                    "--nz", "40",
                    "--steps", "4",
                    "--diag-freq", "2",
                    "--istat", "4",
                    "--stats-dir", str(cli_stats),
                    "--stats-mode", "full",
                    "--checkpoint-dir", str(cli_ckpt),
                    "--output-freq", "4",
                    "--checkpoint-components",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_cli.returncode, 0, msg=f"cuda code.py CLI failed:\n{proc_cli.stdout}\n{proc_cli.stderr}")
            self.assertTrue((cli_stats / "power_OC.dat").exists())
            self.assertTrue((cli_stats / "NU" / "moment1st070.dat").exists())
            self.assertTrue((cli_ckpt / "domInfo.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
