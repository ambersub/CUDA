"""Unit and integration tests covering Phase 1 parity fixes."""

from __future__ import annotations

import importlib.util
import math
import os
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
FORTRAN_REF_SRC = Path(__file__).resolve().parent / "fortran_ref_parity.f90"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cuda_mod = _load_module("phase1_cuda_code", CUDA_SCRIPT)
CUDA_AVAILABLE = bool(cuda_mod.cuda.is_available())
GFORTRAN_PATH = shutil.which("gfortran")


def _write_tecplot_3d(
    path: Path,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    var: np.ndarray,
    time_val: float,
    nstep_val: int,
    i_range: tuple[int, int] | None = None,
    j_range: tuple[int, int] | None = None,
    k_range: tuple[int, int] | None = None,
) -> None:
    """Writes a Tecplot POINT file (either full domain or MPI component slice)."""
    nx, ny, nz = len(x1), len(x2), len(x3)
    i0, i1 = i_range or (0, nx)
    j0, j1 = j_range or (0, ny)
    k0, k1 = k_range or (0, nz)
    li, lj, lk = i1 - i0, j1 - j0, k1 - k0

    out = np.empty((lk, lj, li, 10), dtype=np.float64)
    for kk, k in enumerate(range(k0, k1)):
        out[kk, :, :, 0] = x1[None, i0:i1]
        out[kk, :, :, 1] = x2[j0:j1, None]
        out[kk, :, :, 2] = x3[k]
        out[kk, :, :, 3:] = var[i0:i1, j0:j1, k, :].transpose(1, 0, 2)

    header = (
        f'TITLE ="{time_val:20.10f} {nstep_val:9d}"\n'
        f'Variables = "x","y","z","Rho","u","v","w","e","T","p"\n'
        f"ZONE k={lk},j={lj},i={li},DATAPACKING=POINT"
    )
    fmt = " %10.6f %10.6f %10.6f " + " ".join(["%22.15e"] * 7)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, out.reshape(-1, 10), fmt=fmt, header=header, comments="")


def _synthetic_channel_prim(
    cfg: cuda_mod.SolverConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1, x2, x3, *_ = cuda_mod.make_mesh(cfg)
    X, Y, Z = np.meshgrid(x1, x2, x3, indexing="ij")
    rho = 1.0 + 0.05 * np.sin(0.5 * X) * np.cos(1.5 * Y) * np.sin(0.5 * math.pi * Z)
    u = (Z * (2.0 - Z)) * (1.0 + 0.08 * np.sin(X) * np.cos(1.5 * Y))
    v = 0.04 * np.sin(0.5 * math.pi * Z) * np.sin(X + 1.5 * Y)
    w = 0.03 * np.sin(math.pi * Z) * np.cos(X) * np.sin(1.5 * Y)
    T = 1.0 + 0.04 * np.sin(0.5 * math.pi * Z)
    E = T + 0.5 * cfg.ggM * (u * u + v * v + w * w)
    P = rho * T / (cuda_mod.GAMMA * cfg.mach * cfg.mach)

    var = np.stack([rho, u, v, w, E, T, P], axis=-1).astype(np.float64)
    q = np.empty((cfg.nx, cfg.ny, cfg.nz, 5), dtype=np.float64)
    q[..., 0] = rho
    q[..., 1] = rho * u
    q[..., 2] = rho * v
    q[..., 3] = rho * w
    q[..., 4] = rho * E
    return x1, x2, x3, var, q


class TestPhase1ConfigAndMesh(unittest.TestCase):
    """Tests param.dat parsing, CLI precedence, meshes, stencils, and file readers."""

    def test_load_param_dat_v19_repo(self):
        cfg = cuda_mod.SolverConfig()
        inp = cuda_mod.load_param_dat(str(REPO_ROOT / "param.dat"), cfg)
        self.assertEqual(cfg.description, "unc")
        self.assertEqual(cfg.restart_option, 1)
        self.assertEqual(cfg.topo, (1, 1, 4))
        self.assertAlmostEqual(cfg.reynolds, 3000.0)
        self.assertAlmostEqual(cfg.mach, 1.5)
        self.assertAlmostEqual(cfg.dt, 1.0e-4)
        self.assertEqual(cfg.statistic_frequency, 100)
        self.assertAlmostEqual(cfg.peclet_x, 2.0)
        self.assertAlmostEqual(cfg.peclet_y, 1.0)
        self.assertAlmostEqual(cfg.peclet_z, 0.1)
        self.assertEqual(cfg.debug_op, 0)
        self.assertEqual(inp, "../input/3D.dat")

    def test_load_param_dat_v20_control_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "param_v2.dat"
            p.write_text(
                "\n".join(
                    [
                        "Case Description : oc_v",
                        "Restart Option : 2",
                        "Process Topology : 1 2 8",
                        "Reynolds Number : 3000",
                        "Mach Number : 1.5",
                        "Timestep : 0.0002",
                        "Frequency of Calculating Spatial Statistics : 50",
                        "Peclet in x direction : 2.0",
                        "Peclet in y direction : 1.0",
                        "Peclet in z direction : 0.1",
                        "Debug Option : 1",
                        "Relative path of initial solution : ../input/3D.dat",
                        "Kappa : 1.25",
                        "Omega : 0.75",
                        "Amp_v : 0.04",
                        "Beta : 2.0",
                        "Amp_ls : 0.001",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = cuda_mod.SolverConfig()
            cuda_mod.load_param_dat(str(p), cfg)
            self.assertEqual(cfg.description, "oc_v")
            self.assertEqual(cfg.restart_option, 2)
            self.assertEqual(cfg.topo, (1, 2, 8))
            self.assertAlmostEqual(cfg.dt, 2.0e-4)
            self.assertEqual(cfg.statistic_frequency, 50)
            self.assertEqual(cfg.debug_op, 1)
            self.assertAlmostEqual(cfg.actuation_wavenumber, 1.25)
            self.assertAlmostEqual(cfg.actuation_omega, 0.75)
            self.assertAlmostEqual(cfg.actuation_amplitude, 0.04)
            self.assertAlmostEqual(cfg.actuation_beta, 2.0)
            self.assertAlmostEqual(cfg.actuation_amplitude_ls, 0.001)

    def test_cli_overrides_precedence(self):
        cfg, q0, t0, step0, label = cuda_mod.configure_and_load_state(
            param_path=str(REPO_ROOT / "param.dat"),
            demo=True,
            overrides={
                "nx": 12,
                "ny": 12,
                "nz": 40,
                "re": 1500.0,
                "dt": 5.0e-5,
                "oc": 0,
                "amp_v": 0.05,
                "mesh": "cosine",
                "wall_order": 2,
            },
        )
        self.assertEqual(label, "demo")
        self.assertEqual((t0, step0), (0.0, 0))
        self.assertEqual(q0.shape, (12, 12, 40, 5))
        self.assertAlmostEqual(cfg.reynolds, 1500.0)
        self.assertAlmostEqual(cfg.dt, 5.0e-5)
        self.assertEqual(cfg.enable_oc, 0)
        self.assertAlmostEqual(cfg.actuation_amplitude, 0.05)
        self.assertEqual(cfg.mesh_type, "cosine")
        self.assertEqual(cfg.wall_order, 2)
        self.assertTrue(cfg.dynamic_source)

        # Providing fixed tau disables dynamic_source unless explicitly re-enabled
        cfg_fixed, *_ = cuda_mod.configure_and_load_state(
            param_path=str(REPO_ROOT / "param.dat"),
            demo=True,
            overrides={"nx": 12, "ny": 12, "nz": 40, "tau_lower": 0.003, "tau_upper": -0.003},
        )
        self.assertFalse(cfg_fixed.dynamic_source)
        self.assertAlmostEqual(cfg_fixed.tau_lower, 0.003)

    def test_mesh_generation_detection_and_10_tuple_api(self):
        for mesh_kind in ("cosine", "tanh"):
            cfg = cuda_mod.SolverConfig(nx=12, ny=12, nz=64, mesh_type=mesh_kind)
            tup = cuda_mod.make_mesh(cfg)
            self.assertEqual(len(tup), 10)
            x1, x2, x3, jac, inv_jac, dx, dy, dxi, low3, high3 = tup
            self.assertAlmostEqual(x3[0], 0.0)
            self.assertAlmostEqual(x3[-1], 2.0)
            self.assertEqual(cuda_mod.detect_mesh_type(x3), mesh_kind)
            self.assertTrue(np.all(np.isfinite(inv_jac)))
            if mesh_kind == "cosine":
                self.assertAlmostEqual(inv_jac[0], 0.0)
                self.assertAlmostEqual(inv_jac[-1], 0.0)

        # Standalone "auto" defaults to tanh for backward compat
        _, _, z_auto, *_ = cuda_mod.make_mesh(cuda_mod.SolverConfig(nx=12, ny=12, nz=64, mesh_type="auto"))
        self.assertEqual(cuda_mod.detect_mesh_type(z_auto), "tanh")

    def test_wall_stencil_polynomial_exactness(self):
        for mesh_kind in ("cosine", "tanh"):
            x3, _, _ = cuda_mod.compute_z_grid(64, mesh_kind)
            coeffs = cuda_mod.compute_wall_stencil_coeffs(x3)

            # Quadratic: exact for 2nd-order 1st derivs
            f_q = 1.5 - 0.7 * x3 + 2.3 * (x3 ** 2)
            df_q = -0.7 + 4.6 * x3
            self.assertAlmostEqual(float(np.dot(coeffs["low2"], f_q[:3])), df_q[0], places=11)
            self.assertAlmostEqual(float(np.dot(coeffs["high2"], f_q[-1:-4:-1])), df_q[-1], places=11)

            # Cubic: exact for 3rd-order 1st derivs and 2nd-order 4-pt 2nd derivs
            f_c = 1.0 - 2.0 * x3 + 3.5 * (x3 ** 2) - 0.8 * (x3 ** 3)
            df_c = -2.0 + 7.0 * x3 - 2.4 * (x3 ** 2)
            d2f_c = 7.0 - 4.8 * x3
            self.assertAlmostEqual(float(np.dot(coeffs["low3"], f_c[:4])), df_c[0], places=10)
            self.assertAlmostEqual(float(np.dot(coeffs["high3"], f_c[-1:-5:-1])), df_c[-1], places=10)
            self.assertAlmostEqual(float(np.dot(coeffs["low2_d2"], f_c[:4])), d2f_c[0], places=7)
            self.assertAlmostEqual(float(np.dot(coeffs["high2_d2"], f_c[-1:-5:-1])), d2f_c[-1], places=7)

    def test_write_mesh_tecplot_and_restart_readers_resop1_resop2(self):
        for mesh_kind in ("cosine", "tanh"):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                cfg = cuda_mod.SolverConfig(nx=10, ny=10, nz=24, mach=1.5, mesh_type=mesh_kind)
                x1, x2, x3, var, _ = _synthetic_channel_prim(cfg)

                mesh_file = tmp_path / "Mesh.dat"
                cuda_mod.write_mesh_tecplot(str(mesh_file), x1, x2, x3)
                lines = mesh_file.read_text(encoding="utf-8").splitlines()
                self.assertIn("k=24,j=10,i=10", lines[2])
                self.assertEqual(len(lines) - 3, 10 * 10 * 24)

                # Write combined resOp=1 file
                comb_file = tmp_path / "3D.dat"
                _write_tecplot_3d(comb_file, x1, x2, x3, var, time_val=12.5, nstep_val=42)

                # Write 2 partitioned z-ranks + domInfo.txt for resOp=2
                dom_file = tmp_path / "output" / "domInfo.txt"
                dom_file.parent.mkdir(parents=True, exist_ok=True)
                dom_file.write_text("  10  10  24\n  2\n  1 10  1 10  1 12\n  1 10  1 10 13 24\n", encoding="utf-8")
                _write_tecplot_3d(tmp_path / "output" / "3D001.dat", x1, x2, x3, var, 12.5, 42, k_range=(0, 12))
                _write_tecplot_3d(tmp_path / "output" / "3D002.dat", x1, x2, x3, var, 12.5, 42, k_range=(12, 24))

                cfg1 = cuda_mod.SolverConfig(mesh_type="auto")
                q1, t1, s1 = cuda_mod.read_combined_input(str(comb_file), cfg1)

                cfg2 = cuda_mod.SolverConfig(mesh_type="auto")
                q2, t2, s2 = cuda_mod.read_component_inputs(str(dom_file), cfg2)

                self.assertEqual(cfg1.mesh_type, mesh_kind)
                self.assertEqual(cfg2.mesh_type, mesh_kind)
                self.assertAlmostEqual(t1, 12.5)
                self.assertAlmostEqual(t2, 12.5)
                self.assertEqual((s1, s2), (42, 42))
                self.assertEqual(q1.shape, (10, 10, 24, 5))
                self.assertLess(float(np.max(np.abs(q1 - q2))), 1.0e-12)


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA GPU required")
class TestPhase1CUDASolverBehavior(unittest.TestCase):
    """Tests CUDA kernels, wall BCs/OC, Peclet scaling, blown check, and runner scripts."""

    def test_cosine_2nd_order_no_oc_halos_and_finite(self):
        cfg = cuda_mod.SolverConfig(
            nx=12,
            ny=12,
            nz=60,
            reynolds=3000.0,
            mach=1.5,
            dt=1.0e-4,
            mesh_type="cosine",
            wall_order=2,
            enable_oc=0,
            actuation_amplitude=0.0,
            statistic_frequency=5,
        )
        *_, q0 = _synthetic_channel_prim(cfg)
        solver = cuda_mod.CUDASolver(cfg, q0)
        for _ in range(6):
            solver.step()
        cuda_mod.cuda.synchronize()
        var = solver.download_primitives()
        self.assertTrue(np.all(np.isfinite(var)))
        self.assertAlmostEqual(solver.time_value, 6.0e-4, places=12)
        # Walls stay non-slip isothermal with zero w
        for kw in (0, -1):
            self.assertAlmostEqual(float(np.max(np.abs(var[1:-1, 1:-1, kw, 1]))), 0.0)
            self.assertAlmostEqual(float(np.max(np.abs(var[1:-1, 1:-1, kw, 2]))), 0.0)
            self.assertAlmostEqual(float(np.max(np.abs(var[1:-1, 1:-1, kw, 3]))), 0.0)
            self.assertTrue(np.allclose(var[1:-1, 1:-1, kw, 5], 1.0))
        # Periodic halos match opposite interior columns along interior rows
        self.assertTrue(np.allclose(var[0, 1:-1, :, :], var[-2, 1:-1, :, :]))
        self.assertTrue(np.allclose(var[-1, 1:-1, :, :], var[1, 1:-1, :, :]))
        self.assertTrue(np.allclose(var[1:-1, 0, :, :], var[1:-1, -2, :, :]))
        self.assertTrue(np.allclose(var[1:-1, -1, :, :], var[1:-1, 1, :, :]))

    def test_oc_sensor_indices_and_decoupled_amp_v(self):
        cfg = cuda_mod.SolverConfig(
            nx=12,
            ny=12,
            nz=60,
            reynolds=3000.0,
            mach=1.5,
            dt=1.0e-4,
            opposition_distance=15,
            actuation_wavenumber=1.0,
            actuation_omega=0.5,
            actuation_amplitude=0.05,
            enable_oc=1,
            mesh_type="tanh",
            wall_order=3,
        )
        *_, q0 = _synthetic_channel_prim(cfg)
        solver = cuda_mod.CUDASolver(cfg, q0)
        solver.step()
        cuda_mod.cuda.synchronize()
        var = solver.download_primitives()
        q = solver.download_solution()

        # OC must not be wiped to 0 and must match exact sensor planes y_d-1 and nz-y_d
        self.assertGreater(float(np.max(np.abs(var[1:-1, 1:-1, 0, 3]))), 1.0e-4)
        self.assertTrue(np.allclose(q[1:-1, 1:-1, 0, 3], -q[1:-1, 1:-1, 14, 3]))
        self.assertTrue(np.allclose(q[1:-1, 1:-1, -1, 3], -q[1:-1, 1:-1, 45, 3]))

        # Expected spanwise wall oscillation uses step-updated time (t = dt after step 1)
        expected_v = 0.05 * np.sin(1.0 * solver.x1_h[1:-1] - 0.5 * solver.time_value)
        for j in range(1, cfg.ny - 1):
            self.assertTrue(np.allclose(var[1:-1, j, 0, 2], expected_v))
            self.assertTrue(np.allclose(var[1:-1, j, -1, 2], expected_v))

        # Disable OC while keeping Amp_v non-zero: v actuation remains active!
        cfg_no_oc = cuda_mod.SolverConfig(
            nx=12,
            ny=12,
            nz=60,
            reynolds=3000.0,
            mach=1.5,
            dt=1.0e-4,
            actuation_wavenumber=1.0,
            actuation_omega=0.5,
            actuation_amplitude=0.05,
            enable_oc=0,
            mesh_type="tanh",
            wall_order=3,
        )
        solver2 = cuda_mod.CUDASolver(cfg_no_oc, q0)
        solver2.step()
        cuda_mod.cuda.synchronize()
        var2 = solver2.download_primitives()
        self.assertAlmostEqual(float(np.max(np.abs(var2[:, :, 0, 3]))), 0.0)
        self.assertGreater(float(np.max(np.abs(var2[1:-1, 1:-1, 0, 2]))), 1.0e-3)

    def test_convective_peclet_re_jac_and_wall_skip(self):
        # Verify Re enters Pe in x and z, Pe_z scales by jac[k] (not inv_jac[k]),
        # and wall planes k=0 and k=nz-1 are untouched by convective_kernel.
        cfg_low = cuda_mod.SolverConfig(
            nx=12,
            ny=12,
            nz=40,
            reynolds=1.0,
            mach=1.5,
            peclet_x=10.0,
            peclet_y=10.0,
            peclet_z=0.5,
            mesh_type="cosine",
            wall_order=2,
            enable_oc=0,
        )
        cfg_hi = cuda_mod.SolverConfig(
            nx=12,
            ny=12,
            nz=40,
            reynolds=5000.0,
            mach=1.5,
            peclet_x=10.0,
            peclet_y=10.0,
            peclet_z=0.5,
            mesh_type="cosine",
            wall_order=2,
            enable_oc=0,
        )
        *_, q0 = _synthetic_channel_prim(cfg_low)
        # Ensure strong non-zero w in interior so Pe_z clearly crosses 0.5 at Re=5000
        x1, x2, x3, jac, inv_jac, dx, dy, dxi, *_ = cuda_mod.make_mesh(cfg_low)
        X, Y, Z = np.meshgrid(x1, x2, x3, indexing="ij")
        w_test = 0.2 * np.sin(0.5 * math.pi * Z) * (1.0 + 0.3 * np.cos(X)) * (1.0 + 0.2 * np.sin(1.5 * Y))
        q0[..., 3] = q0[..., 0] * w_test

        s_low = cuda_mod.CUDASolver(cfg_low, q0)
        s_hi = cuda_mod.CUDASolver(cfg_hi, q0)

        s_low.d_F.copy_to_device(np.zeros_like(s_low.d_F.copy_to_host()))
        s_low.d_G.copy_to_device(np.zeros_like(s_low.d_G.copy_to_host()))
        s_low.d_H.copy_to_device(np.zeros_like(s_low.d_H.copy_to_host()))
        s_hi.d_F.copy_to_device(np.zeros_like(s_hi.d_F.copy_to_host()))
        s_hi.d_H.copy_to_device(np.zeros_like(s_hi.d_H.copy_to_host()))
        s_low.convective()
        s_hi.convective()
        cuda_mod.cuda.synchronize()

        f_low, f_hi = s_low.d_F.copy_to_host(), s_hi.d_F.copy_to_host()
        h_low, h_hi = s_low.d_H.copy_to_host(), s_hi.d_H.copy_to_host()

        # Wall planes k=0 and k=nz-1 remain untouched by convective kernel
        self.assertEqual(float(np.max(np.abs(f_low[:, :, 0, :]))), 0.0)
        self.assertEqual(float(np.max(np.abs(h_low[:, :, -1, :]))), 0.0)
        # Interior fluxes differ because Re flips Pe between central and upwind
        self.assertGreater(float(np.max(np.abs(f_hi[2:-2, 2:-2, 2:-2, :] - f_low[2:-2, 2:-2, 2:-2, :]))), 1.0e-5)
        self.assertGreater(float(np.max(np.abs(h_hi[2:-2, 2:-2, 3:-3, :] - h_low[2:-2, 2:-2, 3:-3, :]))), 1.0e-5)

        # Explicitly verify jac[k] vs inv_jac[k] at near-wall plane k=2 (cosine jac[2] ~ 0.25, inv_jac[2] ~ 4.0):
        # choose Pez threshold strictly between (Re * |w| * dxi * jac[2]) and (Re * |w| * dxi * inv_jac[2])
        q_h = s_hi.download_solution()
        mu_h = s_hi.d_mu.copy_to_host()
        k_probe = 2
        pe_jac = cfg_hi.reynolds * np.abs(q_h[2:-2, 2:-2, k_probe, 3]) * dxi * jac[k_probe] / mu_h[2:-2, 2:-2, k_probe]
        pe_inv = cfg_hi.reynolds * np.abs(q_h[2:-2, 2:-2, k_probe, 3]) * dxi * inv_jac[k_probe] / mu_h[2:-2, 2:-2, k_probe]
        max_pe_jac = float(np.max(pe_jac))
        min_pe_inv = float(np.min(pe_inv))
        self.assertLess(max_pe_jac, min_pe_inv)
        cfg_jac_check = cuda_mod.SolverConfig(
            nx=12, ny=12, nz=40, reynolds=cfg_hi.reynolds, mach=1.5,
            peclet_x=10.0, peclet_y=10.0, peclet_z=0.5 * (max_pe_jac + min_pe_inv),
            mesh_type="cosine", wall_order=2, enable_oc=0,
        )
        s_jac = cuda_mod.CUDASolver(cfg_jac_check, q0)
        s_jac.convective()
        cuda_mod.cuda.synchronize()
        h_jac = s_jac.d_H.copy_to_host()
        # With correct * jac[k] Peclet, k=4 uses 4th-order central matching cfg_low (central)
        self.assertLess(
            float(np.max(np.abs(h_jac[2:-2, 2:-2, k_probe, :] - h_low[2:-2, 2:-2, k_probe, :]))),
            1.0e-12,
        )

    def test_defer_source_and_check_blown_dump(self):
        cfg = cuda_mod.SolverConfig(
            nx=12,
            ny=12,
            nz=40,
            reynolds=3000.0,
            mach=1.5,
            mesh_type="cosine",
            wall_order=2,
            enable_oc=0,
            dynamic_source=True,
            defer_source=True,
            statistic_frequency=3,
        )
        *_, q0 = _synthetic_channel_prim(cfg)
        solver = cuda_mod.CUDASolver(cfg, q0)
        self.assertAlmostEqual(cfg.tau_lower, 0.0)
        solver.step()
        self.assertAlmostEqual(cfg.tau_lower, 0.0)
        solver.step()
        solver.step()  # nstep == 3 -> triggers update_source_terms()
        self.assertNotAlmostEqual(cfg.tau_lower, 0.0)

        # Verify blown detection for negative temperature and NaN, dumping debug/3D001.dat in temp cwd
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                var_bad = solver.download_primitives()
                var_bad[3, 3, 5, 5] = -0.1
                var_bad[4, 4, 5, 0] = np.nan
                solver.d_var.copy_to_device(var_bad)
                with self.assertRaises(FloatingPointError):
                    solver.check_blown()
                self.assertTrue((Path(tmp) / "debug" / "3D001.dat").exists())
            finally:
                os.chdir(old_cwd)

    def test_benchmark_and_cudacomplete_cli_end_to_end(self):
        """Exercises both runner scripts end-to-end on small synthetic Tecplot inputs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = cuda_mod.SolverConfig(
                nx=12, ny=12, nz=40, reynolds=3000.0, mach=1.5, dt=1.0e-4, mesh_type="cosine"
            )
            x1, x2, x3, var, _ = _synthetic_channel_prim(cfg)

            inp_path = tmp_path / "input_3D.dat"
            _write_tecplot_3d(inp_path, x1, x2, x3, var, time_val=10.0, nstep_val=100)

            mean_path = tmp_path / "mean.dat"
            prof = np.zeros((cfg.nz, 8), dtype=np.float64)
            prof[:, 0] = x3
            for idx in range(7):
                prof[:, idx + 1] = var[:, :, :, idx].mean(axis=(0, 1))
            np.savetxt(mean_path, prof, fmt="%22.15e")

            param_path = tmp_path / "param.dat"
            param_path.write_text((REPO_ROOT / "param.dat").read_text(encoding="utf-8"), encoding="utf-8")

            for label, script_path in (("cudacomplete", COMPLETE_SCRIPT), ("benchmark", BENCHMARK_SCRIPT)):
                out_dir = tmp_path / label
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--steps", "2",
                    "--param", str(param_path),
                    "--input", str(inp_path),
                    "--mean-file", str(mean_path),
                    "--mesh", "auto",
                    "--wall-order", "2",
                    "--oc", "0",
                    "--no-fluc",
                    "--output", str(out_dir / "3D.dat"),
                    "--rms-output", str(out_dir / "rms.dat"),
                    "--mean-output", str(out_dir / "mean.dat"),
                    "--tke-output", str(out_dir / "tke.dat"),
                ]
                proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, msg=f"{label} failed:\n{proc.stdout}\n{proc.stderr}")
                self.assertIn("Auto-detected mesh stretching: cosine", proc.stdout)
                self.assertTrue((out_dir / "3D.dat").exists())
                self.assertTrue((out_dir / "tauWall.dat").exists())
                self.assertTrue((out_dir / "benchmark_results.json").exists())


@unittest.skipUnless(CUDA_AVAILABLE and GFORTRAN_PATH, "CUDA GPU and gfortran required")
class TestPhase1FortranParity(unittest.TestCase):
    """Compiled Fortran 90 vs CUDA numerical equivalence across both codebases."""

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.exe_path = Path(cls._tmp_dir.name) / "fortran_ref_parity.exe"
        subprocess.run(
            [GFORTRAN_PATH, "-O2", str(FORTRAN_REF_SRC), "-o", str(cls.exe_path)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()

    def _run_parity_case(
        self,
        *,
        mesh: str,
        wall_order: int,
        oc: int,
        amp_v: float,
        kappa: float,
        omega: float,
        y_d: int,
        istat: int,
        nsteps: int,
        defer_source: bool,
        reynolds: float,
    ):
        bin_path = Path(self._tmp_dir.name) / f"{mesh}_{wall_order}_{oc}_{reynolds}.bin"
        subprocess.run(
            [
                str(self.exe_path),
                "0" if mesh == "cosine" else "1",
                str(wall_order),
                str(oc),
                str(amp_v),
                str(kappa),
                str(omega),
                str(y_d),
                str(istat),
                str(nsteps),
                "1" if defer_source else "0",
                str(reynolds),
                str(bin_path),
            ],
            check=True,
        )
        with open(bin_path, "rb") as f:
            tau_l_f, tau_u_f, t_f = struct.unpack("<3d", f.read(24))
            var_f = np.fromfile(f, dtype=np.float64).reshape((80, 16, 16, 7)).transpose(2, 1, 0, 3)

        cfg = cuda_mod.SolverConfig(
            nx=16,
            ny=16,
            nz=80,
            reynolds=reynolds,
            mach=1.5,
            dt=1.0e-4,
            peclet_x=2.0,
            peclet_y=1.0,
            peclet_z=0.1,
            opposition_distance=y_d,
            actuation_wavenumber=kappa,
            actuation_omega=omega,
            actuation_amplitude=amp_v,
            enable_oc=oc,
            mesh_type=mesh,
            wall_order=wall_order,
            dynamic_source=True,
            defer_source=defer_source,
            max_steps=nsteps,
            statistic_frequency=istat,
        )
        *_, q0 = _synthetic_channel_prim(cfg)
        solver = cuda_mod.CUDASolver(cfg, q0)
        for _ in range(nsteps):
            solver.step()
        cuda_mod.cuda.synchronize()
        var_g = solver.download_primitives()

        self.assertAlmostEqual(solver.time_value, t_f, places=14)
        self.assertAlmostEqual(cfg.tau_lower, tau_l_f, places=12)
        self.assertAlmostEqual(cfg.tau_upper, tau_u_f, places=12)

        mask = np.ones((16, 16, 80), dtype=bool)
        for ci in (0, 15):
            for cj in (0, 15):
                mask[ci, cj, :] = False

        for idx in range(7):
            diff = var_g[..., idx][mask] - var_f[..., idx][mask]
            ref = var_f[..., idx][mask]
            rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1.0e-15))
            self.assertLess(rel_l2, 1.0e-12, msg=f"Field {idx} parity error RelL2={rel_l2}")

    def test_codes_v19_cosine_2nd_order_uncontrolled(self):
        self._run_parity_case(
            mesh="cosine",
            wall_order=2,
            oc=0,
            amp_v=0.0,
            kappa=1.0,
            omega=0.5,
            y_d=17,
            istat=5,
            nsteps=10,
            defer_source=False,
            reynolds=3000.0,
        )

    def test_codes1_v20_tanh_3rd_order_oc_and_actuation(self):
        self._run_parity_case(
            mesh="tanh",
            wall_order=3,
            oc=1,
            amp_v=0.05,
            kappa=1.0,
            omega=0.5,
            y_d=17,
            istat=5,
            nsteps=10,
            defer_source=False,
            reynolds=3000.0,
        )

    def test_codes1_low_re_peclet_upwind_and_deferred_source(self):
        self._run_parity_case(
            mesh="tanh",
            wall_order=3,
            oc=1,
            amp_v=0.05,
            kappa=1.0,
            omega=0.5,
            y_d=17,
            istat=4,
            nsteps=8,
            defer_source=True,
            reynolds=100.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
