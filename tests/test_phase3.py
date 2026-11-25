"""Unit and GPU-vs-CPU parity tests for Phase 3 post-processing suite."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
POSTPROC_SCRIPT = REPO_ROOT / "python" / "postprocess_cuda.py"
CUDA_SCRIPT = REPO_ROOT / "cuda code.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pp = _load_module("phase3_postproc", POSTPROC_SCRIPT)
cuda_mod = _load_module("phase3_cuda_code", CUDA_SCRIPT)
CUDA_AVAILABLE = bool(cuda_mod.cuda.is_available())


class TestPhase3PostProcessing(unittest.TestCase):
    """Covers avg_v1.7, avgTauWall, avg_VD, avg_FIK, combine, fluc, and qInvariant."""

    def test_restart_dedup_temporal_avg_tauwall_vd_and_fik(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nz = 24
            z, _, _ = cuda_mod.compute_z_grid(nz, "tanh")

            # Construct synthetic multi-restart moment files with overlapping backward times
            # Times: [1.0, 2.0, 3.0,  2.0(dup), 3.0(dup), 4.0] -> clean: [1.0, 2.0, 3.0, 4.0]
            times = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
            clean_times = [1.0, 2.0, 3.0, 4.0]

            rng = np.random.default_rng(42)
            # Instantaneous samples at clean times (4, nz, 7)
            samples = np.zeros((4, nz, 7), dtype=np.float64)
            u_base = z * (2.0 - z)
            for idx in range(4):
                samples[idx, :, 0] = 1.0 + 0.02 * (idx - 1.5)
                samples[idx, :, 1] = u_base * (1.0 + 0.05 * (idx - 1.5))
                samples[idx, :, 2] = 0.01 * (idx - 1.5)
                samples[idx, :, 3] = 0.02 * np.sin(math.pi * z) * (1.0 if idx % 2 == 0 else -1.0)
                samples[idx, :, 4] = 1.0 + 0.2 * samples[idx, :, 1] ** 2
                samples[idx, :, 5] = 1.0 + 0.1 * (1.0 - (z - 1.0) ** 2)
                samples[idx, :, 6] = samples[idx, :, 0] * samples[idx, :, 5] / (1.4 * 2.25)

            # Duplicate rows with perturbed values so deduplication correctness is provable
            dup_noise = rng.normal(scale=10.0, size=(nz, 9))

            def write_series(fname: str, rows_by_clean_idx):
                with (tmp_path / fname).open("w", encoding="utf-8") as f:
                    map_idx = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3}
                    seen = set()
                    for t in times:
                        arr = rows_by_clean_idx[map_idx[t]]
                        if t in seen:
                            arr = arr + dup_noise[: arr.shape[0], : arr.shape[1]] if arr.ndim == 2 else arr + 99.0
                        seen.add(t)
                        flat = np.asarray(arr, dtype=np.float64).ravel()
                        f.write(f" {t:15.7f}")
                        for val in flat:
                            f.write(f" {float(val):15.7f}")
                        f.write("\n")

            write_series("moment1st.dat", samples)
            write_series("moment2nd.dat", samples ** 2)
            joint_samples = np.stack(
                [samples[..., a] * samples[..., b] for a, b in pp._JOINT_PAIR_INDICES], axis=-1
            )
            write_series("momentJoint.dat", joint_samples)

            tau_rows = np.array([[0.002, -0.002], [0.0022, -0.0022], [0.0018, -0.0018], [0.002, -0.002]])
            write_series("tauWall.dat", tau_rows)

            tau_res = pp.temporal_average_tauwall(tmp_path, reynolds=3000.0)
            self.assertEqual(tau_res["n_samples"], 4)
            self.assertAlmostEqual(tau_res["tau_mean"], 0.002, places=6)
            self.assertTrue((tmp_path / "tauWallAvg.dat").exists())

            mom_res = pp.temporal_average_moments(tmp_path, x3=z)
            self.assertEqual(mom_res["n_samples"], 4)
            expected_mean = np.mean(samples, axis=0)
            expected_rms = np.sqrt(np.maximum(np.mean(samples ** 2, axis=0) - expected_mean ** 2, 0.0))
            np.testing.assert_allclose(mom_res["mean"], expected_mean, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(mom_res["rms"] ** 2, expected_rms ** 2, rtol=1e-5, atol=2e-7)
            self.assertTrue((tmp_path / "mean.dat").exists())
            self.assertTrue((tmp_path / "rms.dat").exists())
            self.assertTrue((tmp_path / "corr.dat").exists())
            self.assertTrue((tmp_path / "AoNDT.dat").exists())

            vd_res = pp.compute_vd_and_tl(
                z,
                mom_res["mean"],
                reynolds=3000.0,
                u_tau=tau_res["u_tau"],
                re_tau=tau_res["Re_tau"],
                out_path=tmp_path / "VD.dat",
            )
            self.assertTrue(np.all(np.diff(vd_res["u_VD_plus"]) >= 0.0))
            self.assertTrue(np.all(np.isfinite(vd_res["u_TL_plus"])))
            self.assertTrue((tmp_path / "VD.dat").exists())

            fik_res = pp.compute_fik(
                z,
                mom_res["mean"],
                mom_res["rms"],
                mom_res["covar"],
                reynolds=3000.0,
                out_path=tmp_path / "FIK.dat",
            )
            self.assertEqual(fik_res["table"].shape, (nz // 2, 7))
            self.assertTrue(np.all(np.isfinite(fik_res["table"])))
            self.assertTrue((tmp_path / "FIK.dat").exists())

    def test_combine_fluc_and_q_invariant_gpu_vs_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = cuda_mod.SolverConfig(
                nx=12, ny=10, nz=24, reynolds=3000.0, mach=1.5, dt=1.0e-4, mesh_type="tanh"
            )
            x1, x2, x3, *_ = cuda_mod.make_mesh(cfg)
            X, Y, Z = np.meshgrid(x1, x2, x3, indexing="ij")

            # Solid-body rotation around z-axis in interior: u = -y, v = x, w = 0 -> omega_z = 2, Q > 0
            rho = np.ones_like(X)
            u = -Y + 0.1 * np.sin(X) * Z
            v = X + 0.1 * np.cos(Y) * Z
            w = 0.05 * np.sin(math.pi * Z) * np.cos(X)
            T = np.ones_like(X)
            E = T + 0.5 * cfg.ggM * (u * u + v * v + w * w)
            P = rho * T / (cuda_mod.GAMMA * cfg.mach * cfg.mach)
            var = np.stack([rho, u, v, w, E, T, P], axis=-1)

            # Write component checkpoint and combine via postprocess_cuda.combine_component_checkpoints
            comp_dir = tmp_path / "comp"
            dom_path = cuda_mod._write_component_checkpoint(
                str(comp_dir), x1, x2, x3, var, time_val=2.5, nstep_val=25
            )
            combined_path = tmp_path / "3D_combined.dat"
            pp.combine_component_checkpoints(dom_path, combined_path)
            self.assertTrue(combined_path.exists())

            snap = pp._read_3d_tecplot_primitives(combined_path)
            np.testing.assert_allclose(snap["var"], var, rtol=1e-12, atol=1e-12)

            # Mean profile & fluc.dat
            mean_arr = np.mean(var, axis=(0, 1))
            mean_path = tmp_path / "mean.dat"
            np.savetxt(mean_path, np.column_stack([x3, mean_arr]), fmt="%22.15e")
            fluc_path = tmp_path / "fluc.dat"
            pp.compute_fluc_3d(combined_path, mean_path, fluc_path)
            fluc_snap = pp._read_3d_tecplot_primitives(fluc_path)
            np.testing.assert_allclose(
                fluc_snap["var"], var - mean_arr[None, None, :, :], rtol=1e-12, atol=1e-12
            )

            # Q-invariant CPU vs GPU (and CLI invocation)
            q_cpu = pp.compute_q_invariant(
                combined_path, tmp_path / "q_cpu.dat", fluc_path=fluc_path, use_gpu=False
            )
            self.assertTrue((tmp_path / "q_cpu.dat").exists())
            if CUDA_AVAILABLE:
                q_gpu = pp.compute_q_invariant(
                    combined_path, tmp_path / "q_gpu.dat", fluc_path=fluc_path, use_gpu=True
                )
                np.testing.assert_allclose(q_gpu["Q"], q_cpu["Q"], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(q_gpu["omega"], q_cpu["omega"], rtol=1e-12, atol=1e-12)

            # CLI smoke test on postprocess_cuda.py
            cli_out = tmp_path / "cli_out"
            cli_out.mkdir()
            np.savetxt(cli_out / "mean.dat", np.column_stack([x3, mean_arr]), fmt="%22.15e")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(POSTPROC_SCRIPT),
                    "--stats-dir", str(cli_out),
                    "--out-dir", str(cli_out),
                    "--snapshot", str(combined_path),
                    "--fluc",
                    "--q-invariant",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"CLI failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertTrue((cli_out / "fluc.dat").exists())
            self.assertTrue((cli_out / "qInvariant.dat").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
