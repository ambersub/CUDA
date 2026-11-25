"""End-to-end Phase 4 pipeline test: Solver -> Live Stats -> Post-Process -> Validation Plots."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import importlib.util

REPO_ROOT = Path(__file__).resolve().parent.parent
CUDA_SCRIPT = REPO_ROOT / "cuda code.py"
POSTPROC_SCRIPT = REPO_ROOT / "python" / "postprocess_cuda.py"
PLOT_SCRIPT = REPO_ROOT / "python" / "plot_validation.py"


def _cuda_available() -> bool:
    spec = importlib.util.spec_from_file_location("phase4_cuda_code", str(CUDA_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase4_cuda_code"] = mod
    spec.loader.exec_module(mod)
    return bool(mod.cuda.is_available())


class TestPhase4EndToEndPipeline(unittest.TestCase):
    """Verifies the complete Fortran-replicated CUDA/Python simulation & analysis workflow."""

    @unittest.skipUnless(_cuda_available(), "CUDA GPU required")
    def test_solver_to_postprocess_to_validation_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_dir = tmp_path / "stats"
            ckpt_dir = tmp_path / "ckpt"
            plots_dir = tmp_path / "plots"

            param_path = tmp_path / "param.dat"
            param_path.write_text((REPO_ROOT / "param.dat").read_text(encoding="utf-8"), encoding="utf-8")

            # 1. Phase 1+2: Run CUDA solver with live stats & component checkpoints
            proc_solve = subprocess.run(
                [
                    sys.executable,
                    str(CUDA_SCRIPT),
                    "--demo",
                    "--param", str(param_path),
                    "--nx", "16",
                    "--ny", "12",
                    "--nz", "40",
                    "--oc", "0",
                    "--steps", "4",
                    "--diag-freq", "2",
                    "--istat", "2",
                    "--stats-dir", str(stats_dir),
                    "--stats-mode", "standard",
                    "--checkpoint-dir", str(ckpt_dir),
                    "--output-freq", "4",
                    "--checkpoint-components",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_solve.returncode, 0, msg=proc_solve.stdout + proc_solve.stderr)

            # 2. Phase 3: Combine component checkpoints, temporal average, VD/FIK, fluc, Q-invariant
            combined_3d = tmp_path / "3D_combined.dat"
            proc_pp = subprocess.run(
                [
                    sys.executable,
                    str(POSTPROC_SCRIPT),
                    "--stats-dir", str(stats_dir),
                    "--out-dir", str(stats_dir),
                    "--combine", str(ckpt_dir / "domInfo.txt"),
                    "--combine-out", str(combined_3d),
                    "--avg",
                    "--vd-fik",
                    "--snapshot", str(combined_3d),
                    "--fluc",
                    "--q-invariant",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_pp.returncode, 0, msg=proc_pp.stdout + proc_pp.stderr)
            for fname in ("mean.dat", "rms.dat", "corr.dat", "tauWallAvg.dat", "VD.dat", "FIK.dat", "fluc.dat", "qInvariant.dat"):
                self.assertTrue((stats_dir / fname).exists(), msg=f"Missing {fname}")

            # 3. Phase 4: Generate CKM validation PNG plots headlessly from the live + post-processed outputs
            expected_plots = [
                "KE_valid.png",
                "massflow_valid.png",
                "rhobulk_valid.png",
                "tauWall_valid.png",
                "pmean_valid.png",
                "rhomean_valid.png",
                "tmean_valid.png",
                "Trms_valid.png",
            ]
            proc_plot = subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPT),
                    "--data-dir", str(stats_dir),
                    "--solver", "cuda",
                    "--full-time",
                    "--output-dir", str(plots_dir),
                    "--plots", *expected_plots,
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_plot.returncode, 0, msg=proc_plot.stdout + proc_plot.stderr)
            for pname in expected_plots:
                p_file = plots_dir / pname
                self.assertTrue(p_file.exists(), msg=f"Missing plot {pname}")
                self.assertGreater(p_file.stat().st_size, 1024, msg=f"Empty plot {pname}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
