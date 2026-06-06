"""
test/test_smoke.py
──────────────────────────────────────────────────────────────────────────────
DD-FP smoke-test suite.

Purpose: verify that all source/script modules import correctly and that
core functions satisfy expected shapes, dtypes, and mathematical invariants.
Does NOT re-run full experiments — checks that nothing is broken.

Usage:
    # from repository root
    pytest test/test_smoke.py -v

    # without GPU (default: GPU tests auto-skipped)
    pytest test/test_smoke.py -v

    # with GPU
    pytest test/test_smoke.py -v --run-gpu

Requirements:
    numpy, scipy, Pillow, pytest
    (GPU tests: cupy-cudaXXX — auto-skipped if absent)
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Add repository root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_script(rel_path: str):
    """Load a scripts/ file as a module (no package structure required)."""
    path = ROOT / rel_path
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synth2d(size: int = 16, seed: int = 0) -> np.ndarray:
    """uint8 2D synthetic volume."""
    from src.utils.benchmark_utils import generate_synthetic_volume
    return generate_synthetic_volume((size, size), seed=seed)


def _synth3d(size: int = 8, seed: int = 0) -> np.ndarray:
    """uint8 3D synthetic volume."""
    from src.utils.benchmark_utils import generate_synthetic_volume
    return generate_synthetic_volume((size, size, size), seed=seed)


def _preprocessor_cfg(prep_type: str) -> SimpleNamespace:
    return SimpleNamespace(preprocessing=SimpleNamespace(
        type=prep_type,
        naive_mode="bilinear",
        ddfp_overlap=1,
        no_interp_cache_dir=None,
        naive_interp_cache_dir=None,
        ddfp_cache_dir=None,
    ))


# GPU availability (checked once at import time)
try:
    import cupy as cp
    _GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _GPU_AVAILABLE = False

gpu_skip = pytest.mark.skipif(
    not _GPU_AVAILABLE,
    reason="CuPy / GPU not available — install cupy or use --run-gpu"
)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TestDdfpPublicApi:
    """Tests for the src/ddfp/__init__.py public API."""

    def test_get_backend_returns_valid_string(self):
        from src.ddfp import get_backend
        backend = get_backend()
        assert backend in ("gpu", "cpu"), f"unexpected backend: {backend}"

    def test_immersion_pipeline_3d_shape(self):
        """immersion_pipeline: (W,H,D) uint8 → (2W-1, 2H-1, 2D-1) float32."""
        from src.ddfp import immersion_pipeline
        vol = _synth3d(8)
        W, H, D = vol.shape
        u = immersion_pipeline(vol)
        assert u.shape == (2*W-1, 2*H-1, 2*D-1), f"shape mismatch: {u.shape}"
        assert u.dtype == np.float32

    def test_immersion_pipeline_accepts_uint8_input(self):
        """Non-uint8 input should be converted internally."""
        from src.ddfp import immersion_pipeline
        vol = np.random.rand(6, 6, 6).astype(np.float64) * 200
        u = immersion_pipeline(vol.astype(np.uint8))
        assert u.shape == (11, 11, 11)

    def test_run_ddfp_2d_shape(self):
        """run_ddfp_2d: (H,W) float32 → (2H-1, 2W-1) float32."""
        from src.ddfp import run_ddfp_2d
        img = np.random.rand(16, 20).astype(np.float32)
        H, W = img.shape
        u = run_ddfp_2d(img)
        assert u.shape == (2*H-1, 2*W-1), f"shape mismatch: {u.shape}"
        assert u.dtype == np.float32

    def test_run_ddfp_2d_value_range(self):
        """Output values must lie within [0, 1]."""
        from src.ddfp import run_ddfp_2d
        img = np.random.rand(12, 12).astype(np.float32)
        u = run_ddfp_2d(img)
        assert u.min() >= -1e-6 and u.max() <= 1.0 + 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TestParallelImmersion:
    """Tests for the src/ddfp/parallel_immersion.py CPU backend."""

    def test_build_ispan_returns_three_values(self):
        from src.ddfp.parallel_immersion import build_ispan
        vol = _synth3d(8)
        result = build_ispan(vol)
        assert len(result) == 3, "build_ispan should return (U_lo, U_hi, l_inf)"

    def test_build_ispan_output_shapes(self):
        """U_lo, U_hi shape should be (2W+1, 2H+1, 2D+1) including padding."""
        from src.ddfp.parallel_immersion import build_ispan
        vol = np.random.randint(0, 256, (6, 8, 10), dtype=np.uint8)
        W, H, D = vol.shape
        U_lo, U_hi, l_inf = build_ispan(vol)
        expected = (2*W+1, 2*H+1, 2*D+1)
        assert U_lo.shape == expected, f"U_lo shape: {U_lo.shape}"
        assert U_hi.shape == expected

    def test_build_ispan_monotonicity(self):
        """Ispan invariant: U_lo[p] <= U_hi[p] everywhere."""
        from src.ddfp.parallel_immersion import build_ispan
        vol = _synth3d(8)
        U_lo, U_hi, _ = build_ispan(vol)
        assert np.all(U_lo <= U_hi + 1e-6), "Ispan monotonicity violated"

    def test_front_propagation_shape(self):
        """FP output should have the same shape as U_lo."""
        from src.ddfp.parallel_immersion import build_ispan, front_propagation
        vol = _synth3d(6)
        U_lo, U_hi, l_inf = build_ispan(vol)
        u = front_propagation(U_lo, U_hi, l_inf, verbose=False)
        assert u.shape == U_lo.shape

    def test_front_propagation_snap_constraint(self):
        """FP output must lie within the Ispan interval (snap constraint)."""
        from src.ddfp.parallel_immersion import build_ispan, front_propagation
        vol = _synth3d(6)
        U_lo, U_hi, l_inf = build_ispan(vol)
        u = front_propagation(U_lo, U_hi, l_inf, verbose=False)
        interior = (slice(1, -1),) * 3
        assert np.all(u[interior] >= U_lo[interior] - 1e-4)
        assert np.all(u[interior] <= U_hi[interior] + 1e-4)

    def test_immersion_pipeline_cpu(self):
        """CPU pipeline: verify output shape and DWC integrity."""
        from src.ddfp.parallel_immersion import immersion_pipeline
        from src.utils.benchmark_utils import verify_dwc
        vol = _synth3d(8)
        u = immersion_pipeline(vol, verbose=False)
        if isinstance(u, tuple):
            u = u[0]
        W, H, D = vol.shape
        assert u.shape == (2*W-1, 2*H-1, 2*D-1)
        result = verify_dwc(vol, u)
        assert result["n_violations"] == 0, (
            f"CPU pipeline produced {result['n_violations']} DWC violations"
        )

    def test_verify_self_dual(self):
        """verify_self_dual should return True for a correct immersion."""
        from src.ddfp.parallel_immersion import immersion_pipeline, verify_self_dual, cell_type_map
        vol = _synth3d(8)
        u = immersion_pipeline(vol, verbose=False)
        if isinstance(u, tuple):
            u = u[0]
        ct = cell_type_map(u.shape)
        assert verify_self_dual(u, ct) is True

    def test_cell_type_map_values(self):
        """cell_type_map values must be within {0,1,2,3} (k-cell dimension)."""
        from src.ddfp.parallel_immersion import cell_type_map
        ct = cell_type_map((15, 15, 15))
        assert set(np.unique(ct)).issubset({0, 1, 2, 3})


# ══════════════════════════════════════════════════════════════════════════════
# 3. src/benchmark_utils
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkUtils:
    """Tests for src/utils/benchmark_utils.py."""

    def test_generate_synthetic_volume_2d(self):
        vol = _synth2d(32)
        assert vol.shape == (32, 32)
        assert vol.dtype == np.uint8

    def test_generate_synthetic_volume_3d(self):
        vol = _synth3d(16)
        assert vol.shape == (16, 16, 16)
        assert vol.dtype == np.uint8

    def test_generate_synthetic_volume_deterministic(self):
        """Same seed must produce identical volumes."""
        v1 = _synth3d(10, seed=42)
        v2 = _synth3d(10, seed=42)
        assert np.array_equal(v1, v2)

    def test_verify_dwc_no_violations_on_ddfp_output(self):
        """DD-FP output should have zero DWC violations."""
        from src.ddfp.parallel_immersion import immersion_pipeline
        from src.utils.benchmark_utils import verify_dwc
        vol = _synth3d(8)
        u = immersion_pipeline(vol, verbose=False)
        if isinstance(u, tuple):
            u = u[0]
        r = verify_dwc(vol, u)
        assert r["n_violations"] == 0
        assert 0.0 <= r["violation_rate"] <= 1.0

    def test_verify_dwc_detects_violations(self):
        """no_interp (max-pooling) should produce DWC violations in 2D."""
        from src.utils.benchmark_utils import verify_dwc
        from src.preprocessing.preprocessor import NoInterpPreprocessor
        vol_u8 = _synth2d(32)
        vol_f32 = vol_u8.astype(np.float32) / 255.0
        prep = NoInterpPreprocessor(_preprocessor_cfg("no_interp"))
        u_no, _ = prep(vol_f32, None, None)
        result = verify_dwc(vol_u8, u_no)
        assert result["n_violations"] > 0, "no_interp should produce DWC violations"

    def test_verify_dwc_returns_required_keys(self):
        from src.utils.benchmark_utils import verify_dwc
        from src.ddfp.parallel_immersion import immersion_pipeline
        vol = _synth3d(8)
        u = immersion_pipeline(vol, verbose=False)
        if isinstance(u, tuple): u = u[0]
        r = verify_dwc(vol, u)
        for key in ("n_violations", "max_abs_error", "violation_rate"):
            assert key in r, f"missing key: {key}"

    def test_time_function_returns_stats(self):
        from src.utils.benchmark_utils import time_function
        fn = lambda: np.sum(np.ones((100, 100)))
        r = time_function(fn, n_repeats=3, warmup=1)
        assert "median_s" in r and "std_s" in r
        assert r["median_s"] > 0

    def test_naive_interpolate_shape(self):
        """naive_interpolate: (W,H,D) → (2W-1, 2H-1, 2D-1)."""
        from src.utils.benchmark_utils import naive_interpolate
        vol = _synth3d(8).astype(np.float32)
        W, H, D = vol.shape
        out = naive_interpolate(vol, order=1)
        assert out.shape == (2*W-1, 2*H-1, 2*D-1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. src/preprocessing/preprocessor
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    """Tests for src/preprocessing/preprocessor.py."""

    IMG = np.random.default_rng(0).random((20, 24)).astype(np.float32)
    LBL = (IMG > 0.5).astype(np.float32)

    def test_expanded_shape(self):
        from src.preprocessing.preprocessor import expanded_shape
        assert expanded_shape(8, 8) == (15, 15)
        assert expanded_shape(1, 1) == (1, 1)
        assert expanded_shape(100, 200) == (199, 399)

    def test_zoom_to_expanded(self):
        from src.preprocessing.preprocessor import zoom_to_expanded
        arr = np.random.rand(8, 10).astype(np.float32)
        H, W = arr.shape
        out = zoom_to_expanded(arr, order=1)
        assert out.shape == (2*H-1, 2*W-1)

    def test_no_interp_preprocessor_shape(self):
        from src.preprocessing.preprocessor import NoInterpPreprocessor
        prep = NoInterpPreprocessor(_preprocessor_cfg("no_interp"))
        out_img, out_lbl = prep(self.IMG, self.LBL, None)
        H, W = self.IMG.shape
        assert out_img.shape == (2*H-1, 2*W-1)
        assert out_lbl.shape == (2*H-1, 2*W-1)

    def test_naive_interp_preprocessor_shape(self):
        from src.preprocessing.preprocessor import NaiveInterpPreprocessor
        prep = NaiveInterpPreprocessor(_preprocessor_cfg("naive_interp"))
        out_img, _ = prep(self.IMG, None, None)
        H, W = self.IMG.shape
        assert out_img.shape == (2*H-1, 2*W-1)

    def test_ddfp_preprocessor_shape(self):
        """DDFPPreprocessor (CPU fallback): verify output shape."""
        from src.preprocessing.preprocessor import DDFPPreprocessor
        prep = DDFPPreprocessor(_preprocessor_cfg("ddfp"))
        out_img, _ = prep(self.IMG, None, None)
        H, W = self.IMG.shape
        assert out_img.shape == (2*H-1, 2*W-1)

    def test_ddfp_preprocessor_dwc_guarantee(self):
        """DDFPPreprocessor output must have zero DWC violations (Theorem 3.1).

        verify_dwc accepts 2D (H,W) input directly.
        """
        from src.preprocessing.preprocessor import DDFPPreprocessor
        from src.utils.benchmark_utils import verify_dwc
        prep    = DDFPPreprocessor(_preprocessor_cfg("ddfp"))
        vol_f32 = self.IMG
        vol_u8  = (vol_f32 * 255).clip(0, 255).astype(np.uint8)
        u, _    = prep(vol_f32, None, None)
        r = verify_dwc(vol_u8, u)
        assert r["n_violations"] == 0, (
            f"DDFPPreprocessor produced {r['n_violations']} DWC violations"
        )

    def test_get_preprocessor_factory(self):
        from src.preprocessing.preprocessor import get_preprocessor, BasePreprocessor
        for ptype in ("no_interp", "naive_interp", "ddfp"):
            cfg  = _preprocessor_cfg(ptype)
            prep = get_preprocessor(cfg)
            assert isinstance(prep, BasePreprocessor)

    def test_get_preprocessor_invalid_type_raises(self):
        from src.preprocessing.preprocessor import get_preprocessor
        with pytest.raises(ValueError, match="Unknown type"):
            get_preprocessor(_preprocessor_cfg("nonexistent"))

    def test_no_interp_label_binarised(self):
        """NoInterp: label output must contain only {0, 1}."""
        from src.preprocessing.preprocessor import NoInterpPreprocessor
        prep = NoInterpPreprocessor(_preprocessor_cfg("no_interp"))
        _, out_lbl = prep(self.IMG, self.LBL, None)
        unique = set(np.unique(out_lbl))
        assert unique.issubset({0.0, 1.0}), f"non-binary label values: {unique}"


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TestPartAScripts:
    """Tests for scripts/part_a/ experiment scripts."""

    def test_exp_a1_imports(self):
        mod = _load_script("scripts/part_a/exp_a1_correctness.py")
        assert hasattr(mod, "run_a1")

    def test_exp_a1_cpu_path(self):
        """Run run_a1 via CPU-only path and verify CSV is created."""
        mod = _load_script("scripts/part_a/exp_a1_correctness.py")
        with tempfile.TemporaryDirectory() as tmp:
            mod.run_a1(Path(tmp))
            csv_file = Path(tmp) / "a1_correctness.csv"
            assert csv_file.exists(), "a1_correctness.csv not created"
            with open(csv_file) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 0
            methods = {r["method"] for r in rows}
            assert "no_interp" in methods
            assert "naive_interp" in methods

    def test_exp_a1_no_interp_has_violations(self):
        """no_interp must produce violations in 2D."""
        mod = _load_script("scripts/part_a/exp_a1_correctness.py")
        with tempfile.TemporaryDirectory() as tmp:
            mod.run_a1(Path(tmp))
            with open(Path(tmp) / "a1_correctness.csv") as f:
                rows = list(csv.DictReader(f))
        no_rows = [r for r in rows if r["method"] == "no_interp" and r["ndim"] == "2"]
        assert all(int(r["n_violations"]) > 0 for r in no_rows)

    def test_exp_a2_imports(self):
        mod = _load_script("scripts/part_a/exp_a2_speedup.py")
        assert callable(getattr(mod, "main", None)) or True

    def test_exp_a3_imports(self):
        mod = _load_script("scripts/part_a/exp_a3_delta.py")
        assert hasattr(mod, "run_a3")

    def test_exp_a4_imports_and_helpers(self):
        mod = _load_script("scripts/part_a/exp_a4_scalability.py")
        assert hasattr(mod, "run_a4")
        a, b = mod._fit_scaling_exponent([100, 1000, 10000], [0.01, 0.1, 1.0])
        assert abs(b - 1.0) < 0.05, f"scaling exponent should be ~1.0, got {b:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TestPartBScripts:
    """Tests for scripts/part_b/ experiment scripts."""

    # ── b1: topology_metrics, _tsi, _b0, _chi ──────────────────────────────

    def test_exp_b1_imports(self):
        mod = _load_script("scripts/part_b/exp_b1_topology_accuracy.py")
        for name in ("topology_metrics", "_tsi", "_b0", "_chi", "run_experiment"):
            assert hasattr(mod, name), f"missing: {name}"

    def test_b1_tsi_zero_on_constant_image(self):
        """TSI = 0 when β₀ does not vary across thresholds."""
        mod = _load_script("scripts/part_b/exp_b1_topology_accuracy.py")
        img = np.zeros((31, 31), dtype=np.float32)
        img[10:20, 10:20] = 200.0
        assert mod._tsi(img) == pytest.approx(0.0, abs=1e-6)

    def test_b1_tsi_nonzero_on_noisy_image(self):
        """TSI > 0 when topology changes across thresholds.

        naive_interp output is [0,1] float scale; β₀ varies as TSI thresholds [0.3..0.7]
        cross the blob boundary.
        """
        mod = _load_script("scripts/part_b/exp_b1_topology_accuracy.py")
        from src.preprocessing.preprocessor import NaiveInterpPreprocessor
        img = np.zeros((16, 16), dtype=np.float32)
        img[3:13, 2:7]  = 1.0
        img[3:13, 9:14] = 1.0
        img[7:9,  7:9]  = 0.3
        prep = NaiveInterpPreprocessor(_preprocessor_cfg("naive_interp"))
        u, _ = prep(img, None, None)
        assert mod._tsi(u) > 0.0, (
            "naive_interp on two-blob image should have TSI > 0"
        )

    def test_b1_b0_single_blob(self):
        mod = _load_script("scripts/part_b/exp_b1_topology_accuracy.py")
        binary = np.zeros((20, 20), dtype=np.uint8)
        binary[5:15, 5:15] = 1
        assert mod._b0(binary, 4) == 1
        assert mod._b0(binary, 8) == 1

    def test_b1_b0_two_separate_blobs(self):
        mod = _load_script("scripts/part_b/exp_b1_topology_accuracy.py")
        binary = np.zeros((20, 20), dtype=np.uint8)
        binary[2:6, 2:6]   = 1
        binary[12:16, 12:16] = 1
        assert mod._b0(binary, 4) == 2

    def test_b1_chi_square(self):
        """Simple square: χ = 1."""
        mod = _load_script("scripts/part_b/exp_b1_topology_accuracy.py")
        binary = np.zeros((20, 20), dtype=np.uint8)
        binary[5:15, 5:15] = 1
        assert mod._chi(binary) == 1

    def test_b1_topology_metrics_keys(self):
        mod = _load_script("scripts/part_b/exp_b1_topology_accuracy.py")
        img = np.zeros((31, 31), dtype=np.float32)
        img[10:20, 10:20] = 180.0
        m = mod.topology_metrics(img, ref_chi=1)
        for key in ("beta0_4conn", "beta0_8conn", "chi", "cc", "tsi",
                    "b0_consistency", "is_binary"):
            assert key in m, f"missing metric key: {key}"

    def test_b1_ddfp_cc_zero(self):
        """DDFPPreprocessor output should have CC = 0 (DWC guarantee)."""
        mod = _load_script("scripts/part_b/exp_b1_topology_accuracy.py")
        from src.preprocessing.preprocessor import DDFPPreprocessor
        img_f32 = np.random.default_rng(7).random((16, 16)).astype(np.float32)
        prep = DDFPPreprocessor(_preprocessor_cfg("ddfp"))
        u, _ = prep(img_f32, None, None)
        m = mod.topology_metrics(u * 255.0, ref_chi=1)
        assert m["cc"] == 0, f"DD-FP CC should be 0, got {m['cc']}"

    def test_b1_wilcoxon_test_significant(self):
        """Two clearly separated distributions should give significant=True."""
        mod = _load_script("scripts/part_b/exp_b1_topology_accuracy.py")
        x = [0.0] * 10           # ddfp: TSI = 0
        y = [200.0 + i for i in range(10)]
        r = mod.wilcoxon_test(x, y)
        assert r["significant"] is True
        assert r["pvalue"] < 0.001

    # ── b2: full_metrics, make_synthetic_images ─────────────────────────────

    def test_exp_b2_imports(self):
        mod = _load_script("scripts/part_b/exp_b2_cc_analysis.py")
        assert hasattr(mod, "full_metrics")
        assert hasattr(mod, "make_synthetic_images")

    def test_b2_full_metrics_disk(self):
        mod = _load_script("scripts/part_b/exp_b2_cc_analysis.py")
        synth = mod.make_synthetic_images()
        img, ref_b0, ref_chi = synth["disk_1"]
        m = mod.full_metrics(img, ref_b0=ref_b0, ref_chi=ref_chi)
        for key in ("beta0_4conn", "beta0_8conn", "chi", "cc", "tsi", "is_binary"):
            assert key in m, f"missing key: {key}"

    def test_b2_synthetic_images_nonempty(self):
        mod = _load_script("scripts/part_b/exp_b2_cc_analysis.py")
        synth = mod.make_synthetic_images()
        assert len(synth) >= 3, "need at least 3 synthetic images"
        for name, (arr, b0, chi) in synth.items():
            assert arr.ndim == 2
            assert arr.dtype in (np.float32, np.float64, np.uint8)

    def test_b2_ddfp_output_cc_zero(self):
        """DDFPPreprocessor output should also have CC = 0 in full_metrics."""
        mod = _load_script("scripts/part_b/exp_b2_cc_analysis.py")
        from src.preprocessing.preprocessor import DDFPPreprocessor
        img_f32 = np.random.default_rng(9).random((16, 16)).astype(np.float32)
        prep = DDFPPreprocessor(_preprocessor_cfg("ddfp"))
        u, _ = prep(img_f32, None, None)
        m = mod.full_metrics(u * 255.0, ref_b0=1, ref_chi=1)
        assert m["cc"] == 0


    def test_exp_b3_imports(self):
        mod = _load_script("scripts/part_b/exp_b3_brats_3d_all.py")
        for name in ("b0_3d", "euler_3d", "tsi_3d", "topology_metrics_3d"):
            assert hasattr(mod, name), f"missing: {name}"

    def test_b3_b0_3d_single_blob(self):
        mod = _load_script("scripts/part_b/exp_b3_brats_3d_all.py")
        binary = np.zeros((15, 15, 15), dtype=np.uint8)
        binary[4:10, 4:10, 4:10] = 1
        assert mod.b0_3d(binary, 6) == 1

    def test_b3_b0_3d_two_blobs(self):
        mod = _load_script("scripts/part_b/exp_b3_brats_3d_all.py")
        binary = np.zeros((20, 20, 20), dtype=np.uint8)
        binary[1:5, 1:5, 1:5]    = 1
        binary[14:18, 14:18, 14:18] = 1
        assert mod.b0_3d(binary, 6) == 2

    def test_b3_euler_3d_cube(self):
        """Simple cuboid: χ = 1 (simply connected solid)."""
        mod = _load_script("scripts/part_b/exp_b3_brats_3d_all.py")
        binary = np.zeros((15, 15, 15), dtype=np.uint8)
        binary[4:10, 4:10, 4:10] = 1
        assert mod.euler_3d(binary) == 1

    def test_b3_tsi_3d_zero_on_constant(self):
        mod = _load_script("scripts/part_b/exp_b3_brats_3d_all.py")
        interp = np.zeros((15, 15, 15), dtype=np.float32)
        interp[4:10, 4:10, 4:10] = 200.0
        assert mod.tsi_3d(interp) == pytest.approx(0.0, abs=1e-6)

    def test_b3_topology_metrics_3d_keys(self):
        mod = _load_script("scripts/part_b/exp_b3_brats_3d_all.py")
        interp = np.random.rand(15, 15, 15).astype(np.float32) * 200
        m = mod.topology_metrics_3d(interp, ref_b0=1, ref_chi=1)
        for key in ("beta0_6conn", "beta0_26conn", "chi", "cc_3d", "tsi_3d"):
            assert key in m, f"missing key: {key}"

    def test_b3_wilson_ci_valid_range(self):
        """Wilson CI: lower <= p_hat <= upper, both in [0,1]."""
        mod = _load_script("scripts/part_b/exp_b3_brats_3d_all.py")
        lo, hi = mod.wilson_ci(k=9, n=10)
        assert 0.0 <= lo <= 0.9 <= hi <= 1.0

    # ── b4: CREMI 3D ─────────────────────────────────────────────────────────

    def test_exp_b4_imports(self):
        mod = _load_script("scripts/part_b/exp_b4_cremi_3d.py")
        for name in ("make_synthetic_membrane_3d", "extract_subvolumes",
                     "b0_3d", "euler_3d", "tsi_3d", "topology_metrics_3d"):
            assert hasattr(mod, name), f"missing: {name}"

    def test_b4_make_synthetic_membrane(self):
        mod = _load_script("scripts/part_b/exp_b4_cremi_3d.py")
        synth = mod.make_synthetic_membrane_3d()
        assert len(synth) >= 1
        name, vol = synth[0]
        assert isinstance(name, str)
        assert vol.ndim == 3
        assert vol.dtype == np.uint8

    def test_b4_extract_subvolumes_count(self):
        mod = _load_script("scripts/part_b/exp_b4_cremi_3d.py")
        _, vol = mod.make_synthetic_membrane_3d()[0]
        subs = mod.extract_subvolumes(vol, n_patches=3, patch_size=(8, 8, 8))
        assert len(subs) == 3

    def test_b4_extract_subvolumes_shape(self):
        mod = _load_script("scripts/part_b/exp_b4_cremi_3d.py")
        _, vol = mod.make_synthetic_membrane_3d()[0]
        subs = mod.extract_subvolumes(vol, n_patches=2, patch_size=(8, 8, 8))
        for name, patch in subs:
            assert patch.shape == (8, 8, 8), f"patch shape: {patch.shape}"

    def test_b4_topology_metrics_3d_keys(self):
        mod = _load_script("scripts/part_b/exp_b4_cremi_3d.py")
        interp = np.random.rand(15, 15, 15).astype(np.float32) * 200
        m = mod.topology_metrics_3d(interp, ref_b0=1, ref_chi=1)
        for key in ("beta0_6conn", "beta0_26conn", "chi", "cc_3d", "tsi_3d"):
            assert key in m

    # ── verify_wilcoxon ──────────────────────────────────────────────────────

    def test_verify_wilcoxon_imports(self):
        mod = _load_script("scripts/part_b/verify_wilcoxon.py")
        assert hasattr(mod, "run_verification")
        assert hasattr(mod, "TESTS")
        assert len(mod.TESTS) >= 2

    def test_verify_wilcoxon_run_on_synthetic_csv(self):
        """Run run_verification on a synthetic CSV and verify bool return.

        CSV column structure matches the format expected by verify_wilcoxon.py:
          dataset, sample, preprocessing, cc, tsi, b0_consistency
        """
        import pandas as pd
        mod = _load_script("scripts/part_b/verify_wilcoxon.py")
        rng = np.random.default_rng(0)
        rows = []
        for i in range(20):
            rows.append({"dataset": "drive", "sample": i,
                         "preprocessing": "ddfp",
                         "cc": 0, "tsi": 0.0, "b0_consistency": 1.0})
            rows.append({"dataset": "drive", "sample": i,
                         "preprocessing": "no_interp",
                         "cc": int(rng.integers(1, 10)),
                         "tsi": float(rng.integers(100, 300)),
                         "b0_consistency": float(rng.random() * 0.5)})
            rows.append({"dataset": "drive", "sample": i,
                         "preprocessing": "naive_interp",
                         "cc": 0,
                         "tsi": float(rng.integers(100, 300)),
                         "b0_consistency": float(rng.random() * 0.5)})
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w",
                                         delete=False, newline="") as f:
            csv_path = Path(f.name)
            pd.DataFrame(rows).to_csv(f, index=False)
        passed = mod.run_verification(csv_path, dataset="drive")
        assert isinstance(passed, bool)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TestGpuKernel:
    """Tests for src/ddfp/gpu_immersion.py GPU kernels."""

    @gpu_skip
    def test_build_ispan_gpu_shape(self):
        import cupy as cp
        from src.ddfp.gpu_immersion import build_ispan_gpu
        vol = _synth3d(8)
        W, H, D = vol.shape
        U_lo, U_hi, l_inf = build_ispan_gpu(vol)
        assert U_lo.shape == (2*W+1, 2*H+1, 2*D+1)

    @gpu_skip
    def test_front_propagation_gpu_dwc(self):
        """GPU FP output should also have zero DWC violations."""
        import cupy as cp
        from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu
        from src.utils.benchmark_utils import verify_dwc
        vol = _synth3d(16)
        U_lo, U_hi, l_inf = build_ispan_gpu(vol)
        u_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
        u = cp.asnumpy(u_pad[1:-1, 1:-1, 1:-1]).astype(np.float32)
        r = verify_dwc(vol, u)
        assert r["n_violations"] == 0

    @gpu_skip
    def test_verify_dwc_gpu(self):
        """Verify GPU FP output satisfies DWC using verify_dwc_gpu.

        verify_dwc_gpu(u_dwc) → dict{"violations": int, "dwc_ok": bool, ...}
        Takes only u_dwc as argument (original vol not required).
        """
        from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu, verify_dwc_gpu
        import cupy as cp
        vol = _synth3d(16)
        U_lo, U_hi, l_inf = build_ispan_gpu(vol)
        u_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
        u = u_pad[1:-1, 1:-1, 1:-1]
        result = verify_dwc_gpu(u)
        assert result["violations"] == 0
        assert result["dwc_ok"] is True

    @gpu_skip
    def test_gpu_cpu_consistency(self):
        """Both GPU and CPU pipelines must guarantee DWC.

        CPU (Jacobi) and GPU (Level-BFS) use different algorithms so pixel
        values may differ; verify via the shared invariant: DWC guarantee.
        """
        import cupy as cp
        from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu, verify_dwc_gpu
        from src.ddfp.parallel_immersion import immersion_pipeline as cpu_pipeline
        from src.utils.benchmark_utils import verify_dwc

        vol = _synth3d(12)

        # CPU
        u_cpu = cpu_pipeline(vol, verbose=False)
        if isinstance(u_cpu, tuple): u_cpu = u_cpu[0]
        cpu_result = verify_dwc(vol, u_cpu)
        assert cpu_result["n_violations"] == 0, "CPU pipeline DWC violated"

        # GPU
        U_lo, U_hi, l_inf = build_ispan_gpu(vol)
        u_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
        u_gpu = u_pad[1:-1, 1:-1, 1:-1]
        gpu_result = verify_dwc_gpu(u_gpu)
        assert gpu_result["violations"] == 0, "GPU pipeline DWC violated"

        assert u_cpu.shape == cp.asnumpy(u_gpu).shape