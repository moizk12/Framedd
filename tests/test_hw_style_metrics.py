from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "moiz_kashif_csc481_final" / "dataset_v2"


def _iter_images():
    if not DATASET_DIR.exists():
        pytest.skip(f"dataset not found: {DATASET_DIR}")
    for p in sorted(DATASET_DIR.rglob("*.jpg")):
        # expects dataset_v2/<category>/<file>.jpg
        if p.parent == DATASET_DIR:
            continue
        yield p


def _load_bgr(path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return bgr


@pytest.mark.parametrize("img_path", list(_iter_images()))
def test_path_a_metrics_schema_ranges_and_determinism(img_path: Path):
    # import inside test so pytest collection doesn't depend on sys.path hacks
    from moiz_kashif_csc481_final.vision_utils import compute_path_a_metrics_from_bgr

    bgr = _load_bgr(img_path)

    m1 = compute_path_a_metrics_from_bgr(bgr)
    m2 = compute_path_a_metrics_from_bgr(bgr)
    assert m1 == m2, "metrics must be deterministic for the same input"

    # HW2–HW7 expected keys
    keys = {
        "green_coverage_percentage",
        "laplacian_variance",
        "edge_density",
        "brightness_mean",
        "contrast_std",
        "hist_entropy",
        "hist_peakiness",
        "equalize_diff_energy",
        "edge_drop_mean_blur",
        "edge_drop_median_blur",
        "canny_sigma_chosen",
        "edge_density_tuned",
        "edge_density_by_sigma",
        "green_component_count",
        "green_component_areas_top5",
        "green_small_component_ratio",
    }
    missing = [k for k in sorted(keys) if k not in m1]
    assert not missing, f"missing keys: {missing}"

    # basic ranges
    assert 0.0 <= float(m1["green_coverage_percentage"]) <= 100.0
    assert float(m1["laplacian_variance"]) >= 0.0
    assert 0.0 <= float(m1["edge_density"]) <= 1.0
    assert 0.0 <= float(m1["edge_density_tuned"]) <= 1.0
    assert 0.0 <= float(m1["brightness_mean"]) <= 255.0
    assert float(m1["contrast_std"]) >= 0.0

    # HW2 ranges (entropy in [0,8] for 256 bins; peakiness in [0,1])
    assert 0.0 <= float(m1["hist_entropy"]) <= 8.1
    assert 0.0 <= float(m1["hist_peakiness"]) <= 1.0
    assert 0.0 <= float(m1["equalize_diff_energy"]) <= 255.0

    # HW3 drops: can be negative in weird cases, but should be bounded
    assert -1.0 <= float(m1["edge_drop_mean_blur"]) <= 1.0
    assert -1.0 <= float(m1["edge_drop_median_blur"]) <= 1.0

    # HW5: chosen sigma must be in the sweep list (stringified in dict keys)
    sig = float(m1["canny_sigma_chosen"])
    by_sigma = m1["edge_density_by_sigma"]
    assert isinstance(by_sigma, dict) and len(by_sigma) >= 3
    assert str(sig) in by_sigma

    # HW4/HW7: components
    assert int(m1["green_component_count"]) >= 0
    assert isinstance(m1["green_component_areas_top5"], list)
    assert 0.0 <= float(m1["green_small_component_ratio"]) <= 1.0


@pytest.mark.parametrize("img_path", list(_iter_images()))
def test_proposal_experiments_consistency_under_blur_and_exposure(img_path: Path):
    """
    Proposal-style experiments: metrics should behave predictably under controlled transforms.
    (CV-only checks; does not run CLIP/YOLO.)
    """
    from moiz_kashif_csc481_final.vision_utils import compute_path_a_metrics_from_bgr

    bgr = _load_bgr(img_path)
    base = compute_path_a_metrics_from_bgr(bgr)

    # Blur: Laplacian variance should not increase (allow tiny epsilon).
    blur_bgr = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2.0, sigmaY=2.0)
    blur_m = compute_path_a_metrics_from_bgr(blur_bgr)
    # At extremely low variance values, quantization/rounding can cause small increases after blur.
    base_lap = float(base["laplacian_variance"])
    blur_lap = float(blur_m["laplacian_variance"])
    tol = max(1e-3, 0.25 * base_lap)  # forgiving, but still catches big regressions
    assert blur_lap <= base_lap + tol
    assert float(blur_m["edge_density_tuned"]) <= float(base["edge_density_tuned"]) + 0.02

    # Exposure: brighten image; brightness_mean should increase.
    bright_bgr = cv2.convertScaleAbs(bgr, alpha=1.0, beta=30)
    bright_m = compute_path_a_metrics_from_bgr(bright_bgr)
    assert float(bright_m["brightness_mean"]) >= float(base["brightness_mean"]) - 1e-3

    # Structural metrics shouldn't explode under a mild brightness shift
    base_lap = float(base["laplacian_variance"]) + 1e-6
    bright_lap = float(bright_m["laplacian_variance"]) + 1e-6
    assert bright_lap / base_lap <= 10.0
    assert abs(float(bright_m["edge_density_tuned"]) - float(base["edge_density_tuned"])) <= 0.25


def test_category_expectations_are_reasonable():
    """
    Weak, forgiving checks to ensure metrics correlate with intended dataset labels.
    """
    from moiz_kashif_csc481_final.vision_utils import compute_path_a_metrics_from_bgr

    cats = {}
    for p in _iter_images():
        cat = p.parent.name
        bgr = _load_bgr(p)
        m = compute_path_a_metrics_from_bgr(bgr)
        cats.setdefault(cat, []).append(m)

    if "nature" in cats and "architecture" in cats:
        n_green = np.median([float(m["green_coverage_percentage"]) for m in cats["nature"]])
        a_green = np.median([float(m["green_coverage_percentage"]) for m in cats["architecture"]])
        assert n_green >= a_green + 1.0
    else:
        pytest.skip("need both nature and architecture categories for this check")

    if "portraits" in cats and "architecture" in cats:
        p_edge = np.median([float(m["edge_density_tuned"]) for m in cats["portraits"]])
        a_edge = np.median([float(m["edge_density_tuned"]) for m in cats["architecture"]])
        assert p_edge <= a_edge + 0.05
    else:
        pytest.skip("need both portraits and architecture categories for this check")

