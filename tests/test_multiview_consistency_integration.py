"""Tests for the multiview 3D consistency integration scaffold.

These tests intentionally use only the standard library so they pass in
the lightweight CI image used to validate the scaffolding PR — the
heavy paths (OpenCV feature matching, the COLMAP CLI) are stubbed in
the integration module and excluded from coverage here.
"""

import math

import pytest

from vqasynth.multiview_consistency_integration import (
    CONSISTENCY_SIGNALS,
    MultiviewConsistencyConfig,
    MultiviewConsistencyEvaluator,
    aggregate_pairwise_scores,
    classify_consistency,
    dense_support_fraction,
    lowes_ratio_filter,
    pixel_to_normalized,
    signal_indicators,
    weighted_consistency_score,
)


# --- Lowe's ratio filter ----------------------------------------------------


def test_lowes_ratio_filter_keeps_clear_winners():
    pairs = [(1.0, 10.0), (2.0, 3.0), (5.0, 5.0)]
    # 1.0 < 0.75 * 10 ✓ ; 2.0 < 0.75 * 3 = 2.25 ✓ ; 5.0 < 0.75 * 5 = 3.75 ✗
    assert lowes_ratio_filter(pairs, ratio=0.75) == 2


def test_lowes_ratio_filter_skips_malformed_pairs():
    # Single-element tuple and a zero second-best are both ignored.
    assert lowes_ratio_filter([(1.0,), (1.0, 0.0)], ratio=0.75) == 0


def test_lowes_ratio_filter_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        lowes_ratio_filter([(1.0, 2.0)], ratio=0.0)
    with pytest.raises(ValueError):
        lowes_ratio_filter([(1.0, 2.0)], ratio=1.5)


# --- Dense support fraction -------------------------------------------------


def test_dense_support_fraction_normal_case():
    assert dense_support_fraction(250, 1000) == pytest.approx(0.25)


def test_dense_support_fraction_clamps_to_unit_interval():
    assert dense_support_fraction(5000, 1000) == 1.0
    assert dense_support_fraction(-10, 1000) == 0.0


def test_dense_support_fraction_zero_pixels_safe():
    assert dense_support_fraction(0, 0) == 0.0
    assert dense_support_fraction(100, 0) == 0.0


# --- Pixel coordinate conversion -------------------------------------------


def test_pixel_to_normalized_center_maps_to_origin():
    nx, ny = pixel_to_normalized(50, 50, 100, 100)
    assert math.isclose(nx, 0.0)
    assert math.isclose(ny, 0.0)


def test_pixel_to_normalized_corners():
    nx, ny = pixel_to_normalized(0, 0, 100, 100)
    assert (nx, ny) == (-1.0, -1.0)
    nx, ny = pixel_to_normalized(100, 100, 100, 100)
    assert (nx, ny) == (1.0, 1.0)


def test_pixel_to_normalized_rejects_bad_size():
    with pytest.raises(ValueError):
        pixel_to_normalized(0, 0, 0, 100)


# --- Signal indicators ------------------------------------------------------


def test_signal_indicators_all_consistent():
    cfg = MultiviewConsistencyConfig()
    inds = signal_indicators(
        num_matches=200,
        inlier_ratio=0.5,
        dense_support=0.3,
        reconstruction_failed=False,
        config=cfg,
    )
    assert inds == {
        "matches": 1.0,
        "registration": 1.0,
        "dense_support": 1.0,
        "failure": 1.0,
    }


def test_signal_indicators_unrelated_views():
    cfg = MultiviewConsistencyConfig()
    inds = signal_indicators(
        num_matches=5,
        inlier_ratio=0.05,
        dense_support=0.02,
        reconstruction_failed=True,
        config=cfg,
    )
    assert inds == {
        "matches": 0.0,
        "registration": 0.0,
        "dense_support": 0.0,
        "failure": 0.0,
    }


def test_signal_indicators_keys_match_canonical_order():
    cfg = MultiviewConsistencyConfig()
    inds = signal_indicators(
        num_matches=0,
        inlier_ratio=0.0,
        dense_support=0.0,
        reconstruction_failed=False,
        config=cfg,
    )
    assert set(inds) == set(CONSISTENCY_SIGNALS)


# --- Weighted aggregation ---------------------------------------------------


def test_weighted_score_uniform_weights():
    cfg = MultiviewConsistencyConfig()
    inds = {"matches": 1.0, "registration": 1.0, "dense_support": 0.0, "failure": 0.0}
    assert weighted_consistency_score(inds, cfg) == pytest.approx(0.5)


def test_weighted_score_respects_custom_weights():
    cfg = MultiviewConsistencyConfig(
        signal_weights={
            "matches": 1.0,
            "registration": 0.0,
            "dense_support": 0.0,
            "failure": 0.0,
        }
    )
    inds = {"matches": 1.0, "registration": 0.0, "dense_support": 0.0, "failure": 0.0}
    assert weighted_consistency_score(inds, cfg) == pytest.approx(1.0)


def test_weighted_score_renormalizes_weights():
    cfg = MultiviewConsistencyConfig(
        signal_weights={
            "matches": 2.0,
            "registration": 2.0,
            "dense_support": 0.0,
            "failure": 0.0,
        }
    )
    inds = {"matches": 1.0, "registration": 1.0, "dense_support": 1.0, "failure": 1.0}
    # Only matches+registration carry weight, both fire -> score 1.0
    assert weighted_consistency_score(inds, cfg) == pytest.approx(1.0)


def test_weighted_score_rejects_zero_weights():
    cfg = MultiviewConsistencyConfig(
        signal_weights={k: 0.0 for k in CONSISTENCY_SIGNALS}
    )
    with pytest.raises(ValueError):
        weighted_consistency_score({k: 1.0 for k in CONSISTENCY_SIGNALS}, cfg)


# --- Classification ---------------------------------------------------------


def test_classify_consistency_threshold():
    cfg = MultiviewConsistencyConfig(consistency_threshold=0.5)
    assert classify_consistency(0.51, cfg) == "consistent"
    assert classify_consistency(0.5, cfg) == "consistent"  # boundary -> consistent
    assert classify_consistency(0.49, cfg) == "hallucinated"


# --- Pairwise aggregation ---------------------------------------------------


def test_aggregate_pairwise_scores_basic_stats():
    stats = aggregate_pairwise_scores([0.1, 0.5, 0.9])
    assert stats["n"] == 3
    assert stats["min"] == 0.1
    assert stats["max"] == 0.9
    assert stats["mean"] == pytest.approx(0.5)


def test_aggregate_pairwise_scores_empty():
    stats = aggregate_pairwise_scores([])
    assert stats == {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}


# --- Evaluator class smoke tests --------------------------------------------


def test_evaluator_default_config():
    ev = MultiviewConsistencyEvaluator()
    assert isinstance(ev.config, MultiviewConsistencyConfig)
    assert ev.config.detector == "ORB"


def test_evaluator_flags_unrelated_images_as_hallucination():
    ev = MultiviewConsistencyEvaluator()
    result = ev.score_from_measurements(
        num_matches=3,
        inlier_ratio=0.02,
        dense_support=0.01,
        reconstruction_failed=True,
    )
    assert result["label"] == "hallucinated"
    assert result["score"] == pytest.approx(0.0)


def test_evaluator_accepts_real_scene_measurements():
    ev = MultiviewConsistencyEvaluator()
    result = ev.score_from_measurements(
        num_matches=400,
        inlier_ratio=0.6,
        dense_support=0.4,
        reconstruction_failed=False,
    )
    assert result["label"] == "consistent"
    assert result["score"] == pytest.approx(1.0)


def test_evaluator_aggregate_scene_min_rule():
    ev = MultiviewConsistencyEvaluator(
        MultiviewConsistencyConfig(consistency_threshold=0.5)
    )
    # One bad pair drags the scene to "hallucinated"
    bad = ev.aggregate_scene([0.95, 0.9, 0.1])
    assert bad["label"] == "hallucinated"
    assert bad["min"] == pytest.approx(0.1)

    good = ev.aggregate_scene([0.6, 0.7, 0.9])
    assert good["label"] == "consistent"


def test_evaluator_aggregate_scene_empty_is_hallucinated():
    ev = MultiviewConsistencyEvaluator()
    assert ev.aggregate_scene([])["label"] == "hallucinated"


def test_run_colmap_is_stubbed():
    ev = MultiviewConsistencyEvaluator()
    with pytest.raises(NotImplementedError):
        ev.run_colmap([], "/tmp/whatever")
