"""
Multiview 3D consistency evaluation (experimental).

Scaffolding for the paper:
    "Can These Views Be One Scene? Evaluating Multiview 3D Consistency
     when 3D Foundation Models Hallucinate"
    (arXiv:2605.18754v1)

VQASynth reconstructs 3D scenes with VGGT (see vqasynth.scene_fusion).
The paper shows that VGGT, MASt3R, DUSt3R, and Fast3R can hallucinate
dense geometry and cross-view support for unrelated scenes, repeated
images, and pure noise — and proposes COLMAP-based, failure-aware
consistency signals that correlate up to ~4x better with human judgment
than MEt3R.

This module provides the entry point to score the geometric integrity of
VQASynth's reconstructions using those signals:

    matches       — count of geometrically verified keypoint matches
    registration  — did classical SfM register the views into one pose graph
    dense support — fraction of pixels with triangulated 3D support
    failure       — did reconstruction outright fail (a positive signal:
                    failure on unrelated views is *correct* behavior)

The parametric family in the paper combines a neural backbone, a residual
term, and an aggregation; the helpers below cover the aggregation +
classical-verification half. The full COLMAP CLI integration and the
neural backbone are left as documented TODOs so this module does not
silently fake work that needs external binaries / checkpoints.

Heavy dependencies (numpy, cv2, PIL) are imported lazily inside the
methods that need them so the pure-Python utilities below can be unit
tested without those packages installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CONSISTENCY_SIGNALS = ("matches", "registration", "dense_support", "failure")


@dataclass
class MultiviewConsistencyConfig:
    """Hyperparameters for COLMAP-style consistency scoring.

    Defaults are taken from values reported / suggested in the paper; tune
    per dataset. ``signal_weights`` must contain all keys in
    :data:`CONSISTENCY_SIGNALS` and sum to a positive number — they are
    renormalized internally.
    """

    min_matches: int = 50
    """Lowe-filtered match count below which a view pair is considered
    geometrically unrelated."""

    min_inlier_ratio: float = 0.3
    """Minimum RANSAC inlier ratio for a fundamental/essential matrix
    estimate to count as a successful pairwise registration."""

    min_dense_support: float = 0.10
    """Minimum fraction of pixels that must have triangulated 3D support
    for the dense_support signal to fire."""

    lowes_ratio: float = 0.75
    """Lowe's ratio test threshold for filtering raw descriptor matches."""

    consistency_threshold: float = 0.5
    """Aggregate score above which a reconstruction is labelled
    'consistent' (vs. 'hallucinated')."""

    detector: str = "ORB"
    """Feature detector for the lightweight matching path. ORB ships with
    OpenCV's BSD-licensed build; SIFT is also accepted if available."""

    max_features: int = 2000
    """Cap on detected features per image."""

    signal_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "matches": 0.25,
            "registration": 0.25,
            "dense_support": 0.25,
            "failure": 0.25,
        }
    )
    """Weighting for the parametric-family aggregation. See paper §3."""

    def normalized_weights(self) -> Dict[str, float]:
        """Return weights renormalized to sum to 1.0, in the canonical
        signal order. Missing keys default to 0."""
        raw = {k: float(self.signal_weights.get(k, 0.0)) for k in CONSISTENCY_SIGNALS}
        total = sum(raw.values())
        if total <= 0:
            raise ValueError(
                "signal_weights must contain at least one positive entry"
            )
        return {k: v / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Concrete utilities — pure Python, no heavy deps
# ---------------------------------------------------------------------------


def lowes_ratio_filter(
    distance_pairs: Iterable[Sequence[float]],
    ratio: float = 0.75,
) -> int:
    """Count how many descriptor matches survive Lowe's ratio test.

    ``distance_pairs`` is an iterable of ``(d_best, d_second)`` descriptor
    distances (smaller = better match), as produced by OpenCV's
    ``knnMatch(k=2)``. A pair survives when ``d_best < ratio * d_second``.

    Pure Python so it can be unit tested without OpenCV.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    kept = 0
    for pair in distance_pairs:
        if len(pair) < 2:
            continue
        d_best, d_second = float(pair[0]), float(pair[1])
        if d_second <= 0:
            continue
        if d_best < ratio * d_second:
            kept += 1
    return kept


def dense_support_fraction(num_valid_points: int, num_pixels: int) -> float:
    """Fraction of image pixels that received triangulated 3D support.

    Returns 0.0 when ``num_pixels`` is 0 to keep the signal well defined
    on empty inputs.
    """
    if num_pixels <= 0:
        return 0.0
    return max(0.0, min(1.0, float(num_valid_points) / float(num_pixels)))


def pixel_to_normalized(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """Convert pixel coordinates to centered, unit-scaled coords in
    [-1, 1] x [-1, 1] (origin at image center).

    Mirrors the convention used inside COLMAP's image undistortion and
    is the form expected by ``cv2.findFundamentalMat`` after normalization.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}, {height}")
    nx = (2.0 * x / width) - 1.0
    ny = (2.0 * y / height) - 1.0
    return nx, ny


def signal_indicators(
    num_matches: int,
    inlier_ratio: float,
    dense_support: float,
    reconstruction_failed: bool,
    config: MultiviewConsistencyConfig,
) -> Dict[str, float]:
    """Turn raw measurements into the four [0, 1] consistency indicators.

    The ``failure`` signal is inverted: reconstruction *failure on
    unrelated views* is correct behavior, so it counts against the
    "this is one scene" hypothesis only when reconstruction *succeeded*.
    """
    matches = 1.0 if num_matches >= config.min_matches else 0.0
    registration = 1.0 if inlier_ratio >= config.min_inlier_ratio else 0.0
    dense = 1.0 if dense_support >= config.min_dense_support else 0.0
    # Failure signal: a successful reconstruction supports consistency.
    failure = 0.0 if reconstruction_failed else 1.0
    return {
        "matches": matches,
        "registration": registration,
        "dense_support": dense,
        "failure": failure,
    }


def weighted_consistency_score(
    indicators: Dict[str, float],
    config: MultiviewConsistencyConfig,
) -> float:
    """Combine signal indicators into a single [0, 1] consistency score.

    This is the aggregation half of the paper's parametric family — the
    backbone and residual terms live in the (stubbed)
    :meth:`MultiviewConsistencyEvaluator.score_pair` neural path.
    """
    weights = config.normalized_weights()
    score = 0.0
    for signal in CONSISTENCY_SIGNALS:
        score += weights[signal] * float(indicators.get(signal, 0.0))
    return max(0.0, min(1.0, score))


def classify_consistency(score: float, config: MultiviewConsistencyConfig) -> str:
    """Map a scalar score to ``"consistent"`` or ``"hallucinated"``."""
    return "consistent" if score >= config.consistency_threshold else "hallucinated"


def aggregate_pairwise_scores(scores: Sequence[float]) -> Dict[str, float]:
    """Summarize a set of pairwise consistency scores across a scene.

    The paper reports min/mean over all (i, j) view pairs; min is the
    most informative — a single inconsistent pair flags scene-level
    hallucination.
    """
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    vals = [float(s) for s in scores]
    return {
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


# ---------------------------------------------------------------------------
# Evaluator class — heavy paths stubbed
# ---------------------------------------------------------------------------


class MultiviewConsistencyEvaluator:
    """Score VQASynth reconstructions for cross-view geometric consistency.

    The constructor is intentionally cheap: no model weights, no COLMAP
    process. Heavy work happens lazily inside the methods that need it,
    and the COLMAP CLI path is documented but not implemented (see
    :meth:`run_colmap`).

    Intended usage:
        evaluator = MultiviewConsistencyEvaluator()
        signals = evaluator.signals_no_colmap(num_matches=12,
                                              inlier_ratio=0.05,
                                              dense_support=0.02,
                                              reconstruction_failed=True)
        score = evaluator.aggregate(signals)
        label = evaluator.classify(score)
    """

    def __init__(self, config: Optional[MultiviewConsistencyConfig] = None):
        self.config = config or MultiviewConsistencyConfig()

    # -- Pure-Python convenience surface -------------------------------------

    def signals_no_colmap(
        self,
        num_matches: int,
        inlier_ratio: float,
        dense_support: float,
        reconstruction_failed: bool,
    ) -> Dict[str, float]:
        """Build the indicator dict from already-computed measurements."""
        return signal_indicators(
            num_matches=num_matches,
            inlier_ratio=inlier_ratio,
            dense_support=dense_support,
            reconstruction_failed=reconstruction_failed,
            config=self.config,
        )

    def aggregate(self, indicators: Dict[str, float]) -> float:
        return weighted_consistency_score(indicators, self.config)

    def classify(self, score: float) -> str:
        return classify_consistency(score, self.config)

    def score_from_measurements(
        self,
        num_matches: int,
        inlier_ratio: float,
        dense_support: float,
        reconstruction_failed: bool,
    ) -> Dict[str, object]:
        """One-shot: measurements -> indicators -> score -> label."""
        indicators = self.signals_no_colmap(
            num_matches=num_matches,
            inlier_ratio=inlier_ratio,
            dense_support=dense_support,
            reconstruction_failed=reconstruction_failed,
        )
        score = self.aggregate(indicators)
        return {
            "indicators": indicators,
            "score": score,
            "label": self.classify(score),
        }

    # -- Heavy paths (stubbed) ----------------------------------------------

    def count_matches(self, image_a, image_b) -> int:
        """Detect features in two images and count Lowe-filtered matches.

        Lazy-imports cv2/numpy so the rest of the module remains usable
        in environments without OpenCV. Returns the count of matches
        passing :meth:`MultiviewConsistencyConfig.lowes_ratio`.

        TODO: extend to symmetric ratio test + RANSAC inlier filtering
        with ``cv2.findFundamentalMat`` to mirror the paper's
        "geometrically verified" match count exactly.
        """
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "count_matches requires opencv-python and numpy"
            ) from exc

        detector_name = self.config.detector.upper()
        if detector_name == "ORB":
            detector = cv2.ORB_create(nfeatures=self.config.max_features)
            norm = cv2.NORM_HAMMING
        elif detector_name == "SIFT":
            detector = cv2.SIFT_create(nfeatures=self.config.max_features)
            norm = cv2.NORM_L2
        else:
            raise ValueError(f"Unknown detector: {self.config.detector}")

        gray_a = self._to_gray(image_a, cv2, np)
        gray_b = self._to_gray(image_b, cv2, np)

        _, des_a = detector.detectAndCompute(gray_a, None)
        _, des_b = detector.detectAndCompute(gray_b, None)
        if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
            return 0

        matcher = cv2.BFMatcher(norm)
        knn = matcher.knnMatch(des_a, des_b, k=2)
        distance_pairs = [(m[0].distance, m[1].distance) for m in knn if len(m) == 2]
        return lowes_ratio_filter(distance_pairs, ratio=self.config.lowes_ratio)

    @staticmethod
    def _to_gray(image, cv2, np):
        """Best-effort PIL/ndarray -> grayscale ndarray conversion."""
        if hasattr(image, "convert"):  # PIL Image
            arr = np.array(image.convert("L"))
            return arr
        arr = np.asarray(image)
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr

    def run_colmap(self, image_paths: List[str], workdir: str) -> Dict[str, object]:
        """Run the classical COLMAP pipeline over a set of images.

        TODO: shell out to the COLMAP CLI:
          1. ``colmap feature_extractor`` — SIFT keypoints + descriptors
          2. ``colmap exhaustive_matcher`` — pairwise + RANSAC verification
          3. ``colmap mapper`` — incremental SfM, returns registered cameras
          4. ``colmap image_undistorter`` + ``patch_match_stereo`` — dense

        Should return a dict with keys ``num_matches``, ``inlier_ratio``,
        ``dense_support`` (in [0, 1]), and ``reconstruction_failed``
        (bool), matching the inputs expected by
        :meth:`signals_no_colmap`.

        Not implemented in this PR because COLMAP is a system binary the
        VQASynth Docker image does not currently install; integrating it
        is a separate piece of infra work tracked alongside this module.
        """
        raise NotImplementedError(
            "COLMAP CLI integration is not yet wired up; see TODO above. "
            "Use signals_no_colmap() with pre-computed measurements until "
            "the COLMAP runner lands."
        )

    # -- Scene-level convenience --------------------------------------------

    def aggregate_scene(
        self,
        pairwise_scores: Sequence[float],
    ) -> Dict[str, object]:
        """Roll pairwise scores up to a scene-level summary + label.

        The scene is flagged as hallucinated when the *min* pairwise
        score falls below the threshold — one bad pair is enough.
        """
        stats = aggregate_pairwise_scores(pairwise_scores)
        scene_label = (
            "consistent"
            if stats["n"] > 0 and stats["min"] >= self.config.consistency_threshold
            else "hallucinated"
        )
        return {**stats, "label": scene_label}
