"""Tests for the SegCompass integration scaffolding."""
from __future__ import annotations

import numpy as np
import pytest

from vqasynth.segcompass_integration import (
    SegCompassAligner,
    SegCompassConfig,
    align_concepts_to_masks,
    concept_caption_overlap,
    coverage_report,
    extract_answer_block,
    extract_concepts,
    extract_think_block,
    mask_area_fraction,
    mask_iou,
)


# ---------------------------------------------------------------------------
# CoT block extraction
# ---------------------------------------------------------------------------
def test_extract_think_block_basic():
    text = "<think>The red forklift is on the right.</think><answer>Right.</answer>"
    assert extract_think_block(text) == "The red forklift is on the right."


def test_extract_think_block_multiline_and_case():
    text = "noise <THINK>\n line one\n line two \n</THINK> trailing"
    assert extract_think_block(text) == "line one\n line two"


def test_extract_think_block_missing_returns_empty():
    assert extract_think_block("no tags here") == ""
    assert extract_think_block("") == ""
    assert extract_think_block(None) == ""  # type: ignore[arg-type]


def test_extract_answer_block_basic():
    text = "<think>...</think><answer> 2-3 feet </answer>"
    assert extract_answer_block(text) == "2-3 feet"


def test_extract_answer_block_missing_returns_empty():
    assert extract_answer_block("<think>only think</think>") == ""


# ---------------------------------------------------------------------------
# Concept extraction
# ---------------------------------------------------------------------------
def test_extract_concepts_filters_stopwords_and_short_tokens():
    text = "The red forklift is on the right of the brown boxes."
    concepts = extract_concepts(text)
    assert "forklift" in concepts
    assert "boxes" in concepts
    assert "brown" in concepts
    # stopwords / short tokens dropped
    assert "the" not in concepts
    assert "is" not in concepts
    assert "on" not in concepts
    assert "of" not in concepts


def test_extract_concepts_dedupes_preserving_order():
    text = "forklift boxes forklift pallet boxes pallet"
    assert extract_concepts(text) == ["forklift", "boxes", "pallet"]


def test_extract_concepts_respects_max_concepts():
    text = "alpha bravo charlie delta echo foxtrot"
    assert extract_concepts(text, max_concepts=3) == ["alpha", "bravo", "charlie"]


def test_extract_concepts_empty_input():
    assert extract_concepts("") == []
    assert extract_concepts("  ") == []


# ---------------------------------------------------------------------------
# Mask geometry
# ---------------------------------------------------------------------------
def test_mask_iou_identical_masks_is_one():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:6, 2:6] = 255
    assert mask_iou(m, m) == pytest.approx(1.0)


def test_mask_iou_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b[6:9, 6:9] = True
    assert mask_iou(a, b) == 0.0


def test_mask_iou_partial_overlap():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:4, 0:4] = True  # 16 px
    b[2:6, 2:6] = True  # 16 px, overlap 4 px, union 28
    assert mask_iou(a, b) == pytest.approx(4 / 28)


def test_mask_iou_both_empty_is_zero():
    z = np.zeros((4, 4), dtype=bool)
    assert mask_iou(z, z) == 0.0


def test_mask_iou_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mask_iou(np.zeros((3, 3), dtype=bool), np.zeros((4, 4), dtype=bool))


def test_mask_area_fraction():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[:5, :] = 255  # 50 of 100 pixels
    assert mask_area_fraction(m) == pytest.approx(0.5)


def test_mask_area_fraction_empty():
    assert mask_area_fraction(np.zeros((5, 5), dtype=bool)) == 0.0


# ---------------------------------------------------------------------------
# Lexical alignment
# ---------------------------------------------------------------------------
def test_concept_caption_overlap_exact_token_is_one():
    assert concept_caption_overlap("forklift", "a red forklift moves boxes") == 1.0


def test_concept_caption_overlap_morphological_variant_is_partial():
    s = concept_caption_overlap("box", "stacked brown boxes on a pallet")
    assert 0.0 < s < 1.0


def test_concept_caption_overlap_unrelated_is_low():
    s = concept_caption_overlap("airplane", "stacked brown boxes on a pallet")
    assert s < 0.2


def test_concept_caption_overlap_empty_inputs():
    assert concept_caption_overlap("", "anything") == 0.0
    assert concept_caption_overlap("forklift", "") == 0.0


# ---------------------------------------------------------------------------
# Concept-to-mask alignment
# ---------------------------------------------------------------------------
def _toy_masks(n: int = 2, h: int = 8, w: int = 8) -> list[np.ndarray]:
    masks = []
    for i in range(n):
        m = np.zeros((h, w), dtype=np.uint8)
        m[: h // 2, i * (w // n) : (i + 1) * (w // n)] = 255
        masks.append(m)
    return masks


def test_align_concepts_picks_best_caption_per_concept():
    masks = _toy_masks(2)
    captions = ["red forklift on the right", "brown cardboard boxes"]
    concepts = ["forklift", "boxes"]
    results = align_concepts_to_masks(concepts, masks, captions)
    assert [r["concept"] for r in results] == concepts
    assert results[0]["best_caption"] == "red forklift on the right"
    assert results[0]["best_mask_index"] == 0
    assert results[0]["matched"] is True
    assert results[1]["best_caption"] == "brown cardboard boxes"
    assert results[1]["best_mask_index"] == 1
    assert results[1]["matched"] is True


def test_align_concepts_unmatched_below_threshold():
    masks = _toy_masks(1)
    captions = ["red forklift"]
    results = align_concepts_to_masks(
        ["airplane"], masks, captions, score_threshold=0.5
    )
    assert results[0]["matched"] is False
    assert results[0]["score"] < 0.5


def test_align_concepts_handles_empty_masks_and_captions():
    results = align_concepts_to_masks(["forklift"], [], [])
    assert len(results) == 1
    assert results[0]["best_mask_index"] == -1
    assert results[0]["best_caption"] is None
    assert results[0]["area_fraction"] == 0.0
    assert results[0]["matched"] is False


def test_align_concepts_length_mismatch_raises():
    with pytest.raises(ValueError):
        align_concepts_to_masks(["x"], _toy_masks(2), ["one caption"])


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
def test_coverage_report_full_match():
    cot = (
        "<think>The red forklift is engaging the brown boxes on a pallet.</think>"
        "<answer>Right side.</answer>"
    )
    masks = _toy_masks(3)
    captions = ["red forklift", "brown boxes", "wooden pallet"]
    report = coverage_report(cot, masks, captions)
    assert report["n_masks"] == 3
    assert report["answer"] == "Right side."
    assert report["coverage"] > 0.0
    # forklift, boxes, pallet should each be matched
    matched_concepts = {r["concept"] for r in report["per_concept"] if r["matched"]}
    assert {"forklift", "boxes", "pallet"} <= matched_concepts


def test_coverage_report_no_concepts_means_zero_coverage():
    # Only stopwords / short tokens in the think block
    cot = "<think>the it is on of as so</think>"
    report = coverage_report(cot, _toy_masks(1), ["red forklift"])
    assert report["n_concepts"] == 0
    assert report["coverage"] == 0.0
    assert report["per_concept"] == []


def test_coverage_report_falls_back_to_raw_text_without_tags():
    cot = "no tags here but mentions a forklift"
    report = coverage_report(cot, _toy_masks(1), ["red forklift"])
    assert any(
        r["concept"] == "forklift" and r["matched"] for r in report["per_concept"]
    )


# ---------------------------------------------------------------------------
# Aligner class (smoke tests for the no-checkpoint scaffold)
# ---------------------------------------------------------------------------
def test_segcompass_config_defaults_are_reasonable():
    cfg = SegCompassConfig()
    assert cfg.sae_hidden_dim > cfg.sae_input_dim  # SAE expansion
    assert 0 < cfg.sae_top_k <= cfg.sae_hidden_dim
    assert 0 < cfg.score_threshold < 1


def test_segcompass_aligner_no_checkpoint_runs_lexical_path():
    aligner = SegCompassAligner()
    assert aligner.has_sae is False
    cot = "<think>The forklift moves the boxes.</think><answer>Yes.</answer>"
    masks = _toy_masks(2)
    captions = ["red forklift", "brown boxes"]
    report = aligner.analyze(cot, masks, captions)
    assert report["n_masks"] == 2
    assert report["coverage"] > 0


def test_segcompass_aligner_with_checkpoint_path_is_stubbed():
    # Passing a checkpoint should raise NotImplementedError until the
    # SAE loading is implemented — better than silently misleading the
    # caller into thinking they got a real SegCompass forward pass.
    with pytest.raises(NotImplementedError):
        SegCompassAligner(sae_checkpoint="/nonexistent/sae.pt")


def test_segcompass_aligner_apply_transform_writes_report_column():
    aligner = SegCompassAligner()
    example = {
        "output": "<think>The forklift moves boxes.</think><answer>Yes.</answer>",
        "masks": _toy_masks(2),
        "captions": ["red forklift", "brown boxes"],
    }
    out = aligner.apply_transform(example)
    assert "segcompass_alignment" in out
    assert out["segcompass_alignment"]["n_masks"] == 2
    assert out["segcompass_alignment"]["coverage"] > 0


def test_segcompass_aligner_apply_transform_tolerates_missing_columns():
    aligner = SegCompassAligner()
    example: dict = {}
    out = aligner.apply_transform(example)
    rep = out["segcompass_alignment"]
    assert rep["n_concepts"] == 0
    assert rep["n_masks"] == 0
    assert rep["coverage"] == 0.0
