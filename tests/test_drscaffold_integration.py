"""Tests for vqasynth.drscaffold_integration."""

import pytest

from vqasynth.drscaffold_integration import (
    DRBENCH_REASONING_LAYERS,
    DRBENCH_TASK_CATEGORIES,
    QWEN_BBOX_CANVAS,
    SCAFFOLD_STAGES,
    DRScaffoldConfig,
    DRScaffoldIntegration,
    adapt_vqasynth_example,
    classify_drbench_layer,
    compute_stage_completeness,
    denormalize_bbox,
    format_scaffolded_target,
    normalize_bbox,
    parse_scaffolded_response,
)


# ---------------------------------------------------------------------------
# Bounding-box pixel conversions
# ---------------------------------------------------------------------------

class TestNormalizeBbox:
    def test_full_image_maps_to_full_canvas(self):
        assert normalize_bbox([0, 0, 640, 480], 640, 480) == (0, 0, 1000, 1000)

    def test_half_image(self):
        assert normalize_bbox([0, 0, 320, 240], 640, 480) == (0, 0, 500, 500)

    def test_clamps_overflow_to_canvas(self):
        out = normalize_bbox([0, 0, 1000, 1000], 640, 480)
        assert out[2] == QWEN_BBOX_CANVAS
        assert out[3] == QWEN_BBOX_CANVAS

    def test_negative_input_clamps_to_zero(self):
        assert normalize_bbox([-10, -10, 100, 100], 640, 480)[0] == 0
        assert normalize_bbox([-10, -10, 100, 100], 640, 480)[1] == 0

    def test_custom_canvas(self):
        assert normalize_bbox([0, 0, 640, 480], 640, 480, canvas=500) == (
            0,
            0,
            500,
            500,
        )

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            normalize_bbox([0, 0, 1, 1], 0, 480)
        with pytest.raises(ValueError):
            normalize_bbox([0, 0, 1, 1], 640, -1)

    def test_invalid_bbox_shape_raises(self):
        with pytest.raises(ValueError):
            normalize_bbox([0, 0, 1], 640, 480)


class TestDenormalizeBbox:
    def test_round_trip(self):
        original = (10, 20, 320, 240)
        canvas = normalize_bbox(original, 640, 480)
        back = denormalize_bbox(canvas, 640, 480)
        # Quantization introduces +/-1 pixel error at most.
        for a, b in zip(original, back):
            assert abs(a - b) <= 1

    def test_zero_maps_to_zero(self):
        assert denormalize_bbox([0, 0, 0, 0], 640, 480) == (0, 0, 0, 0)

    def test_full_canvas_maps_to_full_image(self):
        assert denormalize_bbox(
            [0, 0, QWEN_BBOX_CANVAS, QWEN_BBOX_CANVAS], 640, 480
        ) == (0, 0, 640, 480)

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            denormalize_bbox([0, 0, 1, 1], 0, 480)

    def test_invalid_bbox_shape_raises(self):
        with pytest.raises(ValueError):
            denormalize_bbox([0, 0, 1], 640, 480)


# ---------------------------------------------------------------------------
# Scaffold parse / format
# ---------------------------------------------------------------------------

class TestFormatScaffoldedTarget:
    def test_emits_all_four_stages_in_order(self):
        out = format_scaffolded_target("a chair", "wooden", "left of table", "yes")
        assert "<entities>a chair</entities>" in out
        assert "<attributes>wooden</attributes>" in out
        assert "<relations>left of table</relations>" in out
        assert "<conclusion>yes</conclusion>" in out
        assert out.index("<entities>") < out.index("<attributes>")
        assert out.index("<attributes>") < out.index("<relations>")
        assert out.index("<relations>") < out.index("<conclusion>")

    def test_strips_whitespace(self):
        out = format_scaffolded_target("  a  ", "b", "c", "d")
        assert "<entities>a</entities>" in out


class TestParseScaffoldedResponse:
    def test_round_trip(self):
        original = format_scaffolded_target("chair", "wooden", "near table", "yes")
        parsed = parse_scaffolded_response(original)
        assert parsed["entities"] == "chair"
        assert parsed["attributes"] == "wooden"
        assert parsed["relations"] == "near table"
        assert parsed["conclusion"] == "yes"

    def test_missing_stages_are_none(self):
        parsed = parse_scaffolded_response("<entities>chair</entities>")
        assert parsed["entities"] == "chair"
        assert parsed["attributes"] is None
        assert parsed["relations"] is None
        assert parsed["conclusion"] is None

    def test_case_insensitive(self):
        parsed = parse_scaffolded_response("<Entities>X</Entities>")
        assert parsed["entities"] == "X"

    def test_multiline_content(self):
        text = "<entities>line one\nline two</entities>"
        assert parse_scaffolded_response(text)["entities"] == "line one\nline two"

    def test_empty_text(self):
        parsed = parse_scaffolded_response("")
        assert all(v is None for v in parsed.values())


class TestComputeStageCompleteness:
    def test_full_response_is_one(self):
        text = format_scaffolded_target("a", "b", "c", "d")
        assert compute_stage_completeness(text) == 1.0

    def test_empty_response_is_zero(self):
        assert compute_stage_completeness("nothing here") == 0.0

    def test_partial_response(self):
        text = "<entities>a</entities><conclusion>z</conclusion>"
        assert compute_stage_completeness(text) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# DRBench taxonomy
# ---------------------------------------------------------------------------

class TestClassifyDrbenchLayer:
    def test_known_categories(self):
        assert classify_drbench_layer("object_existence") == "perception"
        assert classify_drbench_layer("attribute_recognition") == "perception"
        assert classify_drbench_layer("spatial_relation") == "grounding"
        assert classify_drbench_layer("counting") == "grounding"
        assert classify_drbench_layer("compositional_reasoning") == "inference"

    def test_normalizes_separators_and_case(self):
        assert classify_drbench_layer("Spatial Relation") == "grounding"
        assert classify_drbench_layer("Spatial-Relation") == "grounding"
        assert classify_drbench_layer("  COUNTING  ") == "grounding"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            classify_drbench_layer("hallucinated_category")

    def test_taxonomy_constants_consistent(self):
        # Every declared task category resolves to a declared layer.
        for cat in DRBENCH_TASK_CATEGORIES:
            assert classify_drbench_layer(cat) in DRBENCH_REASONING_LAYERS


# ---------------------------------------------------------------------------
# VQASynth dataset adapter
# ---------------------------------------------------------------------------

class TestAdaptVqasynthExample:
    def _example(self, conversation):
        return {"images": "fake-image", "messages": conversation}

    def test_extracts_first_user_assistant_pair(self):
        ex = self._example(
            [
                {"role": "user", "content": [{"type": "text", "text": "Q1?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "A1"}]},
                {"role": "user", "content": [{"type": "text", "text": "Q2?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "A2"}]},
            ]
        )
        out = adapt_vqasynth_example(ex)
        assert out["question"] == "Q1?"
        assert out["answer"] == "A1"
        assert out["image"] == "fake-image"

    def test_no_pair_returns_nones(self):
        out = adapt_vqasynth_example(self._example([]))
        assert out["question"] is None
        assert out["answer"] is None

    def test_skips_non_text_content(self):
        ex = self._example(
            [
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
            ]
        )
        out = adapt_vqasynth_example(ex)
        assert out["question"] is None
        assert out["answer"] is None

    def test_custom_columns(self):
        ex = {
            "img": "X",
            "convo": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            ],
        }
        out = adapt_vqasynth_example(ex, image_column="img", conversation_column="convo")
        assert out["image"] == "X"
        assert out["question"] == "hi"
        assert out["answer"] == "ok"


# ---------------------------------------------------------------------------
# Class scaffolding smoke tests
# ---------------------------------------------------------------------------

class TestDRScaffoldConfig:
    def test_defaults_match_paper(self):
        cfg = DRScaffoldConfig()
        assert "Qwen2.5-VL-3B" in cfg.base_model
        assert cfg.num_epochs == 3
        assert cfg.learning_rate == 1e-5
        assert cfg.bbox_canvas == QWEN_BBOX_CANVAS
        assert tuple(cfg.stages) == SCAFFOLD_STAGES

    def test_override(self):
        cfg = DRScaffoldConfig(num_epochs=5, learning_rate=5e-6)
        assert cfg.num_epochs == 5
        assert cfg.learning_rate == 5e-6


class TestDRScaffoldIntegration:
    def test_starts_not_ready(self):
        integ = DRScaffoldIntegration()
        assert integ.is_ready() is False

    def test_predict_without_checkpoint_returns_empty_stages(self):
        integ = DRScaffoldIntegration()
        out = integ.predict(image=None, question="anything?")
        assert set(out.keys()) == set(SCAFFOLD_STAGES)
        assert all(v is None for v in out.values())

    def test_load_checkpoint_is_todo(self):
        integ = DRScaffoldIntegration()
        with pytest.raises(NotImplementedError):
            integ.load_checkpoint("/tmp/does-not-exist")

    def test_score_predictions_aggregates(self):
        integ = DRScaffoldIntegration()
        # Two predictions: first has all stages; second has only entities.
        preds = [
            {"entities": "a", "attributes": "b", "relations": "c", "conclusion": "d"},
            {"entities": "a", "attributes": None, "relations": None, "conclusion": None},
        ]
        result = integ.score_predictions(preds)
        assert result["present_entities"] == 1.0
        assert result["present_attributes"] == 0.5
        assert result["present_relations"] == 0.5
        assert result["present_conclusion"] == 0.5
        # Average across the four stages: (1 + 0.5 + 0.5 + 0.5) / 4 = 0.625
        assert result["completeness"] == pytest.approx(0.625)

    def test_score_predictions_empty(self):
        integ = DRScaffoldIntegration()
        result = integ.score_predictions([])
        assert result["completeness"] == 0.0
        for stage in SCAFFOLD_STAGES:
            assert result[f"present_{stage}"] == 0.0

    def test_accepts_custom_config(self):
        cfg = DRScaffoldConfig(num_epochs=7)
        integ = DRScaffoldIntegration(config=cfg)
        assert integ.config.num_epochs == 7
