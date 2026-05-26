"""Tests for vqasynth.sense_integration.

The pixel-range conversion logic is the only fully-implemented piece
of the SENSE scaffolding (the model loading is a TODO), so it's the
only thing we test concretely. SENSEGenerator's interface is covered
by a smoke test that verifies the no-checkpoint path returns
``(None, metadata)`` as documented.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vqasynth.sense_integration import (
    SENSEConfig,
    SENSEGenerator,
    standardize_pixel_range,
)


class TestStandardizePixelRange:
    """The SENSE → downstream pixel conversion is load-bearing for
    drop-in compatibility with localize.py (Molmo + SAM2 expect uint8)."""

    def test_unit_range_to_uint8(self):
        """SENSE's [0, 1] output → [0, 255] uint8 for Molmo."""
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        result = standardize_pixel_range(img, (0.0, 1.0), (0.0, 255.0))
        assert result.dtype == np.uint8
        assert result.shape == (1, 1, 3)
        np.testing.assert_array_equal(result, np.array([[[0, 127, 255]]], dtype=np.uint8))

    def test_uint8_to_unit_range(self):
        """Reverse direction: [0, 255] → [0, 1] float32."""
        img = np.array([[[0, 127, 255]]], dtype=np.uint8)
        result = standardize_pixel_range(img, (0.0, 255.0), (0.0, 1.0))
        assert result.dtype == np.float32
        np.testing.assert_allclose(
            result, np.array([[[0.0, 0.49803922, 1.0]]]), atol=1e-5
        )

    def test_pil_image_input(self):
        """PIL Images are converted via np.asarray."""
        pil = Image.fromarray(np.array([[[10, 20, 30]]], dtype=np.uint8))
        result = standardize_pixel_range(pil, (0.0, 255.0), (0.0, 1.0))
        assert result.dtype == np.float32
        np.testing.assert_allclose(
            result[0, 0], np.array([10 / 255, 20 / 255, 30 / 255]), atol=1e-5
        )

    def test_overshoot_clipped(self):
        """Slight diffusion overshoot beyond [0, 1] is clipped, not propagated."""
        img = np.array([[[-0.1, 0.5, 1.1]]], dtype=np.float32)
        result = standardize_pixel_range(img, (0.0, 1.0), (0.0, 255.0))
        np.testing.assert_array_equal(result, np.array([[[0, 127, 255]]], dtype=np.uint8))

    def test_round_trip_lossy_but_bounded(self):
        """uint8 → float → uint8 round-trip stays within ±1 due to quantization."""
        original = np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)
        float_form = standardize_pixel_range(original, (0.0, 255.0), (0.0, 1.0))
        back = standardize_pixel_range(float_form, (0.0, 1.0), (0.0, 255.0))
        # Quantization noise is bounded by ±1 in the uint8 domain
        assert np.abs(back.astype(int) - original.astype(int)).max() <= 1


class TestSENSEGenerator:
    """Smoke tests for the SENSEGenerator interface (model loading
    itself is a TODO and not tested here)."""

    def test_no_checkpoint_returns_none(self):
        """Without a checkpoint path the generator stays in scaffold mode
        — generate() returns (None, metadata) instead of raising."""
        gen = SENSEGenerator(SENSEConfig())
        img, meta = gen.generate("a test prompt")
        assert img is None
        assert meta["prompt"] == "a test prompt"
        assert meta["model"] is None
        assert meta["instances_target"] == 8  # default

    def test_config_defaults_match_paper(self):
        """Defaults mirror the SENSE paper's reported hyperparameters."""
        cfg = SENSEConfig()
        assert cfg.guidance_scale == 7.5
        assert cfg.num_inference_steps == 50
        assert cfg.min_instances_per_image == 8
        assert cfg.output_range == (0.0, 1.0)
        assert cfg.target_range_for_downstream == (0.0, 255.0)

    def test_metadata_carries_config(self):
        """Generated metadata exposes the active config so downstream
        eval (SpatialScore et al.) can tag SENSE-sourced samples."""
        cfg = SENSEConfig(min_instances_per_image=12, guidance_scale=9.0)
        gen = SENSEGenerator(cfg)
        _, meta = gen.generate("scene")
        assert meta["instances_target"] == 12
        assert meta["guidance_scale"] == 9.0
