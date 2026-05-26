"""
sense_integration.py — Integration scaffolding for the SENSE framework
(Zhang et al. 2026, https://arxiv.org/abs/2605.19289) into the VQASynth
synthetic data pipeline.

SENSE is a model-agnostic framework that produces synthetic images with
dense scene composition and fine instance fidelity, shown to yield more
discriminative spatial representations for downstream segmentation tasks.
Reference implementation: https://github.com/zhang0jhon/SENSE.

This module provides the integration POINT — the class interface, config
plumbing, and pixel-value standardization needed to slot SENSE-generated
data into VQASynth's existing pipeline alongside the current scene
synthesis stage. The actual SENSE checkpoint loading is left as a TODO;
it depends on the SENSE codebase's API stability and license review.

Status: experimental, opt-in. Downstream stages (localize.py, scene_fusion.py,
evaluation.py) require no changes — the SENSE generator emits
``np.uint8`` arrays in ``[0, 255]`` that are drop-in compatible with the
existing Localizer (Molmo + SAM2) and Scene Fusion stages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SENSEConfig:
    """Configuration for the SENSE synthetic data generator.

    Defaults mirror the SENSE paper's reported hyperparameters
    (Section 4.2, Table 3). Per-dataset tuning may shift these:
    Cityscapes / COCO / ADE20K each have slightly different optimal
    settings in the paper's benchmarks.

    Attributes:
        checkpoint_path: Path or HF Hub id of a SENSE-compatible
            diffusion checkpoint. When ``None``, ``SENSEGenerator``
            stays in scaffold mode and callers should fall back to the
            standard VQASynth scene generator.
        min_instances_per_image: SENSE finds dense scenes
            (``n_instances >= 8``) yield more discriminative spatial
            representations. Sampling will reject scenes with fewer
            instances.
        guidance_scale: Classifier-free guidance strength. Paper default 7.5.
        num_inference_steps: Diffusion sampling steps. Paper default 50.
        output_range: SENSE's diffusion output pixel value range.
        target_range_for_downstream: Range expected by downstream
            VQASynth stages (Localizer expects uint8 [0, 255]).
    """

    checkpoint_path: Optional[str] = None
    min_instances_per_image: int = 8
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    output_range: tuple[float, float] = (0.0, 1.0)
    target_range_for_downstream: tuple[float, float] = (0.0, 255.0)


def standardize_pixel_range(
    image: Union[np.ndarray, Image.Image],
    source_range: tuple[float, float] = (0.0, 1.0),
    target_range: tuple[float, float] = (0.0, 255.0),
) -> np.ndarray:
    """Convert pixel values between ranges.

    SENSE's diffusion output lives in ``[0, 1]``; VQASynth's downstream
    Localizer (Molmo + SAM2) expects ``[0, 255]`` uint8. This function
    handles either-direction conversion idempotently.

    Args:
        image: Input image, either a numpy array or a PIL Image.
        source_range: ``(lo, hi)`` of the input pixel values.
        target_range: ``(lo, hi)`` of the desired output range. When
            ``(0.0, 255.0)`` the result is cast to ``uint8``; otherwise
            it's returned as ``float32``.

    Returns:
        Numpy array with pixel values in ``target_range``.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image, dtype=np.float32)
    else:
        image = np.asarray(image, dtype=np.float32)

    src_lo, src_hi = source_range
    tgt_lo, tgt_hi = target_range

    # Clip to source range to handle slight overshoot from diffusion sampling
    image = np.clip(image, src_lo, src_hi)
    normalized = (image - src_lo) / (src_hi - src_lo + 1e-9)
    scaled = normalized * (tgt_hi - tgt_lo) + tgt_lo

    # Standard uint8 conversion for downstream Molmo / SAM2 compatibility
    if (tgt_lo, tgt_hi) == (0.0, 255.0):
        return np.clip(scaled, 0, 255).astype(np.uint8)
    return scaled.astype(np.float32)


class SENSEGenerator:
    """Wraps SENSE-style synthetic image generation for VQASynth.

    Interface mirrors VQASynth's existing scene-generation stage: takes
    a prompt, generates an image, returns a ``(image_array, metadata)``
    tuple that downstream stages (``localize.py``, ``scene_fusion.py``)
    can consume directly.

    Currently a SCAFFOLD — actual SENSE model loading is a TODO. The
    interface and pixel-range plumbing are in place so the downstream
    pipeline doesn't need to change once a SENSE checkpoint is wired in.

    Example:
        >>> from vqasynth.sense_integration import SENSEConfig, SENSEGenerator
        >>> cfg = SENSEConfig(checkpoint_path="zhang0jhon/sense-base")
        >>> gen = SENSEGenerator(cfg)
        >>> img, meta = gen.generate("a dense urban street scene at dusk")
        >>> # img is np.uint8 [0, 255], drop-in for localize.py
    """

    def __init__(self, config: Optional[SENSEConfig] = None):
        self.config = config or SENSEConfig()
        self._model = None  # Lazy-loaded on first generate() call.
        if self.config.checkpoint_path is None:
            logger.warning(
                "SENSEGenerator initialized without a checkpoint path — "
                "generate() will return (None, metadata). Set "
                "config.checkpoint_path to enable SENSE-based generation."
            )

    def _ensure_loaded(self):
        """Load the SENSE checkpoint on first use.

        TODO(feature-finder/sense): wire up against
        https://github.com/zhang0jhon/SENSE once its checkpoint format +
        loading API stabilizes. Suggested integration shape::

            from sense.models import SENSEUNet
            self._model = SENSEUNet.from_pretrained(self.config.checkpoint_path)

        Until then, ``self._model`` remains ``None`` and ``generate()``
        returns ``(None, metadata)`` so callers can branch on
        availability without exception handling.
        """
        if self._model is not None or self.config.checkpoint_path is None:
            return
        logger.info(
            "SENSEGenerator: checkpoint loading is a TODO (see "
            "vqasynth/sense_integration.py). Falling back — downstream "
            "callers should use VQASynth's default scene generator."
        )

    def generate(self, prompt: str) -> tuple[Optional[np.ndarray], dict]:
        """Generate one image from a prompt.

        Args:
            prompt: Natural-language description of the desired scene.
                SENSE's training data emphasizes dense compositions, so
                prompts describing scenes with multiple instances tend
                to produce the highest-utility synthetic data.

        Returns:
            Tuple of ``(image, metadata)``:

            - ``image``: ``uint8`` numpy array in ``[0, 255]``, shape
              ``(H, W, 3)``. Suitable for direct handoff to
              ``localize.py``. ``None`` if SENSE is not loaded.
            - ``metadata``: ``dict`` with ``prompt``, ``model``,
              ``instances_target``, ``guidance_scale``, and any
              SENSE-specific diagnostic fields surfaced by the model.
        """
        self._ensure_loaded()
        metadata = {
            "prompt": prompt,
            "model": "SENSE" if self._model else None,
            "instances_target": self.config.min_instances_per_image,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
        }
        if self._model is None:
            return None, metadata

        # TODO(feature-finder/sense): replace with actual SENSE forward
        # pass once the checkpoint is wired in. The scaffolding below
        # documents the expected shape:
        #
        #   raw = self._model.sample(
        #       prompt=prompt,
        #       guidance_scale=self.config.guidance_scale,
        #       num_inference_steps=self.config.num_inference_steps,
        #   )
        #   image_uint8 = standardize_pixel_range(
        #       raw,
        #       source_range=self.config.output_range,
        #       target_range=self.config.target_range_for_downstream,
        #   )
        #   metadata["instances_detected"] = self._count_instances(image_uint8)
        #   return image_uint8, metadata
        raise NotImplementedError(
            "SENSE forward pass not yet wired in. See "
            "vqasynth/sense_integration.py:SENSEGenerator._ensure_loaded."
        )


__all__ = ["SENSEConfig", "SENSEGenerator", "standardize_pixel_range"]
