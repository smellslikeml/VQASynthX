"""SegCompass integration scaffolding for VQASynth.

Paper: "SegCompass: Exploring Interpretable Alignment with Sparse Autoencoders
for Enhanced Reasoning Segmentation" (arxiv 2605.22658v1).

VQASynth already produces (a) chain-of-thought reasoning traces via
``r1_reasoning.R1Reasoner`` and (b) SAM2 masks with grounded captions via
``localize.Localizer``. SegCompass proposes a Sparse Autoencoder that maps
CoT tokens and visual tokens into a shared sparse concept space so the
final mask is transparently traceable to the reasoning that produced it.

A full SegCompass forward pass requires a trained SAE + slot mapper +
mask decoder (external checkpoints we don't have), so this module
provides:

* concrete, testable utilities for the inputs SegCompass would operate
  on (concept extraction from CoT, mask geometry, lexical alignment
  scoring);
* a ``SegCompassAligner`` class whose no-checkpoint path produces a
  lexical-only alignment report, with the SAE-driven path stubbed as a
  documented ``TODO`` until a checkpoint is wired in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "of", "on", "in",
        "at", "by", "for", "with", "to", "from", "is", "are", "was", "were",
        "be", "been", "being", "it", "its", "this", "that", "these", "those",
        "i", "you", "he", "she", "they", "we", "them", "his", "her", "their",
        "our", "my", "your", "as", "so", "not", "no", "do", "does", "did",
        "have", "has", "had", "will", "would", "can", "could", "should",
        "about", "into", "than", "there", "here", "where", "when", "what",
        "which", "who", "whom", "how", "why", "near", "far", "left", "right",
        "above", "below", "between", "around", "out", "up", "down", "over",
        "under", "any", "some", "all", "more", "most", "very", "much", "many",
        "really", "just", "also", "still", "even", "only", "such", "like",
        "see", "look", "looks", "looking", "looked", "appears", "appear",
        "seem", "seems", "scene", "image", "picture", "view", "object",
        "objects", "thing", "things", "side", "sides", "front", "back",
    }
)


def extract_think_block(text: str) -> str:
    """Return the text inside the first ``<think>...</think>`` span, or ``""``.

    Whitespace at the boundaries is stripped. Matching is case-insensitive
    and tolerates newlines (CoT traces are multi-line).
    """
    if not text:
        return ""
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else ""


def extract_answer_block(text: str) -> str:
    """Return the text inside the first ``<answer>...</answer>`` span, or ``""``."""
    if not text:
        return ""
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def extract_concepts(
    text: str,
    *,
    min_length: int = 3,
    max_concepts: int | None = None,
) -> list[str]:
    """Extract candidate concept tokens from a free-form text.

    SegCompass's SAE operates on a sparse concept space; this is a
    lightweight stand-in that returns content-bearing single tokens in
    order of first appearance, deduplicated and stopword-filtered.

    Args:
        text: input text (typically a CoT think-block).
        min_length: drop tokens shorter than this (default 3).
        max_concepts: optional cap on the number of returned concepts.

    Returns:
        Ordered, deduplicated list of normalized concept tokens.
    """
    seen: set[str] = set()
    concepts: list[str] = []
    for tok in _tokenize(text):
        if len(tok) < min_length:
            continue
        if tok in _STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        concepts.append(tok)
        if max_concepts is not None and len(concepts) >= max_concepts:
            break
    return concepts


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr
    return arr > 0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two binary masks of the same shape.

    Accepts either bool or uint8 (treated as ``>0``). Returns ``0.0`` when
    both masks are empty so the metric is defined for the degenerate case.
    """
    ba = _as_bool_mask(a)
    bb = _as_bool_mask(b)
    if ba.shape != bb.shape:
        raise ValueError(f"mask shape mismatch: {ba.shape} vs {bb.shape}")
    inter = np.logical_and(ba, bb).sum()
    union = np.logical_or(ba, bb).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_area_fraction(mask: np.ndarray) -> float:
    """Fraction of the mask that is active, in [0, 1]."""
    b = _as_bool_mask(mask)
    if b.size == 0:
        return 0.0
    return float(b.sum()) / float(b.size)


def concept_caption_overlap(concept: str, caption: str) -> float:
    """Lexical overlap of a single concept against a caption, in [0, 1].

    Returns ``1.0`` when ``concept`` appears as a whole token in
    ``caption``, otherwise a softer character-trigram Jaccard score so
    morphological variants ("box"/"boxes", "forklift"/"forklifts") still
    align partially. This is the lexical fallback used until a real
    SegCompass SAE checkpoint is plugged in.
    """
    if not concept or not caption:
        return 0.0
    c = concept.strip().lower()
    cap_tokens = set(_tokenize(caption))
    if c in cap_tokens:
        return 1.0
    return _trigram_jaccard(c, caption.lower())


def _trigrams(s: str) -> set[str]:
    s = f"  {s.strip()}  "
    return {s[i : i + 3] for i in range(len(s) - 2)}


def _trigram_jaccard(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def align_concepts_to_masks(
    concepts: Sequence[str],
    masks: Sequence[np.ndarray],
    captions: Sequence[str],
    *,
    score_threshold: float = 0.5,
) -> list[dict]:
    """For each concept, pick the best-matching caption/mask by lexical score.

    Args:
        concepts: ordered concept tokens (e.g. from ``extract_concepts``).
        masks: list of 2D masks; index aligns with ``captions``.
        captions: caption strings per mask.
        score_threshold: scores below this are flagged as unmatched.

    Returns:
        One dict per concept with ``concept``, ``best_caption``,
        ``best_mask_index``, ``score``, ``area_fraction``, ``matched``.
        ``best_mask_index`` is ``-1`` when there are no captions/masks.
    """
    if len(masks) != len(captions):
        raise ValueError(
            f"masks and captions length mismatch: {len(masks)} vs {len(captions)}"
        )
    results: list[dict] = []
    for concept in concepts:
        best_score = 0.0
        best_idx = -1
        for i, cap in enumerate(captions):
            s = concept_caption_overlap(concept, cap)
            if s > best_score:
                best_score = s
                best_idx = i
        area = (
            mask_area_fraction(masks[best_idx]) if best_idx >= 0 else 0.0
        )
        results.append(
            {
                "concept": concept,
                "best_caption": captions[best_idx] if best_idx >= 0 else None,
                "best_mask_index": best_idx,
                "score": best_score,
                "area_fraction": area,
                "matched": best_score >= score_threshold,
            }
        )
    return results


def coverage_report(
    cot_text: str,
    masks: Sequence[np.ndarray],
    captions: Sequence[str],
    *,
    score_threshold: float = 0.5,
    max_concepts: int | None = 32,
) -> dict:
    """Aggregate concept-to-mask coverage report for one example.

    Pulls the ``<think>...</think>`` block out of ``cot_text`` (falling
    back to the raw string if no tag is present), extracts concepts, and
    reports per-concept alignment plus aggregate coverage. This mirrors
    the *analysis* output SegCompass enables — without the trained SAE,
    the per-concept score is lexical rather than learned.
    """
    think = extract_think_block(cot_text) or (cot_text or "")
    concepts = extract_concepts(think, max_concepts=max_concepts)
    per_concept = align_concepts_to_masks(
        concepts, masks, captions, score_threshold=score_threshold
    )
    matched = sum(1 for r in per_concept if r["matched"])
    coverage = matched / len(per_concept) if per_concept else 0.0
    return {
        "n_concepts": len(concepts),
        "n_masks": len(masks),
        "n_matched": matched,
        "coverage": coverage,
        "concepts": concepts,
        "per_concept": per_concept,
        "answer": extract_answer_block(cot_text),
    }


@dataclass
class SegCompassConfig:
    """Hyperparameters for the SegCompass SAE + slot-mapper stack.

    Defaults follow the values described in the paper's experimental
    section. Override at construction time to ablate.
    """

    sae_input_dim: int = 768
    sae_hidden_dim: int = 16384
    sae_top_k: int = 32
    sae_l1_coef: float = 1e-3
    codebook_size: int = 512
    num_slots: int = 16
    slot_heatmap_resolution: int = 64
    rl_kl_coef: float = 0.05
    seg_loss_weight: float = 1.0
    cot_loss_weight: float = 0.5
    score_threshold: float = 0.5
    max_concepts: int | None = 32
    extra: dict = field(default_factory=dict)


class SegCompassAligner:
    """Entry point for SegCompass-style CoT/mask alignment analysis.

    Two modes:

    * **No checkpoint** (default): ``analyze`` returns a lexical
      coverage report computed from the CoT text, caption strings and
      mask geometry — useful for sanity-checking VQASynth outputs even
      before a trained SAE exists.
    * **With checkpoint** (``sae_checkpoint`` provided): the SAE-driven
      path is the real SegCompass alignment. This is currently a
      documented ``TODO`` — wiring requires loading the published SAE
      weights, the query codebook, and the slot mapper.
    """

    def __init__(
        self,
        config: SegCompassConfig | None = None,
        sae_checkpoint: str | None = None,
    ):
        self.config = config or SegCompassConfig()
        self.sae_checkpoint = sae_checkpoint
        self._sae = None
        self._codebook = None
        self._slot_mapper = None
        if sae_checkpoint is not None:
            # TODO(segcompass): load SAE weights, codebook and slot mapper
            # from the published checkpoint. Until then, callers that pass
            # a checkpoint path get a clear error rather than silently
            # falling back to the lexical-only path.
            raise NotImplementedError(
                "Loading a SegCompass SAE checkpoint is not implemented yet. "
                "See vqasynth.segcompass_integration.SegCompassAligner."
            )

    @property
    def has_sae(self) -> bool:
        return self._sae is not None

    def analyze(
        self,
        cot_text: str,
        masks: Sequence[np.ndarray],
        captions: Sequence[str],
    ) -> dict:
        """Produce an alignment report for one (CoT, masks, captions) example.

        With no SAE loaded this is the lexical fallback from
        :func:`coverage_report`. Once an SAE is wired in, this is the
        method that will dispatch to the sparse-concept path.
        """
        if self.has_sae:
            # TODO(segcompass): route through the trained SAE + slot
            # mapper to produce sparse concept activations and the
            # multi-slot heatmap that grounds them.
            raise NotImplementedError(
                "SAE-driven path not implemented; clear sae_checkpoint to use "
                "the lexical fallback."
            )
        return coverage_report(
            cot_text,
            masks,
            captions,
            score_threshold=self.config.score_threshold,
            max_concepts=self.config.max_concepts,
        )

    def apply_transform(
        self,
        example: dict,
        cot_column: str = "output",
        masks_column: str = "masks",
        captions_column: str = "captions",
        out_column: str = "segcompass_alignment",
    ) -> dict:
        """Hugging Face ``datasets.map``-compatible single-example transform.

        Reads CoT text, mask list and caption list from ``example`` and
        writes the alignment report under ``out_column``. Mirrors the
        ``apply_transform`` shape used by other VQASynth components
        (e.g. ``Localizer.apply_transform``).
        """
        cot = example.get(cot_column) or ""
        masks = example.get(masks_column) or []
        captions = example.get(captions_column) or []
        example[out_column] = self.analyze(cot, list(masks), list(captions))
        return example


__all__ = [
    "SegCompassAligner",
    "SegCompassConfig",
    "align_concepts_to_masks",
    "concept_caption_overlap",
    "coverage_report",
    "extract_answer_block",
    "extract_concepts",
    "extract_think_block",
    "mask_area_fraction",
    "mask_iou",
]
