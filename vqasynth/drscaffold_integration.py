"""
DRScaffold integration scaffolding.

Adapted from "DRScaffold: Boosting Dense-Scene Reasoning in Lightweight Vision
Language Models" (arXiv:2605.26038). DRScaffold proposes a supervised
fine-tuning framework that decomposes the supervision target into four
causally ordered stages (entities, attributes, relations, conclusion) and a
companion benchmark, DRBench (14,573 questions across 2,943 images,
five task categories spanning three progressive reasoning layers).

This module is experimental scaffolding produced by the Feature Finder
pipeline. It provides:

- ``DRScaffoldConfig``: a dataclass holding the paper's reported defaults.
- ``DRScaffoldIntegration``: an entry-point class scaffold whose heavy
  lifting (loading external checkpoints, running fine-tuning, calling
  DRBench loaders) is intentionally left as documented TODOs.
- Concretely-implemented utility functions for stage parsing/formatting,
  bounding-box pixel conversions in the Qwen2.5-VL convention used by the
  paper, and DRBench task-category to reasoning-layer mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# DRBench task taxonomy
# ---------------------------------------------------------------------------
# Five task categories across three progressive reasoning layers, as
# described in the DRScaffold paper. The mapping is exposed so the
# evaluation pipeline (vqasynth.benchmarks) can roll DRBench results up
# alongside the existing SpatialScore / OmniSpatial / SpaCE-10 / MindCube
# layouts.

DRBENCH_TASK_CATEGORIES: Tuple[str, ...] = (
    "object_existence",
    "attribute_recognition",
    "spatial_relation",
    "counting",
    "compositional_reasoning",
)

DRBENCH_REASONING_LAYERS: Tuple[str, ...] = (
    "perception",
    "grounding",
    "inference",
)

_CATEGORY_TO_LAYER: Dict[str, str] = {
    "object_existence": "perception",
    "attribute_recognition": "perception",
    "spatial_relation": "grounding",
    "counting": "grounding",
    "compositional_reasoning": "inference",
}


# ---------------------------------------------------------------------------
# Four causally ordered scaffolding stages
# ---------------------------------------------------------------------------
# Stage names mirror the four-stage decomposition introduced by DRScaffold.
# They are emitted in fixed order inside the assistant response, wrapped by
# XML-style tags so they can be parsed back out for scoring or analysis.

SCAFFOLD_STAGES: Tuple[str, ...] = (
    "entities",
    "attributes",
    "relations",
    "conclusion",
)

_STAGE_TAG_RE = {
    stage: re.compile(rf"<{stage}>\s*(.*?)\s*</{stage}>", re.IGNORECASE | re.DOTALL)
    for stage in SCAFFOLD_STAGES
}


# ---------------------------------------------------------------------------
# Bounding-box pixel conversions
# ---------------------------------------------------------------------------
# DRScaffold trains on Qwen2.5-VL, which emits bounding boxes on a fixed
# 0-1000 normalized canvas regardless of input resolution. The helpers below
# convert between that representation and pixel coordinates so the four
# scaffolding stages can be aligned with actual image regions during
# evaluation.

QWEN_BBOX_CANVAS: int = 1000


def normalize_bbox(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    canvas: int = QWEN_BBOX_CANVAS,
) -> Tuple[int, int, int, int]:
    """Convert a pixel-space ``[x1, y1, x2, y2]`` bbox to Qwen-style canvas coords.

    The result is integer-quantized onto ``[0, canvas]`` and clamped to that
    range. Raises ``ValueError`` for non-positive image dimensions.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    if len(bbox) != 4:
        raise ValueError("bbox must have exactly four elements: [x1, y1, x2, y2]")

    x1, y1, x2, y2 = bbox
    scale_x = canvas / image_width
    scale_y = canvas / image_height

    def _clamp(v: float) -> int:
        return max(0, min(canvas, int(round(v))))

    return (
        _clamp(x1 * scale_x),
        _clamp(y1 * scale_y),
        _clamp(x2 * scale_x),
        _clamp(y2 * scale_y),
    )


def denormalize_bbox(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    canvas: int = QWEN_BBOX_CANVAS,
) -> Tuple[int, int, int, int]:
    """Inverse of :func:`normalize_bbox` — Qwen canvas coords back to pixels."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    if len(bbox) != 4:
        raise ValueError("bbox must have exactly four elements: [x1, y1, x2, y2]")

    x1, y1, x2, y2 = bbox
    scale_x = image_width / canvas
    scale_y = image_height / canvas

    return (
        max(0, min(image_width, int(round(x1 * scale_x)))),
        max(0, min(image_height, int(round(y1 * scale_y)))),
        max(0, min(image_width, int(round(x2 * scale_x)))),
        max(0, min(image_height, int(round(y2 * scale_y)))),
    )


# ---------------------------------------------------------------------------
# Scaffolded-response parsing / formatting
# ---------------------------------------------------------------------------

def format_scaffolded_target(
    entities: str,
    attributes: str,
    relations: str,
    conclusion: str,
) -> str:
    """Render the four causally ordered stages into a single supervision target.

    The output uses lowercased XML-style tags in the canonical order so it
    can be round-tripped through :func:`parse_scaffolded_response`.
    """
    return (
        f"<entities>{entities.strip()}</entities>\n"
        f"<attributes>{attributes.strip()}</attributes>\n"
        f"<relations>{relations.strip()}</relations>\n"
        f"<conclusion>{conclusion.strip()}</conclusion>"
    )


def parse_scaffolded_response(text: str) -> Dict[str, Optional[str]]:
    """Extract the four DRScaffold stages from a model response.

    Missing stages map to ``None``. Tags are matched case-insensitively and
    may span multiple lines.
    """
    out: Dict[str, Optional[str]] = {stage: None for stage in SCAFFOLD_STAGES}
    for stage, pattern in _STAGE_TAG_RE.items():
        match = pattern.search(text)
        if match:
            out[stage] = match.group(1).strip()
    return out


def compute_stage_completeness(text: str) -> float:
    """Return the fraction (0..1) of the four scaffolding stages present in ``text``.

    Used as a fast diagnostic for whether a fine-tuned model has learned the
    causal-ordering supervision target.
    """
    parsed = parse_scaffolded_response(text)
    present = sum(1 for v in parsed.values() if v)
    return present / len(SCAFFOLD_STAGES)


def classify_drbench_layer(task_category: str) -> str:
    """Map a DRBench task category to its progressive reasoning layer.

    Raises ``ValueError`` for unknown categories so calling code does not
    silently miscategorize new tasks.
    """
    key = task_category.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _CATEGORY_TO_LAYER:
        valid = ", ".join(DRBENCH_TASK_CATEGORIES)
        raise ValueError(
            f"Unknown DRBench task category '{task_category}'. Valid: {valid}"
        )
    return _CATEGORY_TO_LAYER[key]


def adapt_vqasynth_example(
    example: Dict,
    image_column: str = "images",
    conversation_column: str = "messages",
) -> Dict:
    """Adapt a VQASynth dataset row into DRScaffold's expected input shape.

    Pulls the user question and assistant answer out of the conversation and
    repackages them with the image so downstream supervised fine-tuning can
    treat the assistant text as the scaffolding target. Rows without a clear
    user→assistant turn pair are returned with ``question`` / ``answer`` set
    to ``None`` so the caller can filter them out.
    """
    images = example.get(image_column)
    conversation = example.get(conversation_column) or []

    question: Optional[str] = None
    answer: Optional[str] = None
    for i in range(len(conversation) - 1):
        cur, nxt = conversation[i], conversation[i + 1]
        if cur.get("role") == "user" and nxt.get("role") == "assistant":
            q_texts = [
                c.get("text", "")
                for c in cur.get("content", [])
                if c.get("type") == "text" and c.get("text")
            ]
            a_texts = [
                c.get("text", "")
                for c in nxt.get("content", [])
                if c.get("type") == "text" and c.get("text")
            ]
            if q_texts and a_texts:
                question, answer = q_texts[0], a_texts[0]
                break

    return {
        "image": images,
        "question": question,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Config + integration entry-point scaffolding
# ---------------------------------------------------------------------------

@dataclass
class DRScaffoldConfig:
    """Reported hyperparameters for DRScaffold supervised fine-tuning.

    Defaults follow Table 3 / Appendix of arXiv:2605.26038. Override any
    field when wiring this into a custom training run.
    """

    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    learning_rate: float = 1e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_seq_length: int = 4096
    image_max_pixels: int = 1280 * 28 * 28
    seed: int = 42
    stages: Tuple[str, ...] = field(default_factory=lambda: SCAFFOLD_STAGES)
    bbox_canvas: int = QWEN_BBOX_CANVAS
    checkpoint_path: Optional[str] = None
    drbench_split: str = "test"


class DRScaffoldIntegration:
    """Entry-point scaffold for DRScaffold fine-tuning + inference.

    This class deliberately does not load model weights or DRBench data on
    construction; doing so would require ``transformers``, ``torch``, and
    pulling >GB checkpoints from HuggingFace, which is out of scope for the
    initial scaffolding PR. The TODO methods below mark where that work
    plugs in.
    """

    def __init__(self, config: Optional[DRScaffoldConfig] = None) -> None:
        self.config = config or DRScaffoldConfig()
        self._model = None
        self._processor = None

    # --- lifecycle ----------------------------------------------------------

    def load_checkpoint(self, path: Optional[str] = None) -> None:
        """Load a DRScaffold-finetuned checkpoint.

        TODO: instantiate ``AutoModelForVision2Seq`` and ``AutoProcessor``
        from ``path`` (or ``self.config.checkpoint_path``) and assign them
        to ``self._model`` / ``self._processor``. Left unimplemented in the
        scaffolding PR so we do not pretend to perform work that requires
        external checkpoint downloads.
        """
        raise NotImplementedError(
            "load_checkpoint is a TODO in the DRScaffold scaffolding PR"
        )

    def is_ready(self) -> bool:
        """Return True when a checkpoint has been loaded."""
        return self._model is not None and self._processor is not None

    # --- inference path -----------------------------------------------------

    def predict(self, image, question: str) -> Dict[str, Optional[str]]:
        """Run scaffolded inference on a single (image, question) pair.

        When no checkpoint is loaded, returns a sensible default: empty
        stages and a ``conclusion`` of ``None`` so downstream scoring can
        cleanly mark the row as unscored rather than crashing.
        """
        if not self.is_ready():
            return {stage: None for stage in SCAFFOLD_STAGES}
        # TODO: tokenize (image, question) via self._processor, generate
        # with self._model, then parse the result with parse_scaffolded_response.
        raise NotImplementedError(
            "predict with a loaded checkpoint is a TODO in the scaffolding PR"
        )

    # --- evaluation path ----------------------------------------------------

    def score_predictions(
        self,
        predictions: List[Dict[str, Optional[str]]],
    ) -> Dict[str, float]:
        """Aggregate scaffolding completeness across a batch of predictions.

        Returns a dict with the mean fraction of stages produced and the
        per-stage presence rate. Used as a fast diagnostic over a held-out
        DRBench slice without requiring the full DRBench scorer.
        """
        if not predictions:
            return {"completeness": 0.0, **{f"present_{s}": 0.0 for s in SCAFFOLD_STAGES}}

        per_stage_present = {s: 0 for s in SCAFFOLD_STAGES}
        for pred in predictions:
            for stage in SCAFFOLD_STAGES:
                if pred.get(stage):
                    per_stage_present[stage] += 1

        n = len(predictions)
        result = {
            f"present_{s}": per_stage_present[s] / n for s in SCAFFOLD_STAGES
        }
        result["completeness"] = sum(result[f"present_{s}"] for s in SCAFFOLD_STAGES) / len(
            SCAFFOLD_STAGES
        )
        return result
