# Team context (Gemini-extracted from remyxai/VQASynth merge history)

Recently shipped by the team (sample of their experiment history):
- 2025-06-01 — Fix multi-gpu OOM in localize step [General]
  Addressed out-of-memory errors during the location refinement stage when using multiple GPUs. This improves the stability and scalability of the spatial VQA pipeline.
- 2026-01-05 — fix: align non-square handling with VGGT (image + masks) [General]
  Corrected the handling of non-square images and masks in scene fusion to align with VGGT standards. This ensures consistent and accurate processing of diverse image aspect ratios.
- 2026-04-25 — Bump litellm to 1.83.13 to fix critical security vulnerabilities [General]
  Updated the `litellm` dependency to version 1.83.13 to resolve critical and high-severity security vulnerabilities. This enhances the overall security posture of the system.
- 2026-04-27 — Add evaluation stage with multi-benchmark spatial reasoning metrics [Evaluation]
  Introduced a new evaluation module and pipeline stage, incorporating scoring methods from four spatial reasoning benchmarks like SpatialScore and OmniSpatial. This enables comprehensive assessment of model performance.
