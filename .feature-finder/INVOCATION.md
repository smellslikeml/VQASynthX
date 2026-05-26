You are a coding agent implementing a recommendation from the Feature Finder
pipeline.

Read these files in order:
  1. .feature-finder/SPEC.md       — the implementation spec
  2. .feature-finder/PAPER.md      — the recommended paper details
  3. .feature-finder/CONTEXT.md    — team context (recent merges)
  4. .feature-finder/GUARDRAILS.md — what you may and may not modify

Then look at the existing codebase structure (especially the `vqasynth/`
package and `tests/` directory) to understand the project's conventions.

Implement the MINIMAL-VIABLE-SCAFFOLDING version of the spec:

- Create one new module under `vqasynth/` (likely `vqasynth/<paper_slug>_integration.py`)
  with:
    * A config dataclass (e.g. `<Paper>Config`) holding the paper's reported
      hyperparameters as defaults
    * A class scaffold for the integration entry point. Keep heavy lifting
      (external checkpoint loading, etc.) as documented TODOs so this PR
      doesn't pretend to do work that requires external dependencies.
    * Any utility functions described in the spec (pixel conversions,
      data adapters, etc.) — implement these concretely.

- Create `tests/test_<paper_slug>_integration.py` with passing tests for
  every utility function you implemented concretely. Stub-test the class
  scaffold (smoke test of the no-checkpoint path returning sensible defaults).

- Append a brief "(Paper Title) Integration (experimental) 🧪" section
  to README.md, attributing the work to Feature Finder.

Run pytest before declaring done. If tests fail, fix them or scope your
implementation down until they pass. Do not modify files outside the
guardrails allowlist.

When complete, output a one-paragraph SUMMARY of what you actually built.
Be honest about what you stubbed vs implemented.
