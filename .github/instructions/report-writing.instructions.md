---
description: "Use when drafting report sections, generating publication figures/tables, editing LaTeX/Markdown manuscripts, or preparing presentation materials. Covers NeurIPS 8-page constraints, scientific framing, and mandatory course disclosures."
applyTo: "report/**, docs/REPORT_BLUEPRINT.md"
---
# Academic Report & Visualization Guidelines

## Course Grading Priorities (MSDM 5054)
- **Strict 8-page limit** in NeurIPS conference format. Prioritize compact multi-panel figures (3–4 figures + 3 tables total across the whole paper).
- **Scientific reasoning > raw accuracy tables**: Frame every section around a testable hypothesis, controlled experiment, and Statistical ML interpretation (bias-variance trade-off, model capacity under low SNR, inductive bias, calibration, distribution shift). Negative or inverted-U findings are equally valuable when rigorously explained.
- **Mandatory sections**:
  1. Explicit bulleted **Team Member Contributions**;
  2. Explicit **AI / Large Language Model Usage Statement** summarizing how AI assisted coding, debugging, and polishing.

## Figure & Table Standards
- Export figures at `dpi=300` (`.pdf` vector for LaTeX + `.png` preview for Markdown/Slides) with consistent colorblind-safe palettes (`seaborn` `colorblind` or `viridis`/`coolwarm` for Grad-CAM overlays).
- Report OOS Accuracy in `%` (2 decimal places, e.g., `52.84%`), Cross-Entropy Loss (4 decimals), AUC (4 decimals), ECE (4 decimals), and annualized Equal-Weighted Long-Short Decile Sharpe Ratio (2 decimals).
