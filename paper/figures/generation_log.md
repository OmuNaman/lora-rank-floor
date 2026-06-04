# Figure Generation Report — lora-rank-floor

- **Total: 6** (1 schematic diagram + 5 data plots) | PASS first try: 4 | PASS after fix: 2 | FAIL: 0
- **Method:** all figures built with code-based **matplotlib** from the REAL ledger
  (`experiments/results.jsonl`, 162 rows) via `make_figures.py`. Chosen over the PaperBanana
  Gemini pipeline for exact numbers, CI error bars, log axes, shaded significance bands, and full
  styling control with no API dependency (the skill explicitly permits the matplotlib route for plots).
  All saved at 300 dpi, white background, into `output/`.
- **Every plot maps to real `results.jsonl` rows** (means ± 95% CI, 3 seeds, Welch-t-derived r*).

| # | file | type | status | attempts | notes |
|---|------|------|--------|----------|-------|
| 1 | fig1_rank_floor_curve.png | plot | PASS | 1 | HERO. Acc vs rank (log2), 95% CI, full-FT band + rank-0 probe lines, r* stars (SST-2 r=4, AG News r=32). |
| 2 | fig2_method_paperbanana.png | diagram | PASS | 1 | **Used in paper.** PaperBanana `generate` (Nano Banana Pro `gemini-3-pro-image`, VLM `gemini-3.1-pro-preview`), key from project-root `.env`. Critic: "publication-ready", 1 iteration. Verified: correct labels, padlock-for-frozen, W0+B/A zoom, head, bracket strip. |
| 2b | fig2_method_overview.png | diagram | PASS (fixed) | 2 | matplotlib schematic (kept as fallback). Fix: removed ❄ tofu glyph; replaced literal `ℝ^{768×768}`. |
| 3 | fig3_accuracy_vs_params.png | plot | PASS | 1 | Pareto: acc vs adapter-param % (log), r annotations, full-FT references. |
| 4 | fig4_placement_ablation.png | plot | PASS | 1 | attn-only vs attn+MLP grouped bars, both datasets, full-FT line. |
| 5 | fig5_scaling_ablation.png | plot | PASS | 1 | const s=2 / fixed α=8 / rsLoRA grouped bars (SST-2). |
| 6 | fig6_data_efficiency.png | plot | PASS (fixed) | 2 | acc vs train size. Fix: full-FT recolored black (was green, colliding with LoRA r=16). |

All figures verified by reading the rendered PNGs. No FAILs; all six are paper-ready.
