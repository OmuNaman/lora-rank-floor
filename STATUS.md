# Research Status: How Low Can the Rank Go?

- **Slug**: lora-rank-floor
- **Statement**: Systematically study the relationship between LoRA rank and downstream text-classification performance for small transformers, and find the minimum viable rank.
- **Started**: 2026-06-04
- **Author block**: Prit Kudale¹ (prit.kudale@gmail.com), Naman Dwivedi², Raj Dandekar², Rajat Dandekar², Sreedath Panat² — ²Vizuara AI Labs, hello@vizuara.com
- **Repo**: https://github.com/OmuNaman/lora-rank-floor
- **Website**: https://omunaman.github.io/lora-rank-floor/

## Pipeline Stages
| # | Stage | Status | Artifact | Notes |
|---|-------|--------|----------|-------|
| 1 | Literature review     | ✅ done | literature/literature_review.md | 18 papers, 13 PDFs; gap = rank-floor for small encoders |
| 2 | Experiment planning   | ✅ done | plan/experiment_plan.md | ~90 runs; r∈{1..64}×2 datasets×3 seeds + FT/probe/r64 + α & placement ablations; 12-round cap |
| 3 | Experiments (Modal)   | ✅ done | experiments/experiments_log.md | 162 cells, 0 errors, 4/12 rounds: core grid + α & placement ablations + data-efficiency. All gates pass |
| 4 | Paper outline         | ✅ done | paper/outline.md | 🟢 full empirical study; 6 figs, 4 tables, 4 eqs, 20 refs |
| 5 | Figures               | ✅ done | paper/figures/output/ | 6/6 figs PASS (matplotlib, real data, 300dpi); all visually verified |
| 6 | Write paper           | ✅ done | paper/paper.tex | IEEE conf; 6 figs, 4 tables, 4 eqs, 19 refs; real numbers only |
| 7 | Compile paper         | ✅ done | paper/paper.pdf | 7 pages, clean build (latexmk/MiKTeX), 0 undefined refs, 0 overfull, all figs embedded |
| 8 | GitHub repo           | ✅ done | https://github.com/OmuNaman/lora-rank-floor | public; code+results+figs+paper.pdf pushed |
| 9 | Project website       | ✅ done | https://omunaman.github.io/lora-rank-floor/ | static page (docs/), GitHub Pages enabled (main/docs) |

## Key Decisions Log
- 2026-06-04: Slug `lora-rank-floor`. Modal CLI invoked via `python -m modal` (not on PATH directly); profile `teamvizuara` authenticated. `gh` logged in as OmuNaman.
- 2026-06-04: Lit review confirms closest prior art = LoRA §7.2 ("rank 1 suffices") + intrinsic-dim (~200 params → 90% full-FT). Our niche: statistically-grounded rank floor for DistilBERT-class encoders, difficulty-stratified, with linear-probe (rank-0) + full-FT brackets and α-scaling + module-placement ablations. Published SST-2 LoRA numbers are RoBERTa-base, not DistilBERT → must establish in-house references.
- 2026-06-04: Modal app batches many cells/container via `run_cells.map(batches)` (amortizes tokenization + cold start); HF cache on Volume; per-cell commit to `/vol/cells/<exp_id>.json` (crash-safe). Param accounting splits adapter vs head vs total — head (~592k, 0.88%) is a fixed cost in every config; rank only controls adapter params. Forced `PYTHONUTF8=1` for Modal CLI on Windows (cp1252 ✓ encode crash).

## Current Best Result
**Core grid complete (54/54, 0 errors). All sanity gates pass.**
- **SST-2** (full-FT 0.9029, probe 0.8452): **minimum viable rank = 4** (0.8979, indist. from full-FT & r64; 0.11% params). r=1 already closes ~72% of probe→FT gap with 18k adapter params (0.027%).
- **AG News** (full-FT 0.9457, probe 0.9169): **minimum viable rank = 32** (strict stat. equiv.); **r≈4** under a practical ≤0.5% tolerance.
- Headline: rank floor is real, low, and **tracks task difficulty** (easy→r4, harder→r32). Probe (rank-0) decisively worse on both → low-rank adapter does necessary work.
- Round 3 ablations (α-scaling + module placement) running.
