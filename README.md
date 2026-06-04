# How Low Can the Rank Go?

*Finding the LoRA rank floor on small transformer encoders (DistilBERT, SST-2, AG News).*

A statistically grounded study of the **LoRA rank floor**: how few trainable parameters a small
transformer encoder actually needs to match full fine-tuning. We fine-tune DistilBERT-base on SST-2
(easy) and AG News (harder), sweep rank `r ∈ {1,2,4,8,16,32,64}` with 3 seeds per cell, bracket it with
a frozen-encoder rank-0 probe and full fine-tuning, and define the **minimum viable rank** as the smallest
`r` whose accuracy is statistically indistinguishable (Welch t-test, α=0.05) from full fine-tuning and
high-rank LoRA.

📄 **Paper:** [paper/paper.pdf](paper/paper.pdf) — IEEE conference format, 7 pages
🌐 **Project page:** https://omunaman.github.io/lora-rank-floor/

## Summary
- **Problem:** LoRA's rank `r` is the single knob trading capacity for trainable parameters, but the *lower
  boundary* — how low `r` can go before adaptation degrades — has never been characterized with statistical
  rigor for small encoder classifiers.
- **Approach:** a controlled, difficulty-stratified rank sweep on DistilBERT-base with 3 seeds, 95% CIs, and
  Welch t-tests, plus α-scaling, module-placement, and data-efficiency ablations. **162 real runs on Modal
  A100 GPUs, 0 errors.**
- **Key result:** the rank floor is astonishingly low and tracks task difficulty — **r=4 on SST-2** (0.11% of
  parameters) and **r=32 on AG News** under strict equivalence (≈4 under a practical 0.5% tolerance). Even
  **r=1** (0.027% of parameters) closes 59–72% of the probe→full-FT gap. The only true breakdown is at rank 0.

## Headline results
Test accuracy (mean over 3 seeds). Full table in [experiments/experiments_log.md](experiments/experiments_log.md).

| Task | Full FT | Probe (r=0) | LoRA r=1 | LoRA r=4 | LoRA r=64 | Min viable rank |
|------|--------:|------------:|---------:|---------:|----------:|-----------------|
| **SST-2** (dev) | 0.903 | 0.845 | 0.887 | **0.898** | 0.896 | **r = 4** (0.11% params) |
| **AG News** (test) | 0.946 | 0.917 | 0.934 | 0.939 | 0.945 | **r = 32** strict / ≈4 practical |

Additional findings:
- **Scaling-robust:** the floor barely moves across `α/r`, fixed-α, and rsLoRA `α/√r`; at r=1 a larger scaling recovers ~0.5 pt.
- **Placement:** attention+MLP lowers the *rank* floor (AG News attn+MLP r=16 ≈ full-FT) but attention-only is most *parameter*-efficient.
- **Data efficiency:** at 500–10,000 training examples, low-rank LoRA **matches or beats** full fine-tuning (regularization effect).

## Reproduce
Experiments run on [Modal](https://modal.com) serverless A100 GPUs.
```bash
pip install modal && modal setup
cd experiments/modal
# smoke test (1 cell)
python -m modal run modal_app.py --config-json @configs_smoke.json
# full core grid / ablations / data-efficiency (batched, parallel containers)
python -m modal run modal_app.py --config-json @configs_core.json
python -m modal run modal_app.py --config-json @configs_ablation.json
python -m modal run modal_app.py --config-json @configs_dataeff.json
```
Then aggregate + analyze locally:
```bash
python experiments/analyze.py            # minimum-viable-rank significance tests
python experiments/analyze_ablations.py  # ablation + data-efficiency tables
python paper/figures/make_figures.py     # regenerate all 6 figures from the ledger
```

## Repository layout
- `paper/` — LaTeX source (`paper.tex`), compiled `paper.pdf`, and figures (`figures/output/*.png` + `make_figures.py`)
- `experiments/` — Modal training app (`modal/modal_app.py`), config generators, the append-only results ledger (`results.jsonl`), analysis scripts, and the narrative log
- `plan/` — the experiment plan (matrix, success criteria, stop conditions)
- `literature/` — the literature review
- `STATUS.md` — pipeline state and key-decision log

## Citation
```bibtex
@misc{kudale2026lorarankfloor,
  title  = {How Low Can the Rank Go?},
  author = {Kudale, Prit and Dwivedi, Naman and Dandekar, Raj and Dandekar, Rajat and Panat, Sreedath},
  year   = {2026},
  note   = {Vizuara AI Labs}
}
```

---
*Produced by an autonomous research pipeline (literature review → experiment planning → Modal GPU experiments → figures → paper → repo → website). All numbers are real, from `experiments/results.jsonl`.*
