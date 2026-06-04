# Experiment Plan: How Low Can the LoRA Rank Go? — Finding the Minimum Viable Rank

_Generated 2026-06-04_

## Hypothesis

For small pretrained encoders (DistilBERT-base) on text classification, the LoRA update needed
to recover full-fine-tuning accuracy lives in a **very low-rank subspace** — so low that accuracy
is statistically flat across a wide range of ranks and only collapses toward the frozen-encoder
(rank-0) probe at the extreme. We further hypothesize the **rank floor scales with task difficulty**:
the easy binary task (SST-2) will be statistically indistinguishable from high-rank LoRA / full-FT
at a smaller rank than the harder 4-way task (AG News). Confirmation = a rank-vs-accuracy curve
where, for each dataset, there exists a smallest rank r\* whose mean accuracy is **not significantly
worse** (Welch's t-test, α=0.05, and overlapping 95% CIs) than r=64 LoRA and full fine-tuning, with
r\*(SST-2) ≤ r\*(AG News).

## Approach

**Primary approach.** Vanilla LoRA (Hu et al. 2021) on `distilbert-base-uncased`, sweeping rank
r ∈ {1, 2, 4, 8, 16, 32, 64} on two datasets of differing difficulty, ≥3 seeds per cell, with the
frozen-encoder head-only probe (rank-0) and full fine-tuning as explicit lower/upper brackets, and
r=64 LoRA as the "standard high-rank LoRA" reference. For every cell we log test accuracy, total
trainable params, **adapter-only trainable params** (the part rank actually controls), %-of-model,
and wall-clock training time.

**Why DistilBERT-base-uncased (66M).** Canonical small encoder, well-supported by HF + PEFT, retains
~97% of BERT GLUE quality, trains in minutes on one A100 → lets us afford ≥3 seeds across the whole
grid. (DistilRoBERTa is a possible robustness extension if budget allows; kept out of the core grid
to control compute and isolate one backbone.)

**Critical design note — the head is a fixed cost.** `DistilBertForSequenceClassification` has a
freshly-initialized head (`pre_classifier`: 768→768 ≈ 590k params, + `classifier`: 768→C) that
**must** be trainable in every configuration or the model cannot classify. So the head (~0.9% of the
model) is a constant present in LoRA, full-FT, and the probe alike. We therefore report **adapter
params separately** from head params; the rank-floor question is about *adapter* capacity on top of
a fixed head. The frozen-encoder probe (train head only, encoder frozen) is precisely the **rank-0
analogue**: same head, zero adapter.

**Alternatives compared (ablations, not separate methods):**
1. **α-scaling axis** — does the apparent floor depend on the LoRA scaling rule? Compare the main
   sweep's constant effective scaling (α=2r ⇒ s=α/r=2) against fixed α=8 (s=8/r, grows at low r) and
   rsLoRA-style α/√r (Kalajdzievski 2023). Tells us whether a "broken" low rank is really capacity or
   just a scaling/optimization artifact.
2. **Module-placement axis** — attention-only (`q_lin`,`v_lin`, the LoRA Wq/Wv convention) vs
   attention+MLP (`q_lin`,`v_lin`,`lin1`,`lin2`). Does spending the same tiny rank on more modules
   lower the floor?

## Datasets

| Dataset | Download method | Size | Preprocessing | Split |
|---------|-----------------|------|---------------|-------|
| SST-2 (GLUE) | HF `datasets`: `load_dataset("glue","sst2")` | 67,349 train / 872 dev / 1,821 test (labels hidden) | tokenize (DistilBERT tokenizer), max_len 128, truncate/pad | train on train; **report dev accuracy** (test labels hidden — GLUE convention). |
| AG News | HF `datasets`: `load_dataset("ag_news")` | 120,000 train / 7,600 test | concat title+desc, tokenize, max_len 128 | train on train; **report test accuracy** (public test split). |

Both download programmatically (no auth). Cached to a Modal Volume so repeated runs don't re-download.

## Baselines

| Baseline | Source | Reported metric | We reproduce? |
|----------|--------|-----------------|---------------|
| Full fine-tuning (upper bound) | Hu et al. 2021; Sanh et al. 2019 | RoBERTa-base SST-2 dev 94.8; DistilBERT retains ~97% BERT GLUE | **Yes** — in-house DistilBERT numbers (no standard DistilBERT SST-2/AG News LoRA refs exist) |
| Frozen-encoder head-only probe (rank-0 lower bound) | Selective PEFT / linear-probe convention | — | **Yes** — train head only, encoder frozen |
| Standard high-rank LoRA (r=64) | Hu et al. 2021; Rathore et al. 2025 (r=32–64 default) | "rank 1 suffices" §7.2 | **Yes** — it is the top of our sweep; serves as the reference for the significance test |

## Success Criterion

- **Primary metric:** classification **accuracy** (SST-2 dev, AG News test), mean ± std over ≥3 seeds.
- **Study-level success (the loop's "done" gate)** — ALL of:
  1. **Reference sanity gates pass:** full-FT achieves **≥ 0.89** on SST-2 (dev) and **≥ 0.93** on AG News
     (test); r=64 LoRA is within **1.0 pt** of full-FT on each; the frozen probe is **clearly below**
     full-FT (establishing a non-degenerate bracket). These anchor the curve to credible references
     (DistilBERT SST-2 ≈ 0.90–0.91, AG News ≈ 0.94–0.95 are the literature expectations).
  2. **The full grid completes** with valid metrics for every (dataset × rank × seed) cell + baselines.
  3. **A minimum viable rank r\* is identifiable** on each dataset via the significance test (Welch
     t-test p>0.05 vs r=64 LoRA AND overlapping 95% CIs). The rank-vs-accuracy trade-off curve is the
     deliverable.
- **Secondary metrics:** total trainable params, adapter-only params, %-of-model trainable, wall-clock
  training time per cell; (optional) F1 for AG News.

## Stop Conditions

- **Stop when** the full grid is complete AND the three sanity gates pass AND r\* is determined for both
  datasets, **OR**
- **Stop after 12 experiment rounds** (max-iteration cap). A "round" = one Modal launch batch. Planned
  rounds: (1) smoke test, (2) SST-2 core grid, (3) AG News core grid, (4) ablations, then up to 8
  remediation/extension rounds if a sanity gate fails or the floor is not cleanly bracketed (e.g. add
  intermediate ranks, more seeds, fix instability). Never infinite.

## Improvement Rule (what to try, in order, if a sanity gate fails or results look wrong)

1. **Reference too low** (full-FT or r=64 below sanity gate): raise epochs (SST-2 3→4, AG News 2→3),
   tune LR (full-FT 2e-5; LoRA 5e-4 → try {3e-4, 1e-3}), add warmup (10%), check label mapping/tokenization.
2. **Low-rank instability / NaNs:** lower LR for that cell, enable warmup, verify α-scaling (very large
   s=α/r at low r can destabilize — this is exactly what the α-ablation diagnoses).
3. **Floor not bracketed** (no rank is significantly worse, so r\*=1 trivially): confirm the probe (rank-0)
   is genuinely below — if even the probe matches, the task is too easy; report that finding and (optional)
   add a harder dataset or reduce training data to expose the floor.
4. **High seed variance** blurring significance: add seeds (3→5) for the contested ranks only.
5. **Compute pressure:** AG News dominates cost — if needed, cap AG News train epochs at 2 (it converges
   fast on 120k); never fabricate — only reduce scope, and record it in STATUS.md.

## Experiment Matrix

Each row below is an **experiment group**; every group runs over **seeds {0,1,2}** (≥3 per cell, per the
statement). Configs are dicts the Modal training script consumes directly. Backbone = `distilbert-base-uncased`,
batch_size 32, max_len 128, AdamW, linear schedule + 10% warmup, fp16. LoRA dropout 0.05.

Defaults: SST-2 epochs=3, AG News epochs=2 (probe uses +2 epochs since only the head trains). LoRA LR=5e-4;
full-FT LR=2e-5; probe LR=1e-3. Main sweep uses **attention-only** target modules `[q_lin, v_lin]` and
**α=2r** (constant effective scaling s=2). All on **A100-40GB** (DistilBERT @ bs32/len128 uses <3 GB — 40GB is
ample; 80GB not needed).

| group_id | type | dataset | method | rank r | α (scaling) | target modules | epochs | LR | seeds | GPU | est. runtime/seed |
|----------|------|---------|--------|--------|-------------|----------------|--------|------|-------|-----|-------------------|
| exp00 | **smoke** | sst2 (2k subset) | lora | 8 | 16 (α/r) | q_lin,v_lin | 1 | 5e-4 | 1 | A100 | <1 min |
| **— SST-2 core grid —** |
| ft-sst2 | baseline | sst2 | full_ft | — | — | (all) | 3 | 2e-5 | 0,1,2 | A100 | ~3 min |
| lp-sst2 | baseline | sst2 | probe | 0 | — | head only (encoder frozen) | 5 | 1e-3 | 0,1,2 | A100 | ~2 min |
| lora-sst2-r1 | sweep | sst2 | lora | 1 | 2 (α/r) | q_lin,v_lin | 3 | 5e-4 | 0,1,2 | A100 | ~3 min |
| lora-sst2-r2 | sweep | sst2 | lora | 2 | 4 | q_lin,v_lin | 3 | 5e-4 | 0,1,2 | A100 | ~3 min |
| lora-sst2-r4 | sweep | sst2 | lora | 4 | 8 | q_lin,v_lin | 3 | 5e-4 | 0,1,2 | A100 | ~3 min |
| lora-sst2-r8 | sweep | sst2 | lora | 8 | 16 | q_lin,v_lin | 3 | 5e-4 | 0,1,2 | A100 | ~3 min |
| lora-sst2-r16 | sweep | sst2 | lora | 16 | 32 | q_lin,v_lin | 3 | 5e-4 | 0,1,2 | A100 | ~3 min |
| lora-sst2-r32 | sweep | sst2 | lora | 32 | 64 | q_lin,v_lin | 3 | 5e-4 | 0,1,2 | A100 | ~3 min |
| lora-sst2-r64 | sweep/ref | sst2 | lora | 64 | 128 | q_lin,v_lin | 3 | 5e-4 | 0,1,2 | A100 | ~3 min |
| **— AG News core grid —** |
| ft-agnews | baseline | ag_news | full_ft | — | — | (all) | 2 | 2e-5 | 0,1,2 | A100 | ~7 min |
| lp-agnews | baseline | ag_news | probe | 0 | — | head only (encoder frozen) | 4 | 1e-3 | 0,1,2 | A100 | ~5 min |
| lora-agnews-r{1,2,4,8,16,32,64} | sweep | ag_news | lora | 1..64 | 2r (α/r=2) | q_lin,v_lin | 2 | 5e-4 | 0,1,2 | A100 | ~6 min |
| **— Ablation A: α-scaling (SST-2, ranks 1,4,16) —** |
| absc-sst2-fixed8 | ablation | sst2 | lora | 1,4,16 | 8 fixed (α/r=8/r) | q_lin,v_lin | 3 | 5e-4 | 0,1,2 | A100 | ~3 min |
| absc-sst2-rslora | ablation | sst2 | lora | 1,4,16 | 2r, **α/√r** scaling | q_lin,v_lin | 3 | 5e-4 | 0,1,2 | A100 | ~3 min |
| _(main-sweep α=2r at r∈{1,4,16} is the 3rd comparison point — already computed)_ |
| **— Ablation B: module placement (attention+MLP, ranks 1,4,16) —** |
| place-sst2-attnmlp | ablation | sst2 | lora | 1,4,16 | 2r (α/r=2) | q_lin,v_lin,lin1,lin2 | 3 | 5e-4 | 0,1,2 | A100 | ~4 min |
| place-agnews-attnmlp | ablation | ag_news | lora | 1,4,16 | 2r (α/r=2) | q_lin,v_lin,lin1,lin2 | 2 | 5e-4 | 0,1,2 | A100 | ~7 min |

**Run-count budget.** Core grids: SST-2 (2 baselines + 7 ranks)×3 = 27; AG News (2 + 7)×3 = 27. Ablations:
A = 2 modes × 3 ranks × 3 = 18 (SST-2); B = 2 datasets × 3 ranks × 3 = 18. Total ≈ **90 runs** + smoke.
DistilBERT is tiny, so to amortize container cold-start and the (cached) model/data download, the
run-experiments stage should pack many configs **per Modal container** (loop over configs in one GPU
session, reuse loaded dataset/model) rather than one container per run. Estimated GPU time ≈ 5–7 A100-hours
(SST-2 cells ~3 min, AG News cells ~6–7 min) — comfortably a "small compute budget" on a single A100.

## Risks

- **HF dataset/hub availability or schema drift.** Mitigation: cache to Modal Volume on first run; pin
  `datasets`/`transformers`/`peft` versions; `glue/sst2` columns = `sentence,label,idx`, `ag_news` = `text,label`.
- **OOM:** not expected (DistilBERT @ bs32/len128 ≪ 40GB). Mitigation: A100-40GB, fp16, gradient accumulation if ever needed.
- **Low-rank training instability (large s=α/r at r=1 with fixed α):** diagnosed by the α-ablation; mitigate with warmup + the α=2r constant-scaling default for the main sweep.
- **SST-2 test labels hidden:** report **dev** accuracy (stated explicitly) — standard GLUE practice; no leaderboard submission needed.
- **"Floor too low to see" (everything ties, even rank-1):** that is itself a publishable finding ("rank 1 suffices for DistilBERT-class encoders"); if even the rank-0 probe ties, note the task is saturated and (optional remediation) add a harder/low-data condition to expose the floor.
- **Cost overrun from AG News × many cells:** pack configs per container; cap AG News at 2 epochs; the 12-round cap bounds total spend.
