# Literature Review: How Low Can the LoRA Rank Go? — The Rank Floor of Parameter-Efficient Fine-Tuning

_Generated 2026-06-04 for: a systematic study of LoRA rank vs. downstream accuracy on small encoders (DistilBERT/DistilRoBERTa) for SST-2 and AG News, identifying the minimum viable rank with statistical rigor against full-FT, linear-probe, and high-rank LoRA baselines._

## 1. Summary

Parameter-efficient fine-tuning (PEFT) is now the default way to adapt pretrained transformers, and Low-Rank Adaptation (LoRA, Hu et al. 2021) is its most widely used instance: it freezes the backbone and learns a low-rank update ΔW = BA scaled by α/r. The theoretical justification predates LoRA — work on the *intrinsic dimension* of objective landscapes (Li et al. 2018) and of language-model fine-tuning (Aghajanyan et al. 2020) showed that pretrained models can be adapted in a surprisingly small subspace (e.g. ~200 parameters reach 90% of full-FT on MRPC for RoBERTa). The original LoRA paper itself reports the striking result that "a rank as small as one suffices" for adapting Wq/Wv on GLUE-scale tasks, yet the field has since pushed in the *opposite* direction — AdaLoRA, DoRA, VeRA, rsLoRA and the recent rank-trade-off studies mostly explore *how to spend a rank budget well* or *how to scale to higher rank*, on large generative LLMs. The frontier is well charted for billion-parameter decoders, but there is **no clean, statistically-grounded characterization of the rank floor for small encoder classifiers**: the smallest r whose accuracy is indistinguishable from full LoRA / full fine-tuning, as a function of task difficulty, with multiple seeds, error bars, and the linear-probe (rank-0) and attention-only-vs-MLP ablations laid side by side. That gap — a rank-vs-accuracy trade-off curve that pinpoints how few trainable parameters are actually needed on DistilBERT-class models — is exactly what this project fills.

## 2. Key Papers

| # | Title | Year | Venue | Method | Datasets | Headline result | Link | Local PDF |
|---|-------|------|-------|--------|----------|-----------------|------|-----------|
| 1 | LoRA: Low-Rank Adaptation of Large Language Models | 2021 | ICLR 2022 | Low-rank update ΔW=BA, scaled α/r, on Wq/Wv | GLUE (incl. SST-2), E2E, GPT-2/3, RoBERTa, DeBERTa | r=8 on RoBERTa-base = 0.3M params, SST-2 95.1±0.2 vs full-FT 94.8; "rank as small as 1 suffices" for Wq+Wv | [arXiv 2106.09685](https://arxiv.org/abs/2106.09685) | papers/2106.09685.pdf |
| 2 | Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning | 2020 | ACL 2021 | Random-subspace reparameterization (intrinsic dim d90) | MRPC, QQP, GLUE | ~200 trainable params reach 90% of full-FT on MRPC (RoBERTa); larger models have lower intrinsic dim | [arXiv 2012.13255](https://arxiv.org/abs/2012.13255) | papers/2012.13255.pdf |
| 3 | Measuring the Intrinsic Dimension of Objective Landscapes | 2018 | ICLR 2018 | Train in random low-dim subspace; find d_int | MNIST, CIFAR, RL | Many tasks solved in a subspace far smaller than parameter count; d_int stable across model sizes | [arXiv 1804.08838](https://arxiv.org/abs/1804.08838) | papers/1804.08838.pdf |
| 4 | AdaLoRA: Adaptive Budget Allocation for PEFT | 2023 | ICLR 2023 | SVD-form ΔW; prune singular values by importance | GLUE, SQuAD, NLG | Reallocating a fixed rank budget beats uniform LoRA, esp. in low-budget regime | [arXiv 2303.10512](https://arxiv.org/abs/2303.10512) | papers/2303.10512.pdf |
| 5 | QLoRA: Efficient Finetuning of Quantized LLMs | 2023 | NeurIPS 2023 | 4-bit NF4 frozen base + LoRA adapters | Vicuna, MMLU, instruction data | Finetune 65B on one 48GB GPU; matches 16-bit full-FT quality | [arXiv 2305.14314](https://arxiv.org/abs/2305.14314) | papers/2305.14314.pdf |
| 6 | DoRA: Weight-Decomposed Low-Rank Adaptation | 2024 | ICML 2024 (Oral) | Decompose W into magnitude + direction; LoRA on direction | Commonsense reasoning, LLaVA, VL-BART | Closes LoRA↔full-FT gap; consistently > LoRA at matched/lower rank | [arXiv 2402.09353](https://arxiv.org/abs/2402.09353) | papers/2402.09353.pdf |
| 7 | VeRA: Vector-based Random Matrix Adaptation | 2023 | ICLR 2024 | Shared frozen random A,B; learn tiny scaling vectors | GLUE, E2E, image, instruct | RoBERTa-base GLUE: 0.043M params (≈7× < LoRA's 0.3M), SST-2 94.6±0.1 vs LoRA 95.1 | [arXiv 2310.11454](https://arxiv.org/abs/2310.11454) | papers/2310.11454.pdf |
| 8 | A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA) | 2023 | preprint | Replace α/r scaling with α/√r | LLM fine-tuning | α/r collapses gradients as r grows; α/√r unlocks gains from higher rank | [arXiv 2312.03732](https://arxiv.org/abs/2312.03732) | papers/2312.03732.pdf |
| 9 | LoRA Learns Less and Forgets Less | 2024 | TMLR 2024 | Empirical LoRA vs full-FT across ranks | Code & math continued pretraining/IFT | Full-FT finds rank 10–100× higher than LoRA; LoRA under-fits but forgets less (regularizer) | [arXiv 2405.09673](https://arxiv.org/abs/2405.09673) | papers/2405.09673.pdf |
| 10 | How Much is Too Much? Exploring LoRA Rank Trade-offs | 2025 | IJCNLP-AACL 2025 Findings | Rank sweep r∈{8,16,32,64,128} | GSM8K, MMLU, MedMCQA, MathQA, LegalMCQ | No single best rank; r=32–64 a practical default; **no significance testing**; large LLMs only | [arXiv 2512.15634](https://arxiv.org/abs/2512.15634) | (HTML only) |
| 11 | DistilBERT, a distilled version of BERT | 2019 | NeurIPS 2019 EMC² WS | Knowledge distillation (triple loss) | GLUE, SQuAD | 40% smaller, 60% faster, retains 97% of BERT GLUE; our backbone | [arXiv 1910.01108](https://arxiv.org/abs/1910.01108) | papers/1910.01108.pdf |
| 12 | Scaling Down to Scale Up: A Guide to PEFT | 2023 | preprint (survey) | Survey/taxonomy of PEFT | — | Taxonomy: additive / selective / reparameterization (LoRA) / hybrid | [arXiv 2303.15647](https://arxiv.org/abs/2303.15647) | papers/2303.15647.pdf |
| 13 | PEFT in Large Models: A Survey of Methodologies | 2024 | preprint (survey) | Survey, 100+ papers Jun'19–Jul'24 | — | Additive/reparam/subtractive + hybrid/quantization/multi-task | [arXiv 2410.19878](https://arxiv.org/abs/2410.19878) | papers/2410.19878.pdf |
| 14 | Parameter-Efficient Transfer Learning for NLP (Adapters) | 2019 | ICML 2019 | Bottleneck adapter modules | GLUE | Within 0.4% of full-FT adding only 3.6% params/task | [arXiv 1902.00751](https://arxiv.org/abs/1902.00751) | papers/1902.00751.pdf |
| 15 | RoBERTa: A Robustly Optimized BERT Pretraining Approach | 2019 | preprint | Better-trained BERT | GLUE, SQuAD, RACE | SOTA GLUE; base/distilled variant is our alt backbone | [arXiv 1907.11692](https://arxiv.org/abs/1907.11692) | (abstract) |
| 16 | GLUE: A Multi-Task Benchmark for NLU | 2018 | ICLR 2019 / EMNLP WS | NLU benchmark incl. SST-2 | SST-2, MRPC, QQP, MNLI… | Defines SST-2 train/dev splits + metric (accuracy) | [arXiv 1804.07461](https://arxiv.org/abs/1804.07461) | (dataset paper) |
| 17 | Character-level Convolutional Networks for Text Classification | 2015 | NeurIPS 2015 | char-CNN; introduces AG News | AG News, Yelp, DBpedia… | AG News: 4 classes, 120k train / 7.6k test | [arXiv 1509.01626](https://arxiv.org/abs/1509.01626) | (dataset paper) |
| 18 | Recursive Deep Models for Semantic Compositionality (SST) | 2013 | EMNLP 2013 | RNTN; introduces Stanford Sentiment Treebank | SST | Origin of SST/SST-2 sentiment data | [aclanthology D13-1170](https://aclanthology.org/D13-1170/) | (dataset paper) |

### Detailed notes (top papers)

**1. LoRA (Hu et al. 2021) — arXiv 2106.09685.** *Problem:* full fine-tuning of large LMs is storage- and serving-expensive (one full copy per task). *Method:* freeze pretrained W0, learn ΔW = BA where B∈R^{d×r}, A∈R^{r×k}, r≪min(d,k); scale the update by α/r; at inference merge into W0 so there is no added latency. Default applies LoRA to attention Wq and Wv only. *Key results (directly load-bearing for us):* RoBERTa-base on GLUE uses r=8 on Wq,Wv → **0.3M trainable params**, **SST-2 95.1±0.2** vs **full-FT 94.8**; the §7.2 ablation sweeps r∈{1,2,4,8,64} and finds "a rank as small as one suffices for adapting both Wq and Wv on these datasets" — the single most relevant prior result and the springboard for our study. Convention: "set α to the first r we try and do not tune it." *Limitation for us:* the rank ablation is on RoBERTa/DeBERTa large GLUE/WikiSQL tasks, reports point estimates without a multi-seed significance test, and does not contrast attention-only vs attention+MLP placement systematically or include a linear-probe floor. *Takeaway:* our project is essentially a rigorous, small-encoder, difficulty-stratified extension of LoRA §7.2.

**2. Intrinsic Dimensionality (Aghajanyan et al. 2020) — arXiv 2012.13255.** *Problem:* why does fine-tuning huge models on small data generalize? *Method:* reparameterize fine-tuning in a random d-dimensional subspace (à la Li et al. 2018) and find d90 — the smallest subspace dimension reaching 90% of full-FT. *Key result:* RoBERTa reaches 90% of full-FT on **MRPC with ~200 trainable parameters**; pretraining implicitly lowers intrinsic dimension, and bigger models have *lower* intrinsic dim. *Limitation:* uses random projections rather than the structured low-rank LoRA used in practice, and d90 is reported per task without the alpha/module-placement axes. *Takeaway:* gives the theoretical reason to expect a very low rank floor and a concrete success criterion ("90% / statistically indistinguishable").

**3. Measuring Intrinsic Dimension (Li et al. 2018) — arXiv 1804.08838.** *Problem:* how many parameters does a task really need? *Method:* train networks inside a randomly-oriented subspace of increasing dimension; the dimension where solutions first appear is the intrinsic dimension. *Key result:* many tasks are solvable in subspaces far smaller than the native parameter count, and d_int is roughly stable across very different model sizes. *Takeaway:* foundational motivation; the "find the smallest dimension that still works" methodology is exactly our rank-floor protocol, transplanted onto LoRA rank.

**4. AdaLoRA (Zhang et al. 2023) — arXiv 2303.10512.** *Problem:* a uniform rank across all weight matrices wastes budget. *Method:* parameterize ΔW in SVD form PΛQ, score singular triplets by importance, and prune unimportant ones — allocating more rank where it matters. *Key result:* beats uniform LoRA especially at low budgets on GLUE/SQuAD. *Limitation:* added machinery and importance scheduling; still aimed at allocation, not at characterizing the floor. *Takeaway:* a relevant "smarter than uniform" baseline and a reason to also test which *modules* get rank (our attention-only vs +MLP axis).

**5. QLoRA (Dettmers et al. 2023) — arXiv 2305.14314.** *Problem:* memory cost of finetuning large models. *Method:* 4-bit NF4 quantized frozen base + LoRA adapters + double quantization + paged optimizers. *Key result:* finetune a 65B model on a single 48GB GPU at 16-bit quality. *Limitation:* orthogonal to the rank-floor question (it is about memory, not minimal rank) and targets generative LLMs. *Takeaway:* context for why LoRA dominates practice; not a baseline we need on DistilBERT but worth citing for the field picture.

**6. DoRA (Liu et al. 2024) — arXiv 2402.09353.** *Problem:* LoRA still trails full-FT in accuracy. *Method:* decompose each pretrained weight into magnitude + direction; apply LoRA only to the direction. *Key result:* consistently outperforms LoRA at matched (and sometimes lower) rank on commonsense reasoning and multimodal tasks; analysis shows LoRA and full-FT have qualitatively different magnitude/direction update patterns. *Limitation:* evaluated on large LLM/VLM tasks, not small encoders. *Takeaway:* shows accuracy at a given rank is not fixed by rank alone — parameterization matters — which motivates pairing rank with the alpha/scaling and placement studies in our design.

**7. VeRA (Kopiczko et al. 2023) — arXiv 2310.11454.** *Problem:* even LoRA's parameter count can be cut. *Method:* share a single pair of *frozen random* low-rank matrices across all layers and learn only tiny per-layer scaling vectors. *Key result:* RoBERTa-base GLUE with rank 1024 yet only **0.043M trainable params** (≈7× fewer than LoRA's 0.3M), SST-2 **94.6±0.1** vs LoRA **95.1**. *Limitation:* trades a small accuracy drop for extreme parameter savings; high nominal rank but few learned params (a different efficiency axis than ours). *Takeaway:* a concrete "ultra-low-parameter" reference point and a reminder to report *learned-parameter* count, not just nominal rank.

**8. LoRA Learns Less and Forgets Less (Biderman et al. 2024) — arXiv 2405.09673.** *Problem:* when and why does LoRA underperform full-FT? *Method:* careful empirical comparison across ranks on code/math continued-pretraining and instruction-tuning. *Key result:* full-FT learns weight perturbations of rank **10–100× larger** than typical LoRA configs, which explains gaps on hard generative tasks; but LoRA forgets less, acting as a regularizer that beats weight decay/dropout. *Limitation:* large decoder LLMs and generative tasks — exactly the *high*-difficulty regime; says little about easy/medium classification where low rank may fully suffice. *Takeaway:* sharpens the difficulty hypothesis — we expect SST-2 (easy) to have a much lower rank floor than a harder task, which is why difficulty stratification is central to our design.

**(also useful) How Much is Too Much? (Rathore et al. 2025) — arXiv 2512.15634.** Sweeps r∈{8,16,32,64,128} on LLaMA-3.1-8B/Qwen-2.5-7B over Q&A datasets; finds no universally best rank, recommends r=32–64, and notes LoRA acts as a regularizer. Crucially, it reports **no confidence intervals or significance testing**, evaluates only two large instruction-tuned models, and omits Adapter/Prefix/QLoRA comparisons — it explicitly leaves the statistically-grounded, small-model, low-rank (r<8) floor unaddressed, which is our niche.

## 3. Methods Landscape

PEFT methods (surveys: Lialin et al. 2023, arXiv 2303.15647; Han/Xu et al. 2024, arXiv 2410.19878) fall into a few families:

- **Reparameterization / low-rank (most relevant).** LoRA (BA update, α/r scaling) and its descendants. Variants change *what is low-rank* (AdaLoRA: SVD + pruning), *how it is parameterized* (DoRA: magnitude+direction), *what is learned* (VeRA: frozen random matrices + scaling vectors), or *how it is scaled* (rsLoRA: α/√r to keep gradients stable as r grows). This is the family we sweep over; the open question they all sidestep is the *lower* boundary — how small r can be before accuracy degrades.
- **Additive — adapters.** Houlsby et al. 2019 bottleneck adapters; within 0.4% of full-FT on GLUE with 3.6% params. Insert serial modules with added inference latency (unlike LoRA, which merges). Useful as a conceptual rank-0-plus baseline but not our focus.
- **Additive — soft prompts / prefix tuning.** Learn continuous prompt/prefix vectors. Strong on large models, weaker and less stable on small encoders; not central here.
- **Selective.** BitFit (bias-only), layer freezing, and the limiting case of a **frozen-encoder linear probe** — train only the classification head. This is exactly our "rank-0 analogue" lower-bound baseline.
- **Quantization-hybrid.** QLoRA — orthogonal memory axis; not needed for DistilBERT on an A100.

**What suits us:** vanilla LoRA on a DistilBERT/DistilRoBERTa-base encoder, swept over r∈{1,2,4,8,16,32,64}, with the α/r vs (fixed-α, scaled-α, α/√r per rsLoRA) scaling axis, and the placement axis (attention-only Wq/Wv vs attention+MLP). Baselines: full fine-tuning (upper bound), linear probe on frozen encoder (rank-0 lower bound), and a high-rank LoRA (e.g. r=64) as the "standard LoRA" reference. Cheap enough for a single A100 and ≥3 seeds per cell.

## 4. Datasets (candidates)

| Dataset | Task | Size | Source / URL | License | Notes |
|---------|------|------|--------------|---------|-------|
| SST-2 (GLUE) | Binary sentiment (positive/negative) | 67,349 train / 872 dev / 1,821 test | HF `glue`, config `sst2` (also `stanfordnlp/sst2`) | CC0 / per GLUE terms; underlying SST from Socher et al. 2013 | **Easy task** end of difficulty axis. Test labels hidden → standard practice is to report **dev accuracy** (or hold out from train). Metric = accuracy. |
| AG News | 4-way news topic (World/Sports/Business/Sci-Tech) | 120,000 train / 7,600 test (30k+1.9k per class) | HF `ag_news` (Zhang et al. 2015) | Custom non-commercial research use (Zhang/LeCun) | **Medium-difficulty** end (4 classes, longer text). Public test split available → report **test accuracy** directly. Metric = accuracy. |
| (optional 3rd, if a harder cell is wanted) e.g. TREC / Banking77 / Yahoo Answers | Fine-grained topic/intent | 5k–1.4M | HF `datasets` | varies | Only if we want a 3rd difficulty point; statement requires ≥2, SST-2 + AG News satisfy this. |

Both load with one line via HuggingFace `datasets` (`load_dataset("glue","sst2")`, `load_dataset("ag_news")`), small enough to fine-tune DistilBERT-base in minutes per cell on one A100.

## 5. Baselines & SOTA

| Method | Dataset | Metric | Reported value | Source |
|--------|---------|--------|----------------|--------|
| Full fine-tuning (RoBERTa-base) | SST-2 (GLUE dev) | accuracy | 94.8 | Hu et al. 2021 (Table 2) |
| LoRA r=8, Wq+Wv (RoBERTa-base, 0.3M params) | SST-2 (GLUE dev) | accuracy | 95.1 ± 0.2 | Hu et al. 2021 (Table 2) |
| LoRA rank ablation r∈{1,2,4,8,64} | WikiSQL/MultiNLI (RoBERTa/DeBERTa) | accuracy | "rank as small as 1 suffices" (near-flat) | Hu et al. 2021 (§7.2, Table 6) |
| VeRA (RoBERTa-base, 0.043M params, rank 1024) | SST-2 (GLUE dev) | accuracy | 94.6 ± 0.1 | Kopiczko et al. 2023 (Table 2) |
| Adapter (Houlsby) | GLUE (avg) | accuracy | within 0.4% of full-FT, +3.6% params | Houlsby et al. 2019 |
| Intrinsic-dim random subspace (RoBERTa) | MRPC | accuracy | 90% of full-FT with ~200 params | Aghajanyan et al. 2020 |
| DistilBERT-base full-FT (general GLUE) | GLUE | accuracy | retains ~97% of BERT-base | Sanh et al. 2019 |
| Full-FT on AG News (typical BERT/DistilBERT) | AG News | test accuracy | ~94–95 (commonly reported for BERT-family) | derived from Zhang et al. 2015 setup + standard HF baselines (to be reproduced in-house) |

Note: published SST-2 LoRA numbers are for **RoBERTa-base**, not DistilBERT; AG News + DistilBERT LoRA numbers are not standardized in the literature, so our experiments must establish the in-house full-FT / linear-probe / high-rank-LoRA reference points rather than copying external values. The LoRA §7.2 ablation is the closest prior art and confirms a low floor is plausible.

## 6. Standard Metrics

- **Primary metric: classification accuracy** for both tasks (SST-2 and AG News are both reported in accuracy; this is the GLUE convention for SST-2).
- **SST-2 split caveat:** the official GLUE *test* labels are hidden (scored only via the leaderboard), so the literature reports **dev-set accuracy**. We should report dev accuracy (or carve a fixed held-out split from train with a fixed seed) and state this explicitly.
- **AG News:** has a public **test split** (7,600 examples) → report **test accuracy** directly.
- **Statistical reporting:** ≥3 seeds per cell; report mean ± std (or 95% CI), and for the "minimum viable rank" claim use a significance test (e.g. paired/Welch t-test or non-overlapping CIs) between each rank and the full-LoRA/full-FT reference — this is the rigor that prior rank studies (Hu §7.2, Rathore et al. 2025) lack.
- **Secondary reported quantities (per the statement):** trainable parameter count and **%** of total params, and wall-clock training time per cell.

## 7. Gap & Opportunity

**The gap.** Three lines of work bracket the question but none answers it directly: (a) intrinsic-dimension theory (Li 2018; Aghajanyan 2020) says the floor *should* be very low but uses random projections, not LoRA, and only on a couple of GLUE tasks; (b) the LoRA paper's §7.2 ablation hints "rank 1 suffices" but reports point estimates on large models with no seeds/significance and no linear-probe floor or placement study; (c) recent rank-trade-off work (Biderman 2024; Rathore 2025) and the newer LoRA variants (AdaLoRA, DoRA, VeRA, rsLoRA) almost exclusively target **large generative LLMs** at **mid-to-high ranks (r≥8)** and explicitly report **no statistical significance testing**. There is no clean, reproducible characterization of the **rank floor for small encoder classifiers** that (i) sweeps the very-low regime r∈{1,2,4,8,…,64}, (ii) stratifies by task difficulty (easy SST-2 vs harder AG News), (iii) includes the rank-0 linear-probe and full-FT brackets, and (iv) declares a minimum viable rank with an explicit statistical test and error bars.

**Concrete directions for this project:**
1. **The rank-floor curve.** For each (dataset, backbone), produce rank-vs-accuracy with ≥3 seeds and CIs; define the minimum viable rank as the smallest r whose accuracy is statistically indistinguishable from full-LoRA/full-FT. Expect SST-2's floor to be lower than AG News's, quantifying the difficulty→floor relationship.
2. **Bracketing baselines.** Place the linear probe (rank-0) and full fine-tuning as explicit lower/upper bounds on the same axes, plus a "standard" high-rank LoRA (r=64) reference — turning the curve into an interpretable trade-off between trainable-parameter % and accuracy.
3. **Scaling and placement ablations.** Test α scaling (fixed-α, the standard α/r, and rsLoRA's α/√r) to check whether the apparent floor is partly a scaling artifact; and attention-only (Wq/Wv) vs attention+MLP placement to see whether spending the same tiny rank on more modules lowers the floor — a small, A100-friendly factorial.
4. **Efficiency framing.** Report params, %-of-model, and training time per cell so the deliverable answers the literal question — "how few trainable parameters are actually needed" — in absolute terms for DistilBERT-class models.

## 8. Candidate References (for the paper)

1. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685.
2. Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL 2021. arXiv:2012.13255.
3. Li, C., Farkhoor, H., Liu, R., & Yosinski, J. (2018). *Measuring the Intrinsic Dimension of Objective Landscapes.* ICLR 2018. arXiv:1804.08838.
4. Zhang, Q., Chen, M., Bukharin, A., He, P., Cheng, Y., Chen, W., & Zhao, T. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR 2023. arXiv:2303.10512.
5. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314.
6. Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024. arXiv:2402.09353.
7. Kopiczko, D. J., Blankevoort, T., & Asano, Y. M. (2023). *VeRA: Vector-based Random Matrix Adaptation.* ICLR 2024. arXiv:2310.11454.
8. Kalajdzievski, D. (2023). *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* arXiv:2312.03732.
9. Biderman, D., Portes, J., Ortiz, J. J. G., et al. (2024). *LoRA Learns Less and Forgets Less.* Transactions on Machine Learning Research (TMLR). arXiv:2405.09673.
10. Rathore, D., Kumar, V., Bansal, C., & Moitra, A. (2025). *How Much is Too Much? Exploring LoRA Rank Trade-offs for Retaining Knowledge and Domain Robustness.* Findings of IJCNLP-AACL 2025. arXiv:2512.15634.
11. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* NeurIPS 2019 EMC² Workshop. arXiv:1910.01108.
12. Lialin, V., Deshpande, V., & Rumshisky, A. (2023). *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning.* arXiv:2303.15647.
13. Han, Z., Gao, C., Liu, J., Zhang, J., & Zhang, S. Q. (2024). *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* arXiv:2403.14608. (companion to the methodologies survey arXiv:2410.19878).
14. Xin, Y., et al. (2024). *Parameter-Efficient Fine-Tuning in Large Models: A Survey of Methodologies.* arXiv:2410.19878.
15. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). *Parameter-Efficient Transfer Learning for NLP.* ICML 2019. arXiv:1902.00751.
16. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
17. Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2018). *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.* ICLR 2019 / EMNLP 2018 BlackboxNLP. arXiv:1804.07461.
18. Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015. arXiv:1509.01626. (AG News dataset.)
19. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP 2013. (Stanford Sentiment Treebank / SST-2 source.)
20. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. arXiv:1810.04805. (Backbone family reference.)
