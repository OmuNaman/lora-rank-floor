# Research Statement

**Date:** 2026-06-04

How low can the LoRA rank go before parameter-efficient fine-tuning breaks down? Systematically study the relationship between LoRA rank and downstream performance when fine-tuning a small pretrained transformer (DistilBERT or DistilRoBERTa, base size) on text classification. Sweep rank r across a wide range (e.g. r = 1, 2, 4, 8, 16, 32, 64) on at least two datasets of differing difficulty (e.g. SST-2 sentiment and AG News topic classification), and identify the minimum viable rank — the smallest r whose accuracy is statistically indistinguishable from full LoRA / full fine-tuning. Compare against three baselines: full fine-tuning, a frozen-encoder linear probe (rank 0 analogue), and standard high-rank LoRA. Also examine the effect of the LoRA alpha/scaling and which modules get adapted (attention-only vs attention+MLP). Report, for every configuration, test accuracy, the number and percentage of trainable parameters, and training time, with at least 3 random seeds per cell and error bars. The central deliverable is a rank-vs-accuracy trade-off curve that pinpoints how few trainable parameters are actually needed. Keep models small and datasets modest so all experiments fit on a single A100 within a small compute budget.

## User notes / constraints
- It's acceptable to run additional experiments if something looks wrong — iterate as needed.
- Experiments must be **concrete and real**, executed on Modal (the GPU workspace). Write code freely.
- Read each stage's skill before entering that stage.
