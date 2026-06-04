"""Generate all publication figures for the lora-rank-floor paper from the REAL ledger.
Reads experiments/results.jsonl, computes mean / 95% CI per config, renders 6 figures @300 dpi
into paper/figures/output/. Pure matplotlib (exact numbers, full control, no external API)."""
import json
import os
import math
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/lora-rank-floor
LED = os.path.join(ROOT, "experiments", "results.jsonl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

# ---- publication style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9.5, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
})
C = {"sst2": "#2166ac", "ag_news": "#b2182b", "ft": "#1a9850", "probe": "#7f7f7f",
     "attn": "#2166ac", "attn_mlp": "#f4a582", "s2": "#2166ac", "a8": "#92c5de", "rs": "#f4a582",
     "r1": "#d6604d", "r4": "#4393c3", "r16": "#1a9850"}
DS_NAME = {"sst2": "SST-2 (binary sentiment)", "ag_news": "AG News (4-way topic)"}


def load():
    by_id = {}
    for line in open(LED):
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        if m.get("status") == "error" or "accuracy" not in m:
            continue
        by_id[m["exp_id"]] = m
    return list(by_id.values())


ROWS = load()


def accs(pred):
    return [m["accuracy"] for m in ROWS if pred(m)]


def stat(xs):
    if not xs:
        return None
    m = sum(xs) / len(xs)
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    tmult = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(xs), 1.96)
    ci = tmult * s / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, s, ci


RANKS = [1, 2, 4, 8, 16, 32, 64]
MVR = {"sst2": 4, "ag_news": 32}  # minimum viable rank (vs full-FT)


def sweep(ds, r):
    return stat(accs(lambda m: m["dataset"] == ds and m["method"] == "lora" and m.get("rank") == r
                     and m.get("type") == "sweep" and m.get("placement") == "attn" and not m.get("use_rslora")))


def ft(ds):
    return stat(accs(lambda m: m["dataset"] == ds and m["method"] == "full_ft" and m.get("type") == "baseline"))


def probe(ds):
    return stat(accs(lambda m: m["dataset"] == ds and m["method"] == "probe"))


# ============================ FIGURE 1 — rank-floor curve (hero) ============================
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, ds in zip(axes, ["sst2", "ag_news"]):
        ms = [sweep(ds, r) for r in RANKS]
        y = [a[0] for a in ms]
        ci = [a[2] for a in ms]
        ftm, _, ftci = ft(ds)
        pm, _, _ = probe(ds)
        # full-FT band
        ax.axhspan(ftm - ftci, ftm + ftci, color=C["ft"], alpha=0.13, zorder=0)
        ax.axhline(ftm, color=C["ft"], ls="--", lw=1.6, label=f"Full fine-tuning ({ftm:.3f})")
        ax.axhline(pm, color=C["probe"], ls=":", lw=1.6, label=f"Rank-0 probe ({pm:.3f})")
        ax.errorbar(RANKS, y, yerr=ci, marker="o", ms=6, lw=2, color=C[ds], capsize=3,
                    label="LoRA (attn-only)", zorder=5)
        # mark minimum viable rank
        rstar = MVR[ds]
        ys = y[RANKS.index(rstar)]
        ax.scatter([rstar], [ys], s=240, marker="*", color="#fdae61", edgecolor="k", zorder=6, lw=0.8)
        ax.annotate(f"min. viable rank r*={rstar}", (rstar, ys), textcoords="offset points",
                    xytext=(6, -22), fontsize=9.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", lw=1, color="k"))
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(r) for r in RANKS])
        ax.set_xlabel("LoRA rank  r")
        ax.set_ylabel("Test accuracy")
        ax.set_title(DS_NAME[ds])
        ax.legend(loc="lower right", frameon=True, framealpha=0.95)
        lo = min(pm, min(y)) - 0.01
        hi = max(ftm, max(y)) + 0.012
        ax.set_ylim(lo, hi)
    fig.suptitle("The LoRA Rank Floor: accuracy vs. rank, bracketed by rank-0 probe and full fine-tuning",
                 fontsize=12.5, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_rank_floor_curve.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================ FIGURE 3 — accuracy vs adapter params (Pareto) ============================
def fig3():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for ds in ["sst2", "ag_news"]:
        xs, ys, es, labs = [], [], [], []
        for r in RANKS:
            m = sweep(ds, r)
            padapt = 18432 * r
            pct = 100.0 * padapt / 66955010
            xs.append(pct); ys.append(m[0]); es.append(m[2]); labs.append(r)
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=5.5, lw=1.8, color=C[ds], capsize=2.5,
                    label=DS_NAME[ds])
        for x, y, r in zip(xs, ys, labs):
            ax.annotate(f"r={r}", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=7.5,
                        color=C[ds])
        ftm = ft(ds)[0]
        ax.axhline(ftm, color=C[ds], ls="--", lw=1.0, alpha=0.6)
    ax.text(0.018, ft("ag_news")[0] + 0.001, "full-FT (AG News)", fontsize=7.5, color=C["ag_news"])
    ax.text(0.018, ft("sst2")[0] + 0.001, "full-FT (SST-2)", fontsize=7.5, color=C["sst2"])
    ax.set_xscale("log")
    ax.set_xlabel("Adapter parameters (% of model, log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. trainable adapter parameters\n(full fine-tuning trains 100%; tiny adapters reach the bracket)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_accuracy_vs_params.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================ FIGURE 4 — module placement ablation ============================
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rs = [1, 4, 16]
    for ax, ds in zip(axes, ["sst2", "ag_news"]):
        attn = [sweep(ds, r) for r in rs]
        amlp = [stat(accs(lambda m, r=r: m["dataset"] == ds and m.get("rank") == r
                          and m.get("type") == "ablation_placement" and m.get("placement") == "attn_mlp")) for r in rs]
        x = np.arange(len(rs)); w = 0.36
        ax.bar(x - w / 2, [a[0] for a in attn], w, yerr=[a[2] for a in attn], capsize=3,
               color=C["attn"], label="attention-only (q,v)")
        ax.bar(x + w / 2, [a[0] for a in amlp], w, yerr=[a[2] for a in amlp], capsize=3,
               color=C["attn_mlp"], label="attention + MLP (q,v,lin1,lin2)")
        ftm = ft(ds)[0]
        ax.axhline(ftm, color=C["ft"], ls="--", lw=1.4, label=f"full fine-tuning ({ftm:.3f})")
        ax.set_xticks(x); ax.set_xticklabels([f"r={r}" for r in rs])
        ax.set_ylabel("Test accuracy"); ax.set_title(DS_NAME[ds])
        lo = min(min(a[0] for a in attn), min(a[0] for a in amlp)) - 0.012
        ax.set_ylim(lo, ftm + 0.012)
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Module-placement ablation: spending the same rank on more modules lowers the rank floor",
                 fontsize=12, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_placement_ablation.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================ FIGURE 5 — alpha-scaling ablation (SST-2) ============================
def fig5():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rs = [1, 4, 16]
    const = [sweep("sst2", r) for r in rs]
    fixed = [stat(accs(lambda m, r=r: m["dataset"] == "sst2" and m.get("rank") == r
                       and m.get("type") == "ablation_scaling" and m.get("lora_alpha") == 8 and not m.get("use_rslora"))) for r in rs]
    rsl = [stat(accs(lambda m, r=r: m["dataset"] == "sst2" and m.get("rank") == r
                     and m.get("type") == "ablation_scaling" and m.get("use_rslora"))) for r in rs]
    x = np.arange(len(rs)); w = 0.26
    ax.bar(x - w, [a[0] for a in const], w, yerr=[a[2] for a in const], capsize=3, color=C["s2"], label="constant  s=α/r=2  (α=2r)")
    ax.bar(x, [a[0] for a in fixed], w, yerr=[a[2] for a in fixed], capsize=3, color=C["a8"], label="fixed α=8  (s=8/r)")
    ax.bar(x + w, [a[0] for a in rsl], w, yerr=[a[2] for a in rsl], capsize=3, color=C["rs"], label="rsLoRA α=8  (s=8/√r)")
    ftm = ft("sst2")[0]
    ax.axhline(ftm, color=C["ft"], ls="--", lw=1.4, label=f"full fine-tuning ({ftm:.3f})")
    ax.set_xticks(x); ax.set_xticklabels([f"r={r}" for r in rs])
    ax.set_ylabel("Test accuracy (SST-2)")
    ax.set_title("α-scaling ablation: the floor is scaling-robust;\nat r=1 a larger scaling recovers ~0.5 pt")
    ax.set_ylim(0.875, ftm + 0.012)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_scaling_ablation.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================ FIGURE 6 — data-efficiency x rank ============================
def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sizes = [500, 2000, 10000]
    series = [("ft", "full fine-tuning", "#111111", "s", lambda m: m["method"] == "full_ft"),
              ("r1", "LoRA r=1", C["r1"], "o", lambda m: m["method"] == "lora" and m.get("rank") == 1),
              ("r4", "LoRA r=4", C["r4"], "^", lambda m: m["method"] == "lora" and m.get("rank") == 4),
              ("r16", "LoRA r=16", C["r16"], "D", lambda m: m["method"] == "lora" and m.get("rank") == 16)]
    for ax, ds in zip(axes, ["sst2", "ag_news"]):
        for key, lab, col, mk, pred in series:
            ys, es = [], []
            for n in sizes:
                s = stat(accs(lambda m, n=n, pred=pred: m.get("type") == "data_efficiency"
                              and m["dataset"] == ds and m.get("train_size") == n and pred(m)))
                ys.append(s[0]); es.append(s[2])
            ls = "--" if key == "ft" else "-"
            ax.errorbar(sizes, ys, yerr=es, marker=mk, ms=6, lw=1.9, ls=ls, color=col, capsize=3, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(sizes); ax.set_xticklabels([str(n) for n in sizes])
        ax.set_xlabel("Training-set size (log scale)")
        ax.set_ylabel("Test accuracy")
        ax.set_title(DS_NAME[ds])
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Data-efficiency × rank: under data scarcity, low-rank LoRA matches or beats full fine-tuning",
                 fontsize=12, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_data_efficiency.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================ FIGURE 2 — method overview schematic ============================
def fig2():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    def box(x, y, w, h, label, fc, ec="#333333", fs=10, lw=1.4, style="round,pad=0.02", fontweight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, fontweight=fontweight)

    def arrow(x1, y1, x2, y2, color="#333333", lw=1.6, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=lw, ls=ls))

    # input
    box(2, 24, 13, 8, "Input text\n\"a charming film\"", "#f0f0f0", fs=9)
    arrow(15, 28, 20, 28)
    # frozen encoder
    box(20, 14, 24, 28, "", "#eaf2fb", ec="#2166ac", lw=1.6)
    ax.text(32, 39.5, "DistilBERT encoder  (frozen, 66.4M)", ha="center", fontsize=9.5, color="#2166ac", fontweight="bold")
    for i in range(6):
        yy = 16 + i * 3.5
        box(23, yy, 18, 2.6, f"Transformer layer {i+1}", "#d6e6f7", ec="#9ecae1", fs=7.5, lw=0.8)
    arrow(44, 28, 50, 28)

    # zoom: one attention projection with LoRA
    box(50, 8, 32, 40, "", "#ffffff", ec="#999999", lw=1.2, style="round,pad=0.02")
    ax.text(66, 45.5, "Attention projection   W  (768 × 768)", ha="center", fontsize=9.5, fontweight="bold")
    # frozen W0
    box(53, 30, 12, 10, "W₀\n(frozen)", "#e0e0e0", ec="#7f7f7f", fs=9)
    # plus
    ax.text(67.5, 35, "+", ha="center", va="center", fontsize=18, fontweight="bold")
    # LoRA branch B A
    box(70, 36, 9, 5, "B  (768×r)", "#fddbc7", ec="#d6604d", fs=8)
    box(70, 29, 9, 5, "A  (r×768)", "#fddbc7", ec="#d6604d", fs=8)
    arrow(74.5, 36, 74.5, 34, color="#d6604d", lw=1.4)
    ax.text(85, 35, "trainable\nlow-rank\nΔW = (α/r)·B·A", ha="center", va="center", fontsize=8.2, color="#d6604d", fontweight="bold")
    arrow(65, 35, 70, 38.5, color="#7f7f7f", lw=1.0, ls=":")
    arrow(79, 31.5, 82, 31.5, color="#d6604d", lw=1.0)
    ax.text(66, 12, "only A, B trained  →  18,432·r params (r=1: 0.027%)", ha="center", fontsize=8, color="#d6604d")

    arrow(82, 28, 87, 28)
    # head
    box(87, 22, 11, 12, "Classifier\nhead\n(trainable,\nfixed 0.88%)", "#e6f5d0", ec="#1a9850", fs=8)
    arrow(92.5, 22, 92.5, 18)
    ax.text(92.5, 16, "class", ha="center", fontsize=9, fontweight="bold")

    # brackets
    ax.text(50, 3.2, "rank-0 probe = head only (encoder frozen, no adapter)        full fine-tuning = all 66.96M trainable",
            ha="center", fontsize=8.5, style="italic", color="#555555")
    ax.add_patch(Rectangle((1, 1.6), 98, 3.2, fc="#fafafa", ec="#cccccc", lw=0.8))
    ax.text(50, 53.5, "LoRA adaptation of a small encoder: frozen backbone + low-rank update on attention, with a fixed trainable head",
            ha="center", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig2_method_overview.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 ok")
    fig2(); print("fig2 ok")
    fig3(); print("fig3 ok")
    fig4(); print("fig4 ok")
    fig5(); print("fig5 ok")
    fig6(); print("fig6 ok")
    print("ALL FIGURES ->", OUT)
