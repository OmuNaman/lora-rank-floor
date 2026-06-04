"""Generate the experiment-matrix config files (batches of cells) for modal_app.py.

Writes JSON files into this folder:
  configs_smoke.json     - 1 smoke cell
  configs_core.json      - {"batches": [...]} core grid (SST-2 + AG News): 54 cells
  configs_ablation.json  - {"batches": [...]} alpha-scaling + placement ablations: 36 cells

Each cell is a dict the Modal training fn consumes. A "batch" is a list of cells run in one
container (groups by dataset to amortize tokenization). batches run on parallel containers.
"""
import json
import os

SEEDS = [0, 1, 2]
HERE = os.path.dirname(os.path.abspath(__file__))

# default epochs / lr per (dataset, method)
EPOCHS = {("sst2", "lora"): 3, ("sst2", "full_ft"): 3, ("sst2", "probe"): 5,
          ("ag_news", "lora"): 2, ("ag_news", "full_ft"): 2, ("ag_news", "probe"): 4}
LR = {"lora": 5e-4, "full_ft": 2e-5, "probe": 1e-3}


def cell(exp_id, dataset, method, typ, rank=0, lora_alpha=16, use_rslora=False,
         placement="attn", seed=0, smoke=False):
    return {
        "exp_id": exp_id, "dataset": dataset, "method": method, "type": typ,
        "rank": rank, "lora_alpha": lora_alpha, "use_rslora": use_rslora,
        "placement": placement, "seed": seed,
        "epochs": 1 if smoke else EPOCHS[(dataset, method)],
        "lr": LR[method], "batch_size": 32, "smoke": smoke,
    }


def seeded(make):
    return [make(s) for s in SEEDS]


# ---------------- SMOKE ----------------
smoke = cell("smoke-sst2-r8-s0", "sst2", "lora", "smoke", rank=8, lora_alpha=16, seed=0, smoke=True)
with open(os.path.join(HERE, "configs_smoke.json"), "w") as f:
    json.dump(smoke, f, indent=2)

# ---------------- CORE GRID ----------------
RANKS = [1, 2, 4, 8, 16, 32, 64]


def core_for(ds):
    """Return list of cells for one dataset's core grid: FT, probe, LoRA r in RANKS."""
    cells = []
    cells += seeded(lambda s: cell(f"ft-{ds}-s{s}", ds, "full_ft", "baseline", seed=s))
    cells += seeded(lambda s: cell(f"lp-{ds}-s{s}", ds, "probe", "baseline", rank=0, seed=s))
    for r in RANKS:
        # main sweep: attention-only, alpha = 2r  => constant effective scaling s = 2
        cells += seeded(lambda s, r=r: cell(f"lora-{ds}-r{r}-s{s}", ds, "lora",
                                             "sweep", rank=r, lora_alpha=2 * r, seed=s))
    return cells


# group into batches (lists run in one container) — split for parallelism + de-risk
def chunk_by_groups(cells, group_exp_prefixes):
    """Split a dataset's cells into batches by lists of exp_id prefixes."""
    out = []
    for prefixes in group_exp_prefixes:
        batch = [c for c in cells if any(c["exp_id"].startswith(p) for p in prefixes)]
        if batch:
            out.append(batch)
    return out


sst2_core = core_for("sst2")
agnews_core = core_for("ag_news")

core_batches = []
# SST-2 (fast): 3 batches
core_batches += chunk_by_groups(sst2_core, [
    ["ft-sst2", "lp-sst2", "lora-sst2-r1-", "lora-sst2-r2-"],
    ["lora-sst2-r4-", "lora-sst2-r8-", "lora-sst2-r16-"],
    ["lora-sst2-r32-", "lora-sst2-r64-"],
])
# AG News (slow): 4 batches
core_batches += chunk_by_groups(agnews_core, [
    ["ft-ag_news", "lp-ag_news"],
    ["lora-ag_news-r1-", "lora-ag_news-r2-", "lora-ag_news-r4-"],
    ["lora-ag_news-r8-", "lora-ag_news-r16-"],
    ["lora-ag_news-r32-", "lora-ag_news-r64-"],
])

with open(os.path.join(HERE, "configs_core.json"), "w") as f:
    json.dump({"batches": core_batches}, f, indent=2)

# ---------------- ABLATIONS ----------------
ABL_RANKS = [1, 4, 16]
abl_batches = []

# Ablation A: alpha-scaling on SST-2 at r in {1,4,16}
#   fixed8 : alpha=8, standard scaling => s = 8/r
#   rslora8: alpha=8, rsLoRA          => s = 8/sqrt(r)
#   (main-sweep alpha=2r, s=2, is the 3rd comparison point, already in core grid)
abl_a = []
for r in ABL_RANKS:
    abl_a += seeded(lambda s, r=r: cell(f"absc-sst2-fixed8-r{r}-s{s}", "sst2", "lora",
                                        "ablation_scaling", rank=r, lora_alpha=8, use_rslora=False, seed=s))
for r in ABL_RANKS:
    abl_a += seeded(lambda s, r=r: cell(f"absc-sst2-rslora8-r{r}-s{s}", "sst2", "lora",
                                        "ablation_scaling", rank=r, lora_alpha=8, use_rslora=True, seed=s))
abl_batches.append(abl_a)  # 18 cells, one SST-2 container

# Ablation B: module placement attn+MLP at r in {1,4,16}, both datasets
abl_b_sst2 = []
for r in ABL_RANKS:
    abl_b_sst2 += seeded(lambda s, r=r: cell(f"place-sst2-attnmlp-r{r}-s{s}", "sst2", "lora",
                                             "ablation_placement", rank=r, lora_alpha=2 * r,
                                             placement="attn_mlp", seed=s))
abl_batches.append(abl_b_sst2)  # 9 cells

abl_b_ag = []
for r in ABL_RANKS:
    abl_b_ag += seeded(lambda s, r=r: cell(f"place-ag_news-attnmlp-r{r}-s{s}", "ag_news", "lora",
                                           "ablation_placement", rank=r, lora_alpha=2 * r,
                                           placement="attn_mlp", seed=s))
abl_batches.append(abl_b_ag)  # 9 cells

with open(os.path.join(HERE, "configs_ablation.json"), "w") as f:
    json.dump({"batches": abl_batches}, f, indent=2)

# ---------------- summary ----------------
n_core = sum(len(b) for b in core_batches)
n_abl = sum(len(b) for b in abl_batches)
print(f"smoke: 1 cell")
print(f"core: {len(core_batches)} batches, {n_core} cells")
for b in core_batches:
    print(f"   batch[{len(b):2d}]: {b[0]['exp_id']} ... {b[-1]['exp_id']}")
print(f"ablation: {len(abl_batches)} batches, {n_abl} cells")
for b in abl_batches:
    print(f"   batch[{len(b):2d}]: {b[0]['exp_id']} ... {b[-1]['exp_id']}")
print(f"TOTAL cells (excl smoke): {n_core + n_abl}")
