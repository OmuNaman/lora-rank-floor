"""Round 4 — data-efficiency x rank: does the LoRA rank floor RISE when training data is scarce?

Subsample train to N in {500, 2000, 10000} for SST-2 and AG News, and compare full fine-tuning
against LoRA r in {1,4,16} (attn-only, alpha=2r), 3 seeds. Eval split is always the FULL dev/test set.
Batches are grouped by (dataset, train_size) so each subsample is tokenized once.

Writes configs_dataeff.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
SIZES = [500, 2000, 10000]
RANKS = [1, 4, 16]
# more epochs for less data so both full-FT and LoRA converge under a comparable budget
EPOCHS_BY_SIZE = {500: 20, 2000: 10, 10000: 5}
LR = {"lora": 5e-4, "full_ft": 2e-5}


def cell(exp_id, ds, method, n, rank=0, lora_alpha=16, seed=0):
    return {
        "exp_id": exp_id, "dataset": ds, "method": method, "type": "data_efficiency",
        "rank": rank, "lora_alpha": lora_alpha, "use_rslora": False, "placement": "attn",
        "seed": seed, "train_size": n, "epochs": EPOCHS_BY_SIZE[n],
        "lr": LR[method], "batch_size": 32, "smoke": False,
    }


batches = []
for ds in ["sst2", "ag_news"]:
    for n in SIZES:
        batch = []
        for s in SEEDS:
            batch.append(cell(f"de-{ds}-n{n}-ft-s{s}", ds, "full_ft", n, seed=s))
        for r in RANKS:
            for s in SEEDS:
                batch.append(cell(f"de-{ds}-n{n}-r{r}-s{s}", ds, "lora", n, rank=r, lora_alpha=2 * r, seed=s))
        batches.append(batch)  # 12 cells per (dataset, size)

with open(os.path.join(HERE, "configs_dataeff.json"), "w") as f:
    json.dump({"batches": batches}, f, indent=2)

n = sum(len(b) for b in batches)
print(f"data-efficiency: {len(batches)} batches, {n} cells")
for b in batches:
    print(f"   batch[{len(b):2d}]: {b[0]['exp_id']} ... {b[-1]['exp_id']}  (epochs={b[0]['epochs']})")
