"""Analyze the ablation (alpha-scaling, module placement) and data-efficiency rounds.
Reads results.jsonl, writes analysis CSV/JSON for figures + the paper.

Outputs (in experiments/analysis/):
  scaling_sst2.json/.csv   - SST-2 r in {1,4,16}: scaling rule (const s=2 / fixed a=8 / rsLoRA) vs acc
  placement.json/.csv      - SST-2 & AG News r in {1,4,16}: attn-only vs attn+MLP
  dataeff_<ds>.json/.csv   - acc by (train_size, method/rank)
"""
import json
import os
import math
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results.jsonl")
OUT = os.path.join(HERE, "analysis")
os.makedirs(OUT, exist_ok=True)


def load():
    by_id = {}
    for line in open(LEDGER):
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        if m.get("status") == "error" or "accuracy" not in m:
            continue
        by_id[m["exp_id"]] = m
    return list(by_id.values())


def mean(x):
    return sum(x) / len(x)


def sd(x):
    return st.stdev(x) if len(x) > 1 else 0.0


def ci95(x):
    tmult = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(x) - 1, 1.96)
    return tmult * sd(x) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def grouped(rows, keyfn):
    g = defaultdict(list)
    for m in rows:
        g[keyfn(m)].append(m["accuracy"])
    return g


def write_csv(name, header, rows_):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows_:
            f.write(",".join(str(x) for x in r) + "\n")


def main():
    rows = load()

    # ---------- 1. alpha-scaling (SST-2) ----------
    # const scaling s=2 (alpha=2r) comes from the core SWEEP cells; fixed8 / rslora8 from ablation cells.
    scaling = {}
    for r in [1, 4, 16]:
        const = [m["accuracy"] for m in rows if m["dataset"] == "sst2" and m["method"] == "lora"
                 and m.get("rank") == r and m.get("type") == "sweep" and not m.get("use_rslora")
                 and m.get("placement") == "attn"]
        fixed8 = [m["accuracy"] for m in rows if m["dataset"] == "sst2" and m.get("rank") == r
                  and m.get("type") == "ablation_scaling" and m.get("lora_alpha") == 8 and not m.get("use_rslora")]
        rslora = [m["accuracy"] for m in rows if m["dataset"] == "sst2" and m.get("rank") == r
                  and m.get("type") == "ablation_scaling" and m.get("use_rslora")]
        scaling[r] = {
            "const_s2": {"mean": round(mean(const), 4), "std": round(sd(const), 4)} if const else None,
            "fixed_a8": {"mean": round(mean(fixed8), 4), "std": round(sd(fixed8), 4)} if fixed8 else None,
            "rslora_a8": {"mean": round(mean(rslora), 4), "std": round(sd(rslora), 4)} if rslora else None,
        }
    json.dump(scaling, open(os.path.join(OUT, "scaling_sst2.json"), "w"), indent=2)
    write_csv("scaling_sst2.csv", ["rank", "const_s2", "fixed_a8", "rslora_a8"],
              [[r, (scaling[r]["const_s2"] or {}).get("mean"), (scaling[r]["fixed_a8"] or {}).get("mean"),
                (scaling[r]["rslora_a8"] or {}).get("mean")] for r in [1, 4, 16]])

    # ---------- 2. module placement (SST-2 & AG News) ----------
    placement = {}
    for ds in ["sst2", "ag_news"]:
        placement[ds] = {}
        for r in [1, 4, 16]:
            attn = [m["accuracy"] for m in rows if m["dataset"] == ds and m["method"] == "lora"
                    and m.get("rank") == r and m.get("type") == "sweep" and m.get("placement") == "attn"]
            attnmlp = [m["accuracy"] for m in rows if m["dataset"] == ds and m.get("rank") == r
                       and m.get("type") == "ablation_placement" and m.get("placement") == "attn_mlp"]
            placement[ds][r] = {
                "attn": {"mean": round(mean(attn), 4), "std": round(sd(attn), 4)} if attn else None,
                "attn_mlp": {"mean": round(mean(attnmlp), 4), "std": round(sd(attnmlp), 4)} if attnmlp else None,
            }
    json.dump(placement, open(os.path.join(OUT, "placement.json"), "w"), indent=2)
    for ds in ["sst2", "ag_news"]:
        write_csv(f"placement_{ds}.csv", ["rank", "attn", "attn_mlp"],
                  [[r, (placement[ds][r]["attn"] or {}).get("mean"),
                    (placement[ds][r]["attn_mlp"] or {}).get("mean")] for r in [1, 4, 16]])

    # ---------- 3. data-efficiency ----------
    de = {}
    for ds in ["sst2", "ag_news"]:
        de[ds] = {}
        for n in [500, 2000, 10000]:
            row = {}
            for col, pred in [("ft", lambda m: m["method"] == "full_ft"),
                              ("r1", lambda m: m["method"] == "lora" and m.get("rank") == 1),
                              ("r4", lambda m: m["method"] == "lora" and m.get("rank") == 4),
                              ("r16", lambda m: m["method"] == "lora" and m.get("rank") == 16)]:
                accs = [m["accuracy"] for m in rows if m.get("type") == "data_efficiency"
                        and m["dataset"] == ds and m.get("train_size") == n and pred(m)]
                row[col] = {"mean": round(mean(accs), 4), "std": round(sd(accs), 4)} if accs else None
            de[ds][n] = row
    json.dump(de, open(os.path.join(OUT, "dataeff.json"), "w"), indent=2)
    for ds in ["sst2", "ag_news"]:
        write_csv(f"dataeff_{ds}.csv", ["train_size", "ft", "r1", "r4", "r16"],
                  [[n, (de[ds][n]["ft"] or {}).get("mean"), (de[ds][n]["r1"] or {}).get("mean"),
                    (de[ds][n]["r4"] or {}).get("mean"), (de[ds][n]["r16"] or {}).get("mean")]
                   for n in [500, 2000, 10000]])

    # ---------- console summary ----------
    print("=== alpha-scaling (SST-2) [mean acc] ===")
    print(f"{'rank':>4} {'const s=2':>10} {'fixed a=8':>10} {'rsLoRA a=8':>11}")
    for r in [1, 4, 16]:
        s = scaling[r]
        print(f"{r:>4} {(s['const_s2'] or {}).get('mean','--'):>10} "
              f"{(s['fixed_a8'] or {}).get('mean','--'):>10} {(s['rslora_a8'] or {}).get('mean','--'):>11}")
    print("\n=== placement: attn-only vs attn+MLP [mean acc] ===")
    for ds in ["sst2", "ag_news"]:
        print(f"  {ds}:")
        for r in [1, 4, 16]:
            p = placement[ds][r]
            print(f"    r={r:<2} attn={ (p['attn'] or {}).get('mean','--')}  "
                  f"attn+MLP={(p['attn_mlp'] or {}).get('mean','--')}")
    print("\n=== data-efficiency [mean acc] ===")
    for ds in ["sst2", "ag_news"]:
        print(f"  {ds}: " + "  ".join(
            f"N{n}(ft={ (de[ds][n]['ft'] or {}).get('mean')},r1={(de[ds][n]['r1'] or {}).get('mean')},"
            f"r4={(de[ds][n]['r4'] or {}).get('mean')},r16={(de[ds][n]['r16'] or {}).get('mean')})"
            for n in [500, 2000, 10000]))
    print(f"\nwrote scaling_sst2, placement_*, dataeff_* to {OUT}")


if __name__ == "__main__":
    main()
