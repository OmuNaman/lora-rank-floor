"""Aggregate results.jsonl into summary tables, the minimum-viable-rank significance test,
and CSVs for figures. Run locally after pulling metrics from Modal.

Usage:  python analyze.py            (reads experiments/results.jsonl, writes experiments/analysis/)
"""
import json
import os
import math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results.jsonl")
OUT = os.path.join(HERE, "analysis")
os.makedirs(OUT, exist_ok=True)


def load():
    by_id = {}  # dedup by exp_id, keep last (append-only ledger may have reruns)
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m = rec.get("metrics", rec)
            if m.get("status") == "error" or "accuracy" not in m:
                continue
            by_id[m["exp_id"]] = m
    return list(by_id.values())


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs):
    # t-multiplier for small n (df=n-1): 2,3,4 seeds -> use t .975
    tmult = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(xs) - 1, 1.96)
    return tmult * std(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def welch_t(a, b):
    """Welch's t-test; returns (t, df, p_two_sided approx)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None, None
    ma, mb = mean(a), mean(b)
    va, vb = std(a) ** 2, std(b) ** 2
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return (0.0, na + nb - 2, 1.0)
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # two-sided p via survival function of t-dist (use scipy if available, else normal approx)
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except Exception:
        # normal approximation
        z = abs(t)
        p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return float(t), float(df), float(p)


def group_key(m):
    """Aggregation key: a unique configuration across seeds."""
    return (m["dataset"], m["method"], m.get("rank"), m.get("lora_alpha"),
            bool(m.get("use_rslora")), m.get("placement"), m.get("type"))


def main():
    rows = load()
    print(f"loaded {len(rows)} ok rows")

    groups = defaultdict(list)
    for m in rows:
        groups[group_key(m)].append(m)

    agg = []
    for key, ms in groups.items():
        ds, method, rank, alpha, rslora, placement, typ = key
        accs = [m["accuracy"] for m in ms]
        times = [m.get("train_time_sec", 0) for m in ms]
        rep = ms[0]
        agg.append({
            "dataset": ds, "method": method, "type": typ, "rank": rank,
            "lora_alpha": alpha, "use_rslora": rslora, "placement": placement,
            "scaling": rep.get("scaling"),
            "n_seeds": len(accs),
            "acc_mean": round(mean(accs), 4), "acc_std": round(std(accs), 4),
            "acc_ci95": round(ci95(accs), 4),
            "acc_list": [round(a, 4) for a in accs],
            "params_total": rep.get("params_total"),
            "params_trainable": rep.get("params_trainable"),
            "params_adapter": rep.get("params_adapter"),
            "params_head": rep.get("params_head"),
            "pct_trainable": rep.get("pct_trainable"),
            "pct_adapter": rep.get("pct_adapter"),
            "time_mean_sec": round(mean(times), 1),
        })

    with open(os.path.join(OUT, "aggregated.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # ---- minimum viable rank, per dataset ----
    mvr_report = {}
    for ds in sorted(set(a["dataset"] for a in agg)):
        # reference = main-sweep LoRA r=64 (attn, standard scaling); also full_ft
        def get_accs(pred):
            for k, ms in groups.items():
                if pred(ms[0]):
                    return [m["accuracy"] for m in ms]
            return None

        ref_r64 = get_accs(lambda m: m["dataset"] == ds and m["method"] == "lora" and m.get("rank") == 64
                           and m.get("placement") == "attn" and not m.get("use_rslora") and m.get("type") == "sweep")
        ref_ft = get_accs(lambda m: m["dataset"] == ds and m["method"] == "full_ft")
        probe = get_accs(lambda m: m["dataset"] == ds and m["method"] == "probe")

        rank_rows = []
        for r in [1, 2, 4, 8, 16, 32, 64]:
            accs = get_accs(lambda m, r=r: m["dataset"] == ds and m["method"] == "lora" and m.get("rank") == r
                            and m.get("placement") == "attn" and not m.get("use_rslora") and m.get("type") == "sweep")
            if accs is None:
                continue
            t64, df64, p64 = welch_t(accs, ref_r64) if ref_r64 else (None, None, None)
            tft, dfft, pft = welch_t(accs, ref_ft) if ref_ft else (None, None, None)
            rank_rows.append({
                "rank": r, "acc_mean": round(mean(accs), 4), "acc_std": round(std(accs), 4),
                "p_vs_r64": round(p64, 4) if p64 is not None else None,
                "p_vs_ft": round(pft, 4) if pft is not None else None,
                "indist_r64": bool(p64 is not None and p64 > 0.05),
                "indist_ft": bool(pft is not None and pft > 0.05),
            })
        # minimum viable rank = smallest r with p_vs_r64 > 0.05 (not significantly worse than r=64)
        mvr = next((rr["rank"] for rr in rank_rows if rr["indist_r64"]), None)
        mvr_ft = next((rr["rank"] for rr in rank_rows if rr["indist_ft"]), None)
        mvr_report[ds] = {
            "ref_r64_mean": round(mean(ref_r64), 4) if ref_r64 else None,
            "ref_ft_mean": round(mean(ref_ft), 4) if ref_ft else None,
            "probe_mean": round(mean(probe), 4) if probe else None,
            "min_viable_rank_vs_r64": mvr,
            "min_viable_rank_vs_ft": mvr_ft,
            "rank_rows": rank_rows,
        }

    with open(os.path.join(OUT, "min_viable_rank.json"), "w") as f:
        json.dump(mvr_report, f, indent=2)

    # ---- CSVs for figures ----
    def write_csv(name, header, rows_):
        with open(os.path.join(OUT, name), "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows_:
                f.write(",".join(str(x) for x in r) + "\n")

    for ds in sorted(set(a["dataset"] for a in agg)):
        rr = mvr_report[ds]["rank_rows"]
        write_csv(f"rank_vs_acc_{ds}.csv", ["rank", "acc_mean", "acc_std"],
                  [[x["rank"], x["acc_mean"], x["acc_std"]] for x in rr])

    print("=== Minimum Viable Rank ===")
    for ds, rep in mvr_report.items():
        print(f"{ds}: probe={rep['probe_mean']} ft={rep['ref_ft_mean']} r64={rep['ref_r64_mean']} "
              f"-> min viable rank vs r64 = {rep['min_viable_rank_vs_r64']}, vs ft = {rep['min_viable_rank_vs_ft']}")
    print(f"\nwrote: {OUT}/aggregated.json, min_viable_rank.json, rank_vs_acc_*.csv")


if __name__ == "__main__":
    main()
