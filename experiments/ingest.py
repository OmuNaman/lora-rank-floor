"""Append a results-list JSON (the RESULT_JSON payload, or a file of it) to results.jsonl.

Usage:
  python ingest.py <path-to-json-with-list>     # append each metric dict as a ledger line
The ledger is append-only; analyze.py dedups by exp_id (keep-last).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results.jsonl")


def main(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    n_ok = n_err = 0
    with open(LEDGER, "a") as f:
        for m in data:
            if "status" not in m:
                m["status"] = "ok" if "accuracy" in m else "error"
            f.write(json.dumps(m) + "\n")
            if m["status"] == "ok":
                n_ok += 1
            else:
                n_err += 1
    print(f"appended {len(data)} rows ({n_ok} ok, {n_err} error) -> {LEDGER}")


if __name__ == "__main__":
    main(sys.argv[1])
