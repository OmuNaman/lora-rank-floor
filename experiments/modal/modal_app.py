"""
Modal experiment runner for: How Low Can the LoRA Rank Go? (lora-rank-floor)

Trains DistilBERT-base-uncased on SST-2 / AG News under different adaptation
methods (full fine-tuning, frozen-encoder head-only probe, LoRA at various ranks
/ alpha-scalings / module placements), driven entirely by a `cell` config dict.

Design:
  - run_cells(cells): ONE container runs a LIST of cells. The dataset for each
    cell is loaded + tokenized once and cached in-process, so a batch of cells on
    the same dataset amortizes download/tokenization. Each finished cell is written
    to /vol/cells/<exp_id>.json and committed immediately (crash-safe), and the
    full list of metrics is returned.
  - The local entrypoint accepts either a single cell dict (smoke / one-off) or
    {"batches": [[cell,...], [cell,...]]} and uses run_cells.map(...) to run the
    batches on parallel containers, then prints  RESULT_JSON:[...]  (flattened).

HF model + dataset downloads are cached on the Volume (HF_HOME=/vol/hf) so they
persist across runs.

Run locally (controller shells out):
  py -3.13 -m modal run modal_app.py --config-json '{"exp_id":"exp00","smoke":true,"dataset":"sst2","method":"lora","rank":8,"lora_alpha":16,"seed":0}'
  py -3.13 -m modal run --detach modal_app.py --config-json '{"batches": [[...],[...]]}'
"""
import json
import os
import time

import modal

SLUG = "lora-rank-floor"
APP_NAME = f"research-{SLUG}"
VOL_NAME = f"research-{SLUG}"
DATA_ROOT = "/vol"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "datasets==2.21.0",
        "peft==0.13.2",
        "scikit-learn==1.5.2",
        "numpy<2",
    )
    # keep HF caches on the Volume mount
    .env({"HF_HOME": "/vol/hf", "HF_DATASETS_CACHE": "/vol/hf/datasets", "TOKENIZERS_PARALLELISM": "false"})
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ---- dataset registry -------------------------------------------------------
DATASET_INFO = {
    "sst2":    {"hf": ("glue", "sst2"), "text": "sentence", "eval_split": "validation", "num_labels": 2},
    "ag_news": {"hf": ("ag_news",),     "text": "text",     "eval_split": "test",       "num_labels": 4},
}

# DistilBERT module names
ATTN_MODULES = ["q_lin", "v_lin"]
ATTN_MLP_MODULES = ["q_lin", "v_lin", "lin1", "lin2"]


def _set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_and_tokenize(dataset_name: str, tokenizer, max_len: int, smoke: bool, train_size=None):
    """Load + tokenize a dataset once; returns (train_ds, eval_ds, num_labels)."""
    from datasets import load_dataset

    info = DATASET_INFO[dataset_name]
    raw = load_dataset(*info["hf"])
    text_col = info["text"]
    eval_split = info["eval_split"]

    def tok(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_len)

    train = raw["train"]
    ev = raw[eval_split]
    if smoke:
        train = train.select(range(min(2000, len(train))))
        ev = ev.select(range(min(800, len(ev))))
    elif train_size:
        # fixed-seed subsample so the training SET is identical across model seeds
        train = train.shuffle(seed=42).select(range(min(int(train_size), len(train))))

    keep = ["input_ids", "attention_mask", "label"]
    train = train.map(tok, batched=True, remove_columns=[c for c in train.column_names if c not in ("label",)])
    ev = ev.map(tok, batched=True, remove_columns=[c for c in ev.column_names if c not in ("label",)])
    train = train.rename_column("label", "labels") if "label" in train.column_names else train
    ev = ev.rename_column("label", "labels") if "label" in ev.column_names else ev
    train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    ev.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train, ev, info["num_labels"]


def _build_model(method, num_labels, rank, lora_alpha, use_rslora, target_modules, seed):
    import torch
    from transformers import AutoModelForSequenceClassification

    base = "distilbert-base-uncased"
    _set_seed(seed)  # seed before head init for reproducible head
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=num_labels)

    if method == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        return model, "full_ft"

    if method == "probe":
        # freeze the encoder; train only the classification head (rank-0 analogue)
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("pre_classifier") or n.startswith("classifier")
        return model, "probe"

    if method == "lora":
        from peft import LoraConfig, get_peft_model, TaskType
        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            use_rslora=bool(use_rslora),
            bias="none",
            modules_to_save=["pre_classifier", "classifier"],
        )
        model = get_peft_model(model, cfg)
        return model, "lora"

    raise ValueError(f"unknown method {method}")


def _count_params(model, method):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    head = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and ("classifier" in n) and "lora_" not in n
    )
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "params_adapter": int(adapter),
        "params_head": int(head),
        "pct_trainable": round(100.0 * trainable / total, 4),
        "pct_adapter": round(100.0 * adapter / total, 6),
    }


def train_one_cell(cell, train_ds, eval_ds, num_labels):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding, AutoTokenizer, get_linear_schedule_with_warmup

    exp_id = cell["exp_id"]
    method = cell["method"]
    seed = int(cell.get("seed", 0))
    epochs = int(cell.get("epochs", 3))
    lr = float(cell.get("lr", 5e-4))
    batch_size = int(cell.get("batch_size", 32))
    rank = int(cell.get("rank", 0))
    lora_alpha = int(cell.get("lora_alpha", 16))
    use_rslora = bool(cell.get("use_rslora", False))
    placement = cell.get("placement", "attn")  # "attn" or "attn_mlp"
    target_modules = ATTN_MLP_MODULES if placement == "attn_mlp" else ATTN_MODULES
    weight_decay = float(cell.get("weight_decay", 0.01))
    warmup_ratio = float(cell.get("warmup_ratio", 0.1))

    _set_seed(seed)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collator = DataCollatorWithPadding(tokenizer)

    model, _ = _build_model(method, num_labels, rank, lora_alpha, use_rslora, target_modules, seed)
    model.to(device)
    pcount = _count_params(model, method)

    # effective LoRA scaling factor (for logging)
    if method == "lora":
        import math
        scaling = lora_alpha / (math.sqrt(rank) if use_rslora else rank)
    else:
        scaling = None

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator, generator=g, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False, collate_fn=collator, num_workers=2)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    total_steps = max(1, len(train_loader) * epochs)
    sched = get_linear_schedule_with_warmup(optim, int(warmup_ratio * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler()

    run_dir = os.path.join(DATA_ROOT, "cells")
    os.makedirs(run_dir, exist_ok=True)
    prog_path = os.path.join(run_dir, exp_id + ".progress.json")

    t0 = time.time()
    step = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            step += 1
        # per-epoch progress
        _write_json(prog_path, {
            "exp_id": exp_id, "epoch": epoch + 1, "total_epochs": epochs,
            "step": step, "loss": float(loss.detach().cpu()), "done": False,
        })
        vol.commit()

    # ---- eval ----
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for batch in eval_loader:
            labels = batch["labels"].to(device)
            inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(**inp).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            n += labels.numel()
    acc = correct / max(1, n)
    train_time = time.time() - t0

    metrics = {
        "exp_id": exp_id,
        "type": cell.get("type", ""),
        "dataset": cell["dataset"],
        "method": method,
        "rank": rank if method == "lora" else (0 if method == "probe" else None),
        "lora_alpha": lora_alpha if method == "lora" else None,
        "use_rslora": use_rslora if method == "lora" else None,
        "scaling": round(scaling, 4) if scaling is not None else None,
        "placement": placement if method == "lora" else None,
        "target_modules": target_modules if method == "lora" else None,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "train_size": cell.get("train_size"),
        "accuracy": round(acc, 4),
        "eval_n": n,
        "train_time_sec": round(train_time, 1),
        "smoke": bool(cell.get("smoke", False)),
        **pcount,
    }
    return metrics


@app.function(image=image, volumes={DATA_ROOT: vol}, gpu="A100", timeout=6 * 60 * 60)
def run_cells(cells: list) -> list:
    """Run a list of cells in one container, caching each dataset's tokenized form."""
    from transformers import AutoTokenizer

    os.makedirs(os.path.join(DATA_ROOT, "cells"), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    cache = {}  # dataset_name(+smoke) -> (train_ds, eval_ds, num_labels)
    max_len = 128

    results = []
    for i, cell in enumerate(cells):
        exp_id = cell["exp_id"]
        ds_name = cell["dataset"]
        smoke = bool(cell.get("smoke", False))
        train_size = cell.get("train_size")  # None = full
        key = f"{ds_name}-{'smoke' if smoke else ('n' + str(train_size) if train_size else 'full')}"
        try:
            if key not in cache:
                cache[key] = _load_and_tokenize(ds_name, tokenizer, max_len, smoke, train_size)
                vol.commit()  # persist HF download cache
            train_ds, eval_ds, num_labels = cache[key]
            m = train_one_cell(cell, train_ds, eval_ds, num_labels)
            m["status"] = "ok"
        except Exception as e:
            import traceback
            m = {"exp_id": exp_id, "dataset": ds_name, "status": "error",
                 "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        # crash-safe per-cell write
        _write_json(os.path.join(DATA_ROOT, "cells", exp_id + ".json"), m)
        vol.commit()
        print(f"[{i+1}/{len(cells)}] {exp_id} -> "
              f"{m.get('accuracy', m.get('status'))} ({m.get('train_time_sec','-')}s)")
        results.append(m)
    return results


@app.local_entrypoint()
def main(config_json: str = "{}"):
    # Support "@path" to read config from a local file (avoids command-line length limits).
    if config_json.startswith("@"):
        with open(config_json[1:]) as f:
            config_json = f.read()
    cfg = json.loads(config_json)
    if isinstance(cfg, dict) and "batches" in cfg:
        batches = cfg["batches"]
        all_results = []
        for res in run_cells.map(batches):
            all_results.extend(res)
        print("RESULT_JSON:" + json.dumps(all_results))
    else:
        if isinstance(cfg, dict) and "cells" in cfg:
            cells = cfg["cells"]
        else:
            cells = [cfg]  # single cell (smoke / one-off)
        res = run_cells.remote(cells)
        print("RESULT_JSON:" + json.dumps(res))
