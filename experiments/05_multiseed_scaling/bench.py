"""Paired, multi-seed character GLU transformer benchmark. See README.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mitsu_torch import mitsu

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "shakespeare.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Block(nn.Module):
    def __init__(self, width, layers, family, gain):
        super().__init__()
        self.width, self.heads, self.family, self.gain = width, 4, family, gain
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 3 * width, bias=False)
        self.up = nn.Linear(width, 3 * width, bias=False)
        self.down = nn.Linear(3 * width, width, bias=False)
        self.layers = layers

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_out(attn.transpose(1, 2).reshape(b, t, d))
        h = self.ln2(x)
        g = self.gate(h)
        g = self.gain * mitsu(g) if self.family == "mitsuglu" else F.silu(g)
        return x + self.down(g * self.up(h))


class GPT(nn.Module):
    def __init__(self, vocab, width, layers, context, family, gain):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.position = nn.Parameter(torch.empty(context, width))
        self.blocks = nn.ModuleList([Block(width, layers, family, gain) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.apply(self._init)
        nn.init.normal_(self.position, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.attn_out.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        h = self.embedding(x) + self.position[:x.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


def load_data():
    if not DATA.is_file():
        raise FileNotFoundError("Run python get_data.py first")
    raw = DATA.read_bytes()
    if digest(DATA) != DATA_SHA256:
        raise ValueError("Dataset checksum differs from the frozen dataset")
    text = raw.decode("utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(tokens))
    return tokens[:split], tokens[split:], len(chars)


@torch.no_grad()
def evaluate(net, tokens, context, batch=32):
    net.eval()
    windows = (len(tokens) - 1) // context
    x = tokens[:windows * context].view(windows, context)
    y = tokens[1:windows * context + 1].view(windows, context)
    loss = 0.0
    for start in range(0, windows, batch):
        logits = net(x[start:start+batch])
        loss += F.cross_entropy(logits.flatten(0, 1), y[start:start+batch].flatten(), reduction="sum").item()
    return loss / (windows * context), windows * context


def run_one(cfg, width, seed, family, train, valid, vocab):
    torch.manual_seed(seed)
    net = GPT(vocab, width, cfg["layers"], cfg["context"], family, cfg["gain"])
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(9000 + seed)
    hist, gradnorms, spikes = [], [], []
    diverged = False
    t0 = time.perf_counter()
    for step in range(1, cfg["steps"] + 1):
        start = rng.integers(0, len(train) - cfg["context"] - 1, cfg["batch"])
        ids = torch.from_numpy(start[:, None]) + torch.arange(cfg["context"])[None, :]
        net.train()
        logits = net(train[ids])
        loss = F.cross_entropy(logits.flatten(0, 1), train[ids + 1].flatten())
        if not torch.isfinite(loss):
            diverged = True
            break
        value = float(loss.detach())
        # Frozen operational definition: after 20 prior steps, > prior-20 median + 0.5 nats.
        if len(hist) >= 20 and value > float(np.median(hist[-20:])) + cfg["spike_add_nats"]:
            spikes.append(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"]))
        if not math.isfinite(gn):
            diverged = True
            break
        for group in opt.param_groups:
            group["lr"] = cfg["lr"] * min(1.0, step / cfg["warmup"])
        opt.step()
        hist.append(value)
        gradnorms.append(gn)
    seconds = time.perf_counter() - t0
    eval_start = time.perf_counter()
    val, val_tokens = evaluate(net, valid, cfg["context"]) if not diverged else (None, 0)
    result = dict(width=width, layers=cfg["layers"], family=family, seed=seed,
                  parameters=sum(p.numel() for p in net.parameters()), steps_completed=len(hist),
                  initial_parameter_sha256=initial_hash, diverged=diverged,
                  final_train_loss=hist[-1] if hist else None,
                  train_loss_last20_mean=float(np.mean(hist[-20:])) if hist else None,
                  validation_loss=val, validation_tokens=val_tokens,
                  training_seconds=seconds, evaluation_seconds=time.perf_counter()-eval_start,
                  train_tokens_per_second=len(hist)*cfg["batch"]*cfg["context"]/seconds,
                  gradient_norm_max=max(gradnorms) if gradnorms else None,
                  clipped_steps=sum(g > cfg["clip"] for g in gradnorms),
                  loss_spike_steps=spikes, train_loss_history=hist, gradient_norm_history=gradnorms)
    return result


def freeze(config_path, output):
    cfg = json.loads(Path(config_path).read_text())
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(config=cfg, config_sha256=digest(config_path),
                    script_sha256=digest(__file__), dataset_sha256=digest(DATA),
                    canonical_torch_source_sha256=digest(ROOT / "mitsu_torch" / "__init__.py"),
                    python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    platform=platform.platform(), processor=platform.processor(),
                    cpu_count=os.cpu_count(), device="cpu", dtype="float32", threads=1,
                    freeze_note="Local content hashes recorded before official runs; no independent timestamp registration.")
    target = output / "manifest.json"
    if target.exists():
        old = json.loads(target.read_text())
        for key in ("config_sha256", "script_sha256", "dataset_sha256", "canonical_torch_source_sha256"):
            if old[key] != identity[key]:
                raise ValueError(f"Frozen manifest differs: {key}; choose a new output directory")
    else:
        target.write_text(json.dumps(identity, indent=2)+"\n")
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE/"pilot.json")
    p.add_argument("--output", type=Path, default=HERE/"results"/"pilot")
    p.add_argument("--freeze-only", action="store_true")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        p.error("worker must be between zero and workers-1")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = freeze(args.config, args.output)
    if args.freeze_only:
        print("Frozen manifest", args.output/"manifest.json")
        return
    train, valid, vocab = load_data()
    path = args.output/f"worker{args.worker}.jsonl"
    done = set()
    if path.exists():
        done = {(r["width"],r["seed"],r["family"]) for r in map(json.loads,path.read_text().splitlines())}
    pairs = [(w,s) for w in cfg["widths"] for s in cfg["seeds"]]
    for i, (width, seed) in enumerate(pairs):
        if i % args.workers != args.worker:
            continue
        families = ["mitsuglu", "swiglu"] if seed % 2 == 0 else ["swiglu", "mitsuglu"]
        for family in families:
            if (width, seed, family) in done:
                continue
            r = run_one(cfg, width, seed, family, train, valid, vocab)
            with path.open("a") as stream:
                stream.write(json.dumps(r, allow_nan=False)+"\n")
            print(f"width={width} seed={seed} {family} N={r['parameters']} val={r['validation_loss']} {r['training_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
