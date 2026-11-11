#!/usr/bin/env python3
"""Measure raw activations and parameter-matched gated FFNs without training.

Run from a checkout after ``pip install -e '.[torch]'``. Timing numbers depend
on hardware, PyTorch version, tensor shapes, dtype, compiler and thread count.
CPU allocator peaks are deliberately not estimated using Python tracemalloc.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn


class GatedFFN(nn.Module):
    """Same three bias-free projections for every gate family."""
    def __init__(self, dim, hidden, activation, gain=1.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.activation = activation
        self.gain = gain

    def forward(self, x):
        return self.down((self.gain * self.activation(self.gate(x))) * self.up(x))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def explicit_memory(module, x, output, backward):
    """Known live tensors only; this is NOT a peak-allocation estimate."""
    pbytes = sum(tensor_bytes(p) for p in module.parameters())
    gradbytes = sum(tensor_bytes(p.grad) for p in module.parameters() if p.grad is not None)
    if x.grad is not None:
        gradbytes += tensor_bytes(x.grad)
    out = dict(input=tensor_bytes(x), output=tensor_bytes(output), parameters=pbytes,
               gradients=gradbytes if backward else 0)
    out["total_excluding_saved_tensors_and_workspaces"] = sum(out.values())
    return out


def run_case(module, shape, dtype, device, mode, warmup, repeats, compiled):
    torch.manual_seed(20261111)
    x = torch.randn(*shape, dtype=dtype, device=device, requires_grad=(mode == "forward_backward"))
    module = module.to(device=device, dtype=dtype)
    model = torch.compile(module) if compiled else module

    def operation():
        if mode == "forward":
            with torch.inference_mode():
                return model(x)
        model.zero_grad(set_to_none=True)
        x.grad = None
        y = model(x)
        # A common scalar loss, including the same reduction and fp32 cast
        # overhead for every family. No optimizer or parameter update.
        y.float().square().mean().backward()
        return y

    for _ in range(warmup):
        y = operation()
        del y
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter_ns()
        y = operation()
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
        del y

    # Separate untimed call; peak includes the live inputs/model/gradients.
    baseline = None
    if device.type == "cuda":
        model.zero_grad(set_to_none=True)
        x.grad = None
        synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    y = operation()
    synchronize(device)
    memory = explicit_memory(module, x, y, mode == "forward_backward")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    finite = bool(torch.isfinite(y).all())
    if mode == "forward_backward":
        finite = finite and bool(torch.isfinite(x.grad).all()) and all(
            bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None)
    median = statistics.median(samples)
    return dict(status="ok", median_ms=median, min_ms=min(samples), max_ms=max(samples),
                samples_ms=samples, input_elements_per_second=x.numel() / (median / 1000),
                explicit_tensor_bytes=memory, finite_outputs_and_gradients=finite,
                cuda_peak_allocated_bytes=peak, cuda_baseline_allocated_bytes=baseline,
                cuda_peak_increment_bytes=(peak - baseline) if peak is not None else None,
                cpu_peak_allocator_bytes=None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", help="cpu, cuda or cuda:0")
    p.add_argument("--dtypes", nargs="+", choices=["float32", "float16", "bfloat16"], default=["float32"])
    p.add_argument("--elements", type=int, default=262144, help="elements in raw activation tensor")
    p.add_argument("--tokens", type=int, default=64, help="flattened batch/token rows for FFNs")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=688)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--compile", action="store_true", dest="compiled", help="compile modules; compilation excluded by warmup")
    p.add_argument("--output", type=Path, default=Path("benchmarks/results_cpu.json"))
    args = p.parse_args(argv)
    for name in ["elements", "tokens", "dim", "hidden", "threads", "warmup", "repeats"]:
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    from mitsu_torch import MITSU
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no GPU measurements were produced.")
    configs = dict(device=str(device), dtypes=args.dtypes, elements=args.elements,
                   tokens=args.tokens, dim=args.dim, hidden=args.hidden, threads=args.threads,
                   warmup=args.warmup, repeats=args.repeats, compiled=args.compiled,
                   mitsu_ffn_gain=3.0374, other_ffn_gains=1.0,
                   forward_mode="inference_mode; no autograd",
                   backward_loss="mean(output.float() ** 2); no optimizer updates")
    env = dict(measured_utc=datetime.now(timezone.utc).isoformat(),
               python=sys.version, pytorch=torch.__version__, platform=platform.platform(),
               processor=platform.processor(), logical_cpus=os.cpu_count(),
               cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
               device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               torch_num_threads=torch.get_num_threads(),
               cpu_instruction_capability=torch.backends.cpu.get_cpu_capability())
    results = []
    factories = {"mitsu": MITSU, "silu": nn.SiLU, "gelu": nn.GELU}
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for workload in ("elementwise", "gated_ffn"):
            # Identical projection weights across all families in this cell.
            torch.manual_seed(20261111)
            template = GatedFFN(args.dim, args.hidden, nn.Identity())
            for family, factory in factories.items():
                module = factory() if workload == "elementwise" else copy.deepcopy(template)
                if workload == "gated_ffn":
                    module.activation = factory()
                    module.gain = 3.0374 if family == "mitsu" else 1.0
                shape = (args.elements,) if workload == "elementwise" else (args.tokens, args.dim)
                for mode in ("forward", "forward_backward"):
                    row = dict(workload=workload, family=family, mode=mode, dtype=dtype_name,
                               shape=list(shape))
                    try:
                        row.update(run_case(copy.deepcopy(module), shape, dtype, device, mode,
                                            args.warmup, args.repeats, args.compiled))
                    except (RuntimeError, NotImplementedError) as exc:
                        row.update(status="unsupported_or_failed", reason=f"{type(exc).__name__}: {exc}")
                    results.append(row)
                    print(f"{dtype_name:8s} {workload:11s} {family:5s} {mode:16s} "
                          + (f"{row['median_ms']:.4f} ms" if row['status'] == 'ok' else row['reason']), flush=True)
    payload = dict(schema_version=1, environment=env, config=configs, results=results,
                   memory_notes="Explicit tensor sizes exclude saved autograd tensors, temporary tensors, allocator caching and backend workspaces. CPU native peak allocation is unavailable and reported as null. CUDA allocator peaks, when available, are measured separately from timing. Python tracemalloc is not used.",
                   scope="Hardware-specific microbenchmark; no accuracy, training stability or GPU performance claim follows from CPU timings.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return 0 if all(r["status"] == "ok" and r["finite_outputs_and_gradients"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
