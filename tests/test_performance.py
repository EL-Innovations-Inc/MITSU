"""Small end-to-end benchmark schema check; no timing thresholds."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_cpu_benchmark_records_real_cells_without_fabricated_peak_memory(tmp_path):
    pytest.importorskip("torch.nn")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "measurements.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(root / "benchmarks" / "activation_bench.py"),
         "--elements", "128", "--tokens", "2", "--dim", "8", "--hidden", "16",
         "--warmup", "1", "--repeats", "2", "--output", str(output)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(output.read_text())
    assert len(report["results"]) == 12  # three gates, two workloads, two modes
    for row in report["results"]:
        assert row["status"] == "ok" and row["finite_outputs_and_gradients"]
        assert len(row["samples_ms"]) == 2
        assert row["median_ms"] > 0
        assert row["cpu_peak_allocator_bytes"] is None
        assert row["cuda_peak_allocated_bytes"] is None
