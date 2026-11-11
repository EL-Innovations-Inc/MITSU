"""Check the experimental pairing and validation accounting, not model quality."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch.nn")
import torch
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("multiseed_bench", ROOT / "experiments/05_multiseed_scaling/bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_family_pair_shares_all_initial_weights():
    torch.manual_seed(17)
    a = bench.GPT(11, 16, 1, 8, "mitsuglu", 3.0374)
    torch.manual_seed(17)
    b = bench.GPT(11, 16, 1, 8, "swiglu", 3.0374)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert not torch.equal(a(torch.arange(8)[None]), b(torch.arange(8)[None]))


def test_validation_window_token_accounting():
    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, 7)
    loss, tokens = bench.evaluate(Uniform(), torch.arange(23) % 7, 8, batch=1)
    assert tokens == 16
    assert loss == pytest.approx(torch.log(torch.tensor(7.)).item())


analysis_spec = importlib.util.spec_from_file_location("multiseed_analysis", ROOT / "experiments/05_multiseed_scaling/analyze.py")
analysis = importlib.util.module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analysis)


def _rows(deltas=(.01, .02, -.01)):
    rows = []
    for seed, delta in enumerate(deltas):
        for family, loss in (("mitsuglu", 2.0 + delta), ("swiglu", 2.0)):
            rows.append(dict(width=192, seed=seed, family=family, parameters=1000,
                initial_parameter_sha256=str(seed), diverged=False, steps_completed=100,
                validation_loss=loss, training_seconds=1., train_tokens_per_second=100., loss_spike_steps=[]))
    return rows


def test_missing_or_wrong_horizon_pair_withholds_inference():
    rows = _rows()
    rows[-1]["steps_completed"] = 99
    s = analysis.paired_summary(rows, 192, [0, 1, 2], 100, 512)
    assert s["n_pairs"] == 2
    assert s["paired_ci95"] is None
    assert s["paired_mitsu_mean"] == pytest.approx(2.015)
    assert s["paired_silu_mean"] == 2.0
    assert "horizon" in s["exclusions"][0]["reason"]
    rows = _rows()[:-1]
    assert analysis.paired_summary(rows, 192, [0, 1, 2], 100, 512)["paired_ci95"] is None


def test_degenerate_seed_variance_never_produces_zero_width_interval():
    s = analysis.paired_summary(_rows((.01, .01, .01)), 192, [0, 1, 2], 100, 512)
    assert s["paired_ci95"] is None
    assert "degenerate" in s["inference_status"]


def test_complete_pairs_have_interval_and_exact_exposure():
    s = analysis.paired_summary(_rows(), 192, [0, 1, 2], 100, 512)
    assert s["paired_ci95"][0] < s["paired_mean_delta"] < s["paired_ci95"][1]
    assert s["families"]["mitsuglu"]["training_tokens_total"] == 3 * 100 * 512
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.paired_summary(_rows() + [_rows()[0]], 192, [0, 1, 2], 100, 512)


def test_empty_report_is_safe_and_uses_configured_seed_count(tmp_path):
    cfg = dict(steps=1000, seeds=list(range(5)), batch=8, context=64)
    s = analysis.paired_summary([], 192, cfg["seeds"], 1000, 512)
    analysis.write_report(tmp_path, cfg, [s])
    text = (tmp_path / "REPORT.md").read_text()
    assert "5 paired seeds per size" in text
    assert "512,000 training tokens" in text
    assert "experiment 06" not in text
    assert "withheld" in text


def test_divergence_withholds_inference_even_when_other_pairs_finite():
    rows = _rows()
    rows[0]["diverged"] = True
    rows[0]["validation_loss"] = None
    s = analysis.paired_summary(rows, 192, [0, 1, 2], 100, 512)
    assert s["n_pairs"] == 2
    assert s["paired_ci95"] is None
    assert s["families"]["mitsuglu"]["divergences"] == 1
