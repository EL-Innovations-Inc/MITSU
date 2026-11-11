"""Regression tests for inference, pairing integrity and historical preservation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dial_analysis", ROOT / "experiments/04_dial_family/analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(name, seed, value, diverged=False):
    return dict(cell=name, seed=seed, acc=value, diverged=diverged, params=10)


def comparison(values, baseline=None):
    baseline = baseline if baseline is not None else [0.8] * len(values)
    rows = [row("mitsu", s, value) for s, value in enumerate(baseline)]
    rows += [row("other", s, value) for s, value in enumerate(values)]
    return next(r for r in analysis.table(rows, "cell", "acc", True, 100, "test") if r["name"] == "other")


def test_interval_matches_student_t_and_margin_is_separate():
    delta = np.array([0.1, -0.1, 0.2, 0.0, 0.1])
    got = analysis.paired_statistics(delta, 0.5)
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    expected = delta.mean() + np.array([-1, 1]) * t.ppf(0.975, 4) * se
    assert np.allclose(got["ci95"], expected)
    assert got["equivalent"] is True
    assert analysis.paired_statistics(delta, 0.01)["equivalent"] is False


def test_nonsignificant_does_not_establish_equivalence():
    got = comparison([0.78, 0.82, 0.79, 0.81, 0.80])
    assert got["p_two_sided"] > 0.05
    assert got["verdict"] == "inconclusive"
    assert got["equivalence_established"] is False


def test_small_differences_can_establish_equivalence():
    got = comparison([0.8001, 0.7999, 0.8002, 0.8, 0.8001])
    assert got["verdict"] == "equivalent within margin"
    assert got["equivalence_established"] is True


def test_difference_and_practical_equivalence_are_distinct():
    got = analysis.paired_statistics([0.100, 0.101, 0.102, 0.099, 0.100], 0.5)
    assert got["p_two_sided"] < 0.05
    assert got["equivalent"] is True


def test_degenerate_and_insufficient_samples_withhold_inference():
    assert analysis.paired_statistics([], 0.5)["status"] == "insufficient paired seeds"
    assert analysis.paired_statistics([0.0], 0.5)["ci95"] is None
    assert analysis.paired_statistics([0.1] * 5, 0.5)["status"] == "degenerate paired variance"
    assert comparison([0.8] * 5)["verdict"] == "degenerate paired variance"


def test_duplicate_seed_is_rejected_instead_of_silently_overwritten():
    with pytest.raises(ValueError, match="duplicate result"):
        analysis.table([row("mitsu", 0, 0.8), row("mitsu", 0, 0.9)], "cell", "acc", True, 100, "test")


def test_unmatched_seeds_are_reported_and_inference_withheld():
    rows = [row("mitsu", s, 0.8) for s in range(4)]
    rows += [row("other", s, 0.8 + s * 0.0001) for s in range(3)]
    got = next(r for r in analysis.table(rows, "cell", "acc", True, 100, "test") if r["name"] == "other")
    assert got["missing_variant_seeds"] == [3]
    assert got["n"] == 3
    assert got["verdict"] == "incomplete paired data"
    assert got["equivalence_established"] is False


@pytest.mark.parametrize("bad,diverged,verdict", [(float("nan"), False, "invalid observations"), (0.8, True, "diverged")])
def test_invalid_or_diverged_pair_cannot_be_claimed_equivalent(bad, diverged, verdict):
    rows = [row("mitsu", s, 0.8) for s in range(5)]
    rows += [row("other", s, 0.8 + s * 0.0001) for s in range(4)]
    rows += [row("other", 4, bad, diverged)]
    got = next(r for r in analysis.table(rows, "cell", "acc", True, 100, "test") if r["name"] == "other")
    assert got["verdict"] == verdict
    assert got["n"] == 4
    assert got["inference_eligible"] is False


@pytest.mark.parametrize("margin", [0, -1, float("nan"), float("inf")])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError):
        analysis.paired_statistics([0.0, 0.1], margin)


def test_invalid_differences_rejected():
    with pytest.raises(ValueError, match="finite"):
        analysis.paired_statistics([float("nan"), 1], 0.5)


def test_reanalysis_preserves_historical_summary_and_raw_files():
    folder = ROOT / "experiments/04_dial_family"
    paths = [folder / "prereg7.md", folder / "FROZEN.txt"]
    paths += list((folder / "results").glob("*.jsonl"))
    paths += [folder / "results/summary.json", folder / "results/summary_confirm.json"]
    before = {p: p.read_bytes() for p in paths}
    analysis.main([])
    analysis.main(["_confirm"])
    assert all(p.read_bytes() == before[p] for p in paths)
    assert (folder / "results/summary_v1_0_1.json").exists()
    assert (folder / "results/report_v1_0_1_confirm.md").exists()
