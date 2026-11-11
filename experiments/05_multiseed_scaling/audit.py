"""Verify frozen hashes, complete records, and correlated-horizon prefixes."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(name):
    directory = HERE / "results" / name
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["script_sha256"] == digest(HERE / "bench.py"), "Runner changed since freeze"
    assert manifest["config_sha256"] == digest(HERE / f"{name}.json"), "Config changed since freeze"
    assert manifest["dataset_sha256"] == digest(HERE / "data" / "shakespeare.txt"), "Dataset changed"
    assert manifest["canonical_torch_source_sha256"] == digest(ROOT / "mitsu_torch" / "__init__.py"), "Activation changed"
    cfg = manifest["config"]
    expected = {(w, s, f) for w in cfg["widths"] for s in cfg["seeds"] for f in ("mitsuglu", "swiglu")}
    records = {}
    for path in directory.glob("worker*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row["width"], row["seed"], row["family"]
            assert key not in records, f"Duplicate {key}"
            assert row["steps_completed"] == cfg["steps"], f"Incomplete horizon {key}"
            assert not row["diverged"], f"Diverged {key}"
            records[key] = row
    assert set(records) == expected, "Missing or unexpected runs"
    for w in cfg["widths"]:
        for seed in cfg["seeds"]:
            a, b = records[w, seed, "mitsuglu"], records[w, seed, "swiglu"]
            assert a["initial_parameter_sha256"] == b["initial_parameter_sha256"]
    return cfg, records


def main():
    pilot_cfg, pilot = load_run("pilot")
    extension_cfg, extension = load_run("extension")
    modified = {k for k in set(pilot_cfg) | set(extension_cfg) if pilot_cfg.get(k) != extension_cfg.get(k)}
    assert modified == {"steps"}, "Unexpected recipe change"
    for key, row in extension.items():
        count = pilot[key]["steps_completed"]
        assert row["initial_parameter_sha256"] == pilot[key]["initial_parameter_sha256"]
        assert row["train_loss_history"][:count] == pilot[key]["train_loss_history"]
        assert row["gradient_norm_history"][:count] == pilot[key]["gradient_norm_history"]
    report = dict(hash_checks="PASS", expected_record_checks="PASS", matched_initial_weights="PASS",
        prefix_train_loss_and_gradient_norm_checks="PASS", pilot_records=len(pilot), extension_records=len(extension),
        seed_size_pairs=len(pilot)//2, horizons=2,
        interpretation="The same six seed-size pairs follow identical first-100-step trajectories. The two horizons are correlated observations, not independent replication or 24 independent trials.")
    (HERE / "results" / "extension" / "PREFIX_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
