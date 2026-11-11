"""EXPLORATORY (not pre-registered, added after the prereg8 runs): one learned c
PER UNIT instead of one per layer. Asked because of the question "can c act as a
filter?": do the units of a layer choose different cut-offs when they are free to?

    python3 exploratory_unit.py cls      # MNIST + Fashion-MNIST, seeds 0-4
    python3 exploratory_unit.py gpt      # char-GPT, seeds 0-2

Same benchmarks, seeds and initial weights as prereg8, so each run pairs with the
`mitsu` run already in results/. Writes results/exploratory_unit_<bench>.jsonl.
"""
import json, math, os, sys
import torch
import torch.nn as nn
import variants

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "learnc-unit"


class UnitLayer(variants.Layer):
    """Layer whose learned c is a vector, one entry per unit, each starting at 2."""

    def __init__(self, spec, d_in, width, init_w, bias=True, init_aux=None):
        super().__init__(spec, d_in, width, init_w, bias=bias, init_aux=init_aux)
        self.theta = nn.Parameter(torch.full((width,), math.log(spec["init"])))

    def dials(self):
        return [round(float(v), 4) for v in self.c().detach()]


def specs():
    return {NAME: dict(kind="learn", init=2.0)}


def main(which):
    if which == "cls":
        import bench_cls as B
        B.specs, B.Layer = specs, UnitLayer
        for ds in ("mnist", "fashion"):
            data = B.load(ds)
            with open(os.path.join(HERE, "results", f"exploratory_unit_{ds}.jsonl"), "w") as f:
                for seed in range(5):
                    r = B.run(ds, data, NAME, NAME, 64, seed)
                    f.write(json.dumps(r) + "\n"); f.flush()
                    print(ds, "seed", seed, "acc %.2f" % (100 * r["acc"]), "div", r["diverged"], flush=True)
    else:
        import bench_gpt as B
        B.specs, B.Layer = specs, UnitLayer
        tr, va, V = B.data()
        with open(os.path.join(HERE, "results", "exploratory_unit_gpt.jsonl"), "w") as f:
            for seed in range(3):
                r = B.run(NAME, seed, 2000, tr, va, V)
                f.write(json.dumps(r) + "\n"); f.flush()
                print("gpt seed", seed, "val %.4f" % r["val"], "div", r["diverged"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
