"""EXPLORATORY (not pre-registered): where is the bottom of the char-GPT bowl?
Fits a parabola to the mean paired difference against log2(c) over the six fixed values."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = {r["name"]: r for r in json.load(open(os.path.join(HERE, "results", "summary.json")))["gpt"]}
cs = np.array([0.25, 0.5, 1, 2, 4, 8]); names = ["c=0.25", "c=0.5", "c=1", "mitsu", "c=4", "c=8"]
d = np.array([rows[n]["delta"] for n in names])
a, b, c0 = np.polyfit(np.log2(cs), d, 2)
xmin = -b / (2 * a)
print("delta by c:", dict(zip(cs.tolist(), d.round(4).tolist())))
print(f"parabola in log2(c): minimum at c = {2 ** xmin:.2f}, fitted delta there {a * xmin ** 2 + b * xmin + c0:+.4f} nats")
print(f"fitted delta at c = 2: {a + b + c0:+.4f} nats")
