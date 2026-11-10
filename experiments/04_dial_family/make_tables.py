"""Build the per-family markdown tables for results7.md from results/summary.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "results", "summary.json")))
idx = {k: {r["name"]: r for r in v} for k, v in S.items()}
MARK = {"BETTER (screen)": " ▲", "worse": " ▼", "parity": "", "baseline": "", "diverged": ""}

def cls(ds, name):
    r = idx[ds].get(name)
    if r is None: return "—"
    if r["diverged"]: return f"diverged {r['diverged']}/5"
    if name == "mitsu": return f"{r['mean']:.2f}"
    return f"{r['mean']:.2f} ({r['delta']:+.2f}){MARK[r['verdict']]}"

def gpt(name):
    r = idx["gpt"].get(name)
    if r is None: return "—"
    if r["diverged"]: return f"diverged {r['diverged']}/3"
    if name == "mitsu": return f"{r['mean']:.4f}"
    return f"{r['mean']:.4f} ({r['delta']:+.4f}){MARK[r['verdict']]}"

def row(label, c, g): return f"| {label} | {cls('mnist', c)} | {cls('fashion', c)} | {gpt(g)} |"
HEAD = "| variant | MNIST acc % (Δ) | Fashion acc % (Δ) | char-GPT loss (Δ) |\n|---|---|---|---|"
fixed = ["fixed(0.9,0.9)", "fixed(0.75,1)", "fixed(0.5,1)", "fixed(0.75,0.75)", "fixed(0.5,0.5)", "fixed(1,1.25)", "fixed(1.5,1.5)"]
out = []
def sec(title, rows):
    out.append(f"\n**{title}**\n\n{HEAD}"); out.extend(rows)
sec("References", [row("`mitsu` (baseline, dials 1, 1)", "mitsu", "mitsu")] + [row(f"`{n}`", n, n) for n in ("gelu", "silu", "relu")])
sec("Fixed dials", [row(f"`{n}`", n, n) for n in fixed])
sec("Learnable dials", [row(f"`{n}`", n, n) for n in ("learn01", "learn02")])
sec("Input-driven dials", [row("`inscalar`", "inscalar", "inscalar"), row("`inunit` (matched params)", "inunit@22", "inunit"), row("`inunit` (width 64, 3× params)", "inunit@64", None)])
sec("Three-seat competition", [row("`tri` (matched params)", "tri@22", "tri"), row("`tri` (width 64, 3× params)", "tri@64", None)])
sec("Log versions", [row("`log-mitsu`", "log-mitsu", "log-mitsu")] + [row(f"`log-{n}`", "log-" + n, "log-" + n) for n in fixed + ["learn01", "learn02", "inscalar"]]
    + [row("`log-inunit` (matched params)", "log-inunit@22", "log-inunit"), row("`log-inunit` (width 64)", "log-inunit@64", None),
       row("`log-tri` (matched params)", "log-tri@22", "log-tri"), row("`log-tri` (width 64)", "log-tri@64", None)])
print("\n".join(out))
