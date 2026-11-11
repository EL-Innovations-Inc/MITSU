# Pre-registration 2 — width-matched follow-up
Written BEFORE the run. 2026-09-01. Question: is tri's digits deficit the 3×-preact
parameter tax, or the competition mechanism itself?

## Design
Same protocol as run 1 (5 seeds, same splits, lr 3e-3, clip 5, epochs 60/80).
New cells, both matching directions:
  - tri-WIDE: W = run-1 MLP width (digits 28, cancer 33, wine 36, iris 38).
    Deliberately param-UNmatched (~2.4× the MLP): tests if width alone closes the gap.
  - MLP-NARROW: relu/gelu/mitsu at H = 16 = tri's run-1 width.
    Tests if MLPs also suffer at 16 units/layer.
Run-1 cells (tri-16, MLP-wide) carry over unchanged (deterministic seeds).
Decision-relevant dataset: digits ONLY (the others are ceiling-limited; reported for completeness).

## Predictions
Q1 width helps: tri-28 beats tri-16 on digits by ≥0.5 pts; I expect tri-28 in 96.0–96.9.
Q2 mechanism contributes: at equal width 16, best MLP-16 still beats tri-16 by >1 pooled std.
Q3 gap not fully closed: tri-28 stays >1 pooled std below best MLP-28 (97.07) despite 2.4× params.
Q4 stability: zero divergences at W=28+ (per-triple math is width-independent).

## Pre-committed interpretation
- PURE WIDTH TAX: tri-28 within 1 pooled std of best MLP-28 AND tri-16 within 1 std of best MLP-16.
- PURE MECHANISM: width doesn't help (tri-28 − tri-16 < 0.5) AND tri-16 < best MLP-16.
- MIXED (my prediction): width helps but tri-28 still short by >1 pooled std.
No post-hoc rescue; whichever cell pattern appears is the verdict.
