> Historical experiment report. “Parity” below is the original decision-rule
> terminology, not a new equivalence-test conclusion. See
> [v1.0.1 statistical guidance](../../docs/STATISTICS.md) and
> [the new multi-seed pilot](../05_multiseed_scaling/README.md).

# 03 · Nano scaling ladder

MitsuGLU against SwiGLU at four model sizes on character-level WikiText-2. At each size
("rung") the two families share identical initial weights and batch order; the only
difference is the gate nonlinearity. This is the CPU pilot of the GPU ladder described in
[`../../docs/RING2_PLAYBOOK.md`](../../docs/RING2_PLAYBOOK.md).

Pre-registrations: [`prereg6.md`](prereg6.md) (this run) and
[`prereg5_scaling_ladder.md`](prereg5_scaling_ladder.md) (the GPU design it pilots).

`gate_mult` was measured at setup and frozen at 3.0374.

## Result

Validation loss in nats (`ladder_state.json`); Δ = MitsuGLU − SwiGLU, negative favours MITSU.

| rung | parameters | MitsuGLU | SwiGLU | Δ |
|---|---|---|---|---|
| 0 | 31,936 | 2.3955 | 2.3962 | −0.0007 |
| 1 | 88,032 | 1.8160 | 1.8180 | −0.0021 |
| 2 | 176,976 | 1.5620 | 1.5657 | −0.0037 |
| 3 | 483,184 | 1.3689 | 1.3712 | −0.0022 |

Mean paired Δ = −0.0022 nats. Fitted power laws L(N) = a·N^(−α) + c (`ladder_fit.json`):
α = 0.6093 for MitsuGLU, 0.6063 for SwiGLU. Zero divergences in 8 runs. All three
conditions of the pre-registered nano-gate are met.

The fit's extrapolation to 8 billion parameters (1.1227 vs 1.1232) is reported, as
pre-registered, strictly "for entertainment": four points between 32k and 483k parameters
cannot support a claim at that size.

Limits stated in the pre-registration: one paired seed per rung, standard parametrisation
rather than muP, character-level, CPU float32.

## Reproduce

Needs numpy and scipy. Reuses the transformer primitives from `../02_char_gpt/`.

```
./get_data.sh                 # WikiText-2
python3 ladder.py gradcheck   # gate: both families < 1e-4
python3 ladder.py setup       # freezes vocab, budgets, gate_mult
python3 ladder.py run 600     # resumable; repeat until QUEUE COMPLETE
python3 ladder.py fit         # power laws + extrapolation
```

`setup` and `run` rewrite `ladder_state.json`.
