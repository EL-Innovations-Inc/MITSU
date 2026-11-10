# 02 · Character-GPT twins

Two identical character-level transformers on Tiny Shakespeare that differ in exactly one
line: the MLP activation, MITSU or GELU. Same initial weights, same batch schedule, so any
difference in loss is down to the nonlinearity. Pure numpy, hand-derived backprop.

Pre-registration: [`prereg4.md`](prereg4.md).

Config: context 64, width 64, 2 heads, 2 pre-norm blocks, about 110k parameters, 2000 steps.

## Result

| twin | validation loss (nats) |
|---|---|
| MITSU | 1.7107 |
| GELU | 1.7062 |

Δ = 0.0045 nats, inside the pre-registered parity band of 0.05. The weakly-held direction
pick in the pre-registration (MITSU ≤ GELU) missed by that margin. Neither twin diverged.
This is a single paired run, a pilot; experiment 04 repeats the comparison on three seeds.

`ckpt_mitsu.npz` and `ckpt_gelu.npz` are the trained weights from that run.

## Reproduce

Needs numpy only.

```
./get_data.sh                                  # Tiny Shakespeare
python3 mitsu_gpt.py gradcheck                 # gate
python3 mitsu_gpt.py sample mitsu "ROMEO:"     # sample from the shipped weights
python3 mitsu_gpt.py train mitsu 2000 1200     # <mitsu|gelu> <steps> <max seconds>; resumable
```

Training resumes from `ckpt_<name>.npz` when that file exists, and the shipped checkpoints are
already at step 2000. To train from scratch, move them aside first.

`Modelfile` is the Ollama recipe for the later, larger model; it needs real exported weights
and a MITSU kernel in the runtime before it works (see the comments inside it).
