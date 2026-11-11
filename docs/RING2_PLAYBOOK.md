> Historical research proposal, retained from v1.0.0. Hardware costs and run times
> are unverified planning estimates. No large-model run or stability advantage is
> established by this document. See [STABILITY.md](STABILITY.md) and experiment 05
> for the v1.0.1 interpretation and measured scope.

# RING 2 PLAYBOOK — the GPU ladder, step by step
Prerequisite (Ring 1 handoff): torch repo with MitsuGLU/SwiGLU switch, muP
parametrization passing coord checks, gate_mult frozen, FineWeb-Edu tokenized
to uint16 shards on the NAS (~40 GB covers the biggest single run).

## Ladder design (muP-correct: depth FIXED at 12, width scales)
| rung | d_model | ~params | tokens (20x) | where | wall time |
|------|---------|---------|--------------|-------|-----------|
| R1   | 448     | ~57M    | 1.1B         | A4500 | ~2.5 h/run |
| R2   | 768     | ~135M   | 2.7B         | A4500 | ~23 h/run |
| R3   | 1408    | ~375M   | 7.5B         | A4500 or rent | ~6.7 d/run local, ~13 H100-h rented |
| R4   | 2304    | ~910M   | 18B          | RENT  | ~76 H100-h/run |
Common: L=12, context 2048, vocab 32k, bf16, FlashAttention, torch.compile,
8-bit Adam, grad checkpointing, batch ~0.5M tokens via accumulation, WSD schedule.
Fixed 100M-token validation set, identical for every run.

## Step 1 — Freeze prereg5-final (30 min, before ANY run)
Fill the numeric blanks and commit: the table above; LR sweep grid (8 log-spaced
points, 3e-4 to 3e-2 muP base LR); seeds (2 per family at R1-R2, +1 replicate
pair at R2; 1 per family at R3; 1 per family at R4); spike definition (val/train
loss jump > 0.3 nats within 50 steps, or grad-norm > 10x trailing median);
recovery protocol (rewind 1 ckpt, skip 1 data window, log as RECOVERED; NaN or
non-recovery = UNRECOVERED). GATE (already frozen): MITSU fitted L(8e9) within
+0.010 nats of SwiGLU AND zero UNRECOVERED MITSU incidents AND replicate
|Delta_rep - Delta_orig| < 0.010.

## Step 2 — LR sweeps at R1 (A4500, ~2 days wall, $0)
8 LRs x 2 families x 1.1B tokens. Pick argmin val loss per family; muP transfers
it up the ladder. Sanity: the two optima should land within ~2x of each other.
Log everything to a public wandb project from day one.

## Step 3 — R2 pair + replicate (A4500, ~4 days, $0)
Four 23-hour runs, sequential, cron-guarded, checkpoint to NAS every 30 min.
Compute Delta and Delta_rep. If |Delta_rep - Delta_orig| >= 0.010: STOP, diagnose
variance, add seeds at R2 before spending on R3.

## Step 4 — R3 pair (choose your patience)
Local: ~13.5 days sequential, $0, the A4500 hums while you live your life.
Rented: ~27 H100-hours, ~$55. Either is pre-approved; record which in the log.

## Step 5 — Rental setup (half a day, do during Step 3/4)
Provider at the ~$2/hr tier (RunPod / Lambda / Vast / GMI). One H100 80GB per
run, both runs in parallel. Push code via git; push the 40 GB token shards once
to a persistent volume (or object storage). Checkpoint every 30 min to volume.
Auto-resume script for spot preemption. Test the whole loop with a 15-minute
smoke run BEFORE launching real jobs.

## Step 6 — R4, the summit pair (rented, ~3.5 days wall, ~$350)
Two 76-H100-hour runs in parallel. Monitor loss + the S2 spike tally (automated
logging per the frozen definition) + grad norms. Babysit twice daily, not
continuously. Total ring spend lands ~$355-420 with storage.

## Step 7 — Fit and verdict (2 hours)
Collect final val losses; fit L(N) = a*N^(-alpha) + c per family; bootstrap CIs
over seeds; extrapolate to 8e9; evaluate the three gate conditions; append the
verdict to prereg5-final with the date. PASS -> Ring 3 unlocked. FAIL -> Ring 3
stays locked and the result publishes anyway.

## Step 8 — Step B fires the same week (pass or fail)
arXiv v2 with the GPU curves, seeds, and the spike tally (the S2 answer is
publishable either direction). Close the SR&ED log for the ring: hours, rental
receipts, the dated preregs. Consulting card goes live with the wandb link as
the living portfolio.

## Standing safety rails
Discretionary funds only. Every config change after Step 1 gets a dated
amendment note. Nothing about the gate moves after data exists.
