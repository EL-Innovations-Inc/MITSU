# MITSU-8B — from-scratch pretraining battle plan
2026-09-01. Rung 3 of the ladder. Goal: a base 8B whose FFN nonlinearity is MITSU,
exported to GGUF, pulled via `ollama pull evanlittle/mitsu-8b` (~16 GB fp16).

## 0. The one-line economics
C = 6·N·D. Effective H100 throughput ≈ 4×10^14 FLOPS (989 TF bf16 × ~40% MFU).

| tokens D | FLOPs   | H100-hrs | cost @$2.2 | what it is |
|----------|---------|----------|------------|------------|
| 160B (Chinchilla) | 7.7e21 | ~5.3k  | **~$12-15k** | coherent base model, the target |
| 1T       | 4.8e22  | ~33k     | ~$75k      | notably stronger, optional phase 3 |
| 15T      | 7.2e23  | ~500k    | ~$1.1M     | Llama-3-8B territory — not the mission |

Hardware paths: 64×H100 ≈ 3.5 days, or one 8×H100 node ≈ 28 days (FSDP2 +
activation checkpointing fits: ~16 bytes/param of state = 128 GB sharded over 640 GB).
Renters: Lambda, Together, SF Compute, Voltage Park, RunPod. Reserved > spot for the long run.

## 1. Architecture: Llama-3 skeleton, one organ transplanted
32 layers, d=4096, 32 heads / 8 KV (GQA), RoPE, RMSNorm pre-norm, hidden 14336,
vocab 32k (BPE trained on our data; 128k buys little at 8B). ≈ 8.0B params.
FFN: SwiGLU → **MitsuGLU**: W2( MITSU(W1 x) ⊙ W3 x ). Only the nonlinearity changes.
Note: MITSU's slope at u=0 is ~0.09 vs SiLU's 0.5 — flatter gate at init. μP coord
checks will surface whether a fixed post-activation multiplier (×2–4, tuned once at
50M, transfers) is needed to match signal variance. Hypothesis to test, not assume:
MITSU's bounded derivative (−0.5, 1) may damp the loss-spike modes SiLU-GLU is prone to.

## 2. Stack
TorchTitan (Meta's official pretraining reference: FSDP2, TP, torch.compile, bf16)
or the OLMo 2 stack if we want their full data pipeline. Patch = one module
(mitsu_glu.py in this repo). Data: FineWeb-Edu (free, 15T tokens available; we
stream/pre-tokenize 170B). Pre-tokenized uint16 shards ≈ 340 GB on object storage.
Eval: lm-eval-harness (HellaSwag, ARC, PIQA, LAMBADA, WikiText ppl) + loss curves.
Logging: wandb. Checkpoint every 30 min; loss-spike protocol: rewind 1 ckpt, skip
the offending data window, resume (standard practice).

## 3. THE GATE — scaling ladder before any large spend (prereg5)
Nobody on Earth has MITSU scaling curves. That's the actual science, and it costs <$1k:
μP (maximal update parametrization) so LR tunes once at 50M and transfers.
Matched pairs MitsuGLU vs SwiGLU at 50M → 150M → 400M → 1B params,
Chinchilla-20× tokens each, same data order. Total ≈ 120 H100-hrs ≈ $300.
Pre-registered decision rule (full text in prereg5_scaling_ladder.md):
fit L(N) power laws to both families; proceed to 8B iff MITSU's fitted curve at
N=8B extrapolates within +0.01 nats of SwiGLU's, AND no stability incidents.
Either outcome is publishable; only one outcome spends $13k.

## 4. Phases and calendar (part-time, ~5 weeks)
P0 weekend, $0: tokenizer, data shards, TorchTitan+MitsuGLU patch, μP coord checks (this repo).
P1 ~1 week, <$1k: scaling ladder, fit curves, GATE decision.
P2 ~2 wks wall, $12-15k: the 8B run on 64×H100 (3.5 d) or 1 node (28 d), babysit ckpts.
P3 weekend, ~$300: SFT on open instruct data (Tulu/OpenHermes) → chat model.
P4 weekend, $0: convert_hf_to_gguf.py → llama-quantize Q4_K_M (4.9 GB) / keep fp16 (16 GB);
   write the ggml MITSU kernel (appendix below); `ollama create` + `ollama push`.

## 5. Risks, named
- MFU shortfall on rented interconnect → prefer single-node or NVLink-dense pods.
- Loss spikes at 8B → μP + z-loss + qk-norm option + rewind protocol.
- Ladder says MITSU scales worse → publish the curves, stop, $13k saved. The gate is the point.
- Eval disappointment vs Llama-3-8B → wrong comparison; ours ate 94× fewer tokens.
  The claim is "first MITSU LLM + its scaling law," not SOTA.

## Appendix: ggml kernel sketch (the 4 lines that make Ollama run it)
static inline float ggml_mitsu_f32(float u) {
    float s = (u > 20.f) ? u : log1pf(expf(u));   // softplus
    return s*(s-1.f)/(s+2.f);
}
// register as GGML_UNARY_OP_MITSU; SIMD/CUDA variants mirror the SiLU kernels.
