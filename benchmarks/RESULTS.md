# Recorded cpu performance sample

Measured 2026-11-11T20:38:37.937906+00:00 using PyTorch 2.15.0+cpu, 1 thread(s), eager execution. This was a shared host without process isolation; repeat on deployment hardware before making a performance decision.

Median milliseconds across 15 repeats after 5 warmups. Lower is faster. See `results_cpu.json` for every timing sample, ranges, complete configuration and explicit tensor sizes.

| dtype | workload | operation | MITSU ms | SiLU ms | GELU ms |
|---|---|---|---:|---:|---:|
| float32 | elementwise | forward | 1.6369 | 0.1019 | 0.0876 |
| float32 | elementwise | forward_backward | 4.1453 | 0.8252 | 1.5109 |
| float32 | gated_ffn | forward | 0.6190 | 0.5289 | 0.5442 |
| float32 | gated_ffn | forward_backward | 2.4725 | 1.6294 | 1.7018 |
| bfloat16 | elementwise | forward | 1.3364 | 0.1355 | 0.0872 |
| bfloat16 | elementwise | forward_backward | 5.6066 | 1.3978 | 1.3949 |
| bfloat16 | gated_ffn | forward | 0.9612 | 0.8415 | 0.8373 |
| bfloat16 | gated_ffn | forward_backward | 2.9253 | 2.7343 | 3.0720 |
| float16 | elementwise | forward | 1.3456 | 0.1166 | 0.0902 |
| float16 | elementwise | forward_backward | 5.3744 | 1.3637 | 1.4208 |
| float16 | gated_ffn | forward | 0.6698 | 0.5835 | 0.5427 |
| float16 | gated_ffn | forward_backward | 2.1183 | 1.8880 | 1.8405 |

MITSU's implementation contains multiple tensor operations and promotes low-precision inputs internally for numerical safety; the built-in gates have specialized kernels. Projection work reduces relative activation overhead in FFNs, and small timing differences are sensitive to shared-host noise. These measurements do not establish training quality or an ordering on other hardware.

CPU native peak allocator memory is not measured and remains `null`. Explicit tensor sizes count inputs, outputs, parameters and gradients, excluding saved autograd tensors, temporary tensors and backend workspaces. These counts must not be interpreted as peak-memory measurements. CUDA runs additionally report separate allocator peaks.

36/36 measured cells had finite outputs and gradients.

No CUDA measurements are included; CPU timings cannot establish GPU speed.
