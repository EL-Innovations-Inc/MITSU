# Activation and gated-FFN performance

This benchmark measures MITSU, SiLU and exact GELU on the same hardware, dtype,
input shape and projection weights. It reports raw activation and
parameter-matched gated-FFN costs separately. It does not train the FFNs.

Install the package and PyTorch first (see the repository installation guide):

```bash
python benchmarks/activation_bench.py --device cpu --dtypes float32 bfloat16 float16 --output benchmarks/results_cpu.json
python benchmarks/summarize.py benchmarks/results_cpu.json --output benchmarks/RESULTS.md
# Run on a CUDA-equipped machine; no GPU results are shipped in this release:
python benchmarks/activation_bench.py --device cuda --dtypes float32 bfloat16 float16 --output benchmarks/results_cuda.json
# Optional compiler comparison (compilation happens in warmup):
python benchmarks/activation_bench.py --device cuda --dtypes bfloat16 --compile --warmup 10 --output benchmarks/results_cuda_compiled.json
```

The default tensor contains 262,144 elements. The gated FFNs process 64 token
rows at width 256 and hidden width 688, using identical bias-free gate, up and
down projection matrices. MITSU's FFN gate uses the frozen pilot gain 3.0374;
the other gates use 1.0. Raw activation measurements use gain 1.0 for all.

Each cell uses 5 untimed warmups and 15 timed repeats. Results include every
sample, median, range, input elements/second, environment and thread count.
Forward timing disables autograd. Forward/backward timing includes gradient
clearing and a common fp32 squared-mean loss, with no optimizer updates.
CUDA measurements synchronize before and after each repeat.

`explicit_tensor_bytes` counts the exact storage sizes of input, output,
parameters and live gradients for that cell. It **excludes** intermediates,
saved autograd tensors, temporary tensors, allocator caching and backend
workspaces, so it is not a peak-memory estimate. Native CPU allocator peaks
are unavailable and remain JSON `null`; Python allocation tracing would miss
native tensor allocations. CUDA runs additionally report total allocator peak,
baseline allocation and peak increase from a separate untimed call.

The bundled `results_cpu.json` records an actual CPU run; `RESULTS.md` summarizes the measured medians. Compare like-for-like
rows; timings are specific to the recorded machine, build and workload.
Unsupported dtype/device combinations are recorded explicitly rather than
invented. Compiler fusion, accelerator kernels, shapes and precision can all
change rankings. These measurements do not establish training quality or
stability, and CPU timings cannot establish GPU speed.
