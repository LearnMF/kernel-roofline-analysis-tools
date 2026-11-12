# BW1000 operator profile (measured)

`bw1000_torch271_dtk2604.yaml` is copied unchanged from
[HYGON-AI/hcu-train-simulator](https://github.com/HYGON-AI/hcu-train-simulator)
(`src/hcu_train_simulator/profiles/builtin/`, commit `55da280`, Apache-2.0).
It is the measurement set that simulator's tiered GEMM coefficients were read off.

Environment: BW200 / UBB BW1000, 64 GiB, torch 2.7.1, DTK-26.04-rc4, bf16 only.

What one `mean_ms` is (from `benchmarks/operators.py` in that repo):

- **gemm**: one training step of `y = x @ W; y.float().sum().backward()` with
  `x[m,k]`, `W[k,n]` -- i.e. forward + dX + dW (3 GEMMs, `tflops` uses `6mnk`)
  **plus** the harness's cast/sum/cast-back kernels, averaged over 20 iterations after
  5 warm-ups, timed with CUDA events around the whole step.
- Entries whose `tflops / (6mnk / mean_ms)` is not 1 were timed on a different shape
  than the key names (grouped MoE GEMMs, `topk_router` with softmax/top-k, one
  `attn_proj`, `vision_patch_embed` conv): `kra validate` excludes them and says why.
- **non_gemm**: torch-eager forward+backward of composite ops; `elements` is the
  simulator's module-level element count, not always one tensor.

Use: `python -m kra validate --machine machines/gfx936-bw1000.json --profile examples/bw1000_profile/bw1000_torch271_dtk2604.yaml`
