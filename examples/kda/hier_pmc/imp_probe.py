import sys
m = sys.argv[1]
if m == "torch":
    import torch; torch.empty(1024, device="cuda"); print("torch", torch.__version__)
elif m == "cuda_alloc":
    import torch; t = torch.empty(1024 * 1024, device="cuda"); t.zero_(); import torch.cuda; torch.cuda.synchronize(); print("cuda alloc ok")
elif m == "fla":
    import fla; print("fla", fla.__file__)
elif m == "triton":
    import triton; print("triton", triton.__version__, triton.__file__)
elif m == "fla.ops.kda":
    import fla.ops.kda; print("fla.ops.kda ok")
elif m == "fla_prune_ns4":
    import fla_prune_ns4; print("prune import ok"); fla_prune_ns4.apply(verbose=True); print("prune apply ok")
elif m == "hip_kda":
    import hip_kda; print("hip_kda", hip_kda.__file__)
elif m == "g2_ops":
    import hip_kda.ops.g2_ops; print("g2_ops ok")
elif m == "fla_chunk":
    from fla.ops.kda import chunk_kda; print("fla chunk_kda ok")
elif m == "g2_chunk":
    from hip_kda.fla import chunk_kda; print("g2 chunk_kda ok")
else:
    print("unknown", m)
