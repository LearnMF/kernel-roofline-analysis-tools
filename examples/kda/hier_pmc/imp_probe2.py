import sys, torch  # torch first: flash_attn requires it
m = sys.argv[1]
if m == "flash_attn":
    import flash_attn; print("flash_attn", flash_attn.__version__)
elif m == "fa_utils":
    import flash_attn.utils.sparse_utils; print("fa sparse_utils ok")
elif m == "fla":
    import fla; print("fla", getattr(fla, "__version__", "?"), fla.__file__)
elif m == "fla_ops_kda":
    import fla.ops.kda; print("fla.ops.kda ok")
elif m == "fla_chunk":
    from fla.ops.kda import chunk_kda; print("fla chunk_kda ok")
elif m == "prune_apply":
    import fla_prune_ns4; fla_prune_ns4.apply(verbose=True); print("prune ok")
elif m == "g2_only_bench":
    # emulate the bench setup minus the fla import: does the rest survive?
    import os
    src = open("/opt/kda_env/g2_r5/tests/megatron/kda_op_bench_tp.py").read().split("res, live = {}, []")[0]
    src = src.replace("import fla_prune_ns4", "fla_prune_ns4 = None")
    src = src.replace("from fla.ops.kda import chunk_kda as fla_chunk_kda", "fla_chunk_kda = None")
    g = {"__name__": "bench"}
    exec(compile(src, "kda_op_bench_tp", "exec"), g)
    print("bench setup ok", [k for k in list(g)[:8]])
else:
    print("unknown", m)
