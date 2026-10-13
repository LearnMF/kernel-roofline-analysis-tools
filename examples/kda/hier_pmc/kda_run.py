# Pre-import the module that segfaults when first executed from inside fla's
# import chain (fla.ops.nsa.utils -> flash_attn.utils.sparse_utils), then run
# kda_iter. This keeps the production script untouched and identical semantics
# (fla_prune_ns4.apply still runs inside the bench source).
import flash_attn.utils.sparse_utils  # noqa: F401
import runpy
import sys
sys.argv = ["kda_iter"] + sys.argv[1:]
runpy.run_path("/public/home/tanbo/kra/examples/kda/kda_iter.py", run_name="__main__")
