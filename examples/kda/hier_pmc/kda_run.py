import runpy
import sys
sys.argv = ["kda_iter"] + sys.argv[1:]
runpy.run_path("/public/home/tanbo/kra/examples/kda/kda_iter.py", run_name="__main__")
