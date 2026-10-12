cd /tmp
TREE=$1; T=$2; H=$3; shift 3
export KDA_G2=1 HIP_VISIBLE_DEVICES=${GPU:-4} PYTHONPATH=/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
for E in "$@"; do
  rm -rf /tmp/ktenv; mkdir -p /tmp/ktenv
  env $E hipprof --hip-trace --output-type 0 -o /tmp/ktenv/x python3 /public/home/tanbo/kra/examples/kda/kda_iter.py --tree $TREE --arm g2 --T $T --H $H --iters 3 > /dev/null 2>&1
  echo "== $E"
  PYTHONPATH=/public/home/tanbo/kra python3 - <<PY
from kra.timeline.hipprof_json import load
from kra.timeline.analyze import split_windows
import re, collections
tr = load("/tmp/ktenv/x.json")
w = split_windows(tr, "spin_kernel", None)[-1]
d = collections.OrderedDict()
for o in w:
    if o.kind != "kernel": continue
    m = re.search(r"(g2_\w+?_kernel|p3_wy_\w+?_kernel)", o.name); n = m.group(1) if m else o.name[:24]
    d[n] = d.get(n, 0) + (o.end_ns - o.begin_ns) / 1e3
for n, v in d.items(): print(f"  {n:30s} {v:9.1f}")
print(f"  {TOTAL:30s} {sum(d.values()):9.1f}")
PY
done
