cd /tmp
TREE=${TREE:-/public/home/tanbo/g2_p45}
export KDA_G2=1 HIP_VISIBLE_DEVICES=${GPU:-5} PYTHONPATH=/public/home/tanbo/xplat/bwenv:$TREE:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
G=/public/home/tanbo/kra/examples/kda/bitwise_gate.py
SPECS=${SPECS:-"8192:12:extreme 16384:48:x 8192:12:cu"}
for spec in $SPECS; do
  IFS=: read -r T H mode <<< "$spec"; extra=""
  [ $mode = extreme ] && extra="--gate-extreme"; [ $mode = cu ] && extra="--cu 0,1000,1063,5000,8192"
  KDA_G2_S4BT=0 python3 $G dump --tree $TREE --T $T --H $H --out /tmp/s4bt_b_$$.pt $extra > /dev/null 2>&1
  KDA_G2_S4BT=1 python3 $G dump --tree $TREE --T $T --H $H --out /tmp/s4bt_c_$$.pt $extra > /tmp/s4bt_c_$$.log 2>&1 || tail -3 /tmp/s4bt_c_$$.log
  echo "== $T $H $mode"
  python3 - <<PY
import torch
x, y = torch.load("/tmp/s4bt_b_$$.pt"), torch.load("/tmp/s4bt_c_$$.pt")
for k in x:
    a, b = x[k].float(), y[k].float()
    if torch.equal(x[k], y[k]): print(f"  {k:8s} BITWISE"); continue
    d = (a - b).abs()
    print(f"  {k:8s} max|d| {float(d.max()):.3e}  max|ref| {float(a.abs().max()):.3e}  rel {float(d.max()/a.abs().max()):.2e}  "
          f"mean|d|/mean|ref| {float(d.mean()/a.abs().mean()):.2e}  diff {int((d>0).sum())}/{d.numel()}")
PY
done
