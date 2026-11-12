#!/bin/bash
# abba_fb.sh GPU OUT REF VAR "T:H T:H ..." : mk3_fb.py (official fwd / bwd measurement) per shape, ABBA x2 (8 runs)
source /opt/dtk/env.sh >/dev/null 2>&1; cd /tmp; G=$1; O=$2; R=$3; V=$4; SHS=$5
M=/public/home/tanbo/mk3; B=/public/home/tanbo/b938; rm -f $O $O.done
for sh in $SHS; do T=${sh%%:*}; H=${sh##*:}
  for v in $R $V $V $R $R $V $V $R; do
    PYTHONPATH=$B/$v IMPL=ft HIP_VISIBLE_DEVICES=$G python3 $M/mk3_fb.py $T $H 2>/dev/null | grep "^{" | sed "s/^{/{\"tag\": \"$v\", /" >> $O
  done
done
echo DONE > $O.done
