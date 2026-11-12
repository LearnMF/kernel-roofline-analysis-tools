#!/bin/bash
# fb_guard.sh GPU OUT REF VAR "T:H ..." : abba_fb.sh protocol (mk3_fb.py, ABBA x2 = 8 fresh processes per shape), but every
# process starts only after the GPU's HCU% reads 0 on 3 consecutive samples (3 s apart); each row records the wait and
# whether the GPU was busy right after the process (post_busy: a foreign job may have overlapped).
source /opt/dtk/env.sh >/dev/null 2>&1; cd /tmp; G=$1; O=$2; R=$3; V=$4; SHS=$5
M=/public/home/qianyongjun_zp/tanbo/mk3; B=/public/home/qianyongjun_zp/tanbo/b938; rm -f $O $O.done
hcu() { hy-smi 2>/dev/null | awk -v g=$G '$1 == g { n = 0; for (i = 2; i <= NF; i++) if ($i ~ /%$/) { n++; if (n == 2) { gsub("%", "", $i); print $i + 0; exit } } }'; }
wait_idle() { local t0=$(date +%s) ok=0
  while [ $ok -lt 3 ]; do u=$(hcu); if [ "${u:-100}" = "0" ]; then ok=$((ok + 1)); else ok=0; sleep 27; fi; sleep 3
    [ $(( $(date +%s) - t0 )) -gt 43200 ] && { echo "{\"abort\": \"GPU $G busy > 12 h\"}" >> $O; exit 1; }; done
  echo $(( $(date +%s) - t0 )); }
for sh in $SHS; do T=${sh%%:*}; H=${sh##*:}
  for v in $R $V $V $R $R $V $V $R; do
    w=$(wait_idle)
    line=$(PYTHONPATH=$B/$v IMPL=ft HIP_VISIBLE_DEVICES=$G python3 $M/mk3_fb.py $T $H 2>/dev/null | grep "^{")
    pb=$(hcu); pb=$([ "${pb:-100}" = "0" ] && echo false || echo true)
    echo "$line" | sed "s/^{/{\"tag\": \"$v\", \"host\": \"$(hostname)\", \"gpu\": $G, \"pre_wait_s\": $w, \"post_busy\": $pb, \"ts\": \"$(date +%H:%M:%S)\", /" >> $O
  done
done
echo DONE > $O.done
