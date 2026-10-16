#!/bin/bash
# P-82/P-82b flash-train check (bw7 container tanbo_mega_k3): build a copy of ft_kdag2_git + p82_ft.py -> ft_site_r26;
# bitwise vs old FT (ft_site_r25 = da6ef61) and vs hip_kda (/opt/kda_env/g2_r5), default and keep-state (state port),
# X1 chain, pytest; then keep-state wall A/B (KDA_G2_SAVE_H 0 vs 1).
set -uo pipefail
export DTKROOT=/opt/dtk CMAKE_ARGS="-Dpybind11_DIR=/public/home/tanbo/xplat/pyb11/pybind11/share/cmake/pybind11"
GPU=${1:-4}; export HIP_VISIBLE_DEVICES=$GPU
FT=/public/home/tanbo/ft_kdag2_git; W=/public/home/tanbo/xplat/ft82_build; NEW=/opt/kda_env/ft_site_r26; OLD=/opt/kda_env/ft_site_r25
HIP=/opt/kda_env/g2_r5; K=/public/home/tanbo/kra/examples/kda; C=/tmp/ft_chk_p82
exec 9>/public/home/tanbo/codex_validation/locks/bw7/hcu_$GPU.lock; flock -w 900 9 || { echo LOCK_BUSY; exit 75; }
hy-smi -d $GPU --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
mkdir -p $C
if [ "${SKIP_BUILD:-0}" = 1 ]; then echo "BUILD_SKIPPED"; else
[ -e "$NEW" ] && { echo "$NEW exists"; exit 1; }
[ -e $W/src/primitive/hygon/kda/g2_runner.cpp ] || { echo "no $W (prepare on the host)"; exit 1; }
cd $W && rm -rf /tmp/dist_r26
if ! python3 -m pip wheel . --no-deps -w /tmp/dist_r26 > /tmp/build_r26.log 2>&1; then echo BUILD_FAILED; grep -m10 -E "error" /tmp/build_r26.log; tail -20 /tmp/build_r26.log; exit 1; fi
python3 -m pip install --no-deps --target $NEW /tmp/dist_r26/flash_train-*.whl >> /tmp/build_r26.log 2>&1 || { echo INSTALL_FAILED; exit 1; }
echo "BUILD_OK $(md5sum $NEW/flash_train/libflash_train.so | cut -c1-12)"; fi
fail=0
for ks in 0 1; do
for s in "8192:12:" "8192:48:" "8192:12:0,1000,1063,1064,5000,8192" "8192:48:0,63,4097,8192" "32768:48:"; do
  IFS=: read -r T H CUS <<<"$s"
  tag="${T}_${H}${CUS:+_vl$(echo "$CUS" | md5sum | cut -c1-3)}_ks$ks"; cu=${CUS:+--cu $CUS}
  export FLASH_TRAIN_KDA_KEEP_STATE=$ks
  PYTHONPATH=$OLD:/opt/kda_env:$HIP:/opt/kda_env/takeover python3 $K/ft_dump.py --tree $HIP --T $T --H $H $cu --out $C/old_$tag.pt > /dev/null 2>&1 || { echo "$tag old FT FAILED"; fail=1; continue; }
  PYTHONPATH=$NEW:/opt/kda_env:$HIP:/opt/kda_env/takeover python3 $K/ft_dump.py --tree $HIP --T $T --H $H $cu --out $C/new_$tag.pt > $C/new_$tag.log 2>&1 || { echo "$tag new FT FAILED"; tail -5 $C/new_$tag.log; fail=1; continue; }
  grep -q "$NEW" $C/new_$tag.log || { echo "$tag new FT not loaded from $NEW"; fail=1; }
  unset FLASH_TRAIN_KDA_KEEP_STATE
  [ -e $C/hip_${tag%_ks*}.pt ] || PYTHONPATH=/opt/kda_env:$HIP:/opt/kda_env/takeover KDA_G2=1 python3 $K/bitwise_gate.py dump --tree $HIP --T $T --H $H $cu --out $C/hip_${tag%_ks*}.pt > /dev/null 2>&1 || { echo "$tag hip_kda FAILED"; fail=1; continue; }
  for ref in old hip; do
    f=$C/${ref}_$tag.pt; [ $ref = hip ] && f=$C/hip_${tag%_ks*}.pt
    r=$(python3 $K/bitwise_gate.py cmp $f $C/new_$tag.pt | tail -1)
    echo "$tag ft_r26 vs $ref : $r"; [[ "$r" == *PASS* ]] || fail=1
  done
done; done
for s in "8192:12:" "8192:12:0,1000,1063,1064,5000,8192" "8192:48:"; do
  IFS=: read -r T H CUS <<<"$s"; tag="x1_${T}_${H}${CUS:+_vl}"; cu=${CUS:+--cu $CUS}
  for v in r25 r26; do
    FLASH_TRAIN_KDA_KEEP_STATE=1 PYTHONPATH=/opt/kda_env/ft_site_$v:/opt/kda_env:$HIP:/opt/kda_env/takeover python3 $K/ft_dump.py --tree $HIP --T $T --H $H $cu --no-l2norm --out $C/${v}_$tag.pt > $C/${v}_$tag.log 2>&1 || { echo "$tag $v FAILED"; fail=1; }
  done
  r=$(python3 $K/bitwise_gate.py cmp $C/r25_$tag.pt $C/r26_$tag.pt | tail -1); echo "$tag (keep-state) r26 vs r25: $r"; [[ "$r" == *PASS* ]] || fail=1
done
echo "== pytest torch_kda_test"
cd $FT && PYTHONPATH=$NEW:/opt/kda_env python3 -m pytest -q tests/python/torch_kda_test.py -p no:cacheprovider 2>&1 | tail -3
[ "${PIPESTATUS[0]}" = 0 ] || fail=1
echo "FT_CHECK $([ $fail = 0 ] && echo PASS || echo FAIL)"
[ $fail = 0 ] || exit 1
cd /tmp
for TH in "16384 12" "16384 24"; do set -- $TH
  PYTHONPATH=$NEW:/opt/kda_env:$HIP:/opt/kda_env/takeover python3 $K/ft_pipe_cfg.py --tree $HIP --T $1 --H $2 --reps 8 \
    --cfg ks_mode2:FLASH_TRAIN_KDA_KEEP_STATE=1,KDA_G2_SAVE_H=0 --cfg ks_h:FLASH_TRAIN_KDA_KEEP_STATE=1 --cfg k0:FLASH_TRAIN_KDA_KEEP_STATE=0 2>&1 | grep -E "cfg|FT T=|rror"
done
echo FT82_DONE
