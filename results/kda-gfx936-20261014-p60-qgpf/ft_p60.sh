#!/bin/bash
# P-60 flash-train check (bw7 container tanbo_mega_k3): build ft_kdag2_git (P-58d applied) -> ft_site_r20,
# bitwise vs old FT (ft_site_r19 = 2f92ab3) and vs hip_kda (/opt/kda_env/g2_r5, P-58d), 2 dense + 2 varlen, pytest.
set -uo pipefail
export DTKROOT=/opt/dtk CMAKE_ARGS="-Dpybind11_DIR=/public/home/tanbo/xplat/pyb11/pybind11/share/cmake/pybind11"
export HIP_VISIBLE_DEVICES=6
FT=/public/home/tanbo/ft_kdag2_git; NEW=/opt/kda_env/ft_site_r20; OLD=/opt/kda_env/ft_site_r19
HIP=/opt/kda_env/g2_r5; K=/public/home/tanbo/kra/examples/kda; C=/tmp/ft_chk_p60
exec 9>/public/home/tanbo/codex_validation/locks/bw7/hcu_6.lock; flock -n 9 || { echo LOCK_BUSY; exit 75; }
hy-smi -d 6 --showmemuse 2>&1 | grep -q 'HCU memory use (%): 0' || { echo HBM_BUSY; exit 76; }
mkdir -p $C
if [ "${SKIP_BUILD:-0}" = 1 ]; then echo "BUILD_SKIPPED $(md5sum $NEW/flash_train/libflash_train.so | cut -c1-12)"; else
[ -e "$NEW" ] && { echo "$NEW exists"; exit 1; }
cd $FT && rm -rf /tmp/dist_r20 build-wheel
if ! python3 -m pip wheel . --no-deps -w /tmp/dist_r20 > /tmp/build_r20.log 2>&1; then echo BUILD_FAILED; tail -30 /tmp/build_r20.log; exit 1; fi
python3 -m pip install --no-deps --target $NEW /tmp/dist_r20/flash_train-*.whl >> /tmp/build_r20.log 2>&1 || { echo INSTALL_FAILED; exit 1; }
echo "BUILD_OK $(md5sum $NEW/flash_train/libflash_train.so | cut -c1-12)"; fi
fail=0
for s in "8192:12:" "8192:48:" "8192:12:0,1000,1063,1064,5000,8192" "8192:48:0,63,4097,8192"; do
  IFS=: read -r T H CUS <<<"$s"
  tag="${T}_${H}${CUS:+_vl$(echo "$CUS" | md5sum | cut -c1-3)}"; cu=${CUS:+--cu $CUS}
  PYTHONPATH=$OLD:/opt/kda_env:$HIP:/opt/kda_env/takeover python3 $K/ft_dump.py --tree $HIP --T $T --H $H $cu --out $C/old_$tag.pt > /dev/null 2>&1 || { echo "$tag old FT FAILED"; fail=1; continue; }
  PYTHONPATH=$NEW:/opt/kda_env:$HIP:/opt/kda_env/takeover python3 $K/ft_dump.py --tree $HIP --T $T --H $H $cu --out $C/new_$tag.pt > $C/new_$tag.log 2>&1 || { echo "$tag new FT FAILED"; fail=1; continue; }
  grep -q "$NEW" $C/new_$tag.log || { echo "$tag new FT not loaded from $NEW"; fail=1; }
  PYTHONPATH=/opt/kda_env:$HIP:/opt/kda_env/takeover KDA_G2=1 python3 $K/bitwise_gate.py dump --tree $HIP --T $T --H $H $cu --out $C/hip_$tag.pt > /dev/null 2>&1 || { echo "$tag hip_kda FAILED"; fail=1; continue; }
  for ref in old hip; do
    r=$(python3 $K/bitwise_gate.py cmp $C/${ref}_$tag.pt $C/new_$tag.pt | tail -1)
    echo "$tag ft_r20 vs $ref : $r"; [[ "$r" == *PASS* ]] || fail=1
  done
done
echo "== pytest torch_kda_test"
cd $FT && PYTHONPATH=$NEW:/opt/kda_env python3 -m pytest -q tests/python/torch_kda_test.py -p no:cacheprovider 2>&1 | tail -3
[ "${PIPESTATUS[0]}" = 0 ] || fail=1
echo "FT_CHECK $([ $fail = 0 ] && echo PASS || echo FAIL)"
