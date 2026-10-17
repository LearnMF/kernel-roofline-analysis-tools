#!/bin/bash
K=/public/home/tanbo/kra/examples/kda
cd /public/home/tanbo && rm -rf g2_p91t && cp -a g2_opt g2_p91t && cd g2_p91t && rm -rf build
python3 $K/p91_s4b_dma.py csrc/g2 --tri || exit 1
python3 build_g2.py > /tmp/b91t.log 2>&1 && echo build_ok || { grep -m8 error /tmp/b91t.log; exit 1; }
mkdir -p /tmp/isa91t && cd /tmp/isa91t && rm -f * && cp /public/home/tanbo/g2_p91t/hip_kda/_G2*.so g2.so && PATH=/opt/dtk/aillvm/bin:$PATH /opt/dtk/bin/extractkernel -i g2.so > /dev/null 2>&1
/opt/dtk/aillvm/bin/llvm-readelf --notes g2.so-000-gfx936.hsaco | grep -B3 -A60 "p3_wy_s4b_kernelILb0E" | grep -E "group_segment_fixed_size|vgpr_count|private_segment" | head -3 | tr '\n' ' '; echo
/opt/dtk/aillvm/bin/llvm-objdump -d --symbolize-operands g2.so-000-gfx936.hsaco > s4.dis 2>/dev/null
awk '/<_ZN2g216p3_wy_s4b_kernelILb0EEEvNS_11WyS4BParamsE>:/{f=1} f&&/^$/{f=0} f' s4.dis > s4b.dis
grep -n "buffer_load_dwordx4.*lds\|s_waitcnt\|s_barrier" s4b.dis | awk -F: '{print $1": "$2}' | sed -n 1,400p | grep -A6 "lds" | head -60
echo B91T_DONE
