# SQTT stall attribution — PRODUCTION driver (kda_iter.py) under xprof --enable-sqtt

NOTE: the previous version measured bn_sqtt.hip harness instances; several of those
(fwd_h generic MODE 0/2, dhu<0,false>, prep_a<0,false>) are not what production runs.
This file is regenerated from the production driver; every row's full symbol is shown
and was verified present in the same shape's production hip-trace (verification.md F1).
Extra kernels now covered: dav, gate, l2n_apply, exp2, prep_b.

## 16K/H48

| kernel (full symbol) | avg waves | stall latency 构成 | top bubbles |
|---|---|---|---|
| `_ZN2g214g2_gate_kernelILi1ELb0EEEvNS_10GateParamsE` | 2414 | valu_int_addr 75%, valu_fp 13%, waitcnt 7%, salu 3%, transcendental 1% | salu->valu_int_addr 62%, valu_int_addr->valu_int_addr 17%, valu_fp->valu_fp 4% |
| `_ZN2g214g2_exp2_kernelEPfl` | 1151 | waitcnt 69%, salu 31% | salu->waitcnt 100% |
| `_ZN2g212g2_wu_kernelILb0EEEvNS_8WuParamsE` | 621 | waitcnt 62%, valu_int_addr 22%, lds 6%, salu 2%, valu_fp 2% | valu_int_addr->waitcnt 57%, barrier->waitcnt 12%, lds->waitcnt 10% |
| `_ZN2g213g2_dav_kernelILb0EEEvNS_9DavParamsE` | 853 | waitcnt 55%, global_load 12%, lds 10%, valu_int_addr 9%, salu 7% | salu->waitcnt 32%, valu_int_addr->waitcnt 19%, barrier->waitcnt 10% |
| `_ZN2g220g2_fwd_h_v8o4_kernelILb0ELi1ELb1EEEvNS_10FwdOParamsE` | 698 | global_load 49%, waitcnt 14%, lds 14%, valu_int_addr 10%, mmac 6% | valu_int_addr->global_load 27%, barrier->global_load 14%, global_load->global_load 8% |
| `_ZN2g212_GLOBAL__N_119g2_l2n_apply_kernelEPKsS2_PKfS4_PsS5_l` | 2556 | waitcnt 42%, salu 39%, global_load 11%, valu_int_addr 5%, global_store 3% | global_store->salu 41%, salu->waitcnt 33%, valu_int_addr->global_load 11% |
| `_ZN2g216p3_wy_s4a_kernelILb0ELb0EEEvNS_11WyS4AParamsE` | 616 | waitcnt 39%, valu_int_addr 19%, global_load 18%, global_store 9%, salu 6% | valu_int_addr->waitcnt 26%, barrier->valu_int_addr 14%, salu->waitcnt 9% |
| `_ZN2g216p3_wy_s4b_kernelILb0EEEvNS_11WyS4BParamsE` | 615 | waitcnt 38%, valu_int_addr 19%, global_load 18%, lds 6%, valu_fp 5% | valu_int_addr->waitcnt 32%, valu_int_addr->global_load 16%, barrier->valu_int_addr 10% |
| `_ZN2g22r420g2_fwd_prep_a_kernelILi1ELb0EEEvNS0_11PrepAParamsE` | 927 | valu_int_addr 36%, waitcnt 24%, valu_fp 20%, salu 5%, transcendental 5% | barrier->waitcnt 17%, valu_int_addr->valu_int_addr 16%, valu_int_addr->waitcnt 10% |
| `_ZN2g218g2_bwd_tail_kernelILb0EEEvNS_10TailParamsE` | 1644 | global_load 34%, waitcnt 31%, valu_int_addr 21%, global_store 6%, valu_fp 4% | valu_int_addr->global_load 28%, barrier->waitcnt 13%, valu_int_addr->waitcnt 10% |
| `_ZN2g22r420g2_fwd_prep_b_kernelILb0EEEvNS0_11PrepBParamsE` | 896 | waitcnt 30%, global_load 28%, global_store 15%, valu_int_addr 10%, salu 7% | valu_int_addr->waitcnt 17%, valu_int_addr->global_load 16%, valu_int_addr->global_store 14% |
| `_ZN2g218g2_dhu_v8b4_kernelILi0ELb0EEEvNS_10BwdNParamsE` | 247 | valu_int_addr 28%, global_load 19%, lds 11%, global_store 11%, waitcnt 10% | valu_int_addr->global_load 24%, valu_int_addr->global_store 14%, mmac->mmac 10% |
| `_ZN2g220g2_fwd_h_v8o4_kernelILb0ELi4ELb0EEEvNS_10FwdOParamsE` | 547 | global_load 26%, global_store 21%, valu_int_addr 17%, lds 16%, waitcnt 11% | valu_int_addr->global_store 16%, mmac->global_load 13%, barrier->lds 11% |

## 16K/H12

| kernel (full symbol) | avg waves | stall latency 构成 | top bubbles |
|---|---|---|---|
| `_ZN2g214g2_exp2_kernelEPfl` | 1403 | waitcnt 82%, salu 18% | salu->waitcnt 55%, valu_int_addr->waitcnt 45% |
| `_ZN2g214g2_gate_kernelILi1ELb0EEEvNS_10GateParamsE` | 2229 | valu_int_addr 72%, valu_fp 14%, waitcnt 7%, global_load 3%, salu 2% | salu->valu_int_addr 61%, valu_int_addr->valu_int_addr 14%, barrier->waitcnt 4% |
| `_ZN2g212g2_wu_kernelILb0EEEvNS_8WuParamsE` | 616 | waitcnt 58%, valu_int_addr 25%, lds 6%, global_store 2%, valu_fp 2% | valu_int_addr->waitcnt 55%, barrier->waitcnt 12%, lds->waitcnt 9% |
| `_ZN2g213g2_dav_kernelILb0EEEvNS_9DavParamsE` | 831 | waitcnt 55%, global_load 13%, lds 9%, valu_int_addr 9%, salu 6% | salu->waitcnt 32%, valu_int_addr->waitcnt 21%, valu_int_addr->global_load 10% |
| `_ZN2g212_GLOBAL__N_119g2_l2n_apply_kernelEPKsS2_PKfS4_PsS5_l` | 2501 | waitcnt 42%, salu 38%, global_load 10%, valu_int_addr 5%, global_store 2% | global_store->salu 41%, salu->waitcnt 33%, valu_int_addr->global_load 11% |
| `_ZN2g22r420g2_fwd_prep_a_kernelILi1ELb0EEEvNS0_11PrepAParamsE` | 914 | valu_int_addr 42%, valu_fp 20%, waitcnt 17%, transcendental 5%, salu 5% | valu_int_addr->valu_int_addr 30%, barrier->waitcnt 10%, valu_fp->valu_fp 9% |
| `_ZN2g216p3_wy_s4a_kernelILb0ELb1EEEvNS_11WyS4AParamsE` | 606 | waitcnt 42%, valu_int_addr 19%, global_load 11%, global_store 9%, salu 7% | valu_int_addr->waitcnt 19%, salu->waitcnt 18%, barrier->valu_int_addr 13% |
| `_ZN2g216p3_wy_s4b_kernelILb0EEEvNS_11WyS4BParamsE` | 613 | waitcnt 36%, valu_int_addr 20%, global_load 17%, lds 7%, valu_fp 6% | valu_int_addr->waitcnt 30%, valu_int_addr->global_load 15%, barrier->valu_int_addr 10% |
| `_ZN2g218g2_bwd_tail_kernelILb0EEEvNS_10TailParamsE` | 1736 | global_load 33%, waitcnt 30%, valu_int_addr 22%, global_store 6%, valu_fp 4% | valu_int_addr->global_load 28%, barrier->waitcnt 14%, valu_int_addr->waitcnt 9% |
| `_ZN2g22r420g2_fwd_prep_b_kernelILb0EEEvNS0_11PrepBParamsE` | 879 | waitcnt 30%, global_load 28%, global_store 14%, valu_int_addr 10%, salu 7% | valu_int_addr->waitcnt 17%, valu_int_addr->global_load 17%, valu_int_addr->global_store 13% |
| `_ZN2g220g2_fwd_h_v8o4_kernelILb0ELi4ELb1EEEvNS_10FwdOParamsE` | 91 | waitcnt 29%, lds 25%, valu_int_addr 22%, global_load 9%, mmac 7% | valu_int_addr->lds 17%, lds->waitcnt 16%, valu_int_addr->global_load 13% |
| `_ZN2g217g2_dhu_v8b_kernelILi1ELb1EEEvNS_10BwdNParamsE` | 230 | valu_int_addr 29%, global_load 25%, lds 12%, global_store 9%, salu 9% | barrier->valu_int_addr 22%, mmac->global_load 18%, valu_int_addr->global_store 11% |
| `_ZN2g220g2_fwd_h_v8o4_kernelILb0ELi1ELb1EEEvNS_10FwdOParamsE` | 140 | global_load 26%, waitcnt 24%, lds 20%, valu_int_addr 17%, mmac 9% | valu_int_addr->global_load 16%, valu_int_addr->waitcnt 12%, lds->waitcnt 11% |

