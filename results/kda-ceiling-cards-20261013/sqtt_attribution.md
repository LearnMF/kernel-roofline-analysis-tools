# SQTT stall attribution — production instances (xprof SQTT -> XCompute)

latency_share: stall-latency share per opcode group; bubble: dominant issue-gap transitions.
Instances: fwd_h #1=MODE1(ILi1ELb1), #2=MODE4(ILi4ELb0 raw-Gn at H>=48 / ILi4ELb1);
dhu: 16K/12 = v8b prod instance ILi1ELb1, 16K/48 = v8b4 generic ILi0ELb0 (production).

## 16K/H48

| kernel | avg waves | stall latency 构成 | top bubbles |
|---|---|---|---|
| g2_wu_kernelILb0E | 624 | waitcnt 65%, valu_int_addr 20%, lds 5%, salu 2%, valu_fp 2% | valu_int_addr->waitcnt 58%, barrier->waitcnt 9%, lds->waitcnt 6% |
| p3_wy_s4a_kernelILb0ELb0E | 618 | waitcnt 57%, global_load 14%, valu_int_addr 12%, global_store 6%, salu 4% | global_load->waitcnt 47%, valu_int_addr->waitcnt 12%, barrier->valu_int_addr 7% |
| g2_fwd_h_v8o4_kernelILb0ELi1ELb1E | 556 | global_load 52%, lds 12%, waitcnt 11%, valu_int_addr 10%, global_store 8% | valu_int_addr->global_load 28%, barrier->global_load 15%, waitcnt->global_store 9% |
| p3_wy_s4b_kernelILb0E | 615 | waitcnt 39%, global_load 20%, valu_int_addr 17%, lds 6%, valu_fp 5% | valu_int_addr->waitcnt 30%, valu_int_addr->global_load 17%, barrier->valu_int_addr 9% |
| g2_fwd_prep_a_kernelILi0ELb0E | 932 | waitcnt 37%, valu_int_addr 25%, valu_fp 9%, global_load 7%, global_store 6% | barrier->waitcnt 26%, valu_int_addr->waitcnt 10%, valu_int_addr->valu_int_addr 9% |
| g2_bwd_tail_kernelILb0E | 1629 | global_load 37%, waitcnt 31%, valu_int_addr 17%, global_store 7%, valu_fp 4% | valu_int_addr->global_load 30%, barrier->waitcnt 12%, valu_int_addr->waitcnt 10% |
| g2_dhu_v8b4_kernelILi0ELb0E | 234 | valu_int_addr 27%, global_load 18%, waitcnt 12%, lds 11%, salu 11% | valu_int_addr->global_load 19%, valu_int_addr->global_store 14%, valu_int_addr->waitcnt 8% |
| g2_fwd_h_v8o4_kernelILb0ELi4ELb0E | 564 | global_load 25%, global_store 20%, valu_int_addr 18%, lds 16%, waitcnt 10% | valu_int_addr->global_store 15%, mmac->global_load 12%, barrier->lds 11% |

## 16K/H12

| kernel | avg waves | stall latency 构成 | top bubbles |
|---|---|---|---|
| g2_bwd_tail_reduce_kernelENS_10TailParamsE | 10 | waitcnt 100% |  |
| g2_wu_kernelILb0E | 615 | waitcnt 64%, valu_int_addr 20%, lds 5%, salu 2%, valu_fp 2% | valu_int_addr->waitcnt 57%, barrier->waitcnt 9%, lds->waitcnt 6% |
| p3_wy_s4a_kernelILb0ELb1E | 610 | waitcnt 57%, valu_int_addr 13%, global_load 10%, global_store 6%, salu 5% | global_load->waitcnt 47%, valu_int_addr->waitcnt 9%, salu->waitcnt 8% |
| g2_fwd_prep_a_kernelILi0ELb0E | 910 | waitcnt 36%, valu_int_addr 24%, global_load 9%, valu_fp 9%, global_store 6% | barrier->waitcnt 24%, valu_int_addr->global_load 10%, valu_int_addr->waitcnt 9% |
| p3_wy_s4b_kernelILb0E | 605 | waitcnt 35%, valu_int_addr 20%, global_load 16%, lds 7%, valu_fp 6% | valu_int_addr->waitcnt 27%, valu_int_addr->global_load 15%, barrier->valu_int_addr 10% |
| g2_bwd_tail_kernelILb0E | 1701 | global_load 35%, waitcnt 30%, valu_int_addr 20%, global_store 7%, valu_fp 4% | valu_int_addr->global_load 28%, barrier->waitcnt 12%, valu_int_addr->waitcnt 10% |
| g2_dhu_v8b_kernelILi1ELb1E | 225 | valu_int_addr 30%, global_load 24%, lds 12%, salu 9%, global_store 9% | barrier->valu_int_addr 20%, mmac->global_load 17%, valu_int_addr->global_store 11% |
| g2_fwd_h_v8o4_kernelILb0ELi4ELb1E | 91 | valu_int_addr 27%, waitcnt 26%, lds 22%, global_load 8%, mmac 7% | valu_int_addr->lds 12%, lds->waitcnt 11%, valu_int_addr->waitcnt 9% |
| g2_fwd_h_v8o4_kernelILb0ELi1ELb1E | 94 | global_load 24%, waitcnt 23%, lds 19%, valu_int_addr 19%, mmac 9% | valu_int_addr->global_load 16%, valu_int_addr->waitcnt 12%, lds->waitcnt 11% |

