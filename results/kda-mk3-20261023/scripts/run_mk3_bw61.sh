#!/bin/bash
# Yu Hui's Kimi-K3 3-layer recipe (examples/kimik3/run_kimik3_3layer.sh) on one node (bw61, 8 GPUs),
# with the KDA kernel switched by KDA_IMPL.  usage: run_mk3_bw61.sh <fla|ft> <iters> <tag>
IMPL=$1; ITERS=${2:-30}; TAG=${3:-run}
M=/public4/home/tanbo/mk3/Megatron-LM-k3; D=$M/examples/kimik3; L=/public4/home/tanbo/mk3/e2e_logs
mkdir -p $L
source /opt/dtk/env.sh >/dev/null 2>&1
cd $D
mpirun -np 8 --allow-run-as-root --bind-to none -x KDA_IMPL=$IMPL -x TRAIN_ITERS=$ITERS \
  bash -c "source /opt/dtk/env.sh >/dev/null 2>&1 && source $M/requirements/env.sh && cd $D && \
    bash train_kimik3_3layer.sh 127.0.0.1 11453 --launch_backend=mpirun --checkpoint_path=./ckpt \
    --launch_with_binding=$M/requirements/launch_with_binding.sh --profiling=" > $L/${TAG}_${IMPL}.log 2>&1
echo "rc=$? log=$L/${TAG}_${IMPL}.log"
