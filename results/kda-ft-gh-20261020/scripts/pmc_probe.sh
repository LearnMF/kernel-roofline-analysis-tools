export HIP_VISIBLE_DEVICES=4 KDA_G2=1 PYTHONDONTWRITEBYTECODE=1 BENCH_DENSE=1
export PYTHONPATH=/public/home/tanbo/xplat/ft_site_ghf:/public/home/tanbo/xplat/bwenv:/public/home/tanbo/g2_opt:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
for a in g2 ftc; do
  for t in 1 3; do
    O=/tmp/pmc_${a}_$t; rm -rf $O; mkdir -p $O
    ARMS=$a hipprof --pmc --pmc-type $t --kernel-name p3_wy_s4b_kernel --output-type 0 -o $O/p.csv python3 /public/home/tanbo/xplat/pmc_run.py 8192 24 > $O/log.txt 2>&1
    echo "$a type$t rc=$? $(ls $O | head -3 | tr "\n" " ")"
  done
done
