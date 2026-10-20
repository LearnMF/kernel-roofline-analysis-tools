export HIP_VISIBLE_DEVICES=4 KDA_G2=1 PYTHONDONTWRITEBYTECODE=1 BENCH_DENSE=1
export PYTHONPATH=/public/home/tanbo/xplat/ft_site_exp:/public/home/tanbo/xplat/bwenv:/public/home/tanbo/g2_opt:/public/home/tanbo/codex_validation/stage_g2_bw7/takeover
for st in 0 256 4096 69632 2097920 8400896 33554688; do
  echo "st2=$st $(FTRAIN_KDA_STAGGER_B2=$st ARMS=ftc,ftc python3 /public/home/tanbo/xplat/s4b_probe1.py 8192 24 2>/dev/null | tail -1 | grep -o "s4b_kernel<false>=[0-9.]*")"
done
