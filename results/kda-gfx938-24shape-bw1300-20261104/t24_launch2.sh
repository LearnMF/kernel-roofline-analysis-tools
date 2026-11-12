#!/bin/bash
# t24_launch2.sh OUTDIR "G:T:H ..." (inside tanbo_kda on a p8c node)
M=/public/home/qianyongjun_zp/tanbo/mk3; O=$1; h=$(hostname); mkdir -p $O
for s in $2; do G=${s%%:*}; TH=${s#*:}; nohup bash $M/fb_guard.sh $G $O/t24_${h}_$G.jsonl a16 bc24 "$TH" > $O/log_${h}_$G.txt 2>&1 & done
wait
