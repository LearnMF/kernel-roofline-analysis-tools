#!/bin/bash
# run on each p8c node inside tanbo_kda: t24_launch.sh "G:T:H G:T:H ..."
M=/public/home/qianyongjun_zp/tanbo/mk3; O=/public/home/qianyongjun_zp/tanbo/t24; h=$(hostname)
for s in $1; do G=${s%%:*}; TH=${s#*:}; nohup bash $M/fb_guard.sh $G $O/t24_${h}_$G.jsonl a16 bc24 "$TH" > $O/log_${h}_$G.txt 2>&1 & done
wait
