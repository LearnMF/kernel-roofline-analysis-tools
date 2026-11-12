#!/bin/bash
# fb4.sh REF VAR TAG : mk3_fb ABBA x2 (abba_fb.sh) for the 9 shapes on GPUs 1 2 3 7 -> /tmp/fb_TAG_G.jsonl
M=/public/home/tanbo/mk3; R=$1; V=$2; T=$3
bash $M/abba_fb.sh 1 /tmp/fb_${T}_1.jsonl $R $V "8192:64 1024:16" &
bash $M/abba_fb.sh 2 /tmp/fb_${T}_2.jsonl $R $V "32768:16 2048:8" &
bash $M/abba_fb.sh 3 /tmp/fb_${T}_3.jsonl $R $V "32768:64" &
bash $M/abba_fb.sh 7 /tmp/fb_${T}_7.jsonl $R $V "4096:32 16384:8 2048:16 1024:64" &
wait; echo DONE > /tmp/fb_$T.done
