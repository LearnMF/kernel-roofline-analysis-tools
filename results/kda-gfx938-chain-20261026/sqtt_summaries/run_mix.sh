#!/bin/bash
cd /public4/home/tanbo/sqtt938
nohup python3 chain_mix.py chain_2048_16 kw_2048_16.json > mix_2048_16.txt 2>&1 &
MAXF=4 nohup python3 chain_mix.py chain_8192_64 kw_8192_64.json > mix_8192_64.txt 2>&1 &
