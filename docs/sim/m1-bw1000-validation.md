# M1 验证：shape-only 下限 vs BW1000 实测

数据：`examples/bw1000_profile/bw1000_torch271_dtk2604.yaml`（hcu-train-simulator 内置 BW1000 实测 profile，
torch 2.7.1 / DTK-26.04-rc4，bf16；也是其分档系数 0.15/0.40/0.75/1.0 的来源数据）。
复现：`python -m kra validate --machine machines/gfx936-bw1000.json --profile examples/bw1000_profile/bw1000_torch271_dtk2604.yaml`

## 结论

1. **下限成立**：109 条可用 GEMM + 111 条 non_gemm，`T_theory <= T_meas` 零违例（实测档、手册档峰值都成立）。
2. **同一张表上，`下限 + 两个全局常数` 比分档系数准约 3 倍**（留一法，即每条都用"没见过它"的常数预测）：

   | 预测器 | MAPE | 中位误差 | 最大误差 | Spearman |
   | --- | ---: | ---: | ---: | ---: |
   | hcu 分档系数（4 档 + gemm_efficiency，且是在这张表上读出来的） | 44.4% | 33.9% | 185% | 0.854 |
   | kra：`T_theory / eta`（1 个常数） | 58.4% | 60.4% | 149% | 0.903 |
   | kra：`floor + T_theory / eta`（2 个常数） | **15.0%** | **13.2%** | 52% | 0.901 |

3. **单常数比分档还差，原因是一个 ~530us 的每次调用地板**：最小的 GEMM（如 `16x4096x2048`，下限 45us）实测仍要 680us。
   这不是硬件开销（BW1000 launch 2.2us），而是 benchmark harness 本身：每条计时的是一次 Python 侧
   `y = x @ W; y.float().sum().backward(); zero_grad` 的整步（3 个 GEMM + cast/sum/cast 及 autograd 调度），
   CUDA event 包住整个 Python 调用。分档表里"小 shape 乘 0.15"实际上主要是在吸收这个地板，而不是 GEMM 本身慢。
   拟合值：floor 533us，eta 0.718（大 shape 的 bf16 GEMM 约达到峰值的 72%）。
4. **non_gemm 实测离 HBM 下限 10~100 倍**（eta 中位 0.064）：这批是 torch-eager 拼出来的复合算子，
   时间被多次 kernel launch 和中间张量读写吃掉——正是融合 kernel 能拿回的空间，不是模型误差。

## 口径与排除

- 每条 GEMM 的 `mean_ms` = 前向 + dX + dW 三个 GEMM（profile 里 `tflops` 用 `6mnk`），所以用 `--phase fwd+bwd` 比。
- 排除 21 条：grouped MoE GEMM（key 的 m,n,k 不是被计时的每专家 shape）、`topk_router`（含 softmax/top-k）、
  `vision_patch_embed`（conv，无 tflops）、1 条 `attn_proj`（`tflops` 与 `6mnk/mean_ms` 对不上，像是计时了别的 shape）。
  判据统一为 `tflops / (6mnk / mean_ms)` 偏离 1 超过 1%。
- non_gemm 排除 `gated_delta_rule` / `gdn_causal_conv1d` / `optimizer_step`（递推/卷积/优化器，需要独立插件，属 M2）。
- non_gemm 的下限按"一元算子前反向最少 5 次张量流量（读 x、写 y、读 dy、读 x、写 dx）+ 2 次 launch"，
  `elements` 取 simulator 的模块级元素数，偏保守。

## 这对后续意味着什么

- 给余晖 simulator 的接口可以是：每算子 `T_theory`（无需实测）+ 每个软件栈一组 `(floor, eta)`（一次标定），
  替换掉按最小维度一维分档的系数。
- 地板是 harness 属性：换成 graph/真实训练里的连续 launch，floor 应接近 `launch_stream` 的 2.2us，
  需要用真实训练 trace 再标一次，不能直接搬 533us。
- 剩余误差最大的几条集中在 `lm_head` 大 M（实测只有 ~200 TFLOPS，eta ~0.5）和若干 `N=576` 的 vision 层，
  是 M2 引入 tile/波次量化项（`T_split` 的前身）的第一批靶子。

## 全量明细

## GEMM (fwd + dX + dW per entry)

entries 130, used 109, excluded 21; bound violations: none

eta = T_theory / T_meas: median 0.303, range 0.045 - 0.753

fitted over all entries: per-call floor 533 us, eta 0.718

| predictor | MAPE | median APE | max APE | Spearman |
| --- | ---: | ---: | ---: | ---: |
| hcu_tiered (4-step table + gemm_efficiency, fitted on this data) | 44.4% | 33.9% | 185% | 0.854 |
| kra: T_theory / eta (1 constant, leave-one-out) | 58.4% | 60.4% | 149% | 0.903 |
| kra: floor + T_theory / eta (2 constants, leave-one-out) | 15.0% | 13.2% | 52% | 0.901 |

| bottleneck | n | median eta |
| --- | ---: | ---: |
| compute | 80 | 0.417 |
| memory | 29 | 0.081 |

| part | M | N | K | meas us | meas TFLOPS | T_theory us | bottleneck | eta | kra pred us | tiered us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| vision_merger_fc2 | 2048 | 576 | 1152 | 665.2 | 12.3 | 29.8 | memory | 0.045 | 572.7 | 70.8 |
| vision_deepstack_merger_fc2 | 2048 | 576 | 1152 | 665.2 | 12.3 | 29.8 | memory | 0.045 | 572.7 | 70.8 |
| vision_attn_proj | 1152 | 2304 | 288 | 636.1 | 7.2 | 40.1 | memory | 0.063 | 587.9 | 106.2 |
| vision_attn_proj | 1152 | 2304 | 576 | 686.3 | 13.4 | 44.6 | memory | 0.065 | 593.5 | 79.6 |
| gdn_in_proj_a | 16 | 4096 | 2048 | 679.9 | 1.2 | 44.6 | memory | 0.066 | 593.5 | 18.6 |
| gdn_in_proj_b | 16 | 4096 | 2048 | 667.5 | 1.2 | 44.6 | memory | 0.067 | 593.7 | 18.6 |
| shared_expert_gate | 1 | 4096 | 2048 | 621.2 | 0.1 | 44.1 | memory | 0.071 | 594.0 | 1.2 |
| vision_attn_proj | 1024 | 2304 | 1024 | 719.3 | 20.2 | 51.6 | compute | 0.072 | 603.0 | 125.8 |
| vision_merger_fc2 | 4096 | 576 | 1152 | 758.8 | 21.5 | 55.8 | memory | 0.074 | 608.6 | 141.6 |
| vision_merger_fc1 | 1152 | 576 | 4608 | 622.0 | 29.5 | 46.0 | memory | 0.074 | 596.6 | 159.3 |
| vision_deepstack_merger_fc1 | 1152 | 576 | 4608 | 622.0 | 29.5 | 46.0 | memory | 0.074 | 596.6 | 159.3 |
| shared_expert_gate | 1 | 4096 | 3072 | 849.4 | 0.1 | 62.9 | memory | 0.074 | 618.0 | 1.7 |
| qwen35_k_proj | 256 | 4096 | 2048 | 696.1 | 18.5 | 54.6 | memory | 0.078 | 607.6 | 298.3 |
| qwen35_v_proj | 256 | 4096 | 2048 | 690.6 | 18.7 | 54.6 | memory | 0.079 | 607.7 | 298.3 |
| vision_merger_fc2 | 2048 | 576 | 2304 | 568.7 | 28.7 | 45.2 | compute | 0.080 | 596.9 | 141.6 |
| gdn_in_proj_a | 32 | 4096 | 3072 | 798.4 | 3.0 | 63.9 | memory | 0.080 | 619.7 | 55.9 |
| vision_qkv | 864 | 2304 | 1152 | 579.0 | 23.8 | 46.7 | compute | 0.081 | 598.8 | 119.4 |
| gdn_in_proj_b | 32 | 4096 | 3072 | 791.6 | 3.1 | 63.9 | memory | 0.081 | 619.8 | 55.9 |
| vision_merger_fc2 | 3072 | 576 | 2304 | 802.7 | 30.5 | 67.2 | compute | 0.084 | 624.4 | 212.3 |
| vision_merger_fc2 | 5120 | 576 | 1152 | 749.7 | 27.2 | 69.3 | compute | 0.092 | 627.7 | 176.9 |
| vision_deepstack_merger_fc2 | 5120 | 576 | 1152 | 749.7 | 27.2 | 69.3 | compute | 0.092 | 627.7 | 176.9 |
| qwen35_v_proj | 256 | 4096 | 3072 | 803.1 | 24.1 | 74.6 | memory | 0.093 | 634.8 | 447.4 |
| vision_fc1 | 1076 | 2304 | 1152 | 626.3 | 27.4 | 58.2 | compute | 0.093 | 613.9 | 79.3 |
| vision_fc2 | 1152 | 2304 | 1076 | 625.3 | 27.4 | 59.8 | compute | 0.096 | 616.2 | 79.3 |
| qwen35_k_proj | 256 | 4096 | 3072 | 776.0 | 24.9 | 74.6 | memory | 0.096 | 635.0 | 447.4 |
| gdn_in_proj_a | 8 | 4096 | 4096 | 825.3 | 1.0 | 82.0 | memory | 0.099 | 645.1 | 18.6 |
| gdn_in_proj_b | 8 | 4096 | 4096 | 739.9 | 1.1 | 82.0 | memory | 0.111 | 645.9 | 18.6 |
| shared_expert_fc1 | 512 | 4096 | 2048 | 643.7 | 40.0 | 72.9 | compute | 0.113 | 634.5 | 596.5 |
| vision_merger_fc2 | 2560 | 576 | 4096 | 743.9 | 48.7 | 89.4 | compute | 0.120 | 656.3 | 314.6 |
| vision_deepstack_merger_fc2 | 2560 | 576 | 4096 | 743.9 | 48.7 | 89.4 | compute | 0.120 | 656.3 | 314.6 |
| qwen35_k_proj | 256 | 4096 | 4096 | 752.2 | 34.3 | 94.6 | memory | 0.126 | 663.5 | 596.5 |
| qwen35_v_proj | 256 | 4096 | 4096 | 724.4 | 35.6 | 94.6 | memory | 0.131 | 663.9 | 596.5 |
| gdn_in_proj_a | 12 | 4096 | 5120 | 734.5 | 2.1 | 101.0 | memory | 0.137 | 672.8 | 35.0 |
| vision_fc2 | 1152 | 2304 | 2152 | 689.6 | 49.7 | 95.8 | compute | 0.139 | 666.1 | 158.7 |
| gdn_in_proj_b | 12 | 4096 | 5120 | 722.4 | 2.1 | 101.0 | memory | 0.140 | 673.0 | 35.0 |
| vision_merger_fc1 | 2304 | 576 | 4608 | 591.4 | 62.0 | 89.2 | compute | 0.151 | 658.8 | 318.5 |
| qwen35_v_proj | 256 | 4096 | 5120 | 703.5 | 45.8 | 114.5 | memory | 0.163 | 692.4 | 745.7 |
| qwen35_k_proj | 256 | 4096 | 5120 | 697.7 | 46.2 | 114.5 | memory | 0.164 | 692.5 | 745.7 |
| vision_qkv | 1728 | 2304 | 1152 | 566.1 | 48.6 | 93.5 | compute | 0.165 | 665.8 | 127.4 |
| vision_fc1 | 2152 | 2304 | 1152 | 685.5 | 50.0 | 116.4 | compute | 0.170 | 695.4 | 158.7 |
| shared_expert_fc2 | 2048 | 4096 | 256 | 630.3 | 20.4 | 119.8 | memory | 0.190 | 701.3 | 298.3 |
| vision_fc2 | 1024 | 2304 | 4096 | 714.8 | 81.1 | 143.0 | compute | 0.200 | 732.4 | 503.3 |
| vision_merger_fc1 | 4096 | 576 | 4096 | 702.2 | 82.6 | 143.0 | compute | 0.204 | 732.6 | 503.3 |
| vision_deepstack_merger_fc1 | 4096 | 576 | 4096 | 702.2 | 82.6 | 143.0 | compute | 0.204 | 732.6 | 503.3 |
| qkv_weight | 1280 | 4096 | 2048 | 894.5 | 72.0 | 182.3 | compute | 0.204 | 786.0 | 298.3 |
| mla_kv_up | 4096 | 4096 | 512 | 952.9 | 54.1 | 258.6 | memory | 0.271 | 892.6 | 1193.0 |
| attn_proj | 2048 | 4096 | 1024 | 675.7 | 76.3 | 183.5 | compute | 0.272 | 790.3 | 447.4 |
| shared_expert_fc1 | 1024 | 4096 | 3072 | 727.3 | 106.3 | 200.0 | compute | 0.275 | 812.7 | 671.1 |
| shared_expert_fc2 | 3072 | 4096 | 512 | 705.2 | 54.8 | 194.0 | memory | 0.275 | 804.6 | 894.8 |
| vision_qkv | 3072 | 2304 | 1024 | 559.7 | 77.7 | 154.8 | compute | 0.277 | 753.2 | 377.5 |
| mla_kv_down | 576 | 4096 | 7168 | 844.1 | 120.2 | 234.3 | compute | 0.278 | 859.5 | 880.8 |
| shared_expert_fc1 | 512 | 4096 | 7168 | 732.0 | 123.2 | 208.3 | compute | 0.285 | 824.4 | 2087.8 |
| vision_fc1 | 4096 | 2304 | 1024 | 720.4 | 80.5 | 206.4 | compute | 0.286 | 821.9 | 503.3 |
| gdn_in_proj_qkv | 2560 | 4096 | 5120 | 2561.6 | 125.8 | 770.8 | compute | 0.301 | 1600.1 | 1118.5 |
| mla_q_up | 3072 | 4096 | 1536 | 1177.0 | 98.5 | 356.4 | compute | 0.303 | 1028.1 | 536.9 |
| qkv_weight | 2560 | 4096 | 5120 | 2541.4 | 126.7 | 770.8 | compute | 0.303 | 1600.2 | 1118.5 |
| gdn_in_proj_z | 1024 | 4096 | 4096 | 784.6 | 131.4 | 254.2 | compute | 0.324 | 888.2 | 894.8 |
| mtp_fc | 1024 | 4096 | 4096 | 763.9 | 134.9 | 254.2 | compute | 0.333 | 888.6 | 894.8 |
| lm_head | 37984 | 4096 | 4096 | 28311.8 | 135.1 | 9427.6 | compute | 0.333 | 13507.9 | 13276.4 |
| qwen35_q_proj_gate | 3072 | 4096 | 5120 | 2771.1 | 139.5 | 924.9 | compute | 0.334 | 1813.2 | 1342.2 |
| lm_head | 151936 | 4096 | 2560 | 71305.2 | 134.1 | 25660.2 | compute | 0.360 | 35826.6 | 33190.9 |
| lm_head | 124160 | 4096 | 3072 | 67137.9 | 139.6 | 24251.6 | compute | 0.361 | 33889.9 | 32547.8 |
| mtp_fc | 896 | 4096 | 14336 | 1881.2 | 167.8 | 696.1 | compute | 0.370 | 1498.7 | 2740.3 |
| linear_fc1 | 4608 | 4096 | 7168 | 4801.8 | 169.0 | 1874.6 | compute | 0.390 | 3122.4 | 2818.6 |
| attn_proj | 2048 | 4096 | 2048 | 747.2 | 137.9 | 291.7 | compute | 0.390 | 942.2 | 477.2 |
| gdn_out_proj | 2048 | 4096 | 2048 | 722.4 | 142.7 | 291.7 | compute | 0.404 | 942.8 | 477.2 |
| gdn_in_proj_z | 2048 | 4096 | 2048 | 711.1 | 145.0 | 291.7 | compute | 0.410 | 943.1 | 477.2 |
| gdn_in_proj_qkv | 6144 | 4096 | 3072 | 2884.6 | 160.8 | 1200.1 | compute | 0.416 | 2194.9 | 1610.6 |
| gdn_in_proj_qkv | 2048 | 4096 | 4096 | 1213.3 | 169.9 | 508.3 | compute | 0.419 | 1241.0 | 954.4 |
| qwen35_q_proj_gate | 2048 | 4096 | 4096 | 1203.1 | 171.4 | 508.3 | compute | 0.422 | 1241.1 | 954.4 |
| attn_proj | 4096 | 4096 | 1024 | 867.7 | 118.8 | 366.9 | compute | 0.423 | 1046.3 | 894.8 |
| shared_expert_fc2 | 7168 | 4096 | 256 | 962.7 | 46.8 | 407.5 | memory | 0.423 | 1102.2 | 1043.9 |
| gdn_out_proj | 4096 | 4096 | 1024 | 855.6 | 120.5 | 366.9 | compute | 0.429 | 1046.6 | 894.8 |
| qkv_weight | 6144 | 4096 | 2560 | 2396.2 | 161.3 | 1037.7 | compute | 0.433 | 1971.9 | 1342.2 |
| qwen35_q_proj_gate | 8192 | 4096 | 3072 | 3644.2 | 169.7 | 1600.1 | compute | 0.439 | 2747.2 | 2147.5 |
| linear_fc2 | 4096 | 4096 | 12288 | 6078.3 | 203.5 | 2749.2 | compute | 0.452 | 4330.3 | 4295.0 |
| gdn_in_proj_z | 1536 | 4096 | 5120 | 1016.8 | 190.1 | 462.5 | compute | 0.455 | 1179.1 | 894.8 |
| qkv_weight | 2304 | 4096 | 4096 | 1255.5 | 184.7 | 571.9 | compute | 0.455 | 1330.1 | 805.3 |
| mla_q_down | 1536 | 4096 | 7168 | 1345.3 | 201.1 | 624.9 | compute | 0.464 | 1403.8 | 1252.7 |
| mtp_fc | 1024 | 4096 | 8192 | 995.1 | 207.2 | 470.7 | compute | 0.473 | 1191.4 | 1789.6 |
| gdn_out_proj | 5120 | 4096 | 1536 | 1255.5 | 153.9 | 594.0 | compute | 0.473 | 1361.5 | 894.8 |
| mtp_fc | 1536 | 4096 | 6144 | 1131.2 | 205.0 | 543.7 | compute | 0.481 | 1292.3 | 1073.7 |
| gdn_in_proj_z | 4096 | 4096 | 3072 | 1659.0 | 186.4 | 800.1 | compute | 0.482 | 1646.5 | 1073.7 |
| attn_proj | 5120 | 4096 | 1536 | 1218.9 | 158.6 | 594.0 | compute | 0.487 | 1362.2 | 894.8 |
| lm_head | 151936 | 4096 | 4096 | 76787.4 | 199.2 | 37710.5 | compute | 0.491 | 52480.2 | 53105.5 |
| attn_proj | 4096 | 4096 | 2048 | 1164.5 | 177.0 | 583.5 | compute | 0.501 | 1348.3 | 954.4 |
| attn_proj | 3072 | 4096 | 4096 | 1497.6 | 206.5 | 762.5 | compute | 0.509 | 1596.1 | 1073.7 |
| lm_head | 124160 | 4096 | 2048 | 34149.0 | 183.0 | 17686.7 | compute | 0.518 | 24924.9 | 28931.4 |
| linear_fc1 | 8704 | 4096 | 5120 | 4998.2 | 219.1 | 2620.6 | compute | 0.524 | 4161.8 | 3802.8 |
| linear_fc2 | 4096 | 4096 | 3072 | 1516.1 | 204.0 | 800.1 | compute | 0.528 | 1649.2 | 1073.7 |
| linear_fc1 | 19456 | 4096 | 2560 | 6179.0 | 198.1 | 3285.9 | compute | 0.532 | 5080.7 | 4250.2 |
| lm_head | 62080 | 4096 | 5120 | 34717.9 | 225.0 | 18690.7 | compute | 0.538 | 26325.9 | 27123.2 |
| gdn_out_proj | 3072 | 4096 | 4096 | 1398.4 | 221.1 | 762.5 | compute | 0.545 | 1598.4 | 1073.7 |
| mtp_fc | 1280 | 4096 | 10240 | 1281.3 | 251.4 | 723.8 | compute | 0.565 | 1546.2 | 1491.3 |
| attn_proj | 4096 | 4096 | 4096 | 1786.5 | 230.8 | 1016.6 | compute | 0.569 | 1951.9 | 1431.7 |
| linear_fc2 | 7168 | 4096 | 2304 | 1958.4 | 207.2 | 1115.8 | compute | 0.570 | 2089.3 | 1409.3 |
| attn_proj | 2560 | 4096 | 4096 | 1106.9 | 232.8 | 635.4 | compute | 0.574 | 1424.4 | 894.8 |
| attn_proj | 5120 | 4096 | 2048 | 1263.8 | 203.9 | 729.4 | compute | 0.577 | 1554.8 | 1193.0 |
| linear_fc1 | 12800 | 4096 | 5120 | 6638.6 | 242.6 | 3853.8 | compute | 0.581 | 5875.2 | 5592.4 |
| qkv_weight | 6144 | 4096 | 4096 | 2517.8 | 245.6 | 1524.9 | compute | 0.606 | 2659.6 | 2147.5 |
| linear_fc1 | 6144 | 4096 | 4096 | 2517.1 | 245.7 | 1524.9 | compute | 0.606 | 2659.7 | 2147.5 |
| linear_fc1 | 24576 | 4096 | 4096 | 9922.8 | 249.3 | 6099.8 | compute | 0.615 | 8992.4 | 8589.9 |
| linear_fc2 | 5120 | 4096 | 4352 | 2105.3 | 260.1 | 1338.5 | compute | 0.636 | 2405.1 | 1901.4 |
| lm_head | 16160 | 4096 | 7168 | 10178.4 | 279.7 | 6574.3 | compute | 0.646 | 9664.8 | 9884.6 |
| lm_head | 37984 | 4096 | 2048 | 8133.8 | 235.0 | 5410.9 | compute | 0.665 | 8061.7 | 8850.9 |
| linear_fc2 | 2560 | 4096 | 9728 | 2051.8 | 298.3 | 1379.9 | compute | 0.673 | 2467.7 | 2125.1 |
| linear_fc2 | 5120 | 4096 | 6400 | 2767.9 | 290.9 | 1879.9 | compute | 0.679 | 3164.0 | 2796.2 |
| lm_head | 62080 | 4096 | 4096 | 21061.2 | 296.7 | 15408.3 | compute | 0.732 | 22026.7 | 21698.5 |
| lm_head | 37984 | 4096 | 5120 | 15194.2 | 314.6 | 11436.0 | compute | 0.753 | 16518.5 | 16595.5 |

excluded:

- `part=vision_patch_embed,m=1024,n=2304,k=1536,dtype=bf16`: no tflops recorded: not a plain matmul benchmark
- `part=vision_patch_embed,m=288,n=2304,k=1536,dtype=bf16`: no tflops recorded: not a plain matmul benchmark
- `part=topk_router,m=128,n=1024,k=2048,dtype=bf16`: timed workload is 4.00x the key's 6mnk flops (grouped/fused or a different shape was timed)
- `part=moe_linear_fc1,m=384,n=2048,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=16`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=moe_linear_fc2,m=2048,n=192,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=16`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=moe_linear_fc1,m=1536,n=2048,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=16`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=moe_linear_fc2,m=2048,n=768,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=16`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=topk_router,m=128,n=1024,k=4096,dtype=bf16`: timed workload is 4.00x the key's 6mnk flops (grouped/fused or a different shape was timed)
- `part=moe_linear_fc1,m=3072,n=4096,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=16`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=moe_linear_fc2,m=4096,n=1536,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=16`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=attn_proj,m=7168,n=4096,k=2048,dtype=bf16`: timed workload is 0.44x the key's 6mnk flops (grouped/fused or a different shape was timed)
- `part=topk_router,m=256,n=512,k=7168,dtype=bf16`: timed workload is 8.00x the key's 6mnk flops (grouped/fused or a different shape was timed)
- `part=moe_linear_fc1,m=4096,n=7168,k=128.0,dtype=bf16,grouped_gemm=True,num_gemms=32`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=moe_linear_fc2,m=7168,n=2048,k=128.0,dtype=bf16,grouped_gemm=True,num_gemms=32`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=vision_patch_embed,m=576,n=2304,k=1536,dtype=bf16`: no tflops recorded: not a plain matmul benchmark
- `part=topk_router,m=256,n=2048,k=2048,dtype=bf16`: timed workload is 2.00x the key's 6mnk flops (grouped/fused or a different shape was timed)
- `part=moe_linear_fc1,m=1024,n=2048,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=32`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=moe_linear_fc2,m=2048,n=512,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=32`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=topk_router,m=256,n=2048,k=3072,dtype=bf16`: timed workload is 2.00x the key's 6mnk flops (grouped/fused or a different shape was timed)
- `part=moe_linear_fc1,m=2048,n=3072,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=32`: grouped GEMM: per-expert shape is not the key's m,n,k
- `part=moe_linear_fc2,m=3072,n=1024,k=512.0,dtype=bf16,grouped_gemm=True,num_gemms=32`: grouped GEMM: per-expert shape is not the key's m,n,k

## non_gemm (lower bound only)

entries 120, used 111, excluded 9; bound violations: none

eta median 0.064, range 0.0063 - 0.262: torch-eager composites run far from the HBM bound, which is the headroom a fused kernel would recover.

| part | n | median eta |
| --- | ---: | ---: |
| gdn_gate_preprocess | 4 | 0.0094 |
| moe_swiglu_activation | 5 | 0.0100 |
| vision_rope | 3 | 0.0133 |
| vision_deepstack_add | 3 | 0.0210 |
| vision_residual_add | 3 | 0.0226 |
| vision_deepstack_merger_norm | 2 | 0.0257 |
| vision_norm1 | 3 | 0.0283 |
| vision_norm2 | 3 | 0.0283 |
| vision_merger_norm | 3 | 0.0283 |
| shared_swiglu_activation | 2 | 0.0296 |
| gdn_qk_l2norm | 2 | 0.0332 |
| qk_norm | 7 | 0.0340 |
| gdn_gated_rmsnorm | 4 | 0.0355 |
| vision_position_add | 3 | 0.0371 |
| vision_merger_gelu | 3 | 0.0387 |
| vision_deepstack_merger_gelu | 2 | 0.0415 |
| rope | 8 | 0.0600 |
| residual_dropout_add | 7 | 0.0679 |
| mtp_pre_fc_norm_embedding | 5 | 0.0868 |
| mtp_pre_fc_norm_hidden | 5 | 0.0868 |
| mtp_final_norm | 5 | 0.0868 |
| final_rmsnorm | 6 | 0.0885 |
| attention_rmsnorm | 1 | 0.0896 |
| mlp_rmsnorm | 4 | 0.0975 |
| swiglu_activation | 6 | 0.1026 |
| attention_output_gate | 4 | 0.1197 |
| vision_gelu | 3 | 0.1294 |
| cross_entropy | 5 | 0.1586 |
