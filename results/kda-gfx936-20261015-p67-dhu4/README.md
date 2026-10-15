# P-67：TP8 的 dhu 改走 4-wave 的 v8b4，并加上 SG 与 LDS 批量读（2026-10-15，不合入）

## 新的测量节点 bw46
- 容器 `tanbo_kda_b46`，镜像是 harbor 的 `megatron-0.18.2-ubuntu22.04-dtk2604-py3.10-20260908-0350`，与 bw47 的 `tanbo_mega_k3` 相同。挂载 `/public/home/tanbo`，`/opt/kda_env` 解自 `xplat/kda_env_b46.tgz`（g2_r5、takeover、fla、ft_site_r25）。
- 设备锁目录 `codex_validation/locks/bw46/`，A/B 脚本 `xplat/abk46.sh`。
- 首次使用时 8 张 HCU 全部空闲。在 bw46 上跑 P-67 的逐位门禁全部 PASS，这说明它的环境与 bw7 一致。

## 依据
TP8 16K/12 的关键路径：
- 生产版 8-wave 的 v8b（`../kda-gfx936-20261015-p66-dhu-cp/cpdhu_p65_d1.txt`）：每步 3753 周期，46% 是 VMEM 发射受阻。每个 CU 每步约 112 条 VMEM 指令，其中同一 r1 的两个 wave 会重复加载 KgA。
- 4-wave 的 v8b4（`cpdhu4_p65_d1.txt`，用 `KDA_G2_R1A_DHU=1` 强制）：每步 3718 周期，属于发射受限。主要是 VALU 1173（64 位地址约 300、bf16 舍入约 330）和 LDS 读延迟 475。这和 P-58、P-65 之前前向的情况很像。

## P-67（`examples/kda/p67_dhu4_tp8.py`）
- **改法**：v8b4 加两个模板参数：SG（wave 号走 readfirstlane，即 P-58 的做法）和 KB（每个 GEMM 的 LDS 读一次发完，即 P-65 的做法）。当 4·nseg·H ≤ 80 时，dhu 改用 v8b4<0, lin, SG, KB>。
- **逐位门禁**：10/10，FI=3 PASS（bw46）。
- **A/B**（bw46，HCU 1 / 2）：

| 形态 | 生产版 v8b | P-67 v8b4 | 比值 |
|---|---:|---:|---:|
| 16K/12 | 657.0 µs | 695.1 µs | 0.945 |
| 64K/12 | 2124.6 µs | 2485.4 µs | 0.855 |

- **SQTT**（`cpdhu4_p67_d1.txt`）：SG 把地址运算挪到了 SALU 上，SALU 发射 451 + 等待 242 周期/步。VALU 仍有 905，每个 wave 的工作量是 8-wave 版的两倍。**否决**。

## 结论与下一步
- 4-wave 去重省下的 VMEM 指令（每 CU 每步约 112 降到 96，少 14%）抵不过单 wave 指令变多的代价。
- 在 TP8 上，dhu 的瓶颈是每个 CU 的 TA/VMEM 发射吞吐。
- **结构性的事实**：在 TP8、H = 12 时，三个递推的网格都是 4 个 V-slab × 12 个头 = 48 个 CTA，**只用了 80 个 CU 中的 48 个（60%）**。
- 下一步的候选：
  1. 在 v8b 本身上调整 VMEM 读写顺序：位置 190 的那条写在发射前等了 601 周期（参照 R52 / P-64B）；
  2. 让 TP8 用满 80 个 CU，例如换一种 V 切分或别的工作划分。
