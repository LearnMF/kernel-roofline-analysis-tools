"""Emit results/kda-ceiling-cards-20261013/cards.md from cards.json.

Layout per the task: summary table first, then per shape three tables
(utilization / floor decomposition / work gaps), verification appendix.
Unreliable cells render as "-- r:reason" or gray numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # examples/kda (kda_bound)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root (kra)
from kda_bound import bounds as algo_bounds                     # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "kda-ceiling-cards-20261013"
CARDS = json.load(open(RES / "cards.json"))

MEASURED = {(16384, 48): 26.43, (16384, 24): 13.09, (16384, 12): 6.75,
            (32768, 48): 50.13, (32768, 24): 25.63, (32768, 12): 12.94,
            (65536, 48): 98.37, (65536, 24): 48.36, (65536, 12): 24.38}


def g(x, nd=2, dash="—"):
    return f"{x:.{nd}f}" if x is not None else dash


def md() -> str:
    L = []
    L += ["# KDA G2 (P-52) 逐 kernel 见顶体检卡 — gfx936/bw7", "",
          "冻结树 `/opt/kda_env/g2_card_p52`（指纹 `35b7bd3a8862`）。"
          "口径：每 kernel 5 个抖动进程的 trace 中位数；"
          "带宽/算力取 kra 标定可达值；FLOPs = SQ_INSTS_MMOP × ISA 标注的单条 FLOPs "
          "（纯 bf16 = 8192，纯 fp32x8 = 4096，混合按 ISA 静态占比加权，卡片注明）。",
          "",
          "> **口径提醒**：表2 的“下限”= **现有指令流全速发射所需时间**"
          "（实测 TA/VALU/LDS 活跃周期 ÷ 活跃 CU ÷ 时钟），"
          "它说明“这条指令流已经多接近满速”，**不是算子真下限**；"
          "真正的余量看表3 的指令/字节放大倍数与总表的算子第一性下限。"
          "LDS 冲突计数偏高不等于病灶——消融实验（保工作量不变、消去全部冲突）"
          "在 dav/wu/递推上只快 0.9–2.4% / 0.5–2.5% / 0（R42），大部分冲突被其他瓶颈掩盖。", "",
          "## 总表（每形态合计）", "",
          "| 形态 | Σ kernel 时间 (ms) | Σ 现有指令流下限 (ms) | 算子第一性下限 (ms) | 上限括号含链 (ms) | 实测/Σ下限 | Σ下限/算子下限 | 主要限制（按时间占比） |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for s, c in CARDS.items():
        T, H = c["T"], c["H"]
        tot = sum(v["t_us"] for v in c["cards"].values()) / 1e3
        fl = sum(v["floor_us"] for v in c["cards"].values()) / 1e3
        b = algo_bounds("BW1000", T, H)
        share = {}
        for k, v in c["cards"].items():
            share[v["binding"]] = share.get(v["binding"], 0) + v["t_us"] / 1e3
        bind = ", ".join(f"{k} {v / tot * 100:.0f}%" for k, v in
                         sorted(share.items(), key=lambda x: -x[1]) if v / tot > 0.05)
        L.append(f"| {T // 1024}K/H{H} | {tot:.2f} | {fl:.2f} | {b['bound'] * 1e3:.2f} "
                 f"({b['binding']}) | {b['bracket'] * 1e3:.2f} | {tot / fl:.2f}x | "
                 f"{fl / (b['bound'] * 1e3):.2f}x | {bind} |")
    L += ["", "图A：每形态一张分层 roofline（`figA_<T>_<H>.png`）；图B/C/D 见 `figB.png`/`figC.png`/`figD.png`。",
          "无 MMAC 的搬运型 kernel 不上 roofline，在图A脚注列出。", ""]
    for s, c in sorted(CARDS.items(), key=lambda x: (x[1]["T"], x[1]["H"])):
        T, H, NT = c["T"], c["H"], c["NT"]
        L += [f"## {T // 1024}K / H{H}（NT={NT}）", "",
              "### 表1：实测与利用率",
              "| kernel | µs | CTAs | HBM B/s | L2 B/s | LDS B/s | TA（全部CU） | VALU槽 | LDS idx | MMAC槽 | 冲突/条 | 等待/条 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
            un = " *" if v["t_us"] < 10 else ""
            L.append(f"| {k}{un} | {v['t_us']:.0f} | {v['ctas']} | {v['hbm_gbs']:.0f} GB/s "
                     f"| {v['l2_gbs'] / 1e3:.2f} TB/s | {v['lds_gbs'] / 1e3:.2f} TB/s "
                     f"| {v['u_ta_act']:.2f} | {v['u_valu_act']:.2f} | {v['u_lds_act']:.2f} "
                     f"| {v['u_mmop_act']:.2f} | {v['conf_i']:.2f} | {v['wait_i']:.2f} |")
        L += ["", "*<10µs 的 kernel 计数不可靠。*", "",
              "### 表2：现有指令流下限分解（重要：这是当前指令流全速发射所需的时间，不是算子真下限）",
              "| kernel | µs | 带宽项 | 算力项 | 发射项（TA） | 发射项（VALU） | 发射项（LDS） | 链项 | **下限** | 绑定项 | 实测/下限 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
        for k, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
            L.append(f"| {k} | {v['t_us']:.0f} | {v['f_mem']:.0f} | {v['f_cmp']:.0f} "
                     f"| {v['f_ta']:.0f} | {v['f_val']:.0f} | {v['f_lds']:.0f} | {v['f_chain']:.0f} "
                     f"| **{v['floor_us']:.0f}** | {v['binding']} | {g(v['gap'])}x |")
        L += ["", "### 表3：工作量缺口（实际/必需）",
              "| kernel | 实测 HBM (MB) | 接口必需 (MB) | 字节倍率 | VMEM 指令 | 指令/必需下界 | MMAC FLOPs | MMAC 类型 |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
        for k, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
            req = v["req_bytes"]
            vreq = req / 1024 if req else None          # best-pattern 1 KB per instruction
            wg = v["w_gap"]
            vi = v["vmem"] / vreq if vreq else None
            L.append(f"| {k} | {(v['hbm_rd'] + v['hbm_wr']) / 1e6:.0f} | "
                     f"{f'{req / 1e6:.0f}' if req else '—'} | "
                     f"{f'{wg:.2f}x' if wg else '—'} | {v['vmem'] / 1e6:.2f}M "
                     f"| {f'{vi:.1f}x' if vi else '—'} "
                     f"| {v['flops'] / 1e9:.1f}G | {v['mmac_kind']} |")
        L += ["", "### 表4：判定（任务 §5）",
              "| kernel | 主约束 | 利用率 | 次约束 | 利用率 | 工作量比 | 是否见顶 | 缺口类型 |",
              "|---|---|---:|---|---:|---:|---|---|"]
        for k, v in sorted(c["cards"].items(), key=lambda x: -x[1]["t_us"]):
            wr = v.get("work_ratio")
            L.append(f"| {k} | {v.get('primary', '—')} | {v.get('u_primary', 0):.2f} "
                     f"| {v.get('secondary', '—')} | {v.get('u_secondary', 0):.2f} "
                     f"| {f'{wr:.2f}x' if wr else '—'} | {'是' if v.get('at_ceiling') else '否'} "
                     f"| {v.get('gap_types', '—')} |")
        L += [""]
    L += ["## 附：launcher 接口账目（optrace）与抽查核对", ""]
    for s, c in sorted(CARDS.items(), key=lambda x: (x[1]["T"], x[1]["H"])):
        if not c["launchers"]:
            continue
        L.append(f"**{c['T'] // 1024}K/H{c['H']}**: " + "；".join(
            f"{ln} {v['in_bytes'] / 1e6:.0f}+{v['out_bytes'] / 1e6:.0f}MB"
            for ln, v in c["launchers"].items()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    (RES / "cards.md").write_text(md())
    print("wrote", RES / "cards.md")
