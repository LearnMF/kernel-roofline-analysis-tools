"""通用 ISA 资源检查：读设备汇编（hipcc --cuda-device-only -S 的输出），列出每个 kernel 的 VGPR、SGPR、
LDS、scratch 与指令条数，并可统计指定指令的出现次数；scratch 非 0 时以退出码 1 失败（用于资源预检）。

    /opt/dtk/bin/hipcc -O3 --offload-arch=gfx936 --cuda-device-only -S your_kernels.hip -o k.s
    python -m kra.tools.isa k.s [--kernel 子串] [--count v_mmac --count v_pk_fma_f32 --count global_load_dwordx4]
"""
import argparse
import collections
import json
import re
import sys

META = {"vgpr": r"\.vgpr_count:\s+(\d+)", "sgpr": r"\.sgpr_count:\s+(\d+)",
        "lds": r"\.group_segment_fixed_size:\s+(\d+)", "scratch": r"\.private_segment_fixed_size:\s+(\d+)"}


def parse(text):
    funcs, cur, device = {}, None, True
    for line in text.split("\n"):
        if "__CLANG_OFFLOAD_BUNDLE____START__" in line:      # 混合 bundle 只统计设备端（amdgcn）部分
            device = "amdgcn" in line; cur = None; continue
        if not device:
            continue
        m = re.match(r"^([A-Za-z_.$][\w.$]*):\s*(;.*)?$", line)
        if m and not m.group(1).startswith((".L", ".")):
            cur = m.group(1); funcs.setdefault(cur, collections.Counter()); continue
        if cur and re.match(r"^\s*\.Lfunc_end", line):
            cur = None; continue
        if cur:
            mm = re.match(r"^\s+([sv]_\w+|ds_\w+|global_\w+|buffer_\w+|flat_\w+|scratch_\w+|matrix_\w+)", line)
            if mm:
                funcs[cur][mm.group(1)] += 1
    metas = {}
    for blk in re.split(r"\n  - ", text):
        nm = re.search(r"\n\s+\.name:\s+(\S+)", "\n" + blk)
        if nm and re.search(r"\.vgpr_count:", blk):
            metas[nm.group(1)] = {k: (int(m.group(1)) if (m := re.search(p, blk)) else None) for k, p in META.items()}
    out = []
    for name, cnt in funcs.items():
        if not cnt:
            continue
        meta = metas.get(name, {k: None for k in META})
        out.append({"kernel": name, **meta, "insts": sum(cnt.values()), "_ops": cnt})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m kra.tools.isa", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("asm"); p.add_argument("--kernel", default="")
    p.add_argument("--count", action="append", default=[])
    a = p.parse_args(argv)
    ok = True
    for k in parse(open(a.asm).read()):
        if a.kernel not in k["kernel"]:
            continue
        row = {x: k[x] for x in ("kernel", "vgpr", "sgpr", "lds", "scratch", "insts")}
        for pat in a.count:
            row[pat] = sum(v for op, v in k["_ops"].items() if op.startswith(pat))
        ok &= not k["scratch"]
        print(json.dumps(row))
    print(f"# scratch check: {'PASS' if ok else 'FAIL (scratch > 0)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
