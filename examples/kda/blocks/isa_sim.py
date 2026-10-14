"""ISA-level in-order issue simulator for one CTA on one gfx936 CU (building-block library P0-1).

A loop body (the 'Loop Header' block of a kernel's .s) is executed for N iterations by the CTA's W waves:
  * a wave issues its instructions in order; wave w runs on SIMD w % 4 and shares that SIMD's vector pipe;
  * VALU / MMAC / transcendental results interlock on VGPRs (ready = issue + latency); memory results are only
    ordered by s_waitcnt (vmcnt / lgkmcnt, in-order completion) as on the hardware;
  * LDS instructions occupy the CU's LDS pipe, VMEM instructions the CU's TA (per-instruction occupancy);
  * s_barrier releases when all W waves have arrived (+ barrier cost).
Every cost / latency comes from the building-block library measurements (results/kda-blocks-20261014), nothing
is fitted to the kernels being predicted.  The steady-state time per iteration is reported.
    python3 isa_sim.py kernel.s --waves 4 [--iters 16] [--vmem-lat 700] [--trace]"""
import argparse
import heapq
import re

# ---------------------------------------------------------------------------------- costs (library, 1.5 GHz)
COST = {
    "valu": (4.0, 8.9),        # (issue cycles on the SIMD, result latency): v_fma 4.85 thr @4 waves, 8.9 dependent
    "valu64": (4.0, 7.0),      # v_mov_b64 / v_lshlrev_b64: full rate (4.6 thr, 6.8-7.0 dep)
    "carry": (8.3, 9.4),       # v_add_co / v_addc_co pairs through VCC: 16.7 per pair
    "pk": (6.0, 11.3),         # v_pk_fma_f32 6.06 thr, 11.3 dep
    "trans": (16.0, 20.0),     # v_exp / v_rcp / v_log 16.6 thr, 18-22 dep
    "mmac16": (8.0, 30.0),     # bf16 16x16x16: 9.3 cyc/SIMD saturated; result -> VALU / srcA ~30 (mmac_dep_bench:
                               # 38.5 per mmac->valu->srcA round incl. two VALU); srcC accumulate forwarding: MMAC_FWD
    "mmac8f": (32.0, 60.0),    # f32 16x16x8: ~34 cyc/SIMD saturated (dependent latency not yet re-measured)
    "salu": (1.0, 2.0),
}
LDS_OCC = {   # cycles of the CU's LDS pipe per wave instruction (conflict-free, lds_bench)
    "ds_read_b32": 3.1, "ds_read_b64": 5.1, "ds_read_b128": 8.1, "ds_read2_b32": 8.1, "ds_read2_b64": 9.9, "ds_read2st64_b64": 9.9,
    "ds_read_u16": 4.7, "ds_read_u8": 4.7, "ds_read_u16_d16": 4.7, "ds_read_u16_d16_hi": 4.7,
    "ds_write_b16": 4.7, "ds_write_b16_d16_hi": 4.7, "ds_write_b8": 4.7, "ds_write_b32": 4.7, "ds_write_b64": 8.0,
    "ds_write_b128": 10.3, "ds_write2_b32": 8.0, "ds_write2_b64": 16.0, "ds_bpermute_b32": 6.4,
    "ds_swizzle_b32": 3.1,
}
MMAC_FWD = {"mmac16": 10.5, "mmac8f": 34.0}   # dependent srcC accumulate chain, straight-line (mmac_dep_bench)
LDS_LAT = 80.0          # issue -> data (dependent ds_read_b32 chain 88.6 incl. its own occupancy)
TA_OCC = {"ld": {"dword": 12.0, "dwordx2": 21.0, "dwordx4": 22.0, "ushort": 13.0, "short": 13.0},
          "st": {"dword": 15.0, "dwordx2": 29.0, "dwordx4": 59.0, "short": 11.0}}   # L2-resident contiguous rows
BAR = {4: 57.0, 8: 65.0, 16: 94.0}
BRANCH = 8.0            # taken backward branch (assumption: pipeline refill)
ISSUE = 4.0             # every instruction occupies the wave's issue slot for 4 cycles (SQTT clks = 4)

VREG = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")


def vregs(op):
    out = set()
    for a, b, c in VREG.findall(op):
        if c:
            out.add(int(c))
        else:
            out.update(range(int(a), int(b) + 1))
    return out


def classify(opc):
    if opc.startswith("v_mmac"):
        return "mmac16" if "bf16" in opc or "f16" in opc else "mmac8f"
    if opc.startswith(("v_exp", "v_log", "v_rcp", "v_rsq", "v_sqrt", "v_sin", "v_cos")):
        return "trans"
    if opc.startswith("v_pk_"):
        return "pk"
    if opc.startswith(("v_mov_b64", "v_lshlrev_b64", "v_lshrrev_b64", "v_lshl_add_u64", "v_add_u64")):
        return "valu64"
    if opc.startswith(("v_add_co_u32", "v_addc_co_u32", "v_sub_co_u32", "v_subb_co_u32", "v_subrev_co_u32")):
        return "carry"
    if opc.startswith("v_"):
        return "valu"
    if opc.startswith("ds_"):
        return "lds"
    if opc.startswith(("global_load", "buffer_load", "flat_load")):
        return "vld"
    if opc.startswith(("global_store", "buffer_store", "flat_store", "global_atomic", "buffer_atomic")):
        return "vst"
    if opc.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if opc == "s_waitcnt":
        return "wait"
    if opc == "s_barrier":
        return "bar"
    if opc == "s_nop":
        return "nop"
    if opc.startswith(("s_cbranch", "s_branch")):
        return "br"
    if opc.startswith("s_"):
        return "salu"
    return None


def parse_loop(path, which=0):
    lines = open(path).read().splitlines()
    heads = [i for i, l in enumerate(lines) if "Loop Header" in l]
    h = heads[which]
    label = lines[h].split(":")[0]
    body = []
    for l in lines[h + 1:]:
        s = l.split(";")[0].strip()
        if not s or s.startswith("."):
            if s.startswith(".LBB") and body:
                pass
            continue
        parts = s.split(None, 1)
        opc = parts[0]
        ops = parts[1] if len(parts) > 1 else ""
        cls = classify(opc)
        if cls is None:
            continue
        body.append((opc, ops, cls))
        if cls == "br" and label in ops:
            break
    return body


def decode(opc, ops, cls):
    """-> (dst VGPRs, src VGPRs, extra) for one instruction"""
    fields = [f.strip() for f in ops.split(",")] if ops else []
    dst, src = set(), set()
    if cls in ("valu", "valu64", "carry", "pk", "trans", "mmac16", "mmac8f", "vld"):
        if fields and VREG.match(fields[0]):
            dst = vregs(fields[0])
            src = set().union(*[vregs(f) for f in fields[1:]]) if len(fields) > 1 else set()
        else:
            src = set().union(*[vregs(f) for f in fields]) if fields else set()
    elif cls == "lds":
        if opc.startswith(("ds_read", "ds_bpermute", "ds_swizzle")) and fields:
            dst = vregs(fields[0])
            src = set().union(*[vregs(f) for f in fields[1:]]) if len(fields) > 1 else set()
        else:
            src = set().union(*[vregs(f) for f in fields]) if fields else set()
    elif cls == "vst":
        src = set().union(*[vregs(f) for f in fields]) if fields else set()
    extra = None
    if cls in ("mmac16", "mmac8f") and len(fields) >= 4:
        extra = vregs(fields[3])                       # srcC (accumulator input)
        src = vregs(fields[1]) | vregs(fields[2])
    if cls == "wait":
        vm = re.search(r"vmcnt\((\d+)\)", ops)
        lg = re.search(r"lgkmcnt\((\d+)\)", ops)
        extra = (int(vm.group(1)) if vm else None, int(lg.group(1)) if lg else None)
    if cls == "nop":
        extra = int(ops or 0)
    return dst, src, extra


def simulate(body, W=4, iters=16, vmem_lat=900.0, st_lat=430.0, trace=False):
    prog = [(o, a, c) + decode(o, a, c) for o, a, c in body]
    n = len(prog)
    simd_free = [0.0] * 4
    lds_free = ta_free = 0.0
    vready = [dict() for _ in range(W)]
    vfwd = [dict() for _ in range(W)]          # MMAC results reaching a following MMAC's srcC
    t = [0.0] * W
    pc = [0] * W
    it = [0] * W
    vm_q = [[] for _ in range(W)]     # completion times of outstanding VMEM ops (in order)
    lg_q = [[] for _ in range(W)]
    bar_arr, bar_wait = {}, set()
    iter_end = [[] for _ in range(W)]
    stall = {}
    pos_t = {}
    heap = [(0.0, 0, w) for w in range(W)]
    heapq.heapify(heap)
    while heap:
        tw, _, w = heapq.heappop(heap)
        if it[w] >= iters:
            continue
        opc, ops, cls, dst, src, extra = prog[pc[w]]
        dep = max((vready[w].get(r, 0.0) for r in src), default=0.0)
        if cls in ("mmac16", "mmac8f") and extra:
            dep = max(dep, max((vfwd[w].get(r, vready[w].get(r, 0.0)) for r in extra), default=0.0))
        start = max(t[w], dep)
        if w == 0 and it[w] >= iters // 2 and dep > t[w]:
            stall["dep:" + cls] = stall.get("dep:" + cls, 0.0) + dep - t[w]
        if cls == "wait":
            vm, lg = extra
            if vm is not None and len(vm_q[w]) > vm:
                start = max(start, vm_q[w][len(vm_q[w]) - vm - 1])
            if lg is not None and len(lg_q[w]) > lg:
                start = max(start, lg_q[w][len(lg_q[w]) - lg - 1])
            if w == 0 and it[w] >= iters // 2 and start > t[w]:
                stall["wait:" + ops.split()[0][:6]] = stall.get("wait:" + ops.split()[0][:6], 0.0) + start - t[w]
            vm_q[w] = [c for c in vm_q[w] if c > start]
            lg_q[w] = [c for c in lg_q[w] if c > start]
            t[w] = start + ISSUE
        elif cls == "bar":
            key = (it[w], pc[w])
            bar_arr.setdefault(key, {})[w] = start
            if len(bar_arr[key]) < W:
                bar_wait.add(w)
                t[w] = start
                pc[w] = pc[w]          # stay; released by the last arrival
                continue
            rel = max(bar_arr[key].values()) + BAR.get(W, 60.0)
            if it[0] >= iters // 2 and 0 in bar_arr[key]:
                stall["barrier"] = stall.get("barrier", 0.0) + rel - bar_arr[key][0]
            for v in list(bar_arr[key]):
                t[v] = rel
                if v != w:
                    bar_wait.discard(v)
                    pc[v] += 1
                    if pc[v] == n:
                        pc[v] = 0; it[v] += 1; iter_end[v].append(t[v])
                    heapq.heappush(heap, (t[v], (v - it[v]) % W, v))
            del bar_arr[key]
        elif cls in COST and cls != "salu":
            issue, lat = COST[cls]
            s = w % 4
            if w == 0 and it[w] >= iters // 2 and simd_free[s] > start:
                stall["simd_busy"] = stall.get("simd_busy", 0.0) + simd_free[s] - start
            start = max(start, simd_free[s])
            simd_free[s] = start + issue
            for r in dst:
                vready[w][r] = start + lat
                if cls in MMAC_FWD:
                    vfwd[w][r] = start + MMAC_FWD[cls]
                else:
                    vfwd[w].pop(r, None)
            t[w] = start + issue
        elif cls == "salu":
            t[w] = start + ISSUE
        elif cls == "nop":
            t[w] = start + ISSUE + extra
        elif cls == "br":
            t[w] = start + BRANCH
        elif cls == "lds":
            occ = LDS_OCC.get(opc, 6.0)
            s0 = max(start, lds_free)
            if w == 0 and it[w] >= iters // 2 and s0 > start:
                stall["issue:lds"] = stall.get("issue:lds", 0.0) + s0 - start
            lds_free = s0 + occ
            done = s0 + occ + LDS_LAT
            lg_q[w].append(max(done, lg_q[w][-1] if lg_q[w] else 0.0))
            t[w] = s0 + ISSUE
        elif cls == "smem":
            lg_q[w].append(max(start + 200.0, lg_q[w][-1] if lg_q[w] else 0.0))
            t[w] = start + ISSUE
        elif cls in ("vld", "vst"):
            kind = "ld" if cls == "vld" else "st"
            width = opc.split("_", 2)[-1].replace("_d16_hi", "").replace("_d16", "")
            occ = TA_OCC[kind].get(width, 22.0 if kind == "ld" else 59.0)
            s0 = max(start, ta_free)
            if w == 0 and it[w] >= iters // 2 and s0 > start:
                stall["issue:vmem"] = stall.get("issue:vmem", 0.0) + s0 - start
            ta_free = s0 + occ
            done = s0 + occ + (vmem_lat if cls == "vld" else st_lat)
            vm_q[w].append(max(done, vm_q[w][-1] if vm_q[w] else 0.0))
            t[w] = s0 + ISSUE
        if w == 0 and it[w] >= iters // 2:
            pos_t.setdefault(pc[w], []).append(t[w])
        if trace and w == 0 and it[w] == iters - 2:
            print(f"{t[w]:9.0f} w{w} {opc} {ops[:60]}")
        pc[w] += 1
        if pc[w] == n:
            pc[w] = 0
            it[w] += 1
            iter_end[w].append(t[w])
        heapq.heappush(heap, (t[w], (w - it[w]) % W, w))
    k0 = iters // 2
    per = [(e[-1] - e[k0 - 1]) / (iters - k0) for e in iter_end if len(e) >= iters]
    simulate.stall = {k: v / (iters - k0) for k, v in sorted(stall.items(), key=lambda kv: -kv[1])}
    simulate.pos_t = pos_t
    return max(per), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("isa")
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--vmem-lat", type=float, default=900.0, help="load issue -> vmcnt (vmem_compl_bench 930-1030)")
    ap.add_argument("--st-lat", type=float, default=430.0, help="store issue -> vmcnt (vmem_compl_bench 427-430)")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    body = parse_loop(a.isa)
    cyc, per = simulate(body, a.waves, a.iters, a.vmem_lat, a.st_lat, a.trace)
    mix = {}
    for _, _, c in body:
        mix[c] = mix.get(c, 0) + 1
    print(f"{a.isa}: {len(body)} instructions/iteration {mix}")
    print(f"  steady state: {cyc:.0f} cycles = {cyc / 1500:.3f} us per iteration (waves: {[round(p) for p in per]})")
    print("  wave-0 stalls per iteration (cycles):", {k: round(v) for k, v in simulate.stall.items()})


if __name__ == "__main__":
    main()
