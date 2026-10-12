"""F9 feasibility: discrete-event simulation of a persistent, statically scheduled KDA backward (and
forward) on 80 CUs, at chunk granularity, from MEASURED kernel times (production, PMC run wl13).

Tasks
  parallel, per (head, chunk):  l2n, gate, wu, dav, s4a, s4b, tail       CU-time = t_kernel * 80 / (H*NT)
  chains, per (head, V-slab):   fwd_h_bn steps c = 0..NT-1 (forward), dhu steps c = NT-1..0 (reverse)
                                step time alone = t_kernel / NT (TP8: 1 chain CTA per CU, latency-bound)
Dependencies (chunk c of head h)
  l2n(c), gate(c) -> wu(c) -> fwd_h_bn step c (all slabs) -> dav(c)
  dav(c), wu(c), fwd_h_bn done -> dhu step c (reverse order) -> s4a(c) -> s4b(c) -> tail(c)
Contention (P-41, measured on the forward: recurrence alone 681 us vs 1413 us co-resident with prep,
prep at ~0.71 of its alone speed): while a chain CU also runs parallel work, the chain step runs at
S_CHAIN = 0.48 and the parallel task at S_PAR = 0.71 of their alone speeds.
Policies
  serial      today's kernel-by-kernel execution (sum of kernel times)
  shared      chains pinned to CUs, every CU (chain CUs included) runs parallel work -> contention
  dedicated   chain CUs run only their chain
  idle-fill   chain CUs take parallel work only while their chain waits on a dependency
TP2 (H=48) is not simulated step by step: its 192 chain CTAs exceed the 80 CUs and the recurrence
kernels are throughput (TA) bound, so a persistent schedule can only remove inter-kernel tails.
    python3 f9_sched_sim.py"""
import heapq

NCU = 80
S_CHAIN, S_PAR = 0.48, 0.71
# measured production kernel times (us), 16K
KT = {
    12: dict(l2n=156.0, gate=175.0, wu=645.8, fwd_h_bn=537.4, dav=278.4, dhu=673.3, s4a=595.2,
             s4b=1324.6, tail=708.6, fwd_prep=968.3, fwd_h_o=716.6),
}


def simulate_bwd(H, NT, policy, slabs=4):
    k = KT[H]
    nchunk = H * NT
    cu_t = {n: k[n] * NCU / nchunk for n in ("l2n", "gate", "wu", "dav", "s4a", "s4b", "tail")}
    step_bn, step_dhu = k["fwd_h_bn"] / NT, k["dhu"] / NT
    nchain = H * slabs
    chain_cu = list(range(nchain))                       # chain i pinned to CU i (TP8: 48 < 80)
    par_ok = [True] * NCU
    if policy == "dedicated":
        for c in chain_cu: par_ok[c] = False
    # state
    t = 0.0
    events = []                                           # (time, seq, kind, payload)
    seq = 0

    def push(time, kind, payload):
        nonlocal seq
        heapq.heappush(events, (time, seq, kind, payload)); seq += 1

    # readiness bookkeeping per (h, c)
    done = set()
    ready = []                                           # parallel tasks ready to run: (prio, name, h, c)
    pr = {"l2n": 0, "gate": 0, "wu": 1, "dav": 2, "s4a": 3, "s4b": 4, "tail": 5}
    for h in range(H):
        for c in range(NT):
            heapq.heappush(ready, (c * 0 + pr["l2n"], c, "l2n", h, c))
            heapq.heappush(ready, (pr["gate"], c, "gate", h, c))
    cu_busy = [None] * NCU                               # parallel task running on the CU
    chain_state = {}                                     # chain -> dict(phase, step, busy_until, waiting)
    for i in range(nchain):
        chain_state[i] = dict(phase="bn", step=0, running=False)
    bn_done_step = {}                                    # (h, c) -> number of slabs done for fwd_h_bn step c
    dhu_done_step = {}
    finish_par = 0.0

    def chain_speed(cu):
        return S_CHAIN if (policy in ("shared", "idle-fill") and cu_busy[cu] is not None) else 1.0

    def par_speed(cu):
        if cu < nchain and policy in ("shared", "idle-fill") and chain_state[cu]["running"]:
            return S_PAR
        return 1.0

    def try_start_chain(i, now):
        st = chain_state[i]
        if st["running"] or st["phase"] == "end":
            return
        h = i // slabs
        if st["phase"] == "bn":
            c = st["step"]
            if ("wu", h, c) in done:
                st["running"] = True
                push(now + step_bn / chain_speed(i), "chain_done", i)
        else:                                            # dhu, reverse order
            c = NT - 1 - st["step"]
            if ("dav", h, c) in done and st.get("bn_all", False):
                st["running"] = True
                push(now + step_dhu / chain_speed(i), "chain_done", i)

    def deps_ok(name, h, c):
        if name in ("l2n", "gate"): return True
        if name == "wu": return ("l2n", h, c) in done and ("gate", h, c) in done
        if name == "dav": return bn_done_step.get((h, c), 0) == slabs
        if name == "s4a": return dhu_done_step.get((h, c), 0) == slabs
        if name == "s4b": return ("s4a", h, c) in done
        if name == "tail": return ("s4b", h, c) in done
        return False

    pending = []                                          # tasks whose deps are not met yet
    for h in range(H):
        for c in range(NT):
            for n in ("wu", "dav", "s4a", "s4b", "tail"):
                pending.append((n, h, c))
    pend_set = {(n, h, c) for (n, h, c) in pending}

    def release(now):
        nonlocal pend_set
        newly = [x for x in pend_set if deps_ok(*x)]
        for (n, h, c) in newly:
            pend_set.discard((n, h, c))
            heapq.heappush(ready, (pr[n], c, n, h, c))

    def dispatch(now):
        for cu in range(NCU):
            if cu_busy[cu] is not None or not par_ok[cu] or not ready:
                continue
            if policy == "idle-fill" and cu < nchain and chain_state[cu]["running"]:
                continue
            _, _, n, h, c = heapq.heappop(ready)
            cu_busy[cu] = (n, h, c)
            push(now + cu_t[n] / par_speed(cu), "par_done", cu)

    for i in range(nchain): try_start_chain(i, 0.0)
    dispatch(0.0)
    while events:
        now, _, kind, payload = heapq.heappop(events)
        if kind == "par_done":
            cu = payload
            n, h, c = cu_busy[cu]; cu_busy[cu] = None
            done.add((n, h, c)); finish_par = max(finish_par, now)
        else:
            i = payload; st = chain_state[i]; st["running"] = False
            h = i // slabs
            if st["phase"] == "bn":
                c = st["step"]
                bn_done_step[(h, c)] = bn_done_step.get((h, c), 0) + 1
                st["step"] += 1
                if st["step"] == NT:
                    st["phase"], st["step"], st["bn_all"] = "dhu", 0, True
            else:
                c = NT - 1 - st["step"]
                dhu_done_step[(h, c)] = dhu_done_step.get((h, c), 0) + 1
                st["step"] += 1
                if st["step"] == NT: st["phase"] = "end"
        release(now)
        for j in range(nchain): try_start_chain(j, now)
        dispatch(now)
    return finish_par


def main():
    H, NT = 12, 256
    k = KT[H]
    serial = sum(k[n] for n in ("l2n", "gate", "wu", "fwd_h_bn", "dav", "dhu", "s4a", "s4b", "tail"))
    print(f"16K/TP8 backward (us): serial kernel sum {serial:.0f}")
    for pol in ("shared", "dedicated", "idle-fill"):
        t = simulate_bwd(H, NT, pol)
        print(f"  persistent, {pol:9s}: {t:7.0f} us  ({serial / t:.2f}x, saves {serial - t:5.0f} us = {(serial - t) / 6750:.1%} of the op)")
    # forward, fluid estimate with the same contention (prep || recurrence)
    fp, fh = k["fwd_prep"], k["fwd_h_o"]
    chain_co = fh / S_CHAIN
    cap = (NCU - 48) + 48 * S_PAR                       # CU-equivalents for prep while 48 chains co-run
    prep_cu = fp * NCU
    t_fwd = max(chain_co, prep_cu / (cap)) if prep_cu / cap <= chain_co else chain_co + (prep_cu - cap * chain_co) / NCU
    print(f"16K/TP8 forward (us): serial {fp + fh:.0f}; persistent shared (fluid) {t_fwd:.0f} "
          f"(saves {fp + fh - t_fwd:.0f} us = {(fp + fh - t_fwd) / 6750:.1%} of the op)")


if __name__ == "__main__":
    main()
