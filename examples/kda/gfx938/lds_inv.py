import re, subprocess, collections, sys
s = open(sys.argv[1]).read()
funcs = re.findall(r"^(_Z\S+):[^\n]*\n(.*?)^\s*s_endpgm", s, re.S | re.M)
names = [n for n, _ in funcs]
dem = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.split("\n")
want = ["p3_wy_s4b_kernel<false>", "g2_wu_kernel<false, true, true>", "g2_fwd_prep_a_kernel<1, false, false>", "p3_wy_s4a_kernel<false, false, 1>",
        "g2_bwd_tail_kernel<false>", "g2_dhu_v8b_kernel<1, true>", "g2_fwd_h_v8o4_kernel<false, 5, true, true, true, true, true, true>",
        "g2_fwd_prep_b_kernel<false>", "g2_dav_kernel<false>", "g2_gate_kernel<1, false>"]
print(f"{'kernel':48s} {'wr_b16':>6s} {'wr_b32':>6s} {'wr_b64':>6s} {'wr_b128':>7s} {'rd_b16':>6s} {'rd_b32':>6s} {'rd_b64':>6s} {'rd_b128':>7s} {'bperm':>5s} {'rd_m':>4s} {'mmac':>5s}")
for (n, body), d in zip(funcs, dem):
    if not any(w in d for w in want): continue
    c = collections.Counter(re.findall(r"^\s+(ds_\w+|v_mmac\w*)", body, re.M))
    g = lambda *ks: sum(v for k, v in c.items() if any(k.startswith(x) for x in ks))
    print(f"{d.replace('void ','').replace('g2::','')[:48]:48s} {g('ds_write_b16','ds_write_b8'):6d} {g('ds_write_b32','ds_write2_b32','ds_write2st64_b32'):6d} {g('ds_write_b64','ds_write2_b64'):6d} {g('ds_write_b128'):7d} "
          f"{g('ds_read_u16','ds_read_i16','ds_read_u8'):6d} {g('ds_read_b32','ds_read2_b32','ds_read2st64_b32'):6d} {g('ds_read_b64','ds_read2_b64'):6d} {g('ds_read_b128'):7d} {g('ds_bpermute'):5d} {g('ds_read_m'):4d} {g('v_mmac'):5d}")
