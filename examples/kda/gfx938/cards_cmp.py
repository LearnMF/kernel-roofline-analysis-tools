import json
A = json.load(open('results/kda-gfx938-cards-20261025/cards.json'))
B = json.load(open('results/kda-gfx938-cards-20261027-hd1/cards.json'))
hb = json.load(open('machines/gfx938-bw1102.json'))["peaks"]["hbm_read"]["value"]
for sh in ['2048_16', '8192_64', '16384_8']:
    a, b = A[sh]['cards'], B[sh]['cards']
    ta, tb = sum(c['t_us'] for c in a.values()), sum(c['t_us'] for c in b.values())
    print(f"== {sh}: chain {ta:.0f} -> {tb:.0f} us ({ta/tb:.3f}x vs a83ac15)")
    print(f"   {'kernel':14s} {'a83':>6s} {'HEAD':>6s} {'share':>6s} {'HBM%':>5s} {'TA%':>4s} {'VALU%':>5s} {'MMAC%':>5s} {'floor':>6s} {'gap':>5s} {'gap_us':>6s} primary")
    rows = []
    for k, c in b.items():
        if c['t_us'] < 0.015 * tb: continue
        hbmu = ((c.get('hbm_rd') or 0) + (c.get('hbm_wr') or 0)) / (c['t_us'] * 1e3) / hb
        rows.append((c['t_us'] - c['floor_us'], k, c, hbmu))
    for gap_us, k, c, hbmu in sorted(rows, key=lambda r: -r[0]):
        print(f"   {k.split('#')[0]:14s} {a.get(k, {}).get('t_us', 0):6.0f} {c['t_us']:6.0f} {c['t_us']/tb:6.1%} {hbmu:5.0%} {c.get('u_ta',0):4.0%} {c.get('u_valu',0):5.0%} {c.get('u_mmop',0):5.0%} {c['floor_us']:6.0f} {c['t_us']/c['floor_us']:5.2f} {gap_us:6.0f} {c.get('primary')}")
