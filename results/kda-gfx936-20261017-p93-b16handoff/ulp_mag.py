import torch
for TH in ("8192_12", "16384_48"):
    A = torch.load(f"/tmp/ulp93/g2_r5_{TH}.pt"); B = torch.load(f"/tmp/ulp93/g2_p93q_{TH}.pt")
    for n in ("dq", "dk"):
        a, b = A[n].double(), B[n].double()
        rms = a.pow(2).mean().sqrt()
        d = (a - b).abs()
        big = a.abs() > 0.1 * rms
        rel = d[big] / a.abs()[big]
        print(f"{TH} {n}: RMS {rms:.3e}  max|diff| {d.max():.3e} (= {float(d.max() / rms):.4f} RMS)  "
              f"where |x|>0.1RMS: max rel {float(rel.max()):.4f}, p99.9 rel {float(rel.quantile(0.999)):.5f}, mean rel {float(rel.mean()):.2e}; "
              f"elements with |x|<=0.1RMS: {float((~big).double().mean()):.2%}")
