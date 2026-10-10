"""kra P-14 (global route, round 1): dav's VMEM instruction count 104 -> 64 per lane, values
untouched.  dOT (32 scattered 2 B stores per lane) is emitted AFTER the pass-2 transposed
staging from the slab ([v][j], 4 consecutive j contiguous = one dOT quad) as 16 B stores
(two 8 B LDS reads each, 4 per lane); dA (16 scalar fp32 stores per lane) goes through the
slab (64 x 68 fp32 = exactly the 17.4 KB slab) and leaves as 16 B row stores (4 per lane).
Usage: python3 p14_dav_wide_stores.py <hip_kda tree root>"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) / "csrc" / "g2" / "g2_dav.cuh"
t = P.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, old[:80]
    t = t.replace(old, new)


# ---- dA: C fragments -> slab (fp32 [t][68]) -> 16 B row stores
sub("""            #pragma unroll
            for (int jt = 0; jt < 4; ++jt) {
                const int t = 16 * wv + (l & 15), j0 = 16 * jt + (l >> 4);
                if (!g2_row_ok<M>(t, cl)) continue;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {           // C-frag col = (l>>4) + 4e
                    const float val = (t >= j0 + 4 * e) ? acc[jt][e] * p.scale : 0.f;
                    p.dA[g2_rm_tok(t0 + t, hh, p.H, kBT) + j0 + 4 * e] = val;
                }
            }
        }
""", """            // kra P-14: the same values, staged through the (now free) slab so they leave
            // as 16 B row stores instead of 16 scalar 4 B stores per lane
            __syncthreads();                                      // all waves done with Vn
            float* dAs = reinterpret_cast<float*>(slab);          // [64][kPadJ3] fp32 = slab
            #pragma unroll
            for (int jt = 0; jt < 4; ++jt) {
                const int t = 16 * wv + (l & 15), j0 = 16 * jt + (l >> 4);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {           // C-frag col = (l>>4) + 4e
                    const float val = (t >= j0 + 4 * e) ? acc[jt][e] * p.scale : 0.f;
                    dAs[t * kPadJ3 + j0 + 4 * e] = val;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int it = 0; it < 4; ++it) {                      // 64 rows x 16 quads
                const int c = tid + 256 * it, t = c >> 4, j = (c & 15) * 4;
                if (!g2_row_ok<M>(t, cl)) continue;
                *reinterpret_cast<f4*>(p.dA + g2_rm_tok(t0 + t, hh, p.H, kBT) + j) =
                    *reinterpret_cast<const f4*>(dAs + t * kPadJ3 + j);
            }
        }
""")

# ---- dOT: drop the scattered scalar emit from the staging loop ...
sub("""            const int ibd = (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3)
                            + 256 * (j >> 4) + 1024 * (vq >> 4);
            #pragma unroll
            for (int d = 0; d < 4; ++d) p.dOT[nblk + ibd + 4 * d] = dv_[d];
        }
        __syncthreads();
""", """        }
        __syncthreads();
        // kra P-14: dOT (B form) from the transposed staging, 16 B per store.  dOT index of
        // element (j, v) = (j&3) + 4(v&15) + 64((j>>2)&3) + 256(j>>4) + 1024(v>>4): the 8
        // shorts of chunk c are rows j0..j0+3 at columns v0 and v0+1 -- two 8 B slab reads.
        if (p.dVn != nullptr)
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = tid + 256 * it, x = 8 * c;
            const int j0 = 4 * ((x >> 6) & 3) + 16 * ((x >> 8) & 3);
            const int v0 = ((x >> 2) & 15) + 16 * (x >> 10);
            const v4bh lo = *reinterpret_cast<const v4bh*>(slab + (size_t)v0 * kPadJ3 + j0);
            const v4bh hi = *reinterpret_cast<const v4bh*>(slab + (size_t)(v0 + 1) * kPadJ3 + j0);
            st16x2(p.dOT + nblk + x, lo, hi);
        }
""")
P.write_text(t)
print("P-14 APPLIED")
