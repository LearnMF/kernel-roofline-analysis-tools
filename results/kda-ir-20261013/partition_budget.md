| fused group | LDS peak KB | largest op core KB | on-chip handoffs (KB) | gap to 32 KB (res 2) |
|---|---:|---:|---|---:|
| re_gate+re_wy | 58.2 | 26.2 | g 32 | +26.2 |
| re_l2norm+re_gate+re_wy | 58.2 | 26.2 | g 32 | +26.2 |
| wy_bwd_a+wy_bwd_b | 64.0 | 32.0 | dwT 16, dA 16 | +32.0 |
| dav_a+wy_bwd_a+wy_bwd_b | 80.0 | 32.0 | dAqk 16, dwT 16, dA 16 | +48.0 |
| wy_bwd_b+tail | 104.0 | 32.0 | dq 32, dk 32, dg 8, db 0 | +72.0 |
| dav_a+wy_bwd_a+wy_bwd_b+tail | 152.0 | 32.0 | dAqk 16, dwT 16, dA 16, db 0, dq 32, dk 32, dg 8, db 0 | +120.0 |

| fused group | LDS peak KB | largest op core KB | on-chip handoffs (KB) | gap to 32 KB (res 2) |
|---|---:|---:|---|---:|
| re_gate+re_wy | 42.2 | 26.2 | g 16 | +10.2 |
| re_l2norm+re_gate+re_wy | 42.2 | 26.2 | g 16 | +10.2 |
| wy_bwd_a+wy_bwd_b | 44.0 | 32.0 | dwT 8, dA 4 | +12.0 |
| dav_a+wy_bwd_a+wy_bwd_b | 48.0 | 32.0 | dAqk 4, dwT 8, dA 4 | +16.0 |
| wy_bwd_b+tail | 68.0 | 32.0 | dq 16, dk 16, dg 4, db 0 | +36.0 |
| dav_a+wy_bwd_a+wy_bwd_b+tail | 84.0 | 32.0 | dAqk 4, dwT 8, dA 4, db 0, dq 16, dk 16, dg 4, db 0 | +52.0 |
