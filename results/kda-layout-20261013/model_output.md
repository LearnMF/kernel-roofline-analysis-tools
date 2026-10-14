| kernel | shape | VMEM/wave inventory | VMEM/wave PMC | TA/wave predicted (old table) | TA/wave measured | missing pattern costs |
|---|---|---:|---:|---:|---:|---|
| wy_s4b | 16384_12 | 126.5 | 128.8 | 7294 | 9088 | ld 4B/lane 16x4B; ld 4B/lane 4x64B |
| wy_s4b | 16384_48 | 126.5 | 128.8 | 7294 | 10472 | ld 4B/lane 16x4B; ld 4B/lane 4x64B |
| bwd_tail | 16384_12 | 148.2 | 194.8 | 1730 | 5907 | ld 4B/lane 16x4B; st 4B/lane 16x4B; st 4B/lane 1x256B |
| bwd_tail | 16384_48 | 148.2 | 194.8 | 1730 | 6155 | ld 4B/lane 16x4B; st 4B/lane 16x4B; st 4B/lane 1x256B |

ld 16B/lane 16x64B               169
ld 16B/lane 1x1024B              21
ld 16B/lane 4x256B               43
ld 4B/lane 16x4B                 MISSING
ld 4B/lane 1x256B                10
ld 4B/lane 4x64B                 MISSING
ld 4B/lane 64x4B                 229
ld 8B/lane 16x32B                85
ld 8B/lane 1x512B                22
st 16B/lane 16x64B               67
st 16B/lane 1x1024B              56
st 4B/lane 16x4B                 MISSING
st 4B/lane 1x256B                MISSING

| tensor | access (kernel role n/wave) | row | natA | natB | natC |
|---|---|---|---|---|---|
| Akk | wy_s4b frag8 x4 | ld 8B/lane 16x32B = 85 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 |
| Vn | wy_s4b frag8 x8 | ld 8B/lane 16x32B = 85 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 |
| beta/db | bwd_tail rowvec64 x0.75 | ld 4B/lane 64x4B = 229 | ld 4B/lane 64x4B = 229 | ld 4B/lane 64x4B = 229 | ld 4B/lane 64x4B = 229 |
| dA | wy_s4b frag16f x4 | ld 16B/lane 16x64B = 169 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 |
| dA | wy_s4b gather4 x10 | ld 4B/lane 4x64B = ? | ld 4B/lane 4x64B = ? | ld 4B/lane 4x64B = ? | ld 4B/lane 4x64B = ? |
| dAqk | wy_s4b frag16f x2.5 | ld 16B/lane 16x64B = 169 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 |
| dAqk | wy_s4b gather4 x10 | ld 4B/lane 4x64B = ? | ld 4B/lane 4x64B = ? | ld 4B/lane 4x64B = ? | ld 4B/lane 4x64B = ? |
| dg | wy_s4b frag16f st x8 | st 16B/lane 16x64B = 67 | st 16B/lane 1x1024B = 56 | st 16B/lane 1x1024B = 56 | st 16B/lane 1x1024B = 56 |
| dg | bwd_tail row2f x16 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 +LDS remap | ld 8B/lane 1x512B = 22 +LDS remap | ld 8B/lane 1x512B = 22 +LDS remap |
| dh | wy_s4b native16 x8 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 |
| dk | wy_s4b frag16f st x8 | st 16B/lane 16x64B = 67 | st 16B/lane 1x1024B = 56 | st 16B/lane 1x1024B = 56 | st 16B/lane 1x1024B = 56 |
| dk | bwd_tail row2f x16 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 +LDS remap | ld 8B/lane 1x512B = 22 +LDS remap | ld 8B/lane 1x512B = 22 +LDS remap |
| dq | wy_s4b frag16f st x8 | st 16B/lane 16x64B = 67 | st 16B/lane 1x1024B = 56 | st 16B/lane 1x1024B = 56 | st 16B/lane 1x1024B = 56 |
| dq | bwd_tail row2f x16 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 +LDS remap | ld 8B/lane 1x512B = 22 +LDS remap | ld 8B/lane 1x512B = 22 +LDS remap |
| dwT | wy_s4b tile4x x8 | ld 16B/lane 4x256B = 43 | ld 16B/lane 4x256B = 43 | ld 16B/lane 4x256B = 43 | ld 16B/lane 4x256B = 43 |
| g | wy_s4b frag16f x8 | ld 16B/lane 16x64B = 169 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 |
| g | wy_s4b rowvec16 x8 | ld 4B/lane 16x4B = ? | ld 4B/lane 16x4B = ? | ld 4B/lane 16x4B = ? | ld 4B/lane 16x4B = ? |
| h | wy_s4b native16 x8 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 | ld 16B/lane 1x1024B = 21 |
| kn | wy_s4b frag8 x8 | ld 8B/lane 16x32B = 85 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 |
| kn | bwd_tail row2h x16 | ld 4B/lane 1x256B = 10 | ld 4B/lane 1x256B = 10 +LDS remap | ld 4B/lane 1x256B = 10 +LDS remap | ld 4B/lane 1x256B = 10 +LDS remap |
| partials | bwd_tail rowvec16 st x0.5 | st 4B/lane 16x4B = ? | st 4B/lane 16x4B = ? | st 4B/lane 16x4B = ? | st 4B/lane 16x4B = ? |
| qn | wy_s4b frag8 x8 | ld 8B/lane 16x32B = 85 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 | ld 8B/lane 1x512B = 22 |
| qn | bwd_tail row2h x16 | ld 4B/lane 1x256B = 10 | ld 4B/lane 1x256B = 10 +LDS remap | ld 4B/lane 1x256B = 10 +LDS remap | ld 4B/lane 1x256B = 10 +LDS remap |
| rstd | bwd_tail rowvec16 x2 | ld 4B/lane 16x4B = ? | ld 4B/lane 16x4B = ? | ld 4B/lane 16x4B = ? | ld 4B/lane 16x4B = ? |
