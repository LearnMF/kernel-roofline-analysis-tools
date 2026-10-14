| intermediate | dtype | layout | B/tok*head | writes | reads |
|---|---|---|---:|---:|---:|
| Aqk | bf16 | row | 128 | 1 | 1 |
| AqkA | bf16 | natA | 128 | 1 | 1 |
| Gn | fp32 | vecK | 8 | 1 | 3 |
| Gn_raw | fp32 | vecK | 8 | 0 | 1 |
| KgA | bf16 | natA | 256 | 1 | 1 |
| Kn | bf16 | natB | 256 | 2 | 2 |
| QgA | bf16 | natA | 256 | 1 | 1 |
| Qn | bf16 | natB | 256 | 1 | 1 |
| Akk | bf16 | row | 128 | 1 | 2 |
| Un | bf16 | natC | 256 | 2 | 2 |
| Wb | bf16 | natB | 256 | 1 | 1 |
| Wn | bf16 | natA | 256 | 2 | 2 |
| beta | fp32 | row | 4 | 0 | 3 |
| dA | fp32 | row | 256 | 1 | 1 |
| dAqk | fp32 | row | 256 | 1 | 1 |
| dVn | bf16 | natC | 256 | 1 | 1 |
| db | fp32 | row | 4 | 1 | 1 |
| dg | fp32 | row | 512 | 1 | 1 |
| dh | bf16 | bN | 512 | 1 | 1 |
| dk | fp32 | row | 512 | 1 | 1 |
| dOT | bf16 | natB | 256 | 1 | 1 |
| dq | fp32 | row | 512 | 1 | 1 |
| dv2 | bf16 | aN | 256 | 1 | 1 |
| dwT | bf16 | natB | 256 | 1 | 1 |
| g | fp32 | row | 512 | 1 | 2 |
| h | bf16 | bN | 512 | 1 | 1 |
| kgB | bf16 | vB | 256 | 1 | 1 |
| kn | bf16 | row | 256 | 1 | 3 |
| qn | bf16 | row | 256 | 1 | 3 |
| rstd_k | fp32 | row | 4 | 1 | 2 |
| rstd_q | fp32 | row | 4 | 1 | 2 |
| Vn | bf16 | natC | 256 | 1 | 2 |

intermediate traffic per token*head: write 8596 B, read 10560 B
