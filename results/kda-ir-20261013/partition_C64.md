segments (dav split):
  S0: ['re_l2norm', 're_gate', 're_wy', 'dav_v']
  S1: []
  S2: ['dav_a', 'wy_bwd_a', 'wy_bwd_b', 'tail']

production-like grouping (every op its own kernel): handoff 11816 B/tok*head

| kernels (fused groups only) | handoff B/tok*head | vs production | worst verdict | per fused kernel: LDS KB / VGPR / res | history |
|---|---:|---:|---|---|---|
| re_l2norm+re_gate+re_wy+dav_v; wy_bwd_a+wy_bwd_b | 9768 | -2048 | res1 | 58.2/212/1; 64.0/236/1 | F6 (P-39): LDS does not fit -> slower; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; wy_bwd_a+wy_bwd_b | 9768 | -2048 | res1 | 58.2/212/1; 64.0/236/1 | F6 (P-39): LDS does not fit -> slower; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_gate+re_wy+dav_v; wy_bwd_a+wy_bwd_b | 10280 | -1536 | res1 | 58.2/212/1; 64.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_gate+re_wy; wy_bwd_a+wy_bwd_b | 10280 | -1536 | res1 | 58.2/212/1; 64.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_l2norm+re_gate+re_wy+dav_v; dav_a+wy_bwd_a | 10792 | -1024 | res1 | 58.2/212/1; 29.8/152/2 | F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy+dav_v | 10792 | -1024 | res1 | 58.2/212/1 | F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; dav_a+wy_bwd_a | 10792 | -1024 | res1 | 58.2/212/1; 29.8/152/2 | F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy | 10792 | -1024 | res1 | 58.2/212/1 | F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate; re_wy+dav_v; wy_bwd_a+wy_bwd_b | 10792 | -1024 | res1 | 2.0/96/8; 26.2/212/2; 64.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_l2norm+re_gate; wy_bwd_a+wy_bwd_b | 10792 | -1024 | res1 | 2.0/96/8; 64.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_wy+dav_v; wy_bwd_a+wy_bwd_b | 10792 | -1024 | res1 | 26.2/212/2; 64.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| wy_bwd_a+wy_bwd_b | 10792 | -1024 | res1 | 64.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_gate+re_wy+dav_v; dav_a+wy_bwd_a | 11304 | -512 | res1 | 58.2/212/1; 29.8/152/2 |  |
| re_gate+re_wy+dav_v | 11304 | -512 | res1 | 58.2/212/1 |  |
| re_gate+re_wy; dav_a+wy_bwd_a | 11304 | -512 | res1 | 58.2/212/1; 29.8/152/2 |  |
| re_gate+re_wy | 11304 | -512 | res1 | 58.2/212/1 |  |
| re_l2norm+re_gate; re_wy+dav_v; dav_a+wy_bwd_a | 11816 | +0 | res2+ | 2.0/96/8; 26.2/212/2; 29.8/152/2 |  |
| re_l2norm+re_gate; re_wy+dav_v | 11816 | +0 | res2+ | 2.0/96/8; 26.2/212/2 |  |
| re_l2norm+re_gate; dav_a+wy_bwd_a | 11816 | +0 | res2+ | 2.0/96/8; 29.8/152/2 |  |
| re_l2norm+re_gate | 11816 | +0 | res2+ | 2.0/96/8 |  |
| re_wy+dav_v; dav_a+wy_bwd_a | 11816 | +0 | res2+ | 26.2/212/2; 29.8/152/2 |  |
| re_wy+dav_v | 11816 | +0 | res2+ | 26.2/212/2 |  |
| dav_a+wy_bwd_a | 11816 | +0 | res2+ | 29.8/152/2 |  |
| re_l2norm+re_gate+re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b+tail | 5656 | -6160 | infeasible | 58.2/212/1; 152.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; dav_a+wy_bwd_a+wy_bwd_b+tail | 5656 | -6160 | infeasible | 58.2/212/1; 152.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy+dav_v; wy_bwd_a+wy_bwd_b+tail | 6168 | -5648 | infeasible | 58.2/212/1; 136.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; wy_bwd_a+wy_bwd_b+tail | 6168 | -5648 | infeasible | 58.2/212/1; 136.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_gate+re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b+tail | 6168 | -5648 | infeasible | 58.2/212/1; 152.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy; dav_a+wy_bwd_a+wy_bwd_b+tail | 6168 | -5648 | infeasible | 58.2/212/1; 152.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b+tail | 6680 | -5136 | infeasible | 2.0/96/8; 26.2/212/2; 152.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; dav_a+wy_bwd_a+wy_bwd_b+tail | 6680 | -5136 | infeasible | 2.0/96/8; 152.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy+dav_v; wy_bwd_a+wy_bwd_b+tail | 6680 | -5136 | infeasible | 58.2/212/1; 136.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy; wy_bwd_a+wy_bwd_b+tail | 6680 | -5136 | infeasible | 58.2/212/1; 136.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b+tail | 6680 | -5136 | infeasible | 26.2/212/2; 152.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| dav_a+wy_bwd_a+wy_bwd_b+tail | 6680 | -5136 | infeasible | 152.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; re_wy+dav_v; wy_bwd_a+wy_bwd_b+tail | 7192 | -4624 | infeasible | 2.0/96/8; 26.2/212/2; 136.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; wy_bwd_a+wy_bwd_b+tail | 7192 | -4624 | infeasible | 2.0/96/8; 136.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_wy+dav_v; wy_bwd_a+wy_bwd_b+tail | 7192 | -4624 | infeasible | 26.2/212/2; 136.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| wy_bwd_a+wy_bwd_b+tail | 7192 | -4624 | infeasible | 136.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate+re_wy+dav_v; dav_a+wy_bwd_a; wy_bwd_b+tail | 7200 | -4616 | infeasible | 58.2/212/1; 29.8/152/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy+dav_v; wy_bwd_b+tail | 7200 | -4616 | infeasible | 58.2/212/1; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; dav_a+wy_bwd_a; wy_bwd_b+tail | 7200 | -4616 | infeasible | 58.2/212/1; 29.8/152/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; wy_bwd_b+tail | 7200 | -4616 | infeasible | 58.2/212/1; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_gate+re_wy+dav_v; dav_a+wy_bwd_a; wy_bwd_b+tail | 7712 | -4104 | infeasible | 58.2/212/1; 29.8/152/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy+dav_v; wy_bwd_b+tail | 7712 | -4104 | infeasible | 58.2/212/1; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy; dav_a+wy_bwd_a; wy_bwd_b+tail | 7712 | -4104 | infeasible | 58.2/212/1; 29.8/152/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy; wy_bwd_b+tail | 7712 | -4104 | infeasible | 58.2/212/1; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; re_wy+dav_v; dav_a+wy_bwd_a; wy_bwd_b+tail | 8224 | -3592 | infeasible | 2.0/96/8; 26.2/212/2; 29.8/152/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; re_wy+dav_v; wy_bwd_b+tail | 8224 | -3592 | infeasible | 2.0/96/8; 26.2/212/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; dav_a+wy_bwd_a; wy_bwd_b+tail | 8224 | -3592 | infeasible | 2.0/96/8; 29.8/152/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; wy_bwd_b+tail | 8224 | -3592 | infeasible | 2.0/96/8; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_wy+dav_v; dav_a+wy_bwd_a; wy_bwd_b+tail | 8224 | -3592 | infeasible | 26.2/212/2; 29.8/152/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_wy+dav_v; wy_bwd_b+tail | 8224 | -3592 | infeasible | 26.2/212/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| dav_a+wy_bwd_a; wy_bwd_b+tail | 8224 | -3592 | infeasible | 29.8/152/2; 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| wy_bwd_b+tail | 8224 | -3592 | infeasible | 104.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate+re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b | 9256 | -2560 | infeasible | 58.2/212/1; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; dav_a+wy_bwd_a+wy_bwd_b | 9256 | -2560 | infeasible | 58.2/212/1; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_gate+re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b | 9768 | -2048 | infeasible | 58.2/212/1; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower |
| re_gate+re_wy; dav_a+wy_bwd_a+wy_bwd_b | 9768 | -2048 | infeasible | 58.2/212/1; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower |
| re_l2norm+re_gate; re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b | 10280 | -1536 | infeasible | 2.0/96/8; 26.2/212/2; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower |
| re_l2norm+re_gate; dav_a+wy_bwd_a+wy_bwd_b | 10280 | -1536 | infeasible | 2.0/96/8; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower |
| re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b | 10280 | -1536 | infeasible | 26.2/212/2; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower |
| dav_a+wy_bwd_a+wy_bwd_b | 10280 | -1536 | infeasible | 80.0/236/0 | F6 (P-39): LDS does not fit -> slower |
