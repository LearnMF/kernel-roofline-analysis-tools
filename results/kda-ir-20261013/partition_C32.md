segments (dav split):
  S0: ['re_l2norm', 're_gate', 're_wy', 'dav_v']
  S1: []
  S2: ['dav_a', 'wy_bwd_a', 'wy_bwd_b', 'tail']

production-like grouping (every op its own kernel): handoff 11328 B/tok*head

| kernels (fused groups only) | handoff B/tok*head | vs production | worst verdict | per fused kernel: LDS KB / VGPR / res | history |
|---|---:|---:|---|---|---|
| re_l2norm+re_gate+re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b | 9280 | -2048 | res1 | 42.2/212/1; 48.0/236/1 | F6 (P-39): LDS does not fit -> slower; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; dav_a+wy_bwd_a+wy_bwd_b | 9280 | -2048 | res1 | 42.2/212/1; 48.0/236/1 | F6 (P-39): LDS does not fit -> slower; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy+dav_v; wy_bwd_a+wy_bwd_b | 9536 | -1792 | res1 | 42.2/212/1; 44.0/236/1 | F6 (P-39): LDS does not fit -> slower; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; wy_bwd_a+wy_bwd_b | 9536 | -1792 | res1 | 42.2/212/1; 44.0/236/1 | F6 (P-39): LDS does not fit -> slower; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_gate+re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b | 9792 | -1536 | res1 | 42.2/212/1; 48.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_gate+re_wy; dav_a+wy_bwd_a+wy_bwd_b | 9792 | -1536 | res1 | 42.2/212/1; 48.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_gate+re_wy+dav_v; wy_bwd_a+wy_bwd_b | 10048 | -1280 | res1 | 42.2/212/1; 44.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_gate+re_wy; wy_bwd_a+wy_bwd_b | 10048 | -1280 | res1 | 42.2/212/1; 44.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_l2norm+re_gate+re_wy+dav_v; dav_a+wy_bwd_a | 10304 | -1024 | res1 | 42.2/212/1; 29.8/152/2 | F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy+dav_v | 10304 | -1024 | res1 | 42.2/212/1 | F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; dav_a+wy_bwd_a | 10304 | -1024 | res1 | 42.2/212/1; 29.8/152/2 | F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy | 10304 | -1024 | res1 | 42.2/212/1 | F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate; re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b | 10304 | -1024 | res1 | 2.0/96/8; 26.2/212/2; 48.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_l2norm+re_gate; dav_a+wy_bwd_a+wy_bwd_b | 10304 | -1024 | res1 | 2.0/96/8; 48.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b | 10304 | -1024 | res1 | 26.2/212/2; 48.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| dav_a+wy_bwd_a+wy_bwd_b | 10304 | -1024 | res1 | 48.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_l2norm+re_gate; re_wy+dav_v; wy_bwd_a+wy_bwd_b | 10560 | -768 | res1 | 2.0/96/8; 26.2/212/2; 44.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_l2norm+re_gate; wy_bwd_a+wy_bwd_b | 10560 | -768 | res1 | 2.0/96/8; 44.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_wy+dav_v; wy_bwd_a+wy_bwd_b | 10560 | -768 | res1 | 26.2/212/2; 44.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| wy_bwd_a+wy_bwd_b | 10560 | -768 | res1 | 44.0/236/1 | F6 (P-39): LDS does not fit -> slower |
| re_gate+re_wy+dav_v; dav_a+wy_bwd_a | 10816 | -512 | res1 | 42.2/212/1; 29.8/152/2 |  |
| re_gate+re_wy+dav_v | 10816 | -512 | res1 | 42.2/212/1 |  |
| re_gate+re_wy; dav_a+wy_bwd_a | 10816 | -512 | res1 | 42.2/212/1; 29.8/152/2 |  |
| re_gate+re_wy | 10816 | -512 | res1 | 42.2/212/1 |  |
| re_l2norm+re_gate; re_wy+dav_v; dav_a+wy_bwd_a | 11328 | +0 | res2+ | 2.0/96/8; 26.2/212/2; 29.8/152/2 |  |
| re_l2norm+re_gate; re_wy+dav_v | 11328 | +0 | res2+ | 2.0/96/8; 26.2/212/2 |  |
| re_l2norm+re_gate; dav_a+wy_bwd_a | 11328 | +0 | res2+ | 2.0/96/8; 29.8/152/2 |  |
| re_l2norm+re_gate | 11328 | +0 | res2+ | 2.0/96/8 |  |
| re_wy+dav_v; dav_a+wy_bwd_a | 11328 | +0 | res2+ | 26.2/212/2; 29.8/152/2 |  |
| re_wy+dav_v | 11328 | +0 | res2+ | 26.2/212/2 |  |
| dav_a+wy_bwd_a | 11328 | +0 | res2+ | 29.8/152/2 |  |
| re_l2norm+re_gate+re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b+tail | 5680 | -5648 | infeasible | 42.2/212/1; 84.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; dav_a+wy_bwd_a+wy_bwd_b+tail | 5680 | -5648 | infeasible | 42.2/212/1; 84.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy+dav_v; wy_bwd_a+wy_bwd_b+tail | 5936 | -5392 | infeasible | 42.2/212/1; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; wy_bwd_a+wy_bwd_b+tail | 5936 | -5392 | infeasible | 42.2/212/1; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_gate+re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b+tail | 6192 | -5136 | infeasible | 42.2/212/1; 84.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy; dav_a+wy_bwd_a+wy_bwd_b+tail | 6192 | -5136 | infeasible | 42.2/212/1; 84.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy+dav_v; wy_bwd_a+wy_bwd_b+tail | 6448 | -4880 | infeasible | 42.2/212/1; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy; wy_bwd_a+wy_bwd_b+tail | 6448 | -4880 | infeasible | 42.2/212/1; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b+tail | 6704 | -4624 | infeasible | 2.0/96/8; 26.2/212/2; 84.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; dav_a+wy_bwd_a+wy_bwd_b+tail | 6704 | -4624 | infeasible | 2.0/96/8; 84.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_wy+dav_v; dav_a+wy_bwd_a+wy_bwd_b+tail | 6704 | -4624 | infeasible | 26.2/212/2; 84.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| dav_a+wy_bwd_a+wy_bwd_b+tail | 6704 | -4624 | infeasible | 84.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate+re_wy+dav_v; dav_a+wy_bwd_a; wy_bwd_b+tail | 6712 | -4616 | infeasible | 42.2/212/1; 29.8/152/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy+dav_v; wy_bwd_b+tail | 6712 | -4616 | infeasible | 42.2/212/1; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; dav_a+wy_bwd_a; wy_bwd_b+tail | 6712 | -4616 | infeasible | 42.2/212/1; 29.8/152/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate+re_wy; wy_bwd_b+tail | 6712 | -4616 | infeasible | 42.2/212/1; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x; F1 / P-36: l2n folded into wu, wu +22-27% |
| re_l2norm+re_gate; re_wy+dav_v; wy_bwd_a+wy_bwd_b+tail | 6960 | -4368 | infeasible | 2.0/96/8; 26.2/212/2; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; wy_bwd_a+wy_bwd_b+tail | 6960 | -4368 | infeasible | 2.0/96/8; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_wy+dav_v; wy_bwd_a+wy_bwd_b+tail | 6960 | -4368 | infeasible | 26.2/212/2; 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| wy_bwd_a+wy_bwd_b+tail | 6960 | -4368 | infeasible | 80.0/236/0 | F6 (P-39): LDS does not fit -> slower; B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy+dav_v; dav_a+wy_bwd_a; wy_bwd_b+tail | 7224 | -4104 | infeasible | 42.2/212/1; 29.8/152/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy+dav_v; wy_bwd_b+tail | 7224 | -4104 | infeasible | 42.2/212/1; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy; dav_a+wy_bwd_a; wy_bwd_b+tail | 7224 | -4104 | infeasible | 42.2/212/1; 29.8/152/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_gate+re_wy; wy_bwd_b+tail | 7224 | -4104 | infeasible | 42.2/212/1; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; re_wy+dav_v; dav_a+wy_bwd_a; wy_bwd_b+tail | 7736 | -3592 | infeasible | 2.0/96/8; 26.2/212/2; 29.8/152/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; re_wy+dav_v; wy_bwd_b+tail | 7736 | -3592 | infeasible | 2.0/96/8; 26.2/212/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; dav_a+wy_bwd_a; wy_bwd_b+tail | 7736 | -3592 | infeasible | 2.0/96/8; 29.8/152/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_l2norm+re_gate; wy_bwd_b+tail | 7736 | -3592 | infeasible | 2.0/96/8; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_wy+dav_v; dav_a+wy_bwd_a; wy_bwd_b+tail | 7736 | -3592 | infeasible | 26.2/212/2; 29.8/152/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| re_wy+dav_v; wy_bwd_b+tail | 7736 | -3592 | infeasible | 26.2/212/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| dav_a+wy_bwd_a; wy_bwd_b+tail | 7736 | -3592 | infeasible | 29.8/152/2; 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
| wy_bwd_b+tail | 7736 | -3592 | infeasible | 68.0/236/0 | B4b-lite (P-47): kept dq/dk round trip, 0.88-0.91x |
