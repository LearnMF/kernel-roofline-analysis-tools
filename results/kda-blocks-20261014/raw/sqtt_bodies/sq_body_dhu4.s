.LBBX:  ; =>This Inner Loop Header: Depth=1
	s_add_i32 s16, s16, -2
	s_cmp_gt_i32 s6, s34
	s_cselect_b32 s6, s16, s34
	s_and_b32 s22, s21, 1
	s_lshl_b32 s7, s22, 12
	s_or_b32 s7, s7, 0x2100
	v_lshl_add_u32 v57, v0, 3, s7
	s_waitcnt vmcnt(17)
	ds_write_b64 v57, v[63:64]
	v_lshl_add_u32 v57, v106, 3, s7
	s_ashr_i32 s7, s6, 31
	s_add_u32 s16, s6, s50
	s_addc_u32 s17, s7, s49
	s_lshl_b64 s[6:7], s[16:17], 3
	s_add_u32 s14, s6, s26
	s_addc_u32 s15, s7, s27
	s_lshl_b64 s[14:15], s[14:15], 11
	s_add_u32 s6, s6, s53
	s_waitcnt vmcnt(16)
	ds_write_b64 v57, v[65:66]
	v_or_b32_e32 v57, s14, v131
	s_addc_u32 s7, s7, s54
	v_mov_b32_e32 v58, s15
	v_add_co_u32_e32 v57, vcc, v113, v57
	s_lshl_b64 s[6:7], s[6:7], 11
	v_addc_co_u32_e32 v58, vcc, v114, v58, vcc
	v_or_b32_e32 v59, s6, v131
	v_mov_b32_e32 v60, s7
	v_add_co_u32_e32 v59, vcc, v113, v59
	v_addc_co_u32_e32 v60, vcc, v114, v60, vcc
	global_load_dwordx2 v[63:64], v[57:58], off
	global_load_dwordx2 v[65:66], v[59:60], off
	s_waitcnt lgkmcnt(0)
	OTHER_MSG
	s_barrier
	s_mul_i32 s23, s4, s56
	s_mul_hi_u32 s37, s4, s57
	s_mul_i32 s38, s5, s57
	s_mul_i32 s39, s4, s57
	s_mov_b64 vcc, s[2:3]
	s_cbranch_vccz 346
	s_add_i32 s6, s37, s23
	s_add_i32 s7, s6, s38
	s_add_u32 s6, s39, s35
	s_addc_u32 s7, s7, s48
	s_lshl_b64 s[14:15], s[6:7], 6
	ds_read_u16 v57, v108
	ds_read_u16 v59, v107
	ds_read_u16 v60, v107 offset:264
	ds_read_u16 v58, v107 offset:528
	ds_read_u16 v85, v107 offset:4224
	ds_read_u16 v86, v107 offset:4488
	ds_read_u16 v87, v107 offset:4752
	ds_read_u16 v88, v107 offset:5016
	v_or_b32_e32 v133, s14, v92
	s_waitcnt lgkmcnt(4)
	v_perm_b32 v58, v57, v58, s18
	v_perm_b32 v57, v60, v59, s18
	v_or_b32_e64 v132, s15, 0
	v_mov_b32_e32 v134, s27
	v_add_co_u32_e32 v59, vcc, s26, v133
	v_addc_co_u32_e32 v60, vcc, v132, v134, vcc
	v_lshlrev_b64 v[59:60], 9, v[59:60]
	v_lshlrev_b32_e32 v135, 1, v101
	v_or_b32_e32 v59, v59, v135
	v_mov_b32_e32 v136, s29
	v_add_co_u32_e32 v59, vcc, s28, v59
	v_addc_co_u32_e32 v60, vcc, v136, v60, vcc
	global_store_dwordx2 v[59:60], v[57:58], off
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v58, v88, v87, s18
	v_mov_b32_e32 v87, s59
	v_add_co_u32_e32 v59, vcc, s58, v133
	v_addc_co_u32_e32 v60, vcc, v132, v87, vcc
	v_lshlrev_b64 v[59:60], 9, v[59:60]
	v_perm_b32 v57, v86, v85, s18
	v_or_b32_e32 v59, v59, v135
	v_add_co_u32_e32 v59, vcc, s28, v59
	v_addc_co_u32_e32 v60, vcc, v136, v60, vcc
	v_or_b32_e32 v139, s14, v109
	global_store_dwordx2 v[59:60], v[57:58], off
	v_or_b32_e64 v138, s15, 0
	v_add_co_u32_e32 v59, vcc, s26, v139
	v_addc_co_u32_e32 v60, vcc, v138, v134, vcc
	v_lshlrev_b64 v[59:60], 9, v[59:60]
	ds_read_u16 v57, v111
	ds_read_u16 v85, v110
	ds_read_u16 v86, v110 offset:264
	ds_read_u16 v58, v110 offset:528
	ds_read_u16 v88, v110 offset:4224
	ds_read_u16 v132, v110 offset:4488
	ds_read_u16 v133, v110 offset:4752
	ds_read_u16 v137, v110 offset:5016
	v_or_b32_e32 v59, v59, v135
	v_add_co_u32_e32 v59, vcc, s28, v59
	s_waitcnt lgkmcnt(4)
	v_perm_b32 v58, v57, v58, s18
	v_perm_b32 v57, v86, v85, s18
	v_addc_co_u32_e32 v60, vcc, v136, v60, vcc
	global_store_dwordx2 v[59:60], v[57:58], off
	v_add_co_u32_e32 v59, vcc, s58, v139
	v_addc_co_u32_e32 v60, vcc, v138, v87, vcc
	v_lshlrev_b64 v[59:60], 9, v[59:60]
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v58, v137, v133, s18
	v_or_b32_e32 v59, v59, v135
	v_add_co_u32_e32 v59, vcc, s28, v59
	v_perm_b32 v57, v132, v88, s18
	v_addc_co_u32_e32 v60, vcc, v136, v60, vcc
	global_store_dwordx2 v[59:60], v[57:58], off
	s_mov_b64 s[14:15], 0
	s_cbranch_execnz 4
	ds_read2_b64 v[85:88], v115 offset1:4
	ds_read2_b64 v[132:135], v116 offset1:4
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s37, s36
	v_mov_b64_e32 v[59:60], s[38:39]
	ds_read2_b64 v[140:143], v115 offset0:8 offset1:12
	ds_read2_b64 v[144:147], v116 offset0:8 offset1:12
	v_mov_b64_e32 v[57:58], s[36:37]
	v_mov_b32_e32 v139, v60
	v_mov_b32_e32 v138, v59
	v_mov_b32_e32 v137, v58
	v_mov_b32_e32 v136, v57
	s_waitcnt vmcnt(17) lgkmcnt(3)
	s_nop 0
	v_mmac_f32_16x16x16_bf16 v[136:139], v[13:14], v[85:86], v[136:139]
	s_waitcnt lgkmcnt(2)
	v_mmac_f32_16x16x16_bf16 v[57:60], v[13:14], v[132:133], v[57:60]
	v_mmac_f32_16x16x16_bf16 v[136:139], v[15:16], v[87:88], v[136:139]
	v_mmac_f32_16x16x16_bf16 v[57:60], v[15:16], v[134:135], v[57:60]
	ds_read2_b64 v[13:16], v115 offset0:16 offset1:20
	ds_read2_b64 v[85:88], v116 offset0:16 offset1:20
	s_waitcnt vmcnt(16) lgkmcnt(3)
	v_mmac_f32_16x16x16_bf16 v[136:139], v[9:10], v[140:141], v[136:139]
	s_waitcnt lgkmcnt(2)
	v_mmac_f32_16x16x16_bf16 v[57:60], v[9:10], v[144:145], v[57:60]
	v_mmac_f32_16x16x16_bf16 v[136:139], v[11:12], v[142:143], v[136:139]
	v_mmac_f32_16x16x16_bf16 v[57:60], v[11:12], v[146:147], v[57:60]
	ds_read2_b64 v[9:12], v115 offset0:24 offset1:28
	ds_read2_b64 v[132:135], v116 offset0:24 offset1:28
	s_waitcnt vmcnt(15) lgkmcnt(3)
	v_mmac_f32_16x16x16_bf16 v[136:139], v[5:6], v[13:14], v[136:139]
	s_waitcnt lgkmcnt(2)
	v_mmac_f32_16x16x16_bf16 v[57:60], v[5:6], v[85:86], v[57:60]
	v_mmac_f32_16x16x16_bf16 v[136:139], v[7:8], v[15:16], v[136:139]
	v_mmac_f32_16x16x16_bf16 v[57:60], v[7:8], v[87:88], v[57:60]
	s_lshl_b64 s[14:15], s[16:17], 13
	s_waitcnt vmcnt(14) lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[136:139], v[1:2], v[9:10], v[136:139]
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[57:60], v[1:2], v[132:133], v[57:60]
	v_mov_b32_e32 v2, s15
	v_or_b32_e32 v1, s14, v90
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	v_mmac_f32_16x16x16_bf16 v[136:139], v[3:4], v[11:12], v[136:139]
	v_add_co_u32_e32 v1, vcc, v117, v1
	v_addc_co_u32_e32 v2, vcc, v118, v2, vcc
	v_mmac_f32_16x16x16_bf16 v[57:60], v[3:4], v[134:135], v[57:60]
	global_load_dwordx4 v[13:16], v[1:2], off
	global_load_dwordx4 v[9:12], v[1:2], off offset:1024
	global_load_dwordx4 v[5:8], v[1:2], off offset:2048
	s_nop 0
	global_load_dwordx4 v[1:4], v[1:2], off offset:3072
	s_waitcnt vmcnt(17)
	v_and_b32_e32 v88, 0xffff0000, v67
	v_lshlrev_b32_e32 v87, 16, v67
	v_pk_add_f32 v[87:88], v[87:88], v[136:137]
	s_add_u32 s4, s4, s50
	v_bfe_u32 v132, v88, 16, 1
	s_addc_u32 s5, s5, s49
	v_bfe_u32 v67, v87, 16, 1
	v_add3_u32 v134, v88, v132, s19
	v_and_b32_e32 v133, 0xffff0000, v68
	v_lshlrev_b32_e32 v132, 16, v68
	s_lshl_b64 s[4:5], s[4:5], 5
	v_add3_u32 v135, v87, v67, s19
	v_pk_add_f32 v[67:68], v[132:133], v[138:139]
	v_mov_b32_e32 v86, s5
	v_add_co_u32_e32 v85, vcc, s4, v93
	v_bfe_u32 v132, v67, 16, 1
	v_addc_co_u32_e32 v86, vcc, v86, v94, vcc
	v_bfe_u32 v133, v68, 16, 1
	v_add3_u32 v137, v67, v132, s19
	v_cndmask_b32_e64 v67, 0, 1, s[42:43]
	v_add3_u32 v136, v68, v133, s19
	v_cmp_ne_u32_e64 s[4:5], 1, v67
	v_lshlrev_b64 v[67:68], 9, v[85:86]
	v_lshrrev_b32_e32 v87, 16, v135
	v_lshrrev_b32_e32 v88, 16, v134
	v_lshrrev_b32_e32 v132, 16, v137
	v_lshrrev_b32_e32 v133, 16, v136
	s_andn2_b64 vcc, exec, s[42:43]
	ds_write_b16 v119, v87
	ds_write_b16 v119, v88 offset:544
	ds_write_b16 v119, v132 offset:1088
	ds_write_b16 v119, v133 offset:1632
	s_cbranch_vccnz 8
	s_lshl_b64 s[38:39], s[6:7], 13
	v_cndmask_b32_e64 v68, 0, 1, s[44:45]
	v_or_b32_e64 v67, s39, 0
	v_cmp_ne_u32_e64 s[6:7], 1, v68
	s_andn2_b64 vcc, exec, s[44:45]
	v_or_b32_e32 v68, s38, v112
	s_cbranch_vccnz 15
	v_add_co_u32_e32 v134, vcc, v68, v122
	v_addc_co_u32_e32 v135, vcc, v67, v123, vcc
	v_lshlrev_b64 v[134:135], 1, v[134:135]
	v_mov_b32_e32 v136, s31
	v_add_co_u32_e32 v134, vcc, s30, v134
	v_addc_co_u32_e32 v135, vcc, v136, v135, vcc
	global_store_short v[134:135], v87, off
	global_store_short v[134:135], v88, off offset:128
	global_store_short v[134:135], v132, off offset:256
	global_store_short v[134:135], v133, off offset:384
	s_waitcnt vmcnt(16)
	v_and_b32_e32 v88, 0xffff0000, v61
	v_lshlrev_b32_e32 v87, 16, v61
	v_and_b32_e32 v133, 0xffff0000, v62
	v_lshlrev_b32_e32 v132, 16, v62
	v_pk_add_f32 v[57:58], v[87:88], v[57:58]
	v_pk_add_f32 v[59:60], v[132:133], v[59:60]
	v_bfe_u32 v87, v57, 16, 1
	v_bfe_u32 v61, v58, 16, 1
	v_bfe_u32 v88, v59, 16, 1
	v_bfe_u32 v62, v60, 16, 1
	v_add3_u32 v61, v58, v61, s19
	v_add3_u32 v87, v57, v87, s19
	v_add3_u32 v62, v60, v62, s19
	v_add3_u32 v88, v59, v88, s19
	v_lshrrev_b32_e32 v57, 16, v87
	v_lshrrev_b32_e32 v58, 16, v61
	v_lshrrev_b32_e32 v59, 16, v88
	v_lshrrev_b32_e32 v60, 16, v62
	s_and_b64 vcc, exec, s[4:5]
	ds_write_b16 v119, v57 offset:2176
	ds_write_b16 v119, v58 offset:2720
	ds_write_b16 v119, v59 offset:3264
	ds_write_b16 v119, v60 offset:3808
	s_cbranch_vccnz 10
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz 64713
	v_add_co_u32_e32 v61, vcc, v68, v122
	v_addc_co_u32_e32 v62, vcc, v67, v123, vcc
	v_lshlrev_b64 v[61:62], 1, v[61:62]
	v_mov_b32_e32 v67, s31
	v_add_co_u32_e32 v61, vcc, s30, v61
	v_addc_co_u32_e32 v62, vcc, v67, v62, vcc
	global_store_short v[61:62], v57, off offset:512
	global_store_short v[61:62], v58, off offset:640
	global_store_short v[61:62], v59, off offset:768
	global_store_short v[61:62], v60, off offset:896
	s_branch 64697
	s_lshl_b64 s[4:5], s[16:17], 5
	v_mov_b32_e32 v60, s5
	v_add_co_u32_e32 v57, vcc, s4, v93
	v_addc_co_u32_e32 v58, vcc, v60, v94, vcc
	v_lshlrev_b64 v[57:58], 9, v[57:58]
	v_add_co_u32_e32 v57, vcc, v124, v57
	v_addc_co_u32_e32 v58, vcc, v125, v58, vcc
	v_add_co_u32_e32 v59, vcc, s4, v95
	v_addc_co_u32_e32 v60, vcc, v60, v96, vcc
	v_lshlrev_b64 v[59:60], 9, v[59:60]
	v_add_co_u32_e32 v59, vcc, v124, v59
	v_addc_co_u32_e32 v60, vcc, v125, v60, vcc
	global_load_dwordx2 v[67:68], v[57:58], off
	global_load_dwordx2 v[61:62], v[59:60], off
	s_waitcnt vmcnt(17)
	v_mov_b32_e32 v168, v17
	s_waitcnt vmcnt(16)
	v_mov_b32_e32 v170, v21
	v_mov_b32_e32 v169, v18
	v_mov_b32_e32 v171, v22
	v_mov_b32_e32 v172, v19
	v_mov_b32_e32 v174, v23
	v_mov_b32_e32 v173, v20
	v_mov_b32_e32 v175, v24
	s_lshl_b64 s[4:5], s[16:17], 7
	v_mov_b32_e32 v18, s5
	v_or_b32_e32 v17, s4, v97
	v_lshlrev_b64 v[17:18], 2, v[17:18]
	v_mov_b32_e32 v20, s5
	v_or_b32_e32 v19, s4, v98
	v_add_co_u32_e32 v17, vcc, v126, v17
	v_lshlrev_b64 v[19:20], 2, v[19:20]
	v_addc_co_u32_e32 v18, vcc, v127, v18, vcc
	v_add_co_u32_e32 v21, vcc, v126, v19
	v_addc_co_u32_e32 v22, vcc, v127, v20, vcc
	global_load_dwordx4 v[17:20], v[17:18], off
	s_nop 0
	global_load_dwordx4 v[21:24], v[21:22], off
	s_waitcnt lgkmcnt(0)
	OTHER_MSG
	s_barrier
	v_lshlrev_b32_e32 v57, 3, v89
	v_lshl_or_b32 v57, s22, 12, v57
	v_add_u32_e32 v156, 0x100, v57
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	ds_read2st64_b64 v[57:60], v156 offset0:16 offset1:17
	ds_read2_b64 v[136:139], v128 offset1:4
	ds_read2st64_b64 v[140:143], v156 offset0:18 offset1:19
	s_mov_b32 s37, s36
	v_mov_b64_e32 v[87:88], s[38:39]
	ds_read2_b64 v[148:151], v128 offset0:8 offset1:12
	v_mov_b64_e32 v[85:86], s[36:37]
	v_mov_b32_e32 v147, v88
	v_mov_b32_e32 v135, v88
	v_mov_b32_e32 v146, v87
	v_mov_b32_e32 v145, v86
	v_mov_b32_e32 v144, v85
	v_mov_b32_e32 v134, v87
	v_mov_b32_e32 v133, v86
	v_mov_b32_e32 v132, v85
	s_waitcnt vmcnt(15) lgkmcnt(2)
	v_mmac_f32_16x16x16_bf16 v[144:147], v[136:137], v[53:54], v[144:147]
	v_mmac_f32_16x16x16_bf16 v[132:135], v[57:58], v[41:42], v[132:135]
	v_mmac_f32_16x16x16_bf16 v[144:147], v[138:139], v[55:56], v[144:147]
	v_mmac_f32_16x16x16_bf16 v[132:135], v[59:60], v[43:44], v[132:135]
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[144:147], v[148:149], v[49:50], v[144:147]
	v_mmac_f32_16x16x16_bf16 v[132:135], v[140:141], v[45:46], v[132:135]
	v_mmac_f32_16x16x16_bf16 v[144:147], v[150:151], v[51:52], v[144:147]
	v_mmac_f32_16x16x16_bf16 v[132:135], v[142:143], v[47:48], v[132:135]
	s_nop 1
	v_xor_b32_e32 v145, 0x80000000, v145
	v_xor_b32_e32 v144, 0x80000000, v144
	v_mov_b64_e32 v[176:177], s[24:25]
	v_pk_fma_f32 v[132:133], v[176:177], v[132:133], v[144:145]
	v_xor_b32_e32 v145, 0x80000000, v147
	v_xor_b32_e32 v144, 0x80000000, v146
	v_mov_b64_e32 v[178:179], s[46:47]
	v_pk_fma_f32 v[134:135], v[178:179], v[134:135], v[144:145]
	v_add_u32_e32 v164, 0x800, v128
	v_pk_fma_f32 v[75:76], v[75:76], v[172:173], v[134:135]
	v_pk_fma_f32 v[73:74], v[73:74], v[168:169], v[132:133]
	ds_read2st64_b64 v[132:135], v156 offset0:20 offset1:21
	ds_read2_b64 v[152:155], v164 offset0:16 offset1:20
	ds_read2st64_b64 v[156:159], v156 offset0:22 offset1:23
	ds_read2_b64 v[164:167], v164 offset0:24 offset1:28
	v_mov_b32_e32 v163, v88
	v_mov_b32_e32 v147, v88
	v_mov_b32_e32 v162, v87
	v_mov_b32_e32 v161, v86
	v_mov_b32_e32 v160, v85
	v_mov_b32_e32 v146, v87
	v_mov_b32_e32 v145, v86
	v_mov_b32_e32 v144, v85
	s_waitcnt lgkmcnt(2)
	v_mmac_f32_16x16x16_bf16 v[160:163], v[152:153], v[53:54], v[160:163]
	v_mmac_f32_16x16x16_bf16 v[144:147], v[132:133], v[41:42], v[144:147]
	v_mmac_f32_16x16x16_bf16 v[160:163], v[154:155], v[55:56], v[160:163]
	v_mmac_f32_16x16x16_bf16 v[144:147], v[134:135], v[43:44], v[144:147]
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[160:163], v[164:165], v[49:50], v[160:163]
	v_mmac_f32_16x16x16_bf16 v[144:147], v[156:157], v[45:46], v[144:147]
	v_mmac_f32_16x16x16_bf16 v[160:163], v[166:167], v[51:52], v[160:163]
	v_mmac_f32_16x16x16_bf16 v[144:147], v[158:159], v[47:48], v[144:147]
	s_nop 1
	v_xor_b32_e32 v42, 0x80000000, v161
	v_xor_b32_e32 v41, 0x80000000, v160
	v_xor_b32_e32 v44, 0x80000000, v163
	v_xor_b32_e32 v43, 0x80000000, v162
	v_pk_fma_f32 v[41:42], v[176:177], v[144:145], v[41:42]
	v_pk_fma_f32 v[43:44], v[178:179], v[146:147], v[43:44]
	v_mov_b64_e32 v[45:46], v[85:86]
	v_pk_fma_f32 v[71:72], v[71:72], v[172:173], v[43:44]
	v_pk_fma_f32 v[69:70], v[69:70], v[168:169], v[41:42]
	v_mov_b64_e32 v[41:42], v[85:86]
	v_mov_b64_e32 v[47:48], v[87:88]
	v_mov_b64_e32 v[43:44], v[87:88]
	s_waitcnt vmcnt(11)
	v_mmac_f32_16x16x16_bf16 v[45:48], v[136:137], v[37:38], v[45:48]
	v_mmac_f32_16x16x16_bf16 v[41:44], v[57:58], v[25:26], v[41:44]
	v_mmac_f32_16x16x16_bf16 v[45:48], v[138:139], v[39:40], v[45:48]
	v_mmac_f32_16x16x16_bf16 v[41:44], v[59:60], v[27:28], v[41:44]
	s_waitcnt vmcnt(10)
	v_mmac_f32_16x16x16_bf16 v[45:48], v[148:149], v[33:34], v[45:48]
	v_mmac_f32_16x16x16_bf16 v[41:44], v[140:141], v[29:30], v[41:44]
	v_mmac_f32_16x16x16_bf16 v[45:48], v[150:151], v[35:36], v[45:48]
	v_mmac_f32_16x16x16_bf16 v[41:44], v[142:143], v[31:32], v[41:44]
	s_nop 1
	v_xor_b32_e32 v46, 0x80000000, v46
	v_xor_b32_e32 v45, 0x80000000, v45
	v_pk_fma_f32 v[41:42], v[176:177], v[41:42], v[45:46]
	v_xor_b32_e32 v46, 0x80000000, v48
	v_xor_b32_e32 v45, 0x80000000, v47
	v_pk_fma_f32 v[43:44], v[178:179], v[43:44], v[45:46]
	v_pk_fma_f32 v[83:84], v[83:84], v[174:175], v[43:44]
	v_pk_fma_f32 v[81:82], v[81:82], v[170:171], v[41:42]
	v_mov_b64_e32 v[41:42], v[85:86]
	v_mov_b64_e32 v[43:44], v[87:88]
	v_mmac_f32_16x16x16_bf16 v[85:88], v[152:153], v[37:38], v[85:88]
	s_nop 0
	v_mmac_f32_16x16x16_bf16 v[41:44], v[132:133], v[25:26], v[41:44]
	v_mmac_f32_16x16x16_bf16 v[85:88], v[154:155], v[39:40], v[85:88]
	v_mmac_f32_16x16x16_bf16 v[41:44], v[134:135], v[27:28], v[41:44]
	v_mmac_f32_16x16x16_bf16 v[85:88], v[164:165], v[33:34], v[85:88]
	v_mmac_f32_16x16x16_bf16 v[41:44], v[156:157], v[29:30], v[41:44]
	v_mmac_f32_16x16x16_bf16 v[85:88], v[166:167], v[35:36], v[85:88]
	v_mmac_f32_16x16x16_bf16 v[41:44], v[158:159], v[31:32], v[41:44]
	s_nop 1
	v_xor_b32_e32 v26, 0x80000000, v86
	v_xor_b32_e32 v25, 0x80000000, v85
	v_pk_fma_f32 v[57:58], v[176:177], v[41:42], v[25:26]
	v_xor_b32_e32 v26, 0x80000000, v88
	v_xor_b32_e32 v25, 0x80000000, v87
	v_pk_fma_f32 v[25:26], v[178:179], v[43:44], v[25:26]
	v_pk_fma_f32 v[79:80], v[79:80], v[174:175], v[25:26]
	v_mov_b32_e32 v29, s15
	v_add_co_u32_e32 v25, vcc, s14, v99
	v_addc_co_u32_e32 v26, vcc, 0, v29, vcc
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	v_mov_b32_e32 v30, s11
	v_add_co_u32_e32 v27, vcc, s10, v25
	v_addc_co_u32_e32 v28, vcc, v30, v26, vcc
	v_lshlrev_b32_e32 v31, 1, v91
	v_add_co_u32_e32 v27, vcc, v27, v31
	v_addc_co_u32_e32 v28, vcc, 0, v28, vcc
	v_mov_b32_e32 v32, s13
	v_add_co_u32_e32 v25, vcc, s12, v25
	v_addc_co_u32_e32 v26, vcc, v32, v26, vcc
	v_add_co_u32_e32 v25, vcc, v25, v31
	v_addc_co_u32_e32 v26, vcc, 0, v26, vcc
	global_load_dwordx4 v[41:44], v[27:28], off
	global_load_dwordx4 v[45:48], v[27:28], off offset:1024
	global_load_dwordx4 v[53:56], v[25:26], off
	global_load_dwordx4 v[49:52], v[25:26], off offset:1024
	v_add_co_u32_e32 v25, vcc, s14, v100
	v_addc_co_u32_e32 v26, vcc, 0, v29, vcc
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	v_pk_fma_f32 v[77:78], v[77:78], v[170:171], v[57:58]
	v_add_co_u32_e32 v27, vcc, s10, v25
	v_addc_co_u32_e32 v28, vcc, v30, v26, vcc
	v_add_co_u32_e32 v33, vcc, v27, v31
	v_addc_co_u32_e32 v34, vcc, 0, v28, vcc
	v_add_co_u32_e32 v25, vcc, s12, v25
	v_addc_co_u32_e32 v26, vcc, v32, v26, vcc
	v_add_co_u32_e32 v35, vcc, v25, v31
	v_addc_co_u32_e32 v36, vcc, 0, v26, vcc
	global_load_dwordx4 v[25:28], v[33:34], off
	global_load_dwordx4 v[29:32], v[33:34], off offset:1024
	global_load_dwordx4 v[37:40], v[35:36], off
	s_nop 0
	global_load_dwordx4 v[33:36], v[35:36], off offset:1024
	v_bfe_u32 v57, v73, 16, 1
	v_bfe_u32 v58, v74, 16, 1
	v_add3_u32 v58, v74, v58, s19
	v_add3_u32 v57, v73, v57, s19
	v_lshrrev_b32_e32 v59, 16, v57
	v_lshrrev_b32_e32 v60, 16, v58
	v_perm_b32 v57, v58, v57, s20
	v_bfe_u32 v58, v75, 16, 1
	v_add3_u32 v58, v75, v58, s19
	ds_write_b16 v102, v59
	v_lshrrev_b32_e32 v59, 16, v58
	ds_write_b16 v102, v59 offset:16
	v_bfe_u32 v59, v76, 16, 1
	v_add3_u32 v59, v76, v59, s19
	ds_write_b16 v102, v60 offset:8
	v_lshrrev_b32_e32 v60, 16, v59
	v_perm_b32 v58, v59, v58, s20
	ds_write_b16 v102, v60 offset:24
	v_bfe_u32 v59, v69, 16, 1
	v_bfe_u32 v60, v70, 16, 1
	v_add3_u32 v60, v70, v60, s19
	v_add3_u32 v59, v69, v59, s19
	v_lshrrev_b32_e32 v85, 16, v59
	v_lshrrev_b32_e32 v86, 16, v60
	v_perm_b32 v59, v60, v59, s20
	v_bfe_u32 v60, v71, 16, 1
	v_add3_u32 v60, v71, v60, s19
	ds_write_b16 v102, v85 offset:4224
	v_lshrrev_b32_e32 v85, 16, v60
	ds_write_b16 v103, v85 offset:16
	v_bfe_u32 v85, v72, 16, 1
	v_add3_u32 v85, v72, v85, s19
	ds_write_b16 v103, v86 offset:8
	v_lshrrev_b32_e32 v86, 16, v85
	v_perm_b32 v60, v85, v60, s20
	ds_write_b16 v103, v86 offset:24
	v_bfe_u32 v85, v81, 16, 1
	v_bfe_u32 v86, v82, 16, 1
	v_add3_u32 v86, v82, v86, s19
	v_add3_u32 v85, v81, v85, s19
	v_lshrrev_b32_e32 v87, 16, v85
	v_lshrrev_b32_e32 v88, 16, v86
	v_perm_b32 v85, v86, v85, s20
	v_bfe_u32 v86, v83, 16, 1
	v_add3_u32 v86, v83, v86, s19
	ds_write_b16 v104, v87
	v_lshrrev_b32_e32 v87, 16, v86
	ds_write_b16 v104, v87 offset:16
	v_bfe_u32 v87, v84, 16, 1
	v_add3_u32 v87, v84, v87, s19
	ds_write_b16 v104, v88 offset:8
	v_lshrrev_b32_e32 v88, 16, v87
	v_perm_b32 v86, v87, v86, s20
	ds_write_b16 v104, v88 offset:24
	v_bfe_u32 v87, v77, 16, 1
	v_bfe_u32 v88, v78, 16, 1
	v_add3_u32 v88, v78, v88, s19
	v_add3_u32 v87, v77, v87, s19
	v_lshrrev_b32_e32 v132, 16, v87
	v_lshrrev_b32_e32 v133, 16, v88
	v_perm_b32 v87, v88, v87, s20
	v_bfe_u32 v88, v79, 16, 1
	s_add_i32 s21, s21, 1
	v_add3_u32 v88, v79, v88, s19
	s_add_u32 s8, s8, -1
	ds_write_b16 v104, v132 offset:4224
	v_lshrrev_b32_e32 v132, 16, v88
	s_addc_u32 s9, s9, -1
	ds_write_b16 v105, v132 offset:16
	v_bfe_u32 v132, v80, 16, 1
	s_add_u32 s40, s40, 0xffff8000
	v_add3_u32 v132, v80, v132, s19
	s_addc_u32 s41, s41, -1
	ds_write_b16 v105, v133 offset:8
	v_lshrrev_b32_e32 v133, 16, v132
	v_perm_b32 v88, v132, v88, s20
	s_cmp_lt_i32 s21, s55
	ds_write_b16 v105, v133 offset:24
	s_cbranch_scc0 421
	s_add_u32 s16, s8, s33
	s_add_i32 s6, s16, -1
	s_mov_b64 s[14:15], -1
	s_mov_b64 vcc, s[0:1]
	s_cbranch_vccz 17
	s_andn2_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz 2
	s_add_u32 s4, s8, s51
	s_addc_u32 s5, s9, s52
	s_cbranch_scc1 .LBBX
