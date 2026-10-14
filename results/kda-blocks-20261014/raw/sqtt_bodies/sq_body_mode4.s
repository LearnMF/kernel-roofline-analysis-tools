.LBBX:  ; =>This Inner Loop Header: Depth=1
	s_add_i32 s18, s16, 1
	s_cmp_lt_i32 s18, s17
	s_cselect_b64 s[0:1], -1, 0
	s_waitcnt lgkmcnt(0)
	OTHER_MSG
	s_barrier
	ds_read_u16 v97, v83
	ds_read_u16 v99, v79 offset:264
	ds_read_u16 v98, v79 offset:528
	ds_read_u16 v100, v79 offset:792
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v97, v99, v97, s12
	v_add_co_u32_e32 v99, vcc, v75, v82
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v98, v100, v98, s12
	v_addc_co_u32_e32 v100, vcc, 0, v76, vcc
	global_store_dwordx2 v[99:100], v[97:98], off
	ds_read_u16 v97, v79 offset:4224
	ds_read_u16 v101, v79 offset:4488
	ds_read_u16 v98, v79 offset:4752
	ds_read_u16 v102, v79 offset:5016
	s_mov_b32 s5, s4
	s_and_b64 s[0:1], s[0:1], exec
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v97, v101, v97, s12
	s_cselect_b32 s2, s18, s16
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v98, v102, v98, s12
	global_store_dwordx2 v[99:100], v[97:98], off offset:512
	ds_read_u16 v97, v84
	ds_read_u16 v99, v80 offset:264
	ds_read_u16 v98, v80 offset:528
	ds_read_u16 v100, v80 offset:792
	s_ashr_i32 s3, s2, 31
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v97, v99, v97, s12
	v_add_co_u32_e32 v99, vcc, v93, v82
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v98, v100, v98, s12
	v_addc_co_u32_e32 v100, vcc, 0, v94, vcc
	v_add_co_u32_e32 v99, vcc, s13, v99
	v_addc_co_u32_e32 v100, vcc, 0, v100, vcc
	global_store_dwordx2 v[99:100], v[97:98], off
	ds_read_u16 v97, v80 offset:4224
	ds_read_u16 v101, v80 offset:4488
	ds_read_u16 v98, v80 offset:4752
	ds_read_u16 v102, v80 offset:5016
	ds_read2_b64 v[109:112], v95 offset1:4
	s_waitcnt lgkmcnt(3)
	v_perm_b32 v97, v101, v97, s12
	s_waitcnt lgkmcnt(1)
	v_perm_b32 v98, v102, v98, s12
	global_store_dwordx2 v[99:100], v[97:98], off offset:512
	ds_read2_b64 v[97:100], v81 offset1:4
	v_mov_b64_e32 v[103:104], s[6:7]
	v_mov_b64_e32 v[101:102], s[4:5]
	v_mov_b64_e32 v[107:108], v[103:104]
	v_mov_b64_e32 v[115:116], v[103:104]
	v_mov_b64_e32 v[105:106], v[101:102]
	v_mov_b64_e32 v[113:114], v[101:102]
	s_waitcnt vmcnt(16) lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[105:108], v[20:21], v[97:98], v[105:108]
	v_mmac_f32_16x16x16_bf16 v[113:116], v[20:21], v[109:110], v[113:116]
	v_mmac_f32_16x16x16_bf16 v[105:108], v[22:23], v[99:100], v[105:108]
	v_mmac_f32_16x16x16_bf16 v[113:116], v[22:23], v[111:112], v[113:116]
	ds_read2_b64 v[20:23], v81 offset0:8 offset1:12
	ds_read2_b64 v[97:100], v95 offset0:8 offset1:12
	s_add_u32 s6, s2, s21
	s_addc_u32 s7, s3, s20
	s_lshl_b64 s[2:3], s[6:7], 13
	s_waitcnt vmcnt(15) lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[105:108], v[16:17], v[20:21], v[105:108]
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[113:116], v[16:17], v[97:98], v[113:116]
	v_mmac_f32_16x16x16_bf16 v[105:108], v[18:19], v[22:23], v[105:108]
	v_mmac_f32_16x16x16_bf16 v[113:116], v[18:19], v[99:100], v[113:116]
	ds_read2_b64 v[16:19], v81 offset0:16 offset1:20
	ds_read2_b64 v[20:23], v95 offset0:16 offset1:20
	v_or_b32_e64 v118, s3, 0
	v_or_b32_e32 v117, s2, v62
	s_waitcnt vmcnt(14) lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[105:108], v[12:13], v[16:17], v[105:108]
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[113:116], v[12:13], v[20:21], v[113:116]
	v_mmac_f32_16x16x16_bf16 v[105:108], v[14:15], v[18:19], v[105:108]
	v_mmac_f32_16x16x16_bf16 v[113:116], v[14:15], v[22:23], v[113:116]
	ds_read2_b64 v[12:15], v81 offset0:24 offset1:28
	ds_read2_b64 v[16:19], v95 offset0:24 offset1:28
	s_waitcnt vmcnt(13) lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[105:108], v[8:9], v[12:13], v[105:108]
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[113:116], v[8:9], v[16:17], v[113:116]
	v_lshlrev_b64 v[8:9], 1, v[117:118]
	v_mmac_f32_16x16x16_bf16 v[105:108], v[10:11], v[14:15], v[105:108]
	v_add_co_u32_e32 v97, vcc, v63, v8
	v_addc_co_u32_e32 v98, vcc, v64, v9, vcc
	v_mmac_f32_16x16x16_bf16 v[113:116], v[10:11], v[18:19], v[113:116]
	global_load_dwordx4 v[20:23], v[97:98], off
	global_load_dwordx4 v[16:19], v[97:98], off offset:1024
	global_load_dwordx4 v[12:15], v[97:98], off offset:2048
	global_load_dwordx4 v[8:11], v[97:98], off offset:3072
	s_waitcnt vmcnt(14)
	v_and_b32_e32 v97, 0xffff0000, v24
	v_lshlrev_b32_e32 v24, 16, v24
	v_sub_f32_e32 v24, v24, v105
	v_bfe_u32 v98, v24, 16, 1
	v_add3_u32 v24, v24, v98, s14
	v_lshrrev_b32_e32 v98, 16, v24
	ds_write_b16 v87, v98
	v_and_b32_e32 v98, 0xffff0000, v26
	v_lshlrev_b32_e32 v26, 16, v26
	v_sub_f32_e32 v97, v97, v106
	v_sub_f32_e32 v26, v26, v113
	v_bfe_u32 v99, v97, 16, 1
	v_sub_f32_e32 v98, v98, v114
	v_bfe_u32 v100, v26, 16, 1
	v_add3_u32 v97, v97, v99, s14
	v_bfe_u32 v105, v98, 16, 1
	v_add3_u32 v26, v26, v100, s14
	v_lshrrev_b32_e32 v99, 16, v97
	v_add3_u32 v98, v98, v105, s14
	v_lshrrev_b32_e32 v100, 16, v26
	v_lshrrev_b32_e32 v105, 16, v98
	ds_write_b16 v87, v100 offset:2176
	ds_write_b16 v87, v99 offset:544
	ds_write_b16 v88, v105 offset:2176
	v_and_b32_e32 v99, 0xffff0000, v25
	v_lshlrev_b32_e32 v25, 16, v25
	v_sub_f32_e32 v25, v25, v107
	v_sub_f32_e32 v99, v99, v108
	v_bfe_u32 v100, v25, 16, 1
	v_bfe_u32 v105, v99, 16, 1
	v_add3_u32 v25, v25, v100, s14
	v_perm_b32 v24, v97, v24, s15
	v_and_b32_e32 v97, 0xffff0000, v27
	v_lshlrev_b32_e32 v27, 16, v27
	v_add3_u32 v99, v99, v105, s14
	v_lshrrev_b32_e32 v100, 16, v25
	v_sub_f32_e32 v97, v97, v116
	v_sub_f32_e32 v27, v27, v115
	v_lshrrev_b32_e32 v105, 16, v99
	ds_write_b16 v87, v100 offset:1088
	v_perm_b32 v25, v99, v25, s15
	v_bfe_u32 v99, v27, 16, 1
	v_bfe_u32 v100, v97, 16, 1
	v_add3_u32 v97, v97, v100, s14
	v_add3_u32 v27, v27, v99, s14
	v_lshrrev_b32_e32 v99, 16, v27
	v_lshrrev_b32_e32 v100, 16, v97
	v_perm_b32 v27, v97, v27, s15
	v_add_co_u32_e32 v97, vcc, v77, v82
	v_perm_b32 v26, v98, v26, s15
	v_addc_co_u32_e32 v98, vcc, 0, v78, vcc
	ds_write_b16 v89, v99 offset:2176
	ds_write_b16 v87, v105 offset:1632
	ds_write_b16 v90, v100 offset:2176
	global_store_dwordx2 v[97:98], v[24:25], off
	global_store_dwordx2 v[97:98], v[26:27], off offset:512
	v_add_co_u32_e32 v24, vcc, s2, v46
	v_mov_b32_e32 v25, s3
	v_addc_co_u32_e32 v25, vcc, v25, v47, vcc
	v_lshlrev_b64 v[24:25], 1, v[24:25]
	v_add_co_u32_e32 v24, vcc, v65, v24
	v_addc_co_u32_e32 v25, vcc, v66, v25, vcc
	global_load_dwordx4 v[24:27], v[24:25], off
	s_waitcnt vmcnt(16)
	v_mov_b32_e32 v119, v0
	s_waitcnt vmcnt(15)
	v_mov_b32_e32 v121, v4
	v_mov_b32_e32 v120, v1
	v_mov_b32_e32 v122, v5
	v_mov_b32_e32 v123, v2
	v_mov_b32_e32 v125, v6
	v_mov_b32_e32 v124, v3
	v_mov_b32_e32 v126, v7
	s_lshl_b64 s[6:7], s[6:7], 7
	v_or_b32_e64 v1, s7, 0
	v_or_b32_e32 v0, s6, v67
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_add_co_u32_e32 v97, vcc, v68, v0
	v_addc_co_u32_e32 v98, vcc, v69, v1, vcc
	v_or_b32_e64 v1, s7, 0
	v_or_b32_e32 v0, s6, v70
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_add_co_u32_e32 v99, vcc, v68, v0
	v_addc_co_u32_e32 v100, vcc, v69, v1, vcc
	global_load_dwordx4 v[0:3], v[97:98], off
	global_load_dwordx4 v[4:7], v[99:100], off
	s_waitcnt lgkmcnt(0)
	OTHER_MSG
	s_barrier
	ds_read2_b64 v[97:100], v92 offset1:4
	ds_read2_b64 v[109:112], v92 offset0:8 offset1:12
	v_mov_b64_e32 v[107:108], v[103:104]
	v_mov_b64_e32 v[105:106], v[101:102]
	v_mov_b64_e32 v[115:116], v[103:104]
	s_waitcnt vmcnt(16) lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[105:108], v[97:98], v[32:33], v[105:108]
	v_mmac_f32_16x16x16_bf16 v[105:108], v[99:100], v[34:35], v[105:108]
	s_waitcnt vmcnt(15) lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[105:108], v[109:110], v[28:29], v[105:108]
	v_mmac_f32_16x16x16_bf16 v[105:108], v[111:112], v[30:31], v[105:108]
	s_nop 2
	v_pk_fma_f32 v[52:53], v[52:53], v[123:124], v[107:108]
	v_pk_fma_f32 v[50:51], v[50:51], v[119:120], v[105:106]
	ds_read2_b64 v[105:108], v91 offset1:4
	v_mov_b64_e32 v[113:114], v[101:102]
	s_mov_b32 s16, s18
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[113:116], v[105:106], v[32:33], v[113:116]
	v_mmac_f32_16x16x16_bf16 v[113:116], v[107:108], v[34:35], v[113:116]
	ds_read2_b64 v[32:35], v91 offset0:8 offset1:12
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[113:116], v[32:33], v[28:29], v[113:116]
	v_mmac_f32_16x16x16_bf16 v[113:116], v[34:35], v[30:31], v[113:116]
	v_mov_b32_e32 v28, v101
	v_mov_b32_e32 v29, v102
	v_mov_b32_e32 v30, v103
	v_mov_b32_e32 v31, v104
	s_waitcnt vmcnt(14)
	s_nop 0
	v_mmac_f32_16x16x16_bf16 v[28:31], v[97:98], v[36:37], v[28:31]
	v_mmac_f32_16x16x16_bf16 v[28:31], v[99:100], v[38:39], v[28:31]
	s_waitcnt vmcnt(13)
	v_mmac_f32_16x16x16_bf16 v[28:31], v[109:110], v[40:41], v[28:31]
	v_mmac_f32_16x16x16_bf16 v[28:31], v[111:112], v[42:43], v[28:31]
	s_nop 2
	v_pk_fma_f32 v[58:59], v[58:59], v[121:122], v[28:29]
	v_sub_co_u32_e32 v28, vcc, v117, v71
	v_subbrev_co_u32_e32 v29, vcc, 0, v118, vcc
	v_lshlrev_b64 v[28:29], 1, v[28:29]
	v_mmac_f32_16x16x16_bf16 v[101:104], v[105:106], v[36:37], v[101:104]
	v_mmac_f32_16x16x16_bf16 v[101:104], v[107:108], v[38:39], v[101:104]
	v_add_co_u32_e32 v28, vcc, s10, v28
	v_mov_b32_e32 v38, s11
	v_addc_co_u32_e32 v29, vcc, v38, v29, vcc
	v_add_co_u32_e32 v36, vcc, v28, v96
	v_mmac_f32_16x16x16_bf16 v[101:104], v[32:33], v[40:41], v[101:104]
	v_addc_co_u32_e32 v37, vcc, 0, v29, vcc
	v_pk_fma_f32 v[60:61], v[60:61], v[125:126], v[30:31]
	v_mmac_f32_16x16x16_bf16 v[101:104], v[34:35], v[42:43], v[101:104]
	global_load_dwordx4 v[32:35], v[36:37], off
	global_load_dwordx4 v[28:31], v[36:37], off offset:1024
	v_or_b32_e64 v37, s3, 0
	v_or_b32_e32 v36, s2, v72
	v_lshlrev_b64 v[36:37], 1, v[36:37]
	v_pk_fma_f32 v[44:45], v[44:45], v[119:120], v[113:114]
	v_add_co_u32_e32 v36, vcc, s10, v36
	v_addc_co_u32_e32 v37, vcc, v38, v37, vcc
	v_add_co_u32_e32 v97, vcc, v36, v96
	v_addc_co_u32_e32 v98, vcc, 0, v37, vcc
	global_load_dwordx4 v[36:39], v[97:98], off
	global_load_dwordx4 v[40:43], v[97:98], off offset:1024
	v_bfe_u32 v97, v50, 16, 1
	v_add3_u32 v97, v50, v97, s14
	ds_write_b16_d16_hi v73, v97
	v_bfe_u32 v97, v51, 16, 1
	v_add3_u32 v97, v51, v97, s14
	ds_write_b16_d16_hi v73, v97 offset:8
	v_bfe_u32 v97, v52, 16, 1
	v_add3_u32 v97, v52, v97, s14
	ds_write_b16_d16_hi v73, v97 offset:16
	v_bfe_u32 v97, v53, 16, 1
	v_add3_u32 v97, v53, v97, s14
	ds_write_b16_d16_hi v73, v97 offset:24
	v_bfe_u32 v97, v44, 16, 1
	v_add3_u32 v97, v44, v97, s14
	ds_write_b16_d16_hi v73, v97 offset:4224
	v_bfe_u32 v97, v45, 16, 1
	v_pk_fma_f32 v[48:49], v[48:49], v[123:124], v[115:116]
	v_add3_u32 v97, v45, v97, s14
	ds_write_b16_d16_hi v85, v97 offset:4224
	v_bfe_u32 v97, v48, 16, 1
	v_add3_u32 v97, v48, v97, s14
	ds_write_b16_d16_hi v73, v97 offset:4240
	v_bfe_u32 v97, v49, 16, 1
	v_add3_u32 v97, v49, v97, s14
	ds_write_b16_d16_hi v73, v97 offset:4248
	v_bfe_u32 v97, v58, 16, 1
	v_add3_u32 v97, v58, v97, s14
	ds_write_b16_d16_hi v74, v97
	v_bfe_u32 v97, v59, 16, 1
	v_add3_u32 v97, v59, v97, s14
	ds_write_b16_d16_hi v74, v97 offset:8
	v_bfe_u32 v97, v60, 16, 1
	v_add3_u32 v97, v60, v97, s14
	ds_write_b16_d16_hi v74, v97 offset:16
	v_bfe_u32 v97, v61, 16, 1
	v_pk_fma_f32 v[54:55], v[54:55], v[121:122], v[101:102]
	v_add3_u32 v97, v61, v97, s14
	ds_write_b16_d16_hi v74, v97 offset:24
	v_bfe_u32 v97, v54, 16, 1
	v_add3_u32 v97, v54, v97, s14
	ds_write_b16_d16_hi v86, v97 offset:4224
	v_bfe_u32 v97, v55, 16, 1
	v_pk_fma_f32 v[56:57], v[56:57], v[125:126], v[103:104]
	v_add3_u32 v97, v55, v97, s14
	ds_write_b16_d16_hi v86, v97 offset:4232
	v_bfe_u32 v97, v56, 16, 1
	v_add3_u32 v97, v56, v97, s14
	ds_write_b16_d16_hi v74, v97 offset:4240
	v_bfe_u32 v97, v57, 16, 1
	v_add3_u32 v97, v57, v97, s14
	ds_write_b16_d16_hi v86, v97 offset:4248
	v_add_co_u32_e32 v75, vcc, s8, v75
	v_mov_b32_e32 v97, s9
	v_addc_co_u32_e32 v76, vcc, v76, v97, vcc
	v_add_co_u32_e32 v77, vcc, 0x4000, v77
	v_addc_co_u32_e32 v78, vcc, 0, v78, vcc
	v_add_co_u32_e32 v93, vcc, s8, v93
	v_addc_co_u32_e32 v94, vcc, v94, v97, vcc
	s_mov_b64 vcc, s[0:1]
	s_cbranch_vccnz 65039
	s_cbranch_scc1 .LBBX
