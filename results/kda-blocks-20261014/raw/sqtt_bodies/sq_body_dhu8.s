.LBBX:  ; =>This Inner Loop Header: Depth=1
	s_add_i32 s0, s8, 1
	s_cmp_gt_i32 s0, s24
	s_cselect_b32 s0, s8, s24
	s_and_b32 s1, s22, 1
	s_lshl_b32 s1, s1, 12
	s_ashr_i32 s3, s0, 31
	s_or_b32 s23, s1, 0x2100
	s_add_u32 s2, s0, s29
	s_addc_u32 s3, s3, s28
	v_lshl_add_u32 v79, v0, 3, s23
	s_lshl_b64 s[0:1], s[2:3], 13
	s_waitcnt vmcnt(13)
	ds_write_b64 v79, v[43:44]
	v_mov_b32_e32 v95, s1
	v_add_co_u32_e32 v43, vcc, s0, v39
	v_addc_co_u32_e32 v44, vcc, v95, v40, vcc
	v_add_co_u32_e32 v43, vcc, v43, v60
	v_addc_co_u32_e32 v44, vcc, 0, v44, vcc
	v_lshlrev_b64 v[43:44], 1, v[43:44]
	v_add_co_u32_e32 v43, vcc, v66, v43
	v_addc_co_u32_e32 v44, vcc, v67, v44, vcc
	global_load_dwordx2 v[43:44], v[43:44], off
	s_waitcnt lgkmcnt(0)
	OTHER_MSG
	s_barrier
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_waitcnt vmcnt(11)
	v_lshlrev_b32_e32 v113, 16, v37
	v_and_b32_e32 v114, 0xffff0000, v37
	v_add_co_u32_e32 v37, vcc, s0, v57
	s_mov_b32 s5, s4
	v_or_b32_e64 v100, s1, 0
	v_or_b32_e32 v99, s0, v56
	v_lshlrev_b32_e32 v115, 16, v38
	v_and_b32_e32 v116, 0xffff0000, v38
	v_addc_co_u32_e32 v38, vcc, v95, v58, vcc
	v_mov_b64_e32 v[97:98], s[6:7]
	ds_read_u16 v105, v63
	ds_read_u16 v107, v63 offset:264
	ds_read_u16 v106, v63 offset:528
	ds_read_u16 v108, v63 offset:792
	ds_read_u16 v109, v63 offset:4224
	ds_read_u16 v110, v63 offset:4488
	ds_read_u16 v111, v63 offset:4752
	ds_read_u16 v112, v63 offset:5016
	ds_read2_b64 v[79:82], v68 offset1:4
	ds_read2_b64 v[83:86], v68 offset0:8 offset1:12
	ds_read2_b64 v[87:90], v68 offset0:16 offset1:20
	ds_read2_b64 v[91:94], v68 offset0:24 offset1:28
	v_mov_b64_e32 v[95:96], s[4:5]
	v_lshlrev_b64 v[103:104], 1, v[99:100]
	v_mov_b64_e32 v[101:102], v[97:98]
	v_mov_b64_e32 v[99:100], v[95:96]
	s_waitcnt vmcnt(10) lgkmcnt(3)
	s_nop 0
	v_mmac_f32_16x16x16_bf16 v[99:102], v[5:6], v[79:80], v[99:102]
	v_mmac_f32_16x16x16_bf16 v[99:102], v[7:8], v[81:82], v[99:102]
	s_waitcnt vmcnt(9) lgkmcnt(2)
	v_mmac_f32_16x16x16_bf16 v[99:102], v[1:2], v[83:84], v[99:102]
	v_lshlrev_b64 v[37:38], 1, v[37:38]
	v_add_co_u32_e32 v103, vcc, v70, v103
	v_mmac_f32_16x16x16_bf16 v[99:102], v[3:4], v[85:86], v[99:102]
	v_addc_co_u32_e32 v104, vcc, v71, v104, vcc
	s_waitcnt vmcnt(8) lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[99:102], v[9:10], v[87:88], v[99:102]
	v_add_co_u32_e32 v37, vcc, v73, v37
	v_perm_b32 v106, v108, v106, s30
	v_perm_b32 v105, v107, v105, s30
	v_mmac_f32_16x16x16_bf16 v[99:102], v[11:12], v[89:90], v[99:102]
	v_addc_co_u32_e32 v38, vcc, v74, v38, vcc
	v_perm_b32 v108, v112, v111, s30
	v_perm_b32 v107, v110, v109, s30
	global_store_dwordx2 v[47:48], v[105:106], off
	global_store_dwordx2 v[47:48], v[107:108], off offset:512
	s_waitcnt vmcnt(9) lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[99:102], v[13:14], v[91:92], v[99:102]
	global_load_dwordx2 v[37:38], v[37:38], off
	s_nop 0
	global_load_dwordx4 v[5:8], v[103:104], off
	global_load_dwordx4 v[1:4], v[103:104], off offset:1024
	global_load_dwordx4 v[9:12], v[103:104], off offset:2048
	v_mmac_f32_16x16x16_bf16 v[99:102], v[15:16], v[93:94], v[99:102]
	global_load_dwordx4 v[13:16], v[103:104], off offset:3072
	s_nop 1
	v_add_f32_e32 v79, v113, v99
	v_add_f32_e32 v80, v114, v100
	v_add_f32_e32 v81, v115, v101
	v_add_f32_e32 v82, v116, v102
	v_bfe_u32 v83, v79, 16, 1
	v_bfe_u32 v84, v80, 16, 1
	v_bfe_u32 v85, v81, 16, 1
	v_bfe_u32 v86, v82, 16, 1
	v_add3_u32 v79, v79, v83, s21
	v_add3_u32 v80, v80, v84, s21
	v_add3_u32 v81, v81, v85, s21
	v_add3_u32 v82, v82, v86, s21
	ds_write_b16_d16_hi v72, v79
	ds_write_b16_d16_hi v72, v80 offset:544
	ds_write_b16_d16_hi v72, v81 offset:1088
	ds_write_b16_d16_hi v72, v82 offset:1632
	s_waitcnt vmcnt(13)
	v_mov_b32_e32 v111, v17
	v_mov_b32_e32 v112, v18
	v_mov_b32_e32 v113, v19
	v_mov_b32_e32 v114, v20
	s_lshl_b64 s[2:3], s[2:3], 9
	v_lshl_or_b32 v17, v61, 2, s2
	v_mov_b32_e32 v18, s3
	v_add_co_u32_e32 v17, vcc, v75, v17
	v_addc_co_u32_e32 v18, vcc, v76, v18, vcc
	global_load_dwordx4 v[17:20], v[17:18], off
	s_waitcnt lgkmcnt(0)
	OTHER_MSG
	s_barrier
	v_lshl_add_u32 v109, v55, 3, s23
	ds_read_u16 v116, v77
	ds_read_u16 v117, v77 offset:136
	ds_read_u16 v118, v77 offset:272
	ds_read_u16 v119, v77 offset:408
	ds_read2st64_b64 v[103:106], v109 offset1:1
	v_mov_b64_e32 v[79:80], v[95:96]
	v_mov_b64_e32 v[81:82], v[97:98]
	v_mov_b64_e32 v[101:102], v[97:98]
	v_mov_b64_e32 v[99:100], v[95:96]
	s_waitcnt vmcnt(11) lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[79:82], v[103:104], v[33:34], v[79:82]
	v_mmac_f32_16x16x16_bf16 v[79:82], v[105:106], v[35:36], v[79:82]
	ds_read2st64_b64 v[103:106], v109 offset0:4 offset1:5
	ds_read2_b64 v[83:86], v78 offset1:4
	v_add_u32_e32 v110, 0x800, v78
	v_mov_b64_e32 v[87:88], v[95:96]
	v_mov_b64_e32 v[89:90], v[97:98]
	s_waitcnt lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[99:102], v[103:104], v[33:34], v[99:102]
	v_mmac_f32_16x16x16_bf16 v[99:102], v[105:106], v[35:36], v[99:102]
	ds_read2_b64 v[33:36], v110 offset0:16 offset1:20
	v_mov_b32_e32 v108, s1
	v_or_b32_e32 v107, s0, v59
	s_waitcnt lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[87:90], v[83:84], v[29:30], v[87:90]
	ds_read2_b64 v[91:94], v78 offset0:8 offset1:12
	v_mmac_f32_16x16x16_bf16 v[87:90], v[85:86], v[31:32], v[87:90]
	s_waitcnt lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[95:98], v[33:34], v[29:30], v[95:98]
	ds_read2st64_b64 v[83:86], v109 offset0:2 offset1:3
	ds_read2st64_b64 v[103:106], v109 offset0:6 offset1:7
	v_lshlrev_b64 v[29:30], 1, v[107:108]
	ds_read2_b64 v[107:110], v110 offset0:24 offset1:28
	v_mov_b32_e32 v115, s11
	v_mov_b32_e32 v120, s13
	v_mmac_f32_16x16x16_bf16 v[95:98], v[35:36], v[31:32], v[95:98]
	v_add_co_u32_e32 v31, vcc, s10, v29
	v_add_co_u32_e64 v32, s[0:1], s12, v29
	v_addc_co_u32_e32 v33, vcc, v115, v30, vcc
	v_addc_co_u32_e64 v34, vcc, v120, v30, s[0:1]
	v_add_co_u32_e32 v115, vcc, v31, v69
	v_perm_b32 v30, v119, v118, s30
	v_perm_b32 v29, v117, v116, s30
	v_add_co_u32_e64 v117, s[0:1], v32, v69
	v_addc_co_u32_e32 v116, vcc, 0, v33, vcc
	v_addc_co_u32_e64 v118, vcc, 0, v34, s[0:1]
	global_store_dwordx2 v[45:46], v[29:30], off
	s_waitcnt vmcnt(10) lgkmcnt(3)
	v_mmac_f32_16x16x16_bf16 v[87:90], v[91:92], v[25:26], v[87:90]
	s_waitcnt lgkmcnt(2)
	v_mmac_f32_16x16x16_bf16 v[79:82], v[83:84], v[21:22], v[79:82]
	s_waitcnt lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[99:102], v[103:104], v[21:22], v[99:102]
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[95:98], v[107:108], v[25:26], v[95:98]
	global_load_dwordx4 v[29:32], v[117:118], off
	v_mmac_f32_16x16x16_bf16 v[87:90], v[93:94], v[27:28], v[87:90]
	global_load_dwordx4 v[33:36], v[115:116], off
	v_mmac_f32_16x16x16_bf16 v[79:82], v[85:86], v[23:24], v[79:82]
	v_mmac_f32_16x16x16_bf16 v[99:102], v[105:106], v[23:24], v[99:102]
	v_mmac_f32_16x16x16_bf16 v[95:98], v[109:110], v[27:28], v[95:98]
	global_load_dwordx4 v[21:24], v[115:116], off offset:1024
	global_load_dwordx4 v[25:28], v[117:118], off offset:1024
	v_mov_b64_e32 v[83:84], s[16:17]
	v_xor_b32_e32 v88, 0x80000000, v88
	v_xor_b32_e32 v87, 0x80000000, v87
	v_mov_b64_e32 v[85:86], s[18:19]
	v_xor_b32_e32 v90, 0x80000000, v90
	v_xor_b32_e32 v89, 0x80000000, v89
	v_pk_fma_f32 v[79:80], v[83:84], v[79:80], v[87:88]
	v_xor_b32_e32 v88, 0x80000000, v96
	v_xor_b32_e32 v87, 0x80000000, v95
	v_mov_b32_e32 v92, s20
	v_add_co_u32_e32 v47, vcc, s15, v47
	v_pk_fma_f32 v[81:82], v[85:86], v[81:82], v[89:90]
	v_xor_b32_e32 v90, 0x80000000, v98
	v_xor_b32_e32 v89, 0x80000000, v97
	v_pk_fma_f32 v[51:52], v[51:52], v[111:112], v[79:80]
	v_pk_fma_f32 v[79:80], v[83:84], v[99:100], v[87:88]
	v_addc_co_u32_e32 v48, vcc, v48, v92, vcc
	v_pk_fma_f32 v[53:54], v[53:54], v[113:114], v[81:82]
	v_pk_fma_f32 v[81:82], v[85:86], v[101:102], v[89:90]
	v_pk_fma_f32 v[41:42], v[41:42], v[111:112], v[79:80]
	v_bfe_u32 v79, v51, 16, 1
	s_add_i32 s22, s22, 1
	s_add_i32 s8, s8, -1
	v_mov_b32_e32 v91, s14
	v_add_co_u32_e32 v45, vcc, s9, v45
	v_pk_fma_f32 v[49:50], v[49:50], v[113:114], v[81:82]
	v_bfe_u32 v80, v52, 16, 1
	v_bfe_u32 v81, v53, 16, 1
	v_bfe_u32 v82, v54, 16, 1
	v_add3_u32 v79, v51, v79, s21
	v_bfe_u32 v83, v41, 16, 1
	s_cmp_lt_i32 s22, s31
	v_addc_co_u32_e32 v46, vcc, v46, v91, vcc
	v_add3_u32 v80, v52, v80, s21
	v_add3_u32 v81, v53, v81, s21
	v_add3_u32 v82, v54, v82, s21
	v_bfe_u32 v84, v42, 16, 1
	v_bfe_u32 v85, v49, 16, 1
	v_bfe_u32 v86, v50, 16, 1
	ds_write_b16_d16_hi v62, v79
	ds_write_b16_d16_hi v64, v80 offset:8
	ds_write_b16_d16_hi v64, v81 offset:16
	ds_write_b16_d16_hi v64, v82 offset:24
	v_add3_u32 v79, v41, v83, s21
	v_add3_u32 v80, v42, v84, s21
	v_add3_u32 v81, v49, v85, s21
	v_add3_u32 v82, v50, v86, s21
	ds_write_b16_d16_hi v62, v79 offset:4224
	ds_write_b16_d16_hi v65, v80 offset:8
	ds_write_b16_d16_hi v65, v81 offset:16
	ds_write_b16_d16_hi v65, v82 offset:24
	s_cbranch_scc1 65170
	s_cbranch_scc1 .LBBX
