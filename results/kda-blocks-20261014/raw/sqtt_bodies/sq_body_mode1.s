.LBBX:  ; =>This Inner Loop Header: Depth=1
	s_add_i32 s14, s20, 1
	s_cmp_lt_i32 s14, s21
	s_cselect_b64 s[0:1], -1, 0
	s_waitcnt lgkmcnt(0)
	OTHER_MSG
	s_barrier
	global_load_dwordx4 v[94:97], v[46:47], off offset:-2048
	global_load_dwordx4 v[98:101], v[46:47], off offset:-1024
	global_load_dwordx4 v[102:105], v[46:47], off
	global_load_dwordx4 v[106:109], v[46:47], off offset:1024
	global_load_dwordx4 v[110:113], v[50:51], off offset:-1024
	global_load_dwordx4 v[114:117], v[50:51], off
	ds_read2_b64 v[118:121], v85 offset1:4
	ds_read2_b64 v[126:129], v93 offset1:4
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[124:125], s[6:7]
	ds_read2_b64 v[138:141], v85 offset0:8 offset1:12
	ds_read2_b64 v[142:145], v93 offset0:8 offset1:12
	v_mov_b64_e32 v[122:123], s[4:5]
	v_mov_b64_e32 v[132:133], v[124:125]
	v_mov_b64_e32 v[136:137], v[124:125]
	v_mov_b64_e32 v[130:131], v[122:123]
	v_mov_b64_e32 v[134:135], v[122:123]
	ds_read2_b64 v[146:149], v85 offset0:16 offset1:20
	ds_read2_b64 v[150:153], v93 offset0:16 offset1:20
	s_waitcnt vmcnt(16) lgkmcnt(5)
	v_mmac_f32_16x16x16_bf16 v[130:133], v[24:25], v[118:119], v[130:133]
	s_waitcnt lgkmcnt(4)
	v_mmac_f32_16x16x16_bf16 v[134:137], v[24:25], v[126:127], v[134:137]
	v_mmac_f32_16x16x16_bf16 v[130:133], v[26:27], v[120:121], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[26:27], v[128:129], v[134:137]
	ds_read2_b64 v[154:157], v85 offset0:24 offset1:28
	ds_read2_b64 v[158:161], v93 offset0:24 offset1:28
	s_waitcnt vmcnt(15) lgkmcnt(5)
	v_mmac_f32_16x16x16_bf16 v[130:133], v[20:21], v[138:139], v[130:133]
	s_waitcnt lgkmcnt(4)
	v_mmac_f32_16x16x16_bf16 v[134:137], v[20:21], v[142:143], v[134:137]
	v_mmac_f32_16x16x16_bf16 v[130:133], v[22:23], v[140:141], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[22:23], v[144:145], v[134:137]
	s_waitcnt vmcnt(14) lgkmcnt(3)
	v_mmac_f32_16x16x16_bf16 v[130:133], v[16:17], v[146:147], v[130:133]
	s_waitcnt lgkmcnt(2)
	v_mmac_f32_16x16x16_bf16 v[134:137], v[16:17], v[150:151], v[134:137]
	v_mmac_f32_16x16x16_bf16 v[130:133], v[18:19], v[148:149], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[18:19], v[152:153], v[134:137]
	s_waitcnt vmcnt(13) lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[130:133], v[12:13], v[154:155], v[130:133]
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[134:137], v[12:13], v[158:159], v[134:137]
	s_and_b64 s[0:1], s[0:1], exec
	v_mmac_f32_16x16x16_bf16 v[130:133], v[14:15], v[156:157], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[14:15], v[160:161], v[134:137]
	s_cselect_b32 s2, s14, s20
	s_ashr_i32 s3, s2, 31
	s_add_u32 s6, s2, s34
	s_addc_u32 s7, s3, s33
	s_lshl_b64 s[2:3], s[6:7], 13
	v_or_b32_e64 v13, s3, 0
	v_or_b32_e32 v12, s2, v66
	v_lshlrev_b64 v[12:13], 1, v[12:13]
	v_add_co_u32_e32 v12, vcc, v67, v12
	v_addc_co_u32_e32 v13, vcc, v68, v13, vcc
	global_load_dwordx4 v[24:27], v[12:13], off
	global_load_dwordx4 v[20:23], v[12:13], off offset:1024
	global_load_dwordx4 v[16:19], v[12:13], off offset:2048
	s_nop 0
	global_load_dwordx4 v[12:15], v[12:13], off offset:3072
	s_waitcnt vmcnt(16)
	v_lshlrev_b32_e32 v162, 16, v0
	v_lshlrev_b32_e32 v163, 16, v2
	v_sub_f32_e32 v130, v162, v130
	v_bfe_u32 v162, v130, 16, 1
	v_sub_f32_e32 v134, v163, v134
	v_and_b32_e32 v0, 0xffff0000, v0
	v_add3_u32 v130, v130, v162, s12
	v_bfe_u32 v162, v134, 16, 1
	v_and_b32_e32 v2, 0xffff0000, v2
	v_sub_f32_e32 v0, v0, v131
	v_add3_u32 v134, v134, v162, s12
	ds_write_b16_d16_hi v86, v130
	ds_write_b16_d16_hi v86, v134 offset:2176
	v_bfe_u32 v130, v0, 16, 1
	v_sub_f32_e32 v2, v2, v135
	v_add3_u32 v0, v0, v130, s12
	v_bfe_u32 v130, v2, 16, 1
	v_add3_u32 v2, v2, v130, s12
	ds_write_b16_d16_hi v86, v0 offset:544
	ds_write_b16_d16_hi v87, v2 offset:2176
	v_lshlrev_b32_e32 v0, 16, v1
	v_lshlrev_b32_e32 v2, 16, v3
	v_sub_f32_e32 v0, v0, v132
	v_bfe_u32 v130, v0, 16, 1
	v_sub_f32_e32 v2, v2, v136
	v_add3_u32 v0, v0, v130, s12
	v_bfe_u32 v130, v2, 16, 1
	v_add3_u32 v2, v2, v130, s12
	ds_write_b16_d16_hi v86, v0 offset:1088
	ds_write_b16_d16_hi v88, v2 offset:2176
	v_and_b32_e32 v0, 0xffff0000, v1
	v_and_b32_e32 v1, 0xffff0000, v3
	v_sub_f32_e32 v0, v0, v133
	v_bfe_u32 v2, v0, 16, 1
	v_sub_f32_e32 v1, v1, v137
	v_add3_u32 v0, v0, v2, s12
	v_bfe_u32 v2, v1, 16, 1
	v_add3_u32 v1, v1, v2, s12
	ds_write_b16_d16_hi v86, v0 offset:1632
	ds_write_b16_d16_hi v89, v1 offset:2176
	v_mov_b32_e32 v1, s3
	v_add_co_u32_e32 v0, vcc, s2, v69
	v_addc_co_u32_e32 v1, vcc, v1, v70, vcc
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	v_add_co_u32_e32 v0, vcc, v71, v0
	v_addc_co_u32_e32 v1, vcc, v72, v1, vcc
	global_load_dwordx4 v[0:3], v[0:1], off
	s_waitcnt vmcnt(16)
	v_mov_b32_e32 v162, v4
	s_waitcnt vmcnt(15)
	v_mov_b32_e32 v164, v8
	v_mov_b32_e32 v163, v5
	v_mov_b32_e32 v165, v9
	v_mov_b32_e32 v166, v6
	v_mov_b32_e32 v168, v10
	v_mov_b32_e32 v167, v7
	v_mov_b32_e32 v169, v11
	s_lshl_b64 s[6:7], s[6:7], 7
	v_or_b32_e64 v5, s7, 0
	v_or_b32_e32 v4, s6, v73
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_or_b32_e64 v7, s7, 0
	v_or_b32_e32 v6, s6, v76
	v_add_co_u32_e32 v4, vcc, v74, v4
	v_lshlrev_b64 v[6:7], 2, v[6:7]
	v_addc_co_u32_e32 v5, vcc, v75, v5, vcc
	v_add_co_u32_e32 v8, vcc, v74, v6
	v_addc_co_u32_e32 v9, vcc, v75, v7, vcc
	global_load_dwordx4 v[4:7], v[4:5], off
	s_nop 0
	global_load_dwordx4 v[8:11], v[8:9], off
	v_mov_b64_e32 v[132:133], v[124:125]
	v_mov_b64_e32 v[136:137], v[124:125]
	v_mov_b64_e32 v[130:131], v[122:123]
	v_mov_b64_e32 v[134:135], v[122:123]
	s_waitcnt vmcnt(12)
	v_mmac_f32_16x16x16_bf16 v[130:133], v[94:95], v[118:119], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[94:95], v[126:127], v[134:137]
	v_mmac_f32_16x16x16_bf16 v[130:133], v[96:97], v[120:121], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[96:97], v[128:129], v[134:137]
	s_waitcnt vmcnt(11)
	v_mmac_f32_16x16x16_bf16 v[130:133], v[98:99], v[138:139], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[98:99], v[142:143], v[134:137]
	v_mmac_f32_16x16x16_bf16 v[130:133], v[100:101], v[140:141], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[100:101], v[144:145], v[134:137]
	s_waitcnt vmcnt(10)
	v_mmac_f32_16x16x16_bf16 v[130:133], v[102:103], v[146:147], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[102:103], v[150:151], v[134:137]
	v_mmac_f32_16x16x16_bf16 v[130:133], v[104:105], v[148:149], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[104:105], v[152:153], v[134:137]
	s_waitcnt vmcnt(9)
	v_mmac_f32_16x16x16_bf16 v[130:133], v[106:107], v[154:155], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[106:107], v[158:159], v[134:137]
	v_mmac_f32_16x16x16_bf16 v[130:133], v[108:109], v[156:157], v[130:133]
	v_mmac_f32_16x16x16_bf16 v[134:137], v[108:109], v[160:161], v[134:137]
	s_waitcnt lgkmcnt(0)
	OTHER_MSG
	s_barrier
	ds_read2_b64 v[94:97], v90 offset1:4
	ds_read2_b64 v[98:101], v90 offset0:8 offset1:12
	v_mov_b64_e32 v[102:103], v[122:123]
	v_mov_b64_e32 v[104:105], v[124:125]
	s_waitcnt vmcnt(8) lgkmcnt(1)
	s_nop 0
	v_mmac_f32_16x16x16_bf16 v[102:105], v[110:111], v[94:95], v[102:105]
	v_mmac_f32_16x16x16_bf16 v[102:105], v[112:113], v[96:97], v[102:105]
	s_waitcnt vmcnt(7) lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[102:105], v[114:115], v[98:99], v[102:105]
	v_mmac_f32_16x16x16_bf16 v[102:105], v[116:117], v[100:101], v[102:105]
	s_nop 2
	v_fma_f32 v94, s16, v130, v102
	v_bfe_u32 v97, v94, 16, 1
	v_fma_f32 v95, s16, v131, v103
	v_add3_u32 v94, v94, v97, s12
	ds_write_b16_d16_hi v91, v94
	v_bfe_u32 v94, v95, 16, 1
	v_fma_f32 v96, s16, v132, v104
	v_add3_u32 v94, v95, v94, s12
	ds_write_b16_d16_hi v91, v94 offset:8
	v_bfe_u32 v94, v96, 16, 1
	v_fmac_f32_e32 v105, s16, v133
	v_add3_u32 v94, v96, v94, s12
	ds_write_b16_d16_hi v91, v94 offset:16
	v_bfe_u32 v94, v105, 16, 1
	v_add3_u32 v94, v105, v94, s12
	ds_write_b16_d16_hi v91, v94 offset:24
	s_waitcnt lgkmcnt(0)
	ds_read_b64 v[94:95], v81
	v_mov_b64_e32 v[98:99], v[122:123]
	v_mov_b64_e32 v[100:101], v[124:125]
	s_waitcnt lgkmcnt(0)
	global_store_dwordx2 v[48:49], v[94:95], off offset:-32
	ds_read2_b64 v[94:97], v84 offset1:4
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[98:101], v[110:111], v[94:95], v[98:101]
	v_mmac_f32_16x16x16_bf16 v[98:101], v[112:113], v[96:97], v[98:101]
	ds_read2_b64 v[94:97], v84 offset0:8 offset1:12
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[98:101], v[114:115], v[94:95], v[98:101]
	v_mmac_f32_16x16x16_bf16 v[98:101], v[116:117], v[96:97], v[98:101]
	s_nop 2
	v_fma_f32 v94, s16, v134, v98
	v_bfe_u32 v97, v94, 16, 1
	v_fma_f32 v95, s16, v135, v99
	v_add3_u32 v94, v94, v97, s12
	ds_write_b16_d16_hi v91, v94
	v_bfe_u32 v94, v95, 16, 1
	v_fma_f32 v96, s16, v136, v100
	v_add3_u32 v94, v95, v94, s12
	ds_write_b16_d16_hi v91, v94 offset:8
	v_bfe_u32 v94, v96, 16, 1
	v_fmac_f32_e32 v101, s16, v137
	v_add3_u32 v94, v96, v94, s12
	ds_write_b16_d16_hi v91, v94 offset:16
	v_bfe_u32 v94, v101, 16, 1
	v_add3_u32 v94, v101, v94, s12
	ds_write_b16_d16_hi v91, v94 offset:24
	s_waitcnt lgkmcnt(0)
	ds_read_b64 v[106:107], v81
	ds_read2_b64 v[94:97], v90 offset1:4
	ds_read2_b64 v[98:101], v90 offset0:8 offset1:12
	v_mov_b64_e32 v[102:103], v[122:123]
	v_mov_b64_e32 v[104:105], v[124:125]
	s_waitcnt lgkmcnt(2)
	global_store_dwordx2 v[48:49], v[106:107], off
	s_waitcnt lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[102:105], v[94:95], v[36:37], v[102:105]
	v_mmac_f32_16x16x16_bf16 v[102:105], v[96:97], v[38:39], v[102:105]
	ds_read2_b64 v[106:109], v84 offset1:4
	s_waitcnt lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[102:105], v[98:99], v[40:41], v[102:105]
	v_mmac_f32_16x16x16_bf16 v[102:105], v[100:101], v[42:43], v[102:105]
	s_nop 2
	v_pk_fma_f32 v[60:61], v[60:61], v[166:167], v[104:105]
	v_pk_fma_f32 v[58:59], v[58:59], v[162:163], v[102:103]
	v_mov_b64_e32 v[102:103], v[122:123]
	v_mov_b64_e32 v[104:105], v[124:125]
	ds_read2_b64 v[110:113], v84 offset0:8 offset1:12
	s_waitcnt lgkmcnt(1)
	v_mmac_f32_16x16x16_bf16 v[102:105], v[106:107], v[36:37], v[102:105]
	v_mmac_f32_16x16x16_bf16 v[102:105], v[108:109], v[38:39], v[102:105]
	v_mov_b64_e32 v[36:37], v[122:123]
	v_mov_b64_e32 v[38:39], v[124:125]
	s_nop 1
	v_mmac_f32_16x16x16_bf16 v[36:39], v[94:95], v[28:29], v[36:39]
	v_mmac_f32_16x16x16_bf16 v[122:125], v[106:107], v[28:29], v[122:125]
	v_mmac_f32_16x16x16_bf16 v[36:39], v[96:97], v[30:31], v[36:39]
	v_mmac_f32_16x16x16_bf16 v[122:125], v[108:109], v[30:31], v[122:125]
	s_waitcnt lgkmcnt(0)
	v_mmac_f32_16x16x16_bf16 v[102:105], v[110:111], v[40:41], v[102:105]
	v_mmac_f32_16x16x16_bf16 v[36:39], v[98:99], v[32:33], v[36:39]
	v_mmac_f32_16x16x16_bf16 v[122:125], v[110:111], v[32:33], v[122:125]
	v_mmac_f32_16x16x16_bf16 v[102:105], v[112:113], v[42:43], v[102:105]
	v_mmac_f32_16x16x16_bf16 v[36:39], v[100:101], v[34:35], v[36:39]
	v_mmac_f32_16x16x16_bf16 v[122:125], v[112:113], v[34:35], v[122:125]
	s_nop 0
	v_pk_fma_f32 v[64:65], v[64:65], v[166:167], v[104:105]
	v_pk_fma_f32 v[62:63], v[62:63], v[162:163], v[102:103]
	v_pk_fma_f32 v[52:53], v[52:53], v[168:169], v[38:39]
	v_pk_fma_f32 v[44:45], v[44:45], v[164:165], v[36:37]
	v_pk_fma_f32 v[56:57], v[56:57], v[168:169], v[124:125]
	v_pk_fma_f32 v[54:55], v[54:55], v[164:165], v[122:123]
	v_or_b32_e64 v29, s3, 0
	v_or_b32_e32 v28, s2, v77
	v_lshlrev_b64 v[28:29], 1, v[28:29]
	v_mov_b32_e32 v30, s11
	v_add_co_u32_e32 v28, vcc, s10, v28
	v_addc_co_u32_e32 v29, vcc, v30, v29, vcc
	v_add_co_u32_e32 v28, vcc, v28, v92
	v_addc_co_u32_e32 v29, vcc, 0, v29, vcc
	global_load_dwordx4 v[36:39], v[28:29], off
	global_load_dwordx4 v[40:43], v[28:29], off offset:1024
	v_or_b32_e64 v29, s3, 0
	v_or_b32_e32 v28, s2, v78
	v_lshlrev_b64 v[28:29], 1, v[28:29]
	v_add_co_u32_e32 v28, vcc, s10, v28
	v_addc_co_u32_e32 v29, vcc, v30, v29, vcc
	v_add_co_u32_e32 v94, vcc, v28, v92
	v_addc_co_u32_e32 v95, vcc, 0, v29, vcc
	global_load_dwordx4 v[28:31], v[94:95], off
	global_load_dwordx4 v[32:35], v[94:95], off offset:1024
	v_bfe_u32 v94, v58, 16, 1
	v_add3_u32 v94, v58, v94, s12
	ds_write_b16_d16_hi v79, v94
	v_bfe_u32 v94, v59, 16, 1
	v_add3_u32 v94, v59, v94, s12
	ds_write_b16_d16_hi v79, v94 offset:8
	v_bfe_u32 v94, v60, 16, 1
	v_add3_u32 v94, v60, v94, s12
	ds_write_b16_d16_hi v79, v94 offset:16
	v_bfe_u32 v94, v61, 16, 1
	v_add3_u32 v94, v61, v94, s12
	ds_write_b16_d16_hi v79, v94 offset:24
	v_bfe_u32 v94, v62, 16, 1
	v_add3_u32 v94, v62, v94, s12
	ds_write_b16_d16_hi v79, v94 offset:4224
	v_bfe_u32 v94, v63, 16, 1
	v_add3_u32 v94, v63, v94, s12
	ds_write_b16_d16_hi v82, v94 offset:4224
	v_bfe_u32 v94, v64, 16, 1
	v_add3_u32 v94, v64, v94, s12
	ds_write_b16_d16_hi v79, v94 offset:4240
	v_bfe_u32 v94, v65, 16, 1
	v_add3_u32 v94, v65, v94, s12
	ds_write_b16_d16_hi v79, v94 offset:4248
	v_bfe_u32 v94, v44, 16, 1
	v_add3_u32 v94, v44, v94, s12
	ds_write_b16_d16_hi v80, v94
	v_bfe_u32 v94, v45, 16, 1
	v_add3_u32 v94, v45, v94, s12
	ds_write_b16_d16_hi v80, v94 offset:8
	v_bfe_u32 v94, v52, 16, 1
	v_add3_u32 v94, v52, v94, s12
	ds_write_b16_d16_hi v80, v94 offset:16
	v_bfe_u32 v94, v53, 16, 1
	v_add3_u32 v94, v53, v94, s12
	ds_write_b16_d16_hi v80, v94 offset:24
	v_bfe_u32 v94, v54, 16, 1
	v_add3_u32 v94, v54, v94, s12
	ds_write_b16_d16_hi v83, v94 offset:4224
	v_bfe_u32 v94, v55, 16, 1
	v_add3_u32 v94, v55, v94, s12
	ds_write_b16_d16_hi v83, v94 offset:4232
	v_bfe_u32 v94, v56, 16, 1
	v_add3_u32 v94, v56, v94, s12
	ds_write_b16_d16_hi v80, v94 offset:4240
	v_bfe_u32 v94, v57, 16, 1
	v_add_co_u32_e32 v46, vcc, s13, v46
	v_add3_u32 v94, v57, v94, s12
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	ds_write_b16_d16_hi v83, v94 offset:4248
	v_mov_b32_e32 v94, s9
	v_add_co_u32_e32 v48, vcc, s8, v48
	v_addc_co_u32_e32 v49, vcc, v49, v94, vcc
	v_add_co_u32_e32 v50, vcc, 0x2000, v50
	v_addc_co_u32_e32 v51, vcc, 0, v51, vcc
	s_mov_b32 s20, s14
	s_mov_b64 vcc, s[0:1]
	s_cbranch_vccnz 64970
	s_cbranch_scc1 .LBBX
