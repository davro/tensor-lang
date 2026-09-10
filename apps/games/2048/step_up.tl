// apps/games/2048/step_up.tl
//
// Pure tensor-ops implementation of ONE 2048 move: slide-and-merge every column one step UP.
// This is a sibling of step.tl (which implements LEFT) — see step.tl's
// header comment for the full design writeup of compact_pair/compact_row/
// merge_pair/merge_row (branchless masking, the [4,2,2,0] counter-example,
// why merge_row can't factor out a merge_line3 helper on this compiler,
// etc). None of that is repeated here.
//
// tensor-lang has no import/include mechanism (see tensorlang.lark — every
// .tl file is standalone), so compact_pair/compact_row/merge_pair/merge_row
// are duplicated verbatim from step.tl rather than shared. Keep the two
// files' copies of those four functions in sync if you ever change the
// algorithm itself.
//
// Slide-and-merge every column one step UP. transpose(board) turns each
// column into a row (T[c][:] == board[:,c]), so running the exact same
// LEFT-slide pipeline step.tl uses on each row of T slides every column
// toward index 0 (the top) -- which is exactly what UP means. Transpose
// back at the end to return to board orientation.

// Run directly, for manual testing:
//   python3 tensorlang.py apps/games/2048/step_up.tl

fn compact_pair(pair: Tensor[f32, (1, 2)]) -> Tensor[f32, (1, 2)] {
    let cp_a = pair[0:1, 0:1]
    let cp_b = pair[0:1, 1:2]
    let cp_zero = fill(0.0, (1, 1))
    let cp_one  = fill(1.0, (1, 1))
    let cp_a_is_zero    = equal(cp_a, cp_zero)
    let cp_b_is_nonzero = greater(cp_b, cp_zero)
    let cp_should_swap  = mult(cp_a_is_zero, cp_b_is_nonzero)
    let cp_keep         = minus(cp_one, cp_should_swap)
    let cp_swap_b = mult(cp_should_swap, cp_b)
    let cp_keep_a = mult(cp_keep, cp_a)
    let cp_a_new  = add(cp_swap_b, cp_keep_a)
    let cp_swap_a = mult(cp_should_swap, cp_a)
    let cp_keep_b = mult(cp_keep, cp_b)
    let cp_b_new  = add(cp_swap_a, cp_keep_b)
    let cp_result = concat(cp_a_new, cp_b_new, axis=1)
    return cp_result
}

////////////////////////////////////////////////////////////////////////////
// compact_row — 3 full bubble passes over a 4-cell row using compact_pair.
// Each pass handles pairs (0,1) then (1,2) then (2,3) IN SEQUENCE, feeding
// each pair's output into the next comparison — like one pass of bubble
// sort, not three independent parallel pairs (which would miss cases where
// a value needs to move more than one slot in a single pass).
////////////////////////////////////////////////////////////////////////////
fn compact_row(row: Tensor[f32, (1, 4)]) -> Tensor[f32, (1, 4)] {
    let cr_r0 = row[0:1, 0:1]
    let cr_r1 = row[0:1, 1:2]
    let cr_r2 = row[0:1, 2:3]
    let cr_r3 = row[0:1, 3:4]

    // --- pass 1 ---
    let cr_p01_in  = concat(cr_r0, cr_r1, axis=1)
    let cr_p01_out = compact_pair(cr_p01_in)
    let cr_a1 = cr_p01_out[0:1, 0:1]
    let cr_b1 = cr_p01_out[0:1, 1:2]

    let cr_p12_in  = concat(cr_b1, cr_r2, axis=1)
    let cr_p12_out = compact_pair(cr_p12_in)
    let cr_b2 = cr_p12_out[0:1, 0:1]
    let cr_c2 = cr_p12_out[0:1, 1:2]

    let cr_p23_in  = concat(cr_c2, cr_r3, axis=1)
    let cr_p23_out = compact_pair(cr_p23_in)
    let cr_c3 = cr_p23_out[0:1, 0:1]
    let cr_d3 = cr_p23_out[0:1, 1:2]

    let cr_pass1_top = concat(cr_a1, cr_b2, axis=1)
    let cr_pass1_bot = concat(cr_c3, cr_d3, axis=1)
    let cr_pass1     = concat(cr_pass1_top, cr_pass1_bot, axis=1)

    // --- pass 2 ---
    let cr_s0 = cr_pass1[0:1, 0:1]
    let cr_s1 = cr_pass1[0:1, 1:2]
    let cr_s2 = cr_pass1[0:1, 2:3]
    let cr_s3 = cr_pass1[0:1, 3:4]

    let cr_q01_in  = concat(cr_s0, cr_s1, axis=1)
    let cr_q01_out = compact_pair(cr_q01_in)
    let cr_a2 = cr_q01_out[0:1, 0:1]
    let cr_b3 = cr_q01_out[0:1, 1:2]

    let cr_q12_in  = concat(cr_b3, cr_s2, axis=1)
    let cr_q12_out = compact_pair(cr_q12_in)
    let cr_b4 = cr_q12_out[0:1, 0:1]
    let cr_c4 = cr_q12_out[0:1, 1:2]

    let cr_q23_in  = concat(cr_c4, cr_s3, axis=1)
    let cr_q23_out = compact_pair(cr_q23_in)
    let cr_c5 = cr_q23_out[0:1, 0:1]
    let cr_d5 = cr_q23_out[0:1, 1:2]

    let cr_pass2_top = concat(cr_a2, cr_b4, axis=1)
    let cr_pass2_bot = concat(cr_c5, cr_d5, axis=1)
    let cr_pass2     = concat(cr_pass2_top, cr_pass2_bot, axis=1)

    // --- pass 3 ---
    let cr_t0 = cr_pass2[0:1, 0:1]
    let cr_t1 = cr_pass2[0:1, 1:2]
    let cr_t2 = cr_pass2[0:1, 2:3]
    let cr_t3 = cr_pass2[0:1, 3:4]

    let cr_u01_in  = concat(cr_t0, cr_t1, axis=1)
    let cr_u01_out = compact_pair(cr_u01_in)
    let cr_a3 = cr_u01_out[0:1, 0:1]
    let cr_b5 = cr_u01_out[0:1, 1:2]

    let cr_u12_in  = concat(cr_b5, cr_t2, axis=1)
    let cr_u12_out = compact_pair(cr_u12_in)
    let cr_b6 = cr_u12_out[0:1, 0:1]
    let cr_c6 = cr_u12_out[0:1, 1:2]

    let cr_u23_in  = concat(cr_c6, cr_t3, axis=1)
    let cr_u23_out = compact_pair(cr_u23_in)
    let cr_c7 = cr_u23_out[0:1, 0:1]
    let cr_d7 = cr_u23_out[0:1, 1:2]

    let cr_pass3_top = concat(cr_a3, cr_b6, axis=1)
    let cr_pass3_bot = concat(cr_c7, cr_d7, axis=1)
    let cr_pass3     = concat(cr_pass3_top, cr_pass3_bot, axis=1)

    return cr_pass3
}

////////////////////////////////////////////////////////////////////////////
// merge_pair — if a == b (including the harmless 0==0 case, which is a
// no-op either way), combine them into a doubled tile in the left slot and
// zero the right slot. Otherwise leave both untouched.
////////////////////////////////////////////////////////////////////////////
fn merge_pair(pair: Tensor[f32, (1, 2)]) -> Tensor[f32, (1, 2)] {
    let mp_a = pair[0:1, 0:1]
    let mp_b = pair[0:1, 1:2]
    let mp_one   = fill(1.0, (1, 1))
    let mp_is_eq = equal(mp_a, mp_b)
    let mp_keep  = minus(mp_one, mp_is_eq)
    let mp_sum   = add(mp_a, mp_b)
    let mp_eq_sum = mult(mp_is_eq, mp_sum)
    let mp_keep_a = mult(mp_keep, mp_a)
    let mp_a_new  = add(mp_eq_sum, mp_keep_a)
    let mp_b_new = mult(mp_keep, mp_b)
    let mp_result = concat(mp_a_new, mp_b_new, axis=1)
    return mp_result
}

////////////////////////////////////////////////////////////////////////////
// merge_row — correct 2048 left-move merge for a COMPACTED 4-cell row.
//
// Why not just check pairs (0,1) and (2,3) independently? Counter-example:
// compacted row [4, 2, 2, 0]. Real 2048 scans left-to-right and merges the
// FIRST equal adjacent pair it finds, so cells 1 & 2 (the two 2s) merge
// into [4, 4, 0, 0]. A fixed-pair check at (0,1)/(2,3) never even compares
// (1,2), and would wrongly leave the row unchanged. The correct rule is a
// left-to-right scan with a "don't merge a tile twice" carry, which is
// data-dependent — rather than write real data-dependent control flow
// (which `fn` bodies can't do here anyway), this computes BOTH possible
// worlds — "cell 0 & 1 merge" vs "cell 0 & 1 don't merge" — unconditionally
// and blends them with a mask, recursively, down to a base case that's a
// single merge_pair call.
//
// NOTE ON STRUCTURE: this was originally written as merge_row() calling a
// separate merge_line3() helper, which itself called merge_pair() — i.e.
// two levels of user-function-calling-user-function. That doesn't work on
// the current compiler: build_ast's inliner only expands ONE level of
// nested calls, and only for a function invoked directly from the top
// level of the program (see ast_builder.py's build_ast, the `elif
// stmt.data == 'let_binding'` branch — the one-level nested-expansion loop
// lives there, not inside inline_function_call() itself, so it never
// recurses further). A three-deep chain (top level -> merge_row ->
// merge_line3 -> merge_pair) left an unexpanded user_function_call node in
// the AST and crashed the type checker. So the 3-cell recursive case is
// inlined BY HAND below instead of factored into its own function —
// merge_row now calls merge_pair directly, twice, which fits the
// one-level pattern that's actually proven to work.
//
// Checked by hand against:
//   [2,2,2,2] -> [4,4,0,0]
//   [4,4,4,0] -> [8,4,0,0]
//   [4,2,2,0] -> [4,4,0,0]   (the case fixed-pairing gets wrong)
//   [4,2,8,0] -> [4,2,8,0]   (no merge)
////////////////////////////////////////////////////////////////////////////
fn merge_row(row: Tensor[f32, (1, 4)]) -> Tensor[f32, (1, 4)] {
    let mr_a = row[0:1, 0:1]
    let mr_b = row[0:1, 1:2]
    let mr_c = row[0:1, 2:3]
    let mr_d = row[0:1, 3:4]
    let mr_zero = fill(0.0, (1, 1))

    // outer decision: does a merge with b?
    let mr_m      = equal(mr_a, mr_b)
    let mr_m4_ab  = concat(mr_m, mr_m, axis=1)
    let mr_m4_abc = concat(mr_m4_ab, mr_m, axis=1)
    let mr_m4     = concat(mr_m4_abc, mr_m, axis=1)
    let mr_one4   = fill(1.0, (1, 4))
    let mr_keep4  = minus(mr_one4, mr_m4)

    // world X: a and b DO merge -> [a+b] ++ merge_pair([c, d]) ++ [0]
    let mr_ab_sum     = add(mr_a, mr_b)
    let mr_cd_in      = concat(mr_c, mr_d, axis=1)
    let mr_cd_out     = merge_pair(mr_cd_in)
    let mr_cd0        = mr_cd_out[0:1, 0:1]
    let mr_cd1        = mr_cd_out[0:1, 1:2]
    let mr_worldX_ab  = concat(mr_ab_sum, mr_cd0, axis=1)
    let mr_worldX_abc = concat(mr_worldX_ab, mr_cd1, axis=1)
    let mr_worldX     = concat(mr_worldX_abc, mr_zero, axis=1)

    // world Y: a and b do NOT merge -> [a] ++ merge3(b, c, d), inlined:
    //   inner decision: does b merge with c?
    let mr_bc_m     = equal(mr_b, mr_c)
    let mr_bc_m3_bc = concat(mr_bc_m, mr_bc_m, axis=1)
    let mr_bc_m3    = concat(mr_bc_m3_bc, mr_bc_m, axis=1)
    let mr_one3     = fill(1.0, (1, 3))
    let mr_bc_keep3 = minus(mr_one3, mr_bc_m3)

    //   inner world X2: b and c DO merge -> [b+c, d, 0]
    let mr_bc_sum      = add(mr_b, mr_c)
    let mr_i_worldX_bd = concat(mr_bc_sum, mr_d, axis=1)
    let mr_i_worldX    = concat(mr_i_worldX_bd, mr_zero, axis=1)

    //   inner world Y2: b and c do NOT merge -> [b] ++ merge_pair([c, d])
    let mr_cd2_in    = concat(mr_c, mr_d, axis=1)
    let mr_cd2_out   = merge_pair(mr_cd2_in)
    let mr_cd2_0     = mr_cd2_out[0:1, 0:1]
    let mr_cd2_1     = mr_cd2_out[0:1, 1:2]
    let mr_i_worldY_b = concat(mr_b, mr_cd2_0, axis=1)
    let mr_i_worldY   = concat(mr_i_worldY_b, mr_cd2_1, axis=1)

    let mr_i_mx = mult(mr_bc_m3, mr_i_worldX)
    let mr_i_ky = mult(mr_bc_keep3, mr_i_worldY)
    let mr_bcd  = add(mr_i_mx, mr_i_ky)         // (1,3): merge3(b,c,d)

    let mr_worldY = concat(mr_a, mr_bcd, axis=1)  // (1,4)

    let mr_mx = mult(mr_m4, mr_worldX)
    let mr_ky = mult(mr_keep4, mr_worldY)
    let mr_result = add(mr_mx, mr_ky)
    return mr_result
}

////////////////////////////////////////////////////////////////////////////
let board: Tensor[f32, (4, 4)] = load("cache/apps/games/2048/step_up.tl/board.npy")

let boardT = transpose(board)

let row0 = boardT[0:1, :]
let row1 = boardT[1:2, :]
let row2 = boardT[2:3, :]
let row3 = boardT[3:4, :]

// row0 of boardT == column 0 of board: plain left-slide == UP
    let row0_c1 = compact_row(row0)
    let row0_m  = merge_row(row0_c1)
    let row0_new = compact_row(row0_m)

// row1 of boardT == column 1 of board: plain left-slide == UP
    let row1_c1 = compact_row(row1)
    let row1_m  = merge_row(row1_c1)
    let row1_new = compact_row(row1_m)

// row2 of boardT == column 2 of board: plain left-slide == UP
    let row2_c1 = compact_row(row2)
    let row2_m  = merge_row(row2_c1)
    let row2_new = compact_row(row2_m)

// row3 of boardT == column 3 of board: plain left-slide == UP
    let row3_c1 = compact_row(row3)
    let row3_m  = merge_row(row3_c1)
    let row3_new = compact_row(row3_m)

    let top_half  = concat(row0_new, row1_new, axis=0)
    let bot_half  = concat(row2_new, row3_new, axis=0)
let new_boardT = concat(top_half, bot_half, axis=0)
let new_board = transpose(new_boardT)

save(new_board, "new_board.npy")
