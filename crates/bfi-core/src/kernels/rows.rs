//! `get_rows`: embedding lookup straight from packed weights.
//! `out[t, e] = W[ids[t], e]`, one lane per output element.

use cubecl::prelude::*;

use super::bytes::{weight_f16, weight_f32, weight_q1_0, weight_q8_0};

#[cube(launch_unchecked)]
pub fn rows_q1_0(w: &Array<u32>, ids: &Array<u32>, out: &mut Array<f32>, n_embd: u32, total: u32) {
    let gid = CUBE_POS_X * 64u32 + UNIT_POS;
    if gid < total {
        let row = ids[usize::cast_from(gid / n_embd)];
        out[usize::cast_from(gid)] = weight_q1_0(w, row, gid % n_embd, n_embd);
    }
}

#[cube(launch_unchecked)]
pub fn rows_q8_0(w: &Array<u32>, ids: &Array<u32>, out: &mut Array<f32>, n_embd: u32, total: u32) {
    let gid = CUBE_POS_X * 64u32 + UNIT_POS;
    if gid < total {
        let row = ids[usize::cast_from(gid / n_embd)];
        out[usize::cast_from(gid)] = weight_q8_0(w, row, gid % n_embd, n_embd);
    }
}

#[cube(launch_unchecked)]
pub fn rows_f16(w: &Array<u32>, ids: &Array<u32>, out: &mut Array<f32>, n_embd: u32, total: u32) {
    let gid = CUBE_POS_X * 64u32 + UNIT_POS;
    if gid < total {
        let row = ids[usize::cast_from(gid / n_embd)];
        out[usize::cast_from(gid)] = weight_f16(w, row * n_embd + gid % n_embd);
    }
}

#[cube(launch_unchecked)]
pub fn rows_f32(w: &Array<u32>, ids: &Array<u32>, out: &mut Array<f32>, n_embd: u32, total: u32) {
    let gid = CUBE_POS_X * 64u32 + UNIT_POS;
    if gid < total {
        let row = ids[usize::cast_from(gid / n_embd)];
        out[usize::cast_from(gid)] = weight_f32(w, row * n_embd + gid % n_embd);
    }
}
