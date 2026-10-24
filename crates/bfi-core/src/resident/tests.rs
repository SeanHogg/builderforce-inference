//! Resident (wgpu) kernels vs the CPU reference. Each test returns early when
//! no wgpu adapter enumerates, so `cargo test --features webgpu` stays green on
//! GPU-less CI and is a real check on any machine with Vulkan/Metal/DX12.

use super::*;
use crate::fused_gemv::max_rel_err;
use crate::{Backend, CpuBackend, DType, RopeParams};
use bfi_gguf::{f32_to_f16, quantize_row_q1_0};

fn gpu() -> Option<Gpu> {
    crate::gpu_or_skip(open_native(), "the resident kernel tests")
}

fn read(g: &Gpu, t: &DeviceTensor) -> Vec<f32> {
    block_on(g.read_f32(t)).expect("readback")
}

fn pattern(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 7919 + seed * 104_729) % 2003) as f32) / 1001.5 - 1.0) * 0.5)
        .collect()
}

fn q1_bytes(n_out: usize, n_in: usize) -> Vec<u8> {
    let w = pattern(n_out * n_in, 3);
    let mut out = vec![0u8; n_out * n_in / 128 * 18];
    for (b, chunk) in w.chunks(128).enumerate() {
        quantize_row_q1_0(chunk, &mut out[b * 18..b * 18 + 18]);
    }
    out
}

fn q8_bytes(n_out: usize, n_in: usize) -> Vec<u8> {
    let w = pattern(n_out * n_in, 5);
    let mut out = Vec::with_capacity(n_out * n_in / 32 * 34);
    for chunk in w.chunks(32) {
        let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for v in chunk {
            out.push(((v / d).round().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    out
}

fn f16_bytes(n: usize) -> Vec<u8> {
    pattern(n, 9)
        .iter()
        .flat_map(|v| f32_to_f16(*v).to_le_bytes())
        .collect()
}

fn f32_bytes(n: usize) -> Vec<u8> {
    pattern(n, 11).iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn check_matmul(ggml: u32, dtype: DType, bytes: Vec<u8>, n_out: usize, n_in: usize, n_tokens: usize) {
    let Some(g) = gpu() else { return };
    let cpu = CpuBackend;
    let x = pattern(n_tokens * n_in, 1);
    let w_cpu = if dtype == DType::F32 {
        let f: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        cpu.upload_f32(&[n_out, n_in], &f).unwrap()
    } else {
        cpu.upload_quant(&[n_out, n_in], dtype, &bytes).unwrap()
    };
    let want = cpu
        .matmul_q(&w_cpu, &cpu.upload_f32(&[n_tokens, n_in], &x).unwrap())
        .unwrap();
    let w = g
        .upload_weight("w", ggml, &[n_in as u64, n_out as u64], &bytes)
        .unwrap();
    let got = read(&g, &g.matmul(&w, &g.upload_f32(&x), n_tokens));
    let err = max_rel_err(&got, want.as_f32().unwrap());
    assert!(err <= 1e-3, "{dtype:?} rel err {err}");
}

#[test]
fn matmul_q1_0_decode_and_prefill_tiles() {
    check_matmul(bfi_gguf::TYPE_Q1_0, DType::Q1_0, q1_bytes(70, 384), 70, 384, 1);
    // 11 tokens crosses a TOKEN_TILE boundary.
    check_matmul(bfi_gguf::TYPE_Q1_0, DType::Q1_0, q1_bytes(33, 1024), 33, 1024, 11);
}

#[test]
fn matmul_q8_f16_f32() {
    check_matmul(bfi_gguf::TYPE_Q8_0, DType::Q8_0, q8_bytes(17, 256), 17, 256, 3);
    check_matmul(bfi_gguf::TYPE_F16, DType::F16, f16_bytes(9 * 130), 9, 130, 2);
    check_matmul(bfi_gguf::TYPE_F32, DType::F32, f32_bytes(9 * 130), 9, 130, 9);
}

#[test]
fn unsupported_type_names_the_tensor() {
    let Some(g) = gpu() else { return };
    let err = g
        .upload_weight("blk.3.ffn_up.weight", bfi_gguf::TYPE_Q4_K, &[256, 1], &[0u8; 144])
        .err()
        .expect("Q4_K is not resident");
    let msg = err.to_string();
    assert!(msg.contains("blk.3.ffn_up.weight") && msg.contains("Q4_K"), "{msg}");
}

#[test]
fn get_rows_q1_0_matches_cpu() {
    let Some(g) = gpu() else { return };
    let cpu = CpuBackend;
    let (n_vocab, n_embd) = (40, 256);
    let bytes = q1_bytes(n_vocab, n_embd);
    let ids = [0i32, 39, 7, 7];
    let want = cpu
        .get_rows(
            &cpu.upload_quant(&[n_vocab, n_embd], DType::Q1_0, &bytes).unwrap(),
            &ids,
        )
        .unwrap();
    let w = g
        .upload_weight("tok", bfi_gguf::TYPE_Q1_0, &[n_embd as u64, n_vocab as u64], &bytes)
        .unwrap();
    let ids_u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let got = read(&g, &g.get_rows(&w, &g.upload_u32(&ids_u)));
    assert_eq!(got, want.as_f32().unwrap());
}

#[test]
fn rms_norm_and_rope_neox_match_cpu() {
    let Some(g) = gpu() else { return };
    let cpu = CpuBackend;
    let (t, h, d) = (3usize, 4usize, 128usize);
    let x = pattern(t * h * d, 2);
    let w = pattern(d, 4);
    let want = cpu
        .rms_norm(
            &cpu.upload_f32(&[t, h, d], &x).unwrap(),
            &cpu.upload_f32(&[d], &w).unwrap(),
            1e-6,
        )
        .unwrap();
    let got_t = g.rms_norm(&g.upload_f32(&x), &g.upload_f32(&w), t * h, d, 1e-6);
    let got = read(&g, &got_t);
    assert!(max_rel_err(&got, want.as_f32().unwrap()) <= 1e-4);

    let params = RopeParams {
        n_dims: d,
        n_ctx: 4096,
        freq_base: 1_000_000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        n_rot: d,
        sections: None,
        neox: true,
    };
    let pos = [5i32, 6, 900];
    let mut want_r = want.clone();
    cpu.rope(&mut want_r, &pos, &params).unwrap();
    let inv: Vec<f32> = (0..d / 2)
        .map(|i| 1.0 / params.freq_base.powf((2 * i) as f32 / d as f32))
        .collect();
    let pos_u: Vec<u32> = pos.iter().map(|&p| p as u32).collect();
    g.rope_neox(&got_t, &g.upload_u32(&pos_u), &g.upload_f32(&inv), t, h, d);
    let got_r = read(&g, &got_t);
    assert!(max_rel_err(&got_r, want_r.as_f32().unwrap()) <= 1e-3);
}

#[test]
fn attention_with_f32_cache_matches_cpu() {
    attention_with_cache(false, 1e-3);
}

#[test]
fn attention_with_f16_cache_matches_cpu() {
    attention_with_cache(true, 5e-3);
}

fn attention_with_cache(f16: bool, tol: f32) {
    let Some(g) = gpu() else { return };
    let cpu = CpuBackend;
    let (hq, hkv, d) = (4usize, 2usize, 64usize);
    let (n_past, n_tok, ctx) = (5usize, 3usize, 16usize);
    let filled = n_past + n_tok;
    let q = pattern(n_tok * hq * d, 6);
    let k = pattern(filled * hkv * d, 7);
    let v = pattern(filled * hkv * d, 8);
    let want = cpu
        .attention(
            &cpu.upload_f32(&[n_tok, hq, d], &q).unwrap(),
            &cpu.upload_f32(&[filled, hkv, d], &k).unwrap(),
            &cpu.upload_f32(&[filled, hkv, d], &v).unwrap(),
            true,
        )
        .unwrap();
    // Rows past `filled` stay uninitialised: the kernels must never read them.
    let kc = g.empty_kv(ctx * hkv * d, f16);
    let vc = g.empty_kv(ctx * hkv * d, f16);
    g.write_kv(&g.upload_f32(&k), &kc, 0, filled * hkv * d);
    g.write_kv(&g.upload_f32(&v), &vc, 0, filled * hkv * d);
    let got = read(
        &g,
        &g.attention(
            &g.upload_f32(&q),
            &kc,
            &vc,
            n_tok,
            n_past,
            hq,
            hkv,
            d,
        ),
    );
    assert!(got.iter().all(|v| v.is_finite()));
    let err = max_rel_err(&got, want.as_f32().unwrap());
    assert!(err <= tol, "f16={f16} rel err {err}");
}

#[test]
fn f16_kv_append_at_an_offset_roundtrips() {
    let Some(g) = gpu() else { return };
    let src = pattern(8, 3);
    let kv = g.empty_kv(16, true);
    g.write_kv(&g.upload_f32(&src), &kv, 4, 8);
    let words = read(&g, &kv.words);
    for i in 0..8 {
        let w = words[(4 + i) / 2].to_bits();
        let h = ((w >> (16 * ((4 + i) % 2))) & 0xFFFF) as u16;
        let got = bfi_gguf::f16_to_f32(h);
        assert!((got - src[i]).abs() <= 1e-3 * src[i].abs().max(1.0), "{i}: {got} vs {}", src[i]);
    }
}

#[test]
fn swiglu_add_copy_match_cpu() {
    let Some(g) = gpu() else { return };
    let cpu = CpuBackend;
    let a = pattern(300, 1);
    let b = pattern(300, 2);
    let want = cpu
        .swiglu(
            &cpu.upload_f32(&[300], &a).unwrap(),
            &cpu.upload_f32(&[300], &b).unwrap(),
        )
        .unwrap();
    let ga = g.upload_f32(&a);
    let gb = g.upload_f32(&b);
    let got = read(&g, &g.swiglu(&ga, &gb));
    assert!(max_rel_err(&got, want.as_f32().unwrap()) <= 1e-4);

    g.add_inplace(&ga, &gb);
    let sum = read(&g, &ga);
    for i in 0..300 {
        assert!((sum[i] - (a[i] + b[i])).abs() < 1e-6);
    }

    let dst = g.upload_f32(&vec![0.0; 10]);
    g.copy(&gb, 290, &dst, 2, 5);
    let c = read(&g, &dst);
    assert_eq!(&c[2..7], &b[290..295]);
    assert_eq!(c[0], 0.0);
}

#[test]
fn checked_runs_work_and_a_lost_device_refuses() {
    let Some(shared) = gpu() else { return };
    // Its own health record: marking the process's one shared device lost would
    // fail every other resident test in this binary.
    let g = shared.with_own_health();
    // A clean upload: the closure runs and no device error is recorded.
    let t = block_on(g.checked("upload", || g.upload_f32(&[1.0, 2.0]))).expect("checked");
    assert_eq!(read(&g, &t), vec![1.0, 2.0]);
    block_on(g.drain()).expect("drain");
    assert!(g.lost().is_none());

    g.health.mark_lost("Unknown: test".into());
    let clone = g.clone();
    assert_eq!(clone.lost().as_deref(), Some("Unknown: test"), "clones share the flag");
    let err = block_on(clone.checked("upload", || ())).err().expect("lost");
    assert!(matches!(err, ResidentError::DeviceLost(_)), "{err}");
    assert!(matches!(block_on(clone.drain()), Err(ResidentError::DeviceLost(_))));
    assert!(shared.lost().is_none(), "the shared device is untouched");
}
