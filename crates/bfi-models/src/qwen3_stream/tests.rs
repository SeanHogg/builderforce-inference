//! The streaming CPU load (the browser engine without WebGPU) against the file load the
//! native `bfi` runs: same backend, same `assemble`, so the logits must be IDENTICAL.

use std::sync::Arc;

use bfi_core::{Backend, CpuBackend};
use bfi_gguf::{GgufFile, TensorStream};

use super::*;
use crate::session::CONTEXT_EXCEEDED;
use crate::test_fixtures::bonsai_toy;
use crate::{load_model, SeqState};

fn cpu() -> Arc<dyn Backend> {
    Arc::new(CpuBackend)
}

/// Stream `bytes` into a loader in `chunk`-byte pieces, as `ReadableStream` chunks arrive.
fn streamed(bytes: &[u8], chunk: usize) -> Qwen3StreamLoader {
    let mut stream = TensorStream::new();
    let mut loader: Option<Qwen3StreamLoader> = None;
    let feed = |stream: &TensorStream, loader: &mut Option<Qwen3StreamLoader>, done: Vec<bfi_gguf::StreamedTensor>| {
        if loader.is_none() {
            if let Some(h) = stream.header() {
                *loader = Some(Qwen3StreamLoader::new(cpu(), h).unwrap());
            }
        }
        for t in done {
            loader.as_mut().expect("header precedes tensors").accept(t).unwrap();
        }
    };
    for c in bytes.chunks(chunk) {
        let done = stream.push(c).unwrap();
        feed(&stream, &mut loader, done);
    }
    let tail = stream.flush_header().unwrap();
    feed(&stream, &mut loader, tail);
    stream.finish().unwrap();
    loader.unwrap()
}

fn file_logits(bytes: &[u8], prompt: &[u32], step: u32) -> (Vec<f32>, Vec<f32>) {
    let file = GgufFile::from_bytes("toy.gguf", bytes.to_vec()).unwrap();
    let model = load_model(&file, cpu()).unwrap();
    let mut state = SeqState::allocate(&model.state_layout());
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let a = model.forward(prompt, &pos, &mut state).unwrap().values;
    let b = model.forward(&[step], &[prompt.len() as i32], &mut state).unwrap().values;
    (a, b)
}

#[test]
fn streamed_cpu_load_matches_the_file_load_exactly() {
    for tied in [false, true] {
        let bytes = bonsai_toy(tied);
        let prompt = [1u32, 5, 9, 3, 14];
        let (want_prompt, want_step) = file_logits(&bytes, &prompt, 7);
        for chunk in [7, 4096, bytes.len()] {
            let loader = streamed(&bytes, chunk);
            assert_eq!(
                loader.kv_shape(),
                KvShape { n_layer: 2, n_head_kv: 1, head_dim: 64, trained_ctx: 32 }
            );
            let mut s = loader.finish(32).unwrap();
            assert_eq!((s.architecture(), s.n_vocab(), s.n_ctx()), ("qwen3", 16, 32));
            assert!(s.params() > 0);
            assert_eq!(s.forward(&prompt).unwrap(), want_prompt, "tied={tied} chunk={chunk}");
            assert_eq!(s.forward(&[7]).unwrap(), want_step, "tied={tied} chunk={chunk}");
            assert_eq!(s.n_past(), prompt.len() + 1);
        }
    }
}

#[test]
fn the_kv_budget_caps_the_served_context() {
    let bytes = bonsai_toy(true);
    let mut s = streamed(&bytes, 4096).finish(8).unwrap();
    assert_eq!(s.n_ctx(), 8);
    s.forward(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let over = s.forward(&[1]).unwrap_err().to_string();
    assert!(over.starts_with(CONTEXT_EXCEEDED), "{over}");
    // Never above the trained window.
    assert_eq!(streamed(&bytes, 4096).finish(10_000).unwrap().n_ctx(), 32);
}

#[test]
fn rewind_reuses_the_cached_prefix() {
    let bytes = bonsai_toy(false);
    let mut whole = streamed(&bytes, 4096).finish(32).unwrap();
    let want = whole.forward(&[1, 5, 9, 3]).unwrap();

    let mut s = streamed(&bytes, 4096).finish(32).unwrap();
    s.forward(&[1, 5, 2, 2, 2]).unwrap();
    s.rewind(2);
    assert_eq!(s.n_past(), 2);
    assert_eq!(s.forward(&[9, 3]).unwrap(), want);
    // A rewind past the end is a no-op.
    s.rewind(100);
    assert_eq!(s.n_past(), 4);
}

#[test]
fn rejects_out_of_vocab_tokens_and_other_architectures() {
    let bytes = bonsai_toy(true);
    let mut s = streamed(&bytes, 4096).finish(32).unwrap();
    assert!(s.forward(&[16]).unwrap_err().to_string().contains("n_vocab"));
    assert!(s.forward(&[]).is_err());

    let mut meta = std::collections::BTreeMap::new();
    meta.insert("general.architecture".into(), bfi_gguf::MetaValue::String("llama".into()));
    let other = bfi_gguf::writer::write_gguf(&meta, &[]).unwrap();
    let mut stream = TensorStream::new();
    stream.push(&other).unwrap();
    stream.flush_header().unwrap();
    let err = Qwen3StreamLoader::new(cpu(), stream.header().unwrap()).err().unwrap().to_string();
    assert!(err.contains("llama"), "{err}");
}
