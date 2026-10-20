//! `BfiEngine.chat`: one OpenAI `chat.completions` request on the loaded model.
//!
//! Request parsing, prompt rendering (GGUF chat template, tools), sampling,
//! stop strings, tool-call detection and chunk JSON all come from
//! `bfi_runtime::openai`; this loop only drives the GPU: enqueue a forward,
//! await the logits readback (the single host sync per token, which also yields
//! to the browser event loop), step the turn, emit its chunks. Long prompts are
//! fed in groups of [`Qwen3Resident::prefill_yield_tokens`] with a one-word
//! readback between groups, so an abort lands mid-prompt.

use std::rc::Rc;

use bfi_core::resident::DeviceTensor;
use bfi_models::Qwen3Resident;
use bfi_runtime::openai::{ChatRequest, ChatTurn, Step};
use js_sys::{Function, Object};
use wasm_bindgen::JsValue;

use crate::errors::{classify, error, gpu_error, Kind};
use crate::js;
use crate::loader::Loaded;

pub async fn chat(
    loaded: &mut Loaded,
    request_json: &str,
    on_chunk: &Function,
    signal: Option<&Object>,
    cancelled: impl Fn() -> bool,
) -> Result<String, JsValue> {
    let req = ChatRequest::from_json(request_json).map_err(|e| error(Kind::InvalidRequest, e))?;
    let plan = req
        .plan(loaded.template.as_deref())
        .map_err(|e| error(Kind::InvalidRequest, e))?;
    let ids = loaded
        .tokenizer
        .encode(&plan.prompt, true)
        .map_err(|e| error(Kind::Engine, format!("tokenize: {e:#}")))?;
    if ids.is_empty() {
        return Err(error(Kind::InvalidRequest, "the rendered prompt is empty"));
    }
    let created = js::now_secs();
    let mut turn = ChatTurn::new(
        &plan,
        &ids,
        loaded.model.n_ctx(),
        format!("chatcmpl-bfi-{created}"),
        created,
    )
    .map_err(|e| error(Kind::ContextExceeded, e))?;
    let pieces: Rc<Vec<String>> = if turn.needs_pieces() {
        loaded.pieces()
    } else {
        Rc::new(Vec::new())
    };

    // Reuse the KV cache for the prompt prefix the previous turn already fed;
    // always re-feed at least the last prompt token to get fresh logits.
    let keep = common_prefix(&loaded.cached, &ids).min(ids.len() - 1);
    loaded.model.rewind(keep);
    loaded.cached.truncate(keep);
    let stop = || js::aborted(signal) || cancelled();
    let mut logits = prefill(&mut loaded.model, &mut loaded.cached, &ids[keep..], &stop).await?;

    let gpu = loaded.model.gpu().clone();
    loop {
        if stop() {
            return Err(error(Kind::Aborted, ""));
        }
        let values = gpu
            .read_f32(&logits)
            .await
            .map_err(|e| gpu_error(Kind::Engine, &e))?;
        if stop() {
            return Err(error(Kind::Aborted, ""));
        }
        let (step, chunks) = turn.step(&values, &loaded.tokenizer, &pieces);
        for c in chunks {
            on_chunk
                .call1(&JsValue::NULL, &JsValue::from_str(&c))
                .map_err(|e| error(Kind::Engine, format!("onChunk threw: {e:?}")))?;
        }
        match step {
            Step::Feed(token) => {
                logits = loaded
                    .model
                    .forward(&[token])
                    .map_err(|e| classify(Kind::Engine, &e))?;
                loaded.cached.push(token);
            }
            Step::Done => break,
        }
    }
    debug_assert_eq!(loaded.cached.len(), loaded.model.n_past());
    Ok(turn.completion_json())
}

/// Feed `prompt` and return its last token's logits. All but the final group
/// are enqueued without logits, each followed by a one-word readback that
/// yields to the event loop; `stop` is checked before every group. `cached`
/// tracks exactly what reached the KV cache, so an abort leaves it consistent.
async fn prefill(
    model: &mut Qwen3Resident,
    cached: &mut Vec<u32>,
    prompt: &[u32],
    stop: &impl Fn() -> bool,
) -> Result<DeviceTensor, JsValue> {
    let group = model.prefill_yield_tokens().max(1);
    // The final group carries the logits: the remainder, or a full group.
    let tail = match prompt.len() % group {
        0 => group,
        r => r,
    }
    .min(prompt.len());
    let (lead, last) = prompt.split_at(prompt.len() - tail);
    let gpu = model.gpu().clone();
    for part in lead.chunks(group) {
        if stop() {
            return Err(error(Kind::Aborted, ""));
        }
        model.prefill(part).map_err(|e| classify(Kind::Engine, &e))?;
        cached.extend_from_slice(part);
        gpu.drain().await.map_err(|e| gpu_error(Kind::Engine, &e))?;
    }
    if stop() {
        return Err(error(Kind::Aborted, ""));
    }
    let logits = model.forward(last).map_err(|e| classify(Kind::Engine, &e))?;
    cached.extend_from_slice(last);
    Ok(logits)
}

/// Longest common prefix of the cached ids and a new prompt.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}
