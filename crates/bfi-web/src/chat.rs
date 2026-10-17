//! `BfiEngine.chat`: one OpenAI `chat.completions` request on the loaded model.
//!
//! Request parsing, prompt rendering (GGUF chat template, tools), sampling,
//! stop strings, tool-call detection and chunk JSON all come from
//! `bfi_runtime::openai`; this loop only drives the GPU: enqueue a forward,
//! await the logits readback (the single host sync per token, which also yields
//! to the browser event loop), step the turn, emit its chunks.

use std::rc::Rc;

use bfi_runtime::openai::{ChatRequest, ChatTurn, Step};
use js_sys::{Function, Object};
use wasm_bindgen::JsValue;

use crate::errors::{classify, error, Kind};
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
    let mut logits = loaded
        .model
        .forward(&ids[keep..])
        .map_err(|e| classify(Kind::Engine, &e))?;
    loaded.cached.extend_from_slice(&ids[keep..]);

    let gpu = loaded.model.gpu().clone();
    loop {
        if js::aborted(signal) || cancelled() {
            return Err(error(Kind::Aborted, ""));
        }
        let values = gpu
            .read_f32(&logits)
            .await
            .map_err(|e| error(Kind::Engine, e))?;
        if js::aborted(signal) || cancelled() {
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

/// Longest common prefix of the cached ids and a new prompt.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}
