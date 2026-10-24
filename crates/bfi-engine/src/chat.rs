//! `BfiEngine.chat`: one OpenAI `chat.completions` request on the loaded model.
//!
//! Request parsing, prompt rendering (GGUF chat template, tools), sampling, stop
//! strings, tool-call detection and chunk JSON all come from `bfi_runtime::openai`; this
//! loop only drives the backend ([`Runner`]): feed the prompt, await each step's logits,
//! step the turn, emit its chunks. Long prompts are fed in groups of
//! [`Runner::prefill_group`] tokens with a yield between groups, so an abort lands
//! mid-prompt on either backend.

use std::sync::Arc;

use bfi_runtime::openai::{ChatRequest, ChatTurn, Step};

use crate::errors::{error, EngineError, Kind};
use crate::host::Host;
use crate::loader::Loaded;
use crate::runner::Runner;

/// Run one request. `on_chunk` receives each chunk JSON (when `stream: true`); `stop` is
/// polled before every step (the host's abort, `unload`, a lost device). Resolves with
/// the final completion JSON (`usage`, `finish_reason`, `tool_calls`).
pub async fn chat(
    loaded: &mut Loaded,
    request_json: &str,
    host: &impl Host,
    on_chunk: &mut dyn FnMut(&str) -> Result<(), EngineError>,
    stop: &dyn Fn() -> bool,
) -> Result<String, EngineError> {
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
    let created = host.now_secs();
    let mut turn = ChatTurn::new(
        &plan,
        &ids,
        loaded.model.n_ctx(),
        format!("chatcmpl-bfi-{created}"),
        created,
    )
    .map_err(|e| error(Kind::ContextExceeded, e))?;
    let pieces: Arc<Vec<String>> = if turn.needs_pieces() {
        loaded.pieces()
    } else {
        Arc::new(Vec::new())
    };

    // Reuse the KV cache for the prompt prefix the previous turn already fed;
    // always re-feed at least the last prompt token to get fresh logits.
    let keep = common_prefix(&loaded.cached, &ids).min(ids.len() - 1);
    loaded.model.rewind(keep);
    // A backend that cannot rewind partway (recurrent state) restarts from empty.
    let keep = loaded.model.n_past();
    loaded.cached.truncate(keep);
    let mut logits = prefill(&mut loaded.model, &mut loaded.cached, &ids[keep..], host, stop).await?;

    loop {
        if stop() {
            return Err(error(Kind::Aborted, ""));
        }
        let (step, chunks) = turn.step(&logits, &loaded.tokenizer, &pieces);
        for c in chunks {
            on_chunk(&c)?;
        }
        match step {
            Step::Feed(token) => {
                logits = loaded.model.logits(&[token], host).await?;
                loaded.cached.push(token);
                if stop() {
                    return Err(error(Kind::Aborted, ""));
                }
            }
            Step::Done => break,
        }
    }
    debug_assert_eq!(loaded.cached.len(), loaded.model.n_past());
    Ok(turn.completion_json())
}

/// Feed `prompt` and return its last token's logits. All but the final group are fed
/// without logits, each followed by a yield; `stop` is checked before every group.
/// `cached` tracks exactly what reached the KV cache, so an abort leaves it consistent.
async fn prefill(
    model: &mut Runner,
    cached: &mut Vec<u32>,
    prompt: &[u32],
    host: &impl Host,
    stop: &dyn Fn() -> bool,
) -> Result<Vec<f32>, EngineError> {
    let group = model.prefill_group();
    // The final group carries the logits: the remainder, or a full group.
    let tail = match prompt.len() % group {
        0 => group,
        r => r,
    }
    .min(prompt.len());
    let (lead, last) = prompt.split_at(prompt.len() - tail);
    for part in lead.chunks(group) {
        if stop() {
            return Err(error(Kind::Aborted, ""));
        }
        model.prefill(part, host).await?;
        cached.extend_from_slice(part);
    }
    if stop() {
        return Err(error(Kind::Aborted, ""));
    }
    let logits = model.logits(last, host).await?;
    cached.extend_from_slice(last);
    Ok(logits)
}

/// Longest common prefix of the cached ids and a new prompt.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}
