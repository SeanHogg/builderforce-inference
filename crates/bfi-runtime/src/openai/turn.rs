//! [`ChatTurn`]: one assistant turn's per-token state (sampling, stop strings,
//! tool-call detection, chunk JSON).

use serde_json::{json, Value};

use super::stream_text::StreamText;
use super::{
    chunk_value, completion_value, tool_calls_value, usage_value, ChatPlan, Finish, END_OF_TURN_PIECES,
    TOOL_CALL_OPEN,
};
use crate::sample::{apply_constraint, sample, SampleParams};
use crate::tokenize::Tokenizer;
use crate::tools::{parse_tool_calls, ParsedToolCall};

/// What the caller does after [`ChatTurn::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Feed this token to the model, then call `step` with its logits.
    Feed(u32),
    /// The turn is over; read [`ChatTurn::completion_json`].
    Done,
}

/// One assistant turn: sampling, stop strings, tool-call detection, chunk JSON.
pub struct ChatTurn {
    id: String,
    created: u64,
    model: String,
    params: SampleParams,
    stop: Vec<String>,
    tools_on: bool,
    stream: bool,
    include_usage: bool,
    prompt_tokens: usize,
    max_tokens: usize,
    history: Vec<u32>,
    generated: Vec<u32>,
    /// Decoded `generated`, grown one token at a time (never re-decoded).
    text: StreamText,
    /// Longest stop string (bytes): how far back a new token can complete one.
    max_stop_len: usize,
    emitted: usize,
    sent_role: bool,
    finish: Option<Finish>,
    tool_calls: Option<Vec<ParsedToolCall>>,
}

impl ChatTurn {
    /// `prompt_ids` are what the model has been (or will be) fed; `n_ctx` is the
    /// KV capacity. Errors with a `context_exceeded:` message when the prompt
    /// leaves no room to generate.
    pub fn new(plan: &ChatPlan, prompt_ids: &[u32], n_ctx: usize, id: String, created: u64) -> Result<Self, String> {
        if prompt_ids.len() >= n_ctx {
            return Err(format!(
                "context_exceeded: prompt is {} tokens; the loaded context is {n_ctx}",
                prompt_ids.len()
            ));
        }
        let room = n_ctx - prompt_ids.len();
        let max_tokens = match plan.params.max_tokens {
            0 => room,
            n => (n as usize).min(room),
        };
        Ok(Self {
            id,
            created,
            model: plan.model.clone(),
            params: plan.params.clone(),
            stop: plan.stop.clone(),
            max_stop_len: plan.stop.iter().map(String::len).max().unwrap_or(0),
            tools_on: plan.tools_on,
            stream: plan.stream,
            include_usage: plan.include_usage,
            prompt_tokens: prompt_ids.len(),
            max_tokens,
            history: prompt_ids.to_vec(),
            generated: Vec::new(),
            text: StreamText::default(),
            emitted: 0,
            sent_role: false,
            finish: None,
            tool_calls: None,
        })
    }

    /// Whether [`ChatTurn::step`] needs every token's decoded piece (constrained decode).
    pub fn needs_pieces(&self) -> bool {
        self.params.grammar.is_some() || self.params.json_object || self.params.tool_choice.constrains()
    }

    /// Decide the next token from the logits of the last fed token. Returns the
    /// next action and the chunk JSON strings to emit (empty unless streaming).
    /// `pieces[id]` is `tok.decode(&[id])`, required when [`Self::needs_pieces`].
    pub fn step(&mut self, logits: &[f32], tok: &Tokenizer, pieces: &[String]) -> (Step, Vec<String>) {
        if self.finish.is_some() {
            return (Step::Done, Vec::new());
        }
        let mut masked = logits.to_vec();
        if self.needs_pieces() {
            apply_constraint(&mut masked, &self.generated, &self.params, |id| {
                pieces.get(id as usize).cloned().unwrap_or_default()
            });
        }
        let next = sample(&masked, &self.history, &self.params);
        let end_of_turn = tok.eos() == Some(next)
            || tok
                .decode(&[next])
                .map(|p| END_OF_TURN_PIECES.contains(&p.as_str()))
                .unwrap_or(false);
        let mut feed = None;
        if end_of_turn {
            self.finish = Some(Finish::Stop);
        } else {
            self.generated.push(next);
            self.history.push(next);
            let changed_from = self.text.push(tok, next);
            if self.tools_on {
                if let Some(calls) = parse_tool_calls(self.text.as_str()) {
                    self.tool_calls = Some(calls);
                    self.finish = Some(Finish::ToolCalls);
                }
            }
            if self.finish.is_none() {
                if let Some(at) = self.first_stop(changed_from) {
                    self.text.truncate(at);
                    self.finish = Some(Finish::Stop);
                } else if self.generated.len() >= self.max_tokens {
                    self.finish = Some(Finish::Length);
                } else {
                    feed = Some(next);
                }
            }
        }
        let chunks = self.drain_chunks();
        match feed {
            Some(t) => (Step::Feed(t), chunks),
            None => (Step::Done, chunks),
        }
    }

    /// End the turn early (abort / host error) with `finish_reason: stop`.
    pub fn stop_now(&mut self) -> Vec<String> {
        if self.finish.is_none() {
            self.finish = Some(Finish::Stop);
        }
        self.drain_chunks()
    }

    pub fn finish(&self) -> Option<Finish> {
        self.finish
    }
    pub fn generated(&self) -> &[u32] {
        &self.generated
    }
    pub fn text(&self) -> &str {
        self.text.as_str()
    }

    /// The final `chat.completion` JSON (`usage`, `finish_reason`, `tool_calls`).
    pub fn completion_json(&self) -> String {
        let finish = self.finish.unwrap_or(Finish::Stop);
        let content = if self.tool_calls.is_some() { None } else { Some(self.text()) };
        completion_value(
            &self.id,
            self.created,
            &self.model,
            content,
            self.tool_calls.as_deref(),
            finish.as_str(),
            usage_value(self.prompt_tokens, self.generated.len()),
        )
        .to_string()
    }

    /// Earliest stop-string match. Only text from `changed_from` (where this
    /// token's bytes landed) back by the longest stop string can hold a new
    /// match: anything earlier was already searched on a previous token.
    fn first_stop(&self, changed_from: usize) -> Option<usize> {
        if self.stop.is_empty() {
            return None;
        }
        let text = self.text();
        let mut from = changed_from.saturating_sub(self.max_stop_len).min(text.len());
        while !text.is_char_boundary(from) {
            from -= 1;
        }
        let window = &text[from..];
        self.stop.iter().filter_map(|s| window.find(s.as_str())).min().map(|at| from + at)
    }

    /// Byte length of `text` that is safe to stream now.
    fn streamable_end(&self) -> usize {
        let text = self.text();
        let mut end = text.len();
        if self.tools_on {
            let t = text.trim_start();
            let lead = text.len() - t.len();
            // A reply that may still turn into a JSON / fenced tool call is held whole.
            if t.is_empty() || t.starts_with('{') || t.starts_with("```") || "```".starts_with(t) {
                return lead;
            }
            if let Some(at) = text.find(TOOL_CALL_OPEN) {
                end = end.min(at);
            }
        }
        if self.finish.is_some() {
            return end;
        }
        let mut markers: Vec<&str> = self.stop.iter().map(String::as_str).collect();
        if self.tools_on {
            markers.push(TOOL_CALL_OPEN);
        }
        let held = partial_marker_suffix(&text[..end], &markers);
        end -= held;
        // A trailing U+FFFD is usually half a multi-byte character; the next token completes it.
        while text[..end].ends_with('\u{FFFD}') {
            end -= '\u{FFFD}'.len_utf8();
        }
        end
    }

    fn drain_chunks(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        let end = self.streamable_end().max(self.emitted);
        if end > self.emitted && self.tool_calls.is_none() {
            let delta = self.text()[self.emitted..end].to_string();
            if self.stream {
                out.push(self.content_chunk(&delta));
            }
            self.emitted = end;
        }
        if let Some(finish) = self.finish {
            if self.stream {
                if let Some(calls) = &self.tool_calls {
                    let mut delta = json!({ "tool_calls": tool_calls_value(calls) });
                    if !self.sent_role {
                        delta["role"] = Value::from("assistant");
                        self.sent_role = true;
                    }
                    out.push(chunk_value(&self.id, self.created, &self.model, delta, None).to_string());
                } else if self.emitted < self.text().len() {
                    // Held text that never became a tool call.
                    let rest = self.text()[self.emitted..].to_string();
                    out.push(self.content_chunk(&rest));
                }
                out.push(chunk_value(&self.id, self.created, &self.model, json!({}), Some(finish.as_str())).to_string());
                if self.include_usage {
                    out.push(
                        json!({
                            "id": self.id,
                            "object": "chat.completion.chunk",
                            "created": self.created,
                            "model": self.model,
                            "choices": [],
                            "usage": usage_value(self.prompt_tokens, self.generated.len())
                        })
                        .to_string(),
                    );
                }
            }
            self.emitted = self.text().len();
        }
        out
    }

    fn content_chunk(&mut self, delta: &str) -> String {
        let mut d = json!({ "content": delta });
        if !self.sent_role {
            d["role"] = Value::from("assistant");
            self.sent_role = true;
        }
        chunk_value(&self.id, self.created, &self.model, d, None).to_string()
    }
}

/// Longest suffix of `text` that is a proper prefix of any marker (bytes).
pub(super) fn partial_marker_suffix(text: &str, markers: &[&str]) -> usize {
    let mut best = 0;
    for m in markers {
        for (i, _) in m.char_indices().skip(1) {
            let prefix = &m[..i];
            if i > best && text.ends_with(prefix) {
                best = i;
            }
        }
    }
    best
}
