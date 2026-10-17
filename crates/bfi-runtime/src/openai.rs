//! OpenAI `chat.completions`: request → prompt plan, and the per-token turn
//! state that turns sampled tokens into `chat.completion.chunk` JSON.
//!
//! Host-agnostic and synchronous: the caller owns the model and feeds logits in.
//! `bfi-web` drives it from an async loop (one awaited GPU readback per token).
//! Nothing here reads the clock — `created` is passed in (wasm32 has no
//! `SystemTime`).

use serde::Deserialize;
use serde_json::{json, Value};

use crate::chat::{chatml_prompt, is_chatml, naive_prompt, render_template, ChatMessage};
use crate::sample::{apply_constraint, sample, SampleParams};
use crate::tokenize::Tokenizer;
use crate::tools::{apply_tools, parse_tool_calls, prepend_tools_block, ParsedToolCall, ToolSpec};

/// Sampling temperature when the request omits it (same as `bfi serve`).
pub const DEFAULT_TEMPERATURE: f32 = 0.8;

/// Decoded single-token pieces that end the assistant turn even when they are
/// not `tokenizer.ggml.eos_token_id` (ChatML / Llama-3 / SentencePiece).
pub const END_OF_TURN_PIECES: &[&str] = &["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"];

/// Content after this marker is a tool call, never streamed as text.
const TOOL_CALL_OPEN: &str = "<tool_call>";

/// One OpenAI / Anthropic-style message as clients send it.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct IncomingMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: Value,
    /// Assistant tool calls (OpenAI).
    #[serde(default)]
    pub tool_calls: Option<Value>,
    /// Tool result id (OpenAI `role: tool`).
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

impl IncomingMessage {
    pub fn role_or_user(&self) -> &str {
        if self.role.is_empty() {
            "user"
        } else {
            &self.role
        }
    }

    /// Flattened text: content blocks, then tool calls / tool-call id.
    pub fn text(&self) -> String {
        let mut s = content_text(&self.content);
        if let Some(calls) = &self.tool_calls {
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str(&calls.to_string());
        }
        if let Some(id) = &self.tool_call_id {
            if !s.is_empty() {
                s.push(' ');
            }
            s.push_str(id);
        }
        s
    }

    /// The message as a chat template sees it (`content` flattened to text,
    /// `tool_calls` / `tool_call_id` / `name` kept structured).
    pub fn template_value(&self) -> Value {
        let mut m = json!({ "role": self.role_or_user(), "content": content_text(&self.content) });
        if let Some(calls) = &self.tool_calls {
            let calls: Vec<Value> = calls
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|c| {
                    // Qwen templates read `tool_call.function` or the call itself.
                    let f = c.get("function").cloned().unwrap_or(c.clone());
                    json!({ "type": "function", "function": f, "name": f.get("name"), "arguments": f.get("arguments") })
                })
                .collect();
            m["tool_calls"] = Value::from(calls);
        }
        if let Some(id) = &self.tool_call_id {
            m["tool_call_id"] = Value::from(id.clone());
        }
        if let Some(n) = &self.name {
            m["name"] = Value::from(n.clone());
        }
        m
    }
}

pub fn to_chat_messages(raw: &[IncomingMessage]) -> Vec<ChatMessage> {
    raw.iter()
        .map(|m| ChatMessage {
            role: m.role_or_user().to_string(),
            content: m.text(),
        })
        .collect()
}

/// String content, OpenAI content parts, or Anthropic content blocks → text.
pub fn content_text(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(items) => items.iter().map(content_block_text).collect::<Vec<_>>().join(""),
        Value::Object(_) => content_block_text(v),
        other => other.to_string(),
    }
}

/// OpenAI / Anthropic content blocks, including `tool_use` and `tool_result`.
fn content_block_text(item: &Value) -> String {
    match item {
        Value::String(s) => s.clone(),
        Value::Object(o) => {
            let ty = o.get("type").and_then(Value::as_str).unwrap_or("");
            if let Some(s) = o.get("text").and_then(Value::as_str) {
                return s.to_string();
            }
            match ty {
                "tool_use" => {
                    let name = o.get("name").and_then(Value::as_str).unwrap_or("tool");
                    let id = o.get("id").and_then(Value::as_str).unwrap_or("");
                    let input = o.get("input").cloned().unwrap_or(Value::Null);
                    format!("[tool_use {id} {name} {input}]")
                }
                "tool_result" => {
                    let id = o.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                    let inner = o.get("content").map(content_text).unwrap_or_default();
                    format!("[tool_result {id} {inner}]")
                }
                _ => o.get("content").map(content_text).unwrap_or_default(),
            }
        }
        other => other.to_string(),
    }
}

/// The OpenAI `chat.completions` request fields the engines honour.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub messages: Vec<IncomingMessage>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    /// `"stop"` or `["a", "b"]`.
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    /// `{ "include_usage": true }` → a final usage-only chunk.
    #[serde(default)]
    pub stream_options: Option<Value>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
}

/// Everything a generate loop needs from one request.
#[derive(Debug, Clone)]
pub struct ChatPlan {
    pub model: String,
    pub prompt: String,
    pub params: SampleParams,
    pub tools_on: bool,
    pub stop: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
}

impl ChatRequest {
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("invalid chat.completions request: {e}"))
    }

    /// Validate the request and render its prompt with `template` (the GGUF's
    /// `tokenizer.chat_template`, if any).
    pub fn plan(&self, template: Option<&str>) -> Result<ChatPlan, String> {
        if self.messages.is_empty() {
            return Err("messages must be a non-empty array".into());
        }
        let mut params = SampleParams {
            max_tokens: self.max_completion_tokens.or(self.max_tokens).unwrap_or(0),
            temperature: self.temperature.unwrap_or(DEFAULT_TEMPERATURE).max(0.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k.unwrap_or(0),
            min_p: self.min_p.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            seed: self.seed.unwrap_or(0),
            ..SampleParams::default()
        };
        if let Some(rf) = &self.response_format {
            match rf.get("type").and_then(Value::as_str).unwrap_or("text") {
                "text" => {}
                "json_object" | "json_schema" => params.json_object = true,
                other => return Err(format!("unsupported response_format.type '{other}'")),
            }
        }
        let (tools, tools_on) = apply_tools(&mut params, self.tools.as_ref(), self.tool_choice.as_ref())?;
        let stop = parse_stop(self.stop.as_ref())?;
        let prompt = self.render(template, &tools, tools_on, &params);
        let include_usage = self
            .stream_options
            .as_ref()
            .and_then(|o| o.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(ChatPlan {
            model: self.model.clone(),
            prompt,
            params,
            tools_on,
            stop,
            stream: self.stream,
            include_usage,
        })
    }

    fn render(&self, template: Option<&str>, tools: &[ToolSpec], tools_on: bool, params: &SampleParams) -> String {
        if let Some(t) = template.filter(|t| !t.trim().is_empty()) {
            let msgs: Vec<Value> = self.messages.iter().map(IncomingMessage::template_value).collect();
            let tools_val = if tools_on { self.tools.as_ref() } else { None };
            if let Ok(p) = render_template(t, minijinja::Value::from_serialize(&msgs), tools_val, true) {
                return p;
            }
        }
        let mut messages = to_chat_messages(&self.messages);
        if tools_on {
            prepend_tools_block(&mut messages, tools, &params.tool_choice);
        }
        match template {
            Some(t) if is_chatml(t) => chatml_prompt(&messages),
            _ => naive_prompt(&messages, true),
        }
    }
}

fn parse_stop(v: Option<&Value>) -> Result<Vec<String>, String> {
    Ok(match v {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| x.as_str().map(str::to_string).ok_or("stop entries must be strings"))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err("stop must be a string or an array of strings".into()),
    }
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect())
}

/// Why generation ended (`finish_reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
    ToolCalls,
}

impl Finish {
    pub fn as_str(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
        }
    }
}

/// `chat.completion.chunk` with one choice.
pub fn chunk_value(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
    })
}

pub fn usage_value(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    })
}

pub fn tool_calls_value(calls: &[ParsedToolCall]) -> Value {
    Value::from(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut v = c.openai_json();
                v["index"] = Value::from(i);
                v
            })
            .collect::<Vec<_>>(),
    )
}

/// Final `chat.completion` object.
pub fn completion_value(
    id: &str,
    created: u64,
    model: &str,
    content: Option<&str>,
    tool_calls: Option<&[ParsedToolCall]>,
    finish: &str,
    usage: Value,
) -> Value {
    let mut message = json!({ "role": "assistant", "content": content });
    if let Some(calls) = tool_calls {
        message["tool_calls"] = tool_calls_value(calls);
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "message": message, "finish_reason": finish }],
        "usage": usage
    })
}

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
    text: String,
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
            tools_on: plan.tools_on,
            stream: plan.stream,
            include_usage: plan.include_usage,
            prompt_tokens: prompt_ids.len(),
            max_tokens,
            history: prompt_ids.to_vec(),
            generated: Vec::new(),
            text: String::new(),
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
            self.text = tok.decode(&self.generated).unwrap_or_default();
            if self.tools_on {
                if let Some(calls) = parse_tool_calls(&self.text) {
                    self.tool_calls = Some(calls);
                    self.finish = Some(Finish::ToolCalls);
                }
            }
            if self.finish.is_none() {
                if let Some(at) = self.first_stop() {
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
        &self.text
    }

    /// The final `chat.completion` JSON (`usage`, `finish_reason`, `tool_calls`).
    pub fn completion_json(&self) -> String {
        let finish = self.finish.unwrap_or(Finish::Stop);
        let content = if self.tool_calls.is_some() { None } else { Some(self.text.as_str()) };
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

    fn first_stop(&self) -> Option<usize> {
        self.stop.iter().filter_map(|s| self.text.find(s.as_str())).min()
    }

    /// Byte length of `text` that is safe to stream now.
    fn streamable_end(&self) -> usize {
        let mut end = self.text.len();
        if self.tools_on {
            let t = self.text.trim_start();
            let lead = self.text.len() - t.len();
            // A reply that may still turn into a JSON / fenced tool call is held whole.
            if t.is_empty() || t.starts_with('{') || t.starts_with("```") || "```".starts_with(t) {
                return lead;
            }
            if let Some(at) = self.text.find(TOOL_CALL_OPEN) {
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
        let held = partial_marker_suffix(&self.text[..end], &markers);
        end -= held;
        // A trailing U+FFFD is usually half a multi-byte character; the next token completes it.
        while self.text[..end].ends_with('\u{FFFD}') {
            end -= '\u{FFFD}'.len_utf8();
        }
        end
    }

    fn drain_chunks(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        let end = self.streamable_end().max(self.emitted);
        if end > self.emitted && self.tool_calls.is_none() {
            let delta = self.text[self.emitted..end].to_string();
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
                } else if self.emitted < self.text.len() {
                    // Held text that never became a tool call.
                    let rest = self.text[self.emitted..].to_string();
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
            self.emitted = self.text.len();
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
fn partial_marker_suffix(text: &str, markers: &[&str]) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Vocab where token id == index; ids 0..=4 are words, 5 is `<|im_end|>`.
    fn tok() -> Tokenizer {
        Tokenizer::from_parts(
            ["Hel", "lo", " wor", "ld", "{\"name\":\"f\",\"arguments\":{}}", "<|im_end|>", "ST", "OP"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            vec![],
        )
    }

    fn one_hot(id: usize, n: usize) -> Vec<f32> {
        let mut v = vec![0.0; n];
        v[id] = 10.0;
        v
    }

    fn plan(json: &str) -> ChatPlan {
        ChatRequest::from_json(json).unwrap().plan(None).unwrap()
    }

    fn run(turn: &mut ChatTurn, ids: &[usize]) -> Vec<Value> {
        let t = tok();
        let mut chunks = Vec::new();
        for &id in ids {
            let (step, c) = turn.step(&one_hot(id, 8), &t, &[]);
            chunks.extend(c.iter().map(|s| serde_json::from_str::<Value>(s).unwrap()));
            if step == Step::Done {
                break;
            }
        }
        chunks
    }

    fn content(chunks: &[Value]) -> String {
        chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect()
    }

    #[test]
    fn streams_content_then_stop_on_end_of_turn() {
        let p = plan(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"temperature":0}"#);
        let mut turn = ChatTurn::new(&p, &[1, 2, 3], 100, "c1".into(), 7).unwrap();
        let chunks = run(&mut turn, &[0, 1, 2, 3, 5]);
        assert_eq!(content(&chunks), "Hello world");
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        let done: Value = serde_json::from_str(&turn.completion_json()).unwrap();
        assert_eq!(done["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(done["usage"]["prompt_tokens"], 3);
        assert_eq!(done["usage"]["completion_tokens"], 4);
        assert_eq!(done["created"], 7);
    }

    #[test]
    fn stop_string_is_held_back_and_cut() {
        let p = plan(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"temperature":0,"stop":["STOP"]}"#);
        let mut turn = ChatTurn::new(&p, &[1], 100, "c".into(), 0).unwrap();
        let chunks = run(&mut turn, &[0, 6, 7, 1]);
        assert_eq!(content(&chunks), "Hel");
        assert_eq!(turn.finish(), Some(Finish::Stop));
        assert_eq!(turn.text(), "Hel");
    }

    #[test]
    fn max_tokens_finishes_with_length() {
        let p = plan(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}"#);
        let mut turn = ChatTurn::new(&p, &[1], 100, "c".into(), 0).unwrap();
        let chunks = run(&mut turn, &[0, 1, 2]);
        assert!(chunks.is_empty(), "stream=false emits no chunks");
        assert_eq!(turn.finish(), Some(Finish::Length));
        assert_eq!(turn.text(), "Hello");
    }

    #[test]
    fn tool_call_is_not_streamed_as_content() {
        let p = plan(
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"temperature":0,
                "tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#,
        );
        assert!(p.tools_on);
        let mut turn = ChatTurn::new(&p, &[1], 100, "c".into(), 0).unwrap();
        let chunks = run(&mut turn, &[4, 5]);
        assert_eq!(content(&chunks), "");
        assert_eq!(turn.finish(), Some(Finish::ToolCalls));
        let tc = chunks
            .iter()
            .find(|c| c["choices"][0]["delta"]["tool_calls"].is_array())
            .unwrap();
        assert_eq!(tc["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "f");
        let done: Value = serde_json::from_str(&turn.completion_json()).unwrap();
        assert_eq!(done["choices"][0]["finish_reason"], "tool_calls");
        assert!(done["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn prompt_that_fills_context_is_context_exceeded() {
        let p = plan(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        let err = ChatTurn::new(&p, &[1, 2, 3, 4], 4, "c".into(), 0).err().unwrap();
        assert!(err.starts_with("context_exceeded:"), "{err}");
    }

    #[test]
    fn plan_maps_sampling_fields_and_validates() {
        let p = plan(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
                "max_tokens":5,"temperature":0.2,"top_p":0.9,"seed":3,"stop":"x",
                "stream":true,"stream_options":{"include_usage":true},
                "response_format":{"type":"json_object"}}"#,
        );
        assert_eq!(p.params.max_tokens, 5);
        assert!((p.params.temperature - 0.2).abs() < 1e-6);
        assert!((p.params.top_p - 0.9).abs() < 1e-6);
        assert_eq!(p.params.seed, 3);
        assert!(p.params.json_object);
        assert_eq!(p.stop, vec!["x".to_string()]);
        assert!(p.stream && p.include_usage);
        assert_eq!(p.prompt, "user: hi\nassistant:\n");
        assert!(ChatRequest::from_json(r#"{"messages":[]}"#).unwrap().plan(None).is_err());
        assert!(ChatRequest::from_json("{").is_err());
    }

    #[test]
    fn plan_uses_chatml_fallback_for_unrenderable_chatml_template() {
        let req = ChatRequest::from_json(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        let p = req.plan(Some("<|im_start|>{{ broken( }}")).unwrap();
        assert_eq!(p.prompt, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn partial_marker_holdback() {
        assert_eq!(partial_marker_suffix("abc<tool", &["<tool_call>"]), 5);
        assert_eq!(partial_marker_suffix("abc", &["<tool_call>"]), 0);
        assert_eq!(partial_marker_suffix("xxST", &["STOP"]), 2);
    }
}
