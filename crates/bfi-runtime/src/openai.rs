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
use crate::sample::SampleParams;
use crate::tools::{apply_tools, prepend_tools_block, ParsedToolCall, ToolSpec};

mod stream_text;
mod turn;

pub use turn::{ChatTurn, Step};

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

#[cfg(test)]
mod tests;
