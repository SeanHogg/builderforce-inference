//! HTTP handlers. Generate paths return 503 when no engine is loaded.

use std::path::{Path, PathBuf};

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bfi_runtime::{
    apply_chat_template, list_models, model_path, ChatMessage, Engine, InstalledModel, SampleParams,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{AppState, Loaded};

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "engine": "bfi" }))
}

pub async fn v1_models() -> Json<Value> {
    let data: Vec<Value> = installed()
        .into_iter()
        .map(|m| {
            json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "bfi"
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

pub async fn v1_embeddings() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({ "error": "embeddings not implemented until M4" })),
    )
        .into_response()
}

pub async fn api_version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

pub async fn api_tags() -> Json<Value> {
    let models: Vec<Value> = installed()
        .into_iter()
        .map(|m| {
            json!({
                "name": m.name,
                "model": m.name,
                "size": m.size,
                "digest": "",
                "details": {}
            })
        })
        .collect();
    Json(json!({ "models": models }))
}

pub async fn api_ps(State(state): State<AppState>) -> Json<Value> {
    let models = match state.inner.engine.lock().as_ref() {
        Some(loaded) => vec![json!({ "name": loaded.name, "model": loaded.name })],
        None => Vec::new(),
    };
    Json(json!({ "models": models }))
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    messages: Vec<IncomingMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
    /// Accepted (OpenAI tools). Forward is stubbed — no model tool loop here.
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct IncomingMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Value,
    /// OpenAI tool-call results / assistant tool_calls.
    #[serde(default)]
    tool_calls: Option<Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

impl IncomingMessage {
    fn text(&self) -> String {
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
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    prompt: Value,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    messages: Vec<IncomingMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    system: Option<Value>,
    #[serde(default)]
    stream: bool,
    /// Anthropic tools; `tool_use` / `tool_result` live in message content blocks.
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    messages: Vec<IncomingMessage>,
    #[serde(default)]
    stream: bool,
    options: Option<OllamaOptions>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default = "default_true")]
    stream: bool,
    options: Option<OllamaOptions>,
}

#[derive(Debug, Deserialize, Default)]
struct OllamaOptions {
    temperature: Option<f32>,
    num_predict: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ModelNameRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    stream: Option<bool>,
}

impl ModelNameRequest {
    fn model_name(&self) -> &str {
        if !self.model.is_empty() {
            &self.model
        } else {
            &self.name
        }
    }
}

fn default_true() -> bool {
    true
}

pub async fn v1_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let model = req.model.clone();
    let params = sample_params(req.max_tokens, req.temperature);
    let messages = to_chat_messages(&req.messages);
    let _ = (&req.tools, &req.tool_choice, &req.response_format);
    match generate_chat(&state, &model, &messages, params).await {
        Ok((name, text)) => {
            if req.stream {
                sse_chat(&name, &text)
            } else {
                Json(openai_chat(&name, &text)).into_response()
            }
        }
        Err(err) => err.into_response(),
    }
}

pub async fn v1_completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let model = req.model.clone();
    let prompt = content_text(&req.prompt);
    let params = sample_params(req.max_tokens, req.temperature);
    match generate_raw(&state, &model, prompt, params).await {
        Ok((name, text)) => {
            if req.stream {
                sse_completion(&name, &text)
            } else {
                Json(json!({
                    "id": "cmpl-bfi",
                    "object": "text_completion",
                    "created": now_secs(),
                    "model": name,
                    "choices": [{ "text": text, "index": 0, "finish_reason": "stop" }],
                    "usage": dummy_usage()
                }))
                .into_response()
            }
        }
        Err(err) => err.into_response(),
    }
}

pub async fn v1_messages(State(state): State<AppState>, Json(req): Json<AnthropicRequest>) -> Response {
    let model = req.model.clone();
    let mut messages = Vec::new();
    if let Some(system) = req.system.as_ref() {
        let text = content_text(system);
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: text,
            });
        }
    }
    messages.extend(to_chat_messages(&req.messages));
    let params = sample_params(req.max_tokens, req.temperature);
    let _ = (&req.tools, &req.tool_choice);
    match generate_chat(&state, &model, &messages, params).await {
        Ok((name, text)) => {
            if req.stream {
                sse_anthropic(&name, &text)
            } else {
                Json(json!({
                    "id": "msg_bfi",
                    "type": "message",
                    "role": "assistant",
                    "model": name,
                    "content": [{ "type": "text", "text": text }],
                    "stop_reason": "end_turn",
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }))
                .into_response()
            }
        }
        Err(err) => err.into_response(),
    }
}

pub async fn api_chat(State(state): State<AppState>, Json(req): Json<OllamaChatRequest>) -> Response {
    let model = req.model.clone();
    let messages = to_chat_messages(&req.messages);
    let (temperature, max_tokens) = match &req.options {
        Some(o) => (o.temperature, o.num_predict),
        None => (None, None),
    };
    let params = sample_params(max_tokens, temperature);
    match generate_chat(&state, &model, &messages, params).await {
        Ok((name, text)) => {
            if req.stream {
                let body = format!(
                    "{}\n{}\n",
                    json!({ "model": name, "message": { "role": "assistant", "content": text }, "done": false }),
                    json!({ "model": name, "message": { "role": "assistant", "content": "" }, "done": true })
                );
                (
                    [
                        (header::CONTENT_TYPE, "application/x-ndjson"),
                        (header::CACHE_CONTROL, "no-cache"),
                    ],
                    body,
                )
                    .into_response()
            } else {
                Json(json!({
                    "model": name,
                    "message": { "role": "assistant", "content": text },
                    "done": true
                }))
                .into_response()
            }
        }
        Err(err) => err.into_response(),
    }
}

pub async fn api_generate(State(state): State<AppState>, Json(req): Json<OllamaGenerateRequest>) -> Response {
    let model = req.model.clone();
    let (temperature, max_tokens) = match &req.options {
        Some(o) => (o.temperature, o.num_predict),
        None => (None, None),
    };
    let params = sample_params(max_tokens, temperature);
    match generate_raw(&state, &model, req.prompt, params).await {
        Ok((name, text)) => {
            if req.stream {
                let body = format!(
                    "{}\n{}\n",
                    json!({ "model": name, "response": text, "done": false }),
                    json!({ "model": name, "response": "", "done": true })
                );
                (
                    [
                        (header::CONTENT_TYPE, "application/x-ndjson"),
                        (header::CACHE_CONTROL, "no-cache"),
                    ],
                    body,
                )
                    .into_response()
            } else {
                Json(json!({
                    "model": name,
                    "response": text,
                    "done": true
                }))
                .into_response()
            }
        }
        Err(err) => err.into_response(),
    }
}

pub async fn api_show(Json(req): Json<ModelNameRequest>) -> Response {
    let name = req.model_name();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "model is required" })),
        )
            .into_response();
    }
    match installed().into_iter().find(|m| m.name == name) {
        Some(m) => Json(json!({
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "bfi",
                "parameter_size": "",
                "quantization_level": ""
            },
            "model_info": {},
            "modified_at": "",
            "name": m.name,
            "size": m.size
        }))
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("model '{name}' not found") })),
        )
            .into_response(),
    }
}

pub async fn api_pull(Json(req): Json<ModelNameRequest>) -> Response {
    let name = req.model_name();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "model is required" })),
        )
            .into_response();
    }
    let stream = req.stream.unwrap_or(true);
    if stream {
        (
            [
                (header::CONTENT_TYPE, "application/x-ndjson"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            pull_ndjson_body(),
        )
            .into_response()
    } else {
        Json(json!({ "status": "success" })).into_response()
    }
}

pub async fn api_delete(Json(req): Json<ModelNameRequest>) -> Response {
    let name = req.model_name();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "model is required" })),
        )
            .into_response();
    }
    // Store mutation is owned by the CLI / later tickets. Route exists.
    StatusCode::OK.into_response()
}

enum ApiError {
    NoModel,
    Generate(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::NoModel => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": "no model loaded" })),
            )
                .into_response(),
            ApiError::Generate(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": msg })),
            )
                .into_response(),
        }
    }
}

async fn generate_chat(
    state: &AppState,
    model: &str,
    messages: &[ChatMessage],
    params: SampleParams,
) -> Result<(String, String), ApiError> {
    let state = state.clone();
    let model = model.to_string();
    let messages = messages.to_vec();
    tokio::task::spawn_blocking(move || {
        let mut slot = state.inner.engine.lock();
        ensure_loaded(&mut slot, state.inner.model_path.as_deref(), &model)?;
        let loaded = slot.as_mut().ok_or(ApiError::NoModel)?;
        let prompt = render_prompt(&loaded.engine, &messages);
        let text = loaded
            .engine
            .generate(&prompt, params)
            .map_err(|e| ApiError::Generate(e.to_string()))?;
        Ok((loaded.name.clone(), text))
    })
    .await
    .map_err(|e| ApiError::Generate(e.to_string()))?
}

async fn generate_raw(
    state: &AppState,
    model: &str,
    prompt: String,
    params: SampleParams,
) -> Result<(String, String), ApiError> {
    let state = state.clone();
    let model = model.to_string();
    tokio::task::spawn_blocking(move || {
        let mut slot = state.inner.engine.lock();
        ensure_loaded(&mut slot, state.inner.model_path.as_deref(), &model)?;
        let loaded = slot.as_mut().ok_or(ApiError::NoModel)?;
        let text = loaded
            .engine
            .generate(&prompt, params)
            .map_err(|e| ApiError::Generate(e.to_string()))?;
        Ok((loaded.name.clone(), text))
    })
    .await
    .map_err(|e| ApiError::Generate(e.to_string()))?
}

fn ensure_loaded(slot: &mut Option<Loaded>, configured: Option<&Path>, model: &str) -> Result<(), ApiError> {
    if let Some(loaded) = slot.as_ref() {
        if configured.is_some() {
            return Ok(());
        }
        if model.is_empty() || loaded.name == model {
            return Ok(());
        }
    }

    let path = match configured {
        Some(p) => p.to_path_buf(),
        None if !model.is_empty() => resolve_model(model),
        None => return Err(ApiError::NoModel),
    };

    match Engine::load(&path) {
        Ok(engine) => {
            let name = if model.is_empty() {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("default")
                    .to_string()
            } else {
                model.to_string()
            };
            *slot = Some(Loaded { name, engine });
            Ok(())
        }
        Err(_) => Err(ApiError::NoModel),
    }
}

fn resolve_model(name: &str) -> PathBuf {
    let as_path = PathBuf::from(name);
    if as_path.exists() {
        as_path
    } else {
        model_path(name)
    }
}

fn render_prompt(engine: &Engine, messages: &[ChatMessage]) -> String {
    match engine.chat_template() {
        Some(t) if !t.is_empty() => {
            apply_chat_template(t, messages, true).unwrap_or_else(|_| naive_prompt(messages))
        }
        _ => naive_prompt(messages),
    }
}

fn naive_prompt(messages: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str(&m.role);
        s.push_str(": ");
        s.push_str(&m.content);
        s.push('\n');
    }
    s.push_str("assistant: ");
    s
}

fn to_chat_messages(raw: &[IncomingMessage]) -> Vec<ChatMessage> {
    raw.iter()
        .map(|m| ChatMessage {
            role: if m.role.is_empty() { "user".into() } else { m.role.clone() },
            content: m.text(),
        })
        .collect()
}

fn content_text(v: &Value) -> String {
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
                    let id = o
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let inner = o.get("content").map(content_text).unwrap_or_default();
                    format!("[tool_result {id} {inner}]")
                }
                _ => o
                    .get("content")
                    .map(content_text)
                    .unwrap_or_default(),
            }
        }
        other => other.to_string(),
    }
}

fn sample_params(max_tokens: Option<u32>, temperature: Option<f32>) -> SampleParams {
    SampleParams {
        max_tokens: max_tokens.unwrap_or(256),
        temperature: temperature.unwrap_or(0.8),
        ..SampleParams::default()
    }
}

fn installed() -> Vec<InstalledModel> {
    list_models().unwrap_or_else(|_| Vec::new())
}

fn dummy_usage() -> Value {
    json!({ "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 })
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn openai_chat(model: &str, text: &str) -> Value {
    json!({
        "id": "chatcmpl-bfi",
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }],
        "usage": dummy_usage()
    })
}

fn sse_chat(model: &str, text: &str) -> Response {
    let chunk = json!({
        "id": "chatcmpl-bfi",
        "object": "chat.completion.chunk",
        "created": now_secs(),
        "model": model,
        "choices": [{ "index": 0, "delta": { "content": text }, "finish_reason": null }]
    });
    let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}

fn sse_completion(model: &str, text: &str) -> Response {
    let chunk = json!({
        "id": "cmpl-bfi",
        "object": "text_completion",
        "model": model,
        "choices": [{ "index": 0, "text": text, "finish_reason": null }]
    });
    let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}

/// Anthropic Messages streaming events (not OpenAI SSE).
fn sse_anthropic(model: &str, text: &str) -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        anthropic_sse_body(model, text),
    )
        .into_response()
}

fn anthropic_sse_body(model: &str, text: &str) -> String {
    let start = json!({
        "type": "message_start",
        "message": {
            "id": "msg_bfi",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": null,
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }
    });
    let block_start = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": { "type": "text", "text": "" }
    });
    let delta = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": { "type": "text_delta", "text": text }
    });
    let block_stop = json!({ "type": "content_block_stop", "index": 0 });
    let msg_delta = json!({
        "type": "message_delta",
        "delta": { "stop_reason": "end_turn" },
        "usage": { "output_tokens": 0 }
    });
    let stop = json!({ "type": "message_stop" });
    format!(
        "event: message_start\ndata: {start}\n\n\
event: content_block_start\ndata: {block_start}\n\n\
event: content_block_delta\ndata: {delta}\n\n\
event: content_block_stop\ndata: {block_stop}\n\n\
event: message_delta\ndata: {msg_delta}\n\n\
event: message_stop\ndata: {stop}\n\n"
    )
}

/// NDJSON progress `{status,total,completed}` — the shape `bf-local` parses today.
fn pull_ndjson_body() -> String {
    format!(
        "{}\n{}\n{}\n{}\n",
        json!({ "status": "pulling manifest" }),
        json!({ "status": "downloading", "total": 1, "completed": 0 }),
        json!({ "status": "downloading", "total": 1, "completed": 1 }),
        json!({ "status": "success" })
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_sse_has_event_types() {
        let body = anthropic_sse_body("m", "hello");
        assert!(body.contains("event: message_start"));
        assert!(body.contains("event: content_block_start"));
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("event: content_block_stop"));
        assert!(body.contains("event: message_delta"));
        assert!(body.contains("event: message_stop"));
        assert!(body.contains("hello"));
        assert!(body.contains("text_delta"));
    }

    #[test]
    fn pull_ndjson_has_status_total_completed() {
        let body = pull_ndjson_body();
        let mut last = None;
        let mut saw_progress = false;
        for line in body.lines().filter(|l| !l.is_empty()) {
            let v: Value = serde_json::from_str(line).unwrap();
            if v.get("total").and_then(Value::as_u64) == Some(1)
                && v.get("completed").and_then(Value::as_u64).is_some()
            {
                saw_progress = true;
            }
            last = v.get("status").and_then(Value::as_str).map(str::to_string);
        }
        assert!(saw_progress);
        assert_eq!(last.as_deref(), Some("success"));
    }

    #[test]
    fn content_text_tool_use_and_result() {
        let use_block = json!([{
            "type": "tool_use",
            "id": "tu_1",
            "name": "lookup",
            "input": { "q": "hi" }
        }]);
        let got = content_text(&use_block);
        assert!(got.contains("tool_use"));
        assert!(got.contains("lookup"));

        let result_block = json!([{
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "ok"
        }]);
        let got = content_text(&result_block);
        assert!(got.contains("tool_result"));
        assert!(got.contains("ok"));
    }

    #[test]
    fn tools_fields_deserialize() {
        let raw = r#"{
            "model": "x",
            "messages": [{"role":"user","content":"hi"}],
            "tools": [{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice": "auto",
            "response_format": {"type":"json_object"}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.tool_choice, Some(Value::String("auto".into())));
        assert!(req.response_format.is_some());
    }
}
