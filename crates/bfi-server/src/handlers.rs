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
}

#[derive(Debug, Deserialize)]
pub struct IncomingMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Value,
}

impl IncomingMessage {
    fn text(&self) -> String {
        content_text(&self.content)
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

#[derive(Debug, Deserialize, Default)]
struct OllamaOptions {
    temperature: Option<f32>,
    num_predict: Option<u32>,
}

pub async fn v1_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let model = req.model.clone();
    let params = sample_params(req.max_tokens, req.temperature);
    let messages = to_chat_messages(&req.messages);
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
    match generate_chat(&state, &model, &messages, params).await {
        Ok((name, text)) => {
            if req.stream {
                sse_chat(&name, &text)
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
        Some(t) if !t.is_empty() => apply_chat_template(t, messages).unwrap_or_else(|_| naive_prompt(messages)),
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
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::String(s) => s.clone(),
                Value::Object(o) => o.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join(""),
        Value::Object(o) => o.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
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
    list_models()
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
