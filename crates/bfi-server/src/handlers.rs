//! HTTP handlers. Generate paths return 503 when no engine is loaded.

use std::path::{Path, PathBuf};

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bfi_runtime::openai::{chunk_value, completion_value, tool_calls_value, usage_value};
use bfi_runtime::{
    apply_tools, content_text, list_models, model_path, parse_tool_calls, prepend_tools_block,
    render_prompt, to_chat_messages, ChatMessage, Engine, InstalledModel, SampleParams,
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

pub async fn v1_embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Response {
    let model = req.model.clone();
    let texts = req.input.texts();
    match embed_texts(&state, model, texts).await {
        Ok((name, vectors)) => Json(openai_embeddings(&name, &vectors)).into_response(),
        Err(err) => err.into_response(),
    }
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
pub struct EmbeddingsRequest {
    #[serde(default)]
    model: String,
    input: EmbeddingsInput,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

impl EmbeddingsInput {
    fn texts(&self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s.clone()],
            Self::Many(v) => v.clone(),
        }
    }
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
    /// OpenAI tools. Parsed and applied as decode constraints / tool_calls.
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
    /// llama.cpp / Ollama GBNF. Parsed here; sampler constraint is M4 runtime.
    #[serde(default)]
    grammar: Option<String>,
}

pub use bfi_runtime::IncomingMessage;

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

/// OpenAI `response_format` / llama.cpp `grammar` after parse.
/// Honored as a request constraint; the sampler does not yet constrain tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputConstraint {
    Free,
    JsonObject,
    JsonSchema,
    Grammar,
}

fn parse_output_constraint(
    response_format: Option<&Value>,
    grammar: Option<&str>,
) -> Result<OutputConstraint, String> {
    let grammar = grammar.map(str::trim).filter(|s| !s.is_empty());
    let format = match response_format {
        None | Some(Value::Null) => None,
        Some(v) => Some(parse_response_format(v)?),
    };
    if grammar.is_some() {
        return Ok(OutputConstraint::Grammar);
    }
    Ok(format.unwrap_or(OutputConstraint::Free))
}

fn parse_response_format(v: &Value) -> Result<OutputConstraint, String> {
    match v {
        Value::String(s) => match s.as_str() {
            "text" => Ok(OutputConstraint::Free),
            "json" | "json_object" => Ok(OutputConstraint::JsonObject),
            other => Err(format!(
                "unsupported response_format '{other}'; expected text, json_object, or json_schema"
            )),
        },
        Value::Object(obj) => {
            let ty = obj
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| "response_format.type is required".to_string())?;
            match ty {
                "text" => Ok(OutputConstraint::Free),
                "json" | "json_object" => Ok(OutputConstraint::JsonObject),
                "json_schema" => {
                    let schema = obj.get("json_schema").ok_or_else(|| {
                        "response_format.json_schema is required when type is json_schema"
                            .to_string()
                    })?;
                    if schema.is_null() {
                        return Err(
                            "response_format.json_schema is required when type is json_schema"
                                .into(),
                        );
                    }
                    Ok(OutputConstraint::JsonSchema)
                }
                other => Err(format!(
                    "unsupported response_format.type '{other}'; expected text, json_object, or json_schema"
                )),
            }
        }
        _ => Err("response_format must be an object".to_string()),
    }
}

pub async fn v1_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let constraint = match parse_output_constraint(
        req.response_format.as_ref(),
        req.grammar.as_deref(),
    ) {
        Ok(c) => c,
        Err(msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": msg })),
            )
                .into_response();
        }
    };
    let model = req.model.clone();
    let mut params = sample_params(req.max_tokens, req.temperature);
    apply_output_constraint(&mut params, constraint, req.grammar.as_deref());
    let (tools, tools_on) = match apply_tools(&mut params, req.tools.as_ref(), req.tool_choice.as_ref()) {
        Ok(v) => v,
        Err(msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": msg })),
            )
                .into_response();
        }
    };
    let mut messages = to_chat_messages(&req.messages);
    prepend_tools_block(&mut messages, &tools, &params.tool_choice);
    match generate_chat(&state, &model, &messages, params).await {
        Ok((name, text)) => {
            if req.stream {
                sse_chat(&name, &text, tools_on)
            } else {
                Json(openai_chat(&name, &text, tools_on)).into_response()
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
    let mut params = sample_params(req.max_tokens, req.temperature);
    let (tools, tools_on) = match apply_tools(&mut params, req.tools.as_ref(), req.tool_choice.as_ref()) {
        Ok(v) => v,
        Err(msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": msg })),
            )
                .into_response();
        }
    };
    prepend_tools_block(&mut messages, &tools, &params.tool_choice);
    match generate_chat(&state, &model, &messages, params).await {
        Ok((name, text)) => {
            if req.stream {
                sse_anthropic(&name, &text, tools_on)
            } else {
                Json(anthropic_message(&name, &text, tools_on)).into_response()
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
        let prompt = render_prompt(loaded.engine.chat_template(), &messages);
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

async fn embed_texts(
    state: &AppState,
    model: String,
    texts: Vec<String>,
) -> Result<(String, Vec<Vec<f32>>), ApiError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let mut slot = state.inner.engine.lock();
        ensure_loaded(&mut slot, state.inner.model_path.as_deref(), &model)?;
        let loaded = slot.as_mut().ok_or(ApiError::NoModel)?;
        let mut vectors = Vec::with_capacity(texts.len());
        for t in &texts {
            let v = loaded
                .engine
                .embed(t)
                .map_err(|e| ApiError::Generate(e.to_string()))?;
            vectors.push(v);
        }
        Ok((loaded.name.clone(), vectors))
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

fn apply_output_constraint(
    params: &mut SampleParams,
    c: OutputConstraint,
    grammar: Option<&str>,
) {
    match c {
        OutputConstraint::Free => {}
        OutputConstraint::JsonObject | OutputConstraint::JsonSchema => {
            params.json_object = true;
        }
        OutputConstraint::Grammar => {
            params.grammar = grammar
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
        }
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
    usage_value(0, 0)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn openai_embeddings(model: &str, vectors: &[Vec<f32>]) -> Value {
    json!({
        "object": "list",
        "data": vectors
            .iter()
            .enumerate()
            .map(|(i, embedding)| {
                json!({
                    "object": "embedding",
                    "embedding": embedding,
                    "index": i,
                })
            })
            .collect::<Vec<_>>(),
        "model": model,
        "usage": { "prompt_tokens": 0, "total_tokens": 0 },
    })
}

fn openai_chat(model: &str, text: &str, tools_on: bool) -> Value {
    let calls = if tools_on { parse_tool_calls(text) } else { None };
    match calls {
        Some(calls) => completion_value(
            "chatcmpl-bfi",
            now_secs(),
            model,
            None,
            Some(&calls),
            "tool_calls",
            dummy_usage(),
        ),
        None => completion_value("chatcmpl-bfi", now_secs(), model, Some(text), None, "stop", dummy_usage()),
    }
}

fn anthropic_message(model: &str, text: &str, tools_on: bool) -> Value {
    if tools_on {
        if let Some(calls) = parse_tool_calls(text) {
            let content: Vec<Value> = calls.iter().map(|c| c.anthropic_json()).collect();
            return json!({
                "id": "msg_bfi",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": "tool_use",
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            });
        }
    }
    json!({
        "id": "msg_bfi",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{ "type": "text", "text": text }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 0, "output_tokens": 0 }
    })
}

fn sse_chat(model: &str, text: &str, tools_on: bool) -> Response {
    let calls = if tools_on { parse_tool_calls(text) } else { None };
    let chunk = match calls {
        Some(calls) => chunk_value(
            "chatcmpl-bfi",
            now_secs(),
            model,
            json!({ "role": "assistant", "content": null, "tool_calls": tool_calls_value(&calls) }),
            Some("tool_calls"),
        ),
        None => chunk_value("chatcmpl-bfi", now_secs(), model, json!({ "content": text }), None),
    };
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
fn sse_anthropic(model: &str, text: &str, tools_on: bool) -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        anthropic_sse_body(model, text, tools_on),
    )
        .into_response()
}

fn anthropic_sse_body(model: &str, text: &str, tools_on: bool) -> String {
    if tools_on {
        if let Some(calls) = parse_tool_calls(text) {
            return anthropic_sse_tool_use(model, &calls);
        }
    }
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

fn anthropic_sse_tool_use(model: &str, calls: &[bfi_runtime::ParsedToolCall]) -> String {
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
    let mut body = format!("event: message_start\ndata: {start}\n\n");
    for (i, c) in calls.iter().enumerate() {
        let block_start = json!({
            "type": "content_block_start",
            "index": i,
            "content_block": c.anthropic_json()
        });
        let block_stop = json!({ "type": "content_block_stop", "index": i });
        body.push_str(&format!(
            "event: content_block_start\ndata: {block_start}\n\nevent: content_block_stop\ndata: {block_stop}\n\n"
        ));
    }
    let msg_delta = json!({
        "type": "message_delta",
        "delta": { "stop_reason": "tool_use" },
        "usage": { "output_tokens": 0 }
    });
    let stop = json!({ "type": "message_stop" });
    body.push_str(&format!(
        "event: message_delta\ndata: {msg_delta}\n\nevent: message_stop\ndata: {stop}\n\n"
    ));
    body
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
    use bfi_runtime::ToolChoice;

    #[test]
    fn apply_constraint_json_object_sets_flag() {
        let mut p = SampleParams::default();
        apply_output_constraint(&mut p, OutputConstraint::JsonObject, None);
        assert!(p.json_object);
        assert!(p.grammar.is_none());
    }

    #[test]
    fn apply_constraint_grammar_copies_string() {
        let mut p = SampleParams::default();
        apply_output_constraint(&mut p, OutputConstraint::Grammar, Some("root ::= \"a\""));
        assert_eq!(p.grammar.as_deref(), Some("root ::= \"a\""));
        assert!(!p.json_object);
    }

    #[test]
    fn openai_embeddings_shape() {
        let v = openai_embeddings("m", &[vec![0.5_f32, 1.0]]);
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["object"], "embedding");
        assert_eq!(v["data"][0]["index"], 0);
        assert!(v["data"][0]["embedding"].is_array());
        assert_eq!(v["data"][0]["embedding"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn anthropic_sse_has_event_types() {
        let body = anthropic_sse_body("m", "hello", false);
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
            "response_format": {"type":"json_object"},
            "grammar": "root ::= object"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.tool_choice, Some(Value::String("auto".into())));
        assert!(req.response_format.is_some());
        assert_eq!(req.grammar.as_deref(), Some("root ::= object"));
    }

    #[test]
    fn tool_choice_object_deserializes() {
        let raw = r#"{
            "model": "x",
            "messages": [],
            "tool_choice": {"type":"function","function":{"name":"f"}}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(
            req.tool_choice.as_ref().unwrap()["function"]["name"],
            "f"
        );
    }

    #[test]
    fn apply_tools_required_sets_json_object() {
        let mut p = SampleParams::default();
        let tools = json!([{"type":"function","function":{"name":"lookup","parameters":{}}}]);
        let (parsed, on) = apply_tools(&mut p, Some(&tools), Some(&json!("required"))).unwrap();
        assert!(on);
        assert_eq!(parsed[0].name, "lookup");
        assert!(p.json_object);
        assert_eq!(p.tool_choice, ToolChoice::Required);
        assert_eq!(p.tool_names, vec!["lookup".to_string()]);
    }

    #[test]
    fn apply_tools_required_without_tools_errors() {
        let mut p = SampleParams::default();
        let err = apply_tools(&mut p, None, Some(&json!("required"))).unwrap_err();
        assert!(err.contains("required"), "{err}");
    }

    #[test]
    fn openai_chat_emits_tool_calls() {
        let v = openai_chat("m", r#"{"name":"lookup","arguments":{"q":1}}"#, true);
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert!(v["choices"][0]["message"]["content"].is_null());
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "lookup");
    }

    #[test]
    fn anthropic_message_emits_tool_use() {
        let v = anthropic_message("m", r#"{"name":"lookup","arguments":{}}"#, true);
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["content"][0]["type"], "tool_use");
        assert_eq!(v["content"][0]["name"], "lookup");
    }

    #[test]
    fn output_constraint_honors_json_modes_and_grammar() {
        assert_eq!(
            parse_output_constraint(None, None).unwrap(),
            OutputConstraint::Free
        );
        assert_eq!(
            parse_output_constraint(Some(&json!({"type": "text"})), None).unwrap(),
            OutputConstraint::Free
        );
        assert_eq!(
            parse_output_constraint(Some(&json!({"type": "json_object"})), None).unwrap(),
            OutputConstraint::JsonObject
        );
        assert_eq!(
            parse_output_constraint(Some(&json!("json")), None).unwrap(),
            OutputConstraint::JsonObject
        );
        assert_eq!(
            parse_output_constraint(
                Some(&json!({
                    "type": "json_schema",
                    "json_schema": {"name": "out", "schema": {"type": "object"}}
                })),
                None
            )
            .unwrap(),
            OutputConstraint::JsonSchema
        );
        assert_eq!(
            parse_output_constraint(None, Some("root ::= object")).unwrap(),
            OutputConstraint::Grammar
        );
        assert_eq!(
            parse_output_constraint(Some(&json!({"type": "json_object"})), Some("  ")).unwrap(),
            OutputConstraint::JsonObject
        );
    }

    #[test]
    fn output_constraint_rejects_unknown_and_incomplete() {
        let err = parse_output_constraint(Some(&json!({"type": "xml"})), None).unwrap_err();
        assert!(err.contains("xml"), "{err}");
        let err = parse_output_constraint(Some(&json!({"type": "json_schema"})), None).unwrap_err();
        assert!(err.contains("json_schema"), "{err}");
        let err = parse_output_constraint(Some(&json!(1)), None).unwrap_err();
        assert!(err.contains("object"), "{err}");
    }
}
