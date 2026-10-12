//! OpenAI / Anthropic / Ollama HTTP API for the bfi engine.
//!
//! Bind with [`serve`] or build a [`router`] for tests (`oneshot` without a socket).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use axum::extract::State;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use parking_lot::Mutex;
use rand::RngCore;
use serde_json::json;
use tower_http::cors::CorsLayer;

mod discovery;
mod handlers;

pub use discovery::{
    discovery_path, publish_discovery, publish_discovery_to, read, withdraw_discovery, Discovery,
    DiscoveryGuard,
};

use bfi_runtime::Engine;
use handlers::{
    api_chat, api_delete, api_generate, api_ps, api_pull, api_show, api_tags, api_version, health,
    v1_chat_completions, v1_completions, v1_embeddings, v1_messages, v1_models,
};

/// Listen address and optional auth / default model.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bind host. Default `127.0.0.1`. Do not use `0.0.0.0` unless set explicitly.
    pub host: String,
    /// Bind port. Default `11434` (Ollama-compatible). If taken, the next free port is used.
    pub port: u16,
    /// When `Some`, every route except `/health` requires `Authorization: Bearer <key>`
    /// (or `x-api-key`). Set by `--auth` in the CLI.
    pub api_key: Option<String>,
    /// Prefers this GGUF on first generate if set.
    pub model_path: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 11434,
            api_key: None,
            model_path: None,
        }
    }
}

impl ServerConfig {
    /// `BFI_HOST` overrides the bind host when set and non-empty.
    pub fn apply_env(&mut self) {
        let env = std::env::var("BFI_HOST").ok();
        self.host = resolve_host(env.as_deref(), &self.host);
    }

    /// Enable `--auth`: mint a token when none is configured.
    pub fn with_auth(mut self) -> Self {
        if self.api_key.is_none() {
            self.api_key = Some(mint_token());
        }
        self
    }
}

/// Host resolution: env wins, then config, then loopback.
pub fn resolve_host(env_host: Option<&str>, cfg_host: &str) -> String {
    if let Some(h) = env_host.map(str::trim).filter(|s| !s.is_empty()) {
        return h.to_string();
    }
    if cfg_host.is_empty() {
        "127.0.0.1".into()
    } else {
        cfg_host.to_string()
    }
}

/// 32-byte hex token, same construction as `bf-context` discovery.
pub fn mint_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) struct Loaded {
    pub name: String,
    pub engine: Engine,
}

pub(crate) struct Inner {
    pub engine: Mutex<Option<Loaded>>,
    pub api_key: Option<String>,
    pub model_path: Option<PathBuf>,
}

/// Shared server state. Clone is cheap (`Arc`).
#[derive(Clone)]
pub struct AppState {
    pub(crate) inner: Arc<Inner>,
}

impl AppState {
    pub fn new(cfg: ServerConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                engine: Mutex::new(None),
                api_key: cfg.api_key,
                model_path: cfg.model_path,
            }),
        }
    }

    pub(crate) fn api_key(&self) -> Option<&str> {
        self.inner.api_key.as_deref()
    }
}

/// Bind `start_port`, then walk upward if it is taken (Ollama on 11434).
pub async fn bind_next_free(
    host: &str,
    start_port: u16,
) -> anyhow::Result<(tokio::net::TcpListener, SocketAddr)> {
    if start_port == 0 {
        let listener = tokio::net::TcpListener::bind(format!("{host}:0"))
            .await
            .with_context(|| format!("bind {host}:0"))?;
        let addr = listener.local_addr()?;
        return Ok((listener, addr));
    }
    let mut port = start_port;
    let mut last_err: Option<anyhow::Error> = None;
    loop {
        match tokio::net::TcpListener::bind(format!("{host}:{port}")).await {
            Ok(listener) => {
                let addr = listener.local_addr()?;
                return Ok((listener, addr));
            }
            Err(e) => {
                last_err = Some(e.into());
                if port == u16::MAX {
                    break;
                }
                port = port.saturating_add(1);
                if port == start_port {
                    break;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no free port after {start_port}"))
        .context(format!("no free port at or above {host}:{start_port}")))
}

/// Bind and serve. Returns when the server stops.
///
/// Writes `~/.builderforce/bfi.json` (`{port, pid, version, token}`) after the
/// chosen address is known. `BFI_HOST` overrides the host.
pub async fn serve(cfg: ServerConfig) -> anyhow::Result<()> {
    let mut cfg = cfg;
    cfg.apply_env();
    let host = resolve_host(None, &cfg.host);
    let (listener, addr) = bind_next_free(&host, cfg.port).await?;
    let token = cfg.api_key.clone().unwrap_or_default();
    let discovery = publish_discovery(addr.port(), &token)?;
    let _guard = DiscoveryGuard::new(discovery);
    let app = router(AppState::new(cfg));
    axum::serve(listener, app).await.context("server")?;
    Ok(())
}

/// Build the axum Router so tests can use oneshot without bind.
pub fn router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/v1/models", get(v1_models))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/messages", post(v1_messages))
        .route("/api/version", get(api_version))
        .route("/api/tags", get(api_tags))
        .route("/api/ps", get(api_ps))
        .route("/api/show", post(api_show))
        .route("/api/pull", post(api_pull))
        .route("/api/delete", delete(api_delete))
        .route("/api/chat", post(api_chat))
        .route("/api/generate", post(api_generate))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ));

    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn require_api_key(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if let Some(expected) = state.api_key() {
        if !presented_key_matches(req.headers(), expected) {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "unauthorized" })),
            )
                .into_response();
        }
    }
    next.run(req).await
}

fn presented_key_matches(headers: &HeaderMap, expected: &str) -> bool {
    if let Some(raw) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        let token = raw
            .strip_prefix("Bearer ")
            .or_else(|| raw.strip_prefix("bearer "))
            .unwrap_or(raw)
            .trim();
        if token == expected {
            return true;
        }
    }
    if let Some(raw) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        if raw.trim() == expected {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn body_json(res: axum::http::Response<Body>) -> (StatusCode, serde_json::Value) {
        let status = res.status();
        let bytes = res.into_body().collect().await.expect("body").to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, v)
    }

    async fn send(
        app: Router,
        method: &str,
        uri: &str,
        body: &str,
        auth: Option<&str>,
    ) -> axum::http::Response<Body> {
        let mut b = Request::builder().method(method).uri(uri);
        if !body.is_empty() {
            b = b.header("content-type", "application/json");
        }
        if let Some(t) = auth {
            b = b.header("authorization", format!("Bearer {t}"));
        }
        app.oneshot(b.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap()
    }

    #[test]
    fn router_builds() {
        let _app = router(AppState::new(ServerConfig::default()));
    }

    #[test]
    fn default_bind_is_loopback() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 11434);
        assert!(cfg.api_key.is_none());
    }

    #[test]
    fn resolve_host_prefers_env_then_cfg_then_loopback() {
        assert_eq!(resolve_host(Some("10.0.0.1"), "127.0.0.1"), "10.0.0.1");
        assert_eq!(resolve_host(Some("  "), "127.0.0.1"), "127.0.0.1");
        assert_eq!(resolve_host(None, ""), "127.0.0.1");
        assert_eq!(resolve_host(None, "192.168.1.2"), "192.168.1.2");
    }

    #[test]
    fn with_auth_mints_token() {
        let cfg = ServerConfig::default().with_auth();
        let t = cfg.api_key.expect("token");
        assert_eq!(t.len(), 64);
        let same = ServerConfig {
            api_key: Some("kept".into()),
            ..ServerConfig::default()
        }
        .with_auth();
        assert_eq!(same.api_key.as_deref(), Some("kept"));
    }

    #[tokio::test]
    async fn health_ok() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = send(app, "GET", "/health", "", None).await;
        let (status, v) = body_json(res).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["status"], "ok");
        assert_eq!(v["engine"], "bfi");
    }

    #[tokio::test]
    async fn models_list_shape() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = send(app, "GET", "/v1/models", "", None).await;
        let (status, v) = body_json(res).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["object"], "list");
        assert!(v["data"].is_array());
    }

    #[tokio::test]
    async fn chat_without_model_is_503() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = send(
            app,
            "POST",
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            None,
        )
        .await;
        let (status, v) = body_json(res).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["error"], "no model loaded");
    }

    #[tokio::test]
    async fn api_key_guards_v1_not_health() {
        let app = router(AppState::new(ServerConfig {
            api_key: Some("secret".into()),
            ..ServerConfig::default()
        }));

        let health = send(app.clone(), "GET", "/health", "", None).await;
        assert_eq!(health.status(), StatusCode::OK);

        let denied = send(app.clone(), "GET", "/v1/models", "", None).await;
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);

        let ok = send(app, "GET", "/v1/models", "", Some("secret")).await;
        assert_eq!(ok.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_requires_bearer_when_auth() {
        let app = router(AppState::new(ServerConfig {
            api_key: Some("secret".into()),
            ..ServerConfig::default()
        }));
        let denied = send(app.clone(), "GET", "/api/tags", "", None).await;
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        let ok = send(app, "GET", "/api/tags", "", Some("secret")).await;
        assert_eq!(ok.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn x_api_key_accepted() {
        let app = router(AppState::new(ServerConfig {
            api_key: Some("k".into()),
            ..ServerConfig::default()
        }));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header("x-api-key", "k")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_ps_empty_when_unloaded() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = send(app, "GET", "/api/ps", "", None).await;
        let (status, v) = body_json(res).await;
        assert_eq!(status, StatusCode::OK);
        assert!(v["models"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn embeddings_without_model_is_503() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = send(app, "POST", "/v1/embeddings", r#"{"input":"hi"}"#, None).await;
        let (status, v) = body_json(res).await;
        assert_ne!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["error"], "no model loaded");
    }

    #[test]
    fn embeddings_response_shape_has_data_embedding() {
        let v = serde_json::json!({
            "object": "list",
            "data": [{ "object": "embedding", "embedding": [0.1, 0.2], "index": 0 }],
            "model": "m",
        });
        assert!(v["data"][0]["embedding"].is_array());
        assert_eq!(v["data"][0]["object"], "embedding");
    }

    #[tokio::test]
    async fn api_version_shape() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = send(app, "GET", "/api/version", "", None).await;
        let (status, v) = body_json(res).await;
        assert_eq!(status, StatusCode::OK);
        assert!(v["version"].as_str().unwrap().len() > 0);
    }

    #[tokio::test]
    async fn pull_ndjson_shape() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = send(app, "POST", "/api/pull", r#"{"model":"x"}"#, None).await;
        assert_eq!(res.status(), StatusCode::OK);
        let ctype = res
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ctype.contains("ndjson"), "{ctype}");
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        let mut last = None;
        let mut saw = false;
        for line in body.lines().filter(|l| !l.is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            if v.get("total").is_some() && v.get("completed").is_some() {
                saw = true;
            }
            last = v.get("status").and_then(|s| s.as_str()).map(str::to_string);
        }
        assert!(saw);
        assert_eq!(last.as_deref(), Some("success"));
    }

    #[tokio::test]
    async fn delete_and_show_exist() {
        let app = router(AppState::new(ServerConfig::default()));
        let del = send(app.clone(), "DELETE", "/api/delete", r#"{"model":"x"}"#, None).await;
        assert_eq!(del.status(), StatusCode::OK);
        let show = send(app, "POST", "/api/show", r#"{"model":"missing"}"#, None).await;
        assert_eq!(show.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn chat_tools_accepted_without_model() {
        let app = router(AppState::new(ServerConfig::default()));
        let body = r#"{
            "model":"x",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice":"auto",
            "response_format":{"type":"json_object"}
        }"#;
        let res = send(app, "POST", "/v1/chat/completions", body, None).await;
        assert_ne!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_ne!(res.status(), StatusCode::BAD_REQUEST);
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn chat_required_tool_choice_without_tools_is_400() {
        let app = router(AppState::new(ServerConfig::default()));
        let body = r#"{
            "model":"x",
            "messages":[{"role":"user","content":"hi"}],
            "tool_choice":"required"
        }"#;
        let res = send(app, "POST", "/v1/chat/completions", body, None).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn chat_unknown_tool_choice_is_400() {
        let app = router(AppState::new(ServerConfig::default()));
        let body = r#"{
            "model":"x",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice":"banana"
        }"#;
        let res = send(app, "POST", "/v1/chat/completions", body, None).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn chat_tool_choice_object_accepted_without_model() {
        let app = router(AppState::new(ServerConfig::default()));
        let body = r#"{
            "model":"x",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice":{"type":"function","function":{"name":"f"}}
        }"#;
        let res = send(app, "POST", "/v1/chat/completions", body, None).await;
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn chat_json_schema_honored_without_model() {
        let app = router(AppState::new(ServerConfig::default()));
        let body = r#"{
            "model":"x",
            "messages":[{"role":"user","content":"hi"}],
            "response_format":{"type":"json_schema","json_schema":{"name":"out","schema":{"type":"object"}}}
        }"#;
        let res = send(app, "POST", "/v1/chat/completions", body, None).await;
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn chat_grammar_honored_without_model() {
        let app = router(AppState::new(ServerConfig::default()));
        let body = r#"{
            "model":"x",
            "messages":[{"role":"user","content":"hi"}],
            "grammar":"root ::= object"
        }"#;
        let res = send(app, "POST", "/v1/chat/completions", body, None).await;
        assert_ne!(res.status(), StatusCode::BAD_REQUEST);
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn chat_unknown_response_format_is_400() {
        let app = router(AppState::new(ServerConfig::default()));
        let body = r#"{
            "model":"x",
            "messages":[{"role":"user","content":"hi"}],
            "response_format":{"type":"xml"}
        }"#;
        let res = send(app, "POST", "/v1/chat/completions", body, None).await;
        let (status, v) = body_json(res).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(v["error"].as_str().unwrap().contains("xml"));
    }

    #[tokio::test]
    async fn chat_json_schema_missing_schema_is_400() {
        let app = router(AppState::new(ServerConfig::default()));
        let body = r#"{
            "model":"x",
            "messages":[{"role":"user","content":"hi"}],
            "response_format":{"type":"json_schema"}
        }"#;
        let res = send(app, "POST", "/v1/chat/completions", body, None).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn generate_without_model_is_503() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = send(
            app,
            "POST",
            "/api/generate",
            r#"{"model":"x","prompt":"hi","stream":false}"#,
            None,
        )
        .await;
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn route_table_is_not_404() {
        let app = router(AppState::new(ServerConfig::default()));
        let routes = [
            ("GET", "/health", ""),
            ("GET", "/v1/models", ""),
            (
                "POST",
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            ),
            ("POST", "/v1/completions", r#"{"prompt":"hi"}"#),
            ("POST", "/v1/embeddings", r#"{"input":"hi"}"#),
            (
                "POST",
                "/v1/messages",
                r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            ),
            ("GET", "/api/version", ""),
            ("GET", "/api/tags", ""),
            ("GET", "/api/ps", ""),
            ("POST", "/api/show", r#"{"model":"x"}"#),
            ("POST", "/api/pull", r#"{"model":"x"}"#),
            ("DELETE", "/api/delete", r#"{"model":"x"}"#),
            ("POST", "/api/chat", r#"{"messages":[]}"#),
            ("POST", "/api/generate", r#"{"prompt":"hi","stream":false}"#),
        ];
        for (method, uri, body) in routes {
            let res = send(app.clone(), method, uri, body, None).await;
            // POST /api/show is a real route; Ollama returns 404 when the
            // named model is not installed (see `delete_and_show_exist`).
            if method == "POST" && uri == "/api/show" {
                assert_ne!(
                    res.status(),
                    StatusCode::METHOD_NOT_ALLOWED,
                    "{method} {uri}"
                );
                continue;
            }
            assert_ne!(res.status(), StatusCode::NOT_FOUND, "{method} {uri}");
        }
    }

    #[tokio::test]
    async fn bind_next_free_loopback() {
        let (listener, addr) = bind_next_free("127.0.0.1", 0).await.unwrap();
        assert!(addr.ip().is_loopback());
        assert_ne!(addr.port(), 0);
        drop(listener);
    }

    #[tokio::test]
    async fn bind_skips_taken_port() {
        let occupied = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let taken = occupied.local_addr().unwrap().port();
        let (listener, addr) = bind_next_free("127.0.0.1", taken).await.unwrap();
        assert!(addr.port() >= taken);
        // If SO_REUSEADDR lets us share, still a valid loopback bind.
        assert!(addr.ip().is_loopback());
        drop(listener);
        drop(occupied);
    }
}
