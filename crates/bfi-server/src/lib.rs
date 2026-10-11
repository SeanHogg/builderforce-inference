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
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::Mutex;
use serde_json::json;
use tower_http::cors::CorsLayer;

mod handlers;

use bfi_runtime::Engine;
use handlers::{
    api_chat, api_ps, api_tags, health, v1_chat_completions, v1_completions, v1_messages, v1_models,
};

/// Listen address and optional auth / default model.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bind host. Default `127.0.0.1`. Do not use `0.0.0.0` unless set explicitly.
    pub host: String,
    /// Bind port. Default `11434` (Ollama-compatible).
    pub port: u16,
    /// When `Some`, `/v1/*` requires `Authorization: Bearer <key>` or `x-api-key`.
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

/// Bind and serve. Returns when the server stops.
pub async fn serve(cfg: ServerConfig) -> anyhow::Result<()> {
    let host = if cfg.host.is_empty() {
        "127.0.0.1".to_string()
    } else {
        cfg.host.clone()
    };
    let port = cfg.port;
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("invalid bind address {host}:{port}"))?;
    let app = router(AppState::new(cfg));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    axum::serve(listener, app).await.context("server")?;
    Ok(())
}

/// Build the axum Router so tests can use oneshot without bind.
pub fn router(state: AppState) -> Router {
    let v1 = Router::new()
        .route("/v1/models", get(v1_models))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/messages", post(v1_messages))
        .layer(middleware::from_fn_with_state(state.clone(), require_api_key));

    Router::new()
        .route("/health", get(health))
        .route("/api/tags", get(api_tags))
        .route("/api/ps", get(api_ps))
        .route("/api/chat", post(api_chat))
        .merge(v1)
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn require_api_key(State(state): State<AppState>, req: Request<axum::body::Body>, next: Next) -> Response {
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
    if let Some(raw) = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
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

    #[tokio::test]
    async fn health_ok() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let (status, v) = body_json(res).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["status"], "ok");
        assert_eq!(v["engine"], "bfi");
    }

    #[tokio::test]
    async fn models_list_shape() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let (status, v) = body_json(res).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["object"], "list");
        assert!(v["data"].is_array());
    }

    #[tokio::test]
    async fn chat_without_model_is_503() {
        let app = router(AppState::new(ServerConfig::default()));
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
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

        let health = app
            .clone()
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        let denied = app
            .clone()
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);

        let ok = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
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
        let res = app
            .oneshot(Request::builder().uri("/api/ps").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let (status, v) = body_json(res).await;
        assert_eq!(status, StatusCode::OK);
        assert!(v["models"].as_array().unwrap().is_empty());
    }
}
