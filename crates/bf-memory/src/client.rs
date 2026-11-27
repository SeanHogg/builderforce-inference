//! A synchronous desktop adapter over the official MCP SDK's stdio client.

use crate::spec::LaunchSpec;
use anyhow::{anyhow, bail, Context, Result};
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientConfig, Implementation,
};
use rmcp::service::{RoleClient, RunningService, ServiceError};
use rmcp::transport::TokioChildProcess;
use rmcp::ServiceExt;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::future::Future;
use std::process::Stdio;
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Duration;
use tokio::{process::Command, runtime};

/// First start may download the package through `npx`.
const START_TIMEOUT: Duration = Duration::from_secs(180);
pub const CALL_TIMEOUT: Duration = Duration::from_secs(30);

type Connection = RunningService<RoleClient, ClientConfig>;

pub struct McpClient {
    spec: LaunchSpec,
    label: String,
    env: Mutex<Vec<(String, String)>>,
    conn: Mutex<Option<Connection>>,
}

impl McpClient {
    pub fn new(spec: LaunchSpec, env: Vec<(String, String)>, label: impl Into<String>) -> Self {
        Self {
            spec,
            label: label.into(),
            env: Mutex::new(env),
            conn: Mutex::new(None),
        }
    }

    /// A running child keeps its original environment until it is restarted.
    pub fn set_env(&self, env: Vec<(String, String)>) {
        *self.env.lock().unwrap_or_else(|p| p.into_inner()) = env;
        let mut slot = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        close_connection(&mut slot);
    }

    /// Keep long calls independent of ordinary desktop requests.
    pub fn sibling(&self) -> Self {
        Self::new(
            self.spec.clone(),
            self.env.lock().unwrap_or_else(|p| p.into_inner()).clone(),
            self.label.clone(),
        )
    }

    pub fn call(&self, tool: &str, args: Value) -> Result<Value> {
        self.call_with_timeout(tool, args, CALL_TIMEOUT)
    }

    pub fn call_as<T: DeserializeOwned>(&self, tool: &str, args: Value) -> Result<T> {
        serde_json::from_value(self.call(tool, args)?)
            .with_context(|| format!("{tool}: unexpected result shape"))
    }

    pub fn call_with_timeout(&self, tool: &str, args: Value, timeout: Duration) -> Result<Value> {
        let mut params = CallToolRequestParams::new(tool.to_owned());
        match args {
            Value::Object(arguments) => params = params.with_arguments(arguments),
            Value::Null => {}
            _ => bail!("{tool}: expected an object for tool arguments"),
        }
        let mut slot = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        if slot.as_ref().is_some_and(Connection::is_closed) {
            close_connection(&mut slot);
        }
        if slot.is_none() {
            *slot = Some(self.connect()?);
        }
        let peer = slot.as_ref().expect("connected above").peer().clone();
        match run_io(async move { tokio::time::timeout(timeout, peer.call_tool(params)).await })? {
            Ok(Ok(result)) => tool_result(result),
            Ok(Err(ServiceError::McpError(error))) => bail!("tools/call: {}", error.message),
            Ok(Err(error)) => {
                // A write may have reached the server before a transport error. Reconnect
                // for the next call without replaying the request with an unknown outcome.
                if matches!(
                    error,
                    ServiceError::TransportClosed | ServiceError::TransportSend(_)
                ) {
                    close_connection(&mut slot);
                }
                Err(error).with_context(|| format!("{}: {tool}", self.label))
            }
            Err(_) => {
                close_connection(&mut slot);
                bail!("{} did not answer in time", self.label);
            }
        }
    }

    fn connect(&self) -> Result<Connection> {
        let env = self.env.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let spec = self.spec.clone();
        let label = self.label.clone();
        run_io(async move {
            let mut command = Command::new(&spec.command);
            command.args(&spec.args).envs(env);
            #[cfg(windows)]
            command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

            let (transport, _) = TokioChildProcess::builder(command)
                .stderr(Stdio::null())
                .spawn()
                .with_context(|| format!("could not start {label} ({})", spec.command))?;
            let config = ClientConfig::new(
                ClientCapabilities::default(),
                Implementation::new("synapse", env!("CARGO_PKG_VERSION")),
            );
            tokio::time::timeout(START_TIMEOUT, config.serve(transport))
                .await
                .with_context(|| format!("{label} did not answer in time"))?
                .with_context(|| format!("{label} did not initialise"))
        })?
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        close_connection(self.conn.get_mut().unwrap_or_else(|p| p.into_inner()));
    }
}

fn close_connection(slot: &mut Option<Connection>) {
    if let Some(connection) = slot.take() {
        let _ = run_io(async move { connection.cancel().await });
    }
}

/// Keep SDK I/O alive between calls and allow callers already on a runtime's
/// blocking pool to use the existing synchronous desktop API.
fn run_io<T: Send + 'static>(task: impl Future<Output = T> + Send + 'static) -> Result<T> {
    static IO: OnceLock<std::io::Result<runtime::Runtime>> = OnceLock::new();
    let io = IO
        .get_or_init(|| {
            runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_name("synapse-memory-io")
                .enable_all()
                .build()
        })
        .as_ref()
        .map_err(|error| anyhow!("could not start the memory I/O runtime: {error}"))?;
    let (reply, result) = mpsc::sync_channel(1);
    io.spawn(async move {
        let _ = reply.send(task.await);
    });
    result.recv().context("the memory I/O task stopped")
}

fn tool_result(result: CallToolResult) -> Result<Value> {
    let text = result
        .content
        .iter()
        .filter_map(|part| part.as_text())
        .map(|part| part.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if result.is_error == Some(true) {
        bail!("{text}");
    }
    Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::ContentBlock;
    use serde_json::json;

    #[test]
    fn a_tool_result_is_its_json_text_and_a_tool_error_is_an_err() {
        let ok = CallToolResult::success(vec![ContentBlock::text(r#"{"id":"ep"}"#)]);
        assert_eq!(tool_result(ok).unwrap(), json!({ "id": "ep" }));
        let plain = CallToolResult::success(vec![ContentBlock::text("Remembered \"k\".")]);
        assert_eq!(tool_result(plain).unwrap(), json!("Remembered \"k\"."));
        let parts = CallToolResult::success(vec![
            ContentBlock::text("a"),
            ContentBlock::image("", "image/png"),
            ContentBlock::text("b"),
        ]);
        assert_eq!(tool_result(parts).unwrap(), json!("a\nb"));
        let err = CallToolResult::error(vec![ContentBlock::text("No skill \"x\".")]);
        assert_eq!(tool_result(err).unwrap_err().to_string(), "No skill \"x\".");
    }
}
