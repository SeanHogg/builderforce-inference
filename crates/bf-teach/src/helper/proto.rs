//! The wire framing: one JSON object per line. Requests are `{id, method, params}`;
//! answers are `{id, result}` or `{id, error:{code, message}}`; pushes are `{event, ...}`.

use serde_json::{json, Map, Value};
use std::sync::{Arc, Mutex, MutexGuard};

/// Lock, surviving a poisoned mutex: one panicked thread must not wedge the protocol.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Where every outgoing line goes. The binary writes it to stdout; tests collect it.
pub type Emitter = Arc<dyn Fn(Value) + Send + Sync>;

pub const PROTOCOL: u32 = 1;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: Value,
    pub method: String,
    pub params: Value,
}

/// A failed request: a short machine code plus a sentence a person could read.
#[derive(Debug, Clone, PartialEq)]
pub struct Failure {
    pub code: &'static str,
    pub message: String,
}

impl Failure {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Failure { code, message: message.into() }
    }
}

/// Parse one line. An unreadable line comes back as the error answer to send (id null,
/// or the id when one could be read).
pub fn parse_request(line: &str) -> Result<Request, Value> {
    let v: Value = serde_json::from_str(line)
        .map_err(|e| err(&Value::Null, Failure::new("bad_request", format!("not valid JSON: {e}"))))?;
    let id = v.get("id").cloned().unwrap_or(Value::Null);
    let Some(method) = v.get("method").and_then(Value::as_str) else {
        return Err(err(&id, Failure::new("bad_request", "a request needs a string `method`")));
    };
    let params = v.get("params").cloned().unwrap_or(Value::Null);
    Ok(Request { id, method: method.to_string(), params })
}

pub fn ok(id: &Value, result: Value) -> Value {
    json!({ "id": id, "result": result })
}

pub fn err(id: &Value, f: Failure) -> Value {
    json!({ "id": id, "error": { "code": f.code, "message": f.message } })
}

/// `{event: name, ...fields}`; `fields` must be a JSON object (or null for none).
pub fn event(name: &str, fields: Value) -> Value {
    let mut m = Map::new();
    m.insert("event".into(), Value::String(name.into()));
    if let Value::Object(o) = fields {
        m.extend(o);
    }
    Value::Object(m)
}
