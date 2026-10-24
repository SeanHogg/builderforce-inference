//! Small `js_sys::Reflect` helpers, plus what `bfi-engine` needs from a browser realm.
//! bfi-web reads browser objects (navigator.gpu, ReadableStream, AbortSignal)
//! dynamically so it needs no `web-sys` features and no `--cfg web_sys_unstable_apis`
//! for the WebGPU types.

use bfi_engine::{error, ByteSource, EngineError, Host, Kind};
use js_sys::{Function, Object, Reflect, Uint8Array};
use serde_json::Value;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

pub fn get(target: &JsValue, key: &str) -> JsValue {
    Reflect::get(target, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

pub fn present(v: &JsValue) -> bool {
    !v.is_undefined() && !v.is_null()
}

/// `target[method](...args)`.
pub fn call(target: &JsValue, method: &str, args: &[JsValue]) -> Result<JsValue, JsValue> {
    let f: Function = get(target, method)
        .dyn_into()
        .map_err(|_| JsValue::from_str(&format!("{method} is not a function")))?;
    let arr = js_sys::Array::new();
    for a in args {
        arr.push(a);
    }
    f.apply(target, &arr)
}

/// `signal?.aborted === true`.
pub fn aborted(signal: Option<&Object>) -> bool {
    signal.is_some_and(|s| get(s, "aborted").as_bool() == Some(true))
}

/// An engine error as the JS `Error` hosts branch on (its message is the contract).
pub fn js_error(e: EngineError) -> JsValue {
    js_sys::Error::new(&e.to_string()).into()
}

/// A JSON value as a plain JS object (records are small; `JSON.parse` is exact).
pub fn to_js(v: &Value) -> JsValue {
    js_sys::JSON::parse(&v.to_string()).unwrap_or(JsValue::UNDEFINED)
}

/// A plain JS object as JSON; `None` for anything `JSON.stringify` cannot represent.
pub fn to_json(v: &JsValue) -> Option<Value> {
    if !present(v) {
        return None;
    }
    let text = js_sys::JSON::stringify(v).ok()?.as_string()?;
    serde_json::from_str(&text).ok()
}

/// The browser realm: `Date.now()` and a macrotask yield.
pub struct WebHost;

impl Host for WebHost {
    /// wasm32 has no `SystemTime`.
    fn now_secs(&self) -> u64 {
        (js_sys::Date::now() / 1000.0) as u64
    }

    /// Resolve on the next macrotask (`setTimeout(…, 0)`), so the host's event loop runs
    /// — an `AbortSignal` fired by a click is seen, the page paints. Resolves at once
    /// where there is no `setTimeout`.
    async fn yield_now(&self) {
        let global = js_sys::global();
        let set_timeout = get(&global, "setTimeout");
        if !set_timeout.is_function() {
            return;
        }
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let _ = call(&global, "setTimeout", &[resolve.into(), JsValue::from_f64(0.0)]);
        });
        let _ = JsFuture::from(promise).await;
    }
}

/// The host's `ReadableStream<Uint8Array>`, read chunk by chunk (each copied into wasm
/// memory, then dropped). [`StreamSource::release`] gives the stream back.
pub struct StreamSource {
    reader: JsValue,
}

impl StreamSource {
    pub fn new(chunks: &Object) -> Result<Self, EngineError> {
        let reader = call(chunks, "getReader", &[])
            .map_err(|e| error(Kind::LoadFailed, format!("chunks is not a ReadableStream: {e:?}")))?;
        Ok(Self { reader })
    }

    pub fn release(&self) {
        let _ = call(&self.reader, "releaseLock", &[]);
    }
}

impl ByteSource for StreamSource {
    async fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, EngineError> {
        let read = call(&self.reader, "read", &[])
            .map_err(|e| error(Kind::LoadFailed, format!("reader.read threw: {e:?}")))?;
        let res = JsFuture::from(js_sys::Promise::from(read))
            .await
            .map_err(|e| error(Kind::LoadFailed, format!("stream errored: {e:?}")))?;
        if get(&res, "done").as_bool() == Some(true) {
            return Ok(None);
        }
        Ok(Some(Uint8Array::new(&get(&res, "value")).to_vec()))
    }
}
