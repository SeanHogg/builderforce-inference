//! Small `js_sys::Reflect` helpers. bfi-web reads browser objects (navigator.gpu,
//! ReadableStream, AbortSignal) dynamically so it needs no `web-sys` features
//! and no `--cfg web_sys_unstable_apis` for the WebGPU types.

use js_sys::{Function, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};

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

pub fn object(pairs: &[(&str, JsValue)]) -> JsValue {
    let o = Object::new();
    for (k, v) in pairs {
        let _ = Reflect::set(&o, &JsValue::from_str(k), v);
    }
    o.into()
}

/// `signal?.aborted === true`.
pub fn aborted(signal: Option<&Object>) -> bool {
    signal.is_some_and(|s| get(s, "aborted").as_bool() == Some(true))
}

/// Seconds since the Unix epoch (`Date.now() / 1000`); wasm32 has no `SystemTime`.
pub fn now_secs() -> u64 {
    (js_sys::Date::now() / 1000.0) as u64
}
