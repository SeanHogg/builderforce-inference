//! The exported `BfiEngine` class. Owns the WebGPU device and at most one
//! loaded model; `load` / `chat` are exclusive (a second call while one runs
//! rejects with `busy:`), `unload` during a chat aborts it. Once the device is
//! lost (`device.lost`), the model is dropped and every later `load` / `chat`
//! rejects with `webgpu_unavailable:` — create a new engine.

use std::cell::RefCell;
use std::rc::Rc;

use bfi_core::resident::{open_webgpu, Gpu};
use js_sys::{Function, Object, Promise};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use crate::errors::{error, Kind};
use crate::loader::{self, LoadOptions, Loaded};
use crate::support::{self, AdapterInfo};

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
export interface BfiModelInfo { arch: string; contextTokens: number; vocab: number; params: number; kvF16: boolean }
export interface BfiLoadOptions { kvBudgetBytes?: number }
export interface BfiContextShape {
  layers: number; kvHeads: number; headDim: number; trainedContext: number;
  kvBudgetBytes?: number; f16?: boolean;
  maxStorageBufferBindingSizeBytes?: number;
  maxBufferSizeBytes?: number;
  /** @deprecated Alias of maxBufferSizeBytes (as in BfiSupport); min-ed with the binding limit. */
  maxBufferBytes?: number;
}
export interface BfiSupport {
  contractVersion: 2; supported: boolean; reason?: string;
  maxBufferSizeBytes: number; maxStorageBufferBindingSizeBytes: number;
  /** @deprecated Use maxBufferSizeBytes. */
  maxBufferBytes: number; requestedPowerPreference: string; f16: boolean
}
"#;

struct State {
    gpu: Gpu,
    adapter: AdapterInfo,
    model: Option<Loaded>,
    busy: bool,
    unload_requested: bool,
}

impl State {
    /// `Err(webgpu_unavailable)` once the device is lost; drops the model (its
    /// buffers belong to the dead device, so there is nothing to release).
    fn alive(&mut self) -> Result<(), JsValue> {
        let Some(why) = self.gpu.lost() else {
            return Ok(());
        };
        self.model = None;
        Err(error(
            Kind::WebgpuUnavailable,
            format!("the GPU device was lost ({why}); create a new BfiEngine"),
        ))
    }
}

/// bfi on WebGPU. See the crate docs and `README.md` for the JS contract.
#[wasm_bindgen]
pub struct BfiEngine {
    state: Rc<RefCell<State>>,
}

#[wasm_bindgen]
impl BfiEngine {
    /// WebGPU adapter + device (`shader-f16` is requested when offered).
    /// Rejects with `webgpu_unavailable: <reason>` when there is no adapter.
    pub async fn create() -> Result<BfiEngine, JsValue> {
        console_error_panic_hook::set_once();
        let adapter = support::probe().await;
        if !adapter.supported {
            return Err(error(
                Kind::WebgpuUnavailable,
                adapter.reason.clone().unwrap_or_else(|| "no adapter".into()),
            ));
        }
        // CubeCL awaits requestAdapter/requestDevice itself (wasm cannot block).
        let gpu = open_webgpu().await;
        Ok(BfiEngine {
            state: Rc::new(RefCell::new(State {
                gpu,
                adapter,
                model: None,
                busy: false,
                unload_requested: false,
            })),
        })
    }

    /// Adapter probe only — no device — safe to call on every capability check.
    #[wasm_bindgen(unchecked_return_type = "Promise<BfiSupport>")]
    pub fn support() -> Promise {
        future_to_promise(async { Ok(support::probe().await.to_js()) })
    }

    /// Predict the context `load` will serve for a model shape, before
    /// downloading it (pure; same function the engine uses). Defaults:
    /// `kvBudgetBytes` 2 GiB, `f16` false, buffer cap 128 MiB (WebGPU's
    /// guaranteed binding). The cap is `min(maxStorageBufferBindingSizeBytes,
    /// maxBufferSizeBytes ?? maxBufferBytes)` of whichever are given, exactly as
    /// `load` caps the adapter. The engine may serve less if allocation fails.
    #[wasm_bindgen(js_name = contextFor)]
    pub fn context_for(
        #[wasm_bindgen(unchecked_param_type = "BfiContextShape")] shape: Object,
    ) -> f64 {
        crate::plan::context_for_js(&shape) as f64
    }

    /// Load a GGUF streamed by the host (the host owns download + cache).
    /// Weights stay packed on the GPU, one buffer per tensor.
    #[wasm_bindgen(unchecked_return_type = "Promise<BfiModelInfo>")]
    pub fn load(
        &self,
        #[wasm_bindgen(unchecked_param_type = "ReadableStream<Uint8Array>")] chunks: Object,
        #[wasm_bindgen(js_name = totalBytes)] total_bytes: f64,
        #[wasm_bindgen(js_name = onProgress, unchecked_param_type = "(loaded: number, total: number) => void")]
        on_progress: Option<Function>,
        #[wasm_bindgen(unchecked_param_type = "BfiLoadOptions")] options: Option<Object>,
    ) -> Promise {
        let opts = LoadOptions::from_js(options.as_ref());
        let state = self.state.clone();
        future_to_promise(async move {
            let (gpu, adapter) = {
                let mut s = state.borrow_mut();
                if s.busy {
                    return Err(error(Kind::Busy, "another load or chat is running"));
                }
                s.alive()?;
                s.busy = true;
                s.unload_requested = false;
                if s.model.take().is_some() {
                    s.gpu.release_unused();
                }
                (s.gpu.clone(), s.adapter.clone())
            };
            let result =
                loader::load(&gpu, &adapter, &chunks, total_bytes, on_progress.as_ref(), opts).await;
            let mut s = state.borrow_mut();
            s.busy = false;
            // A lost device outranks whatever the load returned.
            s.alive()?;
            match result {
                Ok(loaded) if !s.unload_requested => {
                    let info = loaded.info();
                    s.model = Some(loaded);
                    Ok(info)
                }
                Ok(_) => {
                    s.unload_requested = false;
                    s.gpu.release_unused();
                    Err(error(Kind::Aborted, ""))
                }
                Err(e) => {
                    s.gpu.release_unused();
                    Err(e)
                }
            }
        })
    }

    /// OpenAI `chat.completions` request JSON → OpenAI chunk JSON strings via
    /// `onChunk` (when `stream: true`); resolves with the final completion JSON
    /// (`usage`, `finish_reason`, `tool_calls`). Rejects with `aborted` on `signal`.
    #[wasm_bindgen(unchecked_return_type = "Promise<string>")]
    pub fn chat(
        &self,
        #[wasm_bindgen(js_name = requestJson)] request_json: String,
        #[wasm_bindgen(js_name = onChunk, unchecked_param_type = "(chunkJson: string) => void")]
        on_chunk: Function,
        #[wasm_bindgen(unchecked_param_type = "AbortSignal")] signal: Option<Object>,
    ) -> Promise {
        let state = self.state.clone();
        future_to_promise(async move {
            let mut loaded = {
                let mut s = state.borrow_mut();
                if s.busy {
                    return Err(error(Kind::Busy, "another load or chat is running"));
                }
                s.alive()?;
                let Some(m) = s.model.take() else {
                    return Err(error(Kind::NotLoaded, "call load() first"));
                };
                s.busy = true;
                m
            };
            let flag = state.clone();
            let result = crate::chat::chat(
                &mut loaded,
                &request_json,
                &on_chunk,
                signal.as_ref(),
                // Stop at the next chunk on unload or a lost device.
                move || {
                    let s = flag.borrow();
                    s.unload_requested || s.gpu.lost().is_some()
                },
            )
            .await;
            let mut s = state.borrow_mut();
            s.busy = false;
            if s.unload_requested || s.gpu.lost().is_some() {
                s.unload_requested = false;
                drop(loaded);
                // Lost: reject webgpu_unavailable (nothing to release on a dead device).
                s.alive()?;
                s.gpu.release_unused();
            } else {
                s.model = Some(loaded);
            }
            result.map(JsValue::from)
        })
    }

    /// Drop the model and its GPU buffers. During a `chat`, the chat rejects
    /// with `aborted` and the model is dropped when it unwinds.
    pub fn unload(&self) {
        let mut s = self.state.borrow_mut();
        if s.busy {
            s.unload_requested = true;
            return;
        }
        if s.model.take().is_some() {
            s.gpu.release_unused();
        }
    }
}
