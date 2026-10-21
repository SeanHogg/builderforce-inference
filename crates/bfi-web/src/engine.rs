//! The exported `BfiEngine` class. Owns its backend (a WebGPU device, or the CPU) and
//! at most one loaded model; `load` / `chat` are exclusive (a second call while one
//! runs rejects with `busy:`), `unload` during a chat aborts it. Once a WebGPU device
//! is lost (`device.lost`), the model is dropped and every later `load` / `chat`
//! rejects with `webgpu_unavailable:` — create a new engine.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Function, Object, Promise};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use crate::device::{BackendChoice, Device};
use crate::errors::{error, Kind};
use crate::loader::{self, LoadOptions, Loaded};
use crate::support;

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
export type BfiBackend = "webgpu" | "cpu";
export interface BfiCreateOptions {
  /** `auto` (default): WebGPU when an adapter exists, else the CPU. */
  backend?: BfiBackend | "auto";
}
export interface BfiModelInfo {
  arch: string; backend: BfiBackend; contextTokens: number; vocab: number; params: number; kvF16: boolean
}
export interface BfiLoadOptions {
  /** Total KV-cache budget. Default 2 GiB on WebGPU, 512 MiB on the CPU. */
  kvBudgetBytes?: number
}
export interface BfiContextShape {
  layers: number; kvHeads: number; headDim: number; trainedContext: number;
  /** Which backend's rule to apply (default `webgpu`). `cpu`: f32 KV, no buffer cap,
   *  512 MiB default budget — what a CPU `load` serves. */
  backend?: BfiBackend;
  kvBudgetBytes?: number; f16?: boolean;
  maxStorageBufferBindingSizeBytes?: number;
  maxBufferSizeBytes?: number;
  /** @deprecated Alias of maxBufferSizeBytes (as in BfiSupport); min-ed with the binding limit. */
  maxBufferBytes?: number;
}
export interface BfiWebGpuSupport {
  supported: boolean; reason?: string;
  maxBufferBytes: number; maxStorageBufferBindingSizeBytes: number; f16: boolean
}
export interface BfiCpuSupport {
  supported: true;
  /** Built with WebAssembly SIMD128. */
  simd: boolean;
  /** Threads the kernels run on (1: single-threaded build). */
  threads: number;
  /** The CPU backend's default KV budget. */
  kvBudgetBytes: number;
  /** wasm32's memory ceiling: weights + KV + working set must fit under it. */
  maxMemoryBytes: number;
  /** `navigator.deviceMemory` in bytes, where the browser reports it. */
  deviceMemoryBytes?: number
}
export interface BfiSupport {
  /** Top level = the WebGPU adapter, unchanged since version 2. */
  contractVersion: 2; supported: boolean; reason?: string;
  maxBufferSizeBytes: number; maxStorageBufferBindingSizeBytes: number;
  /** @deprecated Use maxBufferSizeBytes. */
  maxBufferBytes: number; requestedPowerPreference: string; f16: boolean;
  backends: { webgpu: BfiWebGpuSupport; cpu: BfiCpuSupport }
}
"#;

struct State {
    device: Device,
    model: Option<Loaded>,
    busy: bool,
    unload_requested: bool,
}

impl State {
    /// `Err(webgpu_unavailable)` once the device is lost; drops the model (its
    /// buffers belong to the dead device, so there is nothing to release).
    fn alive(&mut self) -> Result<(), JsValue> {
        let Some(why) = self.device.lost() else {
            return Ok(());
        };
        self.model = None;
        Err(error(
            Kind::WebgpuUnavailable,
            format!("the GPU device was lost ({why}); create a new BfiEngine"),
        ))
    }
}

/// bfi in the browser, on WebGPU or the CPU. See `README.md` for the JS contract.
#[wasm_bindgen]
pub struct BfiEngine {
    state: Rc<RefCell<State>>,
}

#[wasm_bindgen]
impl BfiEngine {
    /// Open a backend: `webgpu` (adapter + device, `shader-f16` when offered; rejects
    /// `webgpu_unavailable: <reason>` without an adapter), `cpu`, or `auto` (default).
    pub async fn create(
        #[wasm_bindgen(unchecked_param_type = "BfiCreateOptions")] options: Option<Object>,
    ) -> Result<BfiEngine, JsValue> {
        console_error_panic_hook::set_once();
        let choice = BackendChoice::from_js(options.as_ref())?;
        let device = Device::open(choice).await?;
        Ok(BfiEngine {
            state: Rc::new(RefCell::new(State {
                device,
                model: None,
                busy: false,
                unload_requested: false,
            })),
        })
    }

    /// The backend this engine runs on.
    #[wasm_bindgen(getter, unchecked_return_type = "BfiBackend")]
    pub fn backend(&self) -> String {
        self.state.borrow().device.name().to_string()
    }

    /// Every backend's capabilities; requests a WebGPU adapter only — no device — so it
    /// is safe on every capability check.
    #[wasm_bindgen(unchecked_return_type = "Promise<BfiSupport>")]
    pub fn support() -> Promise {
        future_to_promise(async { Ok(support::probe().await.to_js()) })
    }

    /// Predict the context `load` will serve for a model shape, before
    /// downloading it (pure; same function the engine uses). WebGPU defaults:
    /// `kvBudgetBytes` 2 GiB, `f16` false, buffer cap 128 MiB (WebGPU's
    /// guaranteed binding); the cap is `min(maxStorageBufferBindingSizeBytes,
    /// maxBufferSizeBytes ?? maxBufferBytes)` of whichever are given, exactly as
    /// `load` caps the adapter. `backend: 'cpu'`: f32 KV, 512 MiB default budget, no
    /// cap. The engine may serve less if a WebGPU allocation fails.
    #[wasm_bindgen(js_name = contextFor)]
    pub fn context_for(
        #[wasm_bindgen(unchecked_param_type = "BfiContextShape")] shape: Object,
    ) -> f64 {
        crate::plan::context_for_js(&shape) as f64
    }

    /// Load a GGUF streamed by the host (the host owns download + cache).
    /// Weights stay packed — one GPU buffer per tensor, or packed in wasm memory.
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
            let device = {
                let mut s = state.borrow_mut();
                if s.busy {
                    return Err(error(Kind::Busy, "another load or chat is running"));
                }
                s.alive()?;
                s.busy = true;
                s.unload_requested = false;
                if s.model.take().is_some() {
                    s.device.release_unused();
                }
                s.device.clone()
            };
            let result = loader::load(&device, &chunks, total_bytes, on_progress.as_ref(), opts).await;
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
                    s.device.release_unused();
                    Err(error(Kind::Aborted, ""))
                }
                Err(e) => {
                    s.device.release_unused();
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
                    s.unload_requested || s.device.lost().is_some()
                },
            )
            .await;
            let mut s = state.borrow_mut();
            s.busy = false;
            if s.unload_requested || s.device.lost().is_some() {
                s.unload_requested = false;
                drop(loaded);
                // Lost: reject webgpu_unavailable (nothing to release on a dead device).
                s.alive()?;
                s.device.release_unused();
            } else {
                s.model = Some(loaded);
            }
            result.map(JsValue::from)
        })
    }

    /// Drop the model and its memory. During a `chat`, the chat rejects with
    /// `aborted` and the model is dropped when it unwinds.
    pub fn unload(&self) {
        let mut s = self.state.borrow_mut();
        if s.busy {
            s.unload_requested = true;
            return;
        }
        if s.model.take().is_some() {
            s.device.release_unused();
        }
    }
}
