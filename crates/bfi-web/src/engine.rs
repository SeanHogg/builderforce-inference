//! The exported `BfiEngine` class — `bfi_engine::EngineState` behind wasm-bindgen
//! promises. Owns its backend (a WebGPU device, or the CPU) and at most one loaded
//! model; `load` / `chat` are exclusive (a second call while one runs rejects with
//! `busy:`), `unload` during a chat aborts it. Once a WebGPU device is lost
//! (`device.lost`), the model is dropped and every later `load` / `chat` rejects with
//! `webgpu_unavailable:` — create a new engine.

use std::cell::RefCell;
use std::rc::Rc;

#[cfg(feature = "webgpu")]
use bfi_core::resident::open_webgpu;
use bfi_engine::{context_for, error, BackendChoice, Device, EngineState, HostProfile, Kind, LoadOptions};
use js_sys::{Function, Object, Promise};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

use crate::js::{self, js_error, StreamSource, WebHost};
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
  /** The memory the CPU backend may plan to use: min(wasm32's 4 GiB, half the device's
   *  RAM; an unreported RAM reads as 2 GiB). Weights + KV + working set must fit. */
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

/// Open the backend `choice` names. The browser cannot block, so the GPU is opened by
/// awaiting CubeCL's setup — and only after `requestAdapter()` confirmed an adapter
/// (CubeCL panics without one, and a wasm panic aborts the module).
async fn open(choice: BackendChoice) -> Result<Device, JsValue> {
    if choice == BackendChoice::Cpu {
        return Ok(Device::Cpu);
    }
    let adapter = support::probe_webgpu().await;
    if !choice
        .wants_gpu(adapter.supported, adapter.reason.as_deref())
        .map_err(js_error)?
    {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "webgpu")]
    {
        let gpu = open_webgpu().await;
        Ok(Device::Gpu { gpu, adapter })
    }
    #[cfg(not(feature = "webgpu"))]
    unreachable!("probe_webgpu reports unsupported in a build without WebGPU")
}

/// bfi in the browser, on WebGPU or the CPU. See `README.md` for the JS contract.
#[wasm_bindgen]
pub struct BfiEngine {
    state: Rc<RefCell<EngineState>>,
}

#[wasm_bindgen]
impl BfiEngine {
    /// Open a backend: `webgpu` (adapter + device, `shader-f16` when offered; rejects
    /// `webgpu_unavailable: <reason>` without an adapter), `cpu`, or `auto` (default).
    pub async fn create(
        #[wasm_bindgen(unchecked_param_type = "BfiCreateOptions")] options: Option<Object>,
    ) -> Result<BfiEngine, JsValue> {
        console_error_panic_hook::set_once();
        let raw = options.as_ref().map(|o| js::get(o, "backend")).filter(js::present);
        let name = raw.map(|v| v.as_string().unwrap_or_else(|| format!("{v:?}")));
        let choice = BackendChoice::parse(name.as_deref()).map_err(js_error)?;
        let device = open(choice).await?;
        Ok(BfiEngine {
            state: Rc::new(RefCell::new(EngineState::new(device))),
        })
    }

    /// The backend this engine runs on.
    #[wasm_bindgen(getter, unchecked_return_type = "BfiBackend")]
    pub fn backend(&self) -> String {
        self.state.borrow().backend().to_string()
    }

    /// Every backend's capabilities; requests a WebGPU adapter only — no device — so it
    /// is safe on every capability check.
    #[wasm_bindgen(unchecked_return_type = "Promise<BfiSupport>")]
    pub fn support() -> Promise {
        future_to_promise(async { Ok(js::to_js(&support::probe().await.to_json())) })
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
        js::to_json(&shape).map_or(0, |s| context_for(&s, &HostProfile::WASM)) as f64
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
        let opts = LoadOptions::from_json(options.as_ref().and_then(|o| js::to_json(o)).as_ref());
        let state = self.state.clone();
        future_to_promise(async move {
            let device = state.borrow_mut().begin_load().map_err(js_error)?;
            let mut progress = |loaded: f64, total: f64| {
                if let Some(cb) = &on_progress {
                    let _ = cb.call2(&JsValue::NULL, &JsValue::from_f64(loaded), &JsValue::from_f64(total));
                }
            };
            let result = match StreamSource::new(&chunks) {
                Ok(mut source) => {
                    let r = EngineState::load(&device, &HostProfile::WASM, &mut source, total_bytes, &mut progress, opts).await;
                    source.release();
                    r
                }
                Err(e) => Err(e),
            };
            let info = state.borrow_mut().finish_load(result).map_err(js_error)?;
            Ok(js::to_js(&info.to_json()))
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
            let mut loaded = state.borrow_mut().begin_chat().map_err(js_error)?;
            let mut emit = |chunk: &str| {
                on_chunk
                    .call1(&JsValue::NULL, &JsValue::from_str(chunk))
                    .map(|_| ())
                    .map_err(|e| error(Kind::Engine, format!("onChunk threw: {e:?}")))
            };
            // Stop at the next step on the signal, an unload, or a lost device.
            let stop = || js::aborted(signal.as_ref()) || state.borrow().should_stop();
            let result = EngineState::chat(&mut loaded, &request_json, &WebHost, &mut emit, &stop).await;
            state
                .borrow_mut()
                .finish_chat(loaded, result)
                .map(JsValue::from)
                .map_err(js_error)
        })
    }

    /// Drop the model and its memory. During a `chat`, the chat rejects with
    /// `aborted` and the model is dropped when it unwinds.
    pub fn unload(&self) {
        self.state.borrow_mut().unload();
    }
}
