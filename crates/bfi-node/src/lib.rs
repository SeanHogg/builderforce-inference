//! `bfi-node`: bfi's own engine inside a Node process — the editor's extension host —
//! as a native addon. The engine is `bfi-engine` (the same state machine, load, chat and
//! `contextFor` `bfi-web` runs in a browser); this crate is its napi face:
//!
//! - [`thread`]: one engine thread per engine — it owns the model and runs one command
//!   at a time, so the JS thread never decodes.
//! - [`support`]: the native capability probe (a wgpu adapter, the machine's RAM) and
//!   the device opener: native wgpu on Vulkan (Metal on macOS), else the CPU on rayon.
//! - [`host`]: the engine thread's clock, executor and the pushed-chunk byte source.
//!
//! The JS face is `index.js` — the `BfiEngine` class with `bfi-web`'s contract
//! (`create` / `support` / `contextFor` / `load(ReadableStream, …)` / `chat` / `unload`),
//! built over this low-level surface: every command answers through its own
//! `(kind, payload)` event function, and `load` pulls the stream through `pushChunk`.

mod host;
mod support;
mod thread;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use bfi_engine::{context_for as plan_context, BackendChoice, LoadOptions};
use napi::bindgen_prelude::{AsyncTask, Buffer};
use napi::Task;
use napi::threadsafe_function::ThreadSafeCallContext;
use napi::{Env, JsFunction, JsObject};
use napi_derive::napi;

use crate::host::{Ack, ChannelSource, Chunk};
use crate::thread::{Command, Events, Shared};

fn reason(msg: impl Into<String>) -> napi::Error {
    napi::Error::from_reason(msg.into())
}

/// A JS `(kind, payload) => void` as the engine thread's event sink.
fn events(f: &JsFunction) -> napi::Result<Events> {
    f.create_threadsafe_function(0, |ctx: ThreadSafeCallContext<(String, String)>| {
        let (kind, payload) = ctx.value;
        Ok(vec![ctx.env.create_string(&kind)?, ctx.env.create_string(&payload)?])
    })
}

/// The capability record as JSON (`BfiSupport`), probed on the libuv pool: the first
/// probe requests a GPU adapter, which must not block the JS thread.
pub struct SupportTask;

impl Task for SupportTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> napi::Result<String> {
        Ok(support::probe().to_json().to_string())
    }

    fn resolve(&mut self, _env: Env, output: String) -> napi::Result<String> {
        Ok(output)
    }
}

#[napi(js_name = "support")]
pub fn support_json() -> AsyncTask<SupportTask> {
    AsyncTask::new(SupportTask)
}

/// `contextFor` over a `BfiContextShape` JSON, under this machine's CPU rules.
#[napi(js_name = "contextFor")]
pub fn context_for(shape_json: String) -> f64 {
    serde_json::from_str(&shape_json).map_or(0, |shape| plan_context(&shape, &support::profile())) as f64
}

/// Start an engine on `backend` (`webgpu` / `cpu` / `auto`, default `auto`). `onEvent`
/// gets `ready` with the backend name once the device is open, or `error`.
#[napi(js_name = "createEngine")]
pub fn create_engine(backend: Option<String>, on_event: JsFunction) -> napi::Result<NativeEngine> {
    let choice = BackendChoice::parse(backend.as_deref()).map_err(|e| reason(e.to_string()))?;
    let shared = Arc::new(Shared::default());
    let commands = thread::spawn(choice, events(&on_event)?, shared.clone())
        .map_err(|e| reason(format!("engine_error: could not start the engine thread: {e}")))?;
    Ok(NativeEngine {
        commands: Mutex::new(Some(commands)),
        shared,
    })
}

#[napi]
pub struct NativeEngine {
    commands: Mutex<Option<Sender<Command>>>,
    shared: Arc<Shared>,
}

impl NativeEngine {
    /// Claim the engine for a load / chat: `busy:` while one is dispatched.
    fn claim(&self) -> napi::Result<()> {
        if self.shared.busy.swap(true, Ordering::SeqCst) {
            return Err(reason("busy: another load or chat is running"));
        }
        Ok(())
    }

    /// Hand a command to the engine thread; false once it is gone.
    fn dispatch(&self, command: Command) -> bool {
        self.commands
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .is_some_and(|tx| tx.send(command).is_ok())
    }

    /// Dispatch a claimed load / chat, releasing the claim when nothing will run it.
    fn run_claimed(&self, command: Command) -> napi::Result<()> {
        if self.dispatch(command) {
            return Ok(());
        }
        self.shared.busy.store(false, Ordering::SeqCst);
        Err(reason("engine_error: the engine was disposed (or its device failed to open)"))
    }

    /// The event sink for a claimed command, releasing the claim when it cannot be made.
    fn claimed_events(&self, on_event: &JsFunction) -> napi::Result<Events> {
        events(on_event).inspect_err(|_| self.shared.busy.store(false, Ordering::SeqCst))
    }

    /// What `dispose` does — and what the finalizer does when the JS side goes away
    /// without one (a worker torn down mid-turn): end the running command and let the
    /// thread exit, so neither it nor the model outlives the engine.
    fn stop(&self) {
        self.shared.unload.store(true, Ordering::SeqCst);
        self.abort();
        // A load blocked on the next chunk: no sender left, so its read fails now.
        self.shared.chunks.lock().unwrap_or_else(|p| p.into_inner()).take();
        self.commands.lock().unwrap_or_else(|p| p.into_inner()).take();
    }

    fn push(&self, chunk: Chunk) -> Result<(), Chunk> {
        match self.shared.chunks.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            Some(tx) => tx.send(chunk).map_err(|e| e.0),
            None => Err(chunk),
        }
    }
}

#[napi]
impl NativeEngine {
    /// Begin a load of a `totalBytes` GGUF; the bytes follow through `pushChunk`, then
    /// `endChunks` (or `failChunks`). `onEvent`: `progress` (`[loaded,total]`), then
    /// `done` (`BfiModelInfo` JSON) or `error`.
    #[napi]
    pub fn load(&self, total_bytes: f64, options_json: String, on_event: JsFunction) -> napi::Result<()> {
        self.claim()?;
        let events = self.claimed_events(&on_event)?;
        let options = LoadOptions::from_json(serde_json::from_str(&options_json).ok().as_ref());
        let (tx, rx) = std::sync::mpsc::channel();
        *self.shared.chunks.lock().unwrap_or_else(|p| p.into_inner()) = Some(tx);
        let chunks = ChannelSource { rx, slot: self.shared.chunks.clone() };
        self.run_claimed(Command::Load { total_bytes, options, chunks, events })
    }

    /// One chunk of the GGUF; the promise settles when the engine takes it.
    #[napi(js_name = "pushChunk", ts_return_type = "Promise<void>")]
    pub fn push_chunk(&self, env: Env, chunk: Buffer) -> napi::Result<JsObject> {
        let (ack, promise): (Ack, JsObject) = env.create_deferred()?;
        if let Err(Chunk::Data(_, ack)) = self.push(Chunk::Data(chunk.to_vec(), ack)) {
            ack.reject(reason("load_failed: no load is reading chunks"));
        }
        Ok(promise)
    }

    /// The stream ended.
    #[napi(js_name = "endChunks")]
    pub fn end_chunks(&self) {
        let _ = self.push(Chunk::End);
    }

    /// The stream failed; the load ends `load_failed: stream errored: <message>`.
    #[napi(js_name = "failChunks")]
    pub fn fail_chunks(&self, message: String) {
        let _ = self.push(Chunk::Fail(message));
    }

    /// One OpenAI `chat.completions` request. `onEvent`: `chunk` (chunk JSON) for each
    /// streamed chunk, then `done` (the completion JSON) or `error`.
    #[napi]
    pub fn chat(&self, request_json: String, on_event: JsFunction) -> napi::Result<()> {
        self.claim()?;
        let events = self.claimed_events(&on_event)?;
        let cancel = Arc::new(AtomicBool::new(false));
        *self.shared.cancel.lock().unwrap_or_else(|p| p.into_inner()) = Some(cancel.clone());
        self.run_claimed(Command::Chat { request_json, cancel, events })
    }

    /// End the running chat at its next step (`aborted`).
    #[napi]
    pub fn abort(&self) {
        if let Some(cancel) = self.shared.cancel.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            cancel.store(true, Ordering::SeqCst);
        }
    }

    /// Drop the model; a running load / chat ends `aborted`.
    #[napi]
    pub fn unload(&self) {
        self.shared.unload.store(true, Ordering::SeqCst);
        if !self.dispatch(Command::Unload) {
            self.shared.unload.store(false, Ordering::SeqCst);
        }
    }

    /// Stop the engine thread: a running chat ends `aborted` at its next step, a running
    /// load at its next chunk (the stream is closed under it), and the model is freed.
    #[napi]
    pub fn dispose(&self) {
        self.stop();
    }
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        self.stop();
    }
}
