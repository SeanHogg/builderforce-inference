//! The engine thread: owns the [`EngineState`] (the model never leaves it), runs one
//! command at a time, and answers each through its command's own event function — so a
//! chat's chunks and its `done` arrive on the JS thread in the order they were made.
//!
//! What must be seen while a command runs is shared state the JS thread writes
//! directly: the running chat's abort flag, and the unload request.
//!
//! A panic inside a command is the native twin of a wasm trap: the engine's state can no
//! longer be trusted, so the command — and every one after it — answers `trap`, which the
//! JS face raises as a `RuntimeError` (what a host's trap handling already keys on: it
//! starts a fresh worker, and with it a fresh engine).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use bfi_engine::{BackendChoice, EngineState, HostProfile, LoadOptions};
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};

use crate::host::{block_on, ChannelSource, ChunkSlot, NativeHost};
use crate::support;

/// `(kind, payload)` to the JS side: `ready` / `progress` / `chunk` / `done` / `error`.
pub type Events = ThreadsafeFunction<(String, String), ErrorStrategy::Fatal>;

pub fn emit(events: &Events, kind: &str, payload: String) {
    events.call((kind.to_string(), payload), ThreadsafeFunctionCallMode::NonBlocking);
}

pub enum Command {
    Load {
        total_bytes: f64,
        options: LoadOptions,
        /// Built when the load is dispatched, so however the command ends — run, refused
        /// by a poisoned or lost engine, or dropped unread with the thread — its `Drop`
        /// closes the slot and rejects every queued `pushChunk`.
        chunks: ChannelSource,
        events: Events,
    },
    Chat {
        request_json: String,
        cancel: Arc<AtomicBool>,
        events: Events,
    },
    Unload,
}

/// What the JS thread and the engine thread both see.
#[derive(Default)]
pub struct Shared {
    /// A load or chat is dispatched and not yet settled — a second is `busy:`.
    pub busy: AtomicBool,
    /// `unload()` was called: the running load / chat ends `aborted`.
    pub unload: AtomicBool,
    /// The running (or queued) chat's abort flag.
    pub cancel: Mutex<Option<Arc<AtomicBool>>>,
    /// Where `pushChunk` sends during a load.
    pub chunks: ChunkSlot,
}

pub fn spawn(choice: BackendChoice, ready: Events, shared: Arc<Shared>) -> std::io::Result<Sender<Command>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("bfi-engine".into())
        .spawn(move || run(choice, rx, ready, shared))?;
    Ok(tx)
}

fn run(choice: BackendChoice, commands: Receiver<Command>, ready: Events, shared: Arc<Shared>) {
    let device = match support::open(choice) {
        Ok(d) => d,
        Err(e) => return emit(&ready, "error", e.to_string()),
    };
    let profile = support::profile();
    let mut state = EngineState::new(device);
    emit(&ready, "ready", state.backend().to_string());
    drop(ready);

    let mut poisoned: Option<String> = None;
    for command in commands {
        let events = match &command {
            Command::Load { events, .. } | Command::Chat { events, .. } => Some(events.clone()),
            Command::Unload => None,
        };
        if let Some(why) = &poisoned {
            // Drop first: a load's source must close its slot before the claim is
            // released, or it would close the NEXT load's.
            drop(command);
            shared.busy.store(false, Ordering::SeqCst);
            if let Some(events) = events {
                emit(&events, "trap", why.clone());
            }
            continue;
        }
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute(command, &mut state, &profile, &shared)
        }));
        if let Err(panic) = ran {
            let why = format!("the native engine panicked: {}", panic_message(&panic));
            shared.busy.store(false, Ordering::SeqCst);
            if let Some(events) = events {
                emit(&events, "trap", why.clone());
            }
            poisoned = Some(why);
        }
    }
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

/// Run one command on the engine thread and answer it.
fn execute(command: Command, state: &mut EngineState, profile: &HostProfile, shared: &Shared) {
    match command {
        Command::Load { total_bytes, options, mut chunks, events } => {
            let result = state.begin_load().and_then(|device| {
                let mut progress = |loaded: f64, total: f64| emit(&events, "progress", format!("[{loaded},{total}]"));
                let loaded = block_on(EngineState::load(&device, profile, &mut chunks, total_bytes, &mut progress, options));
                if shared.unload.load(Ordering::SeqCst) {
                    state.unload();
                }
                state.finish_load(loaded)
            });
            // Close this load's slot before the claim is released (the next load sets its own).
            drop(chunks);
            shared.busy.store(false, Ordering::SeqCst);
            match result {
                Ok(info) => emit(&events, "done", info.to_json().to_string()),
                Err(e) => emit(&events, "error", e.to_string()),
            }
        }
        Command::Chat { request_json, cancel, events } => {
            let result = state.begin_chat().and_then(|mut loaded| {
                let mut chunk = |c: &str| {
                    emit(&events, "chunk", c.to_string());
                    Ok(())
                };
                let stop = || cancel.load(Ordering::SeqCst) || shared.unload.load(Ordering::SeqCst);
                let answer = block_on(EngineState::chat(&mut loaded, &request_json, &NativeHost, &mut chunk, &stop));
                if shared.unload.load(Ordering::SeqCst) {
                    state.unload();
                }
                state.finish_chat(loaded, answer)
            });
            shared.busy.store(false, Ordering::SeqCst);
            match result {
                Ok(completion) => emit(&events, "done", completion),
                Err(e) => emit(&events, "error", e.to_string()),
            }
        }
        Command::Unload => {
            shared.unload.store(false, Ordering::SeqCst);
            state.unload();
        }
    }
}
