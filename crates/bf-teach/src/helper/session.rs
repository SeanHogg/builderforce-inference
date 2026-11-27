//! Request dispatch and the state behind it: at most one run and one recording, each on
//! its own thread so the stdin loop keeps reading (`run.stop` and `approve.answer` arrive
//! while a run is busy).

use super::backend::{Backend, RecordHandle};
use super::hooks::ChannelHooks;
use super::proto::{self, Emitter, Failure, PROTOCOL};
use crate::model::{new_id, now_ms, Episode, Skill};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const COUNT_POLL: Duration = Duration::from_millis(300);

struct ActiveRun {
    stop: Arc<AtomicBool>,
    answers: Sender<(usize, bool)>,
}

struct ActiveRecording {
    handle: Box<dyn RecordHandle>,
    program: String,
    args: Vec<String>,
    started_at: i64,
    watching: Arc<AtomicBool>,
}

type Slot<T> = Arc<Mutex<Option<T>>>;

pub struct Session {
    backend: Arc<dyn Backend>,
    emit: Emitter,
    run: Slot<ActiveRun>,
    recording: Slot<ActiveRecording>,
}

impl Session {
    pub fn new(backend: Arc<dyn Backend>, emit: Emitter) -> Self {
        Session { backend, emit, run: Arc::default(), recording: Arc::default() }
    }

    /// Handle one stdin line, emitting its answer.
    pub fn handle_line(&self, line: &str) {
        let answer = match proto::parse_request(line) {
            Err(answer) => answer,
            Ok(req) => match self.dispatch(&req.method, &req.params) {
                Ok(result) => proto::ok(&req.id, result),
                Err(f) => proto::err(&req.id, f),
            },
        };
        (self.emit)(answer);
    }

    fn dispatch(&self, method: &str, params: &Value) -> Result<Value, Failure> {
        match method {
            "hello" => Ok(json!({
                "protocol": PROTOCOL,
                "platform": self.backend.platform(),
                "supported": self.backend.supported(),
            })),
            "record.start" => self.record_start(params),
            "record.stop" => self.record_stop(),
            "run.start" => self.run_start(params),
            "run.stop" => {
                if let Some(r) = self.run.lock().unwrap().as_ref() {
                    r.stop.store(true, Ordering::SeqCst);
                }
                Ok(json!({}))
            }
            "approve.answer" => self.approve_answer(params),
            other => Err(Failure::new("unknown_method", format!("unknown method `{other}`"))),
        }
    }

    fn require_supported(&self) -> Result<(), Failure> {
        if self.backend.supported() {
            Ok(())
        } else {
            Err(Failure::new("unsupported", format!("not supported on {}", self.backend.platform())))
        }
    }

    // ------------------------------------------------------------ recording

    fn record_start(&self, params: &Value) -> Result<Value, Failure> {
        self.require_supported()?;
        let program = str_param(params, "program")?;
        let args: Vec<String> = parse_field(params, "args")?.unwrap_or_default();
        let shots = PathBuf::from(str_param(params, "shotsDir")?);
        let mut slot = self.recording.lock().unwrap();
        if slot.is_some() {
            return Err(Failure::new("busy", "a recording is already running"));
        }
        let handle = self
            .backend
            .start_recording(&program, &args, Path::new(&shots))
            .map_err(|e| Failure::new("record_failed", format!("{e:#}")))?;
        let watching = Arc::new(AtomicBool::new(true));
        *slot = Some(ActiveRecording { handle, program, args, started_at: now_ms(), watching: watching.clone() });
        drop(slot);
        self.watch_steps(watching);
        Ok(json!({}))
    }

    /// Push `record.step {count}` whenever the number of captured steps changes.
    fn watch_steps(&self, watching: Arc<AtomicBool>) {
        let (rec, emit) = (self.recording.clone(), self.emit.clone());
        let _ = std::thread::Builder::new().name("bf-teach-count".into()).spawn(move || {
            let mut last = 0;
            while watching.load(Ordering::SeqCst) {
                std::thread::sleep(COUNT_POLL);
                let Some(count) = rec.lock().unwrap().as_ref().map(|r| r.handle.count()) else { break };
                if count != last && watching.load(Ordering::SeqCst) {
                    last = count;
                    emit(proto::event("record.step", json!({ "count": count })));
                }
            }
        });
    }

    fn record_stop(&self) -> Result<Value, Failure> {
        let Some(rec) = self.recording.lock().unwrap().take() else {
            return Err(Failure::new("idle", "no recording is running"));
        };
        rec.watching.store(false, Ordering::SeqCst);
        let early_error = rec.handle.error();
        let steps = rec.handle.stop();
        if steps.is_empty() {
            return match early_error {
                Some(e) => Err(Failure::new("record_failed", e)),
                None => Ok(json!({ "episode": null })),
            };
        }
        let name = Path::new(&rec.program)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&rec.program)
            .to_string();
        let episode = Episode {
            id: new_id(),
            name,
            program: rec.program,
            args: rec.args,
            started_at: rec.started_at,
            ended_at: Some(now_ms()),
            steps,
        };
        let episode = serde_json::to_value(episode).map_err(|e| Failure::new("internal", e.to_string()))?;
        Ok(json!({ "episode": episode }))
    }

    // ----------------------------------------------------------------- runs

    fn run_start(&self, params: &Value) -> Result<Value, Failure> {
        self.require_supported()?;
        let skill: Skill = parse_field(params, "skill")?.ok_or_else(|| bad_params("`skill` is required"))?;
        let values: BTreeMap<String, String> = parse_field(params, "values")?.unwrap_or_default();
        // Before anything moves: a missing secret fails the request, not the run.
        crate::engine::resolve_params(&skill, &values).map_err(|e| Failure::new("missing_value", format!("{e:#}")))?;

        let mut slot = self.run.lock().unwrap();
        if slot.is_some() {
            return Err(Failure::new("busy", "a run is already in progress"));
        }
        let (tx, rx) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        *slot = Some(ActiveRun { stop: stop.clone(), answers: tx });
        drop(slot);

        let (backend, emit, run) = (self.backend.clone(), self.emit.clone(), self.run.clone());
        let spawned = std::thread::Builder::new().name("bf-teach-run".into()).spawn(move || {
            let mut hooks = ChannelHooks::new(emit.clone(), stop, rx);
            let outcome = catch_unwind(AssertUnwindSafe(|| backend.run(&skill, &values, &mut hooks)));
            // Free the slot first, so the host can start the next run on `run.finished`.
            *run.lock().unwrap() = None;
            let fields = match outcome {
                Ok(Ok(status)) => json!({ "status": status.as_str() }),
                Ok(Err(e)) => json!({ "status": "failed", "error": format!("{e:#}") }),
                Err(_) => json!({ "status": "failed", "error": "the run stopped unexpectedly" }),
            };
            emit(proto::event("run.finished", fields));
        });
        if let Err(e) = spawned {
            *self.run.lock().unwrap() = None;
            return Err(Failure::new("internal", format!("could not start the run: {e}")));
        }
        Ok(json!({}))
    }

    fn approve_answer(&self, params: &Value) -> Result<Value, Failure> {
        let idx = params.get("idx").and_then(Value::as_u64).ok_or_else(|| bad_params("`idx` must be a number"))?;
        let approved = params.get("approved").and_then(Value::as_bool).ok_or_else(|| bad_params("`approved` must be a boolean"))?;
        match self.run.lock().unwrap().as_ref() {
            Some(r) => {
                let _ = r.answers.send((idx as usize, approved));
                Ok(json!({}))
            }
            None => Err(Failure::new("idle", "no run is waiting for an answer")),
        }
    }
}

fn bad_params(message: &str) -> Failure {
    Failure::new("bad_params", message)
}

fn str_param(params: &Value, key: &str) -> Result<String, Failure> {
    params.get(key).and_then(Value::as_str).map(str::to_string).ok_or_else(|| bad_params(&format!("`{key}` must be a string")))
}

/// An optional typed field: absent or null is `None`; present but malformed is an error.
fn parse_field<T: serde::de::DeserializeOwned>(params: &Value, key: &str) -> Result<Option<T>, Failure> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => serde_json::from_value(v.clone()).map(Some).map_err(|e| bad_params(&format!("`{key}`: {e}"))),
    }
}
