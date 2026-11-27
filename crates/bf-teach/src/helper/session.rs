//! Request dispatch and the state behind it: at most one run and one recording, each on
//! its own thread so the stdin loop keeps reading (`run.stop` and `approve.answer` arrive
//! while a run is busy).

use super::backend::{Backend, RecordHandle};
use super::hooks::{ChannelHooks, Pending};
use super::proto::{self, lock, Emitter, Failure, PROTOCOL};
use crate::model::{new_id, now_ms, Episode, Skill};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const COUNT_POLL: Duration = Duration::from_millis(300);
/// How long the end of stdin waits for a stopped run to release the keyboard and mouse.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(2);

struct ActiveRun {
    run_id: String,
    stop: Arc<AtomicBool>,
    answers: Sender<(usize, bool)>,
    pending: Pending,
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

    /// Answer a line that could not even be read as text (id unknown, so null).
    pub fn reject(&self, message: &str) {
        (self.emit)(proto::err(&Value::Null, Failure::new("bad_request", message)));
    }

    /// stdin closed: stop any run and wait briefly for it to let go of the input devices.
    pub fn shutdown(&self) {
        if let Some(r) = lock(&self.run).as_ref() {
            r.stop.store(true, Ordering::SeqCst);
        }
        let deadline = Instant::now() + SHUTDOWN_WAIT;
        while lock(&self.run).is_some() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        if let Some(rec) = lock(&self.recording).take() {
            rec.watching.store(false, Ordering::SeqCst);
        }
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
                if let Some(r) = lock(&self.run).as_ref() {
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
        let args: Vec<String> = parse_field(params, "args", "`args` must be a list of strings")?.unwrap_or_default();
        let shots = PathBuf::from(str_param(params, "shotsDir")?);
        if lock(&self.run).is_some() {
            return Err(Failure::new("busy", "a run is in progress"));
        }
        let mut slot = lock(&self.recording);
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

    /// Push `record.step {count}` whenever the number of captured steps changes. The
    /// check and the emit happen under the recording lock, so nothing is pushed once
    /// `record.stop` has taken the recording.
    fn watch_steps(&self, watching: Arc<AtomicBool>) {
        let (rec, emit) = (self.recording.clone(), self.emit.clone());
        let _ = std::thread::Builder::new().name("bf-teach-count".into()).spawn(move || {
            let mut last = 0;
            loop {
                std::thread::sleep(COUNT_POLL);
                let guard = lock(&rec);
                let Some(r) = guard.as_ref() else { break };
                if !watching.load(Ordering::SeqCst) {
                    break;
                }
                let count = r.handle.count();
                if count != last {
                    last = count;
                    emit(proto::event("record.step", json!({ "count": count })));
                }
            }
        });
    }

    fn record_stop(&self) -> Result<Value, Failure> {
        let Some(rec) = lock(&self.recording).take() else {
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
        let run_id = str_param(params, "runId")?;
        // Parse errors never quote the input: `values` carries secrets.
        let skill: Skill = parse_field(params, "skill", "`skill` is not a valid skill")?
            .ok_or_else(|| bad_params("`skill` is required"))?;
        let values: BTreeMap<String, String> =
            parse_field(params, "values", "values must map names to strings")?.unwrap_or_default();
        // Before anything moves: a missing secret fails the request, not the run.
        crate::engine::resolve_params(&skill, &values).map_err(|e| Failure::new("missing_value", format!("{e:#}")))?;
        if lock(&self.recording).is_some() {
            return Err(Failure::new("busy", "a recording is in progress"));
        }

        let mut slot = lock(&self.run);
        if slot.is_some() {
            return Err(Failure::new("busy", "a run is already in progress"));
        }
        let (tx, rx) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let pending: Pending = Arc::default();
        *slot = Some(ActiveRun { run_id: run_id.clone(), stop: stop.clone(), answers: tx, pending: pending.clone() });
        drop(slot);

        let (backend, emit, run) = (self.backend.clone(), self.emit.clone(), self.run.clone());
        let spawned = std::thread::Builder::new().name("bf-teach-run".into()).spawn(move || {
            let mut hooks = ChannelHooks::new(emit.clone(), run_id.clone(), stop, rx, pending);
            let outcome = catch_unwind(AssertUnwindSafe(|| backend.run(&skill, &values, &mut hooks)));
            let fields = match outcome {
                Ok(Ok(status)) => json!({ "runId": run_id, "status": status.as_str() }),
                Ok(Err(e)) => json!({ "runId": run_id, "status": "failed", "error": format!("{e:#}") }),
                Err(_) => json!({ "runId": run_id, "status": "failed", "error": "the run stopped unexpectedly" }),
            };
            // Announce and free the slot under one lock: a `run.start` sent on seeing
            // `run.finished` waits for the slot instead of being told `busy`.
            let mut slot = lock(&run);
            emit(proto::event("run.finished", fields));
            *slot = None;
        });
        if let Err(e) = spawned {
            *lock(&self.run) = None;
            return Err(Failure::new("internal", format!("could not start the run: {e}")));
        }
        Ok(json!({}))
    }

    fn approve_answer(&self, params: &Value) -> Result<Value, Failure> {
        let idx = params.get("idx").and_then(Value::as_u64).ok_or_else(|| bad_params("`idx` must be a number"))? as usize;
        let run_id = str_param(params, "runId")?;
        let approved = params.get("approved").and_then(Value::as_bool).ok_or_else(|| bad_params("`approved` must be a boolean"))?;
        let run = lock(&self.run);
        let Some(r) = run.as_ref() else {
            return Err(Failure::new("idle", "no run is waiting for an answer"));
        };
        // Held across the send, so the gate cannot close between the check and the answer.
        // An answer meant for an earlier run (same idx, other runId) is not this gate's.
        let pending = lock(&r.pending);
        if r.run_id != run_id || *pending != Some(idx) {
            return Err(Failure::new("not_waiting", format!("step {idx} is not waiting for an answer")));
        }
        let _ = r.answers.send((idx, approved));
        Ok(json!({}))
    }
}

fn bad_params(message: &str) -> Failure {
    Failure::new("bad_params", message)
}

fn str_param(params: &Value, key: &str) -> Result<String, Failure> {
    params.get(key).and_then(Value::as_str).map(str::to_string).ok_or_else(|| bad_params(&format!("`{key}` must be a string")))
}

/// An optional typed field: absent or null is `None`; present but malformed is an error
/// carrying only `message`, never the serde text (which can echo the value).
fn parse_field<T: serde::de::DeserializeOwned>(params: &Value, key: &str, message: &str) -> Result<Option<T>, Failure> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => serde_json::from_value(v.clone()).map(Some).map_err(|_| bad_params(message)),
    }
}
