//! Protocol tests against a fake backend; nothing here touches the real desktop.

use super::backend::{Backend, RecordHandle};
use super::hooks::ChannelHooks;
use super::proto::{self, Emitter};
use super::session::Session;
use crate::model::{Action, ElementRef, RecordedStep, RunStatus, Skill};
use crate::RunHooks;
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct FakeRec(usize);

impl RecordHandle for FakeRec {
    fn count(&self) -> usize {
        self.0
    }
    fn error(&self) -> Option<String> {
        None
    }
    fn stop(self: Box<Self>) -> Vec<RecordedStep> {
        (0..self.0)
            .map(|i| RecordedStep {
                id: format!("s{i}"),
                at_ms: i as u64,
                action: Action::Click { target: ElementRef::default() },
                screenshot: None,
            })
            .collect()
    }
}

/// A run with one approval-gated step; it ends when the gate is answered.
struct FakeBackend {
    supported: bool,
    recorded_steps: usize,
}

impl Backend for FakeBackend {
    fn platform(&self) -> &'static str {
        "fake"
    }
    fn supported(&self) -> bool {
        self.supported
    }
    fn start_recording(&self, _: &str, _: &[String], _: &Path) -> Result<Box<dyn RecordHandle>> {
        Ok(Box::new(FakeRec(self.recorded_steps)))
    }
    fn run(&self, _: &Skill, _: &BTreeMap<String, String>, hooks: &mut dyn RunHooks) -> Result<RunStatus> {
        hooks.step_started(0);
        if !hooks.approve(0) {
            return Ok(RunStatus::Denied);
        }
        Ok(RunStatus::Succeeded)
    }
}

type Log = Arc<Mutex<Vec<Value>>>;

fn session(supported: bool, recorded_steps: usize) -> (Session, Log) {
    let log: Log = Arc::default();
    let sink = log.clone();
    let emit: Emitter = Arc::new(move |v| sink.lock().unwrap().push(v));
    (Session::new(Arc::new(FakeBackend { supported, recorded_steps }), emit), log)
}

fn wait_for(log: &Log, mut pred: impl FnMut(&Value) -> bool) -> Value {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(v) = log.lock().unwrap().iter().find(|v| pred(v)) {
            return v.clone();
        }
        assert!(Instant::now() < deadline, "timed out; saw {:?}", log.lock().unwrap());
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn skill_json(secret_param: bool) -> Value {
    let params = if secret_param {
        json!([{ "name": "pw", "label": "Password", "secret": true }])
    } else {
        json!([])
    };
    json!({ "id": "s", "name": "n", "program": "p.exe", "args": [], "params": params, "steps": [] })
}

fn call(s: &Session, log: &Log, id: u64, method: &str, params: Value) -> Value {
    s.handle_line(&json!({ "id": id, "method": method, "params": params }).to_string());
    wait_for(log, |v| v["id"] == json!(id))
}

#[test]
fn parses_requests_and_rejects_garbage() {
    let r = proto::parse_request(r#"{"id":7,"method":"hello"}"#).unwrap();
    assert_eq!(r.id, json!(7));
    assert_eq!(r.method, "hello");
    assert!(r.params.is_null());

    let bad = proto::parse_request("not json").unwrap_err();
    assert_eq!(bad["error"]["code"], "bad_request");
    let no_method = proto::parse_request(r#"{"id":"x"}"#).unwrap_err();
    assert_eq!(no_method["id"], "x");
    assert_eq!(no_method["error"]["code"], "bad_request");
}

#[test]
fn hello_reports_protocol_platform_and_support() {
    let (s, log) = session(true, 0);
    let a = call(&s, &log, 1, "hello", Value::Null);
    assert_eq!(a["result"], json!({ "protocol": 1, "platform": "fake", "supported": true }));
}

#[test]
fn unknown_method_is_an_error_answer() {
    let (s, log) = session(true, 0);
    let a = call(&s, &log, 2, "nope", Value::Null);
    assert_eq!(a["error"]["code"], "unknown_method");
}

#[test]
fn unsupported_platform_refuses_to_run() {
    let (s, log) = session(false, 0);
    let a = call(&s, &log, 1, "run.start", json!({ "skill": skill_json(false), "values": {} }));
    assert_eq!(a["error"]["code"], "unsupported");
}

#[test]
fn a_missing_secret_fails_run_start_before_anything_runs() {
    let (s, log) = session(true, 0);
    let a = call(&s, &log, 1, "run.start", json!({ "skill": skill_json(true), "values": {} }));
    assert_eq!(a["error"]["code"], "missing_value");
    assert!(log.lock().unwrap().iter().all(|v| v.get("event").is_none()), "no run events");

    let ok = call(&s, &log, 2, "run.start", json!({ "skill": skill_json(true), "values": { "pw": "x" } }));
    assert_eq!(ok["result"], json!({}));
}

#[test]
fn a_second_run_is_busy_and_the_answer_reaches_the_gate() {
    let (s, log) = session(true, 0);
    let start = json!({ "skill": skill_json(false), "values": {} });
    assert_eq!(call(&s, &log, 1, "run.start", start.clone())["result"], json!({}));
    wait_for(&log, |v| v["event"] == "run.approvalNeeded" && v["idx"] == 0);

    assert_eq!(call(&s, &log, 2, "run.start", start.clone())["error"]["code"], "busy");

    assert_eq!(call(&s, &log, 3, "approve.answer", json!({ "idx": 0, "approved": true }))["result"], json!({}));
    let done = wait_for(&log, |v| v["event"] == "run.finished");
    assert_eq!(done["status"], "succeeded");

    // The slot is free again once the run has finished.
    assert_eq!(call(&s, &log, 4, "run.start", start)["result"], json!({}));
}

#[test]
fn run_stop_declines_a_pending_approval() {
    let (s, log) = session(true, 0);
    call(&s, &log, 1, "run.start", json!({ "skill": skill_json(false), "values": {} }));
    wait_for(&log, |v| v["event"] == "run.approvalNeeded");
    call(&s, &log, 2, "run.stop", Value::Null);
    assert_eq!(wait_for(&log, |v| v["event"] == "run.finished")["status"], "denied");
}

#[test]
fn answering_with_no_run_is_an_error() {
    let (s, log) = session(true, 0);
    let a = call(&s, &log, 1, "approve.answer", json!({ "idx": 0, "approved": true }));
    assert_eq!(a["error"]["code"], "idle");
}

#[test]
fn approval_blocks_until_the_matching_idx_is_answered() {
    let (tx, rx) = channel();
    let events: Log = Arc::default();
    let sink = events.clone();
    let emit: Emitter = Arc::new(move |v| sink.lock().unwrap().push(v));
    let mut hooks = ChannelHooks::new(emit, Arc::new(AtomicBool::new(false)), rx);
    // A stale answer for another step is skipped; the one for idx 3 decides.
    tx.send((7, true)).unwrap();
    tx.send((3, false)).unwrap();
    assert!(!hooks.approve(3));
    assert_eq!(events.lock().unwrap()[0], json!({ "event": "run.approvalNeeded", "idx": 3 }));
    tx.send((4, true)).unwrap();
    assert!(hooks.approve(4));
}

#[test]
fn recording_returns_the_episode_in_the_engines_shape() {
    let (s, log) = session(true, 2);
    call(&s, &log, 1, "record.start", json!({ "program": "C:/x/ledger.exe", "args": ["-a"], "shotsDir": "shots" }));
    let a = call(&s, &log, 2, "record.stop", Value::Null);
    let ep = &a["result"]["episode"];
    assert_eq!(ep["name"], "ledger");
    assert_eq!(ep["args"], json!(["-a"]));
    assert_eq!(ep["steps"].as_array().unwrap().len(), 2);
    assert_eq!(ep["steps"][0]["action"]["kind"], "click");
    assert!(ep["startedAt"].is_i64() && ep["endedAt"].is_i64());
}

#[test]
fn an_empty_recording_is_a_null_episode() {
    let (s, log) = session(true, 0);
    call(&s, &log, 1, "record.start", json!({ "program": "p", "shotsDir": "shots" }));
    assert_eq!(call(&s, &log, 2, "record.stop", Value::Null)["result"], json!({ "episode": null }));
}
