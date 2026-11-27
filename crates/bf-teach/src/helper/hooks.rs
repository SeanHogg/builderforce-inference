//! [`RunHooks`] over the protocol: progress goes out as events, and an approval gate
//! blocks the run thread until the matching `approve.answer` arrives.

use super::proto::{event, lock, Emitter};
use crate::model::StepOutcome;
use crate::RunHooks;
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How often a blocked approval re-checks for Stop.
const STOP_POLL: Duration = Duration::from_millis(200);

/// The step whose gate is open right now, shared with the session so an answer is only
/// accepted for a gate that is actually waiting.
pub type Pending = Arc<Mutex<Option<usize>>>;

pub struct ChannelHooks {
    emit: Emitter,
    stop: Arc<AtomicBool>,
    answers: Receiver<(usize, bool)>,
    pending: Pending,
    run_id: String,
}

impl ChannelHooks {
    pub fn new(emit: Emitter, run_id: String, stop: Arc<AtomicBool>, answers: Receiver<(usize, bool)>, pending: Pending) -> Self {
        ChannelHooks { emit, stop, answers, pending, run_id }
    }

    fn wait_for_answer(&self, idx: usize) -> bool {
        loop {
            match self.answers.recv_timeout(STOP_POLL) {
                Ok((i, approved)) if i == idx => return approved,
                Ok(_) => {}
                // Stop counts as "no", as does a host that went away.
                Err(RecvTimeoutError::Timeout) if self.stop.load(Ordering::SeqCst) => return false,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return false,
            }
        }
    }
}

impl RunHooks for ChannelHooks {
    fn step_started(&mut self, idx: usize) {
        (self.emit)(event("run.stepStarted", json!({ "runId": self.run_id, "idx": idx })));
    }

    fn step_finished(&mut self, idx: usize, outcome: StepOutcome, detail: Option<String>) {
        let mut f = json!({ "runId": self.run_id, "idx": idx, "outcome": outcome.as_str() });
        if let Some(d) = detail {
            f["detail"] = json!(d);
        }
        (self.emit)(event("run.stepFinished", f));
    }

    fn approve(&mut self, idx: usize) -> bool {
        // An answer that arrived before the gate opened is not an answer to it.
        while self.answers.try_recv().is_ok() {}
        *lock(&self.pending) = Some(idx);
        (self.emit)(event("run.approvalNeeded", json!({ "runId": self.run_id, "idx": idx })));
        let approved = self.wait_for_answer(idx);
        *lock(&self.pending) = None;
        approved
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}
