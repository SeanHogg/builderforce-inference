//! [`RunHooks`] over the protocol: progress goes out as events, and an approval gate
//! blocks the run thread until the matching `approve.answer` arrives.

use super::proto::{event, Emitter};
use crate::model::StepOutcome;
use crate::RunHooks;
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

/// How often a blocked approval re-checks for Stop.
const STOP_POLL: Duration = Duration::from_millis(200);

pub struct ChannelHooks {
    emit: Emitter,
    stop: Arc<AtomicBool>,
    answers: Receiver<(usize, bool)>,
}

impl ChannelHooks {
    pub fn new(emit: Emitter, stop: Arc<AtomicBool>, answers: Receiver<(usize, bool)>) -> Self {
        ChannelHooks { emit, stop, answers }
    }
}

impl RunHooks for ChannelHooks {
    fn step_started(&mut self, idx: usize) {
        (self.emit)(event("run.stepStarted", json!({ "idx": idx })));
    }

    fn step_finished(&mut self, idx: usize, outcome: StepOutcome, detail: Option<String>) {
        let mut f = json!({ "idx": idx, "outcome": outcome.as_str() });
        if let Some(d) = detail {
            f["detail"] = json!(d);
        }
        (self.emit)(event("run.stepFinished", f));
    }

    fn approve(&mut self, idx: usize) -> bool {
        (self.emit)(event("run.approvalNeeded", json!({ "idx": idx })));
        loop {
            match self.answers.recv_timeout(STOP_POLL) {
                // An answer for another step (a late one) is not this gate's.
                Ok((i, approved)) if i == idx => return approved,
                Ok(_) => {}
                // Stop counts as "no", as does a host that went away.
                Err(RecvTimeoutError::Timeout) if self.stop.load(Ordering::SeqCst) => return false,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return false,
            }
        }
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}
