//! The helper process's JSON-lines stdio protocol (see the Synapse fork design, §5).
//!
//! Requests arrive one JSON object per line on stdin; answers and events leave one per
//! line on stdout. Secrets travel only inside `run.start` on stdin.

pub mod backend;
pub mod hooks;
pub mod proto;
pub mod session;

#[cfg(test)]
mod tests;

use backend::DesktopBackend;
use proto::Emitter;
use session::Session;
use std::io::{BufRead, Write};
use std::sync::Arc;

/// Serve the protocol on this process's stdin/stdout until stdin closes.
pub fn run_stdio() {
    let emit: Emitter = Arc::new(|v| {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{v}");
        let _ = out.flush();
    });
    let session = Session::new(Arc::new(DesktopBackend), emit);
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if !line.trim().is_empty() {
            session.handle_line(&line);
        }
    }
}
