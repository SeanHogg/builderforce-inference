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
use std::io::{self, BufRead, Read, Write};
use std::sync::Arc;

/// The longest request line accepted. A skill with its screenshots' names is far smaller.
pub const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;

/// What reading one line produced.
#[derive(Debug, PartialEq)]
pub enum Line {
    Eof,
    Text(String),
    /// Not UTF-8; the line was consumed.
    NotUtf8,
    /// Over the cap; the rest of the line was discarded.
    TooLong,
}

/// Read one line without trusting its size or encoding.
pub fn read_line<R: BufRead>(r: &mut R, max: usize) -> io::Result<Line> {
    let mut buf = Vec::new();
    r.by_ref().take(max as u64 + 1).read_until(b'\n', &mut buf)?;
    if buf.is_empty() {
        return Ok(Line::Eof);
    }
    if buf.len() > max && buf.last() != Some(&b'\n') {
        // Over the cap: throw the remainder of this line away, in bounded pieces.
        loop {
            let mut rest = Vec::new();
            r.by_ref().take(64 * 1024).read_until(b'\n', &mut rest)?;
            if rest.is_empty() || rest.last() == Some(&b'\n') {
                break;
            }
        }
        return Ok(Line::TooLong);
    }
    Ok(match String::from_utf8(buf) {
        Ok(s) => Line::Text(s),
        Err(_) => Line::NotUtf8,
    })
}

/// Serve the protocol on this process's stdin/stdout until stdin closes.
pub fn run_stdio() {
    let emit: Emitter = Arc::new(|v| {
        let mut out = io::stdout().lock();
        let _ = writeln!(out, "{v}");
        let _ = out.flush();
    });
    let session = Session::new(Arc::new(DesktopBackend), emit);
    let mut stdin = io::stdin().lock();
    loop {
        match read_line(&mut stdin, MAX_LINE_BYTES) {
            Ok(Line::Text(line)) => {
                if !line.trim().is_empty() {
                    session.handle_line(&line);
                }
            }
            Ok(Line::NotUtf8) => session.reject("request is not valid UTF-8"),
            Ok(Line::TooLong) => session.reject("request line is too long"),
            Ok(Line::Eof) | Err(_) => break,
        }
    }
    session.shutdown();
}
