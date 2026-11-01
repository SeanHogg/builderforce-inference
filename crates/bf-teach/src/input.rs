//! The ONE global input hook, cross-platform: `rdev` delivers the person's clicks and keys
//! on Windows, macOS and X11 alike, so every platform implementation shares this module.
//!
//! `rdev` allows a process one `listen` loop, and it never returns, so it is started once
//! and fans events out to whoever is subscribed — the recorder (what the person does) and
//! replay (the Esc takeover). Only events that matter leave this module: left clicks with
//! their position and key presses with the modifier state. Mouse movement is kept here,
//! as the position of the next click, and never forwarded.

use rdev::{Button, EventType, Key};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Duration;

#[derive(Debug, Clone)]
pub enum Input {
    Click { x: i32, y: i32 },
    Key { key: Key, text: Option<String>, ctrl: bool, alt: bool, shift: bool, meta: bool },
}

/// Backlog of observed input, drained by `next` with a timeout.
static QUEUE: Mutex<VecDeque<Input>> = Mutex::new(VecDeque::new());
static WAKE: Condvar = Condvar::new();
static STARTED: OnceLock<()> = OnceLock::new();
/// Set while replay is injecting input, so the hook does not report the skill's own
/// keystrokes as the person's (an injected Esc must not stop the run).
static INJECTING: AtomicBool = AtomicBool::new(false);

/// Ensure the listen loop is running and the hook is allowed to observe input.
pub fn subscribe() {
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("bf-input-hook".into()).spawn(listen);
    });
}

/// The next observed input, waiting up to `timeout`. `None` on timeout.
pub fn next(timeout: Duration) -> Result<Input, std::sync::mpsc::RecvTimeoutError> {
    let deadline = std::time::Instant::now() + timeout;
    let mut q = QUEUE.lock().unwrap();
    loop {
        if let Some(ev) = q.pop_front() {
            return Ok(ev);
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            return Err(std::sync::mpsc::RecvTimeoutError::Timeout);
        }
        let (guard, _) = WAKE.wait_timeout(q, deadline - now).unwrap();
        q = guard;
    }
}

/// Run `f` with the hook ignoring input, for input replay generates itself.
pub fn injecting<T>(f: impl FnOnce() -> T) -> T {
    INJECTING.store(true, Ordering::SeqCst);
    let out = f();
    // Injected events are asynchronous: give the hook time to see (and drop) the last one.
    std::thread::sleep(Duration::from_millis(60));
    INJECTING.store(false, Ordering::SeqCst);
    out
}

fn publish(input: Input) {
    QUEUE.lock().unwrap().push_back(input);
    WAKE.notify_one();
}

fn listen() {
    let (mut x, mut y) = (0.0f64, 0.0f64);
    let (mut ctrl, mut alt, mut shift, mut meta) = (false, false, false, false);
    let result = rdev::listen(move |event| {
        match event.event_type {
            EventType::MouseMove { x: mx, y: my } => {
                x = mx;
                y = my;
            }
            EventType::KeyPress(k) | EventType::KeyRelease(k) => {
                let down = matches!(event.event_type, EventType::KeyPress(_));
                match k {
                    Key::ControlLeft | Key::ControlRight => ctrl = down,
                    Key::Alt | Key::AltGr => alt = down,
                    Key::ShiftLeft | Key::ShiftRight => shift = down,
                    Key::MetaLeft | Key::MetaRight => meta = down,
                    _ if down && !INJECTING.load(Ordering::SeqCst) => {
                        publish(Input::Key { key: k, text: event.name.clone(), ctrl, alt, shift, meta });
                    }
                    _ => {}
                }
            }
            EventType::ButtonPress(Button::Left) if !INJECTING.load(Ordering::SeqCst) => {
                publish(Input::Click { x: x.round() as i32, y: y.round() as i32 });
            }
            _ => {}
        }
    });
    if let Err(e) = result {
        eprintln!("bf-teach: input hook failed: {e:?}");
    }
}
