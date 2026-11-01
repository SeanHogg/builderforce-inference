//! Recording and replay, written once against [`crate::platform::Platform`].
//!
//! A click becomes "clicked <element>". Typing becomes one "set <field> to <value>" when
//! the person moves on — the value read back from the field itself, so a typo and its
//! correction record as the final text. Enter, Tab, arrows and shortcuts are steps of
//! their own. Password fields record that they were filled, never what with. Input to any
//! other program, including Synapse itself, is ignored.
//!
//! Replay finds each element by accessibility identity (retrying while the UI catches up),
//! acts through the platform and verifies, or — when the element cannot be found — acts at
//! its recorded position inside the window and says so in the audit trail. Steps marked for
//! approval wait for the person. Esc, or Stop in the window, ends the run at once.

use crate::model::{new_id, Action, ElementRef, RecordedStep, RunStatus, Skill, SkillAction, SkillValue, StepOutcome};
use crate::platform::{KeyName, Observed, Platform};
use crate::procs::Scope;
use crate::RunHooks;
use anyhow::{anyhow, Result};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const WINDOW_WAIT: Duration = Duration::from_secs(20);
const ELEMENT_WAIT: Duration = Duration::from_secs(5);
const SETTLE: Duration = Duration::from_millis(400);
const POLL: Duration = Duration::from_millis(250);

// --------------------------------------------------------------- recording

pub struct Recording {
    stop: Arc<AtomicBool>,
    steps: Arc<Mutex<Vec<RecordedStep>>>,
    error: Arc<Mutex<Option<String>>>,
    handle: Option<JoinHandle<()>>,
}

impl Recording {
    /// The steps so far (the window shows them live).
    pub fn steps(&self) -> Vec<RecordedStep> {
        self.steps.lock().unwrap().clone()
    }

    /// Why recording ended early, if it did (the program could not start, for one).
    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }

    pub fn is_running(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    /// Stop and return every step, with any half-typed value flushed.
    pub fn stop(mut self) -> Vec<RecordedStep> {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.steps()
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

/// Launch `program` and start recording it. Screenshots go into `shots_dir`.
pub fn start_recording(program: &str, args: &[String], shots_dir: &Path) -> Result<Recording> {
    if !Path::new(program).exists() {
        return Err(anyhow!("{program} was not found"));
    }
    let child = std::process::Command::new(program)
        .args(args)
        .spawn()
        .map_err(|e| anyhow!("could not start {program}: {e}"))?;
    let stop = Arc::new(AtomicBool::new(false));
    let steps = Arc::new(Mutex::new(Vec::new()));
    let error = Arc::new(Mutex::new(None));
    let ctx = Ctx {
        program: program.to_string(),
        root_pid: child.id(),
        shots: shots_dir.to_path_buf(),
        steps: steps.clone(),
    };
    let (stop_t, error_t) = (stop.clone(), error.clone());
    let handle = std::thread::Builder::new().name("bf-teach-record".into()).spawn(move || {
        if let Err(e) = record_loop(ctx, &stop_t) {
            *error_t.lock().unwrap() = Some(format!("{e:#}"));
        }
    })?;
    Ok(Recording { stop, steps, error, handle: Some(handle) })
}

struct Ctx {
    program: String,
    root_pid: u32,
    shots: PathBuf,
    steps: Arc<Mutex<Vec<RecordedStep>>>,
}

/// A field being typed into, flushed as one SetValue when the person moves on.
struct Typing {
    element: Box<dyn crate::platform::Element>,
    target: ElementRef,
    buffer: String,
    secret: bool,
    at: (i32, i32),
}

fn record_loop(ctx: Ctx, stop: &AtomicBool) -> Result<()> {
    let mut platform = crate::platform::select();
    platform.subscribe()?;
    let mut scope = Scope::new(ctx.root_pid, &ctx.program);
    let started = Instant::now();
    let mut typing: Option<Typing> = None;
    let mut push = |action: Action, at: (i32, i32), platform: &mut dyn Platform| {
        let id = new_id();
        let screenshot = platform.capture(at.0, at.1, &ctx.shots, &id).ok();
        ctx.steps.lock().unwrap().push(RecordedStep {
            id,
            at_ms: started.elapsed().as_millis() as u64,
            action,
            screenshot,
        });
    };

    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        let Some(event) = platform.poll(POLL) else { continue };
        match event {
            Observed::Click { x, y } => {
                let Some(el) = platform.element_at(x, y) else { continue };
                let Some(pid) = el.pid().filter(|p| scope.contains(*p)) else { continue };
                if let Some(t) = typing.take() {
                    flush(t, &mut push, &mut platform);
                }
                let target = el.describe(Some((x, y)), &scope.name(pid));
                push(Action::Click { target }, (x, y), &mut platform);
            }
            Observed::Key { key, text, ctrl, alt, shift, meta } => {
                let Some(el) = platform.focused_element() else { continue };
                let Some(pid) = el.pid().filter(|p| scope.contains(*p)) else { continue };
                // Focus moved to another field: the previous one is done.
                let moved = typing
                    .as_ref()
                    .is_some_and(|t| !platform.same_element(t.element.as_ref(), el.as_ref()));
                if moved {
                    flush(typing.take().expect("checked"), &mut push, &mut platform);
                }
                let secret = el.is_password();
                let printable = text.as_deref().filter(|s| !s.is_empty() && s.chars().all(|c| !c.is_control()));
                if let (Some(s), false, false, false) = (printable, ctrl, alt, meta) {
                    let t = typing.get_or_insert_with(|| {
                        let at = el.centre();
                        let target = platform.describe_element(el.as_ref(), None, &scope.name(pid));
                        Typing { target, element: el, buffer: String::new(), secret, at }
                    });
                    // A password's characters never reach memory, let alone disk.
                    if !t.secret {
                        t.buffer.push_str(s);
                    }
                    continue;
                }
                if key == KeyName::Backspace {
                    if let Some(t) = typing.as_mut() {
                        t.buffer.pop();
                    }
                    continue;
                }
                let Some(label) = key_label(key, ctrl, alt, shift, meta) else { continue };
                if let Some(t) = typing.take() {
                    flush(t, &mut push, &mut platform);
                }
                let at = el.centre();
                let target = el.describe(None, &scope.name(pid));
                push(Action::Keys { keys: label, target: Some(target) }, at, &mut platform);
            }
        }
    }
    if let Some(t) = typing.take() {
        flush(t, &mut push, &mut platform);
    }
    Ok(())
}

fn flush(t: Typing, push: &mut impl FnMut(Action, (i32, i32), &mut dyn Platform), platform: &mut dyn Platform) {
    let value = if t.secret {
        String::new()
    } else {
        // The field's own value is the truth (autocomplete, a corrected typo, a mask);
        // what was typed is the fallback for fields that expose none.
        t.element.value().filter(|v| !v.is_empty()).unwrap_or(t.buffer)
    };
    push(Action::SetValue { target: t.target, value, secret: t.secret }, t.at, platform);
}

/// `Enter`, `Ctrl+S`, `Alt+F4` — the keys that are actions rather than text. None for a
/// key that is neither (a lone modifier, a key with no name).
pub fn key_label(key: KeyName, ctrl: bool, alt: bool, shift: bool, meta: bool) -> Option<String> {
    let name = match key {
        KeyName::Return => "Enter".to_string(),
        KeyName::Tab => "Tab".into(),
        KeyName::Escape => "Escape".into(),
        KeyName::Delete => "Delete".into(),
        KeyName::Up => "Up".into(),
        KeyName::Down => "Down".into(),
        KeyName::Left => "Left".into(),
        KeyName::Right => "Right".into(),
        KeyName::PageUp => "PageUp".into(),
        KeyName::PageDown => "PageDown".into(),
        KeyName::Home => "Home".into(),
        KeyName::End => "End".into(),
        KeyName::Space => "Space".into(),
        KeyName::Char(c) if c.is_ascii_digit() => c.to_string(),
        KeyName::Char(c) if c.is_ascii_alphabetic() => c.to_ascii_uppercase().to_string(),
        KeyName::Function(n) => format!("F{n}"),
        _ => return None,
    };
    let is_plain_char = name.chars().count() == 1 || name == "Space";
    // A plain letter without a shortcut modifier is typing, handled as text.
    if is_plain_char && !ctrl && !alt && !meta {
        return None;
    }
    let mut parts = Vec::new();
    if ctrl {
        parts.push("Ctrl");
    }
    if alt {
        parts.push("Alt");
    }
    if shift {
        parts.push("Shift");
    }
    if meta {
        parts.push("Win");
    }
    parts.push(&name);
    Some(parts.join("+"))
}

// ----------------------------------------------------------------- replay

/// The person took control back (Esc) or pressed Stop.
struct Stop;

struct Runner<'a> {
    /// Secret parameter names: their values are never read back to verify.
    secrets: HashSet<String>,
    platform: Box<dyn Platform>,
    scope: Scope,
    hooks: &'a mut dyn RunHooks,
}

pub fn run_skill(skill: &Skill, values: &BTreeMap<String, String>, hooks: &mut dyn RunHooks) -> Result<RunStatus> {
    // Every parameter resolved up front: the value given, else the demonstration's. A
    // secret has no default, so a missing one stops the run before anything moves.
    let mut resolved = BTreeMap::new();
    for p in &skill.params {
        match values.get(&p.name).cloned().or_else(|| p.default.clone()) {
            Some(v) => {
                resolved.insert(p.name.clone(), v);
            }
            None => return Err(anyhow!("a value for “{}” is needed", p.label)),
        }
    }
    let values = &resolved;
    let child = std::process::Command::new(&skill.program)
        .args(&skill.args)
        .spawn()
        .map_err(|e| anyhow!("could not start {}: {e}", skill.program))?;
    let mut platform = crate::platform::select();
    // Listen for Esc before anything moves, so the person can stop the very first step.
    platform.subscribe()?;
    let mut r = Runner {
        secrets: skill.params.iter().filter(|p| p.secret).map(|p| p.name.clone()).collect(),
        platform,
        scope: Scope::new(child.id(), &skill.program),
        hooks,
    };
    if r.wait_for(WINDOW_WAIT, |r| !r.platform.windows_of(&r.scope.pids()).is_empty()).is_err() {
        return Ok(RunStatus::Stopped);
    }
    for (idx, step) in skill.steps.iter().enumerate() {
        if r.interrupted().is_some() {
            return Ok(RunStatus::Stopped);
        }
        // The person closed the program mid-run: say so, instead of searching for
        // windows that no longer exist until a step times out.
        if !r.scope.root_alive() {
            return Err(anyhow!("{} was closed during the run", skill.program));
        }
        r.hooks.step_started(idx);
        if step.requires_approval {
            if !r.hooks.approve(idx) {
                r.hooks.step_finished(idx, StepOutcome::Denied, None);
                return Ok(RunStatus::Denied);
            }
            r.hooks.step_finished(idx, StepOutcome::Approved, None);
        }
        let (outcome, detail) = match r.perform(&step.action, values) {
            Ok(o) => o,
            Err(StepError::Stop(_)) => return Ok(RunStatus::Stopped),
            Err(StepError::Failed(msg)) => {
                r.hooks.step_finished(idx, StepOutcome::Failed, Some(msg.clone()));
                return Err(anyhow!(msg));
            }
        };
        r.hooks.step_finished(idx, outcome, detail);
        std::thread::sleep(SETTLE);
    }
    Ok(RunStatus::Succeeded)
}

enum StepError {
    Stop(Stop),
    Failed(String),
}

impl From<Stop> for StepError {
    fn from(s: Stop) -> Self {
        StepError::Stop(s)
    }
}

type StepResult = std::result::Result<(StepOutcome, Option<String>), StepError>;

impl Runner<'_> {
    /// Esc from the person (not our own injected input) or Stop from the window.
    fn interrupted(&mut self) -> Option<Stop> {
        while let Some(ev) = self.platform.poll(Duration::ZERO) {
            if matches!(ev, Observed::Key { key, .. } if key.is_escape()) {
                return Some(Stop);
            }
        }
        self.hooks.should_stop().then_some(Stop)
    }

    /// Poll `ready` until it holds or `limit` passes, staying interruptible.
    fn wait_for(&mut self, limit: Duration, mut ready: impl FnMut(&mut Self) -> bool) -> std::result::Result<bool, Stop> {
        let deadline = Instant::now() + limit;
        loop {
            if let Some(s) = self.interrupted() {
                return Err(s);
            }
            if ready(self) {
                return Ok(true);
            }
            if Instant::now() >= deadline {
                return Ok(false);
            }
            std::thread::sleep(POLL);
        }
    }

    /// The window a step was recorded in: same title, else one containing it, else the
    /// program's first window.
    fn window_for(&mut self, target: &ElementRef) -> Option<Box<dyn crate::platform::Window>> {
        let pids = self.scope.pids();
        let mut wins = self.platform.windows_of(&pids);
        let pick = wins
            .iter()
            .position(|w| w.title() == target.window_title)
            .or_else(|| wins.iter().position(|w| !target.window_title.is_empty() && w.title().contains(&target.window_title)))
            .or(if wins.is_empty() { None } else { Some(0) })?;
        Some(wins.swap_remove(pick))
    }

    /// Find the step's element, retrying while the UI settles. Returns the window too,
    /// for the positional fallback.
    fn locate(
        &mut self,
        target: &ElementRef,
    ) -> std::result::Result<(Option<Box<dyn crate::platform::Window>>, Option<Box<dyn crate::platform::Element>>), Stop> {
        let mut found = (None, None);
        self.wait_for(ELEMENT_WAIT, |r| {
            let win = r.window_for(target);
            let el = win.as_ref().and_then(|w| r.platform.find(w.as_ref(), target));
            let done = el.is_some();
            found = (win, el);
            done
        })?;
        Ok(found)
    }

    fn perform(&mut self, action: &SkillAction, values: &BTreeMap<String, String>) -> StepResult {
        match action {
            SkillAction::Click { target } => {
                let (win, el) = self.locate(target)?;
                match (el, win) {
                    (Some(el), _) => {
                        if self.platform.invoke(el.as_ref()) || self.platform.click_element(el.as_ref()) {
                            Ok((StepOutcome::Ok, None))
                        } else {
                            Err(StepError::Failed(format!("could not click “{}”", target.label())))
                        }
                    }
                    (None, Some(w)) => {
                        self.click_at(w.as_ref(), target)?;
                        Ok((StepOutcome::Fallback, Some(format!("“{}” not found; clicked its recorded position", target.label()))))
                    }
                    (None, None) => Err(StepError::Failed(format!("the “{}” window is not open", target.window_title))),
                }
            }
            SkillAction::SetValue { target, value } => {
                let (text, secret) = match value {
                    SkillValue::Literal { text } => (text.clone(), false),
                    SkillValue::Param { name } => (values.get(name).cloned().unwrap_or_default(), self.secrets.contains(name)),
                };
                let (win, el) = self.locate(target)?;
                match (el, win) {
                    (Some(el), _) => {
                        if self.platform.set_value(el.as_ref(), &text) {
                            // Password fields do not read back; every other value is checked.
                            if secret || el.value().is_some_and(|v| v == text) {
                                return Ok((StepOutcome::Ok, None));
                            }
                        }
                        let _ = self.platform.focus(el.as_ref());
                        self.replace_text(&text)?;
                        Ok((StepOutcome::Ok, Some("typed".into())))
                    }
                    (None, Some(w)) => {
                        self.click_at(w.as_ref(), target)?;
                        self.replace_text(&text)?;
                        Ok((StepOutcome::Fallback, Some(format!("“{}” not found; typed at its recorded position", target.label()))))
                    }
                    (None, None) => Err(StepError::Failed(format!("the “{}” window is not open", target.window_title))),
                }
            }
            SkillAction::Keys { keys, target } => {
                if let Some(t) = target {
                    if let (_, Some(el)) = self.locate(t)? {
                        let _ = self.platform.focus(el.as_ref());
                    }
                }
                self.platform.send_keys(keys).map_err(|e| StepError::Failed(format!("{e:#}")))?;
                Ok((StepOutcome::Ok, None))
            }
        }
    }

    fn click_at(&mut self, w: &dyn crate::platform::Window, target: &ElementRef) -> std::result::Result<(), StepError> {
        let (wx, wy) = w.origin();
        self.platform.click_at(wx + target.rel_x, wy + target.rel_y).map_err(|e| StepError::Failed(format!("{e:#}")))
    }

    fn replace_text(&mut self, text: &str) -> std::result::Result<(), StepError> {
        self.platform.replace_text(text).map_err(|e| StepError::Failed(format!("{e:#}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortcut_labels_are_platform_neutral() {
        assert_eq!(key_label(KeyName::Char('s'), true, false, false, false).as_deref(), Some("Ctrl+S"));
        assert_eq!(key_label(KeyName::Function(4), false, true, false, false).as_deref(), Some("Alt+F4"));
        assert_eq!(key_label(KeyName::Return, false, false, false, false).as_deref(), Some("Enter"));
        // A plain letter is typing, not a step of its own.
        assert_eq!(key_label(KeyName::Char('a'), false, false, false, false), None);
        assert_eq!(key_label(KeyName::Backspace, false, false, false, false), None);
        // The Windows logo key and the macOS Command key share one label.
        assert_eq!(key_label(KeyName::Char('q'), false, false, false, true).as_deref(), Some("Win+Q"));
    }
}
