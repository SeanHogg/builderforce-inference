//! bf-teach — the hands and eyes of Self-Directed Agents.
//!
//! **Recording**: launch a program and capture what the person does in *its* windows as
//! semantic steps (the accessibility element acted on, the value set, a screenshot) — not a
//! raw keystroke log. Password fields are redacted at capture.
//!
//! **Replay**: run a Train Once skill — find each element by accessibility identity, fall
//! back to its recorded position, verify values, pause for approval where the skill says
//! so, and stop the moment the person presses Esc.
//!
//! **Vault**: secret parameters live in the OS credential store and are used at replay
//! without ever being shown.
//!
//! The OS-specific work — observing and injecting input, describing and driving UI
//! elements, screenshots — sits behind [`platform::Platform`]; the loops in [`engine`] are
//! written once against it. Windows (UI Automation) and macOS (Accessibility) are the
//! supported platforms; everywhere else the crate builds and reports "not supported" at
//! the moment recording or replay is tried.

pub mod engine;
pub mod model;
pub mod platform;
pub mod procs;
pub mod shot;

pub(crate) mod input;

pub use engine::{run_skill, start_recording, Recording};
pub use model::StepOutcome;

#[cfg(windows)]
pub(crate) mod win;
#[cfg(target_os = "macos")]
pub(crate) mod mac;

/// What a run tells its owner as it goes, and asks it.
pub trait RunHooks {
    fn step_started(&mut self, idx: usize);
    fn step_finished(&mut self, idx: usize, outcome: StepOutcome, detail: Option<String>);
    /// Block until the person approves (true) or declines (false) step `idx`.
    fn approve(&mut self, idx: usize) -> bool;
    /// The person pressed Stop in the window.
    fn should_stop(&self) -> bool;
}

/// The key a person presses to take control back from a running skill.
pub const TAKEOVER_KEY: &str = "Esc";

/// Whether recording and replay work on this OS.
pub fn supported() -> bool {
    matches!(platform::selected_name(), "windows" | "macos")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Windows-only gate is gone: every supported platform resolves to a real
    /// implementation through the same entry points, and no caller branches on the OS.
    #[test]
    fn platform_selection_is_not_windows_only() {
        assert_ne!(platform::selected_name(), "", "a platform is always selected");
        #[cfg(target_os = "macos")]
        assert_eq!(platform::selected_name(), "macos", "macOS resolves to the macOS platform");
        #[cfg(windows)]
        assert_eq!(platform::selected_name(), "windows", "Windows resolves to the Windows platform");
        // macOS is a supported platform — the old `cfg!(windows)` gate would fail here.
        assert!(supported() || platform::selected_name() == "unsupported");
    }

    /// The platform selection returns an object that answers the platform's own name.
    #[test]
    fn select_returns_the_platform_for_this_os() {
        let p = platform::select();
        assert_eq!(p.name(), platform::selected_name());
    }
}
