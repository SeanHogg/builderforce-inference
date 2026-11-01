//! The platform boundary for Self-Directed Agents.
//!
//! Recording and replay need four things from the OS, and nothing else: a way to see what
//! the person did (input observation), a way to describe and find the UI element they acted
//! on (accessibility), a way to do it again (input injection), and a screenshot of the
//! moment. Each OS supplies those differently — Windows through UI Automation and
//! `SendInput`, macOS through the Accessibility API and `CGEvent` — so the platform-specific
//! code lives behind [`Platform`] and the recording/replay loops in [`crate::engine`] are
//! written once against it.
//!
//! [`select`] is the only place that knows which OS this binary runs on. Callers (and
//! tests) never branch on the OS themselves.

use crate::model::ElementRef;
use anyhow::Result;
use std::any::Any;
use std::path::Path;

/// One thing the person did, already filtered to the launched program.
pub enum Observed {
    /// A left click, in screen coordinates.
    Click { x: i32, y: i32 },
    /// A key press. `text` is the character it produced, when it produced one.
    Key { key: KeyName, text: Option<String>, ctrl: bool, alt: bool, shift: bool, meta: bool },
}

/// The keys recording treats as actions in their own right (rather than text).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyName {
    Return,
    Tab,
    Escape,
    Delete,
    Backspace,
    Up,
    Down,
    Left,
    Right,
    PageUp,
    PageDown,
    Home,
    End,
    Space,
    /// A letter or digit key (`a`, `1`).
    Char(char),
    /// A function key, 1..=12.
    Function(u8),
    /// Anything else (a lone modifier, a media key): recording ignores it.
    Other,
}

impl KeyName {
    pub fn is_escape(self) -> bool {
        matches!(self, KeyName::Escape)
    }
}

/// A UI element the platform can act on. Opaque to the engine: only the platform that
/// created it can drive it, and [`Platform`] is how the engine asks.
pub trait Element: Send + Any {
    /// A password field: recorded and replayed, never read back.
    fn is_password(&self) -> bool;
    /// The field's own value, when it exposes one.
    fn value(&self) -> Option<String>;
    /// A point inside the element, for the screenshot.
    fn centre(&self) -> (i32, i32);
    /// The process that owns it.
    fn pid(&self) -> Option<u32>;
    /// Activate through its accessibility action (no pointer). False when there is none.
    fn invoke(&self) -> bool;
    /// Click with the pointer.
    fn click(&self) -> bool;
    /// Set its value through its accessibility attribute. False when it is not writable
    /// that way, so the caller falls back to typing.
    fn set_value(&self, value: &str) -> bool;
    /// Move keyboard focus to it.
    fn focus(&self) -> bool;
}

/// A top-level window of the launched program.
pub trait Window: Send + Any {
    fn title(&self) -> &str;
    /// Screen coordinates of the window's top-left, so a recorded relative point becomes
    /// an absolute one again.
    fn origin(&self) -> (i32, i32);
}

/// What recording and replay ask of the operating system.
///
/// Every method is synchronous and blocking: both loops already run on their own thread,
/// and the accessibility APIs they call are thread-affine on every platform this crate
/// supports, so the platform object is created on that thread and never moved off it.
pub trait Platform: Send {
    /// Begin observing the person's input. Events arrive through [`Self::poll`], which
    /// blocks up to `timeout` and returns `None` when nothing happened.
    fn subscribe(&mut self) -> Result<()>;
    fn poll(&mut self, timeout: std::time::Duration) -> Option<Observed>;

    /// The element under a screen point / the element with keyboard focus.
    fn element_at(&mut self, x: i32, y: i32) -> Option<Box<dyn Element>>;
    fn focused_element(&mut self) -> Option<Box<dyn Element>>;

    /// Describe an element for a recorded step. `click` is where it was clicked (screen
    /// coordinates); without one its centre is recorded.
    fn describe_element(&mut self, element: &dyn Element, click: Option<(i32, i32)>, process_name: &str) -> ElementRef;
    /// The same element as `other` (focus moved, or not).
    fn same_element(&mut self, a: &dyn Element, b: &dyn Element) -> bool;

    /// Top-level windows owned by any of `pids`, frontmost first.
    fn windows_of(&mut self, pids: &[u32]) -> Vec<Box<dyn Window>>;
    /// The recorded element, found again inside `window` by accessibility identity.
    fn find(&mut self, window: &dyn Window, target: &ElementRef) -> Option<Box<dyn Element>>;

    /// Run `f` while the observer ignores input, so a skill's own injected keystrokes are
    /// never reported as the person's (an injected Esc must not stop the run).
    fn injecting(&mut self, f: &mut dyn FnMut());

    fn click_at(&mut self, x: i32, y: i32) -> Result<()>;
    /// Select the focused field's contents and type `text` over them (empty clears it).
    fn replace_text(&mut self, text: &str) -> Result<()>;
    /// Send a shortcut as recorded: `Ctrl+S`, `Enter`, `Alt+F4`.
    fn send_keys(&mut self, combo: &str) -> Result<()>;

    /// Capture the monitor containing (`x`, `y`) into `dir/<name>.jpg`.
    fn capture(&mut self, x: i32, y: i32, dir: &Path, name: &str) -> Result<String>;
}

/// The platform this binary runs on.
///
/// macOS gets a real implementation (Accessibility + CGEvent); Windows keeps its UI
/// Automation implementation; anything else gets [`Unsupported`], which reports the one
/// honest error rather than pretending.
pub fn select() -> Box<dyn Platform> {
    #[cfg(windows)]
    {
        Box::new(crate::win::WinPlatform::new())
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(crate::mac::MacPlatform::new())
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        Box::new(Unsupported)
    }
}

/// The name [`select`] resolves to here — what a test asserts, so the macOS path is
/// proven to be selected without a Mac to run it on.
pub fn selected_name() -> &'static str {
    #[cfg(windows)]
    {
        "windows"
    }
    #[cfg(target_os = "macos")]
    {
        "macos"
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        "unsupported"
    }
}

/// A platform with no desktop automation. Recording and replay fail with one clear error
/// the moment they try to observe input; nothing else is ever reached.
pub struct Unsupported;

impl Platform for Unsupported {
    fn subscribe(&mut self) -> Result<()> {
        anyhow::bail!("recording and replay need Windows or macOS; this platform has no desktop automation yet")
    }
    fn poll(&mut self, timeout: std::time::Duration) -> Option<Observed> {
        std::thread::sleep(timeout);
        None
    }
    fn element_at(&mut self, _: i32, _: i32) -> Option<Box<dyn Element>> {
        None
    }
    fn focused_element(&mut self) -> Option<Box<dyn Element>> {
        None
    }
    fn describe_element(&mut self, _: &dyn Element, _: Option<(i32, i32)>, _: &str) -> ElementRef {
        ElementRef::default()
    }
    fn same_element(&mut self, _: &dyn Element, _: &dyn Element) -> bool {
        false
    }
    fn windows_of(&mut self, _: &[u32]) -> Vec<Box<dyn Window>> {
        Vec::new()
    }
    fn find(&mut self, _: &dyn Window, _: &ElementRef) -> Option<Box<dyn Element>> {
        None
    }
    fn injecting(&mut self, f: &mut dyn FnMut()) {
        f()
    }
    fn click_at(&mut self, _: i32, _: i32) -> Result<()> {
        Ok(())
    }
    fn replace_text(&mut self, _: &str) -> Result<()> {
        Ok(())
    }
    fn send_keys(&mut self, _: &str) -> Result<()> {
        Ok(())
    }
    fn capture(&mut self, _: i32, _: i32, _: &Path, _: &str) -> Result<String> {
        anyhow::bail!("screenshots need Windows or macOS")
    }
}
