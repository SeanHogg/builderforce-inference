//! macOS: the [`Platform`] implementation built on the Accessibility API (element
//! identity) and `CGEvent` (input injection). Input observation is the shared
//! [`crate::input`] hook (`rdev` supports macOS natively) and screenshots are the shared
//! [`crate::shot`] (`xcap` supports macOS natively).
//!
//! Everything here needs Accessibility permission ("System Settings → Privacy & Security →
//! Accessibility" for Synapse). The first API call returns a clear error until the person
//! grants it.

use crate::input;
use crate::platform::{Element, KeyName, Observed, Platform, Window};
use crate::{procs, shot};
use accessibility::{AXAttribute, AXUIElement, AXUIElementActions, AXUIElementAttributes};
use accessibility_sys::{kAXTrustedCheckOptionPrompt, AXIsProcessTrustedWithOptions};
use anyhow::{anyhow, bail, Result};
use core_foundation::base::TCFType;
use core_foundation::boolean::CFBoolean;
use core_foundation::dictionary::CFDictionary;
use core_foundation::string::CFString;
use core_graphics::event::{CGEvent, CGEventTapLocation, CGEventType, CGMouseButton};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use core_graphics::geometry::CGPoint;
use std::any::Any;
use std::path::Path;
use std::time::Duration;

/// The maximum attribute string length, so a walk does not copy megabytes from one
/// broken node.
const MAX_ATTR_LEN: usize = 4096;
/// How far a search walks a window's tree before giving up — large apps have tens of
/// thousands of nodes, and a replay step should fail over to the recorded position well
/// before a full walk.
const MAX_NODES: usize = 6000;

pub struct MacPlatform;

impl MacPlatform {
    pub fn new() -> Self {
        Self
    }
}

/// Whether this process may drive the UI; prompts for permission the first time.
fn ensure_trusted() -> Result<()> {
    let options = CFDictionary::from_CFType_pairs(&[(
        unsafe { CFString::wrap_under_get_rule(kAXTrustedCheckOptionPrompt) },
        CFBoolean::true_value(),
    )]);
    // SAFETY: `options` is a valid CFDictionary holding the one well-known boolean key.
    let trusted = unsafe { AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef()) };
    if trusted {
        Ok(())
    } else {
        bail!("Synapse needs Accessibility permission (System Settings → Privacy & Security → Accessibility); grant it and try again")
    }
}

/// macOS virtual key codes (the ones replay needs; from <Carbon/Carbon.h>).
#[derive(Clone, Copy)]
struct VKey(i16);
#[allow(non_upper_case_globals)]
impl VKey {
    const Return: VKey = VKey(0x24);
    const Tab: VKey = VKey(0x30);
    const Delete: VKey = VKey(0x33); // forward delete
    const Escape: VKey = VKey(0x35);
    const Left: VKey = VKey(0x7B);
    const Right: VKey = VKey(0x7C);
    const Down: VKey = VKey(0x7D);
    const Up: VKey = VKey(0x7E);
    const Shift: VKey = VKey(0x38);
    const Control: VKey = VKey(0x3B);
    const Option: VKey = VKey(0x3A);
    const Command: VKey = VKey(0x37);
    const Space: VKey = VKey(0x31);
    const PageUp: VKey = VKey(0x74);
    const PageDown: VKey = VKey(0x79);
    const Home: VKey = VKey(0x73);
    const End: VKey = VKey(0x77);
}

/// One key press or release through the HID event tap.
fn key(code: VKey, down: bool) -> Result<()> {
    let src = CGEventSource::new(CGEventSourceStateID::CombinedSessionState).ok_or_else(|| anyhow!("no CGEventSource"))?;
    let ev = CGEvent::new_keyboard_event(&src, code.0, down).ok_or_else(|| anyhow!("could not create a key event"))?;
    ev.post(CGEventTapLocation::AnnotatedSession);
    Ok(())
}

/// Type one character through a Unicode keyboard event.
fn unicode(c: char) -> Result<()> {
    let src = CGEventSource::new(CGEventSourceStateID::CombinedSessionState).ok_or_else(|| anyhow!("no CGEventSource"))?;
    let ev = CGEvent::new_keyboard_event(&src, 0, true).ok_or_else(|| anyhow!("could not create a key event"))?;
    ev.set_string(&c.to_string());
    ev.post(CGEventTapLocation::AnnotatedSession);
    Ok(())
}

/// A left click at a screen point, while the input hook ignores injected input.
pub(crate) fn click_at(x: i32, y: i32) -> Result<()> {
    input::injecting(|| {
        let src = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
            .ok_or_else(|| anyhow!("no CGEventSource"))?;
        let point = CGPoint::new(f64::from(x), f64::from(y));
        for t in [CGEventType::LeftMouseDown, CGEventType::LeftMouseUp] {
            let ev = CGEvent::new_mouse_event(&src, t, point, CGMouseButton::Left)
                .ok_or_else(|| anyhow!("could not create a mouse event"))?;
            ev.post(CGEventTapLocation::AnnotatedSession);
        }
        Ok(())
    })
}

/// A named key (`Enter`, `Tab`, arrows, F-keys, single letters).
fn named_key(name: &str) -> Result<()> {
    let tap = |code: VKey| key(code, true).and_then(|_| key(code, false));
    match name {
        "Enter" => tap(VKey::Return),
        "Tab" => tap(VKey::Tab),
        "Escape" => tap(VKey::Escape),
        "Delete" => tap(VKey::Delete),
        "Up" => tap(VKey::Up),
        "Down" => tap(VKey::Down),
        "Left" => tap(VKey::Left),
        "Right" => tap(VKey::Right),
        "PageUp" => tap(VKey::PageUp),
        "PageDown" => tap(VKey::PageDown),
        "Home" => tap(VKey::Home),
        "End" => tap(VKey::End),
        "Space" => tap(VKey::Space),
        f if f.starts_with('F') && f.len() > 1 => {
            let n: i16 = f[1..].parse().map_err(|_| anyhow!("unknown key “{name}”"))?;
            if !(1..=12).contains(&n) {
                return Err(anyhow!("unknown key “{name}”"));
            }
            tap(VKey(0x7A + (n - 1))) // F1..F12
        }
        c if c.chars().count() == 1 => unicode(c.chars().next().unwrap()),
        _ => Err(anyhow!("unknown key “{name}”")),
    }
}

impl Platform for MacPlatform {
    fn name(&self) -> &'static str {
        "macos"
    }

    fn subscribe(&mut self) -> Result<()> {
        ensure_trusted()?;
        input::subscribe();
        Ok(())
    }

    fn poll(&mut self, timeout: Duration) -> Option<Observed> {
        match input::next(timeout) {
            Ok(input::Input::Click { x, y }) => Some(Observed::Click { x, y }),
            Ok(input::Input::Key { key, text, ctrl, alt, shift, meta }) => Some(Observed::Key {
                key: map_key(key),
                text,
                ctrl,
                alt,
                shift,
                meta,
            }),
            Err(_) => None,
        }
    }

    fn element_at(&mut self, x: i32, y: i32) -> Option<Box<dyn Element>> {
        let sys = AXUIElement::system_wide();
        sys.element_at_position(f64::from(x), f64::from(y)).map(|el| Box::new(MacEl(el)) as Box<dyn Element>)
    }

    fn focused_element(&mut self) -> Option<Box<dyn Element>> {
        let sys = AXUIElement::system_wide();
        sys.focused_uielement().ok().flatten().map(|el| Box::new(MacEl(el)) as Box<dyn Element>)
    }

    fn describe_element(&mut self, element: &dyn Element, click: Option<(i32, i32)>, process_name: &str) -> crate::model::ElementRef {
        let Some(el) = element.as_any().downcast_ref::<MacEl>() else { return crate::model::ElementRef::default() };
        let (win_origin, win_title) = window_of(&el.0).unwrap_or(((0, 0), String::new()));
        let win_pid = pid_of(&el.0).and_then(procs::name_of);
        let (px, py) = click.unwrap_or_else(|| centre_of(&el.0));
        crate::model::ElementRef {
            name: attr_string(&el.0, "AXTitle").unwrap_or_default(),
            automation_id: attr_string(&el.0, "AXIdentifier").unwrap_or_default(),
            control_type: attr_string(&el.0, "AXRole").unwrap_or_default(),
            class_name: String::new(),
            window_title: win_title,
            process_name: if process_name.is_empty() { win_pid.unwrap_or_default() } else { process_name.to_string() },
            rel_x: px - win_origin.0,
            rel_y: py - win_origin.1,
        }
    }

    fn same_element(&mut self, a: &dyn Element, b: &dyn Element) -> bool {
        let (Some(a), Some(b)) = (a.as_any().downcast_ref::<MacEl>(), b.as_any().downcast_ref::<MacEl>()) else { return false };
        // An AXUIElement's opaque ref is stable for the same underlying element, and
        // equal refs compare equal.
        a.0 == b.0
    }

    fn windows_of(&mut self, pids: &[u32]) -> Vec<Box<dyn Window>> {
        pids.iter()
            .filter_map(|&pid| {
                let app = AXUIElement::application(pid as i32);
                app.windows().ok().map(|ws| (pid, ws))
            })
            .flat_map(|(pid, ws)| {
                ws.into_iter().filter_map(move |el| {
                    let title = attr_string(&el, "AXTitle").unwrap_or_default();
                    let (x, y) = frame_of(&el)?.0;
                    Some(Box::new(MacWindow { el, pid, title, origin: (x, y) }) as Box<dyn Window>)
                })
            })
            .collect()
    }

    fn find(&mut self, window: &dyn Window, target: &crate::model::ElementRef) -> Option<Box<dyn Element>> {
        let win = window.as_any().downcast_ref::<MacWindow>()?;
        find_in(&win.el, target).map(|el| Box::new(MacEl(el)) as Box<dyn Element>)
    }

    fn injecting(&mut self, f: &mut dyn FnMut()) {
        input::injecting(f)
    }

    fn click_at(&mut self, x: i32, y: i32) -> Result<()> {
        click_at(x, y)
    }

    fn replace_text(&mut self, text: &str) -> Result<()> {
        // Cmd+A selects the focused field; Delete clears it; then the text.
        self.send_keys("Meta+A")?;
        key(VKey::Delete, true)?;
        key(VKey::Delete, false)?;
        for c in text.chars() {
            unicode(c)?;
        }
        Ok(())
    }

    fn send_keys(&mut self, combo: &str) -> Result<()> {
        let mut parts: Vec<&str> = combo.split('+').collect();
        let last = parts.pop().ok_or_else(|| anyhow!("unknown key “{combo}”"))?;
        let mods: Vec<VKey> = parts
            .iter()
            .map(|m| match *m {
                "Ctrl" => Ok(VKey::Control),
                "Alt" => Ok(VKey::Option),
                "Shift" => Ok(VKey::Shift),
                // Recorded as "Win" (the Windows logo key); on macOS the equivalent
                // modifier is Command.
                "Win" => Ok(VKey::Command),
                _ => Err(anyhow!("unknown modifier “{m}”")),
            })
            .collect::<Result<Vec<_>>>()?;
        self.injecting(&mut || {
            for m in &mods {
                key(*m, true)?;
            }
            let r = named_key(last);
            for m in mods.iter().rev() {
                key(*m, false)?;
            }
            r
        })
    }

    fn capture(&mut self, x: i32, y: i32, dir: &Path, name: &str) -> Result<String> {
        shot::capture(x, y, dir, name)
    }
}

fn map_key(key: rdev::Key) -> KeyName {
    use rdev::Key;
    match key {
        Key::Return | Key::KpReturn => KeyName::Return,
        Key::Tab => KeyName::Tab,
        Key::Escape => KeyName::Escape,
        Key::Delete => KeyName::Delete,
        Key::Backspace => KeyName::Backspace,
        Key::UpArrow => KeyName::Up,
        Key::DownArrow => KeyName::Down,
        Key::LeftArrow => KeyName::Left,
        Key::RightArrow => KeyName::Right,
        Key::PageUp => KeyName::PageUp,
        Key::PageDown => KeyName::PageDown,
        Key::Home => KeyName::Home,
        Key::End => KeyName::End,
        Key::Space => KeyName::Space,
        other => {
            let dbg = format!("{other:?}");
            if let Some(letter) = dbg.strip_prefix("Key").filter(|l| l.len() == 1) {
                KeyName::Char(letter.chars().next().unwrap().to_ascii_lowercase())
            } else if let Some(digit) = dbg.strip_prefix("Num").filter(|d| d.len() == 1) {
                KeyName::Char(digit.chars().next().unwrap())
            } else if dbg.starts_with('F') && dbg[1..].parse::<u8>().is_ok() {
                KeyName::Function(dbg[1..].parse().unwrap())
            } else {
                KeyName::Other
            }
        }
    }
}

fn attr_string(el: &AXUIElement, name: &str) -> Option<String> {
    el.attribute(&AXAttribute::new(&name.into()))
        .ok()
        .and_then(|v| v.to_string())
        .filter(|s| s.len() <= MAX_ATTR_LEN)
}

fn pid_of(el: &AXUIElement) -> Option<u32> {
    el.pid().ok().map(|p| p as u32)
}

/// ((x, y), (w, h)) of an element's frame.
fn frame_of(el: &AXUIElement) -> Option<((i32, i32), (i32, i32))> {
    let any = el.attribute(&AXAttribute::new(&"AXFrame".into())).ok()?;
    let v = any.cast::<accessibility_sys::AXValueRef>().ok()?;
    let mut rect = accessibility_sys::CGRect {
        origin: accessibility_sys::CGPoint { x: 0.0, y: 0.0 },
        size: accessibility_sys::CGSize { width: 0.0, height: 0.0 },
    };
    // SAFETY: `v` came from `AXUIElementCopyAttributeValue` for the `AXFrame` attribute,
    // so it holds a CGRect.
    unsafe {
        if accessibility_sys::AXValueGetType(v) == accessibility_sys::AXValueType::CGRect
            && accessibility_sys::AXValueGetValue(v, &mut rect)
        {
            Some(((rect.origin.x as i32, rect.origin.y as i32), (rect.size.width as i32, rect.size.height as i32)))
        } else {
            None
        }
    }
}

fn centre_of(el: &AXUIElement) -> (i32, i32) {
    frame_of(el).map(|(o, s)| (o.0 + s.0 / 2, o.1 + s.1 / 2)).unwrap_or((0, 0))
}

/// The top-level window containing `el`, as (origin, title).
fn window_of(el: &AXUIElement) -> Option<((i32, i32), String)> {
    let mut cur = el.clone();
    for _ in 0..64 {
        if let Some(w) = try_window(&cur) {
            return Some(w);
        }
        cur = cur.parent().ok()??;
    }
    None
}

fn try_window(el: &AXUIElement) -> Option<((i32, i32), String)> {
    let role = attr_string(el, "AXRole")?;
    if !role.contains("Window") && !role.contains("Sheet") && !role.contains("Drawer") {
        return None;
    }
    let title = attr_string(el, "AXTitle").unwrap_or_default();
    let (origin, _) = frame_of(el)?;
    Some((origin, title))
}

/// Find the recorded element inside `window` by accessibility identity, breadth-first and
/// bounded.
fn find_in(window: &AXUIElement, target: &crate::model::ElementRef) -> Option<AXUIElement> {
    let matches = |el: &AXUIElement| {
        let role = attr_string(el, "AXRole").unwrap_or_default();
        if !role.is_empty() && !target.control_type.is_empty() && role != target.control_type {
            return false;
        }
        if !target.automation_id.is_empty() {
            attr_string(el, "AXIdentifier").as_deref() == Some(target.automation_id.as_str())
        } else {
            !target.name.is_empty() && attr_string(el, "AXTitle").as_deref() == Some(target.name.as_str())
        }
    };
    if matches(window) {
        return Some(window.clone());
    }
    let mut queue = std::collections::VecDeque::from([window.clone()]);
    let mut seen = 0usize;
    while let Some(node) = queue.pop_front() {
        if let Ok(children) = node.children() {
            for c in children {
                seen += 1;
                if seen > MAX_NODES {
                    return None;
                }
                if matches(&c) {
                    return Some(c);
                }
                queue.push_back(c);
            }
        }
    }
    None
}

pub(crate) struct MacEl(AXUIElement);

impl Element for MacEl {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn is_password(&self) -> bool {
        // macOS exposes password fields as secure text fields; the subrole is the signal.
        attr_string(&self.0, "AXSubrole").as_deref() == Some("AXSecureTextField")
    }
    fn value(&self) -> Option<String> {
        attr_string(&self.0, "AXValue")
    }
    fn centre(&self) -> (i32, i32) {
        centre_of(&self.0)
    }
    fn pid(&self) -> Option<u32> {
        pid_of(&self.0)
    }
    fn invoke(&self) -> bool {
        // AXPress is how a button/menu is activated without a pointer.
        self.0.press().is_ok()
    }
    fn click(&self) -> bool {
        let (x, y) = centre_of(&self.0);
        click_at(x, y).is_ok()
    }
    fn set_value(&self, value: &str) -> bool {
        self.0.set_value(&value.to_string()).is_ok()
    }
    fn focus(&self) -> bool {
        self.0.focus().is_ok()
    }
}

struct MacWindow {
    el: AXUIElement,
    pid: u32,
    title: String,
    origin: (i32, i32),
}

impl Window for MacWindow {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn title(&self) -> &str {
        &self.title
    }
    fn origin(&self) -> (i32, i32) {
        self.origin
    }
}

#[allow(dead_code)]
fn _pid_of_window(w: &MacWindow) -> u32 {
    w.pid
}
