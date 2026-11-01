//! Windows: the [`Platform`] implementation built on UI Automation (element identity),
//! `rdev` (input observation) and `SendInput` via enigo (injection).

mod uia;

use crate::input;
use crate::platform::{Element, KeyName, Observed, Platform, Window};
use crate::{procs, shot};
use anyhow::{anyhow, Result};
use std::any::Any;
use std::path::Path;
use std::time::Duration;
use uiautomation::UIElement;

/// UI Automation, xcap and enigo errors all implement Debug; not all of them implement
/// `std::error::Error + Send + Sync`, which `?` into anyhow needs.
pub(crate) fn err<E: std::fmt::Debug>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{e:?}")
}

pub struct WinPlatform;

impl WinPlatform {
    pub fn new() -> Self {
        Self
    }
}

impl Platform for WinPlatform {
    fn name(&self) -> &'static str {
        "windows"
    }

    fn subscribe(&mut self) -> Result<()> {
        input::subscribe();
        Ok(())
    }

    fn poll(&mut self, timeout: Duration) -> Option<Observed> {
        match input::next(timeout) {
            Ok(ev) => Some(match ev {
                input::Input::Click { x, y } => Observed::Click { x, y },
                input::Input::Key { key, text, ctrl, alt, shift, meta } => Observed::Key { key: map_key(key), text, ctrl, alt, shift, meta },
            }),
            Err(_) => None,
        }
    }

    fn element_at(&mut self, x: i32, y: i32) -> Option<Box<dyn Element>> {
        let uia = uia::Uia::new().ok()?;
        uia.at(x, y).map(|el| Box::new(WinEl(el)) as Box<dyn Element>)
    }

    fn focused_element(&mut self) -> Option<Box<dyn Element>> {
        let uia = uia::Uia::new().ok()?;
        uia.focused().map(|el| Box::new(WinEl(el)) as Box<dyn Element>)
    }

    fn describe_element(&mut self, element: &dyn Element, click: Option<(i32, i32)>, process_name: &str) -> crate::model::ElementRef {
        let Some(el) = element.as_any().downcast_ref::<WinEl>() else { return crate::model::ElementRef::default() };
        let uia = match uia::Uia::new() {
            Ok(u) => u,
            Err(_) => return crate::model::ElementRef::default(),
        };
        let win = uia.window_of(&el.0);
        let win_pid = uia::pid(&el.0).and_then(|p| procs::name_of(p));
        let (px, py) = click.unwrap_or_else(|| uia::centre(&el.0));
        let (wx, wy) = win.as_ref().map(|w| (w.rect.get_left(), w.rect.get_top())).unwrap_or((0, 0));
        crate::model::ElementRef {
            name: el.0.get_name().unwrap_or_default(),
            automation_id: el.0.get_automation_id().unwrap_or_default(),
            control_type: uia::control_type(&el.0),
            class_name: el.0.get_classname().unwrap_or_default(),
            window_title: win.map(|w| w.title).unwrap_or_default(),
            process_name: if process_name.is_empty() { win_pid.unwrap_or_default() } else { process_name.to_string() },
            rel_x: px - wx,
            rel_y: py - wy,
        }
    }

    fn same_element(&mut self, a: &dyn Element, b: &dyn Element) -> bool {
        let (Some(a), Some(b)) = (a.as_any().downcast_ref::<WinEl>(), b.as_any().downcast_ref::<WinEl>()) else { return false };
        match uia::Uia::new() {
            Ok(uia) => uia::same(&a.0, &b.0, &uia),
            Err(_) => false,
        }
    }

    fn windows_of(&mut self, pids: &[u32]) -> Vec<Box<dyn Window>> {
        let Ok(uia) = uia::Uia::new() else { return Vec::new() };
        uia.windows_of(pids).into_iter().map(|w| Box::new(WinWindow(w)) as Box<dyn Window>).collect()
    }

    fn find(&mut self, window: &dyn Window, target: &crate::model::ElementRef) -> Option<Box<dyn Element>> {
        let win = window.as_any().downcast_ref::<WinWindow>()?;
        let uia = uia::Uia::new().ok()?;
        uia.find(&win.0, target).map(|el| Box::new(WinEl(el)) as Box<dyn Element>)
    }

    fn injecting(&mut self, f: &mut dyn FnMut()) {
        input::injecting(f)
    }

    fn click_at(&mut self, x: i32, y: i32) -> Result<()> {
        use enigo::{Button, Coordinate, Direction, Enigo, Mouse, Settings};
        let mut enigo = Enigo::new(&Settings::default()).map_err(err)?;
        let mut out = Ok(());
        self.injecting(&mut || {
            out = enigo
                .move_mouse(x, y, Coordinate::Abs)
                .and_then(|_| enigo.button(Button::Left, Direction::Click))
                .map_err(err);
        });
        out
    }

    fn replace_text(&mut self, text: &str) -> Result<()> {
        use enigo::{Direction, Enigo, Key, Keyboard, Settings};
        let mut enigo = Enigo::new(&Settings::default()).map_err(err)?;
        let mut out = Ok(());
        self.injecting(&mut || {
            out = (|| {
                enigo.key(Key::Control, Direction::Press).map_err(err)?;
                enigo.key(Key::Unicode('a'), Direction::Click).map_err(err)?;
                enigo.key(Key::Control, Direction::Release).map_err(err)?;
                if text.is_empty() {
                    enigo.key(Key::Delete, Direction::Click).map_err(err)?;
                } else {
                    enigo.text(text).map_err(err)?;
                }
                Ok(())
            })();
        });
        out
    }

    fn send_keys(&mut self, combo: &str) -> Result<()> {
        use enigo::{Direction, Enigo, Keyboard, Settings};
        let (mods, key) = parse_combo(combo).ok_or_else(|| anyhow!("unknown key “{combo}”"))?;
        let mut enigo = Enigo::new(&Settings::default()).map_err(err)?;
        let mut out = Ok(());
        self.injecting(&mut || {
            for m in &mods {
                if let Err(e) = enigo.key(*m, Direction::Press).map_err(err) {
                    out = Err(e);
                    return;
                }
            }
            let r = enigo.key(key, Direction::Click).map_err(err);
            for m in mods.iter().rev() {
                if let Err(e) = enigo.key(*m, Direction::Release).map_err(err) {
                    out = Err(e);
                    return;
                }
            }
            out = r;
        });
        out
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

/// `Ctrl+Shift+S` → ([Control, Shift], Unicode('s')).
fn parse_combo(combo: &str) -> Option<(Vec<enigo::Key>, enigo::Key)> {
    use enigo::Key;
    let mut parts: Vec<&str> = combo.split('+').collect();
    let last = parts.pop()?;
    let mods = parts
        .into_iter()
        .map(|m| match m {
            "Ctrl" => Some(Key::Control),
            "Alt" => Some(Key::Alt),
            "Shift" => Some(Key::Shift),
            "Win" => Some(Key::Meta),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let key = match last {
        "Enter" => Key::Return,
        "Tab" => Key::Tab,
        "Escape" => Key::Escape,
        "Delete" => Key::Delete,
        "Up" => Key::UpArrow,
        "Down" => Key::DownArrow,
        "Left" => Key::LeftArrow,
        "Right" => Key::RightArrow,
        "PageUp" => Key::PageUp,
        "PageDown" => Key::PageDown,
        "Home" => Key::Home,
        "End" => Key::End,
        "Space" => Key::Space,
        f if f.starts_with('F') && f.len() > 1 => function_key(f[1..].parse().ok()?)?,
        c if c.chars().count() == 1 => Key::Unicode(c.chars().next()?.to_ascii_lowercase()),
        _ => return None,
    };
    Some((mods, key))
}

fn function_key(n: u8) -> Option<enigo::Key> {
    use enigo::Key;
    Some(match n {
        1 => Key::F1,
        2 => Key::F2,
        3 => Key::F3,
        4 => Key::F4,
        5 => Key::F5,
        6 => Key::F6,
        7 => Key::F7,
        8 => Key::F8,
        9 => Key::F9,
        10 => Key::F10,
        11 => Key::F11,
        12 => Key::F12,
        _ => return None,
    })
}

pub(crate) struct WinEl(UIElement);

impl Element for WinEl {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn is_password(&self) -> bool {
        uia::is_password(&self.0)
    }
    fn value(&self) -> Option<String> {
        uia::value(&self.0)
    }
    fn centre(&self) -> (i32, i32) {
        uia::centre(&self.0)
    }
    fn pid(&self) -> Option<u32> {
        uia::pid(&self.0)
    }
    fn invoke(&self) -> bool {
        uia::invoke(&self.0)
    }
    fn click(&self) -> bool {
        self.0.click().is_ok()
    }
    fn set_value(&self, value: &str) -> bool {
        uia::set_value(&self.0, value)
    }
    fn focus(&self) -> bool {
        self.0.set_focus().is_ok()
    }
}

struct WinWindow(uia::Window);

impl Window for WinWindow {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn title(&self) -> &str {
        &self.0.title
    }
    fn origin(&self) -> (i32, i32) {
        (self.0.rect.get_left(), self.0.rect.get_top())
    }
}
