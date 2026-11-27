//! The seam between the protocol and the desktop. The real backend drives the platform
//! through [`crate::engine`]; tests substitute a fake so they never touch the desktop.

use crate::model::{RecordedStep, RunStatus, Skill};
use crate::RunHooks;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;

/// A recording in progress.
pub trait RecordHandle: Send {
    /// Steps captured so far.
    fn count(&self) -> usize;
    /// Why recording ended early, if it did.
    fn error(&self) -> Option<String>;
    /// Stop and return every step.
    fn stop(self: Box<Self>) -> Vec<RecordedStep>;
}

pub trait Backend: Send + Sync + 'static {
    fn platform(&self) -> &'static str;
    fn supported(&self) -> bool;
    fn start_recording(&self, program: &str, args: &[String], shots_dir: &Path) -> Result<Box<dyn RecordHandle>>;
    fn run(&self, skill: &Skill, values: &BTreeMap<String, String>, hooks: &mut dyn RunHooks) -> Result<RunStatus>;
}

/// The desktop this process runs on.
pub struct DesktopBackend;

struct DesktopRecording(crate::Recording);

impl RecordHandle for DesktopRecording {
    fn count(&self) -> usize {
        self.0.step_count()
    }
    fn error(&self) -> Option<String> {
        self.0.error()
    }
    fn stop(self: Box<Self>) -> Vec<RecordedStep> {
        let this = *self;
        this.0.stop()
    }
}

impl Backend for DesktopBackend {
    fn platform(&self) -> &'static str {
        crate::platform::selected_name()
    }
    fn supported(&self) -> bool {
        crate::supported()
    }
    fn start_recording(&self, program: &str, args: &[String], shots_dir: &Path) -> Result<Box<dyn RecordHandle>> {
        Ok(Box::new(DesktopRecording(crate::start_recording(program, args, shots_dir)?)))
    }
    fn run(&self, skill: &Skill, values: &BTreeMap<String, String>, hooks: &mut dyn RunHooks) -> Result<RunStatus> {
        crate::run_skill(skill, values, hooks)
    }
}
