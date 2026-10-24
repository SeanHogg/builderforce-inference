//! The engine's state machine: it owns its backend and at most one loaded model;
//! `load` / `chat` are exclusive (a second call while one runs is `busy:`), `unload`
//! during either aborts it, and once a GPU device is lost the model is dropped and every
//! later `load` / `chat` is `webgpu_unavailable:` — create a new engine.
//!
//! The awaits are the glue's (a browser promise, a native engine thread), so each
//! operation is split into a `begin_*` that claims the engine and a `finish_*` that
//! settles it — the claim is never held across the glue's own borrow.

use crate::device::Device;
use crate::errors::{error, EngineError, Kind};
use crate::loader::{Loaded, ModelInfo};

pub struct EngineState {
    device: Device,
    model: Option<Loaded>,
    busy: bool,
    unload_requested: bool,
}

impl EngineState {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            model: None,
            busy: false,
            unload_requested: false,
        }
    }

    /// `"webgpu"` or `"cpu"`.
    pub fn backend(&self) -> &'static str {
        self.device.name()
    }

    /// `Err(webgpu_unavailable)` once the device is lost; drops the model (its buffers
    /// belong to the dead device, so there is nothing to release).
    fn alive(&mut self) -> Result<(), EngineError> {
        let Some(why) = self.device.lost() else {
            return Ok(());
        };
        self.model = None;
        Err(error(
            Kind::WebgpuUnavailable,
            format!("the GPU device was lost ({why}); create a new BfiEngine"),
        ))
    }

    /// Claim the engine for a load: drops any model already loaded, and returns the
    /// device to load onto.
    pub fn begin_load(&mut self) -> Result<Device, EngineError> {
        if self.busy {
            return Err(error(Kind::Busy, "another load or chat is running"));
        }
        self.alive()?;
        self.busy = true;
        self.unload_requested = false;
        if self.model.take().is_some() {
            self.device.release_unused();
        }
        Ok(self.device.clone())
    }

    /// Settle a load: keep the model unless `unload` arrived meanwhile.
    pub fn finish_load(&mut self, result: Result<Loaded, EngineError>) -> Result<ModelInfo, EngineError> {
        self.busy = false;
        // A lost device outranks whatever the load returned.
        self.alive()?;
        match result {
            Ok(loaded) if !self.unload_requested => {
                let info = loaded.info();
                self.model = Some(loaded);
                Ok(info)
            }
            Ok(_) => {
                self.unload_requested = false;
                self.device.release_unused();
                Err(error(Kind::Aborted, ""))
            }
            Err(e) => {
                self.device.release_unused();
                Err(e)
            }
        }
    }

    /// Claim the engine for a chat: hands out the loaded model for the turn.
    pub fn begin_chat(&mut self) -> Result<Loaded, EngineError> {
        if self.busy {
            return Err(error(Kind::Busy, "another load or chat is running"));
        }
        self.alive()?;
        let Some(m) = self.model.take() else {
            return Err(error(Kind::NotLoaded, "call load() first"));
        };
        self.busy = true;
        Ok(m)
    }

    /// Whether a running turn must stop at its next step: `unload` or a lost device.
    pub fn should_stop(&self) -> bool {
        self.unload_requested || self.device.lost().is_some()
    }

    /// Settle a chat: the model comes back unless `unload` or a lost device ended it.
    pub fn finish_chat(&mut self, loaded: Loaded, result: Result<String, EngineError>) -> Result<String, EngineError> {
        self.busy = false;
        if self.should_stop() {
            self.unload_requested = false;
            drop(loaded);
            // Lost: reject webgpu_unavailable (nothing to release on a dead device).
            self.alive()?;
            self.device.release_unused();
        } else {
            self.model = Some(loaded);
        }
        result
    }

    /// Drop the model and its memory. During a `load` or `chat`, that call ends with
    /// `aborted` and the model is dropped when it unwinds.
    pub fn unload(&mut self) {
        if self.busy {
            self.unload_requested = true;
            return;
        }
        if self.model.take().is_some() {
            self.device.release_unused();
        }
    }

    /// Run one request on the claimed model — the [`crate::chat`] loop.
    pub async fn chat(
        loaded: &mut Loaded,
        request_json: &str,
        host: &impl crate::Host,
        on_chunk: &mut dyn FnMut(&str) -> Result<(), EngineError>,
        stop: &dyn Fn() -> bool,
    ) -> Result<String, EngineError> {
        crate::chat::chat(loaded, request_json, host, on_chunk, stop).await
    }

    /// Stream a GGUF onto the claimed device — the [`crate::loader`] path.
    pub async fn load(
        device: &Device,
        profile: &crate::HostProfile,
        source: &mut impl crate::ByteSource,
        total_bytes: f64,
        on_progress: &mut dyn FnMut(f64, f64),
        opts: crate::LoadOptions,
    ) -> Result<Loaded, EngineError> {
        crate::loader::load(device, profile, source, total_bytes, on_progress, opts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_before_load_and_a_second_load_are_refused() {
        let mut s = EngineState::new(Device::Cpu);
        assert_eq!(s.begin_chat().err().unwrap().kind, Kind::NotLoaded);
        s.begin_load().unwrap();
        assert_eq!(s.begin_load().err().unwrap().kind, Kind::Busy);
        let failed = s.finish_load(Err(error(Kind::LoadFailed, "x")));
        assert_eq!(failed.unwrap_err().kind, Kind::LoadFailed);
        // Settled: the engine takes another load.
        assert!(s.begin_load().is_ok());
    }

    #[test]
    fn unload_during_a_load_aborts_it() {
        let mut s = EngineState::new(Device::Cpu);
        s.begin_load().unwrap();
        s.unload();
        assert!(s.should_stop());
    }
}
