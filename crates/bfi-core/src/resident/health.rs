//! Device health: WebGPU error scopes around allocations, and `device.lost`.
//!
//! CubeCL creates buffers without error scopes and never watches `device.lost`,
//! so on WebGPU an out-of-memory upload or KV allocation yields an *invalid*
//! buffer silently and a lost device turns every later call into garbage. The
//! raw `wgpu::Device` kept from CubeCL's setup (`init_setup_async` in the browser,
//! `init_setup` natively, see `device.rs`) lets [`Gpu::checked`] catch an
//! allocating closure's errors, and the device-lost callback marks the [`Gpu`]
//! dead so callers can refuse instead of computing on a lost device.
//!
//! How `checked` catches them differs by target. In the browser CubeCL runs on
//! the calling thread, so out-of-memory + validation error SCOPES wrap the
//! closure. Natively wgpu's scopes are thread-local while CubeCL runs every
//! server call on its own device thread, so a scope pushed here would never see
//! them, and wgpu's default handler panics on an uncaptured error. There the
//! device's uncaptured-error handler records every error instead (`device.rs`),
//! and `checked` syncs the device and reads what was recorded.

use std::sync::{Arc, Mutex, PoisonError};

use super::{Gpu, ResidentError};

/// Shared "device lost" flag (set from wgpu's device-lost callback, which must
/// be `Send`, hence `Arc<Mutex>` rather than `Rc<Cell>`), and natively the first
/// uncaptured device error since it was last read.
#[derive(Clone, Default)]
pub(crate) struct DeviceHealth {
    lost: Arc<Mutex<Option<String>>>,
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    error: Arc<Mutex<Option<String>>>,
}

impl DeviceHealth {
    pub(crate) fn mark_lost(&self, why: String) {
        *self.lost.lock().unwrap_or_else(PoisonError::into_inner) = Some(why);
    }

    pub(crate) fn lost(&self) -> Option<String> {
        self.lost.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Record a device error (the first one wins until it is read).
    #[cfg(not(target_family = "wasm"))]
    fn record(&self, why: String) {
        self.error.lock().unwrap_or_else(PoisonError::into_inner).get_or_insert(why);
    }

    /// The error recorded since the last read, clearing it.
    #[cfg(not(target_family = "wasm"))]
    fn take_error(&self) -> Option<String> {
        self.error.lock().unwrap_or_else(PoisonError::into_inner).take()
    }

    /// Natively: record the device's uncaptured errors here instead of panicking.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn capture(&self, device: &wgpu::Device) {
        let health = self.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| health.record(e.to_string())));
    }

    /// Subscribe to `device.lost`; the reason lands in this flag.
    pub(crate) fn watch(&self, device: &wgpu::Device) {
        let health = self.clone();
        device.set_device_lost_callback(move |reason, msg| {
            health.mark_lost(format!("{reason:?}: {msg}"));
        });
    }
}

impl Gpu {
    /// `Some(reason)` once the device has been lost. A lost device never
    /// comes back: drop everything on it and create a new engine.
    pub fn lost(&self) -> Option<String> {
        self.health.lost()
    }

    fn ensure_alive(&self) -> Result<(), ResidentError> {
        match self.lost() {
            Some(why) => Err(ResidentError::DeviceLost(why)),
            None => Ok(()),
        }
    }

    /// Run `f` (uploads, KV allocation, …) and report the device errors it
    /// caused (see the module docs for how, per target). `Err(Allocation)` when
    /// it caused one (the buffers `f` made are invalid: drop them),
    /// `Err(DeviceLost)` when the device is gone.
    #[cfg(not(target_family = "wasm"))]
    pub async fn checked<T>(&self, what: &str, f: impl FnOnce() -> T) -> Result<T, ResidentError> {
        self.ensure_alive()?;
        // An error from before this call is not this call's.
        let _ = self.health.take_error();
        let out = f();
        let flushed = self.client.flush();
        let synced = self.client.sync().await;
        self.ensure_alive()?;
        if let Some(e) = self.health.take_error() {
            return Err(ResidentError::Allocation(format!("{what}: {e}")));
        }
        flushed.map_err(|e| ResidentError::Allocation(format!("{what}: flush failed: {e:?}")))?;
        synced.map_err(|e| ResidentError::Allocation(format!("{what}: sync failed: {e:?}")))?;
        Ok(out)
    }

    #[cfg(target_family = "wasm")]
    pub async fn checked<T>(&self, what: &str, f: impl FnOnce() -> T) -> Result<T, ResidentError> {
        self.ensure_alive()?;
        let device = &self.probe_device;
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let invalid = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out = f();
        let flushed = self.client.flush();
        let invalid_err = invalid.pop().await;
        let oom_err = oom.pop().await;
        self.ensure_alive()?;
        if let Some(e) = oom_err.or(invalid_err) {
            return Err(ResidentError::Allocation(format!("{what}: {e}")));
        }
        flushed.map_err(|e| ResidentError::Allocation(format!("{what}: flush failed: {e:?}")))?;
        Ok(out)
    }

    /// Whether `count` storage buffers of `buffer_bytes` each can be allocated
    /// right now. Allocates them inside WebGPU out-of-memory + validation error
    /// scopes, awaits the scopes, then destroys them.
    pub async fn kv_fits(&self, buffer_bytes: u64, count: usize) -> bool {
        let device = &self.probe_device;
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let invalid = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buffers: Vec<wgpu::Buffer> = (0..count)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bfi-kv-probe"),
                    size: buffer_bytes.max(4).div_ceil(4) * 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let invalid_err = invalid.pop().await;
        let oom_err = oom.pop().await;
        for b in &buffers {
            b.destroy();
        }
        invalid_err.is_none() && oom_err.is_none()
    }

    /// Await a one-word readback. The queue runs in order, so this resolves
    /// after everything enqueued before it, and awaiting it yields to the host
    /// event loop (an `AbortSignal` can fire). Used between prefill chunks.
    pub async fn drain(&self) -> Result<(), ResidentError> {
        self.ensure_alive()?;
        let probe = self.empty_f32(1);
        self.read_f32(&probe).await?;
        // Natively a device error during the work drained is recorded, not raised.
        #[cfg(not(target_family = "wasm"))]
        if let Some(e) = self.health.take_error() {
            return Err(ResidentError::Allocation(format!("device error: {e}")));
        }
        Ok(())
    }

    /// This `Gpu` with a health record of its own, so a test can mark it lost
    /// without killing the process's one shared device for every other test.
    #[cfg(test)]
    pub(crate) fn with_own_health(&self) -> Self {
        Self { health: DeviceHealth::default(), ..self.clone() }
    }
}
