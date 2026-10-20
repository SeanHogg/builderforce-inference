//! Device health: WebGPU error scopes around allocations, and `device.lost`.
//!
//! CubeCL creates buffers without error scopes and never watches `device.lost`,
//! so on WebGPU an out-of-memory upload or KV allocation yields an *invalid*
//! buffer silently and a lost device turns every later call into garbage. The
//! raw `wgpu::Device` kept from `init_setup_async` (browser path) lets
//! [`Gpu::checked`] run any allocating closure inside out-of-memory +
//! validation scopes, and the device-lost callback marks the [`Gpu`] dead so
//! callers can refuse instead of computing on a lost device.
//!
//! Without a raw device (native wgpu, tests) scopes are skipped and work is
//! trusted, exactly as before.

use std::sync::{Arc, Mutex, PoisonError};

use super::{Gpu, ResidentError};

/// Shared "device lost" flag (set from wgpu's device-lost callback, which must
/// be `Send`, hence `Arc<Mutex>` rather than `Rc<Cell>`).
#[derive(Clone, Default)]
pub(crate) struct DeviceHealth(Arc<Mutex<Option<String>>>);

impl DeviceHealth {
    pub(crate) fn mark_lost(&self, why: String) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(why);
    }

    fn lost(&self) -> Option<String> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).clone()
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

    /// Run `f` (uploads, KV allocation, …) inside WebGPU out-of-memory and
    /// validation error scopes, flush the queued work, and await the scopes.
    /// `Err(Allocation)` when either scope captured an error (the buffers `f`
    /// made are invalid: drop them), `Err(DeviceLost)` when the device is gone.
    pub async fn checked<T>(&self, what: &str, f: impl FnOnce() -> T) -> Result<T, ResidentError> {
        self.ensure_alive()?;
        let Some(device) = &self.probe_device else {
            return Ok(f());
        };
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
    /// scopes, awaits the scopes, then destroys them. Without a probe device
    /// (native wgpu, tests) it trusts the planned size and returns `true`.
    pub async fn kv_fits(&self, buffer_bytes: u64, count: usize) -> bool {
        let Some(device) = &self.probe_device else {
            return true;
        };
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
        self.read_f32(&probe).await.map(|_| ())
    }
}
