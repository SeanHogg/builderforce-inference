//! What `bfi-engine` needs from a native engine thread: a clock, a no-op yield (the JS
//! event loop runs on another thread, so an abort is a flag the engine polls), a way to
//! drive the engine's futures to completion, and the GGUF bytes the JS side pushes.

use std::future::Future;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use bfi_engine::{error, ByteSource, EngineError, Host, Kind};
use napi::{Env, JsDeferred};

/// The engine thread's realm.
pub struct NativeHost;

impl Host for NativeHost {
    fn now_secs(&self) -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
    }

    /// Nothing to yield to: this thread runs only the engine.
    async fn yield_now(&self) {}
}

/// Drive an engine future on this thread. With the GPU this is CubeCL's own executor
/// (it polls the wgpu device while a readback is pending); CPU futures never pend.
#[cfg(feature = "webgpu")]
pub fn block_on<F: Future>(f: F) -> F::Output {
    bfi_core::resident::block_on(f)
}

#[cfg(not(feature = "webgpu"))]
pub fn block_on<F: Future>(f: F) -> F::Output {
    use std::pin::pin;
    use std::task::{Context, Poll, Wake, Waker};
    use std::thread::{self, Thread};

    struct Unpark(Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let mut f = pin!(f);
    let waker = Waker::from(Arc::new(Unpark(thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        thread::park();
    }
}

/// Settles one `pushChunk` promise: resolved when the engine takes the chunk, which is
/// what paces the JS side to one chunk in flight (never the whole file in memory).
pub type Ack = JsDeferred<(), fn(Env) -> napi::Result<()>>;

/// One message from the JS side's stream reader.
pub enum Chunk {
    Data(Vec<u8>, Ack),
    End,
    Fail(String),
}

/// Where `pushChunk` sends while a load runs; `None` between loads.
pub type ChunkSlot = Arc<Mutex<Option<std::sync::mpsc::Sender<Chunk>>>>;

/// The chunks of the load in progress. On drop the slot is closed first (so no push can
/// land after), then every chunk still queued has its promise rejected — a load that
/// ended early (a bad header, an unload) never leaves the JS reader waiting.
pub struct ChannelSource {
    pub rx: Receiver<Chunk>,
    pub slot: ChunkSlot,
}

impl ByteSource for ChannelSource {
    async fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, EngineError> {
        match self.rx.recv() {
            Ok(Chunk::Data(bytes, ack)) => {
                ack.resolve(|_| Ok(()));
                Ok(Some(bytes))
            }
            Ok(Chunk::End) => Ok(None),
            Ok(Chunk::Fail(why)) => Err(error(Kind::LoadFailed, format!("stream errored: {why}"))),
            Err(_) => Err(error(Kind::LoadFailed, "the chunk stream closed before its end")),
        }
    }
}

impl Drop for ChannelSource {
    fn drop(&mut self) {
        self.slot.lock().unwrap_or_else(|p| p.into_inner()).take();
        while let Ok(chunk) = self.rx.try_recv() {
            if let Chunk::Data(_, ack) = chunk {
                ack.reject(napi::Error::from_reason("load_failed: the load ended before this chunk was read"));
            }
        }
    }
}
