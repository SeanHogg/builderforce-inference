use std::sync::Arc;

use anyhow::{anyhow, Context};
use bfi_core::Backend;
use bfi_gguf::GgufFile;

use crate::state::{Logits, SeqState, StateLayout};

/// One registered architecture (qwen3, qwen35, …).
pub trait Architecture: Send + Sync {
    fn name(&self) -> &'static str;
    fn from_gguf(
        &self,
        file: &GgufFile,
        backend: Arc<dyn Backend>,
    ) -> anyhow::Result<Box<dyn Model>>;
}

/// Loaded weights + a forward that writes last-token logits (or every position's).
pub trait Model: Send + Sync {
    fn architecture(&self) -> &'static str;
    fn n_layer(&self) -> usize;
    fn n_embd(&self) -> usize;
    fn n_vocab(&self) -> usize;
    fn n_ctx(&self) -> usize;
    fn n_head(&self) -> usize;
    fn n_head_kv(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn forward(
        &self,
        tokens: &[u32],
        pos: &[i32],
        state: &mut SeqState,
    ) -> anyhow::Result<Logits>;
    /// Logits for EVERY position of `tokens`: row `i` is the next-token distribution after
    /// `tokens[..=i]`, and `state` advances by `tokens.len()` exactly as [`Model::forward`]
    /// does. Speculative decoding verifies a k-token draft with one call. The default runs
    /// one `forward` per token — correct for any architecture, just no speedup; qwen3 and
    /// qwen35 override it with one batched pass.
    fn forward_each(
        &self,
        tokens: &[u32],
        pos: &[i32],
        state: &mut SeqState,
    ) -> anyhow::Result<Vec<Logits>> {
        anyhow::ensure!(
            pos.len() == tokens.len(),
            "forward_each: pos len {} != tokens {}",
            pos.len(),
            tokens.len()
        );
        tokens
            .iter()
            .zip(pos)
            .map(|(&t, &p)| self.forward(&[t], &[p], state))
            .collect()
    }
    fn state_layout(&self) -> StateLayout;
    /// MTP / next-n tensors loaded but not run. Default 0.
    fn mtp_loaded(&self) -> usize {
        0
    }
}

/// Inventory payload. Architectures submit `&'static dyn Architecture`.
pub struct ArchReg(pub &'static dyn Architecture);

inventory::collect!(ArchReg);

pub fn architecture_for(name: &str) -> Option<&'static dyn Architecture> {
    inventory::iter::<ArchReg>()
        .find(|r| r.0.name() == name)
        .map(|r| r.0)
}

/// Dispatch on `general.architecture`.
pub fn load_model(
    file: &GgufFile,
    backend: Arc<dyn Backend>,
) -> anyhow::Result<Box<dyn Model>> {
    let name = file
        .header()
        .architecture()
        .ok_or_else(|| anyhow!("GGUF missing metadata key general.architecture"))?;
    let arch = architecture_for(name).ok_or_else(|| {
        let known: Vec<&str> = inventory::iter::<ArchReg>().map(|r| r.0.name()).collect();
        anyhow!("unsupported architecture '{name}'; registered: {known:?}")
    })?;
    arch.from_gguf(file, backend)
        .with_context(|| format!("loading architecture '{name}' from {}", file.path.display()))
}
