//! Tokenizer, chat templates, sampler, model store, fit math, and generate loop.

mod chat;
mod engine;
mod fit;
mod sample;
mod store;
mod tokenize;
mod tools;

pub use chat::{apply_chat_template, ChatMessage};
pub use engine::{mix_decode, run_queued, Engine, GreedyTrace, Logits, SeqState, StateLayout};
pub use fit::{fit, FitBreakdown, FitReport, FitShape, RUNTIME_OVERHEAD_BYTES};
pub use sample::{apply_constraint, sample, SampleParams};
pub use store::{aliases, list_models, model_path, models_dir, AliasMap, InstalledModel};
pub use tokenize::Tokenizer;
pub use tools::{
    parse_tool_calls, parse_tool_choice, parse_tool_specs, tools_system_block, ParsedToolCall,
    ToolChoice, ToolSpec,
};
