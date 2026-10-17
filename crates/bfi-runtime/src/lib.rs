//! Tokenizer, chat templates, sampler, model store, fit math, and generate loop.
//!
//! Everything except the model store compiles for wasm32 (the browser engine
//! `bfi-web` uses the tokenizer, templates, sampler, tools and [`openai`]).

mod byte_level;
mod chat;
mod engine;
mod fit;
mod jinja_compat;
pub mod openai;
mod sample;
// The model store is the filesystem (`~/.builderforce/models`); browsers have none.
#[cfg(not(target_arch = "wasm32"))]
mod store;
mod tokenize;
mod tools;

pub use chat::{apply_chat_template, chatml_prompt, render_prompt, render_template, ChatMessage};
pub use engine::{mix_decode, run_queued, Engine, GreedyTrace, Logits, SeqState, StateLayout};
pub use fit::{fit, FitBreakdown, FitReport, FitShape, RUNTIME_OVERHEAD_BYTES};
pub use sample::{apply_constraint, sample, SampleParams};
#[cfg(not(target_arch = "wasm32"))]
pub use store::{aliases, list_models, model_path, models_dir, AliasMap, InstalledModel};
pub use tokenize::Tokenizer;
pub use openai::{content_text, to_chat_messages, ChatPlan, ChatRequest, ChatTurn, IncomingMessage};
pub use tools::{
    apply_tools, parse_tool_calls, parse_tool_choice, parse_tool_specs, prepend_tools_block,
    tools_system_block, ParsedToolCall, ToolChoice, ToolSpec,
};
