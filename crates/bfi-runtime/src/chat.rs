//! Chat-template rendering (`tokenizer.chat_template`, minijinja) with fallbacks.
//!
//! Order: the GGUF's own template (with [`crate::jinja_compat`] shims) →
//! ChatML when the template is ChatML-shaped but uses Jinja we cannot run →
//! `role: content` lines.

use anyhow::Result;
use minijinja::{context, Environment, Value};
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Render `template` over `messages`. Empty template falls back to `role: content\n`.
pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    if template.trim().is_empty() {
        return Ok(naive_prompt(messages, add_generation_prompt));
    }
    let msgs: Vec<Value> = messages.iter().map(Value::from_serialize).collect();
    render_template(template, Value::from(msgs), None, add_generation_prompt)
}

/// Full OpenAI-shaped render: `messages` keep `tool_calls` / `tool_call_id` /
/// `name`, and `tools` is exposed to the template (Qwen3 renders its own
/// `<tools>` block and expects `<tool_call>` output).
pub fn render_template(
    template: &str,
    messages: Value,
    tools: Option<&serde_json::Value>,
    add_generation_prompt: bool,
) -> Result<String> {
    let mut env = Environment::new();
    crate::jinja_compat::install(&mut env);
    env.add_template("chat", template)?;
    let tmpl = env.get_template("chat")?;
    let tools = tools
        .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()))
        .map(Value::from_serialize);
    Ok(tmpl.render(context! {
        messages => messages,
        tools => tools,
        add_generation_prompt => add_generation_prompt,
        bos_token => "",
        eos_token => "",
    })?)
}

/// The prompt a model sees for `messages`: the GGUF template when it renders,
/// ChatML for ChatML-shaped templates that do not, else `role: content` lines.
pub fn render_prompt(template: Option<&str>, messages: &[ChatMessage]) -> String {
    match template {
        Some(t) if !t.trim().is_empty() => match apply_chat_template(t, messages, true) {
            Ok(s) => s,
            Err(_) if is_chatml(t) => chatml_prompt(messages),
            Err(_) => naive_prompt(messages, true),
        },
        _ => naive_prompt(messages, true),
    }
}

/// Template uses `<|im_start|>` / `<|im_end|>` (Qwen, Bonsai, many others).
pub fn is_chatml(template: &str) -> bool {
    template.contains("<|im_start|>")
}

/// `<|im_start|>{role}\n{content}<|im_end|>\n` per message, then the assistant header.
pub fn chatml_prompt(messages: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str("<|im_start|>");
        s.push_str(&m.role);
        s.push('\n');
        s.push_str(&m.content);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

pub fn naive_prompt(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str(&m.role);
        s.push_str(": ");
        s.push_str(&m.content);
        s.push('\n');
    }
    if add_generation_prompt {
        s.push_str("assistant:\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(c: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: c.into(),
        }
    }

    #[test]
    fn tiny_jinja() {
        let t = "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}";
        let out = apply_chat_template(t, &[user("hi")], false).unwrap();
        assert_eq!(out, "user:hi\n");
    }

    #[test]
    fn empty_template_fallback() {
        let out = apply_chat_template("", &[user("hi")], true).unwrap();
        assert_eq!(out, "user: hi\nassistant:\n");
    }

    #[test]
    fn chatml_fallback_when_template_cannot_render() {
        let broken = "<|im_start|>{% for m in messages %}{{ m.nope( }}{% endfor %}";
        let out = render_prompt(Some(broken), &[user("hi")]);
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn qwen_style_template_with_tools_and_pycompat() {
        let t = "{% if tools %}<tools>{% for t in tools %}{{ t | tojson }}{% endfor %}</tools>\n{% endif %}\
{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.content is string and m.content.startswith('<tool_response>') %}TR{% else %}{{ m.content }}{% endif %}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let tools = serde_json::json!([{"type": "function", "function": {"name": "f"}}]);
        let msgs = Value::from_serialize(serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "<tool_response>x</tool_response>"}
        ]));
        let out = render_template(t, msgs, Some(&tools), true).unwrap();
        assert!(out.starts_with("<tools>{"), "{out}");
        assert!(out.contains("\"name\":\"f\""), "{out}");
        assert!(out.contains("<|im_start|>tool\nTR<|im_end|>"), "{out}");
        assert!(out.ends_with("<|im_start|>assistant\n"), "{out}");
    }
}
