//! minijinja chat-template rendering.

use anyhow::Result;
use minijinja::{context, Environment};
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
        return Ok(s);
    }

    let msgs: Vec<minijinja::Value> = messages
        .iter()
        .map(|m| {
            context! {
                role => m.role.as_str(),
                content => m.content.as_str(),
            }
        })
        .collect();

    let mut env = Environment::new();
    env.add_template("chat", template)?;
    let tmpl = env.get_template("chat")?;
    Ok(tmpl.render(context! {
        messages => msgs,
        add_generation_prompt => add_generation_prompt,
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_jinja() {
        let t = "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}";
        let out = apply_chat_template(
            t,
            &[ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            false,
        )
        .unwrap();
        assert_eq!(out, "user:hi\n");
    }

    #[test]
    fn empty_template_fallback() {
        let out = apply_chat_template(
            "",
            &[ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            true,
        )
        .unwrap();
        assert_eq!(out, "user: hi\nassistant:\n");
    }
}
