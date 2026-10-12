//! OpenAI / Anthropic tool specs, `tool_choice`, and parse-from-decode.
//!
//! Constrained decode for `required` / named choice lives in [`crate::sample`].
//! This module is the request/response seam: parse schemas, emit tool-call JSON.

use serde_json::{json, Value};

/// One function the model may call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// OpenAI `tool_choice` / Anthropic equivalent.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolChoice {
    /// Do not call tools; emit plain text.
    #[default]
    None,
    /// Model may call a tool or answer in text.
    Auto,
    /// Decode is constrained to a JSON tool-call object.
    Required,
    /// Decode is constrained to a JSON call of this function.
    Named(String),
}

impl ToolChoice {
    /// `required` and named choices mask logits (JSON object / name prefix).
    pub fn constrains(&self) -> bool {
        matches!(self, Self::Required | Self::Named(_))
    }
}

/// One parsed tool call from model text (OpenAI `tool_calls[]` row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ParsedToolCall {
    pub fn openai_json(&self) -> Value {
        json!({
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            }
        })
    }

    pub fn anthropic_json(&self) -> Value {
        let input = serde_json::from_str::<Value>(&self.arguments).unwrap_or(json!({}));
        json!({
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": input,
        })
    }
}

/// Parse OpenAI `{type,function:{name,...}}` or Anthropic `{name,input_schema}` tools.
pub fn parse_tool_specs(v: &Value) -> Result<Vec<ToolSpec>, String> {
    let arr: Vec<Value> = match v {
        Value::Null => return Ok(vec![]),
        Value::Array(a) => a.clone(),
        Value::Object(_) => vec![v.clone()],
        _ => return Err("tools must be an array".into()),
    };
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        out.push(spec_from_value(item).map_err(|e| format!("tools[{i}]: {e}"))?);
    }
    Ok(out)
}

fn spec_from_value(v: &Value) -> Result<ToolSpec, String> {
    let obj = v.as_object().ok_or_else(|| "tool must be an object".to_string())?;
    if let Some(func) = obj.get("function") {
        let name = func
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if name.is_empty() {
            return Err("function.name is required".into());
        }
        return Ok(ToolSpec {
            name: name.into(),
            description: func
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .into(),
            parameters: func.get("parameters").cloned().unwrap_or(json!({})),
        });
    }
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if name.is_empty() {
        return Err("name is required".into());
    }
    Ok(ToolSpec {
        name: name.into(),
        description: obj
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .into(),
        parameters: obj
            .get("input_schema")
            .or_else(|| obj.get("parameters"))
            .cloned()
            .unwrap_or(json!({})),
    })
}

/// Parse OpenAI string/object or Anthropic `{type: auto|any|tool|none}` `tool_choice`.
pub fn parse_tool_choice(v: Option<&Value>) -> Result<ToolChoice, String> {
    let Some(v) = v else {
        return Ok(ToolChoice::None);
    };
    match v {
        Value::Null => Ok(ToolChoice::None),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(ToolChoice::None),
            "auto" => Ok(ToolChoice::Auto),
            "required" | "any" => Ok(ToolChoice::Required),
            other => Err(format!(
                "unsupported tool_choice '{other}'; expected none, auto, or required"
            )),
        },
        Value::Object(obj) => {
            let ty = obj.get("type").and_then(Value::as_str).unwrap_or("");
            match ty {
                "none" => Ok(ToolChoice::None),
                "auto" => Ok(ToolChoice::Auto),
                "required" | "any" => Ok(ToolChoice::Required),
                "function" => named_from_function(obj),
                "tool" => {
                    let name = obj.get("name").and_then(Value::as_str).unwrap_or("").trim();
                    if name.is_empty() {
                        return Err("tool_choice.name is required".into());
                    }
                    Ok(ToolChoice::Named(name.into()))
                }
                "" => {
                    if obj.get("function").is_some() {
                        named_from_function(obj)
                    } else {
                        Err("tool_choice.type is required".into())
                    }
                }
                other => Err(format!("unsupported tool_choice.type '{other}'")),
            }
        }
        _ => Err("tool_choice must be a string or object".into()),
    }
}

fn named_from_function(obj: &serde_json::Map<String, Value>) -> Result<ToolChoice, String> {
    let name = obj
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if name.is_empty() {
        return Err("tool_choice.function.name is required".into());
    }
    Ok(ToolChoice::Named(name.into()))
}

/// System preamble listing tools so the chat template / naive prompt can see them.
pub fn tools_system_block(tools: &[ToolSpec], choice: &ToolChoice) -> Option<String> {
    if tools.is_empty() || matches!(choice, ToolChoice::None) {
        return None;
    }
    let mut s = String::from(
        "You may call tools by emitting a JSON object {\"name\":...,\"arguments\":...}.\nTools:\n",
    );
    for t in tools {
        s.push_str(&format!(
            "- {}: {}\n  parameters: {}\n",
            t.name, t.description, t.parameters
        ));
    }
    match choice {
        ToolChoice::Required => s.push_str("You must call one of these tools.\n"),
        ToolChoice::Named(n) => s.push_str(&format!("You must call the tool `{n}`.\n")),
        _ => {}
    }
    Some(s)
}

/// Parse a complete tool call from decoded text. `None` if not (yet) a call.
pub fn parse_tool_calls(text: &str) -> Option<Vec<ParsedToolCall>> {
    let t = strip_fence(text.trim());
    if t.is_empty() {
        return None;
    }
    if let Some(calls) = from_xml_tool_calls(t) {
        return Some(calls);
    }
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        return calls_from_value(&v);
    }
    if let Some(obj) = first_json_object(t) {
        if let Ok(v) = serde_json::from_str::<Value>(obj) {
            return calls_from_value(&v);
        }
    }
    None
}

fn strip_fence(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s)
        .trim();
    s.strip_suffix("```").unwrap_or(s).trim()
}

fn first_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &c) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
            } else if c == b'\\' {
                escape = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
                if depth < 0 {
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

fn from_xml_tool_calls(t: &str) -> Option<Vec<ParsedToolCall>> {
    let mut out = Vec::new();
    let mut rest = t;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            break;
        };
        let inner = after[..end].trim();
        if let Ok(v) = serde_json::from_str::<Value>(inner) {
            if let Some(c) = call_from_obj(&v, out.len()) {
                out.push(c);
            }
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn calls_from_value(v: &Value) -> Option<Vec<ParsedToolCall>> {
    if let Some(arr) = v.as_array() {
        let mut out = Vec::new();
        for item in arr {
            if let Some(c) = call_from_obj(item, out.len()) {
                out.push(c);
            }
        }
        return if out.is_empty() { None } else { Some(out) };
    }
    if let Some(arr) = v.get("tool_calls").and_then(Value::as_array) {
        let mut out = Vec::new();
        for item in arr {
            if let Some(c) = call_from_obj(item, out.len()) {
                out.push(c);
            }
        }
        return if out.is_empty() { None } else { Some(out) };
    }
    call_from_obj(v, 0).map(|c| vec![c])
}

fn call_from_obj(v: &Value, idx: usize) -> Option<ParsedToolCall> {
    let name = v
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| {
            v.get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
        })?
        .trim();
    if name.is_empty() {
        return None;
    }
    let args = v
        .get("arguments")
        .or_else(|| v.get("input"))
        .or_else(|| {
            v.get("function")
                .and_then(|f| f.get("arguments"))
        });
    let arguments = match args {
        None => "{}".into(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    };
    Some(ParsedToolCall {
        id: format!("call_bfi_{idx}"),
        name: name.to_string(),
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_and_anthropic_specs() {
        let openai = json!([{
            "type": "function",
            "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}
        }]);
        let t = parse_tool_specs(&openai).unwrap();
        assert_eq!(t[0].name, "f");
        assert_eq!(t[0].description, "d");

        let anth = json!([{"name": "g", "input_schema": {"type": "object"}}]);
        let t = parse_tool_specs(&anth).unwrap();
        assert_eq!(t[0].name, "g");
    }

    #[test]
    fn choice_parses_string_and_objects() {
        assert_eq!(
            parse_tool_choice(Some(&json!("auto"))).unwrap(),
            ToolChoice::Auto
        );
        assert_eq!(
            parse_tool_choice(Some(&json!("required"))).unwrap(),
            ToolChoice::Required
        );
        assert_eq!(
            parse_tool_choice(Some(&json!({"type":"function","function":{"name":"f"}}))).unwrap(),
            ToolChoice::Named("f".into())
        );
        assert_eq!(
            parse_tool_choice(Some(&json!({"type":"tool","name":"g"}))).unwrap(),
            ToolChoice::Named("g".into())
        );
        assert_eq!(
            parse_tool_choice(Some(&json!({"type":"any"}))).unwrap(),
            ToolChoice::Required
        );
        assert!(parse_tool_choice(Some(&json!("banana"))).is_err());
    }

    #[test]
    fn parse_json_and_xml_tool_calls() {
        let calls = parse_tool_calls(r#"{"name":"f","arguments":{"q":1}}"#).unwrap();
        assert_eq!(calls[0].name, "f");
        assert!(calls[0].arguments.contains('1'));
        assert_eq!(calls[0].openai_json()["type"], "function");
        assert_eq!(calls[0].anthropic_json()["type"], "tool_use");

        let xml = parse_tool_calls(
            r#"<tool_call>{"name":"lookup","arguments":{}}</tool_call>"#,
        )
        .unwrap();
        assert_eq!(xml[0].name, "lookup");

        assert!(parse_tool_calls("hello").is_none());
        assert!(parse_tool_calls(r#"{"foo":1}"#).is_none());
    }

    #[test]
    fn tools_block_omitted_when_none() {
        let tools = vec![ToolSpec {
            name: "f".into(),
            description: "".into(),
            parameters: json!({}),
        }];
        assert!(tools_system_block(&tools, &ToolChoice::None).is_none());
        assert!(tools_system_block(&tools, &ToolChoice::Required)
            .unwrap()
            .contains("must call"));
    }
}
