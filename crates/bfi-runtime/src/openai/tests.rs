//! `chat.completions` plan + turn tests.

use super::turn::partial_marker_suffix;
use super::*;
use crate::tokenize::Tokenizer;

/// Vocab where token id == index; ids 0..=4 are words, 5 is `<|im_end|>`.
fn tok() -> Tokenizer {
    Tokenizer::from_parts(
        ["Hel", "lo", " wor", "ld", "{\"name\":\"f\",\"arguments\":{}}", "<|im_end|>", "ST", "OP"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        vec![],
    )
}

fn one_hot(id: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0.0; n];
    v[id] = 10.0;
    v
}

fn plan(json: &str) -> ChatPlan {
    ChatRequest::from_json(json).unwrap().plan(None).unwrap()
}

fn run(turn: &mut ChatTurn, ids: &[usize]) -> Vec<Value> {
    let t = tok();
    let mut chunks = Vec::new();
    for &id in ids {
        let (step, c) = turn.step(&one_hot(id, 8), &t, &[]);
        chunks.extend(c.iter().map(|s| serde_json::from_str::<Value>(s).unwrap()));
        if step == Step::Done {
            break;
        }
    }
    chunks
}

fn content(chunks: &[Value]) -> String {
    chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect()
}

#[test]
fn streams_content_then_stop_on_end_of_turn() {
    let p = plan(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"temperature":0}"#);
    let mut turn = ChatTurn::new(&p, &[1, 2, 3], 100, "c1".into(), 7).unwrap();
    let chunks = run(&mut turn, &[0, 1, 2, 3, 5]);
    assert_eq!(content(&chunks), "Hello world");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    let done: Value = serde_json::from_str(&turn.completion_json()).unwrap();
    assert_eq!(done["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(done["usage"]["prompt_tokens"], 3);
    assert_eq!(done["usage"]["completion_tokens"], 4);
    assert_eq!(done["created"], 7);
}

#[test]
fn stop_string_is_held_back_and_cut() {
    let p = plan(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"temperature":0,"stop":["STOP"]}"#);
    let mut turn = ChatTurn::new(&p, &[1], 100, "c".into(), 0).unwrap();
    let chunks = run(&mut turn, &[0, 6, 7, 1]);
    assert_eq!(content(&chunks), "Hel");
    assert_eq!(turn.finish(), Some(Finish::Stop));
    assert_eq!(turn.text(), "Hel");
}

#[test]
fn stop_string_spanning_tokens_is_found_late_in_a_long_reply() {
    // Text grows a token at a time; the stop search only looks at the new tail.
    let p = plan(r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0,"stop":["loST"]}"#);
    let mut turn = ChatTurn::new(&p, &[1], 1000, "c".into(), 0).unwrap();
    let mut ids = [0usize, 1].repeat(20);
    ids.extend([6, 7]);
    run(&mut turn, &ids);
    // "…Hello" + "ST": "loST" spans the "lo" and "ST" tokens.
    assert_eq!(turn.finish(), Some(Finish::Stop));
    assert_eq!(turn.text(), "Hello".repeat(19) + "Hel");
    assert_eq!(turn.generated().len(), 41);
}

#[test]
fn max_tokens_finishes_with_length() {
    let p = plan(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}"#);
    let mut turn = ChatTurn::new(&p, &[1], 100, "c".into(), 0).unwrap();
    let chunks = run(&mut turn, &[0, 1, 2]);
    assert!(chunks.is_empty(), "stream=false emits no chunks");
    assert_eq!(turn.finish(), Some(Finish::Length));
    assert_eq!(turn.text(), "Hello");
}

#[test]
fn tool_call_is_not_streamed_as_content() {
    let p = plan(
        r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"temperature":0,
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#,
    );
    assert!(p.tools_on);
    let mut turn = ChatTurn::new(&p, &[1], 100, "c".into(), 0).unwrap();
    let chunks = run(&mut turn, &[4, 5]);
    assert_eq!(content(&chunks), "");
    assert_eq!(turn.finish(), Some(Finish::ToolCalls));
    let tc = chunks
        .iter()
        .find(|c| c["choices"][0]["delta"]["tool_calls"].is_array())
        .unwrap();
    assert_eq!(tc["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "f");
    let done: Value = serde_json::from_str(&turn.completion_json()).unwrap();
    assert_eq!(done["choices"][0]["finish_reason"], "tool_calls");
    assert!(done["choices"][0]["message"]["content"].is_null());
}

#[test]
fn prompt_that_fills_context_is_context_exceeded() {
    let p = plan(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    let err = ChatTurn::new(&p, &[1, 2, 3, 4], 4, "c".into(), 0).err().unwrap();
    assert!(err.starts_with("context_exceeded:"), "{err}");
}

#[test]
fn plan_maps_sampling_fields_and_validates() {
    let p = plan(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
            "max_tokens":5,"temperature":0.2,"top_p":0.9,"seed":3,"stop":"x",
            "stream":true,"stream_options":{"include_usage":true},
            "response_format":{"type":"json_object"}}"#,
    );
    assert_eq!(p.params.max_tokens, 5);
    assert!((p.params.temperature - 0.2).abs() < 1e-6);
    assert!((p.params.top_p - 0.9).abs() < 1e-6);
    assert_eq!(p.params.seed, 3);
    assert!(p.params.json_object);
    assert_eq!(p.stop, vec!["x".to_string()]);
    assert!(p.stream && p.include_usage);
    assert_eq!(p.prompt, "user: hi\nassistant:\n");
    assert!(ChatRequest::from_json(r#"{"messages":[]}"#).unwrap().plan(None).is_err());
    assert!(ChatRequest::from_json("{").is_err());
}

#[test]
fn plan_uses_chatml_fallback_for_unrenderable_chatml_template() {
    let req = ChatRequest::from_json(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    let p = req.plan(Some("<|im_start|>{{ broken( }}")).unwrap();
    assert_eq!(p.prompt, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
}

#[test]
fn partial_marker_holdback() {
    assert_eq!(partial_marker_suffix("abc<tool", &["<tool_call>"]), 5);
    assert_eq!(partial_marker_suffix("abc", &["<tool_call>"]), 0);
    assert_eq!(partial_marker_suffix("xxST", &["STOP"]), 2);
}
