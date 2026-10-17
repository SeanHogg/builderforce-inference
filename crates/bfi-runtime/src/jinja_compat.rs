//! Python-method shims for GGUF chat templates.
//!
//! HF chat templates are Jinja2 and call Python `str` / `dict` methods
//! (`content.startswith('<tool_response>')`, `content.split('</think>')`,
//! `tool.items()`), which minijinja does not provide. This is the subset the
//! Qwen2/Qwen3 (and Bonsai) templates use, plus `raise_exception`.

use minijinja::value::{Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind, State};

/// Register the shims on `env`.
pub fn install(env: &mut Environment<'_>) {
    env.set_unknown_method_callback(unknown_method);
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
}

fn arg_str(args: &[Value], i: usize) -> Option<String> {
    args.get(i).and_then(|v| v.as_str().map(str::to_string))
}

fn unknown_method(_: &State, value: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    if let Some(s) = value.as_str() {
        return str_method(s, method, args);
    }
    if value.kind() == ValueKind::Map {
        return map_method(value, method, args);
    }
    Err(Error::from(ErrorKind::UnknownMethod))
}

fn str_method(s: &str, method: &str, args: &[Value]) -> Result<Value, Error> {
    let chars = arg_str(args, 0);
    let strip_set = |c: char| match &chars {
        Some(set) => set.contains(c),
        None => c.is_whitespace(),
    };
    Ok(match method {
        "startswith" => Value::from(arg_str(args, 0).is_some_and(|p| s.starts_with(&p))),
        "endswith" => Value::from(arg_str(args, 0).is_some_and(|p| s.ends_with(&p))),
        "strip" => Value::from(s.trim_matches(strip_set)),
        "lstrip" => Value::from(s.trim_start_matches(strip_set)),
        "rstrip" => Value::from(s.trim_end_matches(strip_set)),
        "lower" => Value::from(s.to_lowercase()),
        "upper" => Value::from(s.to_uppercase()),
        "replace" => match (arg_str(args, 0), arg_str(args, 1)) {
            (Some(a), Some(b)) => Value::from(s.replace(&a, &b)),
            _ => return Err(Error::from(ErrorKind::MissingArgument)),
        },
        "split" => {
            let parts: Vec<Value> = match arg_str(args, 0) {
                Some(sep) if !sep.is_empty() => s.split(sep.as_str()).map(Value::from).collect(),
                _ => s.split_whitespace().map(Value::from).collect(),
            };
            Value::from(parts)
        }
        _ => return Err(Error::from(ErrorKind::UnknownMethod)),
    })
}

fn map_method(map: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    let keys: Vec<Value> = map.try_iter()?.collect();
    Ok(match method {
        "items" => Value::from(
            keys.into_iter()
                .map(|k| {
                    let v = map.get_item(&k).unwrap_or_default();
                    Value::from(vec![k, v])
                })
                .collect::<Vec<_>>(),
        ),
        "keys" => Value::from(keys),
        "values" => Value::from(
            keys.iter()
                .map(|k| map.get_item(k).unwrap_or_default())
                .collect::<Vec<_>>(),
        ),
        "get" => {
            let key = args.first().cloned().unwrap_or_default();
            match map.get_item(&key) {
                Ok(v) if !v.is_undefined() => v,
                _ => args.get(1).cloned().unwrap_or(Value::from(())),
            }
        }
        _ => return Err(Error::from(ErrorKind::UnknownMethod)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::context;

    fn render(src: &str, ctx: Value) -> String {
        let mut env = Environment::new();
        install(&mut env);
        env.add_template("t", src).unwrap();
        env.get_template("t").unwrap().render(ctx).unwrap()
    }

    #[test]
    fn string_methods() {
        let out = render(
            "{{ s.startswith('<a>') }}|{{ s.split('</think>')[-1].lstrip('\\n') }}|{{ ' x '.strip() }}",
            context! { s => "<a>think</think>\nanswer" },
        );
        // Jinja2 (and minijinja) print Python booleans: `True`.
        assert_eq!(out, "True|answer|x");
    }

    #[test]
    fn dict_items_and_get() {
        let out = render(
            "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}{{ d.get('zz', 'dflt') }}",
            context! { d => context! { a => 1 } },
        );
        assert_eq!(out, "a=1;dflt");
    }
}
