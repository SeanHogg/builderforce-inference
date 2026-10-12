//! Command implementations dispatched from clap.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use bfi_runtime::{aliases, model_path, models_dir, Engine, SampleParams};
use bfi_server::ServerConfig;

use crate::cli::Commands;
use crate::fit::{self, cpu_budget_bytes, device_label};
use crate::hf::{self, dest_path, download_resumable, parse_spec};
use crate::ollama::{self, ollama_models_dir};
use crate::store::{format_bytes, list_store, remove_from_store};

pub fn dispatch(cmd: Commands) -> Result<()> {
    match cmd {
        Commands::Pull { spec } => cmd_pull(&spec),
        Commands::Run { model, prompt } => cmd_run(&model, prompt.as_deref()),
        Commands::Serve {
            port,
            auth,
            models_dir,
        } => cmd_serve(port, auth, models_dir),
        Commands::Ps { port } => cmd_ps(port),
        Commands::Fit { model, ctx, json } => cmd_fit(&model, ctx, json),
        Commands::Rm { model } => cmd_rm(&model),
        Commands::Ls => cmd_ls(),
        Commands::Bench { model, pp, tg } => cmd_bench(&model, pp, tg),
        Commands::BenchKernel {
            gguf,
            n_out,
            n_in,
            iters,
            dtype,
        } => crate::bench_kernel::run(gguf, n_out, n_in, iters, &dtype),
        Commands::ImportOllama { ollama_dir } => cmd_import_ollama(ollama_dir),
    }
}

fn cmd_pull(spec_raw: &str) -> Result<()> {
    let aliases = aliases().unwrap_or_default();
    let spec = parse_spec(spec_raw, &aliases);
    if hf::looks_like_path(&spec.repo) {
        bail!("pull expects a Hugging Face repo[:quant] or alias, got {}", spec.repo);
    }
    let remote = hf::resolve_remote(&spec)?;
    let filename = Path::new(&remote.path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.gguf");
    let dest = dest_path(&models_dir(), &spec, filename);
    let path = download_resumable(&remote, &dest)?;
    println!("pulled {}", path.display());
    Ok(())
}

fn cmd_run(model: &str, prompt: Option<&str>) -> Result<()> {
    let path = resolve_local_model(model)?;
    eprintln!("loading {}", path.display());
    let mut engine = Engine::load(&path).with_context(|| format!("load {}", path.display()))?;
    match prompt {
        Some(p) => {
            let out = engine.generate(p, SampleParams::default())?;
            println!("{out}");
        }
        None => interactive(&mut engine)?,
    }
    Ok(())
}

fn interactive(engine: &mut Engine) -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut lines = stdin.lock().lines();
    eprintln!("bfi chat — empty line or /bye to quit");
    loop {
        write!(stdout, "> ")?;
        stdout.flush()?;
        let Some(line) = lines.next() else {
            break;
        };
        let line = line?;
        let t = line.trim();
        if t.is_empty() || t == "/bye" || t == "/exit" {
            break;
        }
        let out = engine.generate(t, SampleParams::default())?;
        println!("{out}");
    }
    Ok(())
}

fn cmd_serve(port: u16, auth: bool, models_dir: Option<PathBuf>) -> Result<()> {
    if let Some(dir) = models_dir {
        std::env::set_var("BFI_MODELS_DIR", &dir);
    }
    let api_key = if auth {
        Some(api_key_for_auth())
    } else {
        None
    };
    let cfg = ServerConfig {
        host: "127.0.0.1".into(),
        port,
        api_key,
        model_path: None,
    };
    eprintln!("bfi serve on 127.0.0.1:{port}");
    let rt = tokio::runtime::Runtime::new().context("tokio runtime")?;
    rt.block_on(bfi_server::serve(cfg))?;
    Ok(())
}

fn api_key_for_auth() -> String {
    if let Ok(k) = std::env::var("BFI_API_KEY") {
        if !k.is_empty() {
            return k;
        }
    }
    use sha2::{Digest, Sha256};
    let raw = format!(
        "bfi-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let mut h = Sha256::new();
    h.update(raw.as_bytes());
    let k = hex::encode(&h.finalize()[..16]);
    eprintln!("bfi: --auth enabled, API key: {k} (set BFI_API_KEY to pin it)");
    k
}

fn cmd_ps(port: u16) -> Result<()> {
    let url = format!("http://127.0.0.1:{port}/api/ps");
    let body: serde_json::Value = match ureq::get(&url).call() {
        Ok(resp) => resp.into_json().context("/api/ps JSON")?,
        Err(ureq::Error::Status(status, resp)) => {
            bail!("server HTTP {status}: {}", resp.into_string().unwrap_or_default());
        }
        Err(e) => {
            eprintln!("bfi ps: server not running at {url} ({e})");
            print_ps_table(&[]);
            return Ok(());
        }
    };
    let models = body
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let ram = sys.used_memory();
    let device = device_label();

    println!(
        "{:<24} {:<8} {:>10} {:>10} {:>8}",
        "NAME", "DEVICE", "RAM", "VRAM", "TOK/S"
    );
    if models.is_empty() {
        println!("(no models loaded)");
        return Ok(());
    }
    for m in models {
        let name = m
            .get("name")
            .or_else(|| m.get("model"))
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let tok = m
            .get("tokPerSec")
            .or_else(|| m.get("tok_s"))
            .and_then(|v| v.as_f64())
            .map(|t| format!("{t:.1}"))
            .unwrap_or_else(|| "-".into());
        let vram = m
            .get("vramBytes")
            .and_then(|v| v.as_u64())
            .map(format_bytes)
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<24} {:<8} {:>10} {:>10} {:>8}",
            name,
            m.get("device")
                .and_then(|v| v.as_str())
                .unwrap_or(&device),
            format_bytes(ram),
            vram,
            tok
        );
    }
    Ok(())
}

fn print_ps_table(rows: &[String]) {
    println!(
        "{:<24} {:<8} {:>10} {:>10} {:>8}",
        "NAME", "DEVICE", "RAM", "VRAM", "TOK/S"
    );
    if rows.is_empty() {
        println!("(no models loaded)");
    }
}

fn cmd_fit(model: &str, ctx: usize, json: bool) -> Result<()> {
    let header = fit::resolve_header(model)?;
    let report = fit::fit_from_header(&header, ctx, cpu_budget_bytes(), &device_label())?;
    fit::print_fit(&report, json)
}

fn cmd_ls() -> Result<()> {
    let dir = models_dir();
    let models = list_store(&dir)?;
    if models.is_empty() {
        return Ok(());
    }
    for m in models {
        println!("{:<32} {}", m.name, format_bytes(m.size));
    }
    Ok(())
}

fn cmd_rm(name: &str) -> Result<()> {
    let path = remove_from_store(&models_dir(), name)?;
    println!("removed {}", path.display());
    Ok(())
}

fn cmd_bench(model: &str, pp: usize, tg: usize) -> Result<()> {
    let path = resolve_local_model(model)?;
    let mut engine = Engine::load(&path).with_context(|| format!("load {}", path.display()))?;
    let prompt: String = "The ".repeat(pp.max(1));

    let mut pp_params = SampleParams::default();
    pp_params.max_tokens = 1;
    pp_params.temperature = 0.0;
    let t0 = Instant::now();
    let _ = engine.generate(&prompt, pp_params)?;
    let pp_s = t0.elapsed().as_secs_f64().max(1e-9);
    let pp_ts = pp as f64 / pp_s;

    let mut tg_params = SampleParams::default();
    tg_params.max_tokens = tg as u32;
    tg_params.temperature = 0.0;
    let t1 = Instant::now();
    let _ = engine.generate("Hi", tg_params)?;
    let tg_s = t1.elapsed().as_secs_f64().max(1e-9);
    let tg_ts = tg as f64 / tg_s;

    let size = path.metadata().map(|m| m.len()).unwrap_or(0);
    println!(
        "| model | size | backend | test | t/s |"
    );
    println!(
        "| {} | {} | {} | pp{pp} | {pp_ts:.2} |",
        path.display(),
        format_bytes(size),
        device_label()
    );
    println!(
        "| {} | {} | {} | tg{tg} | {tg_ts:.2} |",
        path.display(),
        format_bytes(size),
        device_label()
    );
    Ok(())
}

fn cmd_import_ollama(ollama_dir: Option<PathBuf>) -> Result<()> {
    let src = ollama_dir.unwrap_or_else(ollama_models_dir);
    let n = ollama::import_ollama(&src, &models_dir())?;
    println!("imported {n} model(s) from {}", src.display());
    Ok(())
}

fn resolve_local_model(model: &str) -> Result<PathBuf> {
    let p = PathBuf::from(model);
    if p.exists() {
        return Ok(p);
    }
    let via_runtime = model_path(model);
    if via_runtime.exists() {
        return Ok(via_runtime);
    }
    let entries = list_store(&models_dir())?;
    if let Some(hit) = entries.iter().find(|e| e.name == model) {
        return Ok(hit.path.clone());
    }
    Err(anyhow!(
        "model not found: {model} (try `bfi pull` or `bfi ls`)"
    ))
}
