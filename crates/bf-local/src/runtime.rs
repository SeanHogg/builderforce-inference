//! The local runtime Synapse manages: `bfi`.
//!
//! Synapse ships `bfi` as a sidecar and supervises `bfi serve --auth --models-dir
//! ~/.builderforce/models` as a child process. It reads `~/.builderforce/bfi.json`
//! (port, pid, version, token), health-checks `/health`, and restarts with backoff
//! on crash — at most 5 times in 10 minutes — then surfaces an error. The child is
//! stopped when this handle is dropped (Synapse quitting).
//!
//! [`version`], [`installed`], [`pull`] and [`delete`] speak bfi's Ollama-compatible
//! `/api/*` (with the discovery token when `--auth` is on).

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const PROBE_TIMEOUT: Duration = Duration::from_millis(1200);
const CALL_TIMEOUT: Duration = Duration::from_secs(30);
const PULL_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);
const READY_POLL: Duration = Duration::from_millis(100);
const WATCH_POLL: Duration = Duration::from_millis(150);
const DEFAULT_READY: Duration = Duration::from_secs(30);
const MAX_CRASHES: usize = 5;
const CRASH_WINDOW: Duration = Duration::from_secs(10 * 60);

static AUTH_TOKEN: Mutex<Option<String>> = Mutex::new(None);
static LAST_VERSION: Mutex<Option<String>> = Mutex::new(None);

/// A model installed in the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Installed {
    pub name: String,
    pub size: u64,
    pub parameter_size: Option<String>,
    pub quantization: Option<String>,
}

/// Progress of a pull, as the runtime reports it.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PullProgress {
    pub status: String,
    pub total: Option<u64>,
    pub completed: Option<u64>,
}

/// `~/.builderforce/bfi.json` as `bfi serve` writes it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BfiJson {
    pub port: u16,
    #[serde(default)]
    pub pid: u32,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub token: String,
}

/// How the supervisor reacts to a child crash.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    max_in_window: usize,
    window: Duration,
    delay: Option<Duration>,
    crashes: Vec<Instant>,
}

impl RestartPolicy {
    pub fn standard() -> Self {
        Self { max_in_window: MAX_CRASHES, window: CRASH_WINDOW, delay: None, crashes: Vec::new() }
    }

    /// Test helper: fixed delay (including zero) and a custom give-up limit.
    pub fn with_limit(max_in_window: usize, window: Duration, delay: Duration) -> Self {
        Self { max_in_window, window, delay: Some(delay), crashes: Vec::new() }
    }

    /// Record a crash at `now`. `Some(wait)` means spawn again after `wait`; `None` means give up.
    pub fn on_crash(&mut self, now: Instant) -> Option<Duration> {
        self.crashes.retain(|t| now.saturating_duration_since(*t) < self.window);
        self.crashes.push(now);
        if self.crashes.len() > self.max_in_window {
            return None;
        }
        if let Some(d) = self.delay {
            return Some(d);
        }
        let n = self.crashes.len() - 1;
        Some(Duration::from_secs(1u64 << n.min(6)))
    }
}

/// Where and how to spawn `bfi serve`. Tests override every field; production uses [`Default`].
pub struct Config {
    pub bin: PathBuf,
    pub args: Vec<String>,
    pub discovery: PathBuf,
    pub models_dir: PathBuf,
    pub ready_timeout: Duration,
    pub policy: RestartPolicy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bin: bfi_bin(),
            args: Vec::new(),
            discovery: discovery_path(),
            models_dir: models_dir(),
            ready_timeout: DEFAULT_READY,
            policy: RestartPolicy::standard(),
        }
    }
}

struct Inner {
    stop: AtomicBool,
    base: Mutex<String>,
    error: Mutex<Option<String>>,
    child: Mutex<Option<Child>>,
    /// True after this process merged `models` into desktop.json; stop withdraws only then.
    published: AtomicBool,
}

/// Live supervisor. Dropping it stops the child.
pub struct Handle {
    inner: Arc<Inner>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Handle {
    /// Start `bfi serve` as a supervised child.
    pub fn start() -> Self {
        Self::start_with(Config::default())
    }

    pub fn start_with(cfg: Config) -> Self {
        let inner = Arc::new(Inner {
            stop: AtomicBool::new(false),
            base: Mutex::new(String::new()),
            error: Mutex::new(None),
            child: Mutex::new(None),
            published: AtomicBool::new(false),
        });
        let run_inner = inner.clone();
        let thread = thread::Builder::new().name("synapse-bfi".into()).spawn(move || supervise(run_inner, cfg)).ok();
        Self { inner, thread: Mutex::new(thread) }
    }

    pub fn base(&self) -> String {
        self.inner.base.lock().unwrap().clone()
    }

    pub fn error(&self) -> Option<String> {
        self.inner.error.lock().unwrap().clone()
    }

    /// Stop the child and the supervisor thread. Synapse calls this on quit.
    pub fn stop(&self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        reap(&self.inner);
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
        *AUTH_TOKEN.lock().unwrap() = None;
        *LAST_VERSION.lock().unwrap() = None;
        if self.inner.published.swap(false, Ordering::SeqCst) {
            let _ = crate::desktop::withdraw_default();
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// `Authorization: Bearer …` value for the live child, if `--auth` issued a token.
pub fn authorization() -> Option<String> {
    AUTH_TOKEN.lock().unwrap().as_ref().filter(|t| !t.is_empty()).map(|t| format!("Bearer {t}"))
}

pub fn discovery_path() -> PathBuf {
    if let Ok(p) = std::env::var("BFI_DISCOVERY") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    builderforce_dir().join("bfi.json")
}

pub fn models_dir() -> PathBuf {
    if let Ok(p) = std::env::var("BFI_MODELS_DIR") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    builderforce_dir().join("models")
}

fn builderforce_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".builderforce")
}

/// Tauri sidecar on-disk name: `bfi-<target-triple>[.exe]`.
pub fn sidecar_file_name() -> String {
    let triple = env!("BFI_TARGET_TRIPLE");
    if cfg!(windows) {
        format!("bfi-{triple}.exe")
    } else {
        format!("bfi-{triple}")
    }
}

/// A `bfi` binary shipped next to this executable (Tauri sidecar or sibling).
pub fn sidecar_bfi() -> Option<PathBuf> {
    let name = if cfg!(windows) { "bfi.exe" } else { "bfi" };
    let sidecar = sidecar_file_name();
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    for n in [name, sidecar.as_str()] {
        let p = dir.join(n);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

pub fn bfi_bin() -> PathBuf {
    if let Ok(p) = std::env::var("BFI_BIN") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if let Some(p) = sidecar_bfi() {
        return p;
    }
    PathBuf::from(if cfg!(windows) { "bfi.exe" } else { "bfi" })
}

pub fn serve_args(models_dir: &Path) -> Vec<String> {
    vec!["serve".into(), "--auth".into(), "--models-dir".into(), models_dir.display().to_string()]
}

pub fn read_discovery(path: &Path) -> Result<BfiJson> {
    let raw = std::fs::read_to_string(path).map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
    Ok(serde_json::from_str(&raw)?)
}

/// `GET /health` — unauthenticated, as the server publishes it.
pub fn health_ok(base: &str) -> bool {
    if base.is_empty() {
        return false;
    }
    ureq::get(&format!("{base}/health")).timeout(PROBE_TIMEOUT).call().is_ok()
}

fn authed(req: ureq::Request) -> ureq::Request {
    match authorization() {
        Some(h) => req.set("Authorization", &h),
        None => req,
    }
}

/// The runtime's version, or `None` when nothing answers.
pub fn version(base: &str) -> Option<String> {
    if base.is_empty() {
        return None;
    }
    if let Some(v) = http_version(base) {
        return Some(v);
    }
    LAST_VERSION.lock().unwrap().clone().filter(|s| !s.is_empty())
}

fn http_version(base: &str) -> Option<String> {
    let v: Value = authed(ureq::get(&format!("{base}/api/version")).timeout(PROBE_TIMEOUT)).call().ok()?.into_json().ok()?;
    v.get("version").and_then(Value::as_str).map(str::to_string)
}

pub fn installed(base: &str) -> Result<Vec<Installed>> {
    let v: Value = authed(ureq::get(&format!("{base}/api/tags")).timeout(CALL_TIMEOUT)).call()?.into_json()?;
    Ok(parse_tags(&v))
}

fn parse_tags(v: &Value) -> Vec<Installed> {
    v.get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| {
            Some(Installed {
                name: m.get("name").or_else(|| m.get("model"))?.as_str()?.to_string(),
                size: m.get("size").and_then(Value::as_u64).unwrap_or(0),
                parameter_size: m.pointer("/details/parameter_size").and_then(Value::as_str).map(str::to_string),
                quantization: m.pointer("/details/quantization_level").and_then(Value::as_str).map(str::to_string),
            })
        })
        .collect()
}

fn pull_line(line: &str) -> Result<Option<PullProgress>> {
    let Ok(v) = serde_json::from_str::<Value>(line) else { return Ok(None) };
    if let Some(e) = v.get("error").and_then(Value::as_str) {
        bail!("{e}");
    }
    Ok(Some(PullProgress {
        status: v.get("status").and_then(Value::as_str).unwrap_or("").to_string(),
        total: v.get("total").and_then(Value::as_u64),
        completed: v.get("completed").and_then(Value::as_u64),
    }))
}

/// Download `model` into the runtime, telling `on_progress` as it goes.
pub fn pull(base: &str, model: &str, mut on_progress: impl FnMut(&PullProgress)) -> Result<()> {
    let resp = authed(ureq::post(&format!("{base}/api/pull")).timeout(PULL_TIMEOUT)).send_json(json!({ "model": model, "stream": true }))?;
    let mut last = None;
    for line in BufReader::new(resp.into_reader()).lines() {
        if let Some(p) = pull_line(&line?)? {
            on_progress(&p);
            last = Some(p.status);
        }
    }
    match last.as_deref() {
        Some("success") => Ok(()),
        other => Err(anyhow!("the download of {model} ended without finishing ({})", other.unwrap_or("no progress"))),
    }
}

pub fn delete(base: &str, model: &str) -> Result<()> {
    authed(ureq::request("DELETE", &format!("{base}/api/delete")).timeout(CALL_TIMEOUT)).send_json(json!({ "model": model }))?;
    Ok(())
}

fn spawn_bfi(cfg: &Config) -> std::io::Result<Child> {
    std::fs::create_dir_all(&cfg.models_dir)?;
    let args = if cfg.args.is_empty() { serve_args(&cfg.models_dir) } else { cfg.args.clone() };
    let mut cmd = Command::new(&cfg.bin);
    cmd.args(&args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    cmd.spawn()
}

fn apply_discovery(inner: &Inner, d: &BfiJson) {
    *inner.base.lock().unwrap() = format!("http://127.0.0.1:{}", d.port);
    *AUTH_TOKEN.lock().unwrap() = if d.token.is_empty() { None } else { Some(d.token.clone()) };
    *LAST_VERSION.lock().unwrap() = if d.version.is_empty() { None } else { Some(d.version.clone()) };
    if crate::desktop::publish_default(&crate::desktop::from_bfi(d)).is_ok() {
        inner.published.store(true, Ordering::SeqCst);
    }
}

fn wait_ready(child: &mut Child, cfg: &Config, stop: &AtomicBool) -> Result<BfiJson> {
    let deadline = Instant::now() + cfg.ready_timeout;
    while Instant::now() < deadline {
        if stop.load(Ordering::Relaxed) {
            bail!("stopped");
        }
        if let Some(status) = child.try_wait()? {
            bail!("bfi exited before becoming ready ({status})");
        }
        if let Ok(d) = read_discovery(&cfg.discovery) {
            if d.pid == child.id() || d.pid == 0 {
                let base = format!("http://127.0.0.1:{}", d.port);
                if health_ok(&base) {
                    return Ok(d);
                }
            }
        }
        thread::sleep(READY_POLL);
    }
    bail!("bfi did not become ready (no healthy server at {})", cfg.discovery.display())
}

fn watch_until_exit(inner: &Inner) {
    loop {
        if inner.stop.load(Ordering::Relaxed) {
            return;
        }
        let mut slot = inner.child.lock().unwrap();
        if let Some(c) = slot.as_mut() {
            match c.try_wait() {
                Ok(Some(_)) | Err(_) => {
                    slot.take();
                    return;
                }
                Ok(None) => {}
            }
        } else {
            return;
        }
        drop(slot);
        thread::sleep(WATCH_POLL);
    }
}

fn reap(inner: &Inner) {
    if let Some(mut c) = inner.child.lock().unwrap().take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

fn supervise(inner: Arc<Inner>, cfg: Config) {
    let mut policy = cfg.policy.clone();
    loop {
        if inner.stop.load(Ordering::Relaxed) {
            reap(&inner);
            return;
        }
        match spawn_bfi(&cfg) {
            Ok(mut child) => match wait_ready(&mut child, &cfg, &inner.stop) {
                Ok(disc) => {
                    apply_discovery(&inner, &disc);
                    *inner.error.lock().unwrap() = None;
                    *inner.child.lock().unwrap() = Some(child);
                    watch_until_exit(&inner);
                    if inner.stop.load(Ordering::Relaxed) {
                        reap(&inner);
                        return;
                    }
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    if inner.stop.load(Ordering::Relaxed) {
                        return;
                    }
                    *inner.error.lock().unwrap() = Some(format!("{e:#}"));
                }
            },
            Err(e) => {
                *inner.error.lock().unwrap() = Some(format!("could not start bfi: {e:#}"));
                while !inner.stop.load(Ordering::Relaxed) {
                    thread::sleep(WATCH_POLL);
                }
                return;
            }
        }
        match policy.on_crash(Instant::now()) {
            None => {
                *inner.error.lock().unwrap() = Some("the local engine crashed 5 times in 10 minutes".into());
                return;
            }
            Some(delay) => {
                let until = Instant::now() + delay;
                while Instant::now() < until {
                    if inner.stop.load(Ordering::Relaxed) {
                        return;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crashing_command() -> (PathBuf, Vec<String>) {
        if cfg!(windows) {
            (PathBuf::from("cmd.exe"), vec!["/C".into(), "exit 1".into()])
        } else {
            (PathBuf::from("/bin/false"), vec![])
        }
    }

    fn sleeping_command() -> (PathBuf, Vec<String>) {
        if cfg!(windows) {
            (PathBuf::from("ping.exe"), vec!["-n".into(), "30".into(), "127.0.0.1".into()])
        } else {
            (PathBuf::from("sleep"), vec!["30".into()])
        }
    }

    #[test]
    fn serve_args_pass_auth_and_models_dir() {
        let args = serve_args(Path::new("/home/u/.builderforce/models"));
        assert_eq!(args, ["serve", "--auth", "--models-dir", "/home/u/.builderforce/models"]);
    }

    #[test]
    fn fifth_crash_still_restarts_sixth_gives_up() {
        let mut p = RestartPolicy::standard();
        let t0 = Instant::now();
        for i in 0..5 {
            let d = p.on_crash(t0 + Duration::from_secs(i as u64 * 10));
            assert_eq!(d, Some(Duration::from_secs(1u64 << i)), "crash {}", i + 1);
        }
        assert_eq!(p.on_crash(t0 + Duration::from_secs(50)), None);
    }

    #[test]
    fn crashes_outside_the_window_do_not_count() {
        let mut p = RestartPolicy::standard();
        let t0 = Instant::now();
        for i in 0..5 {
            assert!(p.on_crash(t0 + Duration::from_secs(i as u64)).is_some());
        }
        assert_eq!(p.on_crash(t0 + Duration::from_secs(30)), None);
        let mut q = RestartPolicy::standard();
        for i in 0..5 {
            assert!(q.on_crash(t0 + Duration::from_secs(i as u64)).is_some());
        }
        assert!(q.on_crash(t0 + CRASH_WINDOW + Duration::from_secs(1)).is_some());
    }

    #[test]
    fn discovery_reads_bfi_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bfi.json");
        std::fs::write(&path, r#"{"port":11440,"pid":9,"version":"1.0","token":"abc"}"#).unwrap();
        let d = read_discovery(&path).unwrap();
        assert_eq!(d, BfiJson { port: 11440, pid: 9, version: "1.0".into(), token: "abc".into() });
    }

    #[test]
    fn discovery_missing_is_error() {
        assert!(read_discovery(Path::new("/no/such/bfi.json")).is_err());
    }

    #[test]
    fn health_check_fails_when_nothing_answers() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        drop(server);
        assert!(!health_ok(&format!("http://127.0.0.1:{port}")));
        assert!(!health_ok(""));
    }

    #[test]
    fn health_check_fails_on_error_status() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        std::thread::spawn(move || {
            for req in server.incoming_requests() {
                let _ = req.respond(tiny_http::Response::from_string("no").with_status_code(tiny_http::StatusCode(500)));
            }
        });
        assert!(!health_ok(&format!("http://127.0.0.1:{port}")));
    }

    #[test]
    fn health_check_ok_on_200() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        std::thread::spawn(move || {
            for req in server.incoming_requests() {
                let _ = req.respond(tiny_http::Response::from_string(r#"{"status":"ok","engine":"bfi"}"#));
            }
        });
        assert!(health_ok(&format!("http://127.0.0.1:{port}")));
    }

    #[test]
    fn wait_ready_fails_when_discovery_points_at_an_unhealthy_server() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        std::thread::spawn(move || {
            for req in server.incoming_requests() {
                let _ = req.respond(tiny_http::Response::from_string("no").with_status_code(tiny_http::StatusCode(500)));
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bfi.json");
        std::fs::write(&path, format!(r#"{{"port":{port},"pid":0,"version":"x","token":""}}"#)).unwrap();
        let d = read_discovery(&path).unwrap();
        let base = format!("http://127.0.0.1:{}", d.port);
        assert!(!health_ok(&base), "500 /health must not count as ready");
    }

    #[test]
    fn installed_models_and_pull_lines_parse() {
        let tags = json!({ "models": [{ "name": "qwen2.5:7b-instruct-q4_K_M", "size": 4683087332u64, "details": { "parameter_size": "7.6B", "quantization_level": "Q4_K_M" } }, { "nope": 1 }] });
        let got = parse_tags(&tags);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(pull_line(r#"{"status":"downloading","total":10,"completed":4}"#).unwrap().unwrap().completed, Some(4));
        assert!(pull_line(r#"{"error":"pull model manifest: file does not exist"}"#).is_err());
        assert_eq!(pull_line("not json").unwrap(), None);
    }

    #[test]
    fn supervisor_gives_up_after_max_crashes_of_a_mock_child() {
        let dir = tempfile::tempdir().unwrap();
        let (bin, args) = crashing_command();
        let h = Handle::start_with(Config {
            bin,
            args,
            discovery: dir.path().join("bfi.json"),
            models_dir: dir.path().join("models"),
            ready_timeout: Duration::from_millis(200),
            policy: RestartPolicy::with_limit(5, Duration::from_secs(60), Duration::ZERO),
        });
        let deadline = Instant::now() + Duration::from_secs(8);
        let mut err = None;
        while Instant::now() < deadline {
            err = h.error();
            if err.as_deref().is_some_and(|e| e.contains("crashed")) {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        let err = err.expect("supervisor should give up");
        assert!(err.contains("crashed"), "{err}");
        drop(h);
    }

    #[test]
    fn supervisor_reports_when_a_mock_child_never_becomes_healthy() {
        let dir = tempfile::tempdir().unwrap();
        let (bin, args) = sleeping_command();
        let h = Handle::start_with(Config {
            bin,
            args,
            discovery: dir.path().join("bfi.json"),
            models_dir: dir.path().join("models"),
            ready_timeout: Duration::from_millis(150),
            policy: RestartPolicy::with_limit(1, Duration::from_secs(60), Duration::ZERO),
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut err = None;
        while Instant::now() < deadline {
            err = h.error();
            if err.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        let err = err.expect("supervisor should report a health/ready failure");
        assert!(err.contains("ready") || err.contains("crashed") || err.contains("health") || err.contains("bfi"), "{err}");
        drop(h);
    }

    #[test]
    fn stop_kills_a_mock_child() {
        let dir = tempfile::tempdir().unwrap();
        let (bin, args) = sleeping_command();
        let h = Handle::start_with(Config {
            bin,
            args,
            discovery: dir.path().join("bfi.json"),
            models_dir: dir.path().join("models"),
            ready_timeout: Duration::from_secs(8),
            policy: RestartPolicy::with_limit(1, Duration::from_secs(60), Duration::ZERO),
        });
        thread::sleep(Duration::from_millis(80));
        let t0 = Instant::now();
        h.stop();
        assert!(t0.elapsed() < Duration::from_secs(3), "stop should reap the mock child promptly");
    }

    #[test]
    fn start_always_supervises_bfi() {
        // Production Handle::start() is start_with(Config::default()) — no Ollama hatch.
        // Empty args means spawn_bfi fills serve_args (auth + models-dir).
        let cfg = Config::default();
        assert!(cfg.bin.file_name().is_some());
        assert!(cfg.args.is_empty());
        let args = serve_args(&cfg.models_dir);
        assert!(args.iter().any(|a| a == "serve"));
        assert!(args.iter().any(|a| a == "--auth"));
        assert!(args.iter().any(|a| a == "--models-dir"));
    }

    #[test]
    fn sidecar_file_name_is_bfi_plus_target_triple() {
        let n = sidecar_file_name();
        assert!(n.starts_with("bfi-"), "{n}");
        if cfg!(windows) {
            assert!(n.ends_with(".exe"), "{n}");
        } else {
            assert!(!n.ends_with(".exe"), "{n}");
        }
        assert!(!n.contains("unknown") || env!("BFI_TARGET_TRIPLE") == "unknown");
    }

    #[test]
    fn bfi_bin_honours_bfi_bin_env() {
        let prev = std::env::var_os("BFI_BIN");
        std::env::set_var("BFI_BIN", "/tmp/custom-bfi");
        assert_eq!(bfi_bin(), PathBuf::from("/tmp/custom-bfi"));
        match prev {
            Some(v) => std::env::set_var("BFI_BIN", v),
            None => std::env::remove_var("BFI_BIN"),
        }
    }
}
