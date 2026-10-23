//! Repo map — the codebase's shape in a token budget: files ordered by personalized
//! PageRank over the file reference graph ([`crate::graph`]), each with its most-referenced
//! definitions and their signature lines. This is what an agent gets on turn one instead
//! of exploring blind.

use crate::graph::{module_name, FileGraph};
use crate::refs::{self, RefKind};
use crate::store::{GraphRows, SymbolRow};
use crate::terms;
use std::collections::{HashMap, HashSet};

/// Rough chars-per-token for code; the map is trimmed to `max_tokens * CHARS_PER_TOKEN`.
const CHARS_PER_TOKEN: usize = 4;
const MAX_SYMBOLS_PER_FILE: usize = 8;
/// Query words shorter than this never name a file (`a`, `fix`, `the`).
const MIN_QUERY_STEM: usize = 4;

#[derive(Debug, Clone, Copy, Default)]
pub struct RepoMapOptions<'a> {
    pub max_tokens: usize,
    /// Paths (or path prefixes) the caller is working in; their files rank first and
    /// personalize the PageRank (their neighbourhood ranks next).
    pub focus: &'a [String],
    /// The task or query in hand. Files it names (a path, a file stem, or a symbol they
    /// define) personalize the PageRank the same way focus paths do.
    pub query: Option<&'a str>,
}

pub fn render(rows: &GraphRows, opts: &RepoMapOptions) -> String {
    let graph = FileGraph::build(
        rows.paths.iter().map(String::as_str).chain(rows.symbols.iter().map(|s| s.path.as_str())),
        &rows.refs,
        &rows.imports,
    );
    let rank = graph.pagerank(&personalization(&graph, &rows.symbols, opts));

    let mut by_file: HashMap<&str, Vec<(&SymbolRow, f64)>> = HashMap::new();
    for s in &rows.symbols {
        let degree = graph.index_of(&s.path).map(|i| graph.symbol_in_degree(i, &s.name)).unwrap_or(0.0);
        by_file.entry(s.path.as_str()).or_default().push((s, degree));
    }
    let mut files: Vec<(&str, bool, f64, Vec<(&SymbolRow, f64)>)> = by_file
        .into_iter()
        .map(|(path, mut syms)| {
            syms.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.line.cmp(&b.0.line)));
            let r = graph.index_of(path).map(|i| rank[i]).unwrap_or(0.0);
            (path, is_focus(path, opts.focus), r, syms)
        })
        .collect();
    files.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)).then(a.0.cmp(b.0)));

    let budget = opts.max_tokens.max(64) * CHARS_PER_TOKEN;
    let mut out = String::new();
    let mut omitted = 0usize;
    for (path, _, _, syms) in &files {
        let mut block = format!("{path}\n");
        for (s, _) in syms.iter().take(MAX_SYMBOLS_PER_FILE) {
            let indent = if s.container.is_some() { "    " } else { "  " };
            block.push_str(&format!("{indent}{}: {}\n", s.line, s.signature));
        }
        if syms.len() > MAX_SYMBOLS_PER_FILE {
            block.push_str(&format!("  … {} more\n", syms.len() - MAX_SYMBOLS_PER_FILE));
        }
        if out.len() + block.len() > budget {
            omitted += 1;
            continue;
        }
        out.push_str(&block);
    }
    if omitted > 0 {
        out.push_str(&format!("… {omitted} more files (use semantic_search to reach them)\n"));
    }
    out
}

fn is_focus(path: &str, focus: &[String]) -> bool {
    focus.iter().any(|f| path.starts_with(f.as_str()) || f.starts_with(path))
}

/// Teleport weights: 1 per file, plus — when focus paths or the query name any files —
/// a boost that gives those files together as much teleport mass as the rest of the
/// repo. A pure focus-only vector would rank everything unreachable from the focus at
/// zero; this keeps a global order while pulling the focus neighbourhood up.
fn personalization(graph: &FileGraph, symbols: &[SymbolRow], opts: &RepoMapOptions) -> Vec<f64> {
    let mut hot: HashSet<usize> =
        graph.paths().iter().enumerate().filter(|(_, p)| is_focus(p, opts.focus)).map(|(i, _)| i).collect();
    if let Some(q) = opts.query {
        hot.extend(named_files(q, graph, symbols));
    }
    let n = graph.len();
    let mut p = vec![1.0; n];
    if !hot.is_empty() {
        let boost = n as f64 / hot.len() as f64;
        for i in hot {
            p[i] += boost;
        }
    }
    p
}

/// Files a free-text task names: a path (`src/a.ts`), a file stem as a word ("the store
/// module" → `store.rs`), or a code-shaped symbol (`resolveMembership`) they define.
fn named_files(query: &str, graph: &FileGraph, symbols: &[SymbolRow]) -> HashSet<usize> {
    let mut out = HashSet::new();
    let mut wanted_symbols: HashSet<&str> = HashSet::new();
    let found = refs::extract(query);
    for r in &found {
        match r.kind {
            RefKind::Path => {
                for (i, p) in graph.paths().iter().enumerate() {
                    if p == &r.reference || p.ends_with(&format!("/{}", r.reference)) {
                        out.insert(i);
                    }
                }
            }
            RefKind::Symbol => {
                wanted_symbols.insert(r.reference.as_str());
            }
        }
    }
    let words: HashSet<String> =
        terms::identifiers(query, MIN_QUERY_STEM).map(|w| w.to_ascii_lowercase()).collect();
    for (i, p) in graph.paths().iter().enumerate() {
        if words.contains(&module_name(p)) {
            out.insert(i);
        }
    }
    for s in symbols.iter().filter(|s| wanted_symbols.contains(s.name.as_str())) {
        if let Some(i) = graph.index_of(&s.path) {
            out.insert(i);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::RefRow;

    fn sym(name: &str, path: &str) -> SymbolRow {
        SymbolRow {
            name: name.into(),
            kind: "function_declaration".into(),
            path: path.into(),
            line: 1,
            signature: format!("function {name}()"),
            container: None,
        }
    }

    fn rf(from: &str, ident: &str, to: &str) -> RefRow {
        RefRow { from: from.into(), ident: ident.into(), to: to.into() }
    }

    /// `name` used by every file in `users`, defined in `def`.
    fn uses(rows: &mut GraphRows, users: &[String], name: &str, def: &str) {
        for u in users {
            rows.refs.push(rf(u, name, def));
        }
    }

    fn pos(map: &str, path: &str) -> usize {
        map.find(&format!("{path}\n")).unwrap_or_else(|| panic!("{path} missing from map:\n{map}"))
    }

    /// The pre-graph ranking: Σ ln(1 + files mentioning the name) per file.
    fn name_count_score(rows: &GraphRows, path: &str) -> f64 {
        rows.symbols
            .iter()
            .filter(|s| s.path == path)
            .map(|s| {
                let mut files: HashSet<&str> = HashSet::new();
                for r in rows.refs.iter().filter(|r| r.ident == s.name && r.from != s.path) {
                    files.insert(&r.from);
                }
                (1.0 + files.len() as f64).ln()
            })
            .sum()
    }

    /// A hub util with ten helpers, each mentioned by two leaf scripts, vs a core module
    /// with one type that the four service files — which the whole app depends on — use.
    #[test]
    fn pagerank_ranks_core_above_hub_util() {
        let mut rows = GraphRows::default();
        for i in 0..10 {
            rows.symbols.push(sym(&format!("helper{i}"), "src/util.ts"));
        }
        rows.symbols.push(sym("Engine", "src/core.ts"));
        let leaves: Vec<String> = (0..2).map(|i| format!("scripts/leaf{i}.ts")).collect();
        let svcs: Vec<String> = (0..4).map(|i| format!("src/svc{i}.ts")).collect();
        let apps: Vec<String> = (0..8).map(|i| format!("app/page{i}.ts")).collect();
        for i in 0..10 {
            uses(&mut rows, &leaves, &format!("helper{i}"), "src/util.ts");
        }
        uses(&mut rows, &svcs, "Engine", "src/core.ts");
        for (i, svc) in svcs.iter().enumerate() {
            rows.symbols.push(sym(&format!("service{i}"), svc));
            uses(&mut rows, &apps, &format!("service{i}"), svc);
        }
        for p in leaves.iter().chain(&apps) {
            rows.symbols.push(sym("run", p));
        }
        rows.paths = rows.symbols.iter().map(|s| s.path.clone()).collect::<HashSet<_>>().into_iter().collect();

        // The old count puts the hub first…
        assert!(name_count_score(&rows, "src/util.ts") > name_count_score(&rows, "src/core.ts"));
        // …PageRank puts the module the important files depend on first.
        let map = render(&rows, &RepoMapOptions { max_tokens: 4000, ..Default::default() });
        assert!(pos(&map, "src/core.ts") < pos(&map, "src/util.ts"), "{map}");
    }

    /// Two unrelated `Config` types: only the one the core files import collects them.
    #[test]
    fn same_name_collision_resolves_to_imported_definer() {
        let mut rows = GraphRows::default();
        rows.symbols.push(sym("Config", "app/config.ts"));
        rows.symbols.push(sym("Config", "core/config.ts"));
        let users: Vec<String> = (0..3).map(|i| format!("core/user{i}.ts")).collect();
        for (i, u) in users.iter().enumerate() {
            rows.symbols.push(sym(&format!("user{i}"), u));
            rows.imports.push((u.clone(), "config".into()));
            // The join yields one row per definer of the name.
            rows.refs.push(rf(u, "Config", "app/config.ts"));
            rows.refs.push(rf(u, "Config", "core/config.ts"));
        }
        rows.paths = rows.symbols.iter().map(|s| s.path.clone()).collect();

        // The old count cannot tell them apart (a tie, broken alphabetically → app first).
        assert_eq!(name_count_score(&rows, "app/config.ts"), name_count_score(&rows, "core/config.ts"));
        let map = render(&rows, &RepoMapOptions { max_tokens: 4000, ..Default::default() });
        assert!(pos(&map, "core/config.ts") < pos(&map, "app/config.ts"), "{map}");

        let g = FileGraph::build(rows.paths.iter().map(String::as_str), &rows.refs, &rows.imports);
        assert_eq!(g.symbol_in_degree(g.index_of("app/config.ts").unwrap(), "Config"), 0.0);
        assert_eq!(g.symbol_in_degree(g.index_of("core/config.ts").unwrap(), "Config"), 3.0);
    }

    #[test]
    fn symbols_within_a_file_order_by_in_degree() {
        let mut rows = GraphRows::default();
        rows.symbols.push(SymbolRow { line: 1, ..sym("rare", "lib.ts") });
        rows.symbols.push(SymbolRow { line: 9, ..sym("popular", "lib.ts") });
        let users: Vec<String> = (0..3).map(|i| format!("u{i}.ts")).collect();
        uses(&mut rows, &users, "popular", "lib.ts");
        uses(&mut rows, &users[..1], "rare", "lib.ts");
        rows.paths = users.clone();
        let map = render(&rows, &RepoMapOptions { max_tokens: 1000, ..Default::default() });
        assert!(map.find("popular").unwrap() < map.find("rare").unwrap(), "{map}");
    }

    #[test]
    fn budget_respected() {
        let mut rows = GraphRows::default();
        rows.symbols = (0..50).map(|i| sym(&format!("f{i}"), &format!("dir/file{i}.ts"))).collect();
        let tiny = render(&rows, &RepoMapOptions { max_tokens: 64, ..Default::default() });
        assert!(tiny.len() <= 64 * CHARS_PER_TOKEN + 80);
        assert!(tiny.contains("more files"));
    }

    #[test]
    fn focus_ranks_first() {
        let mut rows = GraphRows::default();
        rows.symbols = vec![sym("core", "a/core.ts"), sym("leaf", "b/leaf.ts")];
        let users: Vec<String> = (0..20).map(|i| format!("a/u{i}.ts")).collect();
        uses(&mut rows, &users, "core", "a/core.ts");
        let focus = vec!["b/".to_string()];
        let map = render(&rows, &RepoMapOptions { max_tokens: 1000, focus: &focus, query: None });
        assert!(pos(&map, "b/leaf.ts") < pos(&map, "a/core.ts"));
    }

    #[test]
    fn query_names_lift_files() {
        let mut rows = GraphRows::default();
        rows.symbols = vec![
            sym("alpha", "src/alpha.ts"),
            sym("billingTotals", "src/money.ts"),
            sym("zeta", "src/store.ts"),
            sym("omega", "src/omega.ts"),
        ];
        // alpha and omega are both referenced once; money/store not at all.
        rows.refs = vec![rf("x.ts", "alpha", "src/alpha.ts"), rf("y.ts", "omega", "src/omega.ts")];
        let plain = render(&rows, &RepoMapOptions { max_tokens: 1000, ..Default::default() });
        assert!(pos(&plain, "src/alpha.ts") < pos(&plain, "src/store.ts"));

        let q = "fix the store eviction, then check `billingTotals`";
        let map = render(&rows, &RepoMapOptions { max_tokens: 1000, query: Some(q), ..Default::default() });
        assert!(pos(&map, "src/store.ts") < pos(&map, "src/alpha.ts"), "{map}");
        assert!(pos(&map, "src/money.ts") < pos(&map, "src/omega.ts"), "{map}");
    }
}
