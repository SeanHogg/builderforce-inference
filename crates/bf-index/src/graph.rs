//! The file reference graph behind the repo map, and personalized PageRank over it.
//!
//! Edge A→B carries the number of identifiers A uses that B defines. A name defined in
//! several files is split between the definers A most plausibly means — the ones its
//! imports name, then the ones in its own directory — and a file that defines a name
//! itself resolves it locally (no edge). So two unrelated `Config` types no longer each
//! collect every `Config` mention in the repo, and a file is important when IMPORTANT
//! files depend on it, not merely when many leaf files mention one of its names.

use crate::store::RefRow;
use std::collections::{HashMap, HashSet};

pub const DAMPING: f64 = 0.85;
pub const MAX_ITERS: usize = 30;
/// L1 change between iterations below which the ranks are considered converged.
pub const TOLERANCE: f64 = 1e-6;

/// Stems that name their directory's module rather than themselves.
const MODULE_INDEX_STEMS: &[&str] = &["index", "mod", "lib", "main", "__init__", "init"];

pub struct FileGraph {
    paths: Vec<String>,
    index: HashMap<String, usize>,
    /// Outgoing `(to, weight)` per file, sorted by `to`.
    out: Vec<Vec<(usize, f64)>>,
    /// Resolved references INTO `(file, symbol name)` — orders symbols within a file.
    symbol_in: HashMap<(usize, String), f64>,
}

impl FileGraph {
    /// `paths` are every node (files with no edges still rank); `refs` may name more.
    pub fn build<'a>(
        paths: impl IntoIterator<Item = &'a str>,
        refs: &[RefRow],
        imports: &[(String, String)],
    ) -> Self {
        let mut g = Self { paths: Vec::new(), index: HashMap::new(), out: Vec::new(), symbol_in: HashMap::new() };
        for p in paths {
            g.node(p);
        }
        let mut import_terms: HashMap<usize, HashSet<&str>> = HashMap::new();
        for (path, term) in imports {
            let i = g.node(path);
            import_terms.entry(i).or_default().insert(term.as_str());
        }
        let mut definers: HashMap<(usize, &str), Vec<usize>> = HashMap::new();
        for r in refs {
            let from = g.node(&r.from);
            let to = g.node(&r.to);
            definers.entry((from, r.ident.as_str())).or_default().push(to);
        }

        let mut edges: Vec<HashMap<usize, f64>> = vec![HashMap::new(); g.paths.len()];
        for ((from, ident), mut defs) in definers {
            defs.sort_unstable();
            defs.dedup();
            if defs.contains(&from) {
                continue;
            }
            let chosen = g.best_definers(from, &defs, import_terms.get(&from));
            let w = 1.0 / chosen.len() as f64;
            for to in chosen {
                *edges[from].entry(to).or_default() += w;
                *g.symbol_in.entry((to, ident.to_string())).or_default() += w;
            }
        }
        g.out = edges
            .into_iter()
            .map(|m| {
                let mut v: Vec<(usize, f64)> = m.into_iter().collect();
                v.sort_by_key(|e| e.0);
                v
            })
            .collect();
        g
    }

    fn node(&mut self, path: &str) -> usize {
        if let Some(&i) = self.index.get(path) {
            return i;
        }
        let i = self.paths.len();
        self.paths.push(path.to_string());
        self.index.insert(path.to_string(), i);
        i
    }

    /// The definers `from` most plausibly means: highest score wins, ties share the weight.
    /// Score: the import names the file's module (+2), names its directory (+1), the
    /// definer sits in `from`'s own directory (+1).
    fn best_definers(&self, from: usize, defs: &[usize], imports: Option<&HashSet<&str>>) -> Vec<usize> {
        let from_dir = dir_of(&self.paths[from]);
        let score = |to: usize| -> u32 {
            let path = &self.paths[to];
            let mut s = 0;
            if let Some(terms) = imports {
                if terms.contains(module_name(path).as_str()) {
                    s += 2;
                }
                let dir = dir_name(path).to_ascii_lowercase();
                if !dir.is_empty() && terms.contains(dir.as_str()) {
                    s += 1;
                }
            }
            if dir_of(path) == from_dir {
                s += 1;
            }
            s
        };
        let scored: Vec<(usize, u32)> = defs.iter().map(|&d| (d, score(d))).collect();
        let best = scored.iter().map(|x| x.1).max().unwrap_or(0);
        scored.into_iter().filter(|x| x.1 == best).map(|x| x.0).collect()
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    pub fn paths(&self) -> &[String] {
        &self.paths
    }

    pub fn index_of(&self, path: &str) -> Option<usize> {
        self.index.get(path).copied()
    }

    /// Total edge weight A→B (0 when A does not depend on B).
    pub fn edge(&self, from: usize, to: usize) -> f64 {
        self.out[from].binary_search_by_key(&to, |e| e.0).map(|i| self.out[from][i].1).unwrap_or(0.0)
    }

    /// Resolved references from other files into symbol `name` of file `file`.
    pub fn symbol_in_degree(&self, file: usize, name: &str) -> f64 {
        self.symbol_in.get(&(file, name.to_string())).copied().unwrap_or(0.0)
    }

    /// Personalized PageRank. `personal` is the teleport distribution (normalized here;
    /// all-zero or wrong length means uniform). A file with no outgoing edges hands its
    /// rank back through the teleport distribution, so the ranks always sum to 1.
    pub fn pagerank(&self, personal: &[f64]) -> Vec<f64> {
        let n = self.paths.len();
        if n == 0 {
            return Vec::new();
        }
        let total: f64 = personal.iter().filter(|v| v.is_finite() && **v > 0.0).sum();
        let p: Vec<f64> = if personal.len() == n && total > 0.0 {
            personal.iter().map(|&v| if v.is_finite() && v > 0.0 { v / total } else { 0.0 }).collect()
        } else {
            vec![1.0 / n as f64; n]
        };
        let out_sum: Vec<f64> = self.out.iter().map(|e| e.iter().map(|x| x.1).sum()).collect();
        let mut rank = p.clone();
        for _ in 0..MAX_ITERS {
            let dangling: f64 = (0..n).filter(|&i| out_sum[i] <= 0.0).map(|i| rank[i]).sum();
            let mut next: Vec<f64> = p.iter().map(|&pi| (1.0 - DAMPING) * pi + DAMPING * dangling * pi).collect();
            for (from, edges) in self.out.iter().enumerate() {
                if out_sum[from] <= 0.0 {
                    continue;
                }
                let share = DAMPING * rank[from] / out_sum[from];
                for &(to, w) in edges {
                    next[to] += share * w;
                }
            }
            let delta: f64 = next.iter().zip(&rank).map(|(a, b)| (a - b).abs()).sum();
            rank = next;
            if delta < TOLERANCE {
                break;
            }
        }
        rank
    }
}

/// Lowercased name a file is imported by: its stem, or its directory for an index file
/// (`src/store/mod.rs` → `store`, `lib/config.ts` → `config`).
pub fn module_name(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    let stem = file.split('.').next().unwrap_or(file).to_ascii_lowercase();
    if MODULE_INDEX_STEMS.contains(&stem.as_str()) {
        let dir = dir_name(path).to_ascii_lowercase();
        if !dir.is_empty() {
            return dir;
        }
    }
    stem
}

fn dir_of(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

fn dir_name(path: &str) -> &str {
    let dir = dir_of(path);
    dir.rsplit('/').next().unwrap_or(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(from: &str, ident: &str, to: &str) -> RefRow {
        RefRow { from: from.into(), ident: ident.into(), to: to.into() }
    }

    #[test]
    fn module_names() {
        assert_eq!(module_name("src/lib/config.ts"), "config");
        assert_eq!(module_name("src/store/mod.rs"), "store");
        assert_eq!(module_name("pkg/__init__.py"), "pkg");
        assert_eq!(module_name("Billing.cs"), "billing");
    }

    #[test]
    fn local_definition_is_not_an_edge() {
        let refs = vec![r("a.ts", "Foo", "a.ts"), r("a.ts", "Foo", "b.ts"), r("c.ts", "Foo", "a.ts"), r("c.ts", "Foo", "b.ts")];
        let g = FileGraph::build(["a.ts", "b.ts", "c.ts"], &refs, &[]);
        let (a, b, c) = (g.index_of("a.ts").unwrap(), g.index_of("b.ts").unwrap(), g.index_of("c.ts").unwrap());
        assert_eq!(g.edge(a, b), 0.0, "a defines Foo itself");
        // c cannot tell which Foo it means: the reference is split.
        assert!((g.edge(c, a) - 0.5).abs() < 1e-12);
        assert!((g.edge(c, b) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn imports_then_directory_pick_the_definer() {
        let refs = vec![
            r("core/x.ts", "Config", "core/config.ts"),
            r("core/x.ts", "Config", "app/config.ts"),
            r("core/x.ts", "Helper", "util/helper.ts"),
            r("core/x.ts", "Helper", "core/helper2.ts"),
        ];
        let imports = vec![("core/x.ts".to_string(), "config".to_string()), ("core/x.ts".to_string(), "helper".to_string())];
        let g = FileGraph::build(["core/x.ts"], &refs, &imports);
        let x = g.index_of("core/x.ts").unwrap();
        // Both config.ts match the import; the same-directory one wins.
        assert_eq!(g.edge(x, g.index_of("core/config.ts").unwrap()), 1.0);
        assert_eq!(g.edge(x, g.index_of("app/config.ts").unwrap()), 0.0);
        // The import names util/helper.ts; it beats the same-directory definer.
        assert_eq!(g.edge(x, g.index_of("util/helper.ts").unwrap()), 1.0);
        assert_eq!(g.symbol_in_degree(g.index_of("core/config.ts").unwrap(), "Config"), 1.0);
    }

    #[test]
    fn pagerank_sums_to_one_and_follows_edges() {
        let refs = vec![r("a.ts", "B", "b.ts"), r("b.ts", "C", "c.ts"), r("d.ts", "C", "c.ts")];
        let g = FileGraph::build(["a.ts", "b.ts", "c.ts", "d.ts"], &refs, &[]);
        let rank = g.pagerank(&[]);
        let sum: f64 = rank.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "{sum}");
        let c = rank[g.index_of("c.ts").unwrap()];
        assert!(rank.iter().all(|&v| v <= c), "c is the sink everyone reaches: {rank:?}");

        // Personalization pulls rank toward the chosen file.
        let mut p = vec![1.0; 4];
        p[g.index_of("d.ts").unwrap()] += 20.0;
        let biased = g.pagerank(&p);
        assert!(biased[g.index_of("d.ts").unwrap()] > rank[g.index_of("d.ts").unwrap()]);
    }
}
