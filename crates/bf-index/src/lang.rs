//! Language registry — which tree-sitter grammar parses a file, and which node kinds are
//! definitions, containers and identifiers in it.
//!
//! A new language is a new `LangSpec` entry, never a branch in the chunker: the chunker
//! reads these tables and nothing else.

use std::path::Path;
use tree_sitter::Language;

pub struct LangSpec {
    /// Stable id stored with each file (`files.lang`).
    pub id: &'static str,
    /// Node kinds that are a definition worth its own chunk and a repo-map entry.
    pub def_kinds: &'static [&'static str],
    /// Definition kinds whose body holds member definitions (classes, impls, modules).
    /// An oversized container is split into a header chunk plus one chunk per member.
    pub container_kinds: &'static [&'static str],
    /// Wrapper kinds (`export_statement`, `decorated_definition`) whose inner declaration
    /// is the definition, while the chunk spans the wrapper.
    pub wrapper_kinds: &'static [&'static str],
    /// Node kinds whose text is an identifier reference (feeds the repo-map ranking).
    pub ident_kinds: &'static [&'static str],
    /// Namespace-like blocks walked THROUGH at top level (C# `namespace X { … }`): their
    /// definitions are top-level definitions, the block itself is not a symbol.
    pub transparent_kinds: &'static [&'static str],
    /// Import / use / using statements. Their identifier terms say which files this one
    /// depends on, so the repo-map graph can resolve a name defined in several files to
    /// the one actually imported.
    pub import_kinds: &'static [&'static str],
}

const TS_IDENTS: &[&str] = &[
    "identifier",
    "type_identifier",
    "property_identifier",
    "shorthand_property_identifier",
];

static RUST: LangSpec = LangSpec {
    id: "rust",
    def_kinds: &[
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
        "mod_item",
        "type_item",
        "const_item",
        "static_item",
        "macro_definition",
    ],
    container_kinds: &["impl_item", "trait_item", "mod_item"],
    wrapper_kinds: &[],
    ident_kinds: &["identifier", "type_identifier", "field_identifier"],
    transparent_kinds: &[],
    import_kinds: &["use_declaration"],
};

// TypeScript and TSX share every table; only the grammar and the stored id differ.
// A `static` initialiser cannot read another static, so the shared tables are a const.
static TYPESCRIPT: LangSpec = TYPESCRIPT_FIELDS;
static TSX: LangSpec = LangSpec { id: "tsx", ..TYPESCRIPT_FIELDS };
const TYPESCRIPT_FIELDS: LangSpec = LangSpec {
    id: "typescript",
    def_kinds: &[
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "method_definition",
        "lexical_declaration",
    ],
    container_kinds: &["class_declaration", "abstract_class_declaration"],
    wrapper_kinds: &["export_statement"],
    ident_kinds: TS_IDENTS,
    transparent_kinds: &[],
    import_kinds: &["import_statement"],
};

static JAVASCRIPT: LangSpec = LangSpec {
    id: "javascript",
    def_kinds: &[
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "lexical_declaration",
    ],
    container_kinds: &["class_declaration"],
    wrapper_kinds: &["export_statement"],
    ident_kinds: TS_IDENTS,
    transparent_kinds: &[],
    import_kinds: &["import_statement"],
};

static PYTHON: LangSpec = LangSpec {
    id: "python",
    def_kinds: &["function_definition", "class_definition"],
    container_kinds: &["class_definition"],
    wrapper_kinds: &["decorated_definition"],
    ident_kinds: &["identifier"],
    transparent_kinds: &[],
    import_kinds: &["import_statement", "import_from_statement"],
};

static GO: LangSpec = LangSpec {
    id: "go",
    def_kinds: &["function_declaration", "method_declaration", "type_declaration"],
    container_kinds: &[],
    wrapper_kinds: &[],
    ident_kinds: &["identifier", "type_identifier", "field_identifier"],
    transparent_kinds: &[],
    import_kinds: &["import_declaration"],
};

static JAVA: LangSpec = LangSpec {
    id: "java",
    def_kinds: &[
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "method_declaration",
        "constructor_declaration",
    ],
    container_kinds: &["class_declaration", "interface_declaration", "enum_declaration", "record_declaration"],
    wrapper_kinds: &[],
    ident_kinds: &["identifier", "type_identifier"],
    transparent_kinds: &[],
    import_kinds: &["import_declaration"],
};

static CSHARP: LangSpec = LangSpec {
    id: "csharp",
    def_kinds: &[
        "class_declaration",
        "struct_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "delegate_declaration",
        "method_declaration",
        "constructor_declaration",
        "property_declaration",
    ],
    container_kinds: &["class_declaration", "struct_declaration", "interface_declaration", "record_declaration"],
    wrapper_kinds: &[],
    ident_kinds: &["identifier"],
    transparent_kinds: &["namespace_declaration"],
    import_kinds: &["using_directive"],
};

static RUBY: LangSpec = LangSpec {
    id: "ruby",
    def_kinds: &["method", "singleton_method", "class", "module"],
    container_kinds: &["class", "module"],
    wrapper_kinds: &[],
    ident_kinds: &["identifier", "constant"],
    transparent_kinds: &[],
    // `require_relative` is a plain method call in the grammar; Ruby resolves collisions
    // by same-directory preference only.
    import_kinds: &[],
};

static PHP: LangSpec = LangSpec {
    id: "php",
    def_kinds: &[
        "function_definition",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
        "enum_declaration",
        "method_declaration",
    ],
    container_kinds: &["class_declaration", "interface_declaration", "trait_declaration"],
    wrapper_kinds: &[],
    ident_kinds: &["name"],
    transparent_kinds: &["namespace_definition"],
    import_kinds: &["namespace_use_declaration"],
};

static KOTLIN: LangSpec = LangSpec {
    id: "kotlin",
    def_kinds: &["class_declaration", "object_declaration", "companion_object", "function_declaration", "type_alias"],
    container_kinds: &["class_declaration", "object_declaration", "companion_object"],
    wrapper_kinds: &[],
    ident_kinds: &["identifier"],
    transparent_kinds: &[],
    import_kinds: &["import"],
};

static SWIFT: LangSpec = LangSpec {
    id: "swift",
    def_kinds: &[
        "class_declaration",
        "protocol_declaration",
        "function_declaration",
        "init_declaration",
        "protocol_function_declaration",
        "typealias_declaration",
    ],
    container_kinds: &["class_declaration", "protocol_declaration"],
    wrapper_kinds: &[],
    ident_kinds: &["simple_identifier", "type_identifier"],
    transparent_kinds: &[],
    import_kinds: &["import_declaration"],
};

/// The grammar and spec for a path, or `None` for a file chunked by line windows.
pub fn detect(path: &Path) -> Option<(Language, &'static LangSpec)> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    let pair: (Language, &'static LangSpec) = match ext.as_str() {
        "rs" => (tree_sitter_rust::LANGUAGE.into(), &RUST),
        "ts" | "mts" | "cts" => (tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(), &TYPESCRIPT),
        "tsx" => (tree_sitter_typescript::LANGUAGE_TSX.into(), &TSX),
        "js" | "jsx" | "mjs" | "cjs" => (tree_sitter_javascript::LANGUAGE.into(), &JAVASCRIPT),
        "py" => (tree_sitter_python::LANGUAGE.into(), &PYTHON),
        "go" => (tree_sitter_go::LANGUAGE.into(), &GO),
        "java" => (tree_sitter_java::LANGUAGE.into(), &JAVA),
        "cs" => (tree_sitter_c_sharp::LANGUAGE.into(), &CSHARP),
        "rb" => (tree_sitter_ruby::LANGUAGE.into(), &RUBY),
        "php" => (tree_sitter_php::LANGUAGE_PHP.into(), &PHP),
        "kt" | "kts" => (tree_sitter_kotlin_ng::LANGUAGE.into(), &KOTLIN),
        "swift" => (tree_sitter_swift::LANGUAGE.into(), &SWIFT),
        _ => return None,
    };
    Some(pair)
}

/// Text files worth indexing without a grammar (line-window chunks).
pub fn is_plain_text_source(path: &Path) -> bool {
    const EXTS: &[&str] = &[
        "md", "mdx", "txt", "json", "jsonc", "yaml", "yml", "toml", "css", "scss", "html", "sql", "sh",
        "ps1", "c", "h", "cc", "cpp", "hpp", "vue", "svelte", "graphql",
        "proto", "tf", "dockerfile",
    ];
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    if name == "dockerfile" || name == "makefile" {
        return true;
    }
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}
